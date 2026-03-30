import glob
from pathlib import Path
from typing import Any
from hydra.utils import instantiate
import torch
import time
import os
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, ListConfig, OmegaConf
from loguru import logger

from multi_service.agents.based_agent import BaseSDNAgent
from multi_service.utils import checkpoint_manager
from ..env import SdnWrapper
from ..env.flow_generator import FlowType
from ..utils import CheckpointManager, PPOMemory, AttrDict, BankTrafficManager


class PPOTrainer:

    def __init__(
        self,
        env: SdnWrapper,
        eval_src_nodes: list,
        eval_dst_nodes: list,
        checkpoint_dir: str,
        tensorboard_dir: str,
        batch_size: int,
        train_max_steps: int,
        eval_max_steps: int,
        max_epochs: int,
        save_interval: int,
        qos_reward: DictConfig,
        agent: Any,
        memory: PPOMemory,
        fgprt_bank: BankTrafficManager,
    ) -> None:

        self.start_epoch = 0
        self.global_step = 0
        self.global_flow = 0

        self.env = env
        self.agent = agent
        self.memory = memory
        self.fgprt_bank = fgprt_bank

        self.eval_max_steps = eval_max_steps
        self.train_max_steps = train_max_steps

        self.batch_size = batch_size
        self.qos_reward = qos_reward
        self.max_epochs = max_epochs

        self.val_src_nodes = eval_src_nodes
        self.val_dst_nodes = eval_dst_nodes
        self.save_interval = save_interval

        # 初始化 TensorBoard
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.writer = SummaryWriter(log_dir=tensorboard_dir)

    def train_one_epoch(self):
        """
        训练一个 Epoch：采样数据 -> 存入 Memory -> 更新 Agent
        """
        flow_count = 0
        total_batch_reward = 0  # 用于计算整个 Batch 平均分
        self.agent.train()
        self.env.topo_stats = self.env.train_stats# 确保使用训练集统计数据
        # 1. 生成一批业务流任务
        flows = self.fgprt_bank.generate_batch(self.batch_size)
        logger.debug(f"Starting rollout for {len(flows)} flows...")
        for flow in flows:

            # 覆盖任务设定
            self.env.s_node = flow.src
            self.env.d_node = flow.dst
            self.env.current_flow_type = flow.flow_type

            # 同步环境指针
            self.env.current_node = flow.src
            self.env.path_so_far = [flow.src]
            self.env.step_count = 0

            logger.trace(
                f"[Flow {flow_count+1:02d}] {flow.flow_type.name} Task: Node {flow.src} -> Node {flow.dst}"
            )
            state = self.env.reset_flow()
            state = self.env.get_observation()
            state = state.to(self.agent.device)

            # 准备 Fingerprint
            curr_node = flow.src
            target_node = flow.dst
            fingerprint = flow.fingerprint.to(self.agent.device)
            if fingerprint.dim() == 2:
                fingerprint = fingerprint.unsqueeze(0)

            done = False
            step_count = 0
            episode_reward = 0  # 单条流的累积奖励

            # 逐跳寻路循环
            while not done and step_count < self.train_max_steps:
                with torch.no_grad():
                    # 1. 获取 Embedding
                    node_embeds = self.agent.get_node_embeddings(
                        state, fingerprint)

                    # 2. Critic 估值提早计算 val，防止死胡同里 crash
                    critc_val = self.agent.evaluate_value(  # pyright: ignore[reportCallIssue]
                        node_embeds, curr_node, target_node)

                    # 3. 准备邻居与掩码逻辑
                    edge_attr = state.edge_attr
                    edge_index = state.edge_index

                    assert edge_index is not None, "edge_index 为空"
                    assert edge_attr is not None, "edge_attr 为空"

                    mask = (edge_index[0] == curr_node)
                    neighbor_indices = edge_index[1][mask]
                    neighbor_edge_attrs = edge_attr[mask]

                    # 提取当前动作掩码
                    current_action_mask = state.action_mask[mask]

                    # 6. Agent 决策
                    next_node, log_prob, logits, action_idx = self.agent.get_action(
                        state,
                        node_embeds,
                        current_action_mask,  # pyright: ignore[reportCallIssue]
                        curr_node,
                        target_node,
                        neighbor_indices.tolist(),
                        neighbor_edge_attrs,
                        deterministic=False)

                    # 7. 死胡同处理
                    if next_node is None:
                        logger.warning(
                            f"Dead path{self.env.path_so_far}, ai failed to find path !"
                        )
                        logger.debug(f"neighbor_indices: {neighbor_indices}")
                        # [Fix] Reduce penalty to be consistent with loop/timeout
                        reward = -2.0
                        done = True
                        # 存入记忆
                        self.memory.store(
                            state=state,
                            action=torch.tensor(
                                0, device=self.agent.device
                            ),  # 使用0表示无效action (FullAgent has been fixed to handle this) 
                            log_prob=torch.log(
                                torch.tensor(
                                    1e-10, device=self.agent.device)),  # 极小的概率
                            value=critc_val.item(),
                            reward=float(reward),
                            is_terminal=done,
                            fingerprint=fingerprint,
                            curr_idx=curr_node,
                            target_idx=target_node)
                        self.global_step += 1
                        break

                    assert action_idx is not None, f"action_idx 为空: {action_idx}"
                    assert next_node is not None, f"next_node 为空: {next_node}"

                    # 8. 环境执行
                    assert self.env is not None, "self.env 为空"
                    # 直接使用邻居索引获取对应的边索引
                    selected_edge_idx = torch.where(mask)[0][action_idx].item()
                    next_state, reward, done, info = self.env.step(
                        int(selected_edge_idx))

                    #  Clean Step Logging
                    logger.trace(
                        f"Step {len(self.env.path_so_far)-1:02d} |{curr_node} -> {next_node} | Reward: {reward:.4f}"
                    )

                    # Move detailed info to Debug tag
                    for key, value in info.items():
                        logger.trace(f"{key}: {value}")
                    logger.trace(f"neighbor_indices: {neighbor_indices}")

                    # 9. 存入记忆
                    self.memory.store(state=state,
                                      action=action_idx,
                                      log_prob=log_prob,
                                      value=critc_val.item(),
                                      reward=float(reward),
                                      is_terminal=done,
                                      fingerprint=fingerprint,
                                      curr_idx=curr_node,
                                      target_idx=target_node)

                    # 10. 更新状态流转
                    state = next_state.to(self.agent.device)
                    curr_node = next_node
                    episode_reward += float(reward)  # 累加单条流得分
                    step_count += 1

                    # 11. 每步更新global_step
                    self.global_step += 1

            # 循环结束
            flow_count += 1
            total_batch_reward += episode_reward  # 累加到 Batch 总分

            self.global_flow += 1
            self.writer.add_scalar(f"/Reward/{flow.flow_type.name}",
                                   episode_reward, self.global_flow)

            # 打印的是 episode_reward
            logger.debug(
                f"[Flow {flow_count:02d}] {flow.flow_type.name.upper():<10} Finished | Total Reward: {episode_reward:>8.3f} | Path: {'->'.join(map(str, self.env.path_so_far))}"
            )
        
        # Update Agent
        actor_loss, critic_loss = 0.0, 0.0
        if len(self.memory.rewards) > 0:
            logger.trace("Updating Agent...")
            actor_loss, critic_loss = self.agent.update(
                self.memory)  # pyright: ignore[reportGeneralTypeIssues]

            self.writer.add_scalar(f"/Loss/Actor", actor_loss,
                                   self.global_step)
            self.writer.add_scalar(f"/Loss/Critic", critic_loss,
                                   self.global_step)
        else:
            logger.error(
                "No valid transitions collected, skipping agent update")

        avg_reward = total_batch_reward / flow_count if flow_count > 0 else 0.0
        self.writer.add_scalar("Reward/Average", avg_reward, self.global_step)

        return avg_reward

    def validate(self, epoch):
        """
        在线验证逻辑：
        1. 切换到 eval 模式
        2. 针对验证集节点运行固定任务
        3. 记录指标
        """
        self.env.topo_stats = self.env.eval_stats  # 确保使用验证集统计数据
        if not self.val_src_nodes or not self.val_dst_nodes:
            logger.warning(
                "No validation nodes provided, skipping validation.")
            return -float('inf')

        logger.debug(f"Starting Validation (Epoch {epoch})...")
        self.agent.eval()

        # 使用 self.fgprt_bank 生成 10 个验证任务
        flows = self.fgprt_bank.generate_batch(batch_size=10,
                                               src_nodes=self.val_src_nodes,
                                               dst_nodes=self.val_dst_nodes)

        total_reward = 0
        success_count = 0

        for i, flow in enumerate(flows):
            # 覆盖环境状态
            self.env.s_node = flow.src
            self.env.d_node = flow.dst
            self.env.current_flow_type = flow.flow_type
            self.env.current_node = flow.src
            self.env.path_so_far = [flow.src]
            self.env.step_count = 0

            state = self.env.reset_flow()
            state = self.env.get_observation()
            state = state.to(self.agent.device)

            # 使用生成的 Fingerprint
            fingerprint = flow.fingerprint.to(self.agent.device)
            if fingerprint.dim() == 2:
                fingerprint = fingerprint.unsqueeze(0)

            done = False
            step = 0
            episode_reward = 0

            while not done and step < self.eval_max_steps:
                with torch.no_grad():
                    node_embeds = self.agent.get_node_embeddings(
                        state, fingerprint)

                    edge_attr = state.edge_attr
                    edge_index = state.edge_index

                    assert isinstance(
                        edge_index,
                        torch.Tensor), "edge_index should be a tensor"
                    assert isinstance(
                        edge_attr,
                        torch.Tensor), "edge_attr should be a tensor"

                    mask = (edge_index[0] == self.env.current_node)
                    neighbor_indices = edge_index[1][mask]
                    neighbor_edge_attrs = edge_attr[mask]
                    current_action_mask = state.action_mask[mask]

                    # Deterministic Action for Validation
                    next_node, _, _, action_idx = self.agent.get_action(
                        state,
                        node_embeds,
                        current_action_mask,
                        self.env.current_node,
                        self.env.d_node,
                        neighbor_indices.tolist(),
                        neighbor_edge_attrs,
                        deterministic=True)

                    if next_node is None:
                        episode_reward += self.qos_reward.penalty_dead_end
                        done = True
                        break

                    selected_edge_idx = torch.where(mask)[0][action_idx].item()
                    next_state, reward, done, info = self.env.step(
                        int(selected_edge_idx))

                    state = next_state.to(self.agent.device)
                    episode_reward += float(reward)
                    step += 1

            total_reward += episode_reward
            if done and self.env.current_node == self.env.d_node:
                success_count += 1

            # logger.debug(f"[Val Case {i}] {case['type'].name} {case['src']}->{case['dst']} | Reward: {episode_reward:.2f} | Path: {self.env.path_so_far}")

        avg_reward = total_reward / len(flows)
        success_rate = success_count / len(flows)

        if self.writer:
            self.writer.add_scalar("Val/Reward", avg_reward, epoch)
            self.writer.add_scalar("Val/Success_Rate", success_rate, epoch)

        self.agent.train()  # 恢复训练模式
        return avg_reward

    def run(self):
        """
        主训练入口
        """

        best_train_reward = -float('inf')
        best_eval_reward = -float('inf')
        no_improvement_count = 0

        for epoch in range(self.start_epoch, self.max_epochs):
            start_time = time.time()

            # 1. 训练
            train_reward = self.train_one_epoch()
            self.agent.scheduler.step(train_reward)
            # 2. 保存 Checkpoint
            if epoch % self.save_interval == 0:
                self.checkpoint_manager.save(
                    model=self.agent,
                    epoch=epoch,
                    metrics={'train_reward': train_reward})

            # 3. Validation
            eval_reward = self.validate(epoch)

            # Early Stopping Logic
            if train_reward > best_train_reward or eval_reward > best_eval_reward:
                best_train_reward = max(best_train_reward, train_reward)
                best_eval_reward = max(best_eval_reward, eval_reward)
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # if no_improvement_count >= 100:
            #     logger.info(
            #         f"Early Stopping at Epoch {epoch}: No improvement for 20 consecutive epochs. Best Train: {best_train_reward:.4f}, Best Eval: {best_eval_reward:.4f}"
            #     )
            #     break

            duration = time.time() - start_time
            logger.info(
                f"Epoch {epoch}: Train Reward={train_reward:.4f} | Eval Reward={eval_reward:.4f} | Current LR={self.agent.get_current_lr():.8f} |Time={duration:.2f}s"  # pyright: ignore[reportCallIssue]
            )
        self.checkpoint_manager.save(model=self.agent,
                                        epoch=epoch,
                                        save_file=f"final.pth",
                                        metrics={
                                            'train_reward': train_reward,
                                            'eval_reward': eval_reward
                                        },
                                        optimizer=self.agent.optimizer,
                                        scheduler=self.agent.scheduler
                                        )

    def load_weight(self, checkpoint_path):
        self.checkpoint_manager.load(self.agent, checkpoint_path)