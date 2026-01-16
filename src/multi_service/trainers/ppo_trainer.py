import glob
from typing import Any
import torch
import time
import os
from torch.utils.tensorboard import SummaryWriter

from ..agents import FiLMPPOAgent, AblationAgent
from ..env import SdnWrapper
from ..utils import CheckpointManager, logger, PPOMemory, AttrDict, BankTrafficManager

class PPOTrainer:
  def __init__(self, agent: Any, env: SdnWrapper, memory: PPOMemory, config: AttrDict) -> None:    

    self.env = env
    self.agent = agent
    self.memory = memory
    self.config = config
    self.start_epoch = 0

    # 初始化 TensorBoard
    self.global_step = 0
    self.global_flow = 0
    self.checkpoint_manager = CheckpointManager(config.path.ckpt_dir)
    tb_dir = os.path.join(config.path.log_dir, "tensorboard")
    self.writer = SummaryWriter(log_dir=tb_dir)

  def train_one_epoch(self, traffic_generator: BankTrafficManager):
    """
    训练一个 Epoch：采样数据 -> 存入 Memory -> 更新 Agent
    """
    flow_count   = 0
    total_batch_reward = 0 # 用于计算整个 Batch 平均分
    self.agent.train()
    
    # 1. 生成一批业务流任务
    flows = traffic_generator.generate_batch(self.config.train.batch_size)  
    logger.log(f"Starting rollout for {len(flows)} flows...", tag="Train Init")
    for flow in flows:
        
        # 覆盖任务设定
        self.env.s_node = flow.src
        self.env.d_node = flow.dst
        self.env.current_flow_type = flow.flow_type
        
        # 同步环境指针
        self.env.current_node = flow.src
        self.env.path_so_far = [flow.src]
        self.env.step_count = 0
            
        logger.log(f"[Flow {flow_count+1:02d}] {flow.flow_type.name} Task: Node {flow.src} -> Node {flow.dst}", tag="Flow Start")  
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
        episode_reward = 0 # 单条流的累积奖励
        
        # 逐跳寻路循环
        while not done and step_count < self.config.train.max_steps:
            with torch.no_grad():
                # 1. 获取 Embedding
                node_embeds = self.agent.get_node_embeddings(state, fingerprint)
                
                # 2. Critic 估值提早计算 val，防止死胡同里 crash
                critc_val = self.agent.evaluate_value(node_embeds, curr_node, target_node)
                
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
                    state, node_embeds, current_action_mask, curr_node, target_node, 
                    neighbor_indices.tolist(), neighbor_edge_attrs, deterministic=False 
                )
                
                # 7. 死胡同处理
                if next_node is None:
                    logger.log(f"Dead path{self.env.path_so_far}, ai failed to find path !", tag="Path Failed")
                    logger.log(f"neighbor_indices: {neighbor_indices}", tag="Debug")
                    reward = self.config.qos_reward.penalty_dead_end
                    done = True
                    # 存入记忆
                    self.memory.store(
                        state=state,            
                        action=torch.tensor(0, device=self.agent.device),  # 使用-1表示无效action  
                        log_prob=torch.log(torch.tensor(1e-10, device=self.agent.device)),  # 极小的概率
                        value=critc_val.item(),       
                        reward=float(reward),
                        is_terminal=done,       
                        fingerprint=fingerprint,
                        curr_idx=curr_node,     
                        target_idx=target_node
                    )
                    self.global_step += 1
                    break

                assert action_idx is not None, f"action_idx 为空: {action_idx}"
                assert next_node is not None, f"next_node 为空: {next_node}"

                # 8. 环境执行
                assert self.env is not None, "self.env 为空"
                # 直接使用邻居索引获取对应的边索引
                selected_edge_idx = torch.where(mask)[0][action_idx].item()
                next_state, reward, done, info = self.env.step(int(selected_edge_idx))
                
                # [Optimization] Clean Step Logging
                logger.log(f"{curr_node} -> {next_node} | Reward: {reward:.4f}", tag=f"Step {len(self.env.path_so_far)-1:02d}")
                
                # Move detailed info to Debug tag
                for key, value in info.items():
                    logger.log(f"{key}: {value}", tag="Debug")
                logger.log(f"neighbor_indices: {neighbor_indices}", tag="Debug")
                
                # 9. 存入记忆
                self.memory.store(
                    state=state,            
                    action=action_idx,
                    log_prob=log_prob,      
                    value=critc_val.item(),       
                    reward=float(reward),
                    is_terminal=done,       
                    fingerprint=fingerprint,
                    curr_idx=curr_node,     
                    target_idx=target_node
                )
                
                # 10. 更新状态流转
                state = next_state.to(self.agent.device)
                curr_node = next_node
                episode_reward += float(reward) # 累加单条流得分
                step_count += 1
                
                # 11. 每步更新global_step
                self.global_step += 1
        
        # 循环结束
        flow_count += 1
        total_batch_reward += episode_reward # 累加到 Batch 总分
        
        self.global_flow += 1
        self.writer.add_scalar(f"/Reward/{flow.flow_type.name}", episode_reward, self.global_flow)
        
        # 打印的是 episode_reward
        logger.log(f"[Flow {flow_count:02d}] Finished | Total Reward: {episode_reward:.2f} | Path: {'->'.join(map(str, self.env.path_so_far))}", tag="Flow End", log_to_console=False)

    # Update Agent
    actor_loss, critic_loss = 0.0, 0.0
    if len(self.memory.rewards) > 0:
        logger.log("Updating Agent...", tag="Trainer", log_to_console=False)
        actor_loss, critic_loss = self.agent.update(self.memory)
        
        self.writer.add_scalar(f"/Loss/Actor", actor_loss, self.global_step)
        self.writer.add_scalar(f"/Loss/Critic", critic_loss, self.global_step)
    else:
        logger.log("No valid transitions collected, skipping agent update", tag="Trainer Warn")
    
    avg_reward = total_batch_reward / flow_count if flow_count > 0 else 0.0
    self.writer.add_scalar("Reward/Average", avg_reward, self.global_step)
        
    return avg_reward

  def run(self, traffic_generator):
    """
    主训练入口
    """
    logger.log(f"Start training for {self.config.train.max_epochs} epochs...", tag="System")

    use_curriculum = self.config.env.curriculum

    start_load = self.config.env.start_load
    end_load = self.config.env.end_load

    for epoch in range(self.start_epoch, self.config.train.max_epochs):
        start_time = time.time()

        if use_curriculum:
            progress = epoch / self.config.train.max_epochs
            current_load = start_load + (end_load - start_load) * progress
            
            self.config['load_flow'] = current_load
            
            if self.writer:
                self.writer.add_scalar("Train/Difficulty_Load_Mbps", current_load, epoch)

        # 1. 训练
        train_reward = self.train_one_epoch(traffic_generator)
        
        # 2. 保存 Checkpoint
        if epoch % self.config.train.save_interval == 0:
            self.checkpoint_manager.save(
                model=self.agent,
                epoch=epoch,
                metrics={'train_reward': train_reward}
            )

        duration = time.time() - start_time
        logger.log(f"Epoch {epoch}: Reward={train_reward:.4f} | Time={duration:.2f}s", tag="Train", log_to_console=True)

