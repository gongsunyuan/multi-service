import glob
import torch
import time
import os
from torch.utils.tensorboard import SummaryWriter

from ..utils import CheckpointManager, logger

class PPOTrainer:
  def __init__(self, agent, env, memory, config):

    self.env = env
    self.agent = agent
    self.memory = memory
    self.config = config
    
    # 初始化 TensorBoard
    self.global_step = 0
    self.checkpoint_manager = CheckpointManager(config.path.checkpoint_dir)
    tb_dir = os.path.join(config.path.log_dir, "tensorboard")
    self.writer = SummaryWriter(log_dir=tb_dir)

  def train_one_epoch(self, traffic_generator):
    """
    训练一个 Epoch：采样数据 -> 存入 Memory -> 更新 Agent
    """
    flow_count   = 0
    total_batch_reward = 0 # 用于计算整个 Batch 平均分
    self.agent.train()
    
    # 1. 生成一批业务流任务
    flows = traffic_generator.generate_batch(self.config.train.batch_size)
    logger.log(f"Starting rollout for {len(flows)} flows...", tag="Train Init")
    self.env.reset_bg(self.config.load_flow)
    for flow in flows:
        
        # 覆盖任务设定
        self.env.s_node = flow.src
        self.env.d_node = flow.dst
        self.env.current_flow_type = flow.flow_type
        
        # 同步环境指针
        self.env.current_node = flow.src
        self.env.path_so_far = [flow.src]
        self.env.step_count = 0
        
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
                val = self.agent.evaluate_value(node_embeds, curr_node, target_node)
                
                # 3. 准备邻居与掩码逻辑
                edge_attr = state.edge_attr
                edge_index = state.edge_index
                mask = (edge_index[0] == curr_node)
                neighbor_indices = edge_index[1][mask]
                neighbor_edge_attrs = edge_attr[mask]
                global_edge_indices = torch.where(mask)[0]

                # 禁止回头路逻辑
                visited_set = set(self.env.path_so_far)
                valid_local_indices = []
                for i, neighbor_node in enumerate(neighbor_indices.tolist()):
                    if neighbor_node not in visited_set:
                        valid_local_indices.append(i)
                
                # 4. 死胡同判断
                if len(valid_local_indices) == 0:
                    reward = self.config.qos_reward.penalty_dead_end 
                    # 现在 val 已经定义了，可以安全存储
                    logger.log(f"Dead path{self.env.path_so_far}, ai failed to find path !", tag="Path Failed")
                    dead_end_log_prob = torch.tensor(0.0, device=self.agent.device)
                    self.memory.store(state, 0, dead_end_log_prob, val.item(), float(reward), True, 
                                    fingerprint, curr_node, target_node)
                    break # 结束这条流
                
                # 5. 修改掩码 (Whitelist)
                state.action_mask[global_edge_indices] = False # 先全部屏蔽
                for valid_idx in valid_local_indices:
                    good_global_edge = global_edge_indices[valid_idx].item()
                    state.action_mask[good_global_edge] = True   # 只开合法的

                # 6. Agent 决策
                next_node, log_prob, logits, action_idx = self.agent.get_action(
                    state, node_embeds, curr_node, target_node, 
                    neighbor_indices, neighbor_edge_attrs, deterministic=False 
                )
                if next_node == None:
                    reward = self.config.qos_reward.penalty_dead_end

                # 7. 环境执行
                selected_edge_idx = global_edge_indices[action_idx].item()
                next_state, reward, done, info = self.env.step(selected_edge_idx)
                
                # 8. 存入记忆
                self.memory.store(
                state=state,            
                action=action_idx,
                log_prob=log_prob,      
                value=val.item(),       
                reward=float(reward),
                is_terminal=done,       
                fingerprint=fingerprint,
                curr_idx=curr_node,     
                target_idx=target_node
                )
                
                # 状态流转
                state = next_state.to(self.agent.device)
                curr_node = next_node
                episode_reward += float(reward) # 累加单条流得分
                step_count += 1
        
        # 循环结束
        flow_count += 1
        total_batch_reward += episode_reward # 累加到 Batch 总分
        
        # 打印的是 episode_reward
        logger.log(f"Flow {flow_count} finished. Reward: {episode_reward:.2f}", tag="Flow", log_to_console=False)

    # Update Agent
    actor_loss, critic_loss = 0, 0
    if len(self.memory.rewards) > 0:
        logger.log("Updating Agent...", tag="Trainer", log_to_console=True)
        actor_loss, critic_loss = self.agent.update(self.memory)
        
        self.writer.add_scalar("Loss/Actor", actor_loss, self.global_step)
        self.writer.add_scalar("Loss/Critic", critic_loss, self.global_step)
    
    avg_reward = total_batch_reward / max(1, flow_count)
    self.writer.add_scalar("Reward/Average", avg_reward, self.global_step)
    self.global_step += 1
        
    return avg_reward

  def run(self, traffic_generator):
    """
    主训练入口
    """
    logger.log(f"Start training for {self.config.train.max_epochs} epochs...", tag="System")

    use_curriculum = self.config.env.curriculum

    start_load = self.config.env.start_load
    end_load = self.config.env.end_load

    for epoch in range(self.config.train.max_epochs):
        start_time = time.time()

    if use_curriculum:
        progress = epoch / self.config.train.max_epochs
        current_load = start_load + (end_load - start_load) * progress
        
        self.config.load_flow = current_load
        
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

