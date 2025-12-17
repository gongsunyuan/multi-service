import glob
import torch
import time
import os
from torch.utils.tensorboard import SummaryWriter
from ..utils.verbose_logger import logger, vprint

class PPOTrainer:
  def __init__(self, agent, env, memory, config):

    self.env = env
    self.agent = agent
    self.memory = memory
    self.config = config
    
    # 初始化 TensorBoard
    tb_dir = os.path.join(config.log_dir, "tensorboard")
    self.writer = SummaryWriter(log_dir=tb_dir)
    self.global_step = 0

  def train_one_epoch(self, traffic_generator):
    """
    训练一个 Epoch：采样数据 -> 存入 Memory -> 更新 Agent
    """
    flow_count   = 0
    total_reward = 0
    self.agent.train() # 设置为训练模式
    
    # 1. 生成一批业务流任务
    flows = traffic_generator.generate_batch(self.config.batch_size)
    
    logger.log(f"Starting rollout for {len(flows)} flows...", tag="Train Init")

    for flow in flows:
      # Reset Environment for new flow 
      # flow 对象应包含: src, dst, fingerprint, bandwidth 等
      # 重置环境，并获取初始状态 (state)
      state = self.env.reset(current_load_mbps=self.config.load_flow)
      
      # 显式设置任务
      self.env.s_node = flow.src
      self.env.d_node = flow.dst
      self.env.current_flow_type = flow.flow_type
      
      curr_node = flow.src
      target_node = flow.dst
      
      # 准备 Fingerprint (Batch=1)
      fingerprint = flow.fingerprint.to(self.agent.device)
      if fingerprint.dim() == 2:
        fingerprint = fingerprint.unsqueeze(0)

      done = False
      step_count = 0
      path = [curr_node]
      
      # 逐跳寻路循环 (Rollout) 
      while not done and step_count < self.config.max_steps:
        # 1. 动态获取 Embedding (因为每一步图状态可能变化)
        with torch.no_grad():
          # 通过辅助函数获取 embedding
          node_embeds = self.agent.get_node_embeddings(state, fingerprint)
          
          # 2. 获取有效邻居信息
          # 从 state.edge_index 中提取当前节点的出边
          edge_attr = state.edge_attr
          edge_index = state.edge_index
          
          # 找出起点为 curr_node 的边
          mask = (edge_index[0] == curr_node)
          neighbor_edge_attrs = edge_attr[mask]         # 对应的边特征
          neighbor_indices = edge_index[1][mask]        # 邻居节点 ID
          global_edge_indices = torch.where(mask)[0]    # 这些边在全图 edge_index 中的下标

          # 3. Agent 决策 (Actor)
          # 注意：get_action 返回的 action_idx 是在 neighbor_indices 里的下标 (0~K-1)
          next_node, log_prob, action_local_idx = self.agent.get_action(
            state, node_embeds, curr_node, target_node, 
            neighbor_indices, neighbor_edge_attrs, deterministic=False )
          
          # 4. 估值 (Critic)
          val = self.agent.evaluate_value(node_embeds, curr_node, target_node)

        # 死胡同处理
        if next_node is None:
          reward = self.config.penalty_dead_end # 例如 -1.0
          self.memory.store(state, 0, torch.tensor(0.0), val.item(), reward, True, 
                            fingerprint, curr_node, target_node)
          break

        # 5. 环境执行
        # 找到选中的那条边的全局索引，传给 env.step
        selected_edge_idx = global_edge_indices[action_local_idx].item()
        next_state, reward, done, info = self.env.step(selected_edge_idx)
        
        # 6. 存入记忆 (必须存入 state 对象以便 update 使用)
        self.memory.store(
          state=state,            # PyG Data 对象
          action=action_local_idx,# 邻居列表下标 (Int)
          log_prob=log_prob,      # Tensor
          value=val.item(),       # Float
          reward=reward,          # Float
          is_terminal=done,       # Bool
          fingerprint=fingerprint,# Tensor
          curr_idx=curr_node,     # Int
          target_idx=target_node) # Int
        
        # 更新指针
        state = next_state
        curr_node = next_node
        path.append(curr_node)
        total_reward += reward
        step_count += 1
      
      flow_count += 1
      # 可选: 打印每条流的结果
      # vprint(f"Flow {flow_count} finished. Reward: {total_reward:.2f}", tag="Flow")

    # 一个 Epoch 采样结束，执行 PPO 更新 
    actor_loss, critic_loss = 0, 0
    if len(self.memory.rewards) > 0:
      logger.log("Updating Agent...", tag="Trainer")
      actor_loss, critic_loss = self.agent.update(self.memory)
      
      # 记录日志
      self.writer.add_scalar("Loss/Actor", actor_loss, self.global_step)
      self.writer.add_scalar("Loss/Critic", critic_loss, self.global_step)
    
    avg_reward = total_reward / max(1, flow_count)
    self.writer.add_scalar("Reward/Average", avg_reward, self.global_step)
    self.global_step += 1
        
    return avg_reward

  def run(self, traffic_generator):
    """
    主训练入口
    """
    vprint(f"Start training for {self.config.max_epochs} epochs...", tag="System")
    
    for epoch in range(self.config.max_epochs):
      start_time = time.time()
        
      # 1. 训练
      train_reward = self.train_one_epoch(traffic_generator)
      
      # 2. 保存 Checkpoint
      if epoch % self.config.save_interval == 0:
        self.save_checkpoint(epoch)
      
      duration = time.time() - start_time
      vprint(f"Epoch {epoch}: Reward={train_reward:.4f} | Time={duration:.2f}s", tag="Train")

  def save_checkpoint(self, epoch):
    """保存当前权重并清理旧的 Checkpoint """
    ckpt_path = os.path.join(self.config.ckpt_dir, f"checkpoint_{epoch}.pth") # 
    os.makedirs(self.config.ckpt_dir, exist_ok=True) # 
    
    # 1. 执行标准保存 
    torch.save(self.agent.state_dict(), ckpt_path) # 
    vprint(f"Checkpoint saved: {ckpt_path}", tag="Checkpoint")

    # 2. 自动清理旧文件 (只保留最新的 2 个)
    # 获取目录下所有的 .pth 文件
    checkpoint_files = glob.glob(os.path.join(self.config.ckpt_dir, "*.pth"))
    
    # 按文件修改时间 (mtime) 从旧到新排序
    checkpoint_files.sort(key=os.path.getmtime)
    
    # 如果文件总数超过了 2 个
    if len(checkpoint_files) > 2:
      # 选取除了最后两个（最新的）之外的所有旧文件
      files_to_delete = checkpoint_files[:-self.config.checkpoint_keep_count]
      
      for old_file in files_to_delete:
        try:
          os.remove(old_file)
          vprint(f"Removed old checkpoint: {os.path.basename(old_file)}", tag="Cleanup")
        except Exception as e:
          vprint(f"Failed to delete {old_file}: {e}", tag="Error")