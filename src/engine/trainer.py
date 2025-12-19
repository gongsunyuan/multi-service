# src/engine/trainer.py
import torch
import time
from torch.utils.tensorboard import SummaryWriter

class PPOTrainer:
  def __init__(self, agent, env, memory, config):
    self.agent = agent
    self.env = env
    self.memory = memory
    self.config = config
    self.writer = SummaryWriter(log_dir=config.TB_LOG_DIR)

    # 初始化 Logger, Tensorboard 等

  def train_one_epoch(self, agent, env, memory, traffic_generator, config):
    """
    一个 Epoch 的训练逻辑
    agent: SDNPPOAgent 实例
    env: MininetWrapper 实例 (包含图拓扑)
    memory: PPOMemory 实例
    traffic_generator: 负责生成一批业务流 (fingerprint, src, dst)
    """

    flow_count = 0
    total_reward = 0
    
    # 1. 获取当前网络拓扑的 PyG Data 对象
    graph_data = env.get_graph_data() 
    
    # 2. 生成这一批次的业务流 (例如 10 条流)
    flows = traffic_generator.generate_batch(config.batch_size)
    
    for flow in flows:
      # --- 针对每一条流，先生成业务感知的 Embeddings ---
      # 只要拓扑没变，业务流没变，node_embeds 在这条流的寻路过程中是不变的
      with torch.no_grad(): # 采样阶段不需要梯度
        # film_params = agent.policy.film_gen(flow.fingerprint)
        # node_embeds = agent.policy.gnn(graph_data.x, ..., film_params)
        # 这里我们假设 agent 封装了一个接口来获取 embeds
        node_embeds = agent.get_node_embeddings(graph_data, flow.fingerprint)
      
      curr_node = flow.src
      target_node = flow.dst
      done = False
      step_count = 0
      path = [curr_node]
      
      # --- 逐跳寻路循环 ---
      while not done and step_count < config.max_steps:
        # 1. 获取当前节点的有效邻居及其边特征
        # 注意：env.get_valid_neighbors 应该剔除已访问节点防止死循环
        neighbors, edge_attrs = env.get_neighbors_info(curr_node, path)
        
        if not neighbors: # 死胡同
          reward = config.penalty_dead_end
          memory.store(None, None, None, 0, reward, True)
          break
            
        # 2. Agent 决策
        # 调用 evaluate_value 获取当前状态的价值 (Critic)
        val = agent.policy.evaluate_value(node_embeds, curr_node, target_node)
        
        # 调用 get_action 选下一跳 (Actor)
        next_node, log_prob, _ = agent.choose_action(
          node_embeds, curr_node, target_node, neighbors, edge_attrs
        )
        
        # 3. 环境步进：在 Mininet 中“模拟”走这一步
        # 注意：这里的 reward 可能是即时奖励（如链路利用率反馈）
        # 或者设为 0，等到达终点再给总奖赏
        step_reward, done, info = env.step(curr_node, next_node, target_node)
        
        # 4. 存入记忆
        memory.store(
          state=None, # 如果 update 时重新跑 GNN，这里可以存索引
          action=neighbors.index(next_node), # 存的是邻居列表里的索引
          log_prob=log_prob,
          value=val,
          reward=step_reward,
          is_terminal=done
        )
        
        # 5. 更新状态
        curr_node = next_node
        path.append(curr_node)
        total_reward += step_reward
        step_count += 1
          
      flow_count += 1
      # 在真实 SDN 环境中，一条流规划完后，可能需要更新 env 的带宽占用
      env.update_network_resources(path, flow.bandwidth)

    # --- 所有流跑完后，执行 PPO 更新 ---
    if len(memory.rewards) > 0:
      agent.update(memory)
        
    return total_reward / flow_count

  def run(self):
    """
    主训练循环
    """

    print(f"Start training for {self.config.max_epochs} epochs...")
    
    for epoch in range(self.config.max_epochs):
      start_time = time.time()
        
      # 1. 训练一个 Epoch
      train_reward = self.train_one_epoch()
      
      # 2. (可选) 评估/测试
      # val_reward = self.evaluate()
      
      # 3. 保存模型
      if epoch % self.config.save_interval == 0:
        self.save_checkpoint(epoch)
          
      print(f"Epoch {epoch}: Reward={train_reward:.2f}")

  def save_checkpoint(self, epoch):
    torch.save(self.agent.state_dict(), f"{self.config.ckpt_dir}/model_{epoch}.pth")


    