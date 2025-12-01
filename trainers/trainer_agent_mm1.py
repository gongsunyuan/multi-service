import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import numpy as np
import random
from tqdm import tqdm
from datetime import datetime

# === Import Custom Modules ===
# 注意：这里不导入 Mininet 相关的控制器，只导入计算逻辑
from MS.Env import VerbosePrint as vp
from MS.Agent.ActorCritic import ActorCritic
from MS.Env.FlowGenerator import FlowGenerator, FlowType, FLOW_PROFILES
from MS.Env.NetworkGenerator import TopologyGenerator, get_pyg_data_from_nx
from MS.Env.MininetController import calculate_qoe_reward, sample_path # 只导入计算函数
vprint = vp.vprint

# ==============================================================================
# Configuration
# ==============================================================================
class Config:
  # ===============
  # link capacity
  #================
  MAX_BW = 90
  MIN_BW = 7.5
  MIN_LOSS = 0
  MAX_LOSS = 5
  MIN_DELAY = 0
  MAX_DELAY = 200
  MAX_NODES_NUM = 14
  # --- Training Control ---
  EPOCH = 200              # 理论训练极快，可以多跑几轮
  EPISODES_PER_TOPO = 2000 # 每个拓扑状态下多练几次
  BATCH_SIZE = 64         # 纯计算模式下，Batch 可以大一点
  
  # --- Hyperparameters ---
  LR = 1e-5               # 理论环境比较干净，学习率可以稍大
  GAMMA = 0.99
  ENTROPY_COEF = 0.05
  MAX_GRAD_NORM = 0.5
  CRITIC_LOSS_COEF = 0.5

  # --- System ---
  MODEL_DIR = "./trained_model"
  SAVE_PATH = os.path.join(MODEL_DIR, "trained_agent_mm1.pth")
  PRETRAINED_LSTM = os.path.join(MODEL_DIR, "trained_lstm.pth")
  PRETRAINED_GNN = os.path.join(MODEL_DIR, "trained_gnn_recall_ospf.pth")
  
  GNN_DIM = 256
  LSTM_DIM = 128
  GNN_LAYERS = 6
  
  # --- Environment ---
  N_PACKETS = 30
  START_TIME = datetime.now().strftime("%Y-%m-%d-%H:%M")
  LOG_FILE_PATH=f"train_log/a2c/{START_TIME}.log"
  DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = Config()

# ==============================================================================
# 1. Synthetic Fingerprint Generator (模拟流量指纹)
# ==============================================================================
BANK_PATH = "./dataset/fingerprint_bank.pt"
if os.path.exists(BANK_PATH):
  print(f"[System] Loading Real Fingerprint Bank from {BANK_PATH}")
  FINGERPRINT_BANK = torch.load(BANK_PATH)
else:
  raise FileNotFoundError("[Error] 请先运行 tools/build_fingerprint_bank.py 生成真实指纹库！")

def get_real_fingerprint_from_bank(flow_type):
  """
  从库中随机抽取一个真实的 (Size, IAT) 张量
  """
  key = flow_type.name.lower()
  samples = FINGERPRINT_BANK.get(key)
  
  if not samples:
    raise ValueError(f"Bank empty for {key}")
      
  # 随机选一个
  fp = random.choice(samples) 
  
  # 增加一点点高斯噪声 (Data Augmentation)，防止过拟合，模拟网络波动
  noise = torch.randn_like(fp) * 0.02
  fp_aug = fp + noise
  
  # 增加 Batch 维度 (1, N, 2) 并送入 GPU
  return fp_aug.unsqueeze(0).float().to(CONFIG.DEVICE)

# ==============================================================================
# 2. Feasibility Check (Same as before)
# ==============================================================================
def is_episode_solvable(G_nx, s_node, d_node, flow_type):
  constraints = {
    'voip':      {'max_delay': 150, 'min_bw': 0.1},
    'gaming':    {'max_delay': 60,  'min_bw': 0.5},
    'streaming': {'max_delay': 500, 'min_bw': 5.0} 
  }
  req = constraints.get(flow_type.name.lower(), constraints['streaming'])
  
  def filter_edge(u, v):
    edge_data = G_nx[u][v]
    capacity = edge_data.get('bandwidth', 10.0)
    return capacity >= req['min_bw']

  valid_subgraph = nx.subgraph_view(G_nx, filter_edge=filter_edge)
  
  try:
    path = nx.dijkstra_path(valid_subgraph, s_node, d_node, weight='delay')
    total_delay = sum(G_nx[u][v].get('delay', 1.0) for u, v in zip(path[:-1], path[1:]))
    return total_delay <= req['max_delay']

  except nx.NetworkXNoPath:
    return False

# ==============================================================================
# 3. Theoretical Reward Calculation
# ==============================================================================
def calculate_theoretical_reward(G_nx, path, flow_type):
  """
  不使用 Mininet，直接累加图属性计算 QoS
  """
  path_delay = 0.0
  path_loss_prob = 0.0
  path_bw = 99999.0
  
  # 1. 累加路径属性
  for u, v in zip(path[:-1], path[1:]):
    edge = G_nx[u][v]
    
    # Delay 累加
    path_delay += edge.get('delay', 1.0)
    
    # Bandwidth 取瓶颈 (同时考虑物理带宽和利用率剩余)
    # avail = bw * (1 - util)
    avail_bw = edge.get('bandwidth', 10.0) * (1.0 - edge.get('utilization', 0.0))
    path_bw = min(path_bw, avail_bw)
    
    # Loss 概率合并: 1 - (1-l1)(1-l2)...
    # 近似为累加 (当 loss 很小时)
    path_loss_prob += edge.get('loss', 0.0)

  # 2. 构造虚拟的 Metrics
  mock_metrics = {
    'delay': path_delay,
    'jitter': path_delay * 0.15, # 经验公式: Jitter 约为延迟的 10-20%
    'loss_rate': min(path_loss_prob, 1.0), # 0.0 ~ 1.0
    'bandwidth': path_bw }                 # Mbps
  # 3. 使用 E-model 计算 Reward
  reward = calculate_qoe_reward(mock_metrics, FLOW_PROFILES[flow_type])

  return reward

# ==============================================================================
# Main Training Loop (No Mininet!)
# ==============================================================================
def run_mm1_training():
  vp.LOG_FILE_PATH=CONFIG.LOG_FILE_PATH
  vprint(f"[MM1] Initializing Pure Theoretical Training...")
  
  agent = ActorCritic(
    lstm_hidden_dim=CONFIG.LSTM_DIM,
    gnn_hidden_dim=CONFIG.GNN_DIM,
    gnn_layers=CONFIG.GNN_LAYERS,
    pretrained_lstm_path=CONFIG.PRETRAINED_LSTM,
    pretrained_gnn_path=CONFIG.PRETRAINED_GNN
  ).to(CONFIG.DEVICE)

  # MM1_CHECKPOINT = "checkpoints/a2c/checkpoint.pth" # 确保文件名对
  # if os.path.exists(MM1_CHECKPOINT):
  #   vprint(f"[Transfer] Loading pre-trained MM1 agent from: {MM1_CHECKPOINT}")
  #   state_dict = torch.load(MM1_CHECKPOINT, map_location=CONFIG.DEVICE)
  #   agent.load_state_dict(state_dict)
  # else:
  #   vprint("[Warning] MM1 Checkpoint not found! Starting from scratch.")
      
  agent.train()
  optimizer = optim.Adam(filter(lambda p: p.requires_grad, agent.parameters()), lr=CONFIG.LR)
  
  topo_gen = TopologyGenerator(CONFIG) # 这里不需要 Mininet config
  flow_gen = FlowGenerator()
  
  # 加载 NSFNet 结构
  base_G_nx = topo_gen.load_topology("nsfnet.graphml")
  
  stats_reward = []
  total_steps = 0
  update_count = 0
  
  # 只需要一个巨大的进度条
  total_iterations = CONFIG.EPOCH * CONFIG.EPISODES_PER_TOPO
  pbar = tqdm(range(total_iterations), desc="MM1 Training")
  vp.CURRENT_PBAR = pbar
  for i in pbar:
    try:
      current_G = base_G_nx.copy()
      # 1. [Environment] 动态刷新网络状态 (MM1 数学模型)
      # 这会随机生成 Utilization，并算出 Delay/Loss
      progress = 1

      topo_gen.refresh_dynamic_state(current_G, progress)
      
      # 2. [Task] 随机生成任务 & 拒绝采样
      s_node, d_node = -1, -1
      flow_type, flow_profile = None, None
      
      for _ in range(50):
        s, d = topo_gen.select_source_destination()
        ft, fp = flow_gen.get_random_flow()
        if is_episode_solvable(current_G, s, d, ft):
          s_node, d_node, flow_type, flow_profile = s, d, ft, fp
          break
      
      if s_node == -1: 
        vprint(f"[Skip ] No solvable path found after 50 retries. Flow type: {flow_type}")
        continue # Skip impossible tasks

      # 3. [Observation] 构造输入
      fingerprint = get_real_fingerprint_from_bank(flow_type)

      # B. 图特征 (PyG Data)
      pyg_data, _ = get_pyg_data_from_nx(current_G, s_node, d_node, CONFIG)
      pyg_data = pyg_data.to(CONFIG.DEVICE)
      
      # 4. [Agent] 决策
      dist, value_est, edge_logits = agent(fingerprint, pyg_data)
      
      path, log_prob_sum, success, path_complete = sample_path(
        edge_logits, 
        pyg_data.edge_index, 
        s_node, 
        d_node, 
        G_fallback=current_G, 
        max_steps=30)
      
      # 5. [Reward] 纯数学计算
      reward = -20.0
      if success:
        reward = calculate_theoretical_reward(current_G, path, flow_type)
      elif path_complete:
        reward = calculate_theoretical_reward(current_G, path, flow_type)-1
        # vprint(f"[Penalty] agent find path failed ! reward = {reward}")
      else:
        vprint(f"[Error] failed completely !")

      # 6. [Optimize] 梯度下降
      reward_tensor = torch.tensor([reward], device=CONFIG.DEVICE)
      reward_norm = torch.tanh(reward_tensor / 5.0) # Squash reward
      
      advantage = reward_norm - value_est.detach()
      actor_loss = -log_prob_sum * advantage
      critic_loss = nn.MSELoss()(value_est, reward_norm)
      entropy = dist.entropy().mean()
      
      total_loss = actor_loss + (CONFIG.CRITIC_LOSS_COEF * critic_loss) - (CONFIG.ENTROPY_COEF * entropy)
      
      (total_loss / CONFIG.BATCH_SIZE).backward()
      
      if (total_steps + 1) % CONFIG.BATCH_SIZE == 0:
        torch.nn.utils.clip_grad_norm_(agent.parameters(), CONFIG.MAX_GRAD_NORM)
        optimizer.step()
        optimizer.zero_grad()
        update_count += 1 # 计数 +1
        CKPT_DIR = "checkpoints/a2c/"
        ckpt_path = os.path.join(CKPT_DIR, f"checkpoint_{update_count}.pth")
        torch.save(agent.state_dict(), ckpt_path)
        vprint(f"[train] avg reward: {avg_r:.2f}")
          
      total_steps += 1
      stats_reward.append(reward)
      avg_r = np.mean(stats_reward[-50:])
    
      # Save periodically
      if len(stats_reward) % 500 == 0:
        torch.save(agent.state_dict(), CONFIG.SAVE_PATH)
          
    except Exception as e:
      vprint(f"[Error] Step failed: {e}")
      continue

  # Save final
  torch.save(agent.state_dict(), CONFIG.SAVE_PATH)
  print(f"[MM1] Pre-training Complete! Model saved to {CONFIG.SAVE_PATH}")

if __name__ == '__main__':
  # 不需要 sudo check
  os.makedirs(CONFIG.MODEL_DIR, exist_ok=True)
  run_mm1_training()