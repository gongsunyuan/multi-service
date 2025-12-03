import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import numpy as np
import random
import math # [FIX] 确保 math 库被导入
from tqdm import tqdm
from datetime import datetime

# === Import Custom Modules ===
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
  MAX_BW = 90.0 # [FIX] 保持默认值，与 NetworkGenerator.py 保持一致
  MIN_BW = 7.5
  MIN_LOSS = 0
  MAX_LOSS = 5
  MIN_DELAY = 1.0
  MAX_DELAY = 200.0
  MAX_NODES_NUM = 14

  # --- Training Control ---
  EPOCH = 500               # 增加 Epoch 数量
  EPISODES_PER_TOPO = 800   # 减少 Episodes Per Topo (总步数 N_total = 4000)
  BATCH_SIZE = 64          # [FIX] 纯计算模式下 Batch Size 设为 32
  
  # --- Hyperparameters ---
  # [FIX] 初始 LR 调整为正常 Head LR
  LR = 1e-4                
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
  LOG_FILE_PATH=f"./train_log/a2c/{START_TIME}.log"
  DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = Config()

# [新增] 定义调度器所需学习率和熵系数
LR_GNN_BODY_P1 = 1e-7  # GNN Body 保护性 LR
LR_HEADS_P1 = 1e-4     # 初始 Heads/FiLM LR
LR_HEADS_P2 = 5e-6     # 稳定阶段 Heads LR (降低两个数量级)
ENTROPY_P1 = 0.05
ENTROPY_P3 = 0.005

# ==============================================================================
# 1. Synthetic Fingerprint Generator (模拟流量指纹)
# ==============================================================================
BANK_PATH = "./dataset/fingerprint_bank.pt"
if os.path.exists(BANK_PATH):
  vprint(f"[System] Loading Real Fingerprint Bank from {BANK_PATH}")
  FINGERPRINT_BANK = torch.load(BANK_PATH)
else:
  raise FileNotFoundError("[Error] 请先运行 tools/build_fingerprint_bank.py 生成真实指纹库！")

def get_real_fingerprint_from_bank(flow_type):
  key = flow_type.name.lower()
  samples = FINGERPRINT_BANK.get(key)
  
  if not samples:
    raise ValueError(f"Bank empty for {key}")
      
  fp = random.choice(samples) 
  
  # 增加一点点高斯噪声 (Data Augmentation)
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
    # [修正] 增加对总丢包率的检查
    total_loss = sum(G_nx[u][v].get('loss', 0.0) for u, v in zip(path[:-1], path[1:]))
    
    is_delay_ok = total_delay <= req['max_delay']
    is_loss_ok = total_loss < 0.01 # 假设硬性要求丢包小于 1%
    
    return is_delay_ok and is_loss_ok

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
    
    path_delay += edge.get('delay', 1.0)
    avail_bw = edge.get('bandwidth', 10.0) * (1.0 - edge.get('utilization', 0.0))
    path_bw = min(path_bw, avail_bw)
    
    path_loss_prob += edge.get('loss', 0.0)

  # 2. 构造虚拟的 Metrics
  mock_metrics = {
    'delay': path_delay,
    'jitter': path_delay * 0.15,
    'loss_rate': min(path_loss_prob, 1.0),
    'bandwidth': path_bw 
  }
  
  # 3. 使用 E-model 计算 Reward
  reward = calculate_qoe_reward(mock_metrics, flow_type)

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

  # -------------------------------------------------------------
  # [Phase 0] 权重加载与解冻配置
  # -------------------------------------------------------------
  
  # 1. 加载 MM1 Checkpoint (如果存在，继承策略)
  MM1_CHECKPOINT = None 
  # MM1_CHECKPOINT = "./trained_model/trained_agent_mm1.pth" 
  if not MM1_CHECKPOINT == None :
    if os.path.exists(MM1_CHECKPOINT):
      vprint(f"[Transfer] Loading MM1 agent checkpoint: {MM1_CHECKPOINT}")
      state_dict = torch.load(MM1_CHECKPOINT, map_location=CONFIG.DEVICE)
      agent.load_state_dict(state_dict)
    else:
      vprint("[Warning] MM1 Checkpoint not found! Starting from base GNN/LSTM.")

  # 2. 解冻 GNN Body (允许适应新地图)
  vprint("[Config] Unfreezing GNN Body and setting up Differential LR...")
  
  param_groups = [
    # GNN Backbone (保护性极低 LR)
    {'params': agent.gnn_model.node_embed.parameters(), 'lr': LR_GNN_BODY_P1, 'name': 'GNN_Embed'},
    {'params': agent.gnn_model.convs.parameters(), 'lr': LR_GNN_BODY_P1, 'name': 'GNN_Convs'},
    {'params': agent.gnn_model.layer_norms.parameters(), 'lr': LR_GNN_BODY_P1, 'name': 'GNN_Norms'},
    
    # Heads and FiLM Adapter (初始高学习率)
    {'params': agent.film_generator.parameters(), 'lr': LR_HEADS_P1, 'name': 'FiLM_Gen'},
    {'params': agent.critic_head.parameters(), 'lr': LR_HEADS_P1, 'name': 'Critic_Head'},
    {'params': agent.gnn_model.edge_output_head.parameters(), 'lr': LR_HEADS_P1, 'name': 'Actor_Head'}
  ]

  # 3. 必须解冻 GNN Body 的参数，才能让优化器看到它们
  for param in agent.gnn_model.parameters():
    param.requires_grad = True
  for param in agent.lstm_body.parameters():
    param.requires_grad = False
  
  # 4. 优化器必须在解冻之后初始化
  optimizer = optim.Adam(param_groups)
  scheduler = None

  topo_gen = TopologyGenerator() 
  flow_gen = FlowGenerator()
  base_G_nx = topo_gen.load_topology("nsfnet.graphml")
  
  stats_reward = []
  total_steps = 0
  update_count = 0
  
  # -------------------------------------------------------------
  # [Phase Scheduling Variables]
  # -------------------------------------------------------------
  current_head_lr = LR_HEADS_P1
  current_entropy_coef = ENTROPY_P1
  
  total_iterations = CONFIG.EPOCH * CONFIG.EPISODES_PER_TOPO
  pbar = tqdm(range(total_iterations), desc="MM1 Training")
  vp.CURRENT_PBAR = pbar
  
  # -------------------------------------------------------------
  # Main Training Loop (One-Shot Execution)
  # -------------------------------------------------------------
  for i in pbar:
    try:
      current_G = base_G_nx.copy()
      progress = i / total_iterations

      # ==========================================
      # 1. 自动化调度器 (Phase 2 & 3 Trigger Logic)
      # ==========================================
      
      # [Trigger 1: 40%] PHASE 2: 稳定学习率，消除震荡
      if progress >= 0.4 and current_head_lr == LR_HEADS_P1:
        current_head_lr = LR_HEADS_P2
        for group in optimizer.param_groups:
          if group['name'] in ['FiLM_Gen', 'Critic_Head', 'Actor_Head']:
            group['lr'] = LR_HEADS_P2
            vprint(f"[Scheduler] PHASE 2: Heads LR lowered to {LR_HEADS_P2:.1e}")

      # [Trigger 2: 80%] PHASE 3: 降低探索权重，固化策略
      if progress >= 0.8 and current_entropy_coef == ENTROPY_P1:
        RESTART_PEAK_LR = 5e-5
        for group in optimizer.param_groups:
          if group['name'] in ['FiLM_Gen', 'Critic_Head', 'Actor_Head']:
            group['lr'] = RESTART_PEAK_LR
        current_entropy_coef = ENTROPY_P3
        scheduler = torch.optimlr_scheduler.CosineAnnealingWarmRestarts(
          optimizer, T_0=200, T_mult=1, eta_min=1e-8
        )
        vprint(f"[Scheduler] PHASE 3: Entropy Coef lowered to {ENTROPY_P3:.3f}")
          
      # -------------------------------------------------------------

      # Environment Update
      topo_gen.refresh_dynamic_state(current_G, progress) # Pass progress as difficulty
      
      # Task Generation & Rejection Sampling
      s_node, d_node = -1, -1
      flow_type, flow_profile = None, None
      
      # ... (Task generation and feasibility check loop remains unchanged) ...
      for _ in range(50):
        s, d = topo_gen.select_source_destination()
        ft, fp = flow_gen.get_random_flow()
        if is_episode_solvable(current_G, s, d, ft):
          s_node, d_node, flow_type, flow_profile = s, d, ft, fp
          break
      
      if s_node == -1: 
        vprint(f"[Skip ] No solvable path found after 50 retries.")
        continue 

      # Observation
      fingerprint = get_real_fingerprint_from_bank(flow_type)
      pyg_data, _ = get_pyg_data_from_nx(current_G, s_node, d_node, CONFIG)
      pyg_data = pyg_data.to(CONFIG.DEVICE)
      
      # Agent Decision
      dist, value_est, edge_logits = agent(fingerprint, pyg_data)
      
      # Path Sampling with Dijkstra Fallback
      path, log_prob_sum, ai_success, path_complete = sample_path(
        edge_logits, 
        pyg_data.edge_index, 
        s_node, 
        d_node, 
        G_fallback=current_G, 
        max_steps=100) # [FIX] 使用 100 步
      
      # Reward Calculation & Shaping
      reward = -20.0
      base_reward = -20.0
      
      if path_complete:
        # [FIX] 使用 current_G 计算 Reward
        base_reward = calculate_theoretical_reward(current_G, path, flow_profile)
        
        if ai_success:
          # Agent 独立完成，全额奖励
          reward = base_reward
        else:
          # Dijkstra 兜底完成，惩罚 -0.6
          reward = base_reward - 0.6
          # vprint(f"[Penalty] Dijkstra rescue. Base R:{base_reward:.2f} -> R:{reward:.2f}")
      else:
        # 彻底失败 (物理断连)，惩罚最重
        reward = -2.0 
        
      # 6. Optimization
      reward_tensor = torch.tensor([reward], device=CONFIG.DEVICE)
      
      advantage = reward_tensor - value_est.detach()
      actor_loss = -log_prob_sum * advantage
      critic_loss = nn.MSELoss()(value_est, reward_tensor)
      
      entropy = dist.entropy().mean()
      # [FIX] 使用动态的 Entropy Coef
      total_loss = actor_loss + (CONFIG.CRITIC_LOSS_COEF * critic_loss) - (current_entropy_coef * entropy)
      
      (total_loss / CONFIG.BATCH_SIZE).backward()
      
      if (total_steps + 1) % CONFIG.BATCH_SIZE == 0:
        torch.nn.utils.clip_grad_norm_(agent.parameters(), CONFIG.MAX_GRAD_NORM)
        optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None: scheduler.step()
        update_count += 1
        
        # Checkpoint saving logic
        CKPT_DIR = os.path.join(CONFIG.MODEL_DIR, "checkpoints/a2c_mm1")
        os.makedirs(CKPT_DIR, exist_ok=True)
        vprint(f"[train] avg reward: {avg_r:.2f} | Updates: {update_count}")
          
      total_steps += 1
      stats_reward.append(reward)
      avg_r = np.mean(stats_reward[-50:])
    
      # Save final checkpoint
      if len(stats_reward) % 5000 == 0:
        ckpt_path = os.path.join(CKPT_DIR, f"checkpoint_{update_count}.pth")
        torch.save(agent.state_dict(), ckpt_path)
          
    except Exception as e:
      vprint(f"[Error] Step failed: {e}")
      continue

  torch.save(agent.state_dict(), CONFIG.SAVE_PATH)
  print(f"✅ MM1 Pre-training Complete! Model saved to {CONFIG.SAVE_PATH}")

if __name__ == '__main__':
  os.makedirs(CONFIG.MODEL_DIR, exist_ok=True)
  run_mm1_training()