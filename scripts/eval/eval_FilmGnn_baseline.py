import os
import sys
import torch
import numpy as np
import random
from tqdm import tqdm
import networkx as nx

# 引入项目模块
from src.models.ActorCritic import ActorCritic
from src.env.FlowGenerator import FlowGenerator, FLOW_PROFILES
from src.env.NetworkGenerator import TopologyGenerator, get_pyg_data_from_nx
from src.env.MininetController import calculate_qoe_reward, sample_path

# === 配置 (保持与训练一致) ===
class Config:
  # 这里的参数直接影响物理模拟，需与训练一致
  MAX_BW, MIN_BW = 90, 7.5
  MIN_LOSS, MAX_LOSS = 0, 5
  MIN_DELAY, MAX_DELAY = 0, 200
  MAX_NODES_NUM = 14
  
  MODEL_DIR = "./trained_model"
  PRETRAINED_LSTM = os.path.join(MODEL_DIR, "trained_lstm.pth")
  PRETRAINED_GNN = os.path.join(MODEL_DIR, "trained_gnn_recall_ospf.pth")
  
  GNN_DIM, LSTM_DIM, GNN_LAYERS = 256, 128, 6
  N_PACKETS = 30
  DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = Config()

# 加载指纹库
BANK_PATH = "./dataset/fingerprint_bank.pt"
if os.path.exists(BANK_PATH):
  FINGERPRINT_BANK = torch.load(BANK_PATH)
else:
  raise FileNotFoundError("❌ 指纹库未找到！")

def get_real_fingerprint_from_bank(flow_type):
  key = flow_type.name.lower()
  samples = FINGERPRINT_BANK.get(key)
  fp = random.choice(samples)
  return fp.unsqueeze(0).float().to(CONFIG.DEVICE)

def calculate_theoretical_reward(G_nx, path, flow_type):
  # 直接复用 trainer_agent_mm1.py 里的逻辑
  path_delay = 0.0
  path_loss_prob = 0.0
  path_bw = 99999.0
  for u, v in zip(path[:-1], path[1:]):
    edge = G_nx[u][v]
    path_delay += edge.get('delay', 1.0)
    avail_bw = edge.get('bandwidth', 10.0) * (1.0 - edge.get('utilization', 0.0))
    path_bw = min(path_bw, avail_bw)
    path_loss_prob += edge.get('loss', 0.0)

  mock_metrics = {
    'delay': path_delay,
    'jitter': path_delay * 0.15,
    'loss_rate': min(path_loss_prob, 1.0),
    'bandwidth': path_bw
  }
  return calculate_qoe_reward(mock_metrics, flow_type)

def run_baseline_test():
  print("====== [Baseline Test] Starting Evaluation (Before RL Fine-tuning) ======")
  
  # 1. 初始化 Agent (仅加载 Pretrain Body, 不加载 RL 权重)
  agent = ActorCritic(
    lstm_hidden_dim=CONFIG.LSTM_DIM,
    gnn_hidden_dim=CONFIG.GNN_DIM,
    gnn_layers=CONFIG.GNN_LAYERS,
    pretrained_lstm_path=CONFIG.PRETRAINED_LSTM,
    pretrained_gnn_path=CONFIG.PRETRAINED_GNN
  ).to(CONFIG.DEVICE)
  
  # 切换到评估模式
  agent.eval()
  
  # 2. 环境准备
  topo_gen = TopologyGenerator()
  flow_gen = FlowGenerator()
  base_G_nx = topo_gen.load_topology("nsfnet.graphml")
  
  rewards = []
  TEST_EPISODES = 2000 # 测试 2000 次取平均
  
  print(f"🔍 Testing {TEST_EPISODES} episodes...")
  
  # 3. 测试循环
  for i in tqdm(range(TEST_EPISODES)):
    try:
      # A. 环境随机化 (与训练保持一致)
      current_G = base_G_nx.copy()
      topo_gen.refresh_dynamic_state(current_G, difficulty=1) # 设置一个中等难度
      
      # B. 任务生成 (不做拒绝采样，模拟真实遭遇战，或者做简单的连通性检查)
      s_node, d_node = topo_gen.select_source_destination()
      flow_type, flow_profile = flow_gen.get_random_flow()
      
      if not nx.has_path(current_G, s_node, d_node):
        continue

      # C. 观测与决策
      fingerprint = get_real_fingerprint_from_bank(flow_type)
      pyg_data, _ = get_pyg_data_from_nx(current_G, s_node, d_node, CONFIG)
      pyg_data = pyg_data.to(CONFIG.DEVICE)
      
      with torch.no_grad():
        dist, val, edge_logits = agent(fingerprint, pyg_data)
          
      # D. 路径采样 (使用贪婪策略，因为是测试)
      # 加上 Dijkstra 兜底，看看基准能力
      path, _, ai_success, path_complete = sample_path(
        edge_logits, pyg_data.edge_index, s_node, d_node, 
        G_fallback=current_G, # 允许兜底
        max_steps=30, 
        greedy=True # [关键] 测试时通常使用贪婪策略
      )
      
      # E. 计算得分
      if path_complete:
        r = calculate_theoretical_reward(current_G, path, flow_profile)
        if not ai_success:
          # 如果 Baseline 太笨导致全是 Dijkstra 走的，扣分
          r -= 1 
        rewards.append(r)
      else:
        rewards.append(-1.0)
            
    except Exception as e:
      print(f"Error: {e}")
      continue
          
  # 4. 统计结果
  avg_reward = np.mean(rewards)
  print("\n" + "="*50)
  print(f"📊 Baseline Results:")
  print(f"   Avg Reward: {avg_reward:.4f}")
  print(f"   Success Count: {len([r for r in rewards if r > 0])}/{len(rewards)}")
  print("="*50)

if __name__ == "__main__":
    run_baseline_test()