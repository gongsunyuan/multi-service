import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import numpy as np
from tqdm import tqdm
from torch.distributions import Categorical
from torch_geometric.nn.glob import global_mean_pool

# === 导入自定义模块 ===
# 请确保项目结构正确，并且 __init__.py 文件存在
from MS.Agent.ActorCritic import ActorCritic
from MS.Env.FlowGenerator import FlowGenerator
from MS.Env.TensorLog import append_matrix_to_file
from MS.Env.NetworkGenerator import TopologyGenerator, get_pyg_data_from_nx
from MS.Env.MininetController import get_a_mininet, get_a_fingerprint, measure_path_qos, sample_path, clean_flow_rules, NetworkMonitor, install_path_rules

# ================= 配置参数 =================
class Config:
  # --- 拓扑生成参数 ---
  #
  M_BA = 2
  MIN_BW = 20.0
  MAX_BW = 200.0
  MIN_LOSS = 0.0
  MAX_LOSS = 3.0
  MIN_DELAY = 1.0
  MAX_DELAY = 50.0
  MIN_NODES_NUM = 10
  MAX_NODES_NUM = 20

  # --- 训练控制 ---
  EPOCH = 10 
  BATCH_SIZE = 16
  EPISODES_PER_TOPO = 64   # 每套拓扑训练多少个流 (复用次数)
  
  # --- 优化器参数 ---
  LR = 1e-4                # 学习率 (Phase 2 微调需要较小 LR)
  GAMMA = 0.99             # 折扣因子
  ENTROPY_COEF = 0.01      # 熵正则化 (鼓励探索)
  MAX_GRAD_NORM = 0.5      # 梯度裁剪阈值
  CRITIC_LOSS_COEF = 0.5   # Value Loss 权重

  # --- 路径配置 ---
  MODEL_DIR = "./trained_model"
  SAVE_PATH = os.path.join(MODEL_DIR, "a2c_final.pth")
  # 确保这些预训练模型文件存在
  PRETRAINED_LSTM = os.path.join(MODEL_DIR, "trained_lstm.pth") 
  PRETRAINED_GNN = os.path.join(MODEL_DIR, "trained_gnn_recall_ospf.pth") # 注意文件名拼写是否与您实际一致

  # --- 模型结构 (需与定义保持一致) ---
  GNN_DIM = 256
  LSTM_DIM = 128
  GNN_LAYERS = 6
  
  # --- 环境参数 ---
  N_PACKETS = 30           # 抓包数量 (指纹长度)
  DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = Config()

# ================= 主训练循环 =================

def run_a2c_training():
  # 1. 初始化模型
  print(f"[ms] 正在初始化 A2C Agent (Device: {CONFIG.DEVICE})...")
  agent = ActorCritic(
    lstm_hidden_dim=CONFIG.LSTM_DIM,
    gnn_hidden_dim=CONFIG.GNN_DIM,
    gnn_layers=CONFIG.GNN_LAYERS,
    pretrained_lstm_path=CONFIG.PRETRAINED_LSTM,
    pretrained_gnn_path=CONFIG.PRETRAINED_GNN).to(CONFIG.DEVICE)
    
  agent.train() # 开启训练模式 (主要是 Dropout/BatchNorm, 虽然这里用得少)
  
  # 优化器 (主要训练 FiLM 和新 Heads)
  optimizer = optim.Adam(filter(lambda p: p.requires_grad, agent.parameters()), lr=CONFIG.LR)
  
  # 2. 初始化生成器
  topo_gen = TopologyGenerator(CONFIG)
  flow_gen = FlowGenerator()
  
  # 计算循环次数
  
  total_episodes = 0
  stats_reward = []

  print(f"[ms] 开始训练: 共 {CONFIG.EPOCH-1} 次拓扑变更, 总计 {CONFIG.EPOCH} 回合")
  
  # --- 外层循环：拓扑生命周期 ---
  for topo_idx in range(CONFIG.EPOCH):
      
    # A. 生成新拓扑
    G_nx = topo_gen.generate_topology()
    
    try:
      # B. 启动 Mininet (Context Manager)
      with get_a_mininet(G_nx) as net:
        print(f"[Topo {topo_idx+1}] 启动 Mininet success (Nodes: {len(G_nx.nodes())})...")
        monitor = NetworkMonitor(net)
        # 预取节点对象
        hosts = {i: net.get(f'h{i}') for i in G_nx.nodes()}
        
        # 内层循环：流训练 (Mininet 复用) 
        pbar = tqdm(range(CONFIG.EPISODES_PER_TOPO), desc=f"Topo {topo_idx+1}", leave=False)
        
        for i_step, _ in enumerate(pbar):
          # print("="*50)
          # print(f"flow {i_step}, {_}")
          # 1. 环境准备
          clean_flow_rules(net) # 清理上一回合规则
          s_node, d_node = topo_gen.select_source_destination()
          h_src, h_dst = hosts[s_node], hosts[d_node]
          # print(f"[ms] env ready !")
          # 2. 获取状态 (State)

          # 2.1 采集指纹 (耗时操作)
          TEMP_COOKIE = 0x8888
          try:
            # 计算物理最短路作为临时通路
            temp_path = nx.shortest_path(G_nx, source=s_node, target=d_node)
            install_path_rules(net, temp_path, cookie=TEMP_COOKIE)
            time.sleep(0.05) # 给 OVS 一点时间生效
          except nx.NetworkXNoPath:
            print(f"[Error]: No path between {s_node} and {d_node}")
            continue
          
          flow_type, flow_profile = flow_gen.get_random_flow()
          fingerprint = get_a_fingerprint(
            server=h_dst, client=h_src, 
            flow_type=flow_type, 
            n_packets_to_capture=CONFIG.N_PACKETS).float().to(CONFIG.DEVICE) # Shape: (1, N, 2)
          
          # append_matrix_to_file(fingerprint.detach().cpu(), "test.log", i_step)
          # 2.2 获取图特征
          monitor.sync_state_to_graph(G_nx)

          for u, v in G_nx.edges():
            if G_nx.has_edge(u, v):
              # print(f"     链路 {u}->{v} 利用率: {G_nx[u][v]['utilization']:.2%}")
              break

          pyg_data, _ = get_pyg_data_from_nx(G_nx, s_node, d_node, CONFIG)
          pyg_data = pyg_data.to(CONFIG.DEVICE)
          
          # 3. 模型推理 (Manual Forward 以获取中间变量)
          # print("[ms] agent forwarding")
          dist, value_est, edge_logits = agent(fingerprint, pyg_data)
          # print("[ms] agent forward success")
          # 4. 动作采样 (Action)
          path, log_prob_sum, success = sample_path(
            edge_logits, pyg_data.edge_index, s_node, d_node, max_steps=30)
          
          # 5. 执行与奖励 (Reward)
          if not success:
            reward = -10.0
          else:
            install_path_rules(net, path)
            # 测量真实 QoS
            reward = measure_path_qos(h_src, h_dst, path, flow_type)
          
          # 归一化奖励 (简单除以常数，防止梯度过大)
          # reward_norm = reward / 10.0 
          reward_tensor = torch.tensor([reward], device=CONFIG.DEVICE)
          reward_tanh = torch.tanh(reward_tensor / 10.0)
          
          # 6. 反向传播 (Update)
          # Advantage
          advantage = reward_tanh - value_est.detach()
          
          # Losses
          actor_loss = -log_prob_sum * advantage
          critic_loss = nn.MSELoss()(value_est, reward_tanh)
          
          total_loss = actor_loss + CONFIG.CRITIC_LOSS_COEF * critic_loss
          (total_loss / CONFIG.BATCH_SIZE).backward()     # grad accumulate

          if (i_step + 1) % CONFIG.BATCH_SIZE == 0:
            torch.nn.utils.clip_grad_norm_(agent.parameters(), CONFIG.MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad() # 更新完清空梯度
          
          # 7. 记录与保存
          stats_reward.append(reward)
          avg_r = np.mean(stats_reward[-50:])
          pbar.set_postfix({"R": f"{reward:.1f}", "Avg": f"{avg_r:.1f}", "L": f"{total_loss.item():.2f}"})
          
          if total_episodes % 100 == 0:
            torch.save(agent.state_dict(), CONFIG.SAVE_PATH)

    except Exception as e:
      print(f"\n[Error] Topo {topo_idx} 异常中断: {e}")
      import traceback
      traceback.print_exc()
      continue # 尝试下一个拓扑，而不是直接退出

  print(f"训练结束。最终模型已保存至 {CONFIG.SAVE_PATH}")

if __name__ == '__main__':
  if os.getuid() != 0:
    print("错误: Mininet 必须以 root 权限运行 (sudo python3 ...)")
  else:
    os.makedirs(CONFIG.MODEL_DIR, exist_ok=True)
    # 确保日志目录存在 (如果 MininetController 需要写日志)
    os.makedirs("./train-log", exist_ok=True)
    run_a2c_training()
