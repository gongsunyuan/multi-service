import os
import sys
import time
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import numpy as np
import random
from tqdm import tqdm
from datetime import datetime

# === 导入自定义模块 ===
import MS.Env.VerbosePrint as vp
from MS.Env.VerbosePrint import vprint
from MS.Agent.ActorCritic import ActorCritic
from MS.Env.FlowGenerator import FlowGenerator, FlowType
from MS.Env.NetworkGenerator import TopologyGenerator, get_pyg_data_from_nx
from MS.Env.MininetController import (
  get_a_mininet, get_a_fingerprint, measure_path_qos, 
  sample_path, clean_flow_rules, NetworkMonitor, install_path_rules,
  calculate_qoe_reward
)

# ==============================================================================
# 1. 配置加载器
# ==============================================================================
def load_config(config_path):
  if not os.path.exists(config_path):
    raise FileNotFoundError(f"❌ 配置文件未找到: {config_path}")
  
  with open(config_path, 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)
    
  class DynamicConfig: pass
  config = DynamicConfig()
  
  # 扁平化注入属性
  for section in data.values():
    for k, v in section.items():
      setattr(config, k, v)
      
  # 路径处理
  model_dir = getattr(config, 'MODEL_DIR', './trained_model')
  config.SAVE_PATH = os.path.join(model_dir, config.SAVE_PATH)
  config.MM1_CHECKPOINT = os.path.join(model_dir, config.MM1_CHECKPOINT)
  config.PRETRAINED_LSTM = os.path.join(model_dir, config.PRETRAINED_LSTM)
  config.PRETRAINED_GNN = os.path.join(model_dir, config.PRETRAINED_GNN)
  config.FINGERPRINT_BANK = os.path.join(config.FINGERPRINT_BANK) # 假设是相对路径
  
  config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  # 创建日志目录
  start_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
  config.LOG_FILE = os.path.join(config.LOG_DIR, f"{start_time}.log")
  os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
  
  return config

# ==============================================================================
# 2. 可行性检查 (静态物理约束)
# ==============================================================================
def is_episode_solvable(G_nx, s_node, d_node, flow_type):
  """
  仅检查静态物理连通性和容量上限。
  不检查动态拥塞（那是 Agent 要解决的问题）。
  """
  # 硬编码约束或从 Config 读取
  constraints = {
    'voip':      {'max_delay': 150, 'min_bw': 0.1},
    'gaming':    {'max_delay': 60,  'min_bw': 0.5},
    'streaming': {'max_delay': 500, 'min_bw': 5.0} 
  }
  req = constraints.get(flow_type.name.lower(), constraints['streaming'])
  
  # 1. 带宽初筛 (物理容量 Capacity)
  def filter_edge(u, v):
    cap = G_nx[u][v].get('capacity', 100.0)
    return cap >= (req['min_bw'] * 1.2) # 留 20% 余量

  valid_subgraph = nx.subgraph_view(G_nx, filter_edge=filter_edge)
  
  # 2. 延迟检查 (物理传播延迟 Prop Delay)
  try:
    path = nx.dijkstra_path(valid_subgraph, s_node, d_node, weight='delay')
    total_delay = sum(G_nx[u][v].get('delay', 1.0) for u, v in zip(path[:-1], path[1:]))
    return total_delay <= req['max_delay']
  except nx.NetworkXNoPath:
    return False

# ==============================================================================
# 3. 主训练循环 (Sim-to-Real)
# ==============================================================================
def run_a2c_training(CONFIG):
  print(f"[MS] Initializing Agent on {CONFIG.DEVICE}...")
  vp.MININET_VERBOSE = True
  vp.LOG_TO_CONSOLE = False
  vp.LOG_FILE_PATH = CONFIG.LOG_FILE
  
  # --- A. 初始化 Agent ---
  agent = ActorCritic(
    lstm_hidden_dim=CONFIG.LSTM_DIM,
    gnn_hidden_dim=CONFIG.GNN_DIM,
    gnn_layers=CONFIG.GNN_LAYERS,
    pretrained_lstm_path=CONFIG.PRETRAINED_LSTM,
    pretrained_gnn_path=CONFIG.PRETRAINED_GNN
  ).to(CONFIG.DEVICE)

  # --- B. 加载 MM1 权重 (Transfer Learning) ---
  if os.path.exists(CONFIG.MM1_CHECKPOINT):
    vprint(f"[Transfer] Loading MM1 weights: {CONFIG.MM1_CHECKPOINT}")
    state_dict = torch.load(CONFIG.MM1_CHECKPOINT, map_location=CONFIG.DEVICE)
    agent.load_state_dict(state_dict)
  else:
    vprint(f"[Warning] MM1 checkpoint not found at {CONFIG.MM1_CHECKPOINT}!")

  # --- C. 解冻与优化器设置 ---
  vprint("[Config] Unfreezing GNN Body for fine-tuning...")
  param_groups = [
    # GNN 主体：极低学习率 (1e-7)，保护"常识"
    {'params': agent.gnn_model.node_embed.parameters(), 'lr': CONFIG.LR_GNN_BODY},
    {'params': agent.gnn_model.convs.parameters(),      'lr': CONFIG.LR_GNN_BODY},
    {'params': agent.gnn_model.layer_norms.parameters(),'lr': CONFIG.LR_GNN_BODY},
    # Heads & FiLM：正常微调学习率 (1e-5)
    {'params': agent.film_generator.parameters(),       'lr': CONFIG.LR_HEADS},
    {'params': agent.critic_head.parameters(),          'lr': CONFIG.LR_HEADS},
    {'params': agent.gnn_model.edge_output_head.parameters(), 'lr': CONFIG.LR_HEADS}
  ]
  
  # 必须先设置 requires_grad = True
  for param in agent.gnn_model.parameters(): param.requires_grad = True
  # LSTM 保持冻结 (指纹特征通用)
  for param in agent.lstm_body.parameters(): param.requires_grad = False
    
  agent.train()
  optimizer = optim.Adam(param_groups)

  # --- D. 环境组件 ---
  topo_gen = TopologyGenerator(CONFIG)
  flow_gen = FlowGenerator()
  base_G_nx = topo_gen.load_topology("nsfnet.graphml")
  
  total_steps = 0
  stats_reward = []
  total_iterations = CONFIG.EPOCH * CONFIG.EPISODES_PER_TOPO
  
  # --- E. 启动 Mininet ---
  try:
    with get_a_mininet(base_G_nx) as net:
      vprint(f"===========================================================")
      vprint(f"[System] Mininet Started. Nodes: {len(base_G_nx.nodes())}")
      
      monitor = NetworkMonitor(net)
      hosts = {i: net.get(f'h{i}') for i in base_G_nx.nodes()}
      
      pbar = tqdm(range(total_iterations), desc="Sim-to-Real Training")
      
      for i_step in pbar:
        progress = i_step / total_iterations
        
        # ------------------------------------------------------
        # [Step 1] 构造性保障循环：生成背景流量矩阵
        # ------------------------------------------------------
        tm_dict = {}
        valid_env = False
        
        # 课程学习：难度随进度线性增加
        # Phase 1 (Easy): 30 Mbps -> Phase 3 (Hard): 300 Mbps
        target_load = CONFIG.MIN_BG_LOAD + progress * (CONFIG.MAX_BG_LOAD - CONFIG.MIN_BG_LOAD)
        
        for retry in range(10):
          # 生成 TM (重力模型)
          current_load = target_load * random.uniform(0.8, 1.2)
          tm = flow_gen.generate_traffic_matrix(base_G_nx.nodes(), current_load)
          
          # 内存预演 (Simulation)
          sim_G = base_G_nx.copy()
          flow_gen.simulate_tm_on_graph(sim_G, tm)
          
          # 生成 Agent 任务
          s_node, d_node = topo_gen.select_source_destination()
          flow_type, _ = flow_gen.get_random_flow()
          
          # 检查物理可行性
          if is_episode_solvable(sim_G, s_node, d_node, flow_type):
            valid_env = True
            tm_dict = tm
            break
            
        if not valid_env:
          vprint("[Skip] Failed to generate valid scenario.")
          continue

        # ------------------------------------------------------
        # [Step 2] 实战部署 (Inject & Monitor)
        # ------------------------------------------------------
        clean_flow_rules(net, cookie=0xB000, mask=0xFFFF) # 清理背景流
        clean_flow_rules(net, cookie=0xA000, mask=0xF000) # 清理目标流
        
        # 注入幽灵流量 (Ghost Traffic)
        flow_gen.apply_traffic_matrix_to_mininet(
          net, tm_dict, base_G_nx, install_path_rules, duration=10
        )
        
        # 等待流量铺满网络
        time.sleep(1.5)
        
        # [感知] 同步真实网络状态到图
        monitor.sync_state_to_graph(base_G_nx)
        
        # ------------------------------------------------------
        # [Step 3] Agent 决策
        # ------------------------------------------------------
        h_src, h_dst = hosts[s_node], hosts[d_node]
        
        # 采集指纹 (需要临时通路)
        TEMP_COOKIE = 0x8888
        try:
          temp_path = nx.shortest_path(base_G_nx, s_node, d_node, weight='delay')
          install_path_rules(net, temp_path, cookie=TEMP_COOKIE, do_ping=False)
          
          fingerprint = get_a_fingerprint(
            server=h_dst, client=h_src, 
            flow_type=flow_type, 
            n_packets_to_capture=CONFIG.N_PACKETS
          ).float().to(CONFIG.DEVICE)
          
          clean_flow_rules(net, TEMP_COOKIE)
        except Exception as e:
          vprint(f"[Error] Fingerprint failed: {e}")
          continue

        # GNN 推理
        pyg_data, _ = get_pyg_data_from_nx(base_G_nx, s_node, d_node, CONFIG)
        pyg_data = pyg_data.to(CONFIG.DEVICE)
        
        dist, value_est, edge_logits = agent(fingerprint, pyg_data)
        
        # 路径采样 (含 Dijkstra 兜底)
        path, log_prob_sum, ai_success, path_complete = sample_path(
          edge_logits, pyg_data.edge_index, s_node, d_node,
          G_fallback=base_G_nx, # 兜底地图
          max_steps=50,
          greedy=False
        )

        # ------------------------------------------------------
        # [Step 4] 执行与奖励
        # ------------------------------------------------------
        reward = -2.0 # 默认断连惩罚
        
        if path_complete:
          # 下发实际路径
          install_path_rules(net, path, cookie=0xA005)
          
          # 测量真实 QoS (D-ITG)
          r_measure = measure_path_qos(h_src, h_dst, path, flow_type)
          
          # 奖励重塑
          if ai_success:
            reward = r_measure
          else:
            # 兜底惩罚
            reward = r_measure - 0.6
            vprint(f"[Penalty] Dijkstra rescue. R: {r_measure:.2f} -> {reward:.2f}")
        
        # ------------------------------------------------------
        # [Step 5] 优化 (A2C)
        # ------------------------------------------------------
        reward_tensor = torch.tensor([reward], device=CONFIG.DEVICE)
        
        # 不再使用 tanh 压缩，直接用原始奖励 (-2 ~ 1)
        advantage = reward_tensor - value_est.detach()
        actor_loss = -log_prob_sum * advantage
        critic_loss = nn.MSELoss()(value_est, reward_tensor)
        entropy = dist.entropy().mean()
        
        total_loss = actor_loss + (CONFIG.CRITIC_LOSS_COEF * critic_loss) - (CONFIG.ENTROPY_COEF * entropy)
        
        (total_loss / CONFIG.BATCH_SIZE).backward()
        
        total_steps += 1
        if total_steps % CONFIG.BATCH_SIZE == 0:
          torch.nn.utils.clip_grad_norm_(agent.parameters(), CONFIG.MAX_GRAD_NORM)
          optimizer.step()
          optimizer.zero_grad()
          
          # 打印学习率
          curr_lr = optimizer.param_groups[3]['lr'] # Head LR
          vprint(f"[Train] Step {total_steps} | LR: {curr_lr:.2e} | Reward: {reward:.2f}")
        
        stats_reward.append(reward)
        avg_r = np.mean(stats_reward[-50:])
        
        pbar.set_postfix({
          "Mode": f"{flow_type.name[:3]}",
          "Load": f"{target_load:.0f}",
          "R": f"{reward:.2f}",
          "Avg": f"{avg_r:.2f}"
        })
        
        if total_steps % 100 == 0:
           torch.save(agent.state_dict(), CONFIG.SAVE_PATH)

  except Exception as e:
    print(f"\n[Crash] {e}")
    import traceback
    traceback.print_exc()
  finally:
    # 清理残局
    os.system("sudo killall -9 ITGSend ITGRecv > /dev/null 2>&1")
    print(f"[Done] Model saved to {CONFIG.SAVE_PATH}")

if __name__ == '__main__':
  # 解析命令行参数
  parser = argparse.ArgumentParser()
  parser.add_argument('--yaml', type=str, default='config.yaml', help='Path to config file')
  args = parser.parse_args()
  
  if os.getuid() != 0:
    print("❌ Error: Must run as root (sudo) for Mininet.")
  else:
    try:
      cfg = load_config(args.yaml)
      run_a2c_training(cfg)
    except Exception as e:
      print(f"❌ Init failed: {e}")