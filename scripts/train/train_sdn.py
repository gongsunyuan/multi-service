import os
from pathlib import Path
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
from mininet.cli import CLI
sys.path.append(os.getcwd())
import src.utils.VerbosePrint as vp
from src.utils.VerbosePrint import vprint
from src.models.ActorCritic import ActorCritic
from src.utils.MyParaser import TopoParaser
from src.env.FlowGenerator import FlowGenerator, FlowType, FLOW_PROFILES
from src.env.MininetController import (
  get_a_mininet, 
  get_a_fingerprint, 
  measure_path_qos, 
  sample_path, 
  clean_flow_rules, 
  install_path_rules, 
  calculate_qoe_reward,
  vprint_network_status,
  vprint_path_status,
  NetworkMonitor)
from src.env.NetworkGenerator import get_pyg_data_from_nx, TopologyGenerator

# === 全局常量 ===
AGENT_COOKIE = 0xA001  # 你的目标流 Cookie
BG_COOKIE    = 0xB000  # 背景流 Cookie
BG_MASK      = 0xF000

# 实例化生成器
FLOW_GEN = FlowGenerator()
TOPO_GEN = TopologyGenerator()

# ==============================================================================
# 1. 配置加载器
# ==============================================================================
def load_config(config_path):

  config_path = os.path.join('config/', config_path)
  
  if not os.path.exists(config_path):
    raise FileNotFoundError(f"[Error] Config not found: {config_path}")
  
  with open(config_path, 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)
    
  class DynamicConfig: pass
  config = DynamicConfig()
  for section in data.values():
    for k, v in section.items():
      setattr(config, k, v)
  
  base_dir = Path(__file__).resolve().parents[2]

  # 处理相对路径
  if hasattr(config, 'PRETRAINED_DIR'):
    model_dir = os.path.join(base_dir, getattr(config, 'PRETRAINED_DIR'))
    for path_attr in ['PRETRAINED_LSTM', 'PRETRAINED_GNN', 'PRETRAINED_MM1']:
      if hasattr(config, path_attr):
        setattr(config, path_attr, os.path.join(model_dir, getattr(config, path_attr)))
        print(f"{path_attr}: {getattr(config, path_attr)}")

  # checkpoint_dir
  if hasattr(config, 'CHECKPOINT_DIR'):
    checkpoint_dir = os.path.dirname(os.path.join(base_dir, getattr(config, 'CHECKPOINT_DIR')))
    setattr(config, 'CHECKPOINT_PATH', os.path.join(checkpoint_dir, datetime.now().strftime("sdn_exp_%Y%m%d%H%M%S")))
    p = Path(config.CHECKPOINT_PATH)
    p.mkdir(parents=True, exist_ok=True)

  topo_dir = os.path.join(base_dir, getattr(config, 'TOPOLOGIES_DIR'))
  setattr(config, 'TOPOLOGY_FILE', os.path.join(topo_dir, getattr(config, 'TOPOLOGY_FILE')))
  
  setattr(config, 'FINGERPRINT_BANK', os.path.join(base_dir, getattr(config, 'FINGERPRINT_BANK')))

  # log_dir
  log_dir = os.path.dirname(os.path.join(base_dir, getattr(config, 'LOG_DIR')))
  log_dir_name = f"sdn_train_{datetime.now().strftime("%Y%m%d%H%M%S")}"
  setattr(config, "LOG_FILE_DIR", os.path.join(log_dir, log_dir_name))
  p = Path(config.LOG_FILE_DIR)
  p.mkdir(parents=True, exist_ok=True)
  
  
  return config

# ==============================================================================
# 2. 背景流量管理 (The TCP Hell Manager)
# ==============================================================================

def refresh_background_traffic(net, topo_graph, load_mbps=600.0):
  """
  刷新背景流量：清理旧流 -> 生成新矩阵 -> 注入 Mininet
  """
  vprint(f"[BG-Manager] Refreshing Background Traffic (Target: {load_mbps} Mbps)...")
  
  # 1. 清理旧的背景流
  clean_flow_rules(net, cookie=BG_COOKIE, mask=BG_MASK)
  os.system("sudo killall -9 ITGSend ITGRecv > /dev/null 2>&1")
  time.sleep(1.0)
  
  # 2. 生成新的流量矩阵 (Gravity Model)
  nodes = list(topo_graph.nodes())
  tm_dict = FLOW_GEN.generate_traffic_matrix(nodes, topo_graph, total_load_mbps=load_mbps)
  
  # tm_dict = FLOW_GEN.clip_tm_to_capacity(tm_dict, topo_graph, max_util=0.95)
  # FLOW_GEN.print_theoretical_tm_load(topo_graph, tm_dict)
  # 3. 在图上模拟流量矩阵，更新边的利用率和延迟
  topo_graph = FLOW_GEN.simulate_tm_on_graph(topo_graph, tm_dict)
  # 4. 注入物理网络 (Sim-to-Real 核心)
  # 注意：这里会自动处理 ToS=184 和 Drop 规则
  FLOW_GEN.apply_traffic_matrix_to_mininet(
    net, 
    tm_dict, 
    topo_graph, 
    install_rules_func=install_path_rules, 
    duration=600 # 只要比一个 Epoch 长即可
  )
  
  # 5. 等待流量稳定 (TCP 慢启动需要时间)
  vprint("[BG-Manager] Waiting 5s for congestion stabilization...")
  time.sleep(5.0)

  return topo_graph

def is_episode_solvable(G, s_node, d_node, flow_type_str):
  """
  检查在当前背景流压力下，是否存在满足 QoS 要求的路径。
  基于 G 中的理论状态 (utilization, delay 等) 进行快速预判。
  """
  # 1. 获取业务约束 (Hard Constraints)
  # 这里简单定义一下，或者从 FLOW_PROFILES 获取
  constraints = {
    'VOIP':      {'max_delay': 150, 'min_bw': 0.1},
    'GAMING':    {'max_delay': 60,  'min_bw': 0.5},
    'STREAMING': {'max_delay': 1000, 'min_bw': 2.0} # 视频对带宽要求高
  }
  
  req = constraints.get(flow_type_str, constraints['VOIP'])
  
  # 2. 定义过滤函数
  def filter_edge(u, v):
    edge_data = G[u][v]
    # A. 检查剩余带宽
    # capacity (Mbps) * (1 - utilization)
    cap = edge_data.get('bandwidth', 10.0)
    util = edge_data.get('utilization', 0.0)
    residual_bw = cap * (1.0 - util)
    
    if residual_bw < req['min_bw']:
      return False
    return True

  # 3. 创建视图 (只包含带宽达标的边)
  view = nx.subgraph_view(G, filter_edge=filter_edge)
  
  # 4. 检查连通性 & 延迟
  try:
    # 在过滤后的图中找最短延迟路径
    path = nx.shortest_path(view, source=s_node, target=d_node, weight='delay')
    
    # 计算这条路径的总延迟
    total_delay = 0
    for i in range(len(path)-1):
      u, v = path[i], path[i+1]
      total_delay += G[u][v].get('delay', 0.0)
      
    if total_delay > req['max_delay']:
      return False # 虽然通，但延迟超标
      
    return True # 通过检查
    
  except nx.NetworkXNoPath:
    return False # 物理不连通 (带宽不够)

def get_real_fingerprint_from_bank(flow_type, FINGERPRINT_BANK, CONFIG):
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
# 3. 训练主循环
# ==============================================================================
def train():
  # --- 初始化检查 ---
  if os.geteuid() != 0:
    print("[Error] Must run as root (sudo) to control Mininet.")
    sys.exit(1)

  # --- 参数解析 ---
  parser = TopoParaser()
  args = parser.parse_args()
  
  CONFIG = load_config(args.yaml)
  vp.LOG_TO_CONSOLE = args.verbose # 训练时减少刷屏
  vp.LOG_FILE_PATH = os.path.join(getattr(CONFIG, 'LOG_FILE_DIR', None), 'debug.log')
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  setattr(CONFIG, 'DEVICE', device)
  print(f"=== SDN Agent Training (TCP Hell Edition) ===")
  print(f"Device: {device} | Batch: {CONFIG.BATCH_SIZE}")

  # --- 加载指纹库 ---
  if os.path.exists(CONFIG.FINGERPRINT_BANK):
    print(f"[Init] Loading fingerprint bank from {CONFIG.FINGERPRINT_BANK}...")
    fingerprint_bank = torch.load(CONFIG.FINGERPRINT_BANK)
  else:
    print(f"[Error] : Fingerprint bank not found! {CONFIG.FINGERPRINT_BANK}")
    sys.exit(1)

  # --- 初始化 Agent ---
  agent = ActorCritic(
    lstm_hidden_dim=CONFIG.LSTM_DIM,
    gnn_hidden_dim=CONFIG.GNN_DIM,
    gnn_node_dim=getattr(CONFIG, 'GNN_NODE_DIM', 10), # 根据你的 feature 维度调整
    gnn_layers=CONFIG.GNN_LAYERS,
    pretrained_lstm_path=CONFIG.PRETRAINED_LSTM,
    pretrained_gnn_path=CONFIG.PRETRAINED_GNN).to(device)
  
  if args.checkpoint and os.path.exists(args.checkpoint):
    print(f"[Init] Loading Agent checkpoint from {args.checkpoint}...")
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))

  elif os.path.exists(CONFIG.PRETRAINED_MM1):
    print(f"[Init] Loading MM1 Pretrained Weights from {CONFIG.PRETRAINED_MM1}...")
    state_dict = torch.load(CONFIG.PRETRAINED_MM1, map_location=device)
    
    # 注意：如果 MM1 的结构和 SDN Agent 完全一致，直接加载
    # 如果不一致（比如 Head 维度不同），你需要只加载部分 key
    agent.load_state_dict(state_dict, strict=False) 
  else:
    print("[Init] MM1 weights not found, starting Actor/Critic from scratch.")
    
  param_groups = [
    # GNN Backbone (保护性极低 LR)
    {'params': agent.gnn_model.node_embed.parameters(), 'lr': CONFIG.LR_GNN_BODY, 'name': 'GNN_Embed'},
    {'params': agent.gnn_model.convs.parameters(), 'lr': CONFIG.LR_GNN_BODY, 'name': 'GNN_Convs'},
    {'params': agent.gnn_model.layer_norms.parameters(), 'lr': CONFIG.LR_GNN_BODY, 'name': 'GNN_Norms'},
    
    # Heads and FiLM Adapter (初始高学习率)
    {'params': agent.film_generator.parameters(), 'lr': CONFIG.LR_HEADS, 'name': 'FiLM_Gen'},
    {'params': agent.critic_head.parameters(), 'lr': CONFIG.LR_HEADS, 'name': 'Critic_Head'},
    {'params': agent.gnn_model.edge_output_head.parameters(), 'lr': CONFIG.LR_HEADS, 'name': 'Actor_Head'}]

  # 分层学习率
  optimizer = optim.Adam(param_groups)

  # --- 加载拓扑图 ---
  try:
    base_G_nx = TOPO_GEN.load_topology(CONFIG.TOPOLOGY_FILE)
  except:
    print("[Error]: nsfnet.graphml not found.")
    sys.exit(1)

  # ==========================================
  # 启动 Mininet 环境 (Context Manager)
  # ==========================================
  # 注意：get_a_mininet 内部现在包含了 Disable TSO/GSO 的关键代码
  with get_a_mininet(base_G_nx) as net:
    print("[System] Mininet Started. TSO/GSO Disabled.")
    monitor = NetworkMonitor(net)
    
    total_episodes = CONFIG.EPOCH * CONFIG.EPISODES_PER_TOPO
    num_epochs = total_episodes // CONFIG.BATCH_SIZE
    
    stats_rewards = []
    
    # --- Epoch Loop ---
    for epoch in range(num_epochs):
      
      # [Step 1] 刷新环境 (制造 TCP 地狱)
      # 课程学习：负载从 150 增加到 250 Mbps
      curr_load = CONFIG.START_LOAD + (epoch / num_epochs) * (CONFIG.END_LOAD - CONFIG.START_LOAD)
      current_G = refresh_background_traffic(net, base_G_nx, load_mbps=curr_load)
  
      # --- Batch Loop ---
      optimizer.zero_grad()
      batch_loss = 0
      batch_rewards = []

      
      # [Step 2] 感知网络 (Sync Monitor)
      # 这步至关重要！读取真实的 tx_bytes 和 backlog，写入 current_G
      vprint("[Monitor] Syncing physical network state to Graph...")
      for _ in range(3):
        current_G = monitor.sync_state_to_graph(current_G, duration=0.5)
      vprint_network_status(current_G)

      with tqdm(total=CONFIG.BATCH_SIZE, desc=f"Ep {epoch+1}/{num_epochs} [Load {curr_load:.0f}]", leave=False) as pbar:
        for _ in range(CONFIG.BATCH_SIZE):
          current_G = monitor.sync_state_to_graph(current_G, duration=0.5)
          vp.CURRENT_PBAR = pbar
          max_try = 10
          valid_scenario = False
          
          # 临时变量，用于保存选中的参数
          selected_s = None
          selected_d = None
          selected_flow_type = None
          
          for _ in range(max_try):
            # 1. 随机选择源宿节点
            s, d = TOPO_GEN.select_source_destination()
            flow_type_enum, _ = FLOW_GEN.get_random_flow()
            t_str = flow_type_enum.name 
            
            # 4. 可行性检查
            # 注意：current_G 必须已经包含了 simulate_tm_on_graph 的结果
            if is_episode_solvable(current_G, s, d, t_str):
              selected_s = s
              selected_d = d
              selected_flow_type = flow_type_enum
              valid_scenario = True
              break
          
          if not valid_scenario:
            # 如果随了10次都不行，说明网络太堵了，跳过这次采样
            vprint("[Error] No satisfied path !!!")
            vprint(f"[Ghost Flow] Try to low down the curr_load: {curr_load} to {curr_load*0.9}")
            curr_load = curr_load*0.9
            current_G = refresh_background_traffic(net, base_G_nx, load_mbps=curr_load)
            current_G = monitor.sync_state_to_graph(current_G, duration=0.1)
            continue
          vprint(f"[Agent] s_node: {selected_s} d_node: {selected_d} flow-type: {selected_flow_type}")
          
          try:
            # 直接调用你集成好的函数
            fingerprint = get_real_fingerprint_from_bank(selected_flow_type, fingerprint_bank, CONFIG)
          except ValueError as e:
            # 防止指纹库里没有对应类型的指纹导致崩溃
            print(f"[Warn] Fingerprint error: {e}")
          vprint(f"[Agent] Fingerprint got")
          s_node = selected_s
          d_node = selected_d
          topo_data, _ = get_pyg_data_from_nx(current_G, s_node, d_node, config=CONFIG)
          topo_data = topo_data.to(device)
            
          # B. Agent 决策
          try:
            # GNN + LSTM 推理
            dist, value, edge_logits = agent(fingerprint, topo_data)
            
            # 路径采样 (含 Dijkstra 兜底)
            # 注意：sample_path 需要 edge_index 来还原图结构
            path, log_prob, ai_success, path_complete = sample_path(
              edge_logits, topo_data.edge_index, s_node, d_node, 
              G_fallback=current_G, max_steps=30 )
            
            # C. 执行与奖励
            reward = -1.0 # 默认惩罚
            
            if ai_success and path_complete:
              # === 情况 1: AI 成功 ===
              # 只有 AI 真正成功时，我们才花费昂贵的物理时间去测量 QoS
              
              # 1. 下发规则
              install_path_rules(net, path, cookie=AGENT_COOKIE, do_ping=False)
              h_src = net.get(f'h{path[0]}')
              h_dst = net.get(f'h{path[-1]}')
              
              # 2. 物理测量 (Sim-to-Real)
              r_measure = measure_path_qos(
                h_src, h_dst, path, 
                flow_type=selected_flow_type
              )
              
              # 3. 赋值奖励
              # 如果 r_measure 返回 -1 (物理发包失败)，我们也认，视为环境噪声或严重拥塞
              reward = r_measure
              
              # 4. 清理规则
              clean_flow_rules(net, cookie=AGENT_COOKIE, mask=0xFFFF)
              
            else:
              # === 情况 2: AI 失败 (断路 或 依赖 Dijkstra) ===
              # [关键修正]
              # 即使 sample_path 用 Dijkstra 补全了路径 (path_complete=True)，
              # 我们也必须惩罚 Agent，因为 log_prob 对应的是 Agent 输出的"错误动作"。
              # 绝对不能测量 Dijkstra 的路径并把奖励给 Agent！
              
              reward = -1.0 
              vprint(f"[Penalty] AI failed to route independently.")
            
            vprint_path_status(current_G, path)
            vprint(f"[Result] Path: {path} | Reward: {reward:.2f}")
            # E. 记录 Loss
            r_tensor = torch.tensor(reward, dtype=torch.float32).to(device)
            advantage = r_tensor - value.detach()
            
            act_loss = -(log_prob * advantage) 
            cri_loss = CONFIG.CRITIC_LOSS_COEF * (r_tensor - value).pow(2)
            
            # 如果是分布对象，dist.entropy().mean()；如果是 logits，需自行处理
            # 这里假设 sample_path 处理了采样，dist 是 Distribution 对象
            entropy = dist.entropy().mean()
            
            loss = act_loss + cri_loss - (CONFIG.ENTROPY_COEF * entropy)
            (loss / CONFIG.BATCH_SIZE).backward()
            
            batch_rewards.append(reward)
            pbar.update(1)
            pbar.set_postfix({'R': f"{reward:.2f}"})
            
          except Exception as e:
            print(f"[Step Error] {e}")
            import traceback
            traceback()
            exit()

      # --- End of Batch ---
      torch.nn.utils.clip_grad_norm_(agent.parameters(), CONFIG.MAX_GRAD_NORM)
      optimizer.step()
      
      # 打印统计
      avg_r = np.mean(batch_rewards) if batch_rewards else -2.0
      print(f"Epoch {epoch+1} | Load: {curr_load:.0f}M | Avg Reward: {avg_r:.4f}")
      
      # 保存模型
      if (epoch + 1) % 10 == 0:
        save_path = os.path.join(CONFIG.CHECKPOINT_PATH, f"agent_sdn_ep{epoch+1}.pth")
        torch.save(agent.state_dict(), save_path)
        print(f"Saved model to {save_path}")

    # --- End of Training ---
    print("Training Finished.")
    save_path = os.path.join(CONFIG.CHECKPOINT_PATH, f"finalmodel.pth")
    torch.save(agent.state_dict(), save_path)
    print(f"Saved model to {save_path}")
    # 清理残局
    os.system("sudo killall -9 ITGSend ITGRecv > /dev/null 2>&1")

if __name__ == "__main__":
  os.sched_setaffinity(0, range(64, 128))

  # 2. 强制限制 PyTorch 内部的线程数
  # 即使绑定了核心，过多的线程也会导致开销。通常 8-16 线程足够快了。
  torch.set_num_threads(8) 
  torch.set_num_interop_threads(8)

  print(f"[System] PyTorch restricted to cores 64-127 with 8 threads.")
  try:
    train()
  except KeyboardInterrupt:
    print("\n[Stop] User interrupted.")
    os.system("sudo mn -c > /dev/null 2>&1")
  except Exception as e:
    print(f"\n[Crash] {e}")
    import traceback
    traceback.print_exc()
    os.system("sudo mn -c > /dev/null 2>&1")