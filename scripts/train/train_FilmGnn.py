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
from src.utils.VerbosePrint import vprint, vprint_qos
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
  log_dir_name = f"sdn_train_{datetime.now().strftime('%Y%m%d%H%M%S')}"
  setattr(config, "LOG_FILE_DIR", os.path.join(log_dir, log_dir_name))
  p = Path(config.LOG_FILE_DIR)
  p.mkdir(parents=True, exist_ok=True)
  
  os.chmod(config.LOG_FILE_DIR, 0o777)
  return config

# ==============================================================================
# 2. 辅助函数
# ==============================================================================

def refresh_background_traffic(net, topo_graph, load_mbps=600.0):
  """
  刷新背景流量：清理旧流 -> 生成新矩阵 -> 注入 Mininet
  """
  vprint(f"Refreshing Background Traffic (Target: {load_mbps} Mbps)...", tag="TM Start")
  
  # 1. 清理旧的背景流
  clean_flow_rules(net, cookie=BG_COOKIE, mask=BG_MASK)
  os.system("sudo killall -9 ITGSend ITGRecv > /dev/null 2>&1")
  time.sleep(1.0)
  
  # 2. 生成新的流量矩阵 (Gravity Model)
  nodes = list(topo_graph.nodes())
  tm_dict = FLOW_GEN.generate_traffic_matrix(nodes, topo_graph, total_load_mbps=load_mbps)
  
  # 3. 在图上模拟流量矩阵，更新边的利用率和延迟
  topo_graph = FLOW_GEN.simulate_tm_on_graph(topo_graph, tm_dict)
  
  # 4. 注入物理网络 (Sim-to-Real 核心)
  FLOW_GEN.apply_traffic_matrix_to_mininet(
    net, 
    tm_dict, 
    topo_graph, 
    install_rules_func=install_path_rules, 
    duration=6000) # 只要比一个 Epoch 长即可
  
  
  # 5. 等待流量稳定 (TCP 慢启动需要时间)
  vprint("Waiting 5s for congestion stabilization...", tag="TM Start")
  time.sleep(5.0)

  return topo_graph

def is_episode_solvable(G, s_node, d_node, flow_type_str):
  """
  检查在当前背景流压力下，是否存在满足 QoS 要求的路径。
  """
  constraints = {
    'VOIP':      {'max_delay': 150, 'min_bw': 0.1},
    'GAMING':    {'max_delay': 60,  'min_bw': 0.5},
    'STREAMING': {'max_delay': 1000, 'min_bw': 5.0}
  }
  
  req = constraints.get(flow_type_str, constraints['VOIP'])
  
  def filter_edge(u, v):
    edge_data = G[u][v]
    cap = edge_data.get('bandwidth', 10.0)
    util = edge_data.get('utilization', 0.0)
    residual_bw = cap * (1.0 - util)
    
    if residual_bw < req['min_bw']:
      return False
    return True

  view = nx.subgraph_view(G, filter_edge=filter_edge)
  
  try:
    path = nx.shortest_path(view, source=s_node, target=d_node, weight='delay')
    total_delay = 0
    for i in range(len(path)-1):
      u, v = path[i], path[i+1]
      total_delay += G[u][v].get('delay', 0.0)
      
    if total_delay > req['max_delay']:
      return False 
      
    return True 
    
  except nx.NetworkXNoPath:
    return False 

def get_real_fingerprint_from_bank(flow_type, FINGERPRINT_BANK, CONFIG):
  key = flow_type.name.lower()
  samples = FINGERPRINT_BANK.get(key)
  
  if not samples:
    raise ValueError(f"Bank empty for {key}")
      
  fp = random.choice(samples) 
  noise = torch.randn_like(fp) * 0.02
  fp_aug = fp + noise
  return fp_aug.unsqueeze(0).float().to(CONFIG.DEVICE)

def generate_balanced_flow_list(num_per_type):
  """
  生成平衡的流类型列表，确保每个 epoch 训练等量的各类业务流
  """
  # 假设 FlowType 有 VOIP, GAMING, STREAMING 三种
  # 这里直接使用 FlowGenerator 定义的枚举
  target_types = [FlowType.VOIP, FlowType.GAMING, FlowType.STREAMING]
  
  flow_list = []
  for f_type in target_types:
    flow_list.extend([f_type] * num_per_type)
  
  random.shuffle(flow_list)
  return flow_list

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
  vp.LOG_TO_CONSOLE = args.verbose 
  vp.LOG_FILE_PATH = os.path.join(getattr(CONFIG, 'LOG_FILE_DIR', None), 'debug.log')
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  setattr(CONFIG, 'DEVICE', device)
  print(f"=== SDN Agent Training (Balanced Flow Edition) ===")
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
    gnn_node_dim=getattr(CONFIG, 'GNN_NODE_DIM', 10), 
    gnn_layers=CONFIG.GNN_LAYERS,
    pretrained_lstm_path=CONFIG.PRETRAINED_LSTM,
    pretrained_gnn_path=CONFIG.PRETRAINED_GNN).to(device)
  
  if args.checkpoint and os.path.exists(args.checkpoint):
    print(f"[Init] Loading Agent checkpoint from {args.checkpoint}...")
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))

  elif os.path.exists(CONFIG.PRETRAINED_MM1):
    print(f"[Init] Loading MM1 Pretrained Weights from {CONFIG.PRETRAINED_MM1}...")
    state_dict = torch.load(CONFIG.PRETRAINED_MM1, map_location=device)
    agent.load_state_dict(state_dict, strict=False) 
  else:
    print("[Init] MM1 weights not found, starting Actor/Critic from scratch.")
    
  param_groups = [
    {'params': agent.gnn_model.node_embed.parameters(), 'lr': CONFIG.LR_GNN_BODY, 'name': 'GNN_Embed'},
    {'params': agent.gnn_model.convs.parameters(), 'lr': CONFIG.LR_GNN_BODY, 'name': 'GNN_Convs'},
    {'params': agent.gnn_model.layer_norms.parameters(), 'lr': CONFIG.LR_GNN_BODY, 'name': 'GNN_Norms'},
    {'params': agent.film_generator.parameters(), 'lr': CONFIG.LR_HEADS, 'name': 'FiLM_Gen'},
    {'params': agent.critic_head.parameters(), 'lr': CONFIG.LR_HEADS, 'name': 'Critic_Head'},
    {'params': agent.gnn_model.edge_output_head.parameters(), 'lr': CONFIG.LR_HEADS, 'name': 'Actor_Head'}]

  optimizer = optim.Adam(param_groups)

  # --- 加载拓扑图 ---
  try:
    base_G_nx = TOPO_GEN.load_topology(CONFIG.TOPOLOGY_FILE)
  except:
    print("[Error]: nsfnet.graphml not found.")
    sys.exit(1)
  
  # --- 训练参数配置 ---
  # 每种流每轮训练的数量 (从配置读取，默认30)
  FLOWS_PER_TYPE = getattr(CONFIG, 'FLOWS_PER_TYPE_PER_EPOCH', 30)
  BATCH_SIZE = getattr(CONFIG, 'BATCH_SIZE', 15)

  # ==========================================
  # 启动 Mininet 环境
  # ==========================================
  with get_a_mininet(base_G_nx) as net:
    print("[System] Mininet Started. TSO/GSO Disabled.")
    monitor = NetworkMonitor(net)
    
    # --- Epoch Loop (外层：环境刷新周期) ---
    for epoch in range(CONFIG.EPOCH):
      vprint(f"Epoch {epoch+1}/{CONFIG.EPOCH}"+'='*30, tag="Epoch Start")
      # [Step 1] 刷新环境 (每个 Epoch 一次)
      progress = epoch / CONFIG.EPOCH
      # 1. 计算当前背景流负载
      base_load = CONFIG.START_LOAD + progress * (CONFIG.END_LOAD - CONFIG.START_LOAD)
      noise = np.random.uniform(-0.1, 0.1) * base_load

      # 2. 最终负载 (Clamping 防止变成负数或超出物理极限)
      curr_load = max(20.0, min(base_load + noise, 300.0))

      # 3. 计算当前熵系数
      curr_entropy_coef = CONFIG.ENTROPY_START - progress * (CONFIG.ENTROPY_START - CONFIG.ENTROPY_END)

      current_G = refresh_background_traffic(net, base_G_nx, load_mbps=curr_load)

      # 初始同步网络状态
      vprint("Initial Sync...", tag="Load Net")
      for _ in range(3):
        current_G = monitor.sync_state_to_graph(current_G, duration=0.5)

      vprint_network_status(current_G)
      # 生成本 Epoch 的任务列表 (保证每种流数量一致)
      epoch_flow_tasks = generate_balanced_flow_list(FLOWS_PER_TYPE)
      # 将任务切分成 Batch
      # epoch_flow_batches = [ [FlowType, ...], [FlowType, ...] ]
      epoch_flow_batches = [
        epoch_flow_tasks[i:i + BATCH_SIZE] 
        for i in range(0, len(epoch_flow_tasks), BATCH_SIZE)
      ]
      
      # 统计器 (Key: FlowType Name, Value: list of rewards)
      epoch_stats = { 'VOIP': [], 'GAMING': [], 'STREAMING': [] }

      # --- Update Loop (中层：参数更新周期) ---
      # 这里的 tqdm 进度条显示的是 Update 的进度 (例如 0/10)
      desc_str = f"Ep {epoch+1}/{CONFIG.EPOCH} [Load {curr_load:.0f}M]"
      
      with tqdm(total=len(epoch_flow_batches), desc=desc_str, leave=False) as pbar:

        for batch_flow_types in epoch_flow_batches:
          vprint(f"Batch Start: A new batch of flows", tag="Batch Start")
          
          optimizer.zero_grad()
          batch_rewards = []
          
          # 临时统计当前 Batch 的 Reward，用于打印
          batch_stats_print = { 'VOIP': [], 'GAMING': [], 'STREAMING': [] }
          
          # --- Flow Loop (内层：梯度累积) ---
          for flow_type_enum in batch_flow_types:
            vprint('='*30)
            vprint(f"Processing flow type: {flow_type_enum.name}", tag="Flow Start")
            # 每次决策前同步一次网络 (保证 Agent 看到的是实时状态)
            current_G = monitor.sync_state_to_graph(current_G, duration=0.5)
            vp.CURRENT_PBAR = pbar
            
            # --- 寻找合适流请求 (Solvable Check) ---
            max_try = 10
            valid_scenario = False
            selected_s, selected_d = None, None
            
            for _ in range(max_try):
              s, d = TOPO_GEN.select_source_destination()
              # 使用预定好的 flow_type_enum 进行检查
              if is_episode_solvable(current_G, s, d, flow_type_enum.name):
                selected_s, selected_d = s, d
                valid_scenario = True
                break
            
            if not valid_scenario:
              # 如果网络拥塞严重找不到可行路，跳过此流不计梯度
              # (为了保持 BatchSize 稳定，这里最好不要频繁跳过，但在高负载下难以避免)
              pbar.write(f"[Skip] No solvable path for {flow_type_enum.name}")
              continue

            # --- Agent 交互流程 ---
            try:
              # 1. 获取输入数据
              fingerprint = get_real_fingerprint_from_bank(flow_type_enum, fingerprint_bank, CONFIG)
              topo_data, _ = get_pyg_data_from_nx(current_G, selected_s, selected_d, config=CONFIG)
              topo_data = topo_data.to(device)

              # 2. 前向传播
              dist, value, edge_logits = agent(fingerprint, topo_data)

              # 3. 路径采样
              path, log_prob, ai_success, path_complete = sample_path(
                edge_logits, topo_data.edge_index, selected_s, selected_d, 
                G_fallback=current_G, max_steps=30
              )

              # 4. 执行与奖励计算
              reward = -2.0
              if ai_success and path_complete:
                vprint_path_status(current_G, path)
                install_path_rules(net, path, cookie=AGENT_COOKIE, do_ping=False)
                h_src, h_dst = net.get(f'h{path[0]}'), net.get(f'h{path[-1]}')
                qos_reward, qoe_reward = measure_path_qos(h_src, h_dst, path, flow_type=flow_type_enum, config=CONFIG)
                reward = qos_reward
                vprint(f"Get {flow_type_enum.name} QoS reward {qos_reward}, QoE reward {qoe_reward}.", tag="QoS Reward")
                clean_flow_rules(net, cookie=AGENT_COOKIE, mask=0xFFFF)
              else:
                reward = -2.0 # AI 寻路失败惩罚
                vprint(f"AI failed to find path for {flow_type_enum.name} from {selected_s} to {selected_d}", tag="Agent Fail")

              # 5. 计算 Loss 并反向传播 (梯度累积)
              r_tensor = torch.tensor(reward, dtype=torch.float32).to(device)
              advantage = r_tensor - value.detach()
              
              act_loss = -(log_prob * advantage)

              cri_loss = CONFIG.CRITIC_LOSS_COEF * (r_tensor - value).pow(2)
              entropy = dist.entropy().mean()
              
              loss = act_loss + cri_loss - (curr_entropy_coef * entropy)
              
              # Divide by BATCH_SIZE for accumulation
              (loss / BATCH_SIZE).backward()
              
              # 6. 记录数据
              batch_rewards.append(reward)
              epoch_stats[flow_type_enum.name].append(reward)
              batch_stats_print[flow_type_enum.name].append(reward)
              # vprint('='*30)

            except Exception as e:
              print(f"[Step Error] {e}")
              continue
          
          # --- End of Flow Loop (Batch Complete) ---
          
          # 梯度更新
          torch.nn.utils.clip_grad_norm_(agent.parameters(), CONFIG.MAX_GRAD_NORM)
          optimizer.step()
          
          # 更新进度条信息 (打印各流平均分)
          def get_avg(lst): return np.mean(lst) if lst else -1.0
          pbar.set_postfix({
            'V': f"{get_avg(batch_stats_print['VOIP']):.2f}",
            'G': f"{get_avg(batch_stats_print['GAMING']):.2f}",
            'S': f"{get_avg(batch_stats_print['STREAMING']):.2f}"
          })
          pbar.update(1)

      # --- End of Update Loop (Epoch Complete) ---
      
      # 打印本 Epoch 的最终统计
      def get_epoch_avg(key): 
        return np.mean(epoch_stats[key]) if epoch_stats[key] else -2.0
      
      avg_v = get_epoch_avg('VOIP')
      avg_g = get_epoch_avg('GAMING')
      avg_s = get_epoch_avg('STREAMING')

      all_rewards = epoch_stats['VOIP'] + epoch_stats['GAMING'] + epoch_stats['STREAMING']
      avg_total = np.mean(all_rewards) if all_rewards else -2.0

      print(f"Epoch {epoch+1} Finished | Load: {curr_load:.0f}M")
      print(f"  >> Total Avg: {avg_total:.3f}")
      print(f"  >> Avg Rewards: VOIP={avg_v:.3f} | GAMING={avg_g:.3f} | STREAMING={avg_s:.3f}")
      
      # 保存模型
      if (epoch + 1) % 20 == 0:
        save_path = os.path.join(CONFIG.CHECKPOINT_PATH, f"agent_sdn_ep{epoch+1}.pth")
        torch.save(agent.state_dict(), save_path)
        print(f"Saved model to {save_path}")

    # --- End of Training ---
    print("Training Finished.")
    save_path = os.path.join(CONFIG.CHECKPOINT_PATH, f"finalmodel.pth")
    torch.save(agent.state_dict(), save_path)
    print(f"Saved model to {save_path}")
    os.system("sudo killall -9 ITGSend ITGRecv > /dev/null 2>&1")

if __name__ == "__main__":

  os.sched_setaffinity(0, range(64, 128))
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