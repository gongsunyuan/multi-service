import os
import sys
import time
import yaml
import argparse
import torch
import torch.nn as nn
import networkx as nx
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# === 1. 环境路径设置 ===
sys.path.append(os.getcwd())

# === 2. 项目内模块导入 ===
import src.utils.VerbosePrint as vp
from src.utils.VerbosePrint import vprint
from src.models.ActorCritic import ActorCritic
from src.utils.MyParaser import TopoParaser
from src.env.FlowGenerator import FlowGenerator, FlowType
from src.env.NetworkGenerator import get_pyg_data_from_nx, TopologyGenerator
from src.env.MininetController import (
  get_a_mininet, 
  measure_path_qos, 
  sample_path, 
  clean_flow_rules, 
  install_path_rules, 
  NetworkMonitor,
  vprint_network_status,
  vprint_path_status
)

# === 3. 全局常量 ===
AGENT_COOKIE = 0xA001  
BG_COOKIE    = 0xB000  
BG_MASK      = 0xF000

FLOW_GEN = FlowGenerator()
TOPO_GEN = TopologyGenerator()

# ==============================================================================
# Helper Functions
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
  
  base_dir = Path(__file__).resolve().parents[1]
  if str(base_dir).endswith('scripts'):
    base_dir = base_dir.parent

  topo_dir = os.path.join(base_dir, getattr(config, 'TOPOLOGIES_DIR'))
  setattr(config, 'TOPOLOGY_FILE', os.path.join(topo_dir, getattr(config, 'TOPOLOGY_FILE')))
  setattr(config, 'FINGERPRINT_BANK', os.path.join(base_dir, getattr(config, 'FINGERPRINT_BANK')))
  
  # log_dir
  log_dir = os.path.dirname(os.path.join(base_dir, getattr(config, 'LOG_DIR')))
  log_dir_name = f"agent_eval_{datetime.now().strftime("%Y%m%d%H%M%S")}"
  setattr(config, "LOG_FILE_DIR", os.path.join(log_dir, log_dir_name))
  p = Path(config.LOG_FILE_DIR)
  p.mkdir(parents=True, exist_ok=True)

  return config

def refresh_background_traffic(net, topo_graph, load_mbps=600.0):
  vprint(f"[BG] Refreshing Background Traffic (Target: {load_mbps} Mbps)...")
  clean_flow_rules(net, cookie=BG_COOKIE, mask=BG_MASK)
  os.system("sudo killall -9 ITGSend ITGRecv > /dev/null 2>&1")
  time.sleep(1.0)
  
  nodes = list(topo_graph.nodes())
  tm_dict = FLOW_GEN.generate_traffic_matrix(nodes, topo_graph, total_load_mbps=load_mbps)
  topo_graph = FLOW_GEN.simulate_tm_on_graph(topo_graph, tm_dict)
  
  FLOW_GEN.apply_traffic_matrix_to_mininet(
    net, tm_dict, topo_graph, install_rules_func=install_path_rules, duration=6000.0
  )
  vprint("[BG] Waiting 5s for congestion stabilization...")
  time.sleep(5.0)
  return topo_graph

def get_real_fingerprint_from_bank(flow_type, FINGERPRINT_BANK, CONFIG):
  key = flow_type.name.lower()
  samples = FINGERPRINT_BANK.get(key)
  if not samples: raise ValueError(f"Bank empty for {key}")
  fp = random.choice(samples) 
  return fp.unsqueeze(0).float().to(CONFIG.DEVICE)

# ==============================================================================
# 🧠 核心算法库 (Baseline Library)
# ==============================================================================
def get_baseline_paths(G, s, d):
  """
  一次性计算所有基线算法的路径
  返回字典: {'OSPF': path, 'WSP': path, 'SWP': path, 'RANDOM': path}
  """
  paths = {}
  
  # 0. 预计算所有简单路径 (限制跳数防止爆炸, NSFNet 直径小, cutoff=6 够了)
  # 注意：如果拓扑很大，不能用 all_simple_paths
  try:
    all_paths = list(nx.all_simple_paths(G, s, d, cutoff=8))
  except:
    all_paths = []
    
  if not all_paths:
    return {k: [] for k in ['OSPF', 'WSP', 'SWP', 'RANDOM']}

  # 辅助函数：计算路径指标 (Bottleneck BW, Total Delay)
  def get_metrics(path):
    bw_list = []
    delay_sum = 0
    for u, v in zip(path, path[1:]):
      data = G[u][v]
      # 估算剩余带宽: capacity * (1 - utilization)
      # 注意：验证时 G 里的 info 应该是 sync 过的
      limit = data.get('bandwidth', 100)
      util = data.get('utilization', 0.0)
      avail = limit * (1.0 - util)
      bw_list.append(avail)
      delay_sum += data.get('delay', 1.0)
    return min(bw_list), delay_sum # (Min BW, Total Delay)

  # 为所有路径打分
  candidates = []
  for p in all_paths:
    min_bw, sum_delay = get_metrics(p)
    candidates.append({'path': p, 'bw': min_bw, 'delay': sum_delay})

  # --- Algo 1: OSPF (Shortest Delay) ---
  # 逻辑: Delay 越小越好
  # 排序: (Delay ASC)
  candidates.sort(key=lambda x: x['delay'])
  paths['OSPF'] = candidates[0]['path']

  # --- Algo 2: WSP (Widest-Shortest) "最宽里选最快" ---
  # 逻辑: 瓶颈带宽越大越好 -> 延迟越小越好
  # 排序: (BW DESC, Delay ASC)
  candidates.sort(key=lambda x: (-x['bw'], x['delay']))
  paths['WSP'] = candidates[0]['path']

  # --- Algo 3: SWP (Shortest-Widest) "最快里选最宽" ---
  # 逻辑: 延迟越小越好 -> 瓶颈带宽越大越好 (其实就是 ECMP 增强版)
  # 排序: (Delay ASC, BW DESC)
  # 注意：这可能和 OSPF 是一样的，除非有 Equal Cost
  candidates.sort(key=lambda x: (x['delay'], -x['bw']))
  paths['SWP'] = candidates[0]['path']

  # --- Algo 4: Random (Lower Bound) ---
  paths['RANDOM'] = random.choice(all_paths)

  return paths

# ==============================================================================
# Core Validation Logic
# ==============================================================================
def validate_performance(CONFIG, agent, base_G_nx, fingerprint_bank):
  
  vprint("="*80)
  vprint(f"🚀 STARTING SUPER VALIDATION | Load: {CONFIG.VALIDATION_LOAD} Mbps")
  vprint(f"🥊 Contenders: Agent vs OSPF vs WSP vs SWP vs RANDOM")
  vprint("="*80)

  with get_a_mininet(base_G_nx) as net:
    monitor = NetworkMonitor(net)
    results = {}
    curr_load = CONFIG.VALIDATION_LOAD
    print(f"[Init] Setting up background traffic at {curr_load} Mbps...")
    # 定义我们要测的所有算法
    algos = ['Agent', 'OSPF', 'WSP', 'SWP', 'RANDOM']

    for flow_type_str in CONFIG.VALIDATION_FLOW_TYPES:
      flow_type_enum = getattr(FlowType, flow_type_str.upper())
      vprint(f">>> Testing Flow Type: {flow_type_str}")
      
      current_G = refresh_background_traffic(net, base_G_nx.copy(), load_mbps=curr_load)
      for _ in range(3): current_G = monitor.sync_state_to_graph(current_G.copy(), duration=0.5)
      vprint_network_status(current_G)
      # 初始化数据容器
      metrics = {algo: [] for algo in algos}
      
      total_episodes = CONFIG.VALIDATION_EPISODES_PER_FLOW
      total_batches = total_episodes // CONFIG.VALIDATION_BATCH_SIZE

      for batch_idx in range(total_batches):
        # 每个 Batch 之前，刷新网络状态
        refresh_background_traffic(net, base_G_nx.copy(), load_mbps=curr_load)
        for _ in range(3):
          current_G = monitor.sync_state_to_graph(current_G.copy(), duration=0.5)
        vprint_network_status(current_G)
        
        with tqdm(total=CONFIG.VALIDATION_BATCH_SIZE, desc=f"[{flow_type_str}] Batch {batch_idx+1}/{total_batches}", leave=False) as pbar:
          for _ in range(CONFIG.VALIDATION_BATCH_SIZE):
            current_G = monitor.sync_state_to_graph(current_G.copy(), duration=0.2)
            # 1. 找合法 S-D
            # valid = False
            for _ in range(10):
              s, d = TOPO_GEN.select_source_destination()
              if nx.has_path(current_G, s, d):
                valid = True; break
            # if not valid: continue
            try:
              # 2. 计算所有路径
              # Agent Path
              fingerprint = get_real_fingerprint_from_bank(flow_type_enum, fingerprint_bank, CONFIG)
              topo_data, _ = get_pyg_data_from_nx(current_G, s, d, config=CONFIG)
              topo_data = topo_data.to(CONFIG.DEVICE)

              with torch.no_grad():
                _, _, edge_logits = agent(fingerprint, topo_data)
                agent_path, _, ai_success, _ = sample_path(
                  edge_logits, topo_data.edge_index, s, d, 
                  G_fallback=current_G, max_steps=30, greedy=True
                )
              
              # Baseline Paths
              baseline_paths = get_baseline_paths(current_G, s, d)

              # 整合所有路径
              target_paths = {'Agent': agent_path if ai_success else []}
              target_paths.update(baseline_paths)

              # 3. 依次测量 (耗时操作)
              step_rewards = {}
              for algo_name, path in target_paths.items():
                r = -1.0
                if path and len(path) > 1:
                  # 安装规则 -> 测量 -> 清理
                  install_path_rules(net, path, cookie=AGENT_COOKIE, do_ping=False)
                  r = measure_path_qos(net.get(f'h{s}'), net.get(f'h{d}'), path, flow_type=flow_type_enum)
                  clean_flow_rules(net, cookie=AGENT_COOKIE, mask=0xFFFF)
                  vprint_path_status(current_G.copy(), path)
                  vprint(f"[Result] {algo_name} Path: {path} | Reward: {r:.2f}")

                metrics[algo_name].append(r)
                step_rewards[algo_name] = r

              # 更新进度条 (只显示 Agent 和 OSPF)
              pbar.set_postfix({'AI': f"{step_rewards['Agent']:.2f}", 'OSPF': f"{step_rewards['OSPF']:.2f}"})
              pbar.update(1)
            
            except Exception as e:
              print(e)
              import traceback
              traceback.print_exc()
              sys.exit(1)

      # 4. 统计本轮结果
      summary = {}
      for algo in algos:
        data = metrics[algo]
        avg_r = np.mean(data) if data else -1.0
        succ_rate = sum(1 for x in data if x > 0) / len(data) if data else 0.0
        summary[algo] = {'avg': avg_r, 'succ': succ_rate}
      
      results[flow_type_str] = summary
      
      # 打印中间简报
      print(f"{flow_type_str} Report")
      for algo in algos:
        print(f"  {algo:<8}: Avg={summary[algo]['avg']:.4f} | Succ={summary[algo]['succ']*100:.1f}%")

    # 5. 最终大表输出
    print("="*100)
    print(f"{'Type':<10} | {'Algo':<8} | {'Avg QoE':<10} | {'Success%':<10} | {'vs OSPF Imp%':<12}")
    print("-" * 100)
    
    for ft, data in results.items():
      ospf_base = data['OSPF']['avg']
      # 打印 Agent
      imp = ((data['Agent']['avg'] - ospf_base) / abs(ospf_base)) * 100 if ospf_base != 0 else 0
      print(f"{ft:<10} | {'Agent':<8} | {data['Agent']['avg']:<10.4f} | {data['Agent']['succ']*100:<9.1f}% | {imp:+.1f}%")
      
      # 打印其他基线
      for algo in algos:
        if algo == 'Agent': continue
        print(f"{'':<10} | {algo:<8} | {data[algo]['avg']:<10.4f} | {data[algo]['succ']*100:<9.1f}% | {'--'}")
      print("-" * 100)
    print("="*100 + "\n")

# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
  if os.geteuid() != 0:
    print("[Error] Must run as root (sudo)!")
    sys.exit(1)
  
  os.sched_setaffinity(0, range(64, 128))
  torch.set_num_threads(8)

  parser = TopoParaser()
  args = parser.parse_args()

  vp.LOG_TO_CONSOLE = args.verbose
  CONFIG = load_config(args.yaml)
  vp.LOG_TO_CONSOLE = args.verbose
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  setattr(CONFIG, 'DEVICE', device)
  vp.LOG_FILE_PATH = Path(CONFIG.LOG_FILE_DIR, 'Debug.log')

  # 如果有，使用参数设置评测流量负载
  if args.load_flow != 0:
    setattr(CONFIG, 'VALIDATION_LOAD', args.load_flow)

  print("[Debug] Log File Path:", vp.LOG_FILE_PATH)
  if os.path.exists(CONFIG.FINGERPRINT_BANK):
    print(f"[Init] Bank: {CONFIG.FINGERPRINT_BANK}")
    fingerprint_bank = torch.load(CONFIG.FINGERPRINT_BANK)
  else:
    sys.exit(1)

  print(f"[Init] Loading Agent: {args.checkpoint}")
  agent = ActorCritic(
    lstm_hidden_dim=CONFIG.LSTM_DIM,
    gnn_hidden_dim=CONFIG.GNN_DIM,
    gnn_node_dim=getattr(CONFIG, 'GNN_NODE_DIM', 10),
    gnn_layers=CONFIG.GNN_LAYERS,
    pretrained_lstm_path=None,
    pretrained_gnn_path=None
  ).to(device)
  
  agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
  agent.eval()

  base_G_nx = TOPO_GEN.load_topology(CONFIG.TOPOLOGY_FILE)

  try:
    validate_performance(CONFIG, agent, base_G_nx, fingerprint_bank)
  except KeyboardInterrupt:
    os.system("sudo mn -c > /dev/null 2>&1")
  except Exception as e:
    import traceback
    traceback.print_exc()
    os.system("sudo mn -c > /dev/null 2>&1")