import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import numpy as np
from tqdm import tqdm
from datetime import datetime

# === Import Custom Modules ===
import MS.Env.VerbosePrint as vp
from MS.Env.VerbosePrint import vprint
from MS.Agent.ActorCritic import ActorCritic
from MS.Env.FlowGenerator import FlowGenerator, FlowType
from MS.Env.NetworkGenerator import TopologyGenerator, get_pyg_data_from_nx
from MS.Env.MininetController import (
  get_a_mininet, 
  get_a_fingerprint, 
  measure_path_qos, 
  sample_path, 
  clean_flow_rules, 
  install_path_rules,
  NetworkMonitor
)

# ==============================================================================
# Configuration
# ==============================================================================
class Config:
  # --- Training Control ---
  EPOCH = 20              # Number of distinct Topologies
  EPISODES_PER_TOPO = 100 # Steps per Topology
  BATCH_SIZE = 16         # Gradient accumulation steps
  
  # --- Hyperparameters ---
  LR = 1e-4
  GAMMA = 0.99
  ENTROPY_COEF = 0.05     # Higher entropy to encourage exploration
  MAX_GRAD_NORM = 0.5
  CRITIC_LOSS_COEF = 0.5

  # --- System ---
  MODEL_DIR = "./trained_model"
  SAVE_PATH = os.path.join(MODEL_DIR, "a2c_final_rigorous.pth")
  PRETRAINED_LSTM = os.path.join(MODEL_DIR, "trained_lstm.pth")
  PRETRAINED_GNN = os.path.join(MODEL_DIR, "trained_gnn_recall_ospf.pth")
  
  GNN_DIM = 256
  LSTM_DIM = 128
  GNN_LAYERS = 6
  
  # --- Environment ---
  N_PACKETS = 30
  STEP_DURATION = 4.0     # 4.0s (Matches Video Segment & TCP Warmup)
  DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = Config()

# ==============================================================================
# Feasibility Check (The "God View" / MM1 Logic)
# ==============================================================================
def is_episode_solvable(G_nx, s_node, d_node, flow_type):
  """
  Checks if a feasible path exists in the graph G_nx for the given flow type.
  This uses a 'God View' (Dijkstra on filtered graph) to prevent training on 
  impossible scenarios.
  """
  # 1. Define Hard Constraints (Must match E-model Cliffs)
  constraints = {
    'voip':      {'max_delay': 150, 'min_bw': 0.1}, # VoIP is delay sensitive
    'gaming':    {'max_delay': 60,  'min_bw': 0.5}, # CSa is VERY delay sensitive
    'streaming': {'max_delay': 500, 'min_bw': 5.0}  # Video needs 5Mbps
  }
  
  # Get constraints for current flow (default to streaming if unknown)
  req = constraints.get(flow_type.name.lower(), constraints['streaming'])
  
  # 2. Edge Pruning (Filter out links with insufficient bandwidth)
  # We assume 'utilization' is low at start of episode, or we check capacity.
  def filter_edge(u, v, k):
    edge_data = G_nx[u][v]
    # Check physical bandwidth capacity
    capacity = edge_data.get('bandwidth', 10.0)
    if capacity < req['min_bw']:
      return False # Link is too narrow physically
    return True

  # Create a view of the graph with only valid edges
  valid_subgraph = nx.subgraph_view(G_nx, filter_edge=filter_edge)
  
  # 3. Shortest Path Calculation
  try:
    # Find path with minimum delay on the valid subgraph
    path = nx.dijkstra_path(valid_subgraph, s_node, d_node, weight='delay')
    
    # 4. Verify Total Delay
    total_delay = 0
    for i in range(len(path)-1):
      u, v = path[i], path[i+1]
      total_delay += G_nx[u][v].get('delay', 1.0)
      
    if total_delay <= req['max_delay']:
      return True # Solvable!
    else:
      return False # Best path is still too slow
      
  except nx.NetworkXNoPath:
    return False # Disconnected

# ==============================================================================
# Main Training Loop
# ==============================================================================
def run_a2c_training():
  # --- 1. Init Agent ---
  print(f"[ms] Initializing A2C Agent (Device: {CONFIG.DEVICE})...")
  start_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
  
  agent = ActorCritic(
    lstm_hidden_dim=CONFIG.LSTM_DIM,
    gnn_hidden_dim=CONFIG.GNN_DIM,
    gnn_layers=CONFIG.GNN_LAYERS,
    pretrained_lstm_path=CONFIG.PRETRAINED_LSTM,
    pretrained_gnn_path=CONFIG.PRETRAINED_GNN
  ).to(CONFIG.DEVICE)
  
  agent.train()
  optimizer = optim.Adam(filter(lambda p: p.requires_grad, agent.parameters()), lr=CONFIG.LR)
  
  # --- 2. Init Generators ---
  topo_gen = TopologyGenerator(CONFIG)
  flow_gen = FlowGenerator()
  
  stats_reward = []
  
  # --- 3. Topology Loop ---
  for topo_idx in range(CONFIG.EPOCH):
    
    # Generate Topology
    G_nx = topo_gen.generate_topology()
    
    # Enable Mininet Logging
    vp.MININET_VERBOSE = True
    vp.LOG_TO_CONSOLE = False
    vp.LOG_FILE_PATH = f"./train-log/a2c/{start_time}.log"
    
    try:
      with get_a_mininet(G_nx) as net:
        vprint(f"===========================================================")
        vprint(f"[Topo {topo_idx+1}/{CONFIG.EPOCH}] Mininet Started. Nodes: {len(G_nx.nodes())}")
        
        monitor = NetworkMonitor(net)
        hosts = {i: net.get(f'h{i}') for i in G_nx.nodes()}
        
        pbar = tqdm(range(CONFIG.EPISODES_PER_TOPO), desc=f"Topo {topo_idx+1}", leave=False)
        vp.CURRENT_PBAR = pbar
        
        # --- 4. Episode Loop ---
        for i_step in pbar:
          
          # -----------------------------------------------------------
          # A. Rejection Sampling (Feasibility Check)
          # -----------------------------------------------------------
          # Try to generate a solvable scenario. Give up after 20 tries.
          valid_scenario = False
          for retry in range(20):
            s_node, d_node = topo_gen.select_source_destination()
            flow_type, flow_profile = flow_gen.get_random_flow()
            
            # THE "GOD VIEW" CHECK
            if is_episode_solvable(G_nx, s_node, d_node, flow_type):
              valid_scenario = True
              break # Found a valid one!
            else:
              # Optional: Shuffle graph properties if stuck
              # topo_gen.randomize_links(G_nx) 
              pass
          
          if not valid_scenario:
            vprint("[Warning] Could not generate solvable scenario. Skipping step.")
            continue

          h_src, h_dst = hosts[s_node], hosts[d_node]
          clean_flow_rules(net) # Reset flow tables
          
          # -----------------------------------------------------------
          # B. State Observation (Fingerprint + Graph)
          # -----------------------------------------------------------
          # 1. Install temporary shortest path to allow probe packets
          TEMP_COOKIE = 0x8888
          try:
            temp_path = nx.shortest_path(G_nx, source=s_node, target=d_node)
            install_path_rules(net, temp_path, cookie=TEMP_COOKIE)
          except:
            continue

          # 2. Capture Traffic Fingerprint
          # This sends real packets to get the (Size, IAT) matrix
          fingerprint = get_a_fingerprint(
            server=h_dst, client=h_src,
            flow_type=flow_type,
            n_packets_to_capture=CONFIG.N_PACKETS
          ).float().to(CONFIG.DEVICE)
          
          # Clean temp rules so Agent can decide freely
          clean_flow_rules(net, TEMP_COOKIE)
          
          # 3. Update Graph State (Bandwidth utilization / Queue depth)
          monitor.sync_state_to_graph(G_nx)
          pyg_data, _ = get_pyg_data_from_nx(G_nx, s_node, d_node, CONFIG)
          pyg_data = pyg_data.to(CONFIG.DEVICE)
          
          # -----------------------------------------------------------
          # C. Agent Decision
          # -----------------------------------------------------------
          # Forward pass: LSTM -> FiLM -> GNN -> Logits
          dist, value_est, edge_logits = agent(fingerprint, pyg_data)
          
          # Sample path from edge probabilities
          path, log_prob_sum, success = sample_path(
            edge_logits, pyg_data.edge_index, s_node, d_node, max_steps=30
          )
          
          # -----------------------------------------------------------
          # D. Execution & Reward
          # -----------------------------------------------------------
          reward = -10.0 # Default penalty for failure
          
          if success:
            # Install the chosen path
            install_path_rules(net, path)
            
            # Measure Real QoS using D-ITG
            # Note: This function calls 'calculate_qoe_reward' inside MininetController
            # which in turn calls your new E-model.py
            reward = measure_path_qos(h_src, h_dst, path, flow_type)
          else:
            vprint(f"[Agent] Failed to construct a valid path.")

          # Normalize Reward for Stability (Mapping -100~10 to -1~1 approx)
          # Using tanh to squash cliffs
          reward_tensor = torch.tensor([reward], device=CONFIG.DEVICE)
          reward_norm = torch.tanh(reward_tensor / 5.0) 
          
          # -----------------------------------------------------------
          # E. Optimization (A2C)
          # -----------------------------------------------------------
          # Advantage = Actual Reward - Estimated Value
          advantage = reward_norm - value_est.detach()
          
          # Loss = Actor Loss + Critic Loss - Entropy
          actor_loss = -log_prob_sum * advantage
          critic_loss = nn.MSELoss()(value_est, reward_norm)
          entropy = dist.entropy().mean()
          
          total_loss = actor_loss + (CONFIG.CRITIC_LOSS_COEF * critic_loss) - (CONFIG.ENTROPY_COEF * entropy)
          
          # Accumulate gradients (BATCH_SIZE)
          (total_loss / CONFIG.BATCH_SIZE).backward()
          
          if (i_step + 1) % CONFIG.BATCH_SIZE == 0:
            torch.nn.utils.clip_grad_norm_(agent.parameters(), CONFIG.MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad()
            
          # -----------------------------------------------------------
          # F. Logging
          # -----------------------------------------------------------
          stats_reward.append(reward)
          avg_r = np.mean(stats_reward[-50:])
          
          pbar.set_postfix({
            "Type": f"{flow_type.name[:4]}",
            "R": f"{reward:.1f}", 
            "Avg": f"{avg_r:.1f}",
            "L": f"{total_loss.item():.2f}"
          })
          
          # Save Checkpoint
          if len(stats_reward) % 100 == 0:
            torch.save(agent.state_dict(), CONFIG.SAVE_PATH)

    except Exception as e:
      vp.CURRENT_PBAR = None
      if pbar: pbar.close()
      print(f"\n[Error] Topo {topo_idx} crashed: {e}")
      import traceback
      traceback.print_exc()
      continue # Try next topology

  print(f"[Done] Training finished. Model saved to {CONFIG.SAVE_PATH}")

if __name__ == '__main__':
  if os.getuid() != 0:
    print("❌ Error: Must run as root (sudo) for Mininet.")
  else:
    os.makedirs(CONFIG.MODEL_DIR, exist_ok=True)
    os.makedirs("./train-log", exist_ok=True)
    run_a2c_training()