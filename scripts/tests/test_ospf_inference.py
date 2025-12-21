import os
import sys
import torch
import networkx as nx
import random

# 确保能导入项目模块
sys.path.append(os.getcwd())

from src.utils.config_loadder import load_yaml_config
from src.env.network_generator import TopologyGenerator, get_pyg_data_from_nx
from src.agents.ppo_agent import FiLMPPOAgent
from src.env.sdn_controller import sample_path

def get_path_delay(G, path):
  """
  根据图中边的 'delay' 属性计算路径的总延迟。
  """
  total_delay = 0.0
  # 遍历路径中的每一对相邻节点 (u, v) [cite: 326]
  for u, v in zip(path[:-1], path[1:]):
    if G.has_edge(u, v):
      # 累加物理延迟 + 排队延迟 [cite: 270, 326]
      total_delay += G[u][v].get('delay', 0.0)
    else:
      # 如果路径不连续，返回无穷大 (代表断路)
      return float('inf')
  return total_delay

def run_test(agent, config):
  agent.eval()
  topo_gen = TopologyGenerator()

  # 2. 生成一张全新的随机拓扑 
  # 节点数设为 15，增加一点难度 
  G_test = topo_gen.generate_topology(mode='random', min_nodes=12, max_nodes=15)
  s, d = topo_gen.select_source_destination()
  # print(f"测验开始: 源节点 h{s} -> 目的节点 h{d}")

  # 3. 准备输入数据 
  # 预训练不涉及业务，使用全 1 的占位指纹
  dummy_fingerprint = torch.ones((1, config.model.hidden_dim, 2)) 
  data, _ = get_pyg_data_from_nx(G_test, s, d, config)

  data.target_idx = torch.tensor([d], dtype=torch.long) 
  data.curr_idx = torch.tensor([s], dtype=torch.long)

  # 4. 执行 AI 推理
  with torch.no_grad():
    # 获取全局 Embedding
    node_embeds = agent.get_node_embeddings(data, dummy_fingerprint)
    
    # 获取 Actor 对每一条边的打分 (Logits)  
    u_idx, v_idx = data.edge_index
    target_feats = node_embeds[data.target_idx].repeat(data.edge_index.size(1), 1)
    actor_input = torch.cat([
      node_embeds[u_idx], 
      target_feats, 
      node_embeds[v_idx], 
      data.edge_attr
    ], dim=-1)
    edge_logits = agent.actor(actor_input).squeeze(-1)

    # 5. 使用采样逻辑提取路径 
    ai_path, _, ai_success, _ = sample_path(
      edge_logits, data.edge_index, s, d, greedy=True
    )

  # 6. 获取标准 Dijkstra 答案作为对比
  expert_path = nx.dijkstra_path(G_test, s, d, weight='delay')

  # 7. 结果展示
  expert_delay = get_path_delay(G_test, expert_path)
  ai_delay = get_path_delay(G_test, ai_path)

  return ai_delay == expert_delay and ai_success

if __name__ == "__main__":
  # 请根据你实际生成的 checkpoint 文件夹修改路径
  """
  base: 20251219_174812           -- success: 95.06%  """
  """
  gnn_layer 3->6: 20251219_195558 -- success: 96.06%  """
  """
  gnn_layer 3->6 \
  hidden_dim 256, 20251219_215631 -- success: 96.63%  """

  """
  gnn_layer 3->6,
  hidden_dim 256, 
  lr 0.0003 -> 0.00003, epoch 200 -> 2000

  """

  checkpoint_path = "workspace/checkpoints/ospf_train/20251219_215631_ospf_train/final_model.pth"
  yaml_path = "workspace/checkpoints/ospf_train/20251219_215631_ospf_train/config.yaml"
  success = 0

  # 1. 加载配置与模型
  config = load_yaml_config(yaml_path)
  config.device = "cpu" # 推理用 CPU 即可

  agent = FiLMPPOAgent(config)
  if os.path.exists(checkpoint_path):
    agent.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    print(f" 成功加载权重: {checkpoint_path}")
  else:
    print(f" 找不到权重文件: {checkpoint_path}")

  for _ in range(10000): 
    if run_test(agent, config) : 
      success += 1
  succ = success/10000

  print(f"Success: {succ:.2%}")

