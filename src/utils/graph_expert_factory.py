import torch
import math
import numpy as np
import random
import networkx as nx
from torch.utils.data import IterableDataset
from torch_geometric.data import Data
from ..env.network_generator import get_pyg_data_from_nx
from ..utils import logger

class ExpertPathFactory:
  """
  图数据专家工厂：
  利用 Dijkstra 算法生成最优路径，并为 Actor-Critic 提供监督学习标签。
  """
  def __init__(self, topo_gen, config):
    self.topo_gen = topo_gen
    self.config = config

  def compute_expert_data(self, G, s_node, d_node, edge_index):
    """
    计算专家标签：
    1. Actor 标签 (y): 最短路径上的边设为 1，其余为 0。
    2. Critic 标签 (target_value): 起点到终点的最短路径总延迟。
    """
    try:
      # 使用 single_source_dijkstra 一次性获取长度和路径
      total_delay, path_nodes = nx.single_source_dijkstra(
        G, source=s_node, target=d_node, weight='delay' )
      
    except nx.NetworkXNoPath:
      return None, None

    # --- A. 生成 Actor 标签 (边缘掩码) ---
    path_edges = set()
    for i in range(len(path_nodes) - 1):
      u, v = path_nodes[i], path_nodes[i+1]
      path_edges.add((u, v))
      path_edges.add((v, u)) # 无向图双向标记

    num_edges = edge_index.shape[1]
    actor_labels = torch.zeros(num_edges, dtype=torch.float)
    for i in range(num_edges):
      u, v = edge_index[0, i].item(), edge_index[1, i].item()
      if (u, v) in path_edges:
        actor_labels[i] = 1.0

    # --- B. 生成 Critic 标签 (路径总代价) ---
    # 归一化处理：通常将延迟除以一个预设的最大值，利于模型收敛
    max_expected_delay = self.config.env.max_delay
    critic_target = torch.tensor([total_delay / max_expected_delay], dtype=torch.float)

    return actor_labels, critic_target

  def create_single_sample(self):
    """生成一个完整的带标签样本"""
    while True:
      # 1. 生成拓扑结构 
      G_nx = self.topo_gen.generate_topology(min_nodes=10, max_nodes=20)
      
      try:
        # 2. 随机选择源和宿节点 
        s, d = self.topo_gen.select_source_destination()
        
        # 3. 提取图特征 (调用环境中的统一转换函数) 
        data, G_with_attrs = get_pyg_data_from_nx(G_nx, s, d, self.config)
        
        # 4. 计算专家真值 (Ground Truth)
        y_actor, y_critic = self.compute_expert_data(
          G_with_attrs, s, d, data.edge_index )
        
        if y_actor is not None:
          data.y = y_actor             # Actor 监督目标
          data.target_value = y_critic # Critic 监督目标
          # 存储当前节点和目标节点，方便计算
          data.curr_idx = torch.tensor([s], dtype=torch.long)
          data.target_idx = torch.tensor([d], dtype=torch.long)
          return data
                
      except Exception as e:
        logger.log(f"Generate Topo Error: {e}", tag="Topo Err")
        import traceback
        traceback.print_exc()
        exit()

class SupervisedGraphDataset(IterableDataset):
  """
  支持多进程的高效样本生成器数据集
  """
  def __init__(self, topo_gen, config, max_samples=5000):
    self.factory = ExpertPathFactory(topo_gen, config)
    self.config = config
    self.max_samples = max_samples

  def __iter__(self): 
    worker_info = torch.utils.data.get_worker_info()
    
    if worker_info is None:
      # 单进程模式 
      count = self.max_samples
    else:
      # 多进程模式 
      worker_seed = torch.initial_seed() % (2**32 - 1)
      random.seed(worker_seed)
      np.random.seed(worker_seed)
      
      per_worker = int(math.ceil(self.max_samples / float(worker_info.num_workers)))
      count = min(per_worker, self.max_samples)

    for _ in range(count):
      yield self.factory.create_single_sample()