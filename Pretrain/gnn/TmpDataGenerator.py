import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
import networkx as nx
import math
import numpy as np
import random
from torch.utils.data import IterableDataset
from MS.Env.NetworkGenerator import TopologyGenerator, get_pyg_data_from_nx

# === 专家标签生成 ===
def generate_expert_label(G: nx.Graph, S_node: int, D_node: int, edge_index: torch.Tensor):
  try:
    path_nodes = nx.dijkstra_path(G, S_node, D_node, weight='delay')
  except nx.NetworkXNoPath:
    return None

  path_edges = set()
  for i in range(len(path_nodes) - 1):
    u, v = path_nodes[i], path_nodes[i+1]
    path_edges.add((u, v))
    path_edges.add((v, u)) 

  num_total_edges = edge_index.shape[1]
  labels = torch.zeros(num_total_edges, dtype=torch.float)
  for i in range(num_total_edges):
    u, v = edge_index[0, i].item(), edge_index[1, i].item()
    if (u, v) in path_edges:
      labels[i] = 1.0
  return labels

# === 动态数据集 ===
class DynamicGraphDataset(IterableDataset):
  def __init__(self, topo_gen, global_stats, max_samples):
    self.topo_gen = topo_gen
    self.stats = DEFAULT_CONFIG
    self.max_samples = max_samples

  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
      worker_seed = torch.initial_seed() % (2**32 - 1) + worker_info.id
      random.seed(worker_seed)
      np.random.seed(worker_seed)
    
    if worker_info is None:
      iter_end = self.max_samples
    else:
      per_worker = int(math.ceil(self.max_samples / float(worker_info.num_workers)))
      iter_end = min(worker_info.id * per_worker + per_worker, self.config)
    for _ in range(iter_end):
      yield generate_single_sample(self.topo_gen, self.stats)

  def generate_single_sample(topo_gen, config):
    while True:
      G_nx = topo_gen.generate_topology()
      try:
        S, D = topo_gen.select_source_destination()
      except: continue
      data, G_with_attrs = get_pyg_data_from_nx(G_nx, S, D, config)
      y = generate_expert_label(G_with_attrs, S, D, data.edge_index)
      if y is not None:
        data.y = y
        return data
