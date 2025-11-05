import networkx as nx
import random
import torch

class TopologyGenerator:
  def __init__(self, num_nodes_range=(10, 20), m_ba=2):
    """
    初始化拓扑生成器。
    :param num_nodes_range: 节点数量的范围 (e.g., 10-20个节点)
    :param m_ba: BA模型的附着参数 (每个新节点连接到 m_ba 个现有节点)
    """
    self.num_nodes_range = num_nodes_range
    self.m_ba = m_ba
    self.min_bw = 100.0  # Mbps
    self.max_bw = 1000.0 # Mbps
    self.min_delay = 1.0 # ms
    self.max_delay = 10.0 # ms
    self.G = None

  def generate_topology(self) -> nx.Graph:
    """生成一个随机的BA拓扑，并赋予链路属性。"""
    num_nodes = random.randint(*self.num_nodes_range)
    
    # 1. 生成 BA 无标度网络
    G = nx.barabasi_albert_graph(n=num_nodes, m=self.m_ba)
    
    # 2. 为节点和边添加属性
    for u, v in G.edges():
      # 随机但合理的链路属性
      bw = random.uniform(self.min_bw, self.max_bw)
      delay = random.uniform(self.min_delay, self.max_delay)
      
      # 存储属性
      G[u][v]['bandwidth'] = bw
      G[u][v]['delay'] = delay
      # 存储容量（例如，基于带宽或QCI等级）
      G[u][v]['capacity'] = bw * 0.8 # 假设可用容量是带宽的80%

    self.G = G
    return G

  def select_source_destination(self) -> tuple[int, int]:
    """随机选择不重复的源和目的节点。"""
    if self.G is None:
      raise ValueError("Topology must be generated first.")
        
    nodes = list(self.G.nodes())
    
    # 确保源和目的一定是连通的
    while True:
      s, d = random.sample(nodes, 2)
      if nx.has_path(self.G, s, d):
        return s, d
