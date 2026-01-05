import pickle
import networkx as nx
import random
import torch
import os
import math
from torch_geometric.data import Data
from ..utils import logger

class TopologyGenerator:
  def __init__(self) -> None:
    self.G = None
    pass

  # [Insert into src/env/network_generator.py inside TopologyGenerator class]

  def generate_topology(self, mode='random', min_nodes=10, max_nodes=20) -> nx.Graph:
    """
    生成拓扑的统一入口。
    
    Args:
      mode (str): 'random' (随机生成) 或 'fixed' (返回固定的 NSFNet)
      min_nodes (int): 随机生成的最小节点数
      max_nodes (int): 随机生成的最大节点数
    """
    if mode == 'fixed':
      # 如果还没有加载过固定图，就加载一次
      if not hasattr(self, 'G') or self.G is None:
        # 假设 config 里有 graph_path，或者硬编码默认路径
        default_path = "nsfnet.graphml" 
        self.load_topology(default_path)
      
      # 返回固定图的深拷贝，防止数据污染
      assert self.G is not None, "Fixed graph is not loaded"
      G = self.G.copy()
      # 刷新一下动态状态 (拥塞程度)
      return self.refresh_dynamic_state(G, difficulty=0.5)

    else:
      # === 随机生成模式 (推荐预训练使用) ===
      # 随机决定节点数量
      n = random.randint(min_nodes, max_nodes)
      
      # 生成骨架
      G = self._create_random_graph(n)
      
      # 计算中心性指标 (Betweenness, PageRank...)
      G = self.update_graph_metric(G)
      
      # 初始化动态状态 (赋予随机的拥塞)
      G = self.refresh_dynamic_state(G, difficulty=random.random())

      self.G = G
      
      return G

  def _create_random_graph(self, n_nodes) -> nx.Graph:
    """
    [内部方法] 使用 Barabási-Albert 模型生成无标度网络。
    并初始化物理链路属性 (Bandwidth, Delay)。
    """
    # 1. 生成连通图骨架
    # m=2 表示每个新加入的节点会连接 2 个旧节点 (保证稀疏性但连通)
    while True:
      G = nx.barabasi_albert_graph(n_nodes, m=2)
      if nx.is_connected(G):
        break

    # 2. 初始化物理属性
    for u, v in G.edges():
      cap = random.choice([20.0, 40.0, 50.0, 100.0])
      
      # 1ms ~ 10ms 之间
      delay = random.uniform(1.0, 10.0)
      
      # 写入属性 (注意: 必须与 refresh_dynamic_state 兼容)
      G[u][v]['loss'] = 0.0         # 初始无丢包
      G[u][v]['delay'] = delay      # 基础传播延迟 (refresh 时会加上排队延迟)
      G[u][v]['capacity'] = cap     # 物理容量 (不变)
      G[u][v]['bandwidth'] = cap    # 初始可用带宽 = 容量
      G[u][v]['utilization'] = 0.0  # 初始空闲
      G[u][v]['base_delay'] = delay # 备份基础延迟
    
    # 3. 初始化节点属性
    for n in G.nodes():
      G.nodes[n]['buffer_occupancy'] = 0.0
      G.nodes[n]['proc_delay'] = 0.0

    return G
  
  def load_topology(self, filename: str) -> nx.Graph:
    """
    加载固定的 GraphML 文件 (NSFNet)，并初始化 RL 环境所需的动态属性。
    自动搜索路径：当前目录 -> TopoGraph/ 目录
    """

    #  传完整路径 "TopoGraph/nsfnet.graphml"
    if os.path.exists(filename):
      loadpath = filename
    else:
      raise FileNotFoundError(f"[Error] Load Topology: file not found: {filename} (Checked root and TopoGraph/)")

    # --- 2. 加载文件 (使用找到的 loadpath) ---
    if loadpath.endswith('.graphml'):
      # node_type=int: 尝试将节点 ID 转为整数 (NSFNet graphml 中 id="0")
      try:
        G = nx.read_graphml(loadpath, node_type=int)
      except Exception as e:
        logger.log(f"Load Topology: {e}, trying default loader...", tag="Graph Err")
        G = nx.read_graphml(loadpath) # Fallback

      # [关键] 强制转换节点 Label 为连续整数 (0, 1, 2...)
      # 确保兼容 Mininet 的 h0, h1 命名规则
      G = nx.convert_node_labels_to_integers(G, ordering='sorted')
      
    elif loadpath.endswith('.pkl'):
      with open(loadpath, 'rb') as f:
        G = pickle.load(f)
    else:
      raise ValueError(f"Unsupported format: {loadpath}")

    # --- 3. 初始化动态属性 ---
    self.scale_topology_bandwidth(G, scale=0.1)  # 默认不缩放
    self.G = self.update_graph_metric(G)
    logger.log(f"Loaded Fixed Topo: {loadpath} | Nodes: {len(G.nodes())}", tag="Graph Init")

    return G
  
  def update_graph_metric(self, G):
    """ 
    手动计算图的中心性指标，并更新到 NetworkX 节点属性中。 
    这样 vprint 函数才能读到非零值。 
    """ 
    # 1. 计算介数中心性 (Betweenness Centrality) 
    # weight='delay' 表示计算基于延迟的最短路介数，比默认的跳数更准 
    bet_dict = nx.betweenness_centrality(G, weight='delay', normalized=True)
    
    # 2. 计算 PageRank
    try:
      pr_dict = nx.pagerank(G, weight='bandwidth')
    except:
      # 某些图如果不连通可能会报错，给个兜底
      pr_dict = {n: 0.0 for n in G.nodes()}

    # 3. 计算聚类系数
    clust_dict = nx.clustering(G)
    
    assert isinstance(clust_dict, dict), "clust_dict is not a dict"

    # 4. 赋值回图节点
    for n in G.nodes():
      G.nodes[n]['pagerank'] = pr_dict.get(n, 0.0)
      G.nodes[n]['betweenness'] = bet_dict.get(n, 0.0)
      G.nodes[n]['clustering'] = clust_dict.get(n, 0.0)
    
    return G
  
  def scale_topology_bandwidth(self, G: nx.Graph, scale: float):
    """
    等比例缩放图中所有边的带宽。

    Args:
      G (nx.Graph): NetworkX 图对象 (会被原地修改)
      scale (float): 缩放比例 (例如 1.5 代表增加 50%，0.8 代表减少 20%)

    Returns:
      nx.Graph: 修改后的图对象
    """
    # 遍历图中所有的边
    # data=True 表示我们会获取边的属性字典
    for u, v, data in G.edges(data=True):

      # 检查这条边是否有带宽属性
      # 获取旧带宽
      old_bw = data['bandwidth']
      # 计算新带宽
      new_bw = old_bw * scale
      # 更新属性
      data['capacity'] = new_bw
      data['bandwidth'] = new_bw

    # vprint(f"[Graph] 已将拓扑带宽缩放 {scale} 倍 (Key: {bw_key})")
    return G

  def refresh_dynamic_state(self, G, difficulty=1.0):
    """
    [核心逻辑] 模拟网络动态变化
    1. 随机生成背景流量 (Utilization)
    2. 基于 M/M/1 模型推导 Delay 和 Loss
    """
    for u, v in G.edges():

      for u, v in G.edges():
        if difficulty < 0.3:
          # [简单模式] 几乎无拥塞，Agent 只需要学会连通性
          utilization = random.uniform(0.0, 0.1)
        elif difficulty < 0.7:
          # [中等模式] 偶尔有拥塞
          utilization = random.uniform(0.2, 0.5)
        else:
          # [困难模式] 真实的 Beta 分布，包含严重拥塞
          utilization = random.betavariate(2, 5)
            
      G[u][v]['utilization'] = utilization
      # 获取物理属性
      capacity = float(G[u][v].get('capacity', 100.0)) # Mbps
      base_delay = float(G[u][v].get('delay', 5.0))    # ms (物理传播延迟)
      
      # 2. M/M/1 推导排队延迟 (Queueing Delay)
      # Delay_total = Delay_prop + Delay_queue
      # Delay_queue = (1 / (µ - λ)) - (1/µ)  => 简化为  Base / (1 - rho)
      # 为了防止除以0，rho 上限设为 0.99
      rho = min(utilization, 0.99)
      
      # 估算排队因子 (Scaling Factor)
      # 假设当利用率 0% 时，延迟 = base_delay
      # 当利用率 90% 时，延迟会显著增加
      # 这里使用一个简化的排队公式:
      queue_delay = (10.0 * rho) / (1.0 - rho) # 10.0 是排队系数
      
      total_delay = base_delay + queue_delay
      
      # 3. M/M/1/K 推导丢包率 (Packet Loss)
      # 假设队列长度有限 (K)，根据利用率估算丢包概率
      # 当利用率低时，丢包几乎为0；利用率接近1时，丢包指数上升
      if rho < 0.8:
        loss = 0.0
      else:
        loss = 1.0 * math.pow((rho - 0.8) / 0.2, 2)
          
      # 更新图属性 (供 Agent 观测)
      G[u][v]['delay'] = total_delay
      G[u][v]['loss' ] = loss
      
      # 更新带宽 (剩余带宽)
      # available_bw = capacity * (1 - rho)
      G[u][v]['bandwidth'] = capacity * (1.0 - rho)

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
