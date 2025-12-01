import networkx as nx
import random
import torch
import os
import math
from torch_geometric.data import Data
from MS.Env.VerbosePrint import vprint

class Default_config:
  # 默认拓扑生成参数
  M_BA = 2
  MIN_BW = 20.0
  MAX_BW = 200.0
  MIN_LOSS = 0.0
  MAX_LOSS = 3.0
  MIN_DELAY = 1.0
  MAX_DELAY = 200.0
  MIN_NODES_NUM = 50
  MAX_NODES_NUM = 100


DEFAULT_CONFIG = Default_config()

class TopologyGenerator:
  def __init__(self, config=DEFAULT_CONFIG):
    pass

  def load_topology(self, filename: str) -> nx.Graph:
    """
    加载固定的 GraphML 文件 (NSFNet)，并初始化 RL 环境所需的动态属性。
    自动搜索路径：当前目录 -> TopoGraph/ 目录
    """
    # --- 1. 智能路径搜索 ---
    # 情况 A: 用户传了完整路径 "TopoGraph/nsfnet.graphml"
    if os.path.exists(filename):
      loadpath = filename
    # 情况 B: 用户只传了文件名 "nsfnet.graphml"，自动补全前缀
    elif os.path.exists(f"TopoGraph/{filename}"):
      loadpath = f"TopoGraph/{filename}"
    else:
      raise FileNotFoundError(f"[Error] Load Topology: file not found: {filename} (Checked root and TopoGraph/)")

    # --- 2. 加载文件 (使用找到的 loadpath) ---
    if loadpath.endswith('.graphml'):
      # node_type=int: 尝试将节点 ID 转为整数 (NSFNet graphml 中 id="0")
      try:
        G = nx.read_graphml(loadpath, node_type=int)
      except Exception as e:
        vprint(f"[Error] Load Topology: {e}, trying default loader...")
        G = nx.read_graphml(loadpath) # Fallback

      # [关键] 强制转换节点 Label 为连续整数 (0, 1, 2...)
      # 确保兼容 Mininet 的 h0, h1 命名规则
      G = nx.convert_node_labels_to_integers(G, ordering='sorted')
      
    elif loadpath.endswith('.pkl'):
      with open(loadpath, 'rb') as f:
        G = pickle.load(f)
    else:
      raise ValueError(f"Unsupported format: {loadpath}")
    
    # G = self.scale_topology_bandwidth(G, 0.5)
    self.G = G
    vprint(f"[Graph] Loaded Fixed Topo: {loadpath} | Nodes: {len(G.nodes())}")
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

def get_pyg_data_from_nx(G: nx.Graph, S_node: int, D_node: int, config):
  # --- 1. 性能优化：结构特征缓存 ---
  try:
    if 'betweenness' not in G.graph:
      G.graph['betweenness'] = nx.betweenness_centrality(G)
    if 'pagerank' not in G.graph:
      G.graph['pagerank'] = nx.pagerank(G, alpha=0.85)
    if 'clustering' not in G.graph:
      G.graph['clustering'] = nx.clustering(G)
      
    betweenness = G.graph['betweenness']
    pagerank = G.graph['pagerank']
    clustering = G.graph['clustering']
  except:
    betweenness = nx.betweenness_centrality(G)
    pagerank = nx.pagerank(G, alpha=0.85)
    clustering = nx.clustering(G)

  # --- 2. 边特征处理 ---
  source_nodes, target_nodes = [], []
  edge_attrs_raw = []
  
  for u, v, data in G.edges(data=True):
    source_nodes.extend([u, v])
    target_nodes.extend([v, u])
    
    d = float(data.get('delay', 1.0))
    l = float(data.get('loss', 0.0))
    b = float(data.get('bandwidth', 100.0))
    u_load = float(data.get('utilization', 0.0))
    avail_bw = b * (1.0 - u_load)

    attr = [d, b, l, u_load, avail_bw]
    edge_attrs_raw.extend([attr, attr])

  edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long) 
  edge_attr_tensor = torch.tensor(edge_attrs_raw, dtype=torch.float)

  # 归一化
  edge_attr = torch.zeros_like(edge_attr_tensor)
  # 假设 config 中有这些 MAX/MIN 常量，或者直接写死数值
  # 这里为了稳健，加上 float() 强转
  edge_attr[:, 0] = (edge_attr_tensor[:, 0] - config.MIN_DELAY) / (config.MAX_DELAY - config.MIN_DELAY + 1e-6)
  edge_attr[:, 1] = (edge_attr_tensor[:, 1] - config.MIN_BW) / (config.MAX_BW - config.MIN_BW + 1e-6)
  edge_attr[:, 2] = edge_attr_tensor[:, 2] / 5.0 # Loss (假设最大5%)
  edge_attr[:, 3] = edge_attr_tensor[:, 3] # Util (0-1)
  edge_attr[:, 4] = (edge_attr_tensor[:, 4] - config.MIN_BW) / (config.MAX_BW - config.MIN_BW + 1e-6)
  edge_attr = edge_attr.clamp(0.0, 1.0)

  # --- 3. 节点特征处理 ---
  try:
    dist_from_s = nx.single_source_shortest_path_length(G, S_node)
  except:
    dist_from_s = {n: 999 for n in G.nodes()}
    
  try:
    dist_to_d = nx.single_source_shortest_path_length(G, D_node)
  except:
    dist_to_d = {n: 999 for n in G.nodes()}

  num_nodes = G.number_of_nodes()
  node_features_list = [] # [FIX] 必须初始化这个列表
  deg_max = config.MAX_NODES_NUM 

  for i in range(num_nodes):
    node_data = G.nodes[i] # [OK] 正确获取节点属性
    
    deg = G.degree(i) / (deg_max + 1e-6)
    is_s = 1.0 if i == S_node else 0.0
    is_d = 1.0 if i == D_node else 0.0
    
    ds = 1.0 * min(dist_from_s.get(i, 999), deg_max) / deg_max 
    dd = 1.0 * min(dist_to_d.get(i, 999), deg_max) / deg_max
    
    betw = betweenness.get(i, 0.0)
    clus = clustering.get(i, 0.0)
    pr = pagerank.get(i, 0.0) * 10.0
    
    bo = float(node_data.get('buffer_occupancy', 0.0))
    pd = float(node_data.get('proc_delay', 0.0))

    # 基础特征 (10维)
    basic_feat = [deg, is_s, is_d, ds, dd, betw, clus, pr, bo, pd]
    
    # [FIX] 必须把特征加到列表里！
    node_features_list.append(basic_feat)
    
  # [FIX] 在循环外将列表转为 Tensor
  x = torch.tensor(node_features_list, dtype=torch.float)
  
  return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), G

