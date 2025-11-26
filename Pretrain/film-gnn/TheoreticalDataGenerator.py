import torch
import networkx as nx
import numpy as np
import random
import math
from torch.utils.data import IterableDataset
from MS.Env.NetworkGenerator import TopologyGenerator, DEFAULT_CONFIG

# ==========================================
# 1. M/M/1 排队延迟模型 (数学核心)
# ==========================================
def calculate_theoretical_delay(prop_delay, capacity, utilization):
  """
  基于 M/M/1 启发式公式计算总延迟。
  Total Delay = Propagation + Queueing
  Queueing ~ C * (rho / (1 - rho))
  """
  # 1. 传播延迟 (固定物理属性)
  d_prop = prop_delay
  
  # 2. 排队延迟 (动态拥塞属性)
  # 限制 rho 上限为 0.99，防止除以零
  rho = min(utilization, 0.99)
  
  # 缩放因子 C (根据之前的讨论，设为 10ms)
  # 这意味着当负载 50% 时，排队延迟是 10ms
  C = 10.0 
  
  d_queue = C * (rho / (1.0 - rho + 1e-6))
  
  # 截断最大延迟，防止梯度爆炸 (例如限制在 200ms)
  d_queue = min(d_queue, 200.0)
  
  return d_prop + d_queue

# ==========================================
# 2. 样本生成逻辑 (单次)
# ==========================================
def generate_single_sample(topo_gen, fixed_G):
  """
  在固定拓扑上，随机生成流量状态，并计算 Dijkstra 标签。
  """
  # 1. 复制固定拓扑 (因为我们要修改它的动态属性)
  # 这是一个浅拷贝，结构不变，但属性可以改
  G = fixed_G.copy()
  
  # 2. 域随机化 (Domain Randomization)
  # 为每条链路生成随机的利用率，模拟当前时刻的拥塞
  for u, v, data in G.edges(data=True):
    # 随机利用率 0% - 95%
    # 使用 Beta 分布可以让高负载情况出现得更自然 (可选，这里先用 Uniform)
    load = random.uniform(0.0, 0.95)
    
    # 读取固定的物理属性 (从 GraphML 加载进来的)
    # 注意：NetworkX 读取 GraphML 后属性可能是字符串，需确保转换
    # 你的 generate_std_nsfnet.py 存的时候是 float，这里读出来应该是 float
    capacity = float(data.get('bandwidth', 100.0))
    prop_delay = float(data.get('delay', 5.0))
    
    # 计算理论总延迟 (这是 Agent 要预测的目标！)
    total_delay = calculate_theoretical_delay(prop_delay, capacity, load)
    
    # 更新图属性 (作为 GNN 的输入特征)
    # 我们将 total_delay 写入 'delay'，因为这是当前时刻的真实延迟
    G[u][v]['utilization'] = load
    G[u][v]['delay'] = total_delay
    G[u][v]['loss'] = 0.0 # 预训练阶段暂时忽略丢包，专注延迟
    
    # 同时也更新 NetworkGenerator 需要的属性，防止 get_pyg_data 出错
    G[u][v]['capacity'] = capacity

  # 3. 随机选择源宿节点
  nodes = list(G.nodes())
  s, d = random.sample(nodes, 2)
  
  # 4. 生成标签 (使用 Dijkstra 找理论最优路)
  # 我们希望 Agent 学会寻找 total_delay 最小的路径
  try:
      path_nodes = nx.dijkstra_path(G, s, d, weight='delay')
  except nx.NetworkXNoPath:
      return None

  # 转换为边标签 (0/1)
  # 获取 edge_index (这一步调用 NetworkGenerator 的辅助函数)
  from MS.Env.NetworkGenerator import get_pyg_data_from_nx
  pyg_data, _ = get_pyg_data_from_nx(G, s, d, DEFAULT_CONFIG)
  
  # 标记最短路上的边
  path_edges = set()
  for i in range(len(path_nodes) - 1):
      u, v = path_nodes[i], path_nodes[i+1]
      path_edges.add((u, v))
      path_edges.add((v, u))

  num_edges = pyg_data.edge_index.shape[1]
  labels = torch.zeros(num_edges, dtype=torch.float)
  for i in range(num_edges):
      u = pyg_data.edge_index[0, i].item()
      v = pyg_data.edge_index[1, i].item()
      if (u, v) in path_edges:
          labels[i] = 1.0
          
  pyg_data.y = labels
  return pyg_data

# ==========================================
# 3. Dataset 类
# ==========================================
class DynamicGraphDataset(IterableDataset):
	def __init__(self, config, max_samples=10000):
		self.config = config
		self.max_samples = max_samples
		
		# 初始化生成器并加载固定拓扑
		# 这样只读取一次文件，不用每次生成都读 IO
		self.topo_gen = TopologyGenerator(config)
		# 确保这里路径是对的，指向你刚才生成的标准文件
		self.fixed_G = self.topo_gen.load_topology("nsfnet_standard.graphml")
		print(f"[Dataset] Loaded Fixed Benchmark: {len(self.fixed_G.nodes())} nodes")

	def __iter__(self):
		worker_info = torch.utils.data.get_worker_info()
		if worker_info is None:
			iter_range = range(self.max_samples)
		else:
			# 多进程分片逻辑
			per_worker = int(math.ceil(self.max_samples / float(worker_info.num_workers)))
			iter_range = range(per_worker)
				
		for _ in iter_range:
			sample = generate_single_sample(self.topo_gen, self.fixed_G)
			if sample is not None:
				yield sample