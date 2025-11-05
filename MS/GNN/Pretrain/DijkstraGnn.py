import random
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch.utils.data import IterableDataset
from ...Env.NetworkGenerator import TopologyGenerator

GLOBAL_STATS = {
  'delay': {'min': 1.0, 'max': 10.0},     # 假设 G.edges() 延迟范围
  'bw':    {'min': 100.0, 'max': 1000.0}, # 假设 G.edges() 带宽范围
}

def generate_single_sample(topo_gen: TopologyGenerator, global_stats: dict) -> Data:
  """
  不断尝试生成拓扑，直到获得一个合法的（S->D可达）带标签样本。
  """
  while True:
    # 1. 生成拓扑和 S/D 对
    G_nx = topo_gen.generate_topology()
    try:
      S, D = topo_gen.select_source_destination()
    except ValueError: # 防止偶尔生成的图没有足够的节点或不可达
      continue

    # 2. 转换为 PyG Data 对象
    data, G_nx_with_attrs = get_pyg_data_from_nx(G_nx, S, D, global_stats)
    
    # 3. 计算专家标签
    y_true = generate_expert_label(G_nx_with_attrs, S, D, data.edge_index)
    
    if y_true is not None:
      # [关键] 将标签直接挂载到 Data 对象上，名为 'y'
      data.y = y_true
      return data

# === 新增：动态图数据集 ===
class DynamicGraphDataset(IterableDataset):
  def __init__(self, topo_gen, global_stats, max_samples_per_epoch):
    """
    :param max_samples_per_epoch: 每个 epoch 生成多少个样本后停止，防止无限循环。
    """
    self.topo_gen = topo_gen
    self.global_stats = global_stats
    self.max_samples = max_samples_per_epoch

  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:  # 单进程模式
      iter_start = 0
      iter_end = self.max_samples
    else:  
      # 多进程模式 (DataLoader num_workers > 0)
      # 将总任务量分配给各个 worker
      per_worker = int(math.ceil(self.max_samples / float(worker_info.num_workers)))
      worker_id = worker_info.id
      iter_start = worker_id * per_worker
      iter_end = min(iter_start + per_worker, self.max_samples)
        
    count = iter_start
    while count < iter_end:
      yield generate_single_sample(self.topo_gen, self.global_stats)
      count += 1

def normalize_features(
  edge_attr_tensor: torch.Tensor, 
  degrees_tensor: torch.Tensor, 
  global_stats: dict
  ) -> tuple[torch.Tensor, torch.Tensor]:
  """
  使用全局统计数据 (global_stats) 对边特征和节点度进行 Min-Max 归一化。

  :param edge_attr_tensor: (Num_Edges, 2) 形状的原始 [delay, bw] 张量
  :param degrees_tensor: (Num_Nodes, 1) 形状的原始 [degree] 张量
  :param global_stats: 包含 {'delay': {'min': x, 'max': y}, ...} 的字典
  :return: (edge_attr_norm, degrees_norm) 两个归一化后的张量
  """
  
  # --- 1. 归一化 Edge Attributes (Delay, Bandwidth) ---
  edge_attr_norm = edge_attr_tensor.clone()
  if edge_attr_tensor.numel() > 0:
    # 提取延迟 (delay) 的全局统计
    d_min = global_stats['delay']['min']
    d_range = global_stats['delay']['max'] - d_min
    
    # 提取带宽 (bw) 的全局统计
    b_min = global_stats['bw']['min']
    b_range = global_stats['bw']['max'] - b_min

    # 归一化延迟 (列 0)
    edge_attr_norm[:, 0] = (edge_attr_tensor[:, 0] - d_min) / (d_range + 1e-6)
    # 归一化带宽 (列 1)
    edge_attr_norm[:, 1] = (edge_attr_tensor[:, 1] - b_min) / (b_range + 1e-6)
    
    # 将值裁剪到 [0, 1] 范围，防止因随机波动超出范围
    edge_attr_norm = edge_attr_norm.clamp(0, 1)

  # --- 2. 归一化 Node Degrees ---
  degrees_norm = degrees_tensor.clone()

  return edge_attr_norm, degrees_norm

# 假设的 GNN 输入准备函数
def get_pyg_data_from_nx(G: nx.Graph, S_node: int, D_node: int, global_stats: dict):
  """
  辅助函数：将 NetworkX 图 G 转换为 PyG Data 对象，并添加 S/D 标记。
  """
  # 1. 边索引 (Edge Index)
  source_nodes, target_nodes = [], []
  edge_attrs_raw = []
  for u, v, data in G.edges(data=True): 
    source_nodes.extend([u, v])
    target_nodes.extend([v, u])
    # 归一化的 delay 和 bw (假设已归一化)
    attr = [data.get('delay', 1.0), data.get('bandwidth', 1000.0)] 
    edge_attrs_raw.extend([attr, attr])
      
  edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
  edge_attr_tensor = torch.tensor(edge_attrs_raw, dtype=torch.float)

  

  # 2. 节点特征 (Node Features)
  num_nodes = G.number_of_nodes()
  # 简单使用节点度作为特征
  degrees_tensor = torch.tensor([G.degree(n) for n in range(num_nodes)], dtype=torch.float).unsqueeze(1)
  
  edge_attr, degrees_norm = normalize_features(
    edge_attr_tensor, 
    degrees_tensor, 
    global_stats
  )
    
  # [关键] 标记源和目的节点
  is_source = torch.zeros(num_nodes, 1, dtype=torch.float)
  is_dest = torch.zeros(num_nodes, 1, dtype=torch.float)
  is_source[S_node] = 1.0
  is_dest[D_node] = 1.0
  
  node_features = torch.cat([degrees_norm, is_source, is_dest], dim=1) # (Num_Nodes, 3)

  return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr), G

# 正确答案
def generate_expert_label(G: nx.Graph, S_node: int, D_node: int, edge_index: torch.Tensor):
  """
  使用 Dijkstra (以 'delay' 为权重) 计算专家路径，并生成 GNN 标签。
  返回一个 (num_edges,) 的张量，1.0 代表在最短路径上, 0.0 代表不在。
  """
  # 1. 计算 Dijkstra 最短路径 (基于延迟)
  try:
    path_nodes = nx.dijkstra_path(G, S_node, D_node, weight='delay')
  except nx.NetworkXNoPath:
    return None # 无法到达

  # 2. 将节点路径转换为边集合
  path_edges = set()
  for i in range(len(path_nodes) - 1):
    u, v = path_nodes[i], path_nodes[i+1]
    path_edges.add((u, v))
    path_edges.add((v, u)) # 考虑无向图

  # 3. 创建标签向量 (与 edge_index 严格对应)
  # edge_index 的形状是 (2, num_edges*2)
  num_total_edges = edge_index.shape[1]
  labels = torch.zeros(num_total_edges, dtype=torch.float)

  for i in range(num_total_edges):
    u = edge_index[0, i].item()
    v = edge_index[1, i].item()
    if (u, v) in path_edges:
      labels[i] = 1.0
          
  return labels

# --- 关键架构：FiLM-GNN 模型 (阶段 1B 模式) ---
# 注意：这只是 GNN 主体和输出头，不包含 LSTM 和 FiLM 生成器

class GNNPretrainModel(nn.Module):
  def __init__(self, node_feat_dim, gnn_dim, edge_feat_dim, num_layers=6):
    super().__init__()
    self.gnn_dim = gnn_dim
    self.num_layers = num_layers
      
      # 节点特征嵌入层 (将 (N, 3) 映射到 (N, D_gnn))
    self.node_embed = nn.Linear(node_feat_dim, gnn_dim)
      
    self.convs = nn.ModuleList()
    self.layer_norms = nn.ModuleList()

    for _ in range(num_layers):
      # [关键] 你的自定义 FilmGCNLayer 
      # 在这里我们简化为标准的 LayerNorm + GCNConv
      self.layer_norms.append(nn.LayerNorm(gnn_dim))
      # PyG 的 GCNConv
      self.convs.append(
        pyg_nn.GATConv(gnn_dim, gnn_dim, heads=2, concat=False, edge_dim=edge_feat_dim)
      )
      
    # [关键] 临时路径输出头 (GNN 头)
    # 目标：输出每条边的 logit (分数)
    # 方法：(h_u || h_v || edge_attr) -> MLP -> 1 (logit)
    # 我们简化：(h_u + h_v) -> MLP -> 1 (logit)
    # 更好的方法：使用 pyg_nn.EdgeCNN
      
    # 临时输出头：将最终的 *节点* 嵌入转换为 *边* 的 logits
    self.edge_output_head = nn.Sequential(
      nn.Linear(gnn_dim * 2, gnn_dim), # (h_u || h_v)
      nn.ReLU(),
      nn.Linear(gnn_dim, 1)
    )

  def forward(self, data, manual_gamma, manual_beta):
    """
    GNN 主体的前向传播
    manual_gamma/beta: (L, D_gnn) 的张量，用于中和 FiLM
    """
    # h ：隐藏节点特征
    h = self.node_embed(data.x) # 初始节点特征
    
    for l in range(self.num_layers):
      # 1. 内部归一化 (GNN 主体的一部分)
      h_norm = self.layer_norms[l](h)
      
      # 2. [关键] FiLM 中和 (手动注入)
      # (1, D_gnn) * (N, D_gnn) + (1, D_gnn) -> (N, D_gnn)
      gamma_l = manual_gamma[l].unsqueeze(0)
      beta_l = manual_beta[l].unsqueeze(0)
      
      h_modulated = gamma_l * h_norm + beta_l
      
      # 3. 图卷积 (GNN 主体的一部分)
      h = self.convs[l](
        h_modulated, 
        data.edge_index, 
        edge_attr=data.edge_attr
      )
      h = h.relu()
    
    # 4. 临时路径输出头 (GNN 头)
    # 为 edge_index 中的每条边计算 logit
    edge_src = data.edge_index[0]
    edge_dst = data.edge_index[1]
    
    h_src = h[edge_src] # (Num_Edges, D_gnn)
    h_dst = h[edge_dst] # (Num_Edges, D_gnn)
    
    # 拼接 (h_u || h_v)
    edge_features = torch.cat([h_src, h_dst], dim=-1) # (Num_Edges, D_gnn * 2)
    
    # (Num_Edges, 1) -> (Num_Edges,)
    edge_logits = self.edge_output_head(edge_features).squeeze() 
    
    return edge_logits


