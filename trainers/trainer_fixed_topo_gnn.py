import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.loader import DataLoader 
from torch_geometric.data import Batch
from torch.utils.data import IterableDataset
from tqdm import tqdm
import os
import math
import numpy as np
import random
import argparse
import networkx as nx
import torch.nn.functional as F

# === 导入自定义模块 (假设这些路径在你本地是正确的) ===
from MS.GNN.FiLMGnn import FiLMGnn 
from MS.Env.MininetController import sample_path
from MS.Env.NetworkGenerator import TopologyGenerator, get_pyg_data_from_nx, DEFAULT_CONFIG

# === 辅助函数 ===
def setup(rank, world_size):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12355'
  torch.cuda.set_device(rank)
  dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
  dist.destroy_process_group()

def generate_expert_label(G, S_node, D_node, edge_index):
  """
  使用 Dijkstra 生成最短路标签
  """
  try:
    path_nodes = nx.dijkstra_path(G, S_node, D_node, weight='delay')
  except nx.NetworkXNoPath:
    return None

  path_edges = set()
  for i in range(len(path_nodes) - 1):
    u, v = path_nodes[i], path_nodes[i+1]
    path_edges.add((u, v))
    path_edges.add((v, u)) # 无向图/双向边处理

  num_total_edges = edge_index.shape[1]
  labels = torch.zeros(num_total_edges, dtype=torch.float)
  for i in range(num_total_edges):
    u, v = edge_index[0, i].item(), edge_index[1, i].item()
    if (u, v) in path_edges:
      labels[i] = 1.0
  return labels

# === 配置 ===
class Config:
  MAX_NODES_NUM = 14
  MIN_BW = 50.0
  MAX_BW = 600.0
  MIN_LOSS = 0.0
  MAX_LOSS = 5.0
  MIN_DELAY = 1.0
  MAX_DELAY = 200.0

# === 核心修改：基于固定图的数据生成 ===
def generate_sample_from_fixed_graph(topo_gen, fixed_graph, config):
  """
  从给定的 fixed_graph 中随机选择 S-D 对进行训练
  注意：虽然拓扑结构固定，但我们会重新生成 PyG 数据对象，
  因为 get_pyg_data_from_nx 可能会注入一些随机噪声或特定属性。
  """
  while True:
    try:
      # 在固定的图上随机选择起点和终点
      # 注意：这里需要修改 topo_gen.select_source_destination 
      # 或者我们自己手动在 nodes 中选，因为 topo_gen 内部可能没有绑定这个 fixed_graph
      nodes = list(fixed_graph.nodes())
      S, D = random.sample(nodes, 2)
      
      # 将 NetworkX 对象转换为 PyG Data
      # 注意：如果 get_pyg_data_from_nx 会修改 graph (inplace)，需要传入 fixed_graph.copy()
      # 这里为了安全起见，传入 copy
      data, G_with_attrs = get_pyg_data_from_nx(fixed_graph.copy(), S, D, config)
      
      # 生成标签
      y = generate_expert_label(G_with_attrs, S, D, data.edge_index)
      
      if y is not None:
        data.y = y
        return data
    except Exception as e:
      # print(f"Sample generation error: {e}")
      continue

class FixedGraphDataset(IterableDataset):
  def __init__(self, topo_gen, fixed_graph, config, min_samples_per_epoch, rank, world_size):
    self.topo_gen = topo_gen
    self.fixed_graph = fixed_graph
    self.config = config
    self.rank = rank
    self.world_size = world_size
    
    # 1. 生成所有可能的 (Source, Destination) 对
    nodes = list(fixed_graph.nodes())
    self.all_od_pairs = []
    for u in nodes:
        for v in nodes:
            if u != v:
                self.all_od_pairs.append((u, v))
    
    # 2. 计算每个 Epoch 需要重复遍历列表多少次才能达到 min_samples
    # NSFNet: 182 pairs. 如果 min_samples=6400, 需要重复约 35 次
    self.num_pairs = len(self.all_od_pairs)
    self.rounds_per_epoch = int(math.ceil(min_samples_per_epoch / self.num_pairs))
    
    # 计算每个 GPU 分到的任务量
    total_samples_actual = self.num_pairs * self.rounds_per_epoch
    self.samples_per_gpu = int(math.ceil(total_samples_actual / world_size))

  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    
    # 设置随机种子，确保每次 generate_pyg_data 里的 Delay 都是随机且不同的
    base_seed = torch.initial_seed() 
    if worker_info is not None:
      base_seed += worker_info.id
    
    unique_seed = base_seed + (self.rank * 100000)
    random.seed(unique_seed)
    np.random.seed(unique_seed % (2**32 - 1))
    
    # 确定当前进程/Worker 的迭代范围
    # 我们构建一个虚拟的长列表索引：[0, 1, ..., rounds * num_pairs]
    total_indices = self.num_pairs * self.rounds_per_epoch
    
    # 简单的线性切分：每个 GPU 跑一段
    # 注意：这里我们不需要 shuffle OD pairs，因为我们希望覆盖全
    # 但为了避免 Batch 里全是同一个 Source，我们在每一轮内部可以 shuffle 索引（可选）
    # 这里为了简单直接顺序取，因为 Batch Size 很大，会混合不同 OD
    
    start_idx = self.rank * self.samples_per_gpu
    end_idx = min(start_idx + self.samples_per_gpu, total_indices)
    
    # 考虑 num_workers
    if worker_info is not None:
      per_worker = int(math.ceil((end_idx - start_idx) / worker_info.num_workers))
      worker_start = start_idx + worker_info.id * per_worker
      worker_end = min(worker_start + per_worker, end_idx)
    else:
      worker_start = start_idx
      worker_end = end_idx

    # 开始生成数据
    for i in range(worker_start, worker_end):
      # 取模拿到当前的 OD Pair
      # 这样保证了我们在不断的循环列表
      pair_idx = i % self.num_pairs
      S, D = self.all_od_pairs[pair_idx]
      
      try:
        # 关键：这里每次调用都会基于 config 生成新的随机 delay
        # 传入 fixed_graph.copy() 确保不污染原始拓扑结构
        data, G_with_attrs = get_pyg_data_from_nx(self.fixed_graph.copy(), S, D, self.config)
        
        y = generate_expert_label(G_with_attrs, S, D, data.edge_index)
        if y is not None:
          data.y = y
          yield data
      except Exception as e:
        # print(f"Error: {e}")
        continue

# === 损失函数 (保持不变) ===
class FocalLoss(nn.Module):
  def __init__(self, alpha=0.85, gamma=2.0, logits=True, reduce=True):
    super(FocalLoss, self).__init__()
    self.alpha = alpha
    self.gamma = gamma
    self.logits = logits
    self.reduce = reduce

  def forward(self, inputs, targets):
    if self.logits:
      BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    else:
      BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-BCE_loss)
 
    alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
    F_loss = alpha_t * (1-pt)**self.gamma * BCE_loss

    if self.reduce: return torch.mean(F_loss)
    else: return F_loss

# === 训练主逻辑 ===
def train_worker(rank, world_size):
  try:
    # 1. 初始化
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    
    if rank == 0:
      print(f"🚀 启动 DDP 固定拓扑训练 | GPU: {world_size} | PID: {os.getpid()}")

    # 2. 参数
    EPOCHS = 6000          
    BATCH_SIZE = 512       
    TOTAL_SAMPLES = 6400   
    LEARNING_RATE = 5e-4 # 调大了 LR，因为固定拓扑更容易收敛，太小会很慢
    
    NODE_FEAT_DIM = 10+Config.MAX_NODES_NUM 
    EDGE_FEAT_DIM = 5  
    GNN_DIM = 256
    NUM_LAYERS = 6
    SAVE_PATH = "./trained_model/gnn_nsf_fixed.pth" # 修改保存路径以区分
    LOAD_PATH = SAVE_PATH
    
    if rank == 0:
      os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    # 3. 准备固定拓扑数据
    topo_gen = TopologyGenerator(Config)
    # [关键修改] 在循环外加载一次固定拓扑
    try:
      fixed_graph_nx = topo_gen.load_topology("nsfnet.graphml")
      if rank == 0:
        print(f"✅ 成功加载固定拓扑: NSFNet (Nodes: {len(fixed_graph_nx.nodes)}, Edges: {len(fixed_graph_nx.edges)})")
    except Exception as e:
      if rank == 0: print(f"❌ 加载拓扑失败: {e}")
      return

    # 4. 模型
    model = FiLMGnn(
      node_feat_dim=NODE_FEAT_DIM, 
      gnn_dim=GNN_DIM, 
      edge_feat_dim=EDGE_FEAT_DIM, 
      num_layers=NUM_LAYERS
    ).to(device)

    if os.path.exists(LOAD_PATH):
      map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
      try:
        checkpoint = torch.load(LOAD_PATH, map_location=map_location)
        model.load_state_dict(checkpoint, strict=False)
        print(f"[Rank {rank}] 加载断点: {LOAD_PATH}")
      except Exception as e:
        if rank == 0: print(f"[Rank {rank}] 加载失败: {e}")
    
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)
    loss_fn = FocalLoss(alpha=0.98, gamma=8.0)

    best_recall = 0
    
    # 5. 循环
    for epoch in range(EPOCHS):
      model.train()
      
      # 使用 FixedGraphDataset
      dataset = FixedGraphDataset(
        topo_gen, 
        fixed_graph_nx,  # 传入固定图
        Config, 
        max_samples_per_epoch=TOTAL_SAMPLES, 
        rank=rank, 
        world_size=world_size
      )
      
      train_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=4, 
        pin_memory=True
      )

      total_loss = 0.0
      total_acc = 0.0
      total_tp = 0.0      
      total_real_p = 0.0  
      num_batches = 0
      
      if rank == 0:
        pbar = tqdm(train_loader, leave=False)
      else:
        pbar = train_loader

      for batch in pbar:
        if rank == 0: pbar.set_description(f"[Epoch {epoch+1}]: ")

        batch = batch.to(device)
        optimizer.zero_grad()
        
        edge_logits = model(batch)
        y_true = batch.y
        
        loss = loss_fn(edge_logits, y_true)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 统计
        current_loss = loss.item()
        predicted = (edge_logits > 0.0).float()
        current_acc = (predicted == y_true).float().mean().item()
        tp = ((predicted == 1) & (y_true == 1)).sum().item()
        real_p = (y_true == 1).sum().item() 

        total_loss += current_loss
        total_acc += current_acc
        total_tp += tp
        total_real_p += real_p
        curr_rec = tp / (real_p + 1e-8)

        num_batches += 1

        if rank == 0:
          pbar.set_postfix({"Loss": f"{current_loss:.4f}", "Rec": f"{curr_rec:.2%}"})

      # 汇总 (Reduce)
      metrics = torch.tensor([total_loss, total_acc, num_batches, total_tp, total_real_p], device=device)
      dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
      
      global_batches = metrics[2].item()
      global_tp = metrics[3].item()
      global_real_p = metrics[4].item()
      
      if global_batches > 0:
        avg_loss = metrics[0].item() / global_batches
        avg_acc = metrics[1].item() / global_batches
        avg_recall = global_tp / (global_real_p + 1e-8)
      else:
        avg_loss, avg_acc, avg_recall = 0.0, 0.0, 0.0

      scheduler.step()

      # Rank 0 负责验证和保存
      if rank == 0:
        current_lr = optimizer.param_groups[0]['lr']
        success_str = "N/A"

        # 保存逻辑
        if avg_recall > best_recall:
          best_recall = avg_recall
          torch.save(model.module.state_dict(), SAVE_PATH)
        
        # 验证逻辑：同样基于固定拓扑进行验证
        try:
          model.eval()
          # 验证时也使用同样的 fixed_graph_nx
          test_nodes = list(fixed_graph_nx.nodes())
          test_S, test_D = random.sample(test_nodes, 2)
          
          test_data, _ = get_pyg_data_from_nx(fixed_graph_nx.copy(), test_S, test_D, DEFAULT_CONFIG)
          test_data = test_data.to(device)
          
          with torch.no_grad():
            val_logits = model.module(test_data)
          
          # greedy=True 确保推理确定性
          _, success = sample_path(
            val_logits, test_data.edge_index, 
            test_S, test_D, greedy=True
          )
          success_str = "Yes" if success else "No"

        except Exception as e:
          success_str = f"Err"
        finally:
          model.train()

        pbar.write(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}, Rec: {avg_recall:.2%}, Best: {best_recall:.2%}, Valid: {success_str}")
      
      dist.barrier(device_ids=[rank])

  finally:
    cleanup()

# === 主程序 ===
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="GNN Fixed Topology Training")
  parser.add_argument('--gpus', type=str, default='', help='指定 GPU ID')
  args = parser.parse_args()

  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
  world_size = torch.cuda.device_count()
  
  if world_size < 1:
    print("需要 GPU 才能运行")
  else:
    mp.spawn(train_worker, args=(world_size,), nprocs=world_size, join=True)