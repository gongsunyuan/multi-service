import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import DataParallel
from torch_geometric.loader import DataListLoader
from tqdm import tqdm
import torch.multiprocessing

from .DijkstraGnn import GNNPretrainModel, GLOBAL_STATS, DynamicGraphDataset
from ...Env.NetworkGenerator import TopologyGenerator

if __name__ == "__main__":
  try:
    torch.multiprocessing.set_start_method('spawn', force=True)
  except RuntimeError:
    pass
  try:
    torch.multiprocessing.set_sharing_strategy('file_system')
  except RuntimeError:
    pass

  print("🚀 开始阶段 1B: GNN 主体预训练 (最终冲刺版)...")

  # --- 1. 超参数配置 ---
  EPOCHS = 200          # [冲刺优化] 增加到 150 轮，给低 LR 更多时间
  GNN_DIM = 256         # [冲刺优化] 宽度翻倍：128 -> 256
  NUM_LAYERS = 6
  BATCH_SIZE = 128       
  SAMPLES_PER_EPOCH = 6400
  LEARNING_RATE = 1e-3
  NODE_FEAT_DIM = 5
  EDGE_FEAT_DIM = 2

  if torch.cuda.is_available():
      torch.cuda.init()
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  # --- 2. 初始化组件 ---
  topo_gen = TopologyGenerator(num_nodes_range=(20, 30), m_ba=2)
  model = GNNPretrainModel(NODE_FEAT_DIM, GNN_DIM, EDGE_FEAT_DIM, NUM_LAYERS)
  model = model.to(device)

  if torch.cuda.device_count() > 1:
      print(f"✨ 启用 {torch.cuda.device_count()} 张 GPU 进行 PyG DataParallel 加速")
      model = DataParallel(model)
      
  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

  from torch.optim import lr_scheduler
  # [微调] patience 从 5 增加到 8，让它在降 LR 前多尝试一会儿
  scheduler = lr_scheduler.ReduceLROnPlateau(
      optimizer, mode='min', factor=0.1, patience=8, threshold=0.001, min_lr=1e-6
  )

  POS_WEIGHT_FIXED = torch.tensor([15.0]).to(device)
  loss_fn = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT_FIXED)

  # --- 3. 训练循环 ---
  for epoch in range(EPOCHS):
    model.train()
    dataset = DynamicGraphDataset(topo_gen, GLOBAL_STATS, max_samples=SAMPLES_PER_EPOCH)
    train_loader = DataListLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")

    for batch_data_list in pbar:
      optimizer.zero_grad()
      edge_logits = model(batch_data_list)
      y_true = torch.cat([data.y for data in batch_data_list]).to(device)
      loss = loss_fn(edge_logits, y_true)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()

      current_loss = loss.item()
      total_loss += current_loss
      predicted = (edge_logits > 0.0).float()
      current_acc = (predicted == y_true).float().mean().item()
      total_acc += current_acc
      num_batches += 1
      pbar.set_postfix({"Loss": f"{current_loss:.4f}", "Acc": f"{current_acc:.2%}"})

    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1} 完成. Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.2%}, LR: {current_lr:.2e}")

    scheduler.step(avg_loss)

    if (epoch + 1) % 10 == 0:
      model_to_save = model.module if isinstance(model, DataParallel) else model
      torch.save(model_to_save.state_dict(), f'./MS/GNN/Pretrain/gnn_pretrained_epoch_{epoch+1}.pth')

  print("✅ 阶段 1B 预训练完成！")
  model_to_save = model.module if isinstance(model, DataParallel) else model
  torch.save(model_to_save.state_dict(), './MS/GNN/pretrained_model.pth')