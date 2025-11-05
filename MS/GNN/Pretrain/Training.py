import torch
import torch.nn as nn
import torch.optim as optim

from .DijkstraGnn import get_pyg_data_from_nx, generate_expert_label, GNNPretrainModel, GLOBAL_STATS
from ...Env.NetworkGenerator import TopologyGenerator

if __name__ == "__main__":
  print("🚀 开始阶段 1B: GNN 主体预训练 (模仿 Dijkstra)...")

  # 1. 初始化
  GNN_DIM = 128          # D_gnn 隐藏节点特征数
  EPOCHS = 6000          # 训练轮数
  NUM_LAYERS = 6        # L Gnn 层数
  NODE_FEAT_DIM = 3     # (degree, is_source, is_dest) 阶段特征数
  EDGE_FEAT_DIM = 2     # 边特征数
  LEARNING_RATE = 1e-4  # 学习率
  STEPS_PER_EPOCH = 200 #

  device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")         # 定义device
  topo_gen = TopologyGenerator(num_nodes_range=(20, 30), m_ba=2)                  # Networkx 拓扑生成器
  model = GNNPretrainModel(NODE_FEAT_DIM, GNN_DIM, EDGE_FEAT_DIM, NUM_LAYERS)     # Gnn 预训练模型
  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)                    # Adam 优化器

  model.to(device)                                                                # 将模型移到 device

  # 将 pos_weight 传入损失函数
  base_loss_fn = nn.BCEWithLogitsLoss()# 损失函数：二元交叉熵 (每条边是否在最短路径上)

  # 2. [关键] "中和" FiLM 的参数
  # 我们创建 gamma=1.0 和 beta=0.0 的常量张量
  GAMMA_NEUTRAL = torch.ones((NUM_LAYERS, GNN_DIM), dtype=torch.float).to(device)
  BETA_NEUTRAL = torch.zeros((NUM_LAYERS, GNN_DIM), dtype=torch.float).to(device)


  for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    
    for step in range(STEPS_PER_EPOCH):
      # 3. 生成数据和标签
      G_nx = topo_gen.generate_topology()
      S, D = topo_gen.select_source_destination()
      
      data, G_nx_with_attrs = get_pyg_data_from_nx(G_nx, S, D, GLOBAL_STATS)
      data = data.to(device)

      # 基于 'delay' 计算专家路径标签
      y_true_edge_labels = generate_expert_label(G_nx_with_attrs, S, D, data.edge_index)
      
      # 跳过不可达的图
      if y_true_edge_labels is None:                                
        continue
      
      y_true_edge_labels = y_true_edge_labels.to(device)
      # 4. 训练
      optimizer.zero_grad()
      
      num_pos = y_true_edge_labels.sum()
      num_neg = y_true_edge_labels.shape[0] - num_pos
      # [关键] 调用模型，并传入“中和”参数
      edge_logits = model(data, manual_gamma=GAMMA_NEUTRAL, manual_beta=BETA_NEUTRAL)
      
      if num_pos > 0 and num_neg > 0:
        pos_weight_value = num_neg / num_pos
        # 动态创建损失函数
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_value.to(device))
      else:
        # 如果没有正/负样本, 使用基础损失 (无权重)
        loss_fn = base_loss_fn
      
      # 5. 计算损失
      # 比较 GNN 的输出分数 (logits) 和 Dijkstra 标签
      loss = loss_fn(edge_logits, y_true_edge_labels)
      
      # --- [新] 计算准确率 ---
      # edge_logits 是原始分数, > 0.0 意味着模型预测为 1 (在路径上)
      predicted_labels = (edge_logits > 0.0).float()
      # y_true_edge_labels 是 0.0 或 1.0
      correct_predictions = (predicted_labels == y_true_edge_labels).float()
      accuracy = correct_predictions.mean() # 计算平均正确率

      loss.backward()
      optimizer.step()
      total_loss += loss.item()
      total_acc += accuracy.item() # <--- [新] 累加准确率
        
    avg_loss = total_loss / STEPS_PER_EPOCH
    avg_acc = total_acc / STEPS_PER_EPOCH
    if (epoch + 1) % 10 == 0:
      print(f"Epoch {epoch+1}/{EPOCHS}, 预训练损失: {avg_loss:.6f}, 准确率: {avg_acc*100:.2f}%")

  # 6. 保存预训练好的 GNN 主体
  # 注意：保存的是整个模型，在阶段 2 加载时需要选择性加载
  print("✅ 阶段 1B 完成。保存 GNN 主体权重...")
  # 我们只保存 GNN 主体（卷积层、归一化层）和节点嵌入层的权重
  # 丢弃 self.edge_output_head
  gnn_body_weights = {k: v for k, v in model.state_dict().items() if 'edge_output_head' not in k}
  torch.save(gnn_body_weights, 'pretrained-model-with-posWeight.pth')