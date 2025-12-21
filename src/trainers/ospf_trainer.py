import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os

class OSPFPreTrainer:
  def __init__(self, agent, config, writer=None):
    """
    OSPF 预训练引擎：利用模仿学习初始化 GNN 骨干网络
    """
    self.agent = agent 
    self.config = config
    self.writer = writer
    self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # 预训练阶段主要优化 GNN、Actor 和 Critic
    # 建议过滤掉 FilmGenerator 以保持其初始 Identity 状态
    trainable_params = [
      {'params': self.agent.gnn.parameters()},
      {'params': self.agent.actor.parameters()},
      {'params': self.agent.critic.parameters()}]
    
    self.optimizer = torch.optim.Adam(trainable_params, lr=config.train.lr)
    self.global_step = 0

  def train_epoch(self, loader, epoch):
    self.agent.train()
    # 显式冻结 Film 模块参数
    self.agent.film.requires_grad_(False) 
    
    total_actor_loss = 0
    total_critic_loss = 0
    correct_edges = 0
    total_edges = 0

    for i, batch in enumerate(loader):
      batch = batch.to(self.device)
      # 使用全 1 占位指纹，因为 OSPF 与业务意图无关
      fingerprints = torch.ones((batch.num_graphs, self.config.model.fingerprint_dim, 2)).to(self.device)
      
      # 1. 批量获取节点嵌入 (全图并行)
      node_embeds = self.agent.get_node_embeddings(batch, fingerprints) 
      
      # 2. 向量化构造 Actor 输入 (满足 3H + Edge_Dim 规范) 
      u_idx, v_idx = batch.edge_index
      curr_feats = node_embeds[u_idx]     # (E_total, H)
      neigh_feats = node_embeds[v_idx]    # (E_total, H)
      edge_feats = batch.edge_attr        # (E_total, E_dim) 
      
      # 处理目标节点特征映射
      target_global_idx = batch.target_idx + batch.ptr[:-1] 
      graph_ids_per_edge = batch.batch[u_idx] # 获取每条边属于哪张图
      target_feats = node_embeds[target_global_idx[graph_ids_per_edge]] # (E_total, H)

      # 拼接: [Current | Target | Neighbor | Edge] 
      actor_input = torch.cat([curr_feats, target_feats, neigh_feats, edge_feats], dim=-1) 

      logits = self.agent.actor(actor_input).squeeze(-1) 

      # 3. 计算 Actor Loss (二分类：边是否在最短路上) 
      pos_weight = torch.tensor([5.0]).to(self.device) # 给正样本 5 倍权重
      actor_loss = F.binary_cross_entropy_with_logits(logits, batch.y, pos_weight=pos_weight)

      # 4. 计算 Critic Loss (路径代价 MSE)
      curr_global_idx = batch.curr_idx + batch.ptr[:-1]
      predicted_val = self.agent.evaluate_value(node_embeds, curr_global_idx, target_global_idx).squeeze(-1) 
      critic_loss = F.mse_loss(predicted_val, batch.target_value) 

      # 5. 反向传播
      loss = actor_loss + 0.5 * critic_loss
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      # 统计
      preds = (torch.sigmoid(logits) > 0.5).float()
      correct_edges += (preds == batch.y).sum().item()
      total_edges += batch.y.size(0)
      total_actor_loss += actor_loss.item()
      total_critic_loss += critic_loss.item()
      pos_mask = (batch.y == 1.0)
      if pos_mask.sum() > 0:
        pos_correct = (preds[pos_mask] == 1.0).sum().item()
        pos_total = pos_mask.sum().item()
        recall = pos_correct / pos_total

      if self.writer:
        # step = epoch * len(loader) + i
        self.writer.add_scalar("Pretrain/Actor_Loss_Step", actor_loss.item(), self.global_step)
        self.writer.add_scalar("Pretrain/Critic_Loss_Step", critic_loss.item(), self.global_step)
        self.global_step += 1

    avg_actor = total_actor_loss / (i+1)
    avg_critic = total_critic_loss / (i+1)
    accuracy = correct_edges / total_edges
    
    return avg_actor, avg_critic, accuracy, recall
  

