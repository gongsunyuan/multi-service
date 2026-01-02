
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Warmup Trainer 模块

该模块实现了 FiLMPPOAgent 的预热训练逻辑，用于预训练 GNN 理解网络拓扑和拥塞信息，
以及预训练 Actor 基于这些信息做出决策。

主要功能：
1. 初始化训练器，配置优化器和损失函数
2. 执行训练 epoch，包括数据加载、前向传播、损失计算和反向传播
3. 管理模型检查点
"""

import torch
from torch_geometric.data import DataLoader
from torch_geometric.data.batch import Batch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import IterableDataset

from agents import FiLMPPOAgent
from utils import CheckpointManager, AttrDict

class WarmupTrainer:
    def __init__(self, agent: FiLMPPOAgent, config: AttrDict, writer: SummaryWriter | None = None) -> None:  
        self.agent = agent
        self.config = config
        self.writer = writer
        self.device = torch.device(config.device)
        
        self.checkpoint_manager = CheckpointManager(config.path.checkpoint_dir)
        # 全参数训练：GNN 和 Actor 一起练
        # GNN 负责看懂 "这条路很堵" 和 "那条路通向终点"
        # Actor 负责综合这两个信息做决策
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=config.train.lr)
        # 使用带权重的 BCE Loss (正样本少，负样本多)
        pos_weight = config.train.get('pos_weight', 3.0) if hasattr(config.train, 'get') else getattr(config.train, 'pos_weight', 3.0)
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(self.device))

    def train_epoch(self, dataset: IterableDataset, epoch_idx: int) -> float:
        """
        训练一个 Epoch：
        1. 从数据集加载数据
        2. 前向传播：GNN提取特征，Actor生成预测
        3. 计算损失
        4. 反向传播更新模型参数
        
        params:
            dataset (IterableDataset): 训练数据集
            epoch_idx (int): 当前 Epoch 索引
            
        returns:
            float: 该 Epoch 的平均 Loss
        """
        # 配置批量大小
        batch_size = getattr(self.config.train, 'batch_size', 32) if hasattr(self.config.train, 'batch_size') else 32
        
        # 创建数据加载器
        loader = DataLoader(dataset, batch_size=batch_size)
        
        # 设置模型为训练模式
        self.agent.train()
        
        # 初始化损失统计
        total_loss = 0.0
        total_batches = 0
        
        # 遍历数据批次
        for step, batch in enumerate(loader):
            # 移动到指定设备
            batch = batch.to(self.device)
            
            # 创建虚拟指纹（当前未使用）
            dummy_fp = torch.ones((batch.num_graphs, 2), device=self.device)
            
            # 1. GNN 提取节点特征
            # 输入边属性包含利用率信息，GNN可以学习到拥塞情况
            node_embeds = self.agent.gnn(batch.x, batch.edge_index, batch.edge_attr, dummy_fp)
            
            # 2. 构造 Actor 输入
            # 获取边的源节点和目标节点索引
            row, col = batch.edge_index
            
            # 获取每个图的偏移量
            batch_offset = batch.ptr[:-1]
            
            # 计算目标节点在全局节点索引中的位置
            target_indices = batch.target_node + batch_offset
            
            # 获取源节点、邻居节点和目标节点的特征
            u_feat = node_embeds[row]  # 当前节点特征
            v_feat = node_embeds[col]  # 邻居节点特征
            target_feat = node_embeds[target_indices[batch.batch[row]]]  # 目标节点特征
            
            # 获取边特征
            edge_attrs = batch.edge_attr
            
            # 拼接特征：[当前节点, 目标节点, 邻居节点, 边特征]
            # 与 ppo_agent.py 中的顺序保持一致
            actor_in = torch.cat([u_feat, target_feat, v_feat, edge_attrs], dim=-1)
            
            # 3. Actor 生成预测
            logits = self.agent.actor(actor_in)
            
            # 4. 计算损失
            # 只计算训练掩码内的边
            mask = batch.train_mask
            loss = self.criterion(logits[mask], batch.y_guidance[mask])
            
            # 5. 反向传播更新参数
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 累加损失
            total_loss += loss.item()
            total_batches += 1
            
            # 记录每步的损失
            if self.writer is not None:
                self.writer.add_scalar('train/step_loss', loss.item(), epoch_idx * len(loader) + step)
        
        # 计算平均损失
        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
        
        # 记录Epoch平均损失
        if self.writer is not None:
            self.writer.add_scalar('train/epoch_loss', avg_loss, epoch_idx)
        
        return avg_loss
    
    def validate_epoch(self, dataset: IterableDataset, epoch_idx: int) -> float:
        """
        在验证集上评估模型性能
        
        params:
            dataset (IterableDataset): 验证数据集
            epoch_idx (int): 当前 Epoch 索引
            
        returns:
            float: 该 Epoch 的平均验证 Loss
        """
        # 配置批量大小
        batch_size = getattr(self.config.train, 'batch_size', 32) if hasattr(self.config.train, 'batch_size') else 32
        
        # 创建数据加载器
        loader = DataLoader(dataset, batch_size=batch_size)
        
        # 设置模型为评估模式
        self.agent.eval()
        
        # 初始化损失统计
        total_loss = 0.0
        total_batches = 0
        
        with torch.no_grad():
            # 遍历数据批次
            for step, batch in enumerate(loader):
                # 移动到指定设备
                batch = batch.to(self.device)
                
                # 创建虚拟指纹（当前未使用）
                dummy_fp = torch.ones((batch.num_graphs, 2), device=self.device)
                
                # 1. GNN 提取节点特征
                node_embeds = self.agent.gnn(batch.x, batch.edge_index, batch.edge_attr, dummy_fp)
                
                # 2. 构造 Actor 输入
                row, col = batch.edge_index
                batch_offset = batch.ptr[:-1]
                target_indices = batch.target_node + batch_offset
                
                # 获取特征
                u_feat = node_embeds[row]  # 当前节点特征
                v_feat = node_embeds[col]  # 邻居节点特征
                target_feat = node_embeds[target_indices[batch.batch[row]]]  # 目标节点特征
                
                # 获取边特征
                edge_attrs = batch.edge_attr
                
                # 拼接特征：[当前节点, 目标节点, 邻居节点, 边特征]
                # 与 ppo_agent.py 中的顺序保持一致
                actor_in = torch.cat([u_feat, target_feat, v_feat, edge_attrs], dim=-1)
                
                # 3. Actor 生成预测
                logits = self.agent.actor(actor_in)
                
                # 4. 计算损失
                mask = batch.train_mask
                loss = self.criterion(logits[mask], batch.y_guidance[mask])
                
                # 累加损失
                total_loss += loss.item()
                total_batches += 1
        
        # 计算平均损失
        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
        
        # 记录验证损失
        if self.writer is not None:
            self.writer.add_scalar('val/epoch_loss', avg_loss, epoch_idx)
        
        # 恢复训练模式
        self.agent.train()
        
        return avg_loss