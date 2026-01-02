
from agents import FiLMPPOAgent
from torch_geometric.data import DataLoader
from torch_geometric.data.batch import Batch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import IterableDataset
import torch

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
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=0.001)
        # 使用带权重的 BCE Loss (正样本少，负样本多)
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(self.device))

    def train_epoch(self, dataset: IterableDataset, epoch_idx: int) -> float:
        """
        训练一个 Epoch：采样数据 -> 存入 Memory -> 更新 Agent
        
        params:
            dataset (IterableDataset): 训练数据集
            epoch_idx (int): 当前 Epoch 索引
            
        returns:
            float: 该 Epoch 的平均 Loss
        """
        loader = DataLoader(dataset, batch_size=32, collate_fn=Batch.from_data_list)
        self.agent.train()
        total_loss = 0
        
        for step, batch in enumerate(loader):
            batch = batch.to(self.device)
            dummy_fp = torch.ones((batch.num_graphs, 2), device=self.device)
            
            # 1. GNN 提取特征
            # Input Edge Attr 包含了 utilization，所以 GNN 能看到拥塞
            node_embeds = self.agent.gnn(batch.x, batch.edge_index, batch.edge_attr, dummy_fp)
            
            # 2. 构造 Actor 输入
            row, col = batch.edge_index
            batch_offset = batch.ptr[:-1]
            target_indices = batch.target_node + batch_offset
            
            # 获取特征
            u_feat = node_embeds[row]
            v_feat = node_embeds[col]
            target_feat = node_embeds[target_indices[batch.batch[row]]]
            
            # 拼接：[源, 邻居, 终点]
            # Actor 必须同时看到：v 离 target 近不近？v 堵不堵？
            actor_in = torch.cat([u_feat, v_feat, target_feat], dim=-1)
            
            # 3. 预测
            logits = self.agent.actor(actor_in)
            
            # 4. Loss
            mask = batch.train_mask
            loss = self.criterion(logits[mask], batch.y_guidance[mask])
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(loader)