import torch
import torch.nn as nn
import torch.optim as optim

from multi_service.utils import BankTrafficManager

class LSTMTrainer:
    def __init__(self, lstm, config):
        self.lstm = lstm
        self.config = config
        self.device = config.device  # 建议在 config 中统一指定设备
        self.optimizer = optim.Adam(lstm.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
        
        # 针对分类任务，使用交叉熵损失函数
        self.criterion = nn.CrossEntropyLoss()

    def train_one_epoch(self, traffic_generator):
        self.lstm.train()
        total_loss = 0
        
        # 1. 获取包含多个 flow 对象的列表
        flows = traffic_generator.generate_batch(self.config.train.batch_size)
        if not flows:
            return 0

        # 2. 【核心改进】将 list 转换为 Batch Tensor
        # 假设每个 fingerprint 已经是 (30, 2) 或 (1, 30, 2)
        # 我们使用 torch.stack 将其合并为 (batch_size, 30, 2)
        
        batch_x = []
        batch_y = []
        
        for flow in flows:
            # 确保去掉多余的 batch 维度，统一处理成 (seq_len, feature_dim)
            fp = flow.fingerprint.float()
            if isinstance(fp, torch.Tensor):
                fp = fp.squeeze() # 变为 (30, 2)
            else:
                fp = torch.tensor(fp, dtype=torch.float32)
                
            batch_x.append(fp)
            batch_y.append(flow.label)

        # 堆叠成高维张量
        inputs = torch.stack(batch_x).to(self.device)   # 形状: (batch_size, 30, 2)
        labels = torch.tensor(batch_y, dtype=torch.long).to(self.device) # 形状: (batch_size)

        # 3. 标准批处理训练步骤
        self.optimizer.zero_grad()
        
        # 前向传播 (一次性跑整个 batch)
        outputs = self.lstm(inputs) # 形状: (batch_size, 2)
        
        # 计算损失
        loss = self.criterion(outputs, labels)
        
        # 反向传播与优化
        loss.backward()
        self.optimizer.step()
        
        return loss.item()