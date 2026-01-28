import torch.nn as nn
from multi_service.utils import AttrDict, CheckpointManager

class LSTMClassifier(nn.Module):
    """
    lstm 分类器，用于分类任务。
    
    参数:
        input_dim: fingerprint的长度。
    输出:
        分类结果，通常是一个概率值，范围在 [0, 1] 之间。
    """
    def __init__(self, config:AttrDict):
        super().__init__()
        self.ckpt_manager = CheckpointManager(config.path.ckpt_dir)
        self.lstm = nn.LSTM(config.model.fingerprint_dim, config.model.hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(config.model.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2))
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])  