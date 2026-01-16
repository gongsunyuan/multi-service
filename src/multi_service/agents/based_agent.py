import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch_geometric.data import Data

class BaseSDNAgent(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def get_node_embeddings(self, graph_data, fingerprint):
        """
            [抽象方法] 子类必须实现：如何从图数据和指纹中提取节点嵌入
        """
        pass

    @abstractmethod
    def get_action(self, state, node_embeds, **kwargs):
        """
            [抽象方法] 子类必须实现：具体的动作选择逻辑
        """
        pass

    @abstractmethod
    def evaluate_batch(self, *args, **kwargs):
        """
            [抽象方法] 子类必须实现：用于训练时的批量评估逻辑
        """
        pass

    @abstractmethod
    def update(self, memory):
        """
            [抽象方法] 子类必须实现：参数更新算法（如 PPO, DQN 等）
        """
        pass