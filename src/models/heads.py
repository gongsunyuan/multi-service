import torch.nn as nn

class actor(nn.Module):
    """
    演员网络，用于生成动作。
    
    参数:
        input_dim: 输入维度，通常是状态空间的维度。
            current_node: 当前节点的状态向量。
            target_node: 目标节点的状态向量。
            neighbor_nodes: 当前节点的邻居节点状态向量。
            edge_attrs: 当前节点与邻居节点之间的边特征向量。
    输出:
        动作值，范围通常在 [-1, 1] 之间。
    """
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1))
  
    def forward(self, x):
        return self.net(x)

class critic(nn.Module):
    """
    评论家网络，用于评估状态价值。

    参数:
        input_dim: 输入维度，通常是状态空间的维度。
            current_node: 当前节点的状态向量。
            target_node: 目标节点的状态向量。
    输出:
        状态价值估计值。
    """
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1))
    
    def forward(self, x):
        return self.net(x)  