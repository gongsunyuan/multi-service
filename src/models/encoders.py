import torch.nn as nn
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F

from ..utils import AttrDict

class FilmGenerator(nn.Module):
    """
    [模块1] 意图生成器
    输入: 流量指纹 (Sequence)
    输出: GNN 的调制参数 (Gamma, Beta)
    """

    def __init__(self, config:AttrDict):
        super().__init__()
        self.lstm = nn.LSTM(config.model.fingerprint_dim, config.model.hidden_dim, batch_first=True)
        # 输出层生成所有 GNN 层的 Gamma 和 Beta
        # 每一层需要 2 个参数向量 (gamma, beta)，共 gnn_layers 层
        self.head = nn.Linear(config.model.hidden_dim, config.model.hidden_dim * 2 * config.model.gnn_layers)
        
        # 初始化为 Identity (Gamma=1, Beta=0)
        self._init_weights()

    def _init_weights(self):
        # 将 head 的权重和偏置设为极小值/零，使得初始输出接近 0
        nn.init.constant_(self.head.weight, 0.0)
        nn.init.constant_(self.head.bias, 0.0)

    def forward(self, fingerprint):
        """
        forward 的 Docstring
        
        :param fingerprint: 流量指纹 (Batch, Seq_Len, fingerprint_dim)
        :return: GNN FiLM 调制参数 (Batch, gnn_layers * 2 * hidden_dim)
        """
        # fingerprint: (Batch, Seq_Len, 2)
        _, (h_n, _) = self.lstm(fingerprint)
        # h_n: (1, Batch, Hidden) -> (Batch, Hidden)
        intent = h_n.squeeze(0)

        # 生成参数 (Batch, Layers * 2 * Hidden)
        params = self.head(intent)
        
        # 由于初始化为0，这里 +1.0 让 Gamma 初始为 1
        # Split 逻辑: [Layer1_Gamma, Layer1_Beta, Layer2_Gamma, ...]
        return params

class FilmGNN(nn.Module):
    """
    [模块2] 全局感知骨干
    融合: 节点特征 + 边特征 + FiLM 调制
    """
    def __init__(self, config):
            """
            __init__ 的 Docstring
            
            :param node_dim: 节点特征维度
            :param edge_dim: 边特征维度
            :param hidden_dim: GNN 隐藏层维度
            :param num_layers: GNN 层数
            :param heads: 多头注意力头数
            """
            super().__init__()
            self.num_layers = config.model.gnn_layers
            self.hidden_dim = config.model.hidden_dim

            heads = config.model.gnn_heads  
            node_dim = config.model.node_dim
            edge_dim = config.model.edge_dim
            hidden_dim = config.model.hidden_dim

            # 初始编码
            self.node_encoder = nn.Linear(node_dim, hidden_dim)
            self.edge_encoder = nn.Linear(edge_dim, edge_dim) # 边特征维度通常较小，可先升维或保持
            
            # GAT 层列表
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()
            
            for _ in range(self.num_layers):
                # edge_dim 参数开启边特征融合
                self.convs.append(GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_dim, heads=heads, concat=False))
                self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, x, edge_index, edge_attr, film_params, batch_vector=None):
        
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        # 2. 确保 view 的维度包含 Batch Size
        batch_size = film_params.size(0)
        params = film_params.view(batch_size, self.num_layers, 2, self.hidden_dim)
        
        for i in range(self.num_layers):
            x_in = x
            x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            x = self.norms[i](x)
            
            # 3. 取出第 i 层的参数 (Batch, Hidden)
            layer_params = params[:, i, :, :] 
            gamma = layer_params[:, 0, :] + 1.0 
            beta = layer_params[:, 1, :]        
            
            # ================= [标准解法核心] =================
            if batch_vector is not None:
                # 利用 batch_vector (N,) 从 gamma (B, H) 中查表
                # 结果变为 (N, H)，与节点数量一一对应
                gamma_expanded = gamma[batch_vector] 
                beta_expanded = beta[batch_vector]
                x = x * gamma_expanded + beta_expanded
            else:
                # 兼容单图推理的情况 (batch_size=1)
                x = x * gamma + beta
            # ==================================================

            x = F.relu(x)
            x = x + x_in 
            
        return x

