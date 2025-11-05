import torch
import torch.nn as nn
import torch.nn.functional as F

class PreferenceModule(nn.Module):
  """
  (1) 偏好模块 (RNN 主体)
  读取 (N, C) 的流指纹，输出 (1, D_rnn) 的摘要向量 h_N。
  """
  def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
    super().__init__()
    # input_dim 必须是 2 (PacketSize, IAT)
    # hidden_dim 是 D_rnn
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    
    # 我们使用 LSTM，batch_first=True 使得输入形状为 (B, N, C)
    # B 在我们的 E2E 模型中通常是 1（一次处理一个流）
    self.lstm = nn.LSTM(
      input_size=input_dim,
      hidden_size=hidden_dim,
      num_layers=num_layers,
      batch_first=True,
      bidirectional=False # 路由决策必须是因果的，不能看“未来”的包
    )

  def forward(self, x_fingerprint: torch.Tensor):
    """
    输入:
      x_fingerprint (torch.Tensor): 形状为 (B, N, C)。
      例如 (1, 50, 2)
    输出:
      h_n_summary (torch.Tensor): 形状为 (B, D_rnn)。
      例如 (1, 128)
    """
    # 初始化隐藏状态 (h_0, c_0)
    # (num_layers * num_directions, B, D_rnn)
    h_0 = torch.zeros(self.num_layers, x_fingerprint.size(0), self.hidden_dim).to(x_fingerprint.device)
    c_0 = torch.zeros(self.num_layers, x_fingerprint.size(0), self.hidden_dim).to(x_fingerprint.device)
    
    # LSTM 的 output 是 (B, N, D_rnn)
    # hidden 是一个元组 (h_n, c_n)
    # h_n 是 (num_layers, B, D_rnn)
    _, (h_n, c_n) = self.lstm(x_fingerprint, (h_0, c_0))
    
    # 我们只需要最后一层的最终隐藏状态
    # h_n[-1] 的形状是 (B, D_rnn)
    h_n_summary = h_n[-1]
    
    return h_n_summary