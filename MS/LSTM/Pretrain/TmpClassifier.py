from MS.LSTM.PreferenceModule import PreferenceModule
import torch.nn as nn
import torch

class RNNClassifier(nn.Module):
  """
  阶段 1.A 的完整模型：[LSTM Body] + [临时 MLP 分类头]
  """
  def __init__(self, input_dim: int, rnn_hidden_dim: int, rnn_layers: int, num_classes: int):
    super().__init__()
    
    self.preference_module = PreferenceModule(
      input_dim, rnn_hidden_dim, rnn_layers
    )
    
    # 临时分类头 (用于监督学习，训练完成后将被丢弃)
    self.classifier_head = nn.Sequential(
      nn.Linear(rnn_hidden_dim, rnn_hidden_dim // 2),
      nn.ReLU(),
      nn.Linear(rnn_hidden_dim // 2, num_classes) 
    )

  def forward(self, x_fingerprint: torch.Tensor):
    # (B, N, C) -> (B, D_rnn)
    h_n_summary = self.preference_module(x_fingerprint)
    
    # (B, D_rnn) -> (B, NumClasses)
    logits = self.classifier_head(h_n_summary)
    return logits