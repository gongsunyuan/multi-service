import torch.nn as nn

class actor(nn.Module):
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