import torch

def compute_advantages(rewards, values, is_terminals, gamma=0.99, lam=0.95):
  """
  计算 GAE 优势函数
  rewards: [T], values: [T+1] (需要多存一个最后状态的 value)
  """
  
  advantages = []
  gae = 0
  
  # 从后往前计算
  for i in reversed(range(len(rewards))):
    # TD 误差 delta = 当前奖励 + 下一步折扣价值 - 当前价值
    # 如果是最后一步，next_value 就是 0

    mask = 1.0 - is_terminals[i]
    next_value = 0 if is_terminals[i] else values[i+1]
    delta = rewards[i] + gamma * next_value * mask - values[i]

    # GAE 公式：优势 = 当前 TD 误差 + 后续优势的加权累加
    gae = delta + gamma * lam * mask * gae
    advantages.insert(0, gae)
  
  return torch.tensor(advantages, dtype=torch.float32)


