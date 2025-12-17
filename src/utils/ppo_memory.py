import torch
from torch_geometric.data import Batch # 必须导入这个用于处理图批处理

class PPOMemory:
  def __init__(self, device):
    self.device = device
    self.states = []
    self.values = []
    self.actions = []
    self.rewards = []
    self.log_probs = []
    self.is_terminals = []
    # 为了 evaluate_batch 增加的辅助数据
    self.fingerprints = []  
    self.curr_indices = []  
    self.target_indices = [] 

  def store(self, state, action, log_prob, value, reward, is_terminal,
            fingerprint, curr_idx, target_idx):
    self.states.append(state)
    self.values.append(value)
    self.rewards.append(reward)
    self.actions.append(action)
    self.log_probs.append(log_prob)
    self.curr_indices.append(curr_idx)
    self.is_terminals.append(is_terminal)
    self.fingerprints.append(fingerprint)
    self.target_indices.append(target_idx)

  def get_all(self):
    """
    核心转化函数：将 List 转化为符合训练要求的 Tensor/Batch
    """
    # 1. 图状态特殊处理：将 List[Data] 转化为一个大图 Batch 
    # 注意：不能用 torch.stack，因为图的节点数可能不同
    states_batch = Batch.from_data_list(self.states).to(self.device)
    
    # 2. 基础数据转化
    log_probs = torch.stack(self.log_probs).to(self.device).detach() 
    actions = torch.tensor(self.actions, dtype=torch.long).to(self.device) 
    values = torch.tensor(self.values, dtype=torch.float32).to(self.device) 
    rewards = torch.tensor(self.rewards, dtype=torch.float32).to(self.device) 
    is_terminals = torch.tensor(self.is_terminals, dtype=torch.float32).to(self.device)
    
    # 3. 辅助训练数据转化  
    fingerprints = torch.stack(self.fingerprints).to(self.device).detach()
    curr_indices = torch.tensor(self.curr_indices, dtype=torch.long).to(self.device)
    target_indices = torch.tensor(self.target_indices, dtype=torch.long).to(self.device)

    return (
      states_batch, actions, log_probs, values, rewards, 
      is_terminals, fingerprints, curr_indices, target_indices)

  def clear(self):
    del self.states[:]
    del self.values[:]
    del self.actions[:]
    del self.rewards[:]
    del self.log_probs[:]
    del self.is_terminals[:]
    del self.fingerprints[:]
    del self.curr_indices[:]
    del self.target_indices[:]

  