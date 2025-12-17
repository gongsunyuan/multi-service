import torch
import random
from verbose_logger import logger

class BankTrafficManager:
  """
  使用预存指纹库的流量管理器
  """
  def __init__(self, env, bank_path):
    self.env = env
    # 加载指纹库 [cite: 536]
    self.bank = torch.load(bank_path) 
    logger.log(f"Fingerprint Bank loaded from {bank_path}", tag="System", log_to_console=True)

  def generate_batch(self, batch_size):
    """
    从库中随机采样指纹，而不是实时捕获
    """
    flows = []
    for _ in range(batch_size):
      # 1. 随机选择源宿节点 
      s, d = self.env.topo_gen.select_source_destination()
      
      # 2. 随机选择业务类型 
      f_type, _ = self.env.flow_gen.get_random_flow()
      
      # 3. 从指纹库中采样 
      # 假设 bank 的格式是 {'voip': [t1, t2...], 'gaming': [...]}
      type_key = f_type.name.lower()
      available_fingerprints = self.bank.get(type_key, [])
      
      if not available_fingerprints:
        logger.log(f"No such flow type({f_type.name})! Check your code!", tag="Flow Type Err", log_to_console=True)
        fingerprint = torch.zeros((1, 30, 2))
      else:
        # 随机抽取一个 Tensor 
        fingerprint = random.choice(available_fingerprints)
        # 确保维度是 (1, 30, 2)
        if fingerprint.dim() == 2:
          fingerprint = fingerprint.unsqueeze(0)

      flow_obj = type('Flow', (), {
        'src': s,
        'dst': d,
        'flow_type': f_type,
        'fingerprint': fingerprint
      })
      flows.append(flow_obj)
    return flows