import torch
import random
from .verbose_logger import logger

class BankTrafficManager:
  """
  使用预存指纹库的流量管理器
  """
  def __init__(self, env, bank_path):
    self.env = env
    # [cite_start]加载指纹库 [cite: 1]
    raw_bank = torch.load(bank_path, map_location='cpu')
    
    # 预处理：将所有 Tensor 统一为 (1, 30, 2) 维度
    self.bank = {}
    for k, v_list in raw_bank.items():
      processed_list = []
      for t in v_list:
        if t.dim() == 2:
          t = t.unsqueeze(0)
        processed_list.append(t)
      self.bank[k.lower()] = processed_list
      
    logger.log(f"Fingerprint Bank loaded and pre-processed from {bank_path}", tag="System", log_to_console=True)

  def generate_batch(self, batch_size):
    """
    从库中随机采样指纹，并确保设备与 Agent 一致
    """
    flows = []

    for _ in range(batch_size):
      # 1. 随机选择源宿节点 
      s, d = self.env.topo_gen.select_source_destination()
      
      # 2. 随机选择业务类型
      f_type, _ = self.env.flow_gen.get_random_flow()
      
      # 3. 从预处理过的库中采样 
      type_key = f_type.name.lower()
      available_fingerprints = self.bank.get(type_key, [])
      
      if not available_fingerprints:
        logger.log(f"No fingerprint data for type: {f_type.name}!", tag="Flow Type Err", log_to_console=True)
        # 兜底：生成全 0 的张量
        fingerprint = torch.zeros((1, 30, 2))
      else:
        # 随机抽取并移动到目标设备
        fingerprint = random.choice(available_fingerprints)

      # 构建 Flow 对象
      flow_obj = type('Flow', (), {
        'src': s,
        'dst': d,
        'flow_type': f_type,
        'fingerprint': fingerprint
      })
      flows.append(flow_obj)
      
    return flows