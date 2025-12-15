import torch
import numpy as np
import os # 用于检查文件是否存在
from pathlib import Path

def append_matrix_to_file(tensor: torch.Tensor, filename: str, flow_id: int):
  """
  将 [1, N, M] 形状的 Tensor 写入文件，并保持 N x M 的矩阵结构。
  
  参数:
      tensor (torch.Tensor): 归一化后的特征张量。
      filename (str): 要追加写入的文件名。
      flow_id (int): 当前流的序号。
  """
  
  # 1. 转换为 NumPy 数组并移除 Batch 维度
  # 形状: (N, M) -> 例如 (30, 3)
  data_np = tensor.squeeze(0).numpy()
  Path(filename).parent.mkdir(parents=True, exist_ok=True)
  # 2. 以追加模式打开文件
  try:
    # 使用 'a' 模式 (append)
    with open(filename, 'a') as f:
        
      # [关键] 写入分隔符和头部，清晰地标记矩阵的开始
      f.write(f"\n====================================================================\n")
      f.write(f"Flow ID: {flow_id} | Shape: {data_np.shape}\n")
      f.write("Features: [Size/1600, IAT/0.1]\n")
      f.write(f"--------------------------------------------------------------------\n")
      
      # 3. 使用 np.savetxt 将数据写入到 Python 文件句柄 (f) 中
      # fmt='%.6f' 保证了浮点数有 6 位精度
      # delimiter=',' 使得它依然是 CSV 格式，但结构清晰
      np.savetxt(f, data_np, fmt='%.6f', delimiter=', ')
        
  except IOError as e:
    print(f"写入文件失败: {e}")