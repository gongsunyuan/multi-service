import torch
import os
import numpy as np
import matplotlib.pyplot as plt

# === 配置 ===
BANK_PATH = "./dataset/fingerprint_bank.pt"

# 你的归一化参数 (参考 MininetController.py)
MAX_SIZE = 1600.0  # Bytes
MAX_IAT  = 0.1     # Seconds (100ms)

def inspect_bank():
  if not os.path.exists(BANK_PATH):
    print(f"❌ 文件不存在: {BANK_PATH}")
    print("   请先运行: sudo python -m tools.build_fingerprint_bank")
    return

  print(f"🔍 正在加载指纹库: {BANK_PATH} ...")
  try:
    bank = torch.load(BANK_PATH, map_location='cpu')
  except Exception as e:
    print(f"❌ 加载失败: {e}")
    return

  print("=" * 60)
  print(f"{'Type':<12} | {'Count':<6} | {'Shape (1 Sample)':<15} | {'Avg Size (Norm)':<15} | {'Avg IAT (Norm)':<15}")
  print("-" * 60)

  # 1. 统计概览
  for flow_type, samples in bank.items():
    if len(samples) == 0:
      print(f"{flow_type:<12} | 0      | N/A             | N/A             | N/A")
      continue


    # 堆叠所有样本进行统计
    # samples 是 list of tensors, 每个 tensor shape (N, 2)
    all_data = torch.stack(samples) # (Total_Samples, N_Packets, 2)
    
    avg_size = all_data[:, :, 0].mean().item()
    avg_iat = all_data[:, :, 1].mean().item()
    
    print(f"{flow_type:<12} | {len(samples):<6} | {str(list(samples[0].shape)):<15} | {avg_size:.4f}          | {avg_iat:.4f}")

  print("=" * 60)
  print("\n📋 详细抽样检查 (还原真实物理数值):")

  # 2. 详细抽样
  for flow_type, samples in bank.items():
    if len(samples) == 0: continue
    
    # 随机抽一个
    idx = np.random.randint(len(samples))
    sample = samples[idx] # (N, 2)
    if sample.dim() == 3:
      sample = sample.squeeze(0) # 变成 (30, 2)
    print(f"\n>>> [{flow_type.upper()}] Sample #{idx}")
    
    # 还原物理数值
    # Column 0: Size
    raw_sizes = sample[:, 0] * MAX_SIZE
    # Column 1: IAT
    raw_iats = sample[:, 1] * MAX_IAT
    
    print(f"   Top 10 Packets:")
    print(f"   {'Seq':<4} | {'Size (Bytes)':<12} | {'IAT (ms)':<12}")
    print(f"   {'-'*34}")
    
    for i in range(min(10, sample.shape[0])):
      size_val = raw_sizes[i].item()
      iat_val_ms = raw_iats[i].item() * 1000.0 # s -> ms
      print(f"   {i:<4} | {size_val:<12.1f} | {iat_val_ms:<12.2f}")
      
    print(f"   ... (Total {sample.shape[0]} packets)")
    
    # 简单统计特征分析
    print(f"   [Stats] Mean Size: {raw_sizes.mean():.1f} B | Mean IAT: {raw_iats.mean()*1000:.2f} ms")
    
    # 简单的 ASCII 图形化 (Size)
    # 让你不用绘图库也能直观看到流量模式
    print(f"   [Visual] Packet Size Pattern:")
    visual_str = ""
    for s in sample[:, 0][:30]: # 只看前30个包
      if s > 0.8: visual_str += "█" # 大包
      elif s > 0.5: visual_str += "▓"
      elif s > 0.2: visual_str += "▒"
      else: visual_str += "."       # 小包
    print(f"   Seq: {visual_str}")

if __name__ == "__main__":
  inspect_bank()