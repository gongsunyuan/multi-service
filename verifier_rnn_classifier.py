import os
import sys
import networkx as nx
import numpy as np
import time
from mininet.net import Mininet
from mininet.log import setLogLevel, info

# --- 导入你的环境文件 ---
from MS.Env.NetworkGenerator import TopologyGenerator
from MS.Env.FlowGenerator import FlowGenerator, FlowType
from MS.Env.MininetController import get_flow_fingerprint, measure_path_qos, get_a_mininet

# --- 导入 PyTorch 和你的 RNN 分类器 ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau 
from MS.LSTM.Pretrain.TmpClassifier import RNNClassifier 
from tqdm import tqdm # 用于显示进度条


# -----------------------------------------------
# 核心训练逻辑
# -----------------------------------------------
def run_live_verifying():
  """主运行函数"""
  
  # 验证参数
  VERIFIER_SAMPLE = 2000
  LOAD_MODEL_PATH = "./MS/LSTM/Classify-model.pth"
  
  # --- 模型参数 ---
  INPUT_DIM = 2            # 2个参数 (delay, IAT)
  RNN_LAYERS = 2           # RNN 层数
  NUM_CLASSES = 3          # 3个类别 (VOIP, STREAMING, INTERACTIVE)
  HIDDEN_DIM = 256         # 
  
  # --- 初始化 PyTorch 组件 ---
  info(f"*** 正在初始化模型 (HIDDEN_DIM={HIDDEN_DIM}, NUM_ClASSES={NUM_CLASSES})\n")
  model = RNNClassifier(
    input_dim=INPUT_DIM, 
    rnn_hidden_dim=HIDDEN_DIM, 
    num_classes=NUM_CLASSES,
    rnn_layers=RNN_LAYERS
    )
  
  criterion = nn.CrossEntropyLoss()
  try:
    model.load_state_dict(LOAD_MODEL_PATH)
  except Exception as e:
    print(f"模型加载失败：\n {e}")

  # 将模型置于评估模式
  model.eval()

  try:
    # --- 步骤 1: 生成拓扑 ---
    print("==== 生成拓扑 ====")
    topo_gen = TopologyGenerator(num_nodes_range=(5, 8))
    g = topo_gen.generate_topology()
    print(f"拓扑生成完毕: {len(g.nodes())} 个节点, {len(g.edges())} 条边。")

    # --- 步骤 2: 启动 Mininet ---
    print("\n==== 启动 Mininet ====")

    with get_a_mininet(g) as net:
      # --- 步骤 2.B: 测试 Mininet 连通性 (pingAll) ---
      print("\n==== 测试 Mininet 连通性 (pingAll)====")
      net.pingAll() # 不需要存储返回值
      net.pingAll()
      # --- 步骤 3: 实例化生成器 ---
      flow_gen = FlowGenerator()

      info(f"===== 开始在线评估 (共 {VERIFIER_SAMPLE} 次 ======\n")
      
      pbar = tqdm(range(VERIFIER_SAMPLE))
      best_acc = 0.0
      correct_predictions = 0

      for i in pbar: 
        # 生成一个新样本 (数据+标签)
        flow_type, flow_profile = flow_gen.get_random_flow()
        label = flow_type.value - 1 
        label_tensor = torch.tensor([label], dtype=torch.long)

        # B. 获取 Mininet 中的主机
        s_id, d_id = topo_gen.select_source_destination()
        S_host = net.get(f'h{s_id}')
        D_host = net.get(f'h{d_id}')
        
        # C. 在 Mininet 中"实时"生成指纹 (数据)
        fingerprint_matrix = get_flow_fingerprint(S_host, D_host, flow_profile) 
        
        if fingerprint_matrix is None:
          pbar.write(f"警告: 第 {i+1} 步指纹生成失败，跳过。\n")
          continue
      
        # D. 准备 PyTorch 张量
        # --------------------
        input_tensor = torch.from_numpy(fingerprint_matrix).unsqueeze(0).float() 
        logits = model(input_tensor)            # 前向传播

        # F. 记录统计数据
        # -----------------
        # 关键 2: 记录时，将归一化后的 loss 乘回 ACCUMULATION_STEPS 以恢复原始损失值

        predicted_index = torch.argmax(logits, dim=1).item()
        if predicted_index == label:
          correct_predictions += 1

      info(f"[-] 评估完成，最终准确率{correct_predictions/VERIFIER_SAMPLE*100:.2f}%\n")

  except Exception as e:
    info(f"\n--- 仿真出错 ---")
    info(f"{e}\n")
    import traceback
    traceback.print_exc()

  finally:
    os.system('sudo mn -c')


if __name__ == '__main__':
  setLogLevel('info')
  
  if os.getuid() != 0:
    print("错误：此脚本必须以 root 权限 (sudo) 运行。")
  else:
    run_live_verifying()