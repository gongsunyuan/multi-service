import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 模拟读取你的日志内容 (实际使用时请读取文件)
log_content = open("mm1_train_log", "r").read() 
# 为了演示，这里假设 log_text 包含了上传文件的内容

def parse_training_log(log_text):
  data = {
    "updates": [],
    "reward": [],
    "phases": [] # 存储调度器介入的时间点
  }
  
  # 正则表达式匹配
  # 匹配: [train] avg reward: 0.77 | Updates: 1
  reward_pattern = re.compile(r"avg reward:\s+([\d\.]+)\s+\|\s+Updates:\s+(\d+)")
  
  # 匹配: [Scheduler] PHASE 2: ...
  phase_pattern = re.compile(r"\[Scheduler\]\s+(PHASE\s+\d+)")
  
  current_update = 0
  
  lines = log_text.strip().split('\n')
  for line in lines:
    # 提取 Reward 和 Updates
    match_reward = reward_pattern.search(line)
    if match_reward:
      reward = float(match_reward.group(1))
      update = int(match_reward.group(2))
      data["updates"].append(update)
      data["reward"].append(reward)
      current_update = update
          
      # 提取 Phase 变化
      match_phase = phase_pattern.search(line)
      if match_phase:
        phase_name = match_phase.group(1)
        # 记录当前 Phase 发生的 Update 节点
        # 防止重复记录 (日志中可能连续打印多条 Phase 信息)
        if not data["phases"] or data["phases"][-1][0] != current_update:
          data["phases"].append((current_update, phase_name))
              
  return pd.DataFrame({"updates": data["updates"], "reward": data["reward"]}), data["phases"]

def plot_training_curve(file_path, save_filename="training_reward_curve.png", dpi=300):
  try:
    with open(file_path, 'r', encoding='utf-8') as f:
      content = f.read()
  except FileNotFoundError:
    print("请确保日志文件存在")
    return

  df, phases = parse_training_log(content)
  
  print(df)
  print(phases)
  if df.empty:
    print("未提取到有效数据，请检查日志格式。")
    return

  # 设置学术绘图风格
  plt.style.use('seaborn-v0_8-paper')
  fig, ax = plt.subplots(figsize=(10, 6))

  # 1. 绘制原始数据 (半透明背景)
  ax.plot(df["updates"], df["reward"], color='lightgray', alpha=0.6, label='Raw Reward', linewidth=1)

  # 2. 绘制平滑曲线 (移动平均，窗口大小=20)
  window_size = 20
  df["smooth"] = df["reward"].rolling(window=window_size, min_periods=1).mean()
  ax.plot(df["updates"], df["smooth"], color='#1f77b4', linewidth=2, label=f'Smoothed (MA={window_size})')

  # 3. 标注调度器 Phase 变化 (垂直线)
  colors = ['#d62728', '#2ca02c'] # 红、绿
  for i, (update_point, phase_name) in enumerate(phases):
    color = colors[i % len(colors)]
    plt.axvline(x=update_point, color=color, linestyle='--', alpha=0.8)
    plt.text(update_point + 10, 0.55, phase_name, color=color, rotation=90, fontweight='bold')

  # 4. 图表装饰
  ax.set_title("FiLM-GNN Agent Training Convergence (Theoretical Phase)", fontsize=14, pad=15)
  ax.set_xlabel("Training Updates", fontsize=12)
  ax.set_ylabel("Average Reward", fontsize=12)
  ax.set_ylim(0.4, 1.0) # 根据日志数据范围调整 [cite: 4, 628]
  ax.grid(True, linestyle=':', alpha=0.6)
  ax.legend(loc='lower right')

  plt.tight_layout()
  save_filename = "mm1_train_curve"
  print(f"Saving figure to {save_filename} with DPI={dpi}...")
  plt.savefig(save_filename, dpi=dpi, bbox_inches='tight')
  plt.show()
  plt.close(fig) # 关闭图表对象，释放内存
  print("Save complete.")


plot_training_curve('mm1_train_log')