from datetime import datetime
import os

MININET_VERBOSE = True        # 总开关
LOG_TO_CONSOLE  = True        # 是否打印到终端
LOG_FILE_PATH   = None        # 日志文件路径
CURRENT_PBAR    = None        # tqdm 进度条对象
TAG_WIDTH       = 10          # [配置] TAG 的固定显示宽度，超过截断，不足补空格

def vprint(message, tag=None, timestamp=None, **kwargs):
  """
  核心日志函数。
  支持：终端输出、文件记录、时间戳锁定、TAG对齐。
  
  Args:
    message (str): 消息内容
    tag (str, optional): 标签，如 "Agent"。
    timestamp (datetime, optional): 传入特定的时间对象。
                                    如果不传，则使用当前时间。
                                    用于多行日志保持时间一致。
    **kwargs: 传给 print 的参数
  """
  if not MININET_VERBOSE:
    return

  # 1. 处理时间戳 (关键修改：支持传入 timestamp 以实现 Time Locking)
  if timestamp is None:
    now = datetime.now()
  else:
    now = timestamp
  
  # 格式化时间字符串
  time_str = now.strftime("%H:%M:%S.%f")[:-3]

  # 2. 处理 TAG 前缀
  if tag:
    # 截断并居中对齐
    safe_tag = str(tag)[:TAG_WIDTH]
    # 格式: "[   TAG    ] " (注意末尾有一个空格)
    tag_prefix = f"[{safe_tag:^{TAG_WIDTH}}] "
  else:
    tag_prefix = ""

  # 3. 拼接完整消息
  # 最终格式: [12:00:01.123] [  Agent   ] Message...
  full_msg = f"[{time_str}] {tag_prefix}{message}"

  # 4. 输出到终端 (Console)
  if LOG_TO_CONSOLE:
    if CURRENT_PBAR is not None:
      CURRENT_PBAR.write(full_msg)
    else:
      print(full_msg, **kwargs)

  # 5. 输出到文件 (File)
  if LOG_FILE_PATH:
    try:
      log_dir = os.path.dirname(LOG_FILE_PATH)
      # 仅当路径包含目录时才检查创建
      if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
      
      with open(LOG_FILE_PATH, "a", encoding='utf-8') as f:
        f.write(full_msg + "\n")
    except Exception as e:
      # 兜底防止 IO 错误崩溃
      print(f"[{time_str}] [LogErr] Write failed: {e}")

def vprint_qos(service_type, delay, jitter, bw, loss, tag="Measure"):
  """
  QoS 专用打印函数 (Wrapper)。
  完全复用 vprint 的文件写入和终端打印逻辑，确保格式一致。
  """
  if not MININET_VERBOSE:
    return

  # 1. 锁定时间戳 (Time Locking)
  # 确保 Header 和下面 4 个指标使用完全相同的毫秒时间
  locked_now = datetime.now()

  # 2. 打印 Header
  # 这一行带 TAG，例如 [ Measure  ]
  header_msg = f"Report for {service_type}:"
  vprint(header_msg, tag=tag, timestamp=locked_now)

  # 3. 计算对齐缩进 (Indent Calculation)
  # 我们需要模拟 TAG 的宽度，使指标行的 ">" 对齐到 Header 的文字下方
  # 逻辑: TAG占据 (1 + TAG_WIDTH + 1) 个字符，外加 tag_prefix 末尾的一个空格
  # Total chars in tag_prefix = '[' + TAG_WIDTH + ']' + ' '
  indent_len = 1 + TAG_WIDTH
  indent_str = " " * indent_len

  # 4. 打印指标 (Metrics)
  # 注意：这里 tag=None，因为我们把缩进做在了 message 里
  # 这样 vprint 会打印: [Time] + "" + indent_str + "> Delay..."
  
  # 辅助 lambda 简化代码
  def _p(label, val, unit):
    # :>8.2f 确保数值右对齐
    msg = f"{indent_str}> {label:<10} {val:>8.2f} {unit}"
    vprint(msg, tag=None, timestamp=locked_now)

  _p("Delay:",     delay,  "ms")
  _p("Jitter:",    jitter, "ms")
  _p("Bandwidth:", bw,     "Mbps")
  _p("Loss Rate:", loss,   "%")
# ==========================================
# 测试用例 (你可以直接运行这个文件查看效果)
# ==========================================
if __name__ == "__main__":
  import time
  from tqdm import tqdm

  print("--- 普通测试 ---")
  vprint("System starting...", tag="System")
  vprint("Loading graph...", tag="Topo")
  vprint("Network unstable!", tag="Monitor")
  vprint("This is a message without tag.")
  vprint("Tag is too long will be truncated", tag="VeryLongTagIndeed")
  vprint_qos("STREAMING", 150.23, 12.05, 89.50, 0.50)

  print("\n--- Tqdm 集成测试 ---")
  pbar = tqdm(total=5)
  CURRENT_PBAR = pbar
  
  for i in range(5):
    time.sleep(0.2)
    vprint(f"Processing step {i+1}...", tag="Train")
    pbar.update(1)
  CURRENT_PBAR = None
  print("\nDone.")