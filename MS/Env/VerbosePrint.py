from datetime import datetime
import os

MININET_VERBOSE = False       # 总开关
LOG_TO_CONSOLE  = True        # 是否打印到终端
LOG_FILE_PATH   = None        # 日志文件路径 (例如: "./train-log/mininet_debug.log"), 如果设为 None，则不写入文件
CURRENT_PBAR    = None

def vprint(message, **kwargs):
  """
  Conditionally logs a message. 
  If a tqdm pbar is registered, uses pbar.write() to avoid breaking the bar.
  Otherwise, uses standard print().
  """
  if MININET_VERBOSE:
    # 1. 准备消息
    now = datetime.now()
    time_str = now.strftime("%H:%M:%S.%f")[:-3]
    full_msg = f"[{time_str}] {message}"

    # 2. 输出到终端 (Console)
    if LOG_TO_CONSOLE:
      if CURRENT_PBAR is not None:
        # [核心逻辑] 如果存在进度条，使用 write 方法
        # 注意: pbar.write 会自动换行，不需要 end='\n'
        CURRENT_PBAR.write(full_msg)
      else:
        # 普通模式
        print(full_msg, **kwargs)

    # 3. 输出到文件 (File) - 逻辑保持不变
    if LOG_FILE_PATH:
      try:
        log_dir = os.path.dirname(LOG_FILE_PATH)
        if log_dir and not os.path.exists(log_dir):
          os.makedirs(log_dir, exist_ok=True)
        
        with open(LOG_FILE_PATH, "a", encoding='utf-8') as f:
          f.write(full_msg + "\n")
      except Exception as e:
        # 这种情况下只能用 print 报错了，别无他法
        print(f"[{time_str}] [vprint ERROR] Failed to write log: {e}")