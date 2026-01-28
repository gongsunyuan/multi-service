from datetime import datetime
import os

def create_unique_log_dir(base_log_path, experiment_name="Exp"):
    """
    在基础路径下创建一个带有时间戳的子文件夹
    输入: workspace/logs, "PPO_Training"
    输出: workspace/logs/20251219_143005_PPO_Training
    """
    # 1. 生成时间戳字符串 (年月日_时分秒)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 2. 拼接完整的文件夹路径 
    unique_dir_name = f"{timestamp}_{experiment_name}"
    full_path = os.path.join(base_log_path, unique_dir_name)
    
    # 3. 物理创建文件夹
    os.makedirs(full_path, exist_ok=True)
    
    return full_path