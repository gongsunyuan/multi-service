import yaml
import os
from pathlib import Path
from .create_unique_log_dir import create_unique_log_dir

class AttrDict(dict):
  """
  一个简单的扩展类，允许通过 . 访问字典属性，并支持递归嵌套。
  例如：config.QOS_REWARD_PARAMS.GAMING.max_delay
  """
  def __init__(self, *args, **kwargs):
    super(AttrDict, self).__init__(*args, **kwargs)
    for key, value in self.items():
      if isinstance(value, dict):
        self[key] = AttrDict(value)

  def __getattr__(self, key):
    try:
      return self[key]
    except KeyError:
      raise AttributeError(f"Configuration has no attribute '{key}'")

def load_yaml_config(yaml_path):
  """
  调用 YAML 文件的核心函数。
  """
  if not os.path.exists(yaml_path):
    raise FileNotFoundError(f"Config file not found: {yaml_path}")

  with open(yaml_path, 'r', encoding='utf-8') as f:
    try:
      config_dict = yaml.safe_load(f)
      # 封装为对象形式返回 
      config = AttrDict(config_dict)
      config.path.log_dir = create_unique_log_dir(config.path.log_dir, experiment_name=config.name.exp_name)
      config.path.ckpt_dir = create_unique_log_dir(config.path.ckpt_dir, experiment_name=config.name.exp_name)
      os.makedirs(config.path.ckpt_dir, exist_ok=True)
      os.makedirs(config.path.log_dir , exist_ok=True)
      return config
    except yaml.YAMLError as e:
      print(f"Error parsing YAML file: {e}")
      return None
    