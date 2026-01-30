import yaml
import os
import shutil
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

    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Configuration has no attribute '{key}'")

def load_yaml_config(yaml_path: str) -> AttrDict:
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
            if hasattr(config, 'path'):
                if hasattr(config.path, 'output_dir'):
                    debug_mode = config.get("debug_mode", False)
                    if debug_mode:
                        # In debug mode, redirect everything to /tmp to avoid cluttering the workspace
                        base_tmp = f"workspace/tmp/{config.name.exp_name}_debug"
                        config.path.output_dir = base_tmp
                        config.path.log_dir = os.path.join(base_tmp, "logs/")
                        config.path.ckpt_dir = os.path.join(base_tmp, "checkpoints/")
                        config.path.eval_dir = os.path.join(base_tmp, "eval_results/")
                        config.path.config_dir = os.path.join(base_tmp, "configs/")
                        
                        if os.path.exists(base_tmp):
                            shutil.rmtree(base_tmp)

                        os.makedirs(config.path.log_dir, exist_ok=True)
                        os.makedirs(config.path.ckpt_dir, exist_ok=True)
                        os.makedirs(config.path.eval_dir, exist_ok=True)
                        os.makedirs(config.path.config_dir, exist_ok=True)
                        try:
                            os.chmod(base_tmp, 0o777)
                        except OSError:
                            pass

                    elif not "eval" in config.name.exp_name:
                        config.path.output_dir = create_unique_log_dir(config.path.output_dir, experiment_name=config.name.exp_name)
                        config.path.log_dir = os.path.join(config.path.output_dir, "logs/")
                        config.path.ckpt_dir = os.path.join(config.path.output_dir, "checkpoints/")
                        config.path.eval_dir = os.path.join(config.path.output_dir, "eval_results/")
                        config.path.config_dir = os.path.join(config.path.output_dir, "configs/")

                        os.chmod(config.path.output_dir, mode=0o777)
                        if hasattr(config.path, 'log_dir'):
                            os.makedirs(config.path.log_dir , exist_ok=True)
                            os.chmod(config.path.log_dir, mode=0o777)
                        if hasattr(config.path, 'ckpt_dir'):
                            os.makedirs(config.path.ckpt_dir, exist_ok=True)
                            os.chmod(config.path.ckpt_dir, mode=0o777)
                        if hasattr(config.path, 'eval_dir'):
                            os.makedirs(config.path.eval_dir, exist_ok=True)
                            os.chmod(config.path.eval_dir, mode=0o777)
                        if hasattr(config.path, 'config_dir'):
                            os.makedirs(config.path.config_dir, exist_ok=True)
                            os.chmod(config.path.config_dir, mode=0o777)

            # 延迟导入 save_configs 以避免循环导入
            from .config_saver import save_configs
            return config
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            return None  # pyright: ignore[reportReturnType]
    