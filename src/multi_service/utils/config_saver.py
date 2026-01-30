import os
from omegaconf import DictConfig
import yaml

from loguru import logger

def save_configs(config: DictConfig):
    # 1. 保存完整的训练配置 (Train Config)
    train_cfg_path = os.path.join(config.path.config_dir, "train_eval.yaml")
    eval_cfg_path = os.path.join(config.path.config_dir, "eval_config.yaml")

    with open(train_cfg_path, 'w', encoding='utf-8') as f:
        # 将 AttrDict 转回 dict 进行 dump 
        yaml.dump(dict(config), f, indent=2, sort_keys=False, allow_unicode=True)
    
    # 2. 构造精简的评估配置 (Eval Config)
    # 只挑选评估需要的核心参数，例如模型架构和环境归一化参数 

    eval_dict = {
        "model": dict(config.model),
        "env": dict(config.env) if hasattr(config, "env") else None,
        "path": {
        "eval_output" : config.path.eval_dir,
        "ckpt_dir": config.path.ckpt_dir
        }
    }
    
    with open(eval_cfg_path, 'w', encoding='utf-8') as f:
        yaml.dump(eval_dict, f, indent=2, sort_keys=False, allow_unicode=True)
    
    logger.info(f"Configs saved to {config.path.config_dir}")
