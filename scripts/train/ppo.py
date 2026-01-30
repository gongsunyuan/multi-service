import sys
import os
from omegaconf import DictConfig, OmegaConf
import torch
from pathlib import Path
from mininet.log import setLogLevel
sys.path.append(str(Path(__file__).resolve().parent.parent))

from loguru import logger
from multi_service.agents.ablation_agent import AblationAgent
from multi_service.utils import (
      SdnParaser, PPOMemory, BankTrafficManager, load_yaml_config)

from multi_service.env.sdn_wrapper import SdnWrapper
from multi_service.agents.ppo_agent import FiLMPPOAgent
from multi_service.trainers.ppo_trainer import PPOTrainer

def train_ppo() -> None:
    # 1. 加载配置文件
    parser = SdnParaser()
    args = parser.parse_args()
    
    # 验证配置文件路径
    if not args.yaml or not args.yaml.endswith('.yaml'):
        raise ValueError(f"Invalid config file: {args.yaml}")
    
    config_path = os.path.join("configs/", args.yaml)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = DictConfig(OmegaConf.load(config_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 强制更新或添加
    OmegaConf.update(config, "device", device, force_add=True)

    # 确保日志目录存在
    os.makedirs(config.path.log_dir, exist_ok=True)

    # 2. 环境初始化 (启动 Mininet)
    log_path = os.path.join(config.path.log_dir, "debug.log")
    debug_mode = OmegaConf.select(config, "debug_mode", default=False)
    logger.configure(handlers=[
        {"sink": log_path, "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", "level": "DEBUG" if debug_mode else "INFO"},
        {"sink": sys.stdout, "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", "level": "DEBUG" if debug_mode else "INFO"}
    ])
    
    logger.info(f"Booting Mininet Environment (Debug Mode: {debug_mode})...")
    
    try:
        setLogLevel('critical')
        env = SdnWrapper(config)
    except Exception as e:
        logger.error(f"Failed to initialize Mininet environment: {e}")
        raise
    
    # 3. 初始化 Agent 与 Memory
    agent = AblationAgent(config)
    memory = PPOMemory(config.device)
    
    # 4. 训练流程控制
    trainer = PPOTrainer(agent, env, memory, config)
    
    # 5. 模型加载策略：优先加载PPO checkpoint，否则加载warmup模型
    if args.checkpoint:
        logger.info(f"Loading PPO checkpoint from: {args.checkpoint}")
        try:
            checkpoint_info = trainer.checkpoint_manager.load(
                model=agent,
                checkpoint_path=args.checkpoint,
                device=config.device
            )
            if checkpoint_info:
                logger.info(f"PPO checkpoint loaded successfully!", tag="ckpt load")
                if 'metrics' in checkpoint_info:
                    metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in checkpoint_info['metrics'].items()])
                    logger.info(f"Warmup model metrics: {metrics_str}", tag="Checkpoint")
                if 'start_epoch' in checkpoint_info:
                    trainer.start_epoch = checkpoint_info['start_epoch']
                    logger.info(f"Start from epoch {trainer.start_epoch}", tag="Start epoch")
        except Exception as e:
            logger.info(f"Failed to load PPO checkpoint: {e}", tag="Checkpoint")
            # PPO checkpoint加载失败，尝试加载warmup模型
            if hasattr(config.path, 'warmup_path') and config.path.warmup_path:
                logger.info(f"Attempting to load warmup pre-trained model from: {config.path.warmup_path}", tag="Warmup")
                warmup_info = trainer.checkpoint_manager.load(
                    model=agent,
                    checkpoint_path=config.path.warmup_path,
                    device=config.device
                )
                if warmup_info:
                    logger.info(f"Warmup model loaded successfully! Epoch: {warmup_info.get('start_epoch', 0)-1}", tag="Warmup")
                    if 'metrics' in warmup_info:
                        metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in warmup_info['metrics'].items()])
                        logger.info(f"Warmup model metrics: {metrics_str}", tag="Warmup")
                else:
                    logger.info(f"Failed to load warmup model from: {config.path.warmup_path}", tag="Warmup Warn")
    else:
        # 未提供PPO checkpoint，尝试加载warmup模型
        if hasattr(config.path, 'warmup_path') and config.path.warmup_path:
            logger.info(f"Loading warmup pre-trained model from: {config.path.warmup_path}",)
            warmup_info = trainer.checkpoint_manager.load(
                model=agent,
                checkpoint_path=config.path.warmup_path,
                device=config.device
            )
            if warmup_info:
                logger.info(f"Warmup model loaded successfully! Epoch: {warmup_info.get('start_epoch', 0)-1}")
                if 'metrics' in warmup_info:
                    metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in warmup_info['metrics'].items()])
                    logger.info(f"Warmup model metrics: {metrics_str}")
            else:
                logger.info(f"Failed to load warmup model from: {config.path.warmup_path}")
        
    traffic_gen = BankTrafficManager(config=config, bank_path=config.path.fgprt_path)
    try:
        logger.info("Starting Formal RL Training...")
        trainer.run(traffic_gen)
    finally:
        env.close() # 必须清理 Mininet 资源

if __name__ == "__main__":
	train_ppo()
