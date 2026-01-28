import sys
import os
import torch
from pathlib import Path
from mininet.log import setLogLevel
sys.path.append(str(Path(__file__).resolve().parent.parent))
    
from multi_service.agents.ablation_agent import AblationAgent
from multi_service.utils import (
      SdnParaser, logger, PPOMemory, BankTrafficManager, load_yaml_config)

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
    
    config = load_yaml_config(config_path)
    config['device'] = "cuda" if torch.cuda.is_available() else "cpu"

    # 确保日志目录存在
    os.makedirs(config.path.log_dir, exist_ok=True)

    # 2. 环境初始化 (启动 Mininet)
    log_path = os.path.join(config.path.log_dir, "debug.log")
    debug_mode = config.get("debug_mode", False)
    logger.configure(log_file=log_path, log_to_console=False, debug_mode=debug_mode) # Enable console output by default, controlled by debug_mode
    logger.log(f"Booting Mininet Environment (Debug Mode: {debug_mode})...", log_to_console=True, tag="Init")
    
    try:
        setLogLevel('critical')
        env = SdnWrapper(config)
    except Exception as e:
        logger.log(f"Failed to initialize Mininet environment: {e}", log_to_console=True, tag="Init Error")
        raise
    
    # 3. 初始化 Agent 与 Memory
    agent = AblationAgent(config)
    memory = PPOMemory(config.device)
    
    # 4. 训练流程控制
    trainer = PPOTrainer(agent, env, memory, config)
    
    # 5. 模型加载策略：优先加载PPO checkpoint，否则加载warmup模型
    if args.checkpoint:
        logger.log(f"Loading PPO checkpoint from: {args.checkpoint}", tag="Checkpoint", log_to_console=True)
        try:
            checkpoint_info = trainer.checkpoint_manager.load(
                model=agent,
                checkpoint_path=args.checkpoint,
                device=config.device
            )
            if checkpoint_info:
                logger.log(f"PPO checkpoint loaded successfully!", tag="ckpt load")
                if 'metrics' in checkpoint_info:
                    metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in checkpoint_info['metrics'].items()])
                    logger.log(f"Warmup model metrics: {metrics_str}", tag="Checkpoint")
                if 'start_epoch' in checkpoint_info:
                    trainer.start_epoch = checkpoint_info['start_epoch']
                    logger.log(f"Start from epoch {trainer.start_epoch}", tag="Start epoch")
        except Exception as e:
            logger.log(f"Failed to load PPO checkpoint: {e}", tag="Checkpoint")
            # PPO checkpoint加载失败，尝试加载warmup模型
            if hasattr(config.path, 'warmup_path') and config.path.warmup_path:
                logger.log(f"Attempting to load warmup pre-trained model from: {config.path.warmup_path}", tag="Warmup")
                warmup_info = trainer.checkpoint_manager.load(
                    model=agent,
                    checkpoint_path=config.path.warmup_path,
                    device=config.device
                )
                if warmup_info:
                    logger.log(f"Warmup model loaded successfully! Epoch: {warmup_info.get('start_epoch', 0)-1}", tag="Warmup")
                    if 'metrics' in warmup_info:
                        metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in warmup_info['metrics'].items()])
                        logger.log(f"Warmup model metrics: {metrics_str}", tag="Warmup")
                else:
                    logger.log(f"Failed to load warmup model from: {config.path.warmup_path}", tag="Warmup Warn")
    else:
        # 未提供PPO checkpoint，尝试加载warmup模型
        if hasattr(config.path, 'warmup_path') and config.path.warmup_path:
            logger.log(f"Loading warmup pre-trained model from: {config.path.warmup_path}", tag="Warmup", log_to_console=True)
            warmup_info = trainer.checkpoint_manager.load(
                model=agent,
                checkpoint_path=config.path.warmup_path,
                device=config.device
            )
            if warmup_info:
                logger.log(f"Warmup model loaded successfully! Epoch: {warmup_info.get('start_epoch', 0)-1}", log_to_console=True, tag="Warmup")
                if 'metrics' in warmup_info:
                    metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in warmup_info['metrics'].items()])
                    logger.log(f"Warmup model metrics: {metrics_str}", log_to_console=True, tag="Warmup")
            else:
                logger.log(f"Failed to load warmup model from: {config.path.warmup_path}", log_to_console=True, tag="Warmup Warn")
        
    traffic_gen = BankTrafficManager(config=config, bank_path=config.path.fgprt_path)
    try:
        logger.log("Starting Formal RL Training...", log_to_console=True, tag="Init")
        trainer.run(traffic_gen)
    finally:
        env.close() # 必须清理 Mininet 资源

if __name__ == "__main__":
	train_ppo()
