import sys
import os
import torch
import yaml
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils import (
  SdnParaser, logger, PPOMemory, BankTrafficManager, load_yaml_config)

from src.env.sdn_wrapper import SdnWrapper
from src.agents.ppo_agent import FiLMPPOAgent
from src.trainers.ppo_trainer import PPOTrainer

def main():
  # 1. 加载配置
  parser = SdnParaser()
  args = parser.parse_args()
  config_path = os.path.join("configs/", args.yaml)
  config = load_yaml_config(config_path)
  config['device'] = "cuda" if torch.cuda.is_available() else "cpu"

  # 2. 环境初始化 (启动 Mininet)
  log_path = os.path.join(config.path.log_dir, "debug.log")
  logger.configure(log_file=log_path, console=False)
  logger.log("Booting Mininet Environment...", tag="Init")
  env = SdnWrapper(config)
  
  # 3. 初始化 Agent 与 Memory
  agent = FiLMPPOAgent(config)
  memory = PPOMemory(config.device)
  
  # 加载预训练权重进行热启动
  if(args.checkpoint):
    logger.log(f"Hot Start: Loading weights from {args.checkpoint}", tag="Init")
    agent.load_state_dict(torch.load(args.checkpoint, map_location=config.device))
  
  # 4. 训练流程控制
  trainer = PPOTrainer(agent, env, memory, config)
  traffic_gen = BankTrafficManager(env=env, bank_path=config.path.fgprt_path)
  try:
    logger.log("Starting Formal RL Training...", tag="Init", log_to_console=True)
    trainer.run(traffic_gen)
  finally:
    env.close() # 必须清理 Mininet 资源

if __name__ == "__main__":
  main()