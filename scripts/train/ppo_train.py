import sys
import os
import torch
import yaml
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.sdn_paraser import SdnParaser
from src.utils.verbose_logger import logger, vprint
from src.utils.ppo_memory import PPOMemory
from src.env.sdn_wrapper import SdnWrapper
from src.agents.ppo_agent import FiLMPPOAgent
from src.engine.ppo_trainer import PPOTrainer

def main():
  # 1. 加载配置
  parser = SdnParaser()
  args = parser.parse_args()
  with open(args.yaml, 'r') as f:
    config_dict = yaml.safe_load(f)
  config = type('Config', (), config_dict)
  config.device = "cuda" if torch.cuda.is_available() else "cpu"

  # 2. 环境初始化 (启动 Mininet)
  vprint("Booting Mininet Environment...", tag="System")
  env = SdnWrapper(config)
  
  # 3. 初始化 Agent 与 Memory
  agent = FiLMPPOAgent(config)
  memory = PPOMemory(config.device)
  
  # 关键：加载预训练权重进行热启动
  if args.checkpoint:
    vprint(f"Hot Start: Loading weights from {args.checkpoint}", tag="System")
    agent.load_state_dict(torch.load(args.checkpoint, map_location=config.device))
  elif os.path.exists(os.path.join(config.ckpt_dir, "pretrain_model.pth")):
    vprint("Auto-loading pretrain_model.pth...", tag="System")
    agent.load_state_dict(torch.load(os.path.join(config.ckpt_dir, "pretrain_model.pth")))

  # 4. 训练流程控制
  
  trainer = PPOTrainer(agent, env, memory, config)
  
  # 创建流量管理器 (这里可以整合你之前的 TrafficManager)
  class SimpleTrafficManager:
    def generate_batch(self, batch_size):
      # 逻辑同上一轮对话中的 TrafficManager
      pass 

  try:
    logger.log("Starting Formal RL Training...", tag="System")
    trainer.run(SimpleTrafficManager())
  finally:
    env.close() # 必须清理 Mininet 资源

if __name__ == "__main__":
  main()