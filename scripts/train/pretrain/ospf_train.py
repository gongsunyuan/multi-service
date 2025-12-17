import sys
import os
import torch
import yaml
from torch_geometric.loader import DataLoader
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils import (
  logger, SdnParaser, SupervisedGraphDataset, load_yaml_config)
from src.env.network_generator import TopologyGenerator 
from src.agents.ppo_agent import FiLMPPOAgent 
from src.engine.ospf_trainer import OSPFPreTrainer
from torch.utils.tensorboard import SummaryWriter

def main():
  # 1. 解析参数与配置 
  parser = SdnParaser()
  args = parser.parse_args()
  
  yaml = os.path.join("config/", args.yaml)
  config = load_yaml_config(yaml)
  config.device = args.device if args.device else "cpu"

  # 3. 初始化日志与 TensorBoard
  logger.configure(log_file=os.path.join(config.path.log_dir, "debug.log"))
  writer = SummaryWriter(log_dir=os.path.join(config.path.log_dir, "tensorboard"))
  logger.log(f"Starting Pre-training. Run Directory: {config.path.log_dir}", tag="System")

  # 4. 准备数据与模型
  topo_gen = TopologyGenerator()
  dataset = SupervisedGraphDataset(topo_gen, config, max_samples=5000) 
  loader = DataLoader(dataset, batch_size=32)
  
  agent = FiLMPPOAgent(config)  
  agent.to(config.device)
  
  trainer = OSPFPreTrainer(agent, config, writer)

  # 5. 执行训练循环
  
  for epoch in range(config.train.train_epochs):
    al, cl, acc, recall = trainer.train_epoch(loader, epoch)
    
    # 记录 Epoch 指标
    writer.add_scalar("Pretrain/Epoch_Actor_Loss", al, epoch)
    writer.add_scalar("Pretrain/Epoch_Critic_Loss", cl, epoch)
    writer.add_scalar("Pretrain/Edge_Accuracy", acc, epoch)
    
    logger.log(f"Epoch {(epoch+1):02d} | Actor Loss: {al:.4f} | Accuracy: {acc:.2%} | Recall: {recall:.2%}", tag="Ospf Train")

    # 保存最新的两个 checkpoint
    save_path = os.path.join(config.path.ckpt_dir, f"checkpoint_{epoch}.pth")
    torch.save(agent.state_dict(), save_path)
    
    # 自动清理逻辑 (仅保留最新2个)
    import glob
    ckpt_files = sorted(glob.glob(os.path.join(config.path.ckpt_dir, "*.pth")), key=os.path.getmtime)
    if len(ckpt_files) > 2:
      for old_f in ckpt_files[:-2]:
        os.remove(old_f)

  logger.log("Pre-training completed successfully.", tag="System")
  writer.close()

if __name__ == "__main__":
  main()