import sys
import os
import torch
import logging
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

from src.agents.ppo_agent import FiLMPPOAgent  # 假设你的 Agent 类名
from src.utils import (
    logger, SdnParaser, load_yaml_config, WarmupDataset
)
from src.trainers import WarmupTrainer
from utils.config_loadder import AttrDict

def train_stage1(config: AttrDict) -> None:
    """
    训练 Stage 1：MM1 预训练
    目标：训练一个能理解 "这条路很堵" 和 "那条路通向终点" 的 GNN 模型

    params:
        config (AttrDict): 配置字典
            attributes:
                train.epochs (int): Warmup 预训练的 Epoch 数
                train.samples_per_epoch (int): 每个 Epoch 生成的样本数
    returns:
        None
    """
    # 1. 初始化环境与设备
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.log(f"Starting Stage 1: Warmup Training on {device}...", tag="Pretrain")
    # 2. 初始化 Agent
    # 注意：这里我们加载未经训练的原始 Agent
    agent = FiLMPPOAgent(config).to(device)
    
    # 3. 初始化 TensorBoard
    writer = SummaryWriter(os.path.join(config.path.log_dir, "tensorboard"))
    
    # 4. 初始化数据生成器与训练器
    # max_samples 决定每个 Epoch 跑多少个合成图
    dataset = WarmupDataset(config, max_samples=config.train.samples_per_epoch)
    trainer = WarmupTrainer(agent, config, writer)
    
    # 5. 训练循环
    best_loss = float('inf')

    for epoch in range(config.train.epochs):
        loss = trainer.train_epoch(dataset, epoch)
        
        logger.log(f"Warmup Loss (Log-MSE): {loss:.4f}", tag=f"Epoch {epoch+1}/{config.train.epochs} |")
        
        # 保存最佳权重
        if loss < best_loss:
            best_loss = loss
            trainer.checkpoint_manager.save(model = agent, epoch = epoch, save_file = "warmup_best.pt", metrics={'loss': loss})
            logger.log(f"Saved Best Model at Epoch {epoch+1}")
            
    # 保存最终权重
    trainer.checkpoint_manager.save(model = agent, epoch = epoch, save_file = "warmup_final.pt", metrics={'loss': loss})
    logger.log("Stage 1 Warmup Completed.")

if __name__ == "__main__":
    parser = SdnParaser()
    args = parser.parse_args()
    config_path = os.path.join("configs/", args.yaml)
    config = load_yaml_config(config_path)
    config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    train_stage1(config)
