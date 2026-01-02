import os
import sys

import torch
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

from src.agents.ppo_agent import FiLMPPOAgent
from src.trainers import WarmupTrainer
from src.utils import (
    AttrDict, logger, SdnParaser, load_yaml_config, WarmupDataset
)

def train_stage1(config: AttrDict) -> None:
    """
    训练 Stage 1：MM1 预训练
    目标：训练一个能理解 "这条路很堵" 和 "那条路通向终点" 的 GNN 模型

    params:
        config (AttrDict): 配置字典
            attributes:
                train.epochs (int): Warmup 预训练的 Epoch 数
                train.samples_per_epoch (int): 每个 Epoch 生成的样本数
                train.patience (int): 早停耐心值
                train.min_delta (float): 早停最小改进值
    returns:
        None
    """
    # 1. 初始化环境与设备
    device = config.device
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
    
    # 5. 训练循环配置
    best_val_loss = float('inf')
    patience = getattr(config.train, 'patience', 10)
    min_delta = getattr(config.train, 'min_delta', 0.001)
    epochs = getattr(config.train, 'epochs', 1000)
    counter = 0
    
    # 创建验证数据集
    val_dataset = WarmupDataset(config, max_samples=int(config.train.samples_per_epoch * 0.2))

    for epoch in range(epochs):
        # 训练一个epoch
        train_loss = trainer.train_epoch(dataset, epoch)
        
        # 在验证集上评估
        val_loss = trainer.validate_epoch(val_dataset, epoch)
        
        logger.log(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}", tag=f"Epoch {epoch+1}/{epochs}")
        
        # 保存最佳权重（使用验证损失）
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            trainer.checkpoint_manager.save(model = agent, epoch = epoch, save_file = "warmup_best.pt", metrics={'train_loss': train_loss, 'val_loss': val_loss})
            logger.log(f"Saved Best Model at Epoch {epoch+1}")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                logger.log(f"Early stopping triggered after {epoch+1} epochs. Best val loss: {best_val_loss:.4f}")
                break
            
    # 保存最终权重
    trainer.checkpoint_manager.save(model = agent, save_file = "warmup_final.pt", metrics={'train_loss': train_loss, 'val_loss': val_loss})
    logger.log("Stage 1 Warmup Completed.")
    writer.close()

if __name__ == "__main__":
    parser = SdnParaser()
    args = parser.parse_args()
    config_path = os.path.join("configs/", args.yaml)
    config = load_yaml_config(config_path)
    config['device'] = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    train_stage1(config)
