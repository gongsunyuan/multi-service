import argparse
import os
import sys

import torch
from torch.utils.tensorboard import SummaryWriter

from src.agents.ppo_agent import FiLMPPOAgent
from src.trainers import WarmupTrainer
from src.utils import (
    AttrDict, logger, SdnParaser, load_yaml_config, WarmupDataset
)

def train_stage1(config: AttrDict, args: argparse.Namespace) -> None:
    """
    训练 Stage 1：MM1 预训练
    目标：训练一个能理解 "这条路很堵" 和 "那条路通向终点" 的 GNN 模型

    params:
        args (argparse.Namespace): 命令行参数
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
    
    # 6. 尝试加载检查点，实现断点续训
    start_epoch = 0
    if args.checkpoint is not None:
        checkpoint_info = trainer.checkpoint_manager.load(
            checkpoint_path=args.checkpoint,
            model=agent,
            optimizer=trainer.optimizer,  # 恢复优化器状态
            device=device
        )
        # 如果加载成功，恢复训练状态
        if checkpoint_info:
            start_epoch = checkpoint_info.get('start_epoch', 0)
            # 恢复最佳验证损失和早停计数器
            if 'metrics' in checkpoint_info:
                loaded_val_loss = checkpoint_info['metrics'].get('val_loss', float('inf'))
                if loaded_val_loss < best_val_loss:
                    best_val_loss = loaded_val_loss
            logger.log(f"Resuming training from epoch {start_epoch}", tag="Resume")
    
    # 创建验证数据集
    val_dataset = WarmupDataset(config, max_samples=int(config.train.samples_per_epoch * 0.2))
    
    for epoch in range(start_epoch, epochs):
        # 训练一个epoch
        train_loss = trainer.train_epoch(dataset, epoch)
        
        # 在验证集上评估
        val_loss = trainer.validate_epoch(val_dataset, epoch)
        
        logger.log(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}", tag=f"Epoch {epoch+1}/{epochs}")
        
        if (epoch+1) % config.train.save_interval == 0:
            trainer.checkpoint_manager.save(model = agent, epoch = epoch, save_file = f"warmup_epoch_{epoch+1}.pt", metrics={'train_loss': train_loss, 'val_loss': val_loss})
        
        # 保存最佳权重（使用验证损失）
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            trainer.checkpoint_manager.save(model = agent, epoch = epoch, save_file = "warmup_best.pt", metrics={'train_loss': train_loss, 'val_loss': val_loss})
            logger.log(f"Saved Best Model at Epoch {epoch+1}", tag="Best", log_to_console=True)
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
    if config.path.log_dir is not None:
        logger.configure(log_file=os.path.join(config.path.log_dir, "debug.log"), log_to_console=False)
    config['device'] = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    assert config['device'] in ["cuda", "cpu"], f"Device must be 'cuda' or 'cpu' but {config['device']}"
    train_stage1(config, args)
