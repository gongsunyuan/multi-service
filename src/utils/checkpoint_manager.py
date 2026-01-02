import os
import glob
import re
import json
import subprocess
import torch
import tempfile

from .verbose_logger import logger
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from datetime import datetime

# 假设 logger 已定义

class CheckpointManager:
    def __init__(self, checkpoint_dir: str, keep_count: int = 3, save_optimizer: bool = True):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_count = keep_count
        self.save_optimizer = save_optimizer
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # 维护一个 history 文件来避免加载大模型文件查找 best
        self.history_file = self.checkpoint_dir / "history.json"
        self._history = self._load_history()

    def _load_history(self) -> List[Dict]:
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []

    def _save_history(self) -> None:
        with open(self.history_file, 'w') as f:
            json.dump(self._history, f, indent=2)

    @staticmethod
    def get_git_commit_hash():
        try:
            return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        except Exception:
            return "unknown"

    def save(
        self, 
        epoch: int, 
        model: torch.nn.Module, 
        save_file: str | None = None,
        metrics: Dict[str, float]| None = None, 
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        **kwargs) -> str:
        """
        保存模型检查点，包含模型状态、优化器状态、学习率调度器状态、指标和 Git 提交哈希。
        
        参数:
            model: 要保存的模型实例
            epoch: 当前训练轮次（整数）
            metrics: 训练指标字典（可选）
            optimizer: 优化器实例（可选）
            scheduler: 学习率调度器实例（可选）
            **kwargs: 其他自定义字段（会被保存但不影响加载）
            
        返回:
            检查点文件路径（字符串）
        """
        # --- 前置检查 (Pre-checks) ---
        
        # 1. 检查模型是否为空
        if model is None:
            raise ValueError("Cannot save checkpoint: 'model' is None.")
            
        # 2. 检查 Epoch 格式（防止传入 'last' 这种字符串导致后续排序崩溃）
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError(f"Epoch must be a non-negative integer, got {epoch} (type: {type(epoch)})")
            
        # 3. 检查保存目录是否存在（防止训练途中被人手滑删了文件夹）
        # 如果不存在，这里可以做一个"自愈"操作，自动重建
        if not self.checkpoint_dir.exists():
            logger.log(f"Warning: Checkpoint directory {self.checkpoint_dir} was missing. Re-creating it.", tag="Warn")
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 4. 检查 Metrics 格式
        if metrics is not None and not isinstance(metrics, dict):
            logger.log("Warning: 'metrics' is not a dictionary. It might cause issues in history tracking.", tag="Warn")
            # 这里可以选择报错，也可以选择只是警告，看你对严谨性的要求

        filename = f"checkpoint_epoch_{epoch:06d}.pth"
        if save_file is not None:
            checkpoint_path = self.checkpoint_dir / save_file
        else :
            checkpoint_path = self.checkpoint_dir / filename
        
        # 构造数据
        checkpoint_data = {
            'epoch': epoch,
            'metrics': metrics,
            'model_state_dict': model.state_dict(),

            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,

            'git_commit': self.get_git_commit_hash(),
        }
        
        # 1. 原子化写入：先写临时文件，再 rename
        try:
            tmp_path = checkpoint_path.with_suffix('.tmp')
            torch.save(checkpoint_data, tmp_path)
            tmp_path.rename(checkpoint_path) # 原子操作，防止损坏
            
            # 2. 更新 history
            self._history.append({
                'epoch': epoch,
                'path': str(checkpoint_path),
                'metrics': metrics or {},
                'timestamp': datetime.now().isoformat()
            })
            self._save_history()
            
            # 3. 清理旧权重 (基于 Epoch 排序，而非时间)
            self._cleanup_old_checkpoints()
            
            return str(checkpoint_path)
        except Exception as e:
            logger.log(f"Error saving checkpoint: {e}", tag="Checkpoint Err", log_to_console=True)
            raise e

    def get_best_checkpoint(self, metric_key: str = 'reward', higher_is_better: bool = True) -> Optional[str]:
        """
        获取历史记录中最佳检查点路径，基于指定指标排序。
        
        参数:
            metric_key: 要排序的指标键名（默认 'reward'）
            higher_is_better: 是否指标值越高越好（默认 True）
            
        返回:
            最佳检查点路径（字符串）或 None（如果没有记录或指标不存在）
        """
        if not self._history:
            return None
            
        # 过滤掉没有该指标的记录
        valid_records = [r for r in self._history if r.get('metrics') and metric_key in r['metrics']]
        if not valid_records:
            return None
            
        # 排序查找
        sorted_records = sorted(
            valid_records, 
            key=lambda x: x['metrics'][metric_key], 
            reverse=higher_is_better
        )
<<<<<<< HEAD

=======
        
>>>>>>> 10df8564a669efb4c9baf12ad895ffcd004530f0
        best_record = sorted_records[0]
        
        # 验证文件是否存在
        if os.path.exists(best_record['path']):
            return best_record['path']
        return None

    def get_latest_checkpoint(self) -> str | None:
        """
        获取最新的 checkpoint 路径，基于 Epoch 排序。
        
        返回:
            最新检查点路径（字符串）或 None（如果没有记录）
        """
        if not self._history:
            return None
            
        # 按 epoch 从大到小排序
        sorted_records = sorted(
            self._history, 
            key=lambda x: x['epoch'], 
            reverse=True
        )
        latest_record = sorted_records[0]
        
        # 验证文件是否存在
        if os.path.exists(latest_record['path']):
            return latest_record['path']
        return None
    
    def _cleanup_old_checkpoints(self) -> None:
        # 基于文件名中的 epoch 数字解析，比 getmtime 更可靠
        files = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        
        def parse_epoch(p: Path):
            # 提取 000123
            match = re.search(r'epoch_(\d+)', p.name)
            return int(match.group(1)) if match else -1

        # 按 epoch 从小到大排序
        files.sort(key=parse_epoch)
        
        if len(files) > self.keep_count:
            files_to_delete = files[:-self.keep_count]
            for f in files_to_delete:
                try:
                    f.unlink() # Pathlib 的删除方法
                    # 同步从 history 中移除
                    self._history = [h for h in self._history if h['path'] != str(f)]
                except Exception:
                    pass
            self._save_history()

    def load(
        self,
        model: torch.nn.Module,
        checkpoint_path: str | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        normalizer: Any | None = None,  # 新增：RL 专用的 RunningMeanStd 归一化器
        device: str | torch.device = 'cpu',
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        全能加载函数：支持断点续训和推理评估
        
        Args:
            model: 模型实例 (必须)
            checkpoint_path: 指定路径。如果为 None，自动查找最新的。
            optimizer: 优化器。续训时必须传，推理时为 None。
            scheduler: 学习率调度器。续训时建议传。
            normalizer: RL 的状态归一化器 (RunningMeanStd)。RL 必须传！
            device: 'cpu' 或 'cuda:0'。自动处理跨设备加载。
            strict: 是否严格匹配模型键值 (微调时可设为 False)。
            
        Returns:
            info_dict: 包含 'start_epoch', 'global_step', 'config', 'metrics' 等信息的字典
        """
        # 1. 确定加载哪个文件
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()
            if checkpoint_path is None:
                logger.log("No checkpoint found to load.", tag="Load", log_to_console=True)
                return {} # 返回空字典，由外部判断是否报错
        
        ckpt_path_obj = Path(checkpoint_path)
        if not ckpt_path_obj.exists():
            logger.log(f"Checkpoint file not found: {ckpt_path_obj}", tag="Load Err", log_to_console=True)
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path_obj}")

        logger.log(f"Loading checkpoint from: {ckpt_path_obj} ...", tag="Load")

        try:
            # 2. 加载文件 (处理 map_location 防止 GPU/CPU 不匹配报错)
            checkpoint_data = torch.load(ckpt_path_obj, map_location=device)

            # 3. 加载模型权重 (Model)
            # state_dict key 可能会有 'module.' 前缀 (如果你用了 DataParallel)，这里可以自动修剪
            state_dict = checkpoint_data['model_state_dict']
            # consume_prefix_in_state_dict_if_present(state_dict, "module.") # 如果需要兼容多卡
            model.load_state_dict(state_dict, strict=strict)
            
            # 4. 加载优化器 (Optimizer) - 仅当传入且存在时
            if optimizer is not None:
                if 'optimizer_state_dict' in checkpoint_data:
                    optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                else:
                    logger.log("Warning: Optimizer provided but missing in checkpoint.", tag="Warn", log_to_console=True)
                    
            # 5. 加载调度器 (Scheduler)
            if scheduler is not None:
                if 'scheduler_state_dict' in checkpoint_data:
                    scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
                else:
                    logger.log("Warning: Scheduler provided but missing in checkpoint.", tag="Warn", log_to_console=True)
            
            # 6. 加载 RL 归一化器 (Normalizer) - 关键！
            if normalizer is not None:
                # 兼容不同的命名习惯，尝试找 obs_normalizer 或 running_mean_std
                if 'obs_normalizer' in checkpoint_data:
                    normalizer.load_state_dict(checkpoint_data['obs_normalizer'])
                elif 'obs_rms' in checkpoint_data: # 另一种常见的命名
                    normalizer.load_state_dict(checkpoint_data['obs_rms'])
                else:
                    logger.log("Warning: Normalizer provided but missing in checkpoint!", tag="Warn")

            # 7. 提取元数据用于恢复训练状态
            loaded_epoch = checkpoint_data.get('epoch', -1)
            info = {
                'start_epoch': loaded_epoch + 1,  # 续训应该从下一轮开始
                'global_step': checkpoint_data.get('global_step', 0),
                'config': checkpoint_data.get('config', {}),
                'metrics': checkpoint_data.get('metrics', {}),
                'best_metric': checkpoint_data.get('best_metric', None), # 如果你有存这个
                'checkpoint_path': str(ckpt_path_obj)
            }
            
            logger.log(f"Successfully loaded. Resume from Epoch {info['start_epoch']}.", tag="Load")
            return info

        except Exception as e:
            logger.log(f"Failed to load checkpoint: {e}", tag="Error")
            raise e