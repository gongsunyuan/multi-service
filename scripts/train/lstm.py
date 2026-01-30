import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger

from multi_service.trainers.lstm_trainer import LSTMTrainer
from multi_service.agents.lstm_classifier import LSTMClassifier
from multi_service.utils import (
    SdnParaser, load_yaml_config, BankTrafficManager
)

def train_lstm(args):
    config = load_yaml_config(args.yaml)
    config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    lstmClassifier = LSTMClassifier(config).to(config.device)
    trainer = LSTMTrainer(lstmClassifier, config)
    traffic_generator = BankTrafficManager(config, config.path.fgprt_path)
    for epoch in range(1, config.train.epochs+1):
        epoch_loss = trainer.train_one_epoch(traffic_generator)
        logger.info(f"Epoch {epoch}: Loss {epoch_loss:.4f}")
    lstmClassifier.ckpt_manager.save(lstmClassifier.lstm, save_file="lstm.pth")

if __name__ == "__main__":
    parser = SdnParaser()
    args = parser.parse_args()
    train_lstm(args)
