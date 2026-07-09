# Wrapper for compatibility
import torch
from common.train import run_training

def train_model(model, train_loader, val_loader, class_weights=None, epochs=50, device='cuda', save_path='best_model.pth'):
    class DummyArgs:
        def __init__(self):
            self.architecture = "TinySleepNet"
            self.fold_id = 0
            self.epochs = epochs
            self.seq_len = 20
            self.train_stride = 5
            self.val_stride = 20
            self.test_stride = 20
            self.batch_size = 32
            self.learning_rate = 1e-4
            self.early_stopping_patience = 20
            self.use_wandb = False
            
    args = DummyArgs()
    run_training(model, train_loader, val_loader, val_loader, [], args, device)
