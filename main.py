import os
import random
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from common.splits import get_kfold_splits, check_no_subject_leakage, extract_subject_id
from common.dataset import SleepEDFDataset
from common.train import run_training

# Set reproducibility seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_model(architecture):
    if architecture == "TinySleepNet":
        from models.TinySleepNet.models.tinysleepnet import TinySleepNet
        return TinySleepNet(in_channels=1, num_classes=5)
    elif architecture == "DeepSleepNet":
        from models.DeepSleepNet.deepsleepnet import DeepSleepNet
        return DeepSleepNet(in_channels=1, num_classes=5)
    elif architecture == "SleepTransformer":
        from models.SleepTransformer.sleeptransformer import SleepTransformer
        return SleepTransformer(in_channels=1, num_classes=5)
    elif architecture == "MambaSleep":
        from models.MambaSleep.mambasleep import MambaSleep
        return MambaSleep(in_channels=1, num_classes=5)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

def run_single_fold(args, fold_id, splits, device):
    print(f"\n==========================================")
    print(f"STARTING FOLD {fold_id} / 10 FOR {args.architecture}")
    print(f"==========================================")
    
    train_files, val_files, test_files = splits[fold_id]
    
    # 1. Check subject leakage
    print(f"Fold {fold_id}")
    train_subjects = sorted(list(set(extract_subject_id(f) for f in train_files)))
    val_subjects = sorted(list(set(extract_subject_id(f) for f in val_files)))
    test_subjects = sorted(list(set(extract_subject_id(f) for f in test_files)))
    
    print(f"Train subjects: {', '.join(train_subjects)}")
    print(f"Val subjects: {', '.join(val_subjects)}")
    print(f"Test subjects: {', '.join(test_subjects)}")
    print(f"Train recordings: {len(train_files)}")
    print(f"Val recordings: {len(val_files)}")
    print(f"Test recordings: {len(test_files)}")
    
    check_no_subject_leakage(train_files, val_files, test_files)
    
    # 2. Datasets
    train_dataset = SleepEDFDataset(
        train_files,
        seq_len=args.seq_len,
        stride=args.train_stride,
        split="train",
        pad_last=True,
        dtype=np.float32
    )
    val_dataset = SleepEDFDataset(
        val_files,
        seq_len=args.seq_len,
        stride=args.val_stride,
        split="val",
        pad_last=True,
        dtype=np.float32
    )
    test_dataset = SleepEDFDataset(
        test_files,
        seq_len=args.seq_len,
        stride=args.test_stride,
        split="test",
        pad_last=True,
        dtype=np.float32
    )
    
    # 3. DataLoaders
    # num_workers=0 is safe for Windows / Kaggle
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # 4. Build and initialize model
    model = build_model(args.architecture)
    model.to(device)
    
    # 5. WandB setup
    if args.use_wandb:
        import wandb
        os.environ["WANDB_SILENT"] = "true"
        # We update fold_id inside config so it matches the current fold run
        args.fold_id = fold_id
        
        run_name = f"{args.architecture}_fold{fold_id}_seq{args.seq_len}_stride{args.train_stride}"
        wandb.init(
            entity="giabao240806-fpt-university",
            project="Đồng cam mất ngủ",
            name=run_name,
            save_code=True,
            config={
                "architecture": args.architecture,
                "dataset": "Sleep-EDF-SC",
                "fold_id": fold_id,
                "epochs": args.epochs,
                "seq_len": args.seq_len,
                "train_stride": args.train_stride,
                "val_stride": args.val_stride,
                "test_stride": args.test_stride,
                "dtype": "float32",
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "optimizer": "AdamW",
                "loss": "Weighted Cross Entropy",
                "class_weights": "train_only",
                "split": "subject-wise 10-fold",
                "channel": "Fpz-Cz",
                "early_stopping_patience": args.early_stopping_patience,
            }
        )
        
    # 6. Train model
    run_training(model, train_loader, val_loader, test_loader, train_files, args, device)
    
    if args.use_wandb:
        import wandb
        wandb.finish()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to npz data directory')
    parser.add_argument('--architecture', type=str, required=True, 
                        choices=["DeepSleepNet", "TinySleepNet", "SleepTransformer", "MambaSleep"])
    parser.add_argument('--fold_id', type=int, default=0, help='Fold ID to run (0..9)')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs to train')
    parser.add_argument('--seq_len', type=int, default=20, help='Sequence length')
    parser.add_argument('--train_stride', type=int, default=5, help='Train sequence stride')
    parser.add_argument('--val_stride', type=int, default=20, help='Val sequence stride')
    parser.add_argument('--test_stride', type=int, default=20, help='Test sequence stride')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases')
    parser.add_argument('--run_all_folds', action='store_true', help='Train all folds sequentially')
    parser.add_argument('--seed', type=int, default=42, help='Reproducibility seed')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    set_seed(args.seed)
    
    splits = get_kfold_splits(args.data_dir, k=10, seed=args.seed)
    
    if args.run_all_folds:
        for f in range(10):
            run_single_fold(args, f, splits, device)
    else:
        run_single_fold(args, args.fold_id, splits, device)

if __name__ == "__main__":
    main()
