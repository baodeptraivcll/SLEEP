import os
import argparse
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
import wandb
import gc
from torch.utils.data import DataLoader
from dataset import get_kfold_splits, SleepEDFDataset
from models.tinysleepnet import TinySleepNet
from train import train_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to directory containing .npz files')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size (default: 20 sequences of 15 epochs)')
    parser.add_argument('--k_folds', type=int, default=10, help='Number of cross-validation folds')
    parser.add_argument('--seq_length', type=int, default=15, help='Sequence length for LSTM')
    parser.add_argument('--wandb_project', type=str, default='sleep-staging-benchmark', help='Wandb project name')
    parser.add_argument('--start_fold', type=int, default=1, help='Fold to start from (1-indexed, e.g., 6 to resume from fold 6)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ponytail: Lấy danh sách chia fold (chỉ lấy đường dẫn file, KHÔNG load vào RAM)
    print(f"Preparing K-Fold splits from {args.data_dir}...")
    splits = get_kfold_splits(args.data_dir, k=args.k_folds)
    
    # ponytail: Mảng lưu kết quả các Folds
    all_f1 = []
    all_kappa = []
    all_acc = []

    # Chạy K-Fold Cross Validation
    for fold, (train_files, val_files) in enumerate(splits):
        if fold + 1 < args.start_fold:
            continue
            
        print(f"\n{'='*30}")
        print(f"BẮT ĐẦU FOLD {fold + 1}/{args.k_folds}")
        print(f"{'='*30}")
        
        # Load data CHO ĐÚNG FOLD HIỆN TẠI VÀO RAM
        print("Loading data for this fold...")
        train_dataset = SleepEDFDataset(train_files, augment=True, seq_length=args.seq_length)
        val_dataset = SleepEDFDataset(val_files, augment=False, seq_length=args.seq_length)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
        # Bắt buộc log WandB theo yêu cầu giáo viên (đồng bộ với team)
        wandb.init(
            entity="giabao240806-fpt-university",
            project="Đồng cam mất ngủ",
            config={
                "learning_rate": 1e-4,
                "architecture": "TinySleepNet",
                "dataset": "Sleep-EDF",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "early_stopping_patience": 20,
                "seq_length": args.seq_length,
                "fold": fold + 1
            },
            name=f"TinySleepNet_Fold_{fold+1}",
            save_code=True
        )

        model = TinySleepNet(in_channels=1, num_classes=5)
        
        # ponytail: Weighted Loss phạt N1 nặng hơn do Class Imbalance (giống paper)
        # Các trọng số mặc định cho Sleep-EDF (Wake, N1, N2, N3, REM)
        class_weights = [1.0, 1.5, 1.0, 1.0, 1.0] 

        best_model_path = f"best_model_fold_{fold+1}.pth"
        
        train_model(
            model=model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            class_weights=class_weights, 
            epochs=args.epochs, 
            device=device,
            save_path=best_model_path
        )
        
        # Đánh giá lại mô hình tốt nhất của Fold này
        print(f"\nĐÁNH GIÁ CUỐI CÙNG FOLD {fold + 1}")
        model.load_state_dict(torch.load(best_model_path))
        from evaluate import evaluate_model
        acc, f1, kappa, cm = evaluate_model(model, val_loader, device=device, verbose=True)
        
        # Log final metrics for the fold
        wandb.log({
            "final_fold_acc": acc,
            "final_fold_f1": f1,
            "final_fold_kappa": kappa
        })
        
        # ponytail: Bắn file weights (.pth) lên mây WandB luôn để không bị mất khi Kaggle sập
        wandb.save(best_model_path)
        
        wandb.finish()
        
        all_acc.append(acc)
        all_f1.append(f1)
        all_kappa.append(kappa)
        
        # ponytail: Giải phóng RAM cực mạnh để không bị tràn khi sang fold tiếp theo
        del train_dataset, val_dataset, train_loader, val_loader
        gc.collect()
        
    print(f"\n{'*'*40}")
    print(f"KẾT QUẢ TỔNG HỢP {args.k_folds}-FOLD CV")
    print(f"{'*'*40}")
    print(f"Mean Accuracy: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"Mean Macro F1: {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
    print(f"Mean Kappa   : {np.mean(all_kappa):.4f} ± {np.std(all_kappa):.4f}")

if __name__ == "__main__":
    main()

