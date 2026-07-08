import os
import argparse
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import DataLoader
from dataset import get_kfold_splits, SleepEDFDataset
from models.tinysleepnet import TinySleepNet
from evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to SC_Data")
    parser.add_argument("--model_dir", type=str, default=".", help="Directory containing best_model_fold_X.pth files")
    parser.add_argument("--k_folds", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Preparing K-Fold splits from {args.data_dir}...")
    splits = get_kfold_splits(args.data_dir, k=args.k_folds)
    
    all_acc, all_f1, all_kappa = [], [], []

    for fold in range(args.k_folds):
        model_path = os.path.join(args.model_dir, f"best_model_fold_{fold+1}.pth")
        if not os.path.exists(model_path):
            print(f"Warning: {model_path} not found. Skipping fold {fold+1}.")
            continue
            
        print(f"\n==============================")
        print(f"EVALUATING FOLD {fold+1}/{args.k_folds}")
        print(f"==============================")
        
        train_idx, val_idx = splits[fold]
        val_dataset = SleepEDFDataset(val_idx)
        # num_workers=0 để tránh lỗi đa luồng trên Windows/Kaggle khi evaluate nhanh
        val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False, num_workers=0)
        
        model = TinySleepNet().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        acc, f1, kappa, cm = evaluate_model(model, val_loader, device=device, verbose=True)
        
        all_acc.append(acc)
        all_f1.append(f1)
        all_kappa.append(kappa)

    if len(all_acc) > 0:
        print(f"\n{'*'*40}")
        print(f"FINAL AGGREGATED RESULTS (over {len(all_acc)} folds):")
        print(f"Mean Accuracy: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
        print(f"Mean Macro F1: {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
        print(f"Mean Kappa:    {np.mean(all_kappa):.4f} ± {np.std(all_kappa):.4f}")
        print(f"{'*'*40}\n")
    else:
        print("\nError: No models evaluated. Check your --model_dir.")

if __name__ == "__main__":
    main()
