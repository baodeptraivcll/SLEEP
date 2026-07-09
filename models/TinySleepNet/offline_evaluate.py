import os
import argparse
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from common.splits import get_kfold_splits
from common.dataset import SleepEDFDataset
from common.evaluate import evaluate_model
from models.TinySleepNet.models.tinysleepnet import TinySleepNet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to SC_Data")
    parser.add_argument("--model_dir", type=str, default=".", help="Directory containing fold results or best_model_fold_X.pth files")
    parser.add_argument("--k_folds", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Preparing K-Fold splits from {args.data_dir}...")
    splits = get_kfold_splits(args.data_dir, k=args.k_folds)
    
    all_acc, all_f1, all_kappa = [], [], []

    for fold in range(args.k_folds):
        # Check standard path results/TinySleepNet/fold_X/best_model.pth
        model_path = os.path.join(args.model_dir, "results", "TinySleepNet", f"fold_{fold}", "best_model.pth")
        if not os.path.exists(model_path):
            # Fallback 1: best_model_fold_X.pth
            model_path = os.path.join(args.model_dir, f"best_model_fold_{fold}.pth")
        if not os.path.exists(model_path):
            # Fallback 2: best_model_fold_{X+1}.pth (1-indexed)
            model_path = os.path.join(args.model_dir, f"best_model_fold_{fold+1}.pth")
            
        if not os.path.exists(model_path):
            print(f"Warning: model checkpoint not found for fold {fold}. Checked standard path and fallbacks. Skipping.")
            continue
            
        print(f"\n==============================")
        print(f"EVALUATING FOLD {fold}/{args.k_folds}")
        print(f"==============================")
        
        train_idx, val_idx, test_idx = splits[fold]
        # Evaluate on test set
        test_dataset = SleepEDFDataset(test_idx, seq_len=20, stride=20, split="test", pad_last=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        model = TinySleepNet(in_channels=1, num_classes=5).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        metrics = evaluate_model(model, test_loader, device=device)
        acc = metrics["accuracy"]
        f1 = metrics["macro_f1"]
        kappa = metrics["kappa"]
        
        print(f"Fold {fold} Test Results: ACC={acc:.4f} | Macro-F1={f1:.4f} | Kappa={kappa:.4f}")
        
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
