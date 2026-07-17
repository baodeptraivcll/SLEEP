# ponytail: Simple evaluation script to copy, evaluate and summarize 40 models.
import os
import shutil
import json
import numpy as np
import torch
import random
from torch.utils.data import DataLoader

from common.splits import get_kfold_splits
from common.dataset import SleepEDFDataset
from common.evaluate import evaluate_model
from main import build_model
from tools.summary import compile_results

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(42)
    
    data_dir = "d:/SC_Data"
    downloaded_dir = "d:/SLEEP/downloaded_best_models"
    results_dir = "d:/SLEEP/results"
    
    architectures = ["TinySleepNet", "DeepSleepNet", "SleepTransformer", "MambaSleep"]
    
    splits = get_kfold_splits(data_dir, k=10, seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    for arch in architectures:
        for fold in range(10):
            print(f"\n--- Evaluating {arch} | Fold {fold} ---")
            
            # 1. Setup folders and copy weights
            src_folder = os.path.join(downloaded_dir, f"{arch}_fold{fold}_best_model")
            src_path = os.path.join(src_folder, "best_model.pth")
            
            out_dir = os.path.join(results_dir, arch, f"fold_{fold}")
            os.makedirs(out_dir, exist_ok=True)
            dst_path = os.path.join(out_dir, "best_model.pth")
            
            if not os.path.exists(src_path):
                print(f"Warning: model path {src_path} not found. Skipping evaluation.")
                continue
                
            shutil.copy(src_path, dst_path)
            
            # 2. Get the test files for this fold
            _, _, test_files = splits[fold]
            
            # 3. Create test dataset and dataloader
            test_dataset = SleepEDFDataset(
                test_files,
                seq_len=20,
                stride=20,
                split="test",
                pad_last=True,
                dtype=np.float32
            )
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
            
            # 4. Build and load the model
            model = build_model(arch)
            model.load_state_dict(torch.load(dst_path, map_location=device))
            model.to(device)
            
            # 5. Evaluate
            test_metrics = evaluate_model(model, test_loader, device=device)
            test_acc = test_metrics["accuracy"]
            test_f1 = test_metrics["macro_f1"]
            test_kappa = test_metrics["kappa"]
            
            print(f"Results for {arch} fold {fold}: ACC={test_acc:.4f} | Macro-F1={test_f1:.4f} | Kappa={test_kappa:.4f}")
            
            # 6. Save outputs to results directory
            metrics_json_path = os.path.join(out_dir, "metrics.json")
            metrics_to_save = {
                "fold_id": fold,
                "architecture": arch,
                "test_acc": test_acc,
                "test_macro_f1": test_f1,
                "test_kappa": test_kappa,
                "per_class_f1": test_metrics["per_class_f1"],
                "best_epoch": -1,
                "best_val_f1": -1.0
            }
            with open(metrics_json_path, "w") as f:
                json.dump(metrics_to_save, f, indent=2)
                
            cm_path = os.path.join(out_dir, "confusion_matrix.npy")
            np.save(cm_path, np.array(test_metrics["confusion_matrix"]))
            
            preds_path = os.path.join(out_dir, "predictions.npz")
            np.savez(preds_path, y_true=test_metrics["y_true"], y_pred=test_metrics["y_pred"])
            
            config_path = os.path.join(out_dir, "config.json")
            config_to_save = {
                "data_dir": data_dir,
                "architecture": arch,
                "fold_id": fold,
                "epochs": 50,
                "seq_len": 20,
                "train_stride": 5,
                "val_stride": 20,
                "test_stride": 20,
                "batch_size": 32,
                "learning_rate": 1e-4,
                "early_stopping_patience": 15,
                "use_wandb": False,
                "run_all_folds": False,
                "seed": 42
            }
            with open(config_path, "w") as f:
                json.dump(config_to_save, f, indent=2)
                
    print("\n==========================================")
    print("ALL EVALUATIONS COMPLETE. COMPILING SUMMARY...")
    print("==========================================")
    compile_results(results_dir)

if __name__ == "__main__":
    main()
