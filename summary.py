import os
import json
import argparse
import numpy as np
import pandas as pd

def compile_results(results_dir):
    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' does not exist.")
        return
        
    architectures = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    
    all_summary = []
    
    for arch in architectures:
        arch_dir = os.path.join(results_dir, arch)
        acc_list, f1_list, kappa_list = [], [], []
        f1_classes = {"W": [], "N1": [], "N2": [], "N3": [], "REM": []}
        
        # Look for folds (fold_0 to fold_9)
        for fold in range(10):
            fold_dir = os.path.join(arch_dir, f"fold_{fold}")
            metrics_path = os.path.join(fold_dir, "metrics.json")
            
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                acc_list.append(metrics["test_acc"])
                f1_list.append(metrics["test_macro_f1"])
                kappa_list.append(metrics["test_kappa"])
                
                # Per-class f1
                for c in ["W", "N1", "N2", "N3", "REM"]:
                    f1_classes[c].append(metrics["per_class_f1"].get(c, 0.0))
                    
        if len(acc_list) > 0:
            def format_metric(lst):
                # Returns 'mean ± std' format
                mean = np.mean(lst)
                std = np.std(lst)
                return f"{mean:.4f} ± {std:.4f}"
                
            summary_entry = {
                "Model": arch,
                "ACC": format_metric(acc_list),
                "MF1": format_metric(f1_list),
                "Kappa": format_metric(kappa_list),
                "F1-W": format_metric(f1_classes["W"]),
                "F1-N1": format_metric(f1_classes["N1"]),
                "F1-N2": format_metric(f1_classes["N2"]),
                "F1-N3": format_metric(f1_classes["N3"]),
                "F1-REM": format_metric(f1_classes["REM"])
            }
            all_summary.append(summary_entry)
            print(f"Compiled results for {arch} over {len(acc_list)} folds.")
            
    if len(all_summary) > 0:
        df = pd.DataFrame(all_summary)
        summary_csv_path = os.path.join(results_dir, "summary.csv")
        df.to_csv(summary_csv_path, index=False)
        print(f"\nUnified summary table written to: {summary_csv_path}")
        print("\nSummary Results:")
        print(df.to_markdown(index=False))
    else:
        print("No metrics.json files were found to compile.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='/kaggle/working/results', 
                        help='Path to the directory where fold results are saved')
    args = parser.parse_args()
    compile_results(args.results_dir)

if __name__ == "__main__":
    main()
