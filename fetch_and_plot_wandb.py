import os
import sys
import re
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reconfigure stdout/stderr for unicode safety on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# Constants
ENTITY = "giabao240806-fpt-university"
PROJECT = "Đồng cam mất ngủ"
WANDB_PATH = f"{ENTITY}/{PROJECT}"
OUTPUT_DIR = "results"
CSV_PATH = os.path.join(OUTPUT_DIR, "wandb_history.csv")
PNG_PATH = os.path.join(OUTPUT_DIR, "learning_curves.png")
PDF_PATH = os.path.join(OUTPUT_DIR, "learning_curves.pdf")

def fetch_wandb_data():
    api = wandb.Api()
    # Print project name cleanly in ASCII to prevent console encoding issues
    print(f"Connecting to WandB project: {ENTITY}/[Dong cam mat ngu]...")
    try:
        runs = api.runs(WANDB_PATH)
    except Exception as e:
        print(f"Error accessing WandB project: {e}")
        return None, None

    print(f"Found {len(runs)} runs in the project.")
    
    all_histories = []
    run_runtimes = []

    for run in runs:
        name = run.name
        # Match run name pattern like: DeepSleepNet_fold0_seq20_stride5
        # or similar naming containing architecture and fold
        match = re.match(r"([a-zA-Z0-9]+)_fold(\d+)", name)
        if not match:
            print(f"Skipping run with unexpected name format: {name}")
            continue
            
        arch = match.group(1)
        fold = int(match.group(2))
        
        # Get run summary runtime
        runtime = run.summary.get("_runtime", 0)
        run_runtimes.append({
            "arch": arch,
            "fold": fold,
            "runtime_seconds": runtime
        })
        
        print(f"Fetching history for {arch} | Fold {fold} (Runtime: {runtime:.1f}s)...")
        
        try:
            # Retrieve history of logged metrics
            history = run.history(keys=["epoch", "train_loss", "val_f1", "val_acc", "val_kappa"], samples=100)
            if history.empty:
                # Fallback to scan_history if history is empty
                history = pd.DataFrame(run.scan_history(keys=["epoch", "train_loss", "val_f1", "val_acc", "val_kappa"]))
                
            if not history.empty:
                history["arch"] = arch
                history["fold"] = fold
                all_histories.append(history)
            else:
                print(f"Warning: No history data found for run {name}")
        except Exception as e:
            print(f"Error fetching history for run {name}: {e}")

    if not all_histories:
        print("No history data collected from any runs.")
        return None, None
        
    # Combine all history dataframes
    df_history = pd.concat(all_histories, ignore_index=True)
    
    # Filter columns and rename/reorder
    expected_cols = ["arch", "fold", "epoch", "train_loss", "val_f1", "val_acc", "val_kappa"]
    # Ensure all expected columns exist
    for col in expected_cols:
        if col not in df_history.columns:
            df_history[col] = np.nan
            
    df_history = df_history[expected_cols]
    # Sort history values
    df_history = df_history.sort_values(by=["arch", "fold", "epoch"]).reset_index(drop=True)
    
    df_runtimes = pd.DataFrame(run_runtimes)
    return df_history, df_runtimes

def plot_learning_curves(df):
    print("Plotting learning curves...")
    
    # Get unique architectures
    architectures = sorted(df["arch"].unique())
    
    # Setup subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Layout mapping
    metrics = [
        {"col": "train_loss", "title": "Training Loss", "ylabel": "Loss", "ax": axes[0, 0]},
        {"col": "val_acc", "title": "Validation Accuracy", "ylabel": "Accuracy", "ax": axes[0, 1]},
        {"col": "val_f1", "title": "Validation Macro-F1", "ylabel": "Macro-F1", "ax": axes[1, 0]},
        {"col": "val_kappa", "title": "Validation Cohen's Kappa", "ylabel": "Kappa", "ax": axes[1, 1]}
    ]
    
    # Color scheme for architectures
    colors = {
        "TinySleepNet": "#1f77b4",       # Blue
        "DeepSleepNet": "#ff7f0e",       # Orange
        "SleepTransformer": "#2ca02c",   # Green
        "MambaSleep": "#9467bd"          # Purple
    }
    
    # Preferred display order
    display_order = ["TinySleepNet", "DeepSleepNet", "SleepTransformer", "MambaSleep"]
    
    for metric in metrics:
        ax = metric["ax"]
        col = metric["col"]
        
        ax.set_title(metric["title"], fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(metric["ylabel"], fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)
        
        # Plot each architecture in preferred order
        for arch in display_order:
            if arch not in architectures:
                continue
                
            arch_df = df[df["arch"] == arch]
            
            # Group by epoch and calculate mean and std across folds
            grouped = arch_df.groupby("epoch")[col].agg(["mean", "std"]).reset_index()
            # Drop NaN values
            grouped = grouped.dropna(subset=["mean"])
            
            epochs = grouped["epoch"]
            mean_vals = grouped["mean"]
            std_vals = grouped["std"]
            
            color = colors.get(arch, "#7f7f7f")
            
            # Plot mean line
            ax.plot(epochs, mean_vals, label=f"{arch}-adapted", color=color, linewidth=2.2)
            
            # Fill standard deviation area
            ax.fill_between(epochs, mean_vals - std_vals, mean_vals + std_vals, 
                            color=color, alpha=0.15)
            
        ax.legend(fontsize=10, loc="best")
        
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save both PNG and PDF formats
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    print(f"Learning curves saved to {PNG_PATH} and {PDF_PATH}")
    plt.close(fig)

def print_runtime_summary(df_runtimes):
    if df_runtimes is None or df_runtimes.empty:
        return
        
    print("\n" + "="*45)
    print("          TRAINING RUNTIME SUMMARY")
    print("="*45)
    
    summary_data = []
    
    for arch in sorted(df_runtimes["arch"].unique()):
        arch_df = df_runtimes[df_runtimes["arch"] == arch]
        runtimes = arch_df["runtime_seconds"]
        
        mean_time = runtimes.mean()
        std_time = runtimes.std()
        total_time = runtimes.sum()
        
        def format_duration(seconds):
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            if h > 0:
                return f"{h}h {m}m {s}s"
            elif m > 0:
                return f"{m}m {s}s"
            else:
                return f"{s}s"
                
        summary_data.append({
            "Architecture": f"{arch}-adapted",
            "Avg Fold Time": f"{format_duration(mean_time)} +/- {format_duration(std_time)}",
            "Total Time": format_duration(total_time),
            "Avg (sec)": f"{mean_time:.1f}s",
            "Total (sec)": f"{total_time:.1f}s"
        })
        
    summary_df = pd.DataFrame(summary_data)
    # Using simple text representation to prevent markdown formatting issues with unicode
    for idx, row in summary_df.iterrows():
        print(f"{row['Architecture']:<25} | Avg: {row['Avg Fold Time']:<20} | Total: {row['Total Time']}")
    print("="*45)

def main():
    # 1. Fetch data from WandB
    df_history, df_runtimes = fetch_wandb_data()
    
    if df_history is not None:
        # 2. Save combined history
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df_history.to_csv(CSV_PATH, index=False)
        print(f"Successfully saved WandB history to {CSV_PATH}")
        
        # 3. Plot learning curves
        plot_learning_curves(df_history)
        
        # 4. Print runtime summary
        print_runtime_summary(df_runtimes)
    else:
        print("Failed to retrieve metrics history from WandB.")

if __name__ == "__main__":
    main()
