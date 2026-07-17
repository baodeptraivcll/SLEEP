# ponytail: Load 10 confusion matrices per architecture, aggregate, plot normalized heatmaps.
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]

# Preferred display order
DISPLAY_ORDER = ["TinySleepNet", "DeepSleepNet", "SleepTransformer", "MambaSleep"]


def load_confusion_matrices(results_dir, arch):
    """Load and sum confusion matrices from all 10 folds."""
    arch_dir = os.path.join(results_dir, arch)
    cm_sum = None
    loaded = 0
    for fold in range(10):
        path = os.path.join(arch_dir, f"fold_{fold}", "confusion_matrix.npy")
        if not os.path.exists(path):
            print(f"  Warning: {path} not found, skipping.")
            continue
        cm = np.load(path)
        cm_sum = cm if cm_sum is None else cm_sum + cm
        loaded += 1
    print(f"  {arch}: loaded {loaded}/10 folds")
    return cm_sum


def plot_confusion_matrices(results_dir):
    # Discover architectures
    all_dirs = sorted([
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
    ])
    # Order by DISPLAY_ORDER, then alphabetical for any extras
    architectures = [a for a in DISPLAY_ORDER if a in all_dirs]
    architectures += [a for a in all_dirs if a not in architectures]

    if not architectures:
        print("No architecture directories found.")
        return

    n = len(architectures)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for idx, arch in enumerate(architectures):
        ax = axes[idx]
        cm_sum = load_confusion_matrices(results_dir, arch)

        if cm_sum is None:
            ax.set_title(f"{arch}\n(no data)")
            ax.axis("off")
            continue

        # Row-normalize (recall-based)
        row_sums = cm_sum.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_norm = cm_sum / row_sums

        # Plot heatmap
        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)

        # Annotate cells with percentage and raw count
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                pct = cm_norm[i, j]
                count = int(cm_sum[i, j])
                color = "white" if pct > 0.5 else "black"
                ax.text(j, i, f"{pct:.1%}\n({count})",
                        ha="center", va="center", fontsize=8, color=color)

        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, fontsize=10)
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_yticklabels(CLASS_NAMES, fontsize=10)
        ax.set_xlabel("Predicted", fontsize=11)
        if idx == 0:
            ax.set_ylabel("True", fontsize=11)
        ax.set_title(f"{arch}", fontsize=12, fontweight="bold")

    # Shared colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Recall")

    fig.suptitle("Aggregated Confusion Matrices (10-Fold, Row-Normalized)",
                 fontsize=14, fontweight="bold", y=1.02)

    # Save
    png_path = os.path.join(results_dir, "confusion_matrices.png")
    pdf_path = os.path.join(results_dir, "confusion_matrices.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"\nSaved to {png_path} and {pdf_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot aggregated confusion matrices")
    parser.add_argument("--results_dir", type=str, default="./results",
                        help="Path to results directory")
    args = parser.parse_args()
    plot_confusion_matrices(args.results_dir)


if __name__ == "__main__":
    main()
