# SLEEP: Deep Learning Architecture Benchmark for Sleep Staging

This repository contains the implementation of a comprehensive benchmark comparing State-of-the-Art (SOTA) deep learning architectures for Sleep Staging using the **Sleep-EDF Expanded (Sleep Cassette)** dataset.

## 📌 Project Overview
The main goal of this project is to establish a rigorous, fair, and reproducible benchmarking pipeline to compare 4 classic models:
1. **DeepSleepNet** (CNN-BiLSTM)
2. **TinySleepNet** (CNN-LSTM)
3. **SleepTransformer** (Transformer with Sequence-to-Sequence processing)
4. **MambaSleep** (State Space Model)

---

## 🎯 Shared Evaluation Protocol
To ensure zero data leakage and absolute fairness, all models use the exact same setup:
- **Subject-wise K-Fold**: 10-fold cross-validation split by `Subject ID` (78 subjects), never by recordings.
- **Strict Boundaries**: Sequences are never concatenated across different subjects or different nights.
- **Remainder Handling**: The final epochs of a night that do not perfectly fit into a Sequence Window are padded/masked so no sleep stages are missed.
- **Identical Metrics**: Accuracy, Macro-F1 (fixed to labels `[0,1,2,3,4]`), Cohen's Kappa, and Confusion Matrix.
- **WandB Logs**: Shared log structure and unified metric tracking.

---

## 📁 Repository Structure
```
.
├── common/
│   ├── dataset.py             # Shared Sequence Dataloader with padding + mask support
│   ├── splits.py              # Subject-wise splits with leakage checking
│   ├── metrics.py             # Masked Accuracy, Macro-F1, Kappa, and Class F1 calculation
│   ├── evaluate.py            # Sequence-level evaluation loop
│   └── train.py               # Masked Loss trainer and checkpoint saving
├── models/
│   ├── TinySleepNet/          # Master template implementation wrapper folder
│   ├── DeepSleepNet/          # 1-step end-to-end DeepSleepNet implementation
│   ├── SleepTransformer/      # 1-step end-to-end SleepTransformer (raw EEG variant)
│   └── MambaSleep/            # Pure PyTorch MambaSleep State Space Model
├── tools/
│   ├── dataset_statistics.py  # Calculates global class distribution (Table 1)
│   ├── fetch_and_plot_wandb.py# Fetches WandB runs and plots learning curves
│   ├── plot_confusion_matrix.py# Aggregates and plots normalized confusion matrices
│   ├── statistical_analysis.py# Performs Friedman and pairwise Wilcoxon tests
│   └── summary.py             # Compiles metrics.json fold results into summary.csv
├── main.py                    # Unified entry point for training any model on a specific fold
├── evaluate_all.py            # Sequence evaluation script for all models
├── test_dummy.py              # Smoke test to verify model shapes
├── requirements.txt           # Project dependencies
└── README.md                  # This file
```

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/baodeptraivcll/SLEEP.git
   cd SLEEP
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Dataset**
   ```bash
   https://drive.google.com/drive/folders/1C5_7nPyU-0sohmYsDRUFND7O1-ScLZdL?usp=sharing
   ```
4. **best weights **
   ```bash
   https://drive.google.com/drive/folders/1C5_7nPyU-0sohmYsDRUFND7O1-ScLZdL?usp=sharing](https://drive.google.com/drive/folders/17NweDVmZ--cuTfbR-SHGk_7Mq3V6o1lU?usp=drive_link
   ```
---

## 🚀 Usage

### Smoke Test
Verify that all 4 models compile and run a dummy forward propagation with expected tensor shapes:
```bash
python test_dummy.py
```

### Run Benchmark Training
To train any of the 4 models on a single fold:
```bash
python main.py \
  --data_dir /path/to/npz_dataset \
  --architecture TinySleepNet \
  --fold_id 0 \
  --epochs 50 \
  --use_wandb
```
Replace `--architecture` with one of the following:
- `TinySleepNet`
- `DeepSleepNet`
- `SleepTransformer`
- `MambaSleep`

### Compile Summary Results
Compile all cross-validation fold results into a single CSV and print summary performance:
```bash
python tools/summary.py --results_dir /path/to/results
```

### Dataset Statistics (Table 1)
Calculate the global class distribution across all 153 sleep recordings:
```bash
python tools/dataset_statistics.py
```

### Statistical Analysis
Perform a Friedman test followed by pairwise Wilcoxon signed-rank tests with Holm-Bonferroni correction and effect size (median difference and rank-biserial correlation) calculations:
```bash
python tools/statistical_analysis.py --results_dir /path/to/results
```

### Generate Confusion Matrices
Aggregate confusion matrices from all 10 folds for each architecture and generate normalized heatmap comparisons:
```bash
python tools/plot_confusion_matrix.py --results_dir /path/to/results
```

