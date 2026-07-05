# SLEEP: Deep Learning Architecture Benchmark for Sleep Staging

This repository contains the implementation of a comprehensive benchmark comparing State-of-the-Art (SOTA) deep learning architectures for Sleep Staging using the **Sleep-EDF Expanded (Sleep Cassette)** dataset.

## 📌 Project Overview
The main goal of this group project is to establish a rigorous, fair, and reproducible benchmarking pipeline to compare 5 classic models:
1. **DeepSleepNet** (CNN-BiLSTM)
2. **TinySleepNet** (CNN-LSTM)
3. **AttnSleep** (CNN-Attention)
4. **SleepTransformer** (Transformer with CNN front-end variant)
5. **MambaSleep** (State Space Model)

## 🎯 Current Status
Currently, this repository highlights the complete, highly-optimized PyTorch implementation of **DeepSleepNet**.

### DeepSleepNet Features:
- **True-to-paper architecture**: Dual-branch CNN (Small & Large filters) + 2-layer BiLSTM + Shortcut/Residual Connection.
- **Optimized for Sequence-of-Epochs**: The `forward` pass is heavily optimized to process `(Batch, Seq_Len, C, L)` 4D tensors, maximizing GPU parallelization across sequences.
- **Zero-pad Optimization**: Removed standard PyTorch asymmetric padding overhead by using manual `nn.ZeroPad1d`.

## 📁 Repository Structure
```
.
├── models/
│   └── deepsleepnet.py        # Optimized PyTorch implementation of DeepSleepNet
├── data_preprocessing.py      # Script for Wake Trimming, Bandpass Filtering, and Z-score Norm (WIP)
├── implementation_plan.md     # Detailed blueprint for the 5-model benchmark pipeline
└── README.md                  # This file
```

## 🚀 Data Pipeline Design (Strict Constraints)
To ensure **zero data leakage** and absolute fairness, our dataloader is designed with the following rigid rules:
- **Subject-wise K-Fold**: 10-fold cross-validation split by `Subject ID` (78 subjects), never by recordings.
- **Strict Boundaries**: Sequences are never concatenated across different subjects or different nights.
- **Remainder Handling**: The final epochs of a night that do not perfectly fit into a Sequence Window are padded/masked to ensure no rare sleep stages (N1/REM) are missed during testing.

## 📜 Usage
To test the output shapes and parameters of the DeepSleepNet implementation:
```bash
python models/deepsleepnet.py
```
*Expected Output:* Tensor shape `(Batch, Seq_Len, 5)` corresponding to the 5 AASM sleep stages (W, N1, N2, N3, REM).
