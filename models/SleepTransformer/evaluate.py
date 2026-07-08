import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

from sleeptransformer import SleepTransformer

DATA_DIR = 'd:\\SLEEP\\SC_Data'
WEIGHTS_PATH = 'weights/sleeptransformer_finetune_best.pth'
REPORTS_DIR = 'reports'
os.makedirs(REPORTS_DIR, exist_ok=True)
DEVICE = torch.device('cpu')  # Force CPU since Windows VM might not have CUDA

SEQ_LEN = 20

# 16 Files used for exact Kaggle validation (same as DeepSleepNet)
test_files_names = [
    'SC4382.npz', 'SC4661.npz', 'SC4722.npz', 'SC4031.npz', 
    'SC4541.npz', 'SC4701.npz', 'SC4111.npz', 'SC4062.npz', 
    'SC4462.npz', 'SC4532.npz', 'SC4092.npz', 'SC4702.npz', 
    'SC4321.npz', 'SC4341.npz', 'SC4211.npz', 'SC4551.npz'
]
test_files = [os.path.join(DATA_DIR, f) for f in test_files_names]

class SequenceDataset(Dataset):
    def __init__(self, file_paths, seq_len=20, stride=20):
        self.seq_len = seq_len
        self.data_cache = []
        self.windows_map = []
        
        for idx, p in enumerate(file_paths):
            data = np.load(p)
            self.data_cache.append({
                'x': torch.tensor(data['x'], dtype=torch.float32),
                'y': torch.tensor(data['y'], dtype=torch.long)
            })
            
            num_epochs = len(data['y'])
            for i in range(0, num_epochs - seq_len + 1, stride):
                self.windows_map.append((idx, i))

    def __len__(self):
        return len(self.windows_map)
        
    def __getitem__(self, idx):
        file_idx, start_idx = self.windows_map[idx]
        data = self.data_cache[file_idx]
        x_seq = data['x'][start_idx : start_idx + self.seq_len]
        y_seq = data['y'][start_idx : start_idx + self.seq_len]
        return x_seq, y_seq

print("Đang khởi tạo model SleepTransformer...")
model = SleepTransformer(in_channels=1, num_classes=5)
try:
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    print("✅ Đã load thành công tạ sleeptransformer_finetune_best.pth")
except Exception as e:
    print(f"❌ Lỗi load tạ: {e}")
    exit(1)

model.to(DEVICE)
model.eval()

test_ds = SequenceDataset(test_files, seq_len=SEQ_LEN, stride=SEQ_LEN)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

all_preds, all_labels = [], []
total = 0
correct = 0

start_time = time.time()
print("Đang chạy inference...")
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Evaluating"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs, mode="finetune")
        
        outputs = outputs.view(-1, outputs.size(-1))
        labels = labels.view(-1)
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

infer_time = time.time() - start_time

acc = 100. * correct / total
mf1 = 100. * f1_score(all_labels, all_preds, average='macro')
kappa = cohen_kappa_score(all_labels, all_preds)
report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
n1_f1 = report['1']['f1-score'] * 100

print("="*50)
print(f"✅ Độ chính xác tổng thể (Accuracy): {acc:.2f}%")
print(f"✅ Macro F1: {mf1:.2f}%")
print(f"✅ Cohen's Kappa: {kappa:.4f}")
print(f"✅ F1-Score Pha N1: {n1_f1:.2f}%")
print(f"⏱ Thời gian suy luận {total} epochs: {infer_time:.2f} giây")
print("="*50)

# Lưu kết quả
with open(os.path.join(REPORTS_DIR, 'metrics_report_latest.md'), 'w', encoding='utf-8') as f:
    f.write("# BÁO CÁO ĐÁNH GIÁ: SLEEPTRANSFORMER\n")
    f.write("**Ngày cập nhật:** 08/07/2026\n")
    f.write("**Tập dữ liệu Test:** 16 file được cố định chính xác (giống DeepSleepNet)\n\n")
    f.write("## 1. Các chỉ số tổng quát\n")
    f.write(f"*   **Độ chính xác tổng thể (Accuracy):** {acc:.2f}%\n")
    f.write(f"*   **Chỉ số Macro F1 (MF1):** {mf1:.2f}%\n")
    f.write(f"*   **Hệ số Cohen's Kappa:** {kappa:.4f}\n")
    f.write(f"*   **Thời gian Inference:** {infer_time:.2f} giây (CPU)\n")
    f.write(f"*   **Tổng số tham số mạng (Params):** {sum(p.numel() for p in model.parameters())}\n\n")
    f.write("## 2. Chỉ số chi tiết từng Pha giấc ngủ\n")
    f.write(f"*   F1-Score của pha N1: {n1_f1:.2f}%\n")

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
classes = ['W', 'N1', 'N2', 'N3', 'REM']
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('SleepTransformer Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'confusion_matrix.png'), dpi=300)
plt.close()

# Hypnogram (Plot first 1000 epochs)
plt.figure(figsize=(15, 4))
plot_len = min(1000, len(all_labels))
plt.plot(all_labels[:plot_len], label='True', color='blue', alpha=0.7, linestyle='--', drawstyle='steps-post')
plt.plot(all_preds[:plot_len], label='Predicted (SleepTransformer)', color='red', alpha=0.7, drawstyle='steps-post')
plt.yticks([0, 1, 2, 3, 4], classes)
plt.gca().invert_yaxis()
plt.title('Hypnogram Comparison (First 1000 epochs)')
plt.xlabel('Epochs')
plt.ylabel('Sleep Stage')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'hypnogram_comparison.png'), dpi=300)
plt.close()

print("✅ Đã xuất báo cáo và biểu đồ thành công!")
