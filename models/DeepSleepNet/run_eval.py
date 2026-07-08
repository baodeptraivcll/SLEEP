# Cài đặt thư viện vẽ biểu đồ và tính toán

import os
import time
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, cohen_kappa_score, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("✅ Đã nạp xong thư viện!")


# ====================================================================
# [CẤU HÌNH CƠ BẢN] DÀNH CHO CẢ 4 MODEL
# ====================================================================

# 1. Đường dẫn gốc tới Data

# 2. Đường dẫn tới file trọng số (.pth) tốt nhất của Model
# CẬP NHẬT ĐƯỜNG DẪN TỚI FILE .PTH CỦA NHÓM BẠN LÊN ĐÂY
MODEL_WEIGHT_PATH = './weights/deepsleepnet_finetune_best.pth'

DATA_DIR = '../../SC_Data'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Mapping nhãn giấc ngủ
LABEL_MAPPING = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

print("Cấu hình xong! Đang dùng thiết bị:", DEVICE)


# Dùng Sequence Dataset để mô phỏng dữ liệu test (stride = seq_len để TUYỆT ĐỐI KHÔNG chồng lấp dữ liệu)
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
                self.windows_map.append((idx, i, idx))

    def __len__(self):
        return len(self.windows_map)
        
    def __getitem__(self, idx):
        file_idx, start_idx, patient_idx = self.windows_map[idx]
        data = self.data_cache[file_idx]
        x_seq = data['x'][start_idx : start_idx + self.seq_len]
        y_seq = data['y'][start_idx : start_idx + self.seq_len]
        return x_seq, y_seq, patient_idx


# ====================================================================
# [KHU VỰC DÁN CODE MODEL]
# Copy và Paste class Model của nhóm bạn vào đây (Mamba, Tiny, Salient...)
# Ở dưới đang để mẫu DeepSleepNet để test
# ====================================================================

class DeepSleepNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=5):
        super(DeepSleepNet, self).__init__()
        
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=50, stride=6, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(0.5),
            nn.ZeroPad1d((3, 4)),
            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ZeroPad1d((3, 4)),
            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ZeroPad1d((3, 4)),
            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=400, stride=50, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(0.5),
            nn.ZeroPad1d((2, 3)),
            nn.Conv1d(64, 128, kernel_size=6, stride=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ZeroPad1d((2, 3)),
            nn.Conv1d(128, 128, kernel_size=6, stride=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ZeroPad1d((2, 3)),
            nn.Conv1d(128, 128, kernel_size=6, stride=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        cnn_out_features = 2688
        self.dropout_cnn = nn.Dropout(0.5)
        self.pretrain_classifier = nn.Linear(cnn_out_features, num_classes)
        
        hidden_size = 512
        self.bilstm = nn.LSTM(
            input_size=cnn_out_features, hidden_size=hidden_size, 
            num_layers=2, batch_first=True, bidirectional=True, dropout=0.5
        )
        
        self.shortcut = nn.Linear(cnn_out_features, hidden_size * 2)
        self.finetune_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size * 2, num_classes)
        )

    def _cnn_feature_extraction(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out1 = out1.view(out1.size(0), -1)
        out2 = out2.view(out2.size(0), -1)
        cnn_features = torch.cat((out1, out2), dim=1)
        cnn_features = self.dropout_cnn(cnn_features)
        return cnn_features

    def forward(self, x, mode="finetune"):
        if mode == "pretrain":
            if x.dim() == 4:
                x = x.squeeze(1) 
            cnn_features = self._cnn_feature_extraction(x)
            return self.pretrain_classifier(cnn_features)
            
        elif mode == "finetune":
            B, Seq_Len, C, L = x.shape
            x = x.view(B * Seq_Len, C, L)
            cnn_features = self._cnn_feature_extraction(x)
            seq_features = cnn_features.view(B, Seq_Len, -1)
            lstm_out, _ = self.bilstm(seq_features)
            shortcut_out = self.shortcut(seq_features)
            combined_features = lstm_out + shortcut_out
            return self.finetune_classifier(combined_features)


# Khởi tạo mô hình Universal
try:
    # SỬA TÊN CLASS MODEL CỦA BẠN VÀO ĐÂY (NẾU DÙNG TÊN KHÁC)
    model = DeepSleepNet().to(DEVICE)
    
    # Tính tổng lượng tham số Params
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Đã tải cấu trúc Model. Tổng tham số: {total_params:,} Params.")
    
except Exception as e:
    print(f"❌ Lỗi nạp Model: {e}\n(Kiểm tra lại xem bạn đã paste code class Model vào ô trên chưa)")

try:
    # Nạp trọng số
    if os.path.exists(MODEL_WEIGHT_PATH):
        model.load_state_dict(torch.load(MODEL_WEIGHT_PATH, map_location=DEVICE))
        print("✅ Đã nạp thành công bộ trọng số tốt nhất!")
    else:
        print(f"⚠️ CẢNH BÁO: Không tìm thấy file trọng số tại {MODEL_WEIGHT_PATH}\n(Model hiện đang dùng trọng số ngẫu nhiên rác để test logic vẽ hình)")
except Exception as e:
    print(f"❌ Lỗi nạp Trọng số: {e}")
    
model.eval()

# Nạp Data
all_files = glob.glob(os.path.join(DATA_DIR, "*.npz"))
if len(all_files) == 0:
    print(f"❌ Chưa có data trong {DATA_DIR}. Kiểm tra lại ổ Kaggle!")

# Chúng ta lấy 20 file cuối làm đại diện Test Set (hoặc bạn tự chia K-Fold tuỳ ý)
from sklearn.model_selection import KFold
kf = KFold(n_splits=10, shuffle=True, random_state=42)
_, test_idx = next(kf.split(all_files))
test_files = [all_files[i] for i in test_idx]
test_ds = SequenceDataset(test_files, seq_len=20, stride=20)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# Chạy Inference
all_preds = []
all_labels = []
patient_groups = {i: {"preds": [], "labels": []} for i in range(len(test_files))}

print("Đang đánh giá mô hình, vui lòng chờ...")
start_time = time.time()

with torch.no_grad():
    for inputs, labels, p_idxs in test_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        # Sửa tham số truyền vào tuỳ thuộc Model của bạn (có cần mode="finetune" không)
        outputs = model(inputs, mode="finetune")
        
        outputs = outputs.view(-1, outputs.size(-1))
        labels = labels.view(-1)
        
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Nhóm vào từng bệnh nhân để vẽ Hypnogram
        for i, p_idx in enumerate(p_idxs.cpu().numpy()):
            patient_groups[p_idx]["preds"].extend(predicted.cpu().numpy()[i*20:(i+1)*20])
            patient_groups[p_idx]["labels"].extend(labels.cpu().numpy()[i*20:(i+1)*20])

inference_time = time.time() - start_time

# Tính toán điểm số khắt khe
acc = accuracy_score(all_labels, all_preds) * 100
mf1 = f1_score(all_labels, all_preds, average='macro') * 100
kappa = cohen_kappa_score(all_labels, all_preds)

# F1 riêng biệt cho từng nhãn (Để chê bai N1)
f1_per_class = f1_score(all_labels, all_preds, average=None) * 100
n1_f1 = f1_per_class[1]

print("="*50)
print(f"📊 BÁO CÁO ĐÁNH GIÁ (UNIVERSAL)")
print("="*50)
print(f"• Số lượng tham số (Params): {total_params:,}")
print(f"• Thời gian Inference (Dự đoán {len(all_labels)} epochs): {inference_time:.2f} giây")
print(f"• Độ chính xác tổng thể (Accuracy): {acc:.2f}%")
print(f"• Chỉ số Macro F1 (MF1): {mf1:.2f}%")
print(f"• Hệ số Cohen's Kappa: {kappa:.4f}")
print(f"• [QUAN TRỌNG] F1-Score của pha N1 (pha cực khó): {n1_f1:.2f}%")
print("="*50)


# ====================================================================
# [VẼ MA TRẬN NHẦM LẪN - CONFUSION MATRIX]
# ====================================================================

plt.figure(figsize=(10, 8))
cm = confusion_matrix(all_labels, all_preds)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # Quy ra phần trăm

labels_name = [LABEL_MAPPING[i] for i in range(5)]
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=labels_name, yticklabels=labels_name, annot_kws={"size": 12})

plt.title('Normalized Confusion Matrix (IEEE Standard)', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=14)
plt.xlabel('Predicted Label', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12, rotation=0)

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Đã lưu ảnh Confusion Matrix thành 'confusion_matrix.png' (Nét căng chuẩn in PDF)")


# ====================================================================
# [VẼ HYPNOGRAM - SO SÁNH GIẤC NGỦ THỰC TẾ VÀ AI DỰ ĐOÁN]
# Chọn bừa bệnh nhân số 0 trong tập Validation
# ====================================================================

TARGET_PATIENT_IDX = 0

true_stages = patient_groups[TARGET_PATIENT_IDX]["labels"]
pred_stages = patient_groups[TARGET_PATIENT_IDX]["preds"]

epochs = np.arange(len(true_stages))

# Trick vẽ Hypnogram: Trục Y lộn ngược (Wake ở trên, N3/REM ở dưới)
# Chuyển đổi nhãn để vẽ (Wake=0, REM=1, N1=2, N2=3, N3=4) cho giống chuẩn y khoa
hypno_mapping = {0: 0, 4: -1, 1: -2, 2: -3, 3: -4}
hypno_labels = ['Wake', 'REM', 'N1', 'N2', 'N3']
hypno_ticks = [0, -1, -2, -3, -4]

plot_true = [hypno_mapping[s] for s in true_stages]
plot_pred = [hypno_mapping[s] for s in pred_stages]

fig, ax = plt.subplots(figsize=(20, 6))

# Vẽ AI đoán (Đỏ, mảnh hơn, nằm dưới)
ax.step(epochs, plot_pred, label='AI Predicted', color='red', alpha=0.7, linewidth=1.5, where='post')
# Vẽ Bác sĩ dán nhãn (Xanh, đậm, chèn lên trên)
ax.step(epochs, plot_true, label='Ground Truth (Doctor)', color='darkblue', alpha=0.8, linewidth=2.5, where='post')

ax.set_yticks(hypno_ticks)
ax.set_yticklabels(hypno_labels, fontsize=12)
ax.set_xlabel('Time (Epochs = 30s)', fontsize=14)
ax.set_ylabel('Sleep Stage', fontsize=14)
ax.set_title(f'Hypnogram Comparison - Subject #{TARGET_PATIENT_IDX} (Test Set)', fontsize=16, fontweight='bold', pad=15)
ax.legend(loc='upper right', fontsize=12, framealpha=1)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.grid(axis='x', linestyle=':', alpha=0.3)

plt.tight_layout()
plt.savefig('hypnogram_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Đã lưu ảnh Hypnogram thành 'hypnogram_comparison.png' (Nét căng chuẩn in PDF)")


