import torch
import torch.nn as nn

class DeepSleepNet(nn.Module):
    """
    DeepSleepNet Model for Sleep Staging (Supratak et al., 2017).
    
    This implementation STRICTLY supports the 2-step training mechanism 
    described in the original paper, serving as the "Baseline" model 
    for the benchmark narrative.
    """
    def __init__(self, in_channels=1, num_classes=5):
        super(DeepSleepNet, self).__init__()
        
        # ==========================================
        # 1. REPRESENTATION LEARNING (CNNs)
        # ==========================================
        # Branch 1: Small Filter (High Frequency / Temporal focus)
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
        
        # Branch 2: Large Filter (Low Frequency / Spatial focus)
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
        
        # Flattened CNN features size
        cnn_out_features = 2688  # Exact flattened size after max pooling
        self.dropout_cnn = nn.Dropout(0.5)
        
        # Dedicated Classifier for Step 1 (Pre-training)
        self.pretrain_classifier = nn.Linear(cnn_out_features, num_classes)
        
        # ==========================================
        # 2. SEQUENCE LEARNING (BiLSTM)
        # ==========================================
        hidden_size = 512
        self.bilstm = nn.LSTM(
            input_size=cnn_out_features, 
            hidden_size=hidden_size, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.5
        )
        
        # Shortcut connection mapping
        self.shortcut = nn.Linear(cnn_out_features, hidden_size * 2)
        
        # Dedicated Classifier for Step 2 (Fine-tuning)
        self.finetune_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size * 2, num_classes)
        )

    def freeze_cnn(self):
        """Khóa toàn bộ trọng số của nhánh CNN để phục vụ Fine-tuning (Step 2)"""
        for param in self.branch1.parameters():
            param.requires_grad = False
        for param in self.branch2.parameters():
            param.requires_grad = False
            
    def unfreeze_cnn(self):
        """Mở khóa toàn bộ trọng số của nhánh CNN"""
        for param in self.branch1.parameters():
            param.requires_grad = True
        for param in self.branch2.parameters():
            param.requires_grad = True

    def _cnn_feature_extraction(self, x):
        """Hàm dùng chung để đẩy dữ liệu qua CNN"""
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        
        out1 = out1.view(out1.size(0), -1)
        out2 = out2.view(out2.size(0), -1)
        
        cnn_features = torch.cat((out1, out2), dim=1)
        cnn_features = self.dropout_cnn(cnn_features)
        return cnn_features

    def forward(self, x, mode="finetune"):
        """
        Chuyển mạch dựa trên mode:
        - "pretrain": Input (B, 1, 3000). Đầu ra lấy trực tiếp từ CNN.
        - "finetune": Input (B, Seq_Len, 1, 3000). Đầu ra qua BiLSTM.
        """
        if mode == "pretrain":
            # Input: (Batch, Channels, Length) e.g., (128, 1, 3000)
            if x.dim() == 4:
                # Nếu lỡ đưa 4D tensor với Seq=1 vào, ta bóp nó về 3D
                x = x.squeeze(1) 
            
            cnn_features = self._cnn_feature_extraction(x)
            logits = self.pretrain_classifier(cnn_features)  # (Batch, Num_Classes)
            return logits
            
        elif mode == "finetune":
            # Input: (Batch, Seq_Len, Channels, Length) e.g., (32, 20, 1, 3000)
            B, Seq_Len, C, L = x.shape
            x = x.view(B * Seq_Len, C, L)
            
            # 1. Trích xuất đặc trưng
            cnn_features = self._cnn_feature_extraction(x)
            seq_features = cnn_features.view(B, Seq_Len, -1)  # (B, Seq_Len, 2432)
            
            # 2. Xử lý qua BiLSTM
            lstm_out, _ = self.bilstm(seq_features)  # (B, Seq_Len, 1024)
            
            # 3. Kết nối Shortcut
            shortcut_out = self.shortcut(seq_features) # (B, Seq_Len, 1024)
            combined_features = lstm_out + shortcut_out
            
            # 4. Phân loại theo chuỗi
            logits = self.finetune_classifier(combined_features)  # (B, Seq_Len, Num_Classes)
            return logits
            
        else:
            raise ValueError("mode must be 'pretrain' or 'finetune'")

# ---------------------------------------------------------
# KIỂM TRA MÔ HÌNH VỚI 2 CHẾ ĐỘ (2-STEP TRAINING)
# ---------------------------------------------------------
if __name__ == "__main__":
    model = DeepSleepNet(in_channels=1, num_classes=5)
    
    print("="*40)
    print("STEP 1: PRE-TRAINING (CNN Only)")
    print("="*40)
    # Dummy data: Batch=128, Channel=1, Length=3000
    dummy_x_pretrain = torch.randn(128, 1, 3000)
    out_pretrain = model(dummy_x_pretrain, mode="pretrain")
    print("Pre-train Input shape:", dummy_x_pretrain.shape)
    print("Pre-train Output shape:", out_pretrain.shape) # Expected: (128, 5)
    
    print("\n" + "="*40)
    print("STEP 2: FINE-TUNING (CNN + BiLSTM)")
    print("="*40)
    model.freeze_cnn() # Thử đóng băng nhánh CNN
    # Dummy data: Batch=32, Seq_Len=20, Channel=1, Length=3000
    dummy_x_finetune = torch.randn(32, 20, 1, 3000)
    out_finetune = model(dummy_x_finetune, mode="finetune")
    print("Fine-tune Input shape:", dummy_x_finetune.shape)
    print("Fine-tune Output shape:", out_finetune.shape) # Expected: (32, 20, 5)
    
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nParameters Tracking:")
    print(f"- Total Frozen (CNN branch): {frozen_params}")
    print(f"- Total Trainable (BiLSTM + FC): {trainable_params}")
