import torch
import torch.nn as nn

class TinySleepNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=5):
        super(TinySleepNet, self).__init__()
        
        # ponytail: Chuẩn hóa 100% theo bản TF gốc
        # Tính toán kích thước theo TF 'SAME' padding:
        # L_in = 3000 -> Conv1 -> 500 -> MaxPool1 -> 63
        # Conv2,3,4 -> 63 -> MaxPool2 -> 16
        # Kích thước Flatten: 16 * 128 = 2048
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=50, stride=6, padding=22), # 3000 -> 500
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8, padding=2), # 500 -> 63
            nn.Dropout(0.5),
            
            nn.Conv1d(128, 128, kernel_size=8, stride=1, padding='same'), # 63 -> 63
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Conv1d(128, 128, kernel_size=8, stride=1, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Conv1d(128, 128, kernel_size=8, stride=1, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=2) # 63 -> 16
        )
        
        # LSTM học sự chuyển tiếp giữa các Epochs (Sequence)
        # Input size = 16 * 128 = 2048
        self.lstm = nn.LSTM(input_size=2048, hidden_size=128, num_layers=1, 
                            batch_first=True, dropout=0.0) 
        
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x, mask=None):
        # x shape: (B, Seq_Len, 1, 3000)
        B, Seq_Len, C, L = x.shape
        
        # Gộp Batch và Sequence lại để tống qua CNN (CNN chỉ biết xử lý từng Epoch riêng rẽ)
        x_cnn = x.view(B * Seq_Len, C, L) # (B * Seq_Len, 1, 3000)
        
        out_cnn = self.cnn(x_cnn) # (B * Seq_Len, 128, 16)
        
        # Flatten các kênh đặc trưng của CNN
        out_cnn = out_cnn.view(B * Seq_Len, -1) # Flatten (B * Seq_Len, 2048)
        
        out_cnn = self.dropout(out_cnn)
        
        # Reshape lại để tách Batch và Sequence
        out_seq = out_cnn.view(B, Seq_Len, -1) # (B, Seq_Len, 2048)
        
        # Đưa vào LSTM
        out_lstm, _ = self.lstm(out_seq) # (B, Seq_Len, 128)
        
        out_lstm = self.dropout(out_lstm)
        
        # Tính logits cho tất cả các bước trong sequence
        logits = self.fc(out_lstm) # (B, Seq_Len, num_classes)
        
        return logits

