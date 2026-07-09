import torch
import torch.nn as nn

class DeepSleepNet(nn.Module):
    """
    DeepSleepNet Model for Sleep Staging (Supratak et al., 2017).
    Standardized to 1-step end-to-end sequence training.
    """
    def __init__(self, in_channels=1, num_classes=5):
        super(DeepSleepNet, self).__init__()
        
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
        
        # Sequence Learning (BiLSTM)
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
        
        # Classifier
        self.classifier = nn.Sequential(
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

    def forward(self, x, mask=None):
        # Input shape: (B, L, 1, 3000)
        B, L, C, Length = x.shape
        x_flat = x.view(B * L, C, Length)
        
        # 1. Feature extraction
        cnn_features = self._cnn_feature_extraction(x_flat) # (B * L, 2688)
        seq_features = cnn_features.view(B, L, -1) # (B, L, 2688)
        
        # 2. BiLSTM
        lstm_out, _ = self.bilstm(seq_features) # (B, L, 1024)
        
        # 3. Shortcut
        shortcut_out = self.shortcut(seq_features) # (B, L, 1024)
        combined_features = lstm_out + shortcut_out
        
        # 4. Classifier
        logits = self.classifier(combined_features) # (B, L, 5)
        return logits
