import torch
import torch.nn as nn

class DeepSleepNet(nn.Module):
    """
    DeepSleepNet Model for Sleep Staging.
    Architecture based on Supratak et al. (2017).
    Expected Input: (Batch_Size, Seq_Len, Channels=1, Signal_Length=3000)
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
        
        # Calculate Flattened Size Dynamically
        # Input: 3000
        # B1_Conv1: (3000-50)/6 + 1 = 492 -> MaxPool1: 492/8 = 61
        # B1_Conv2-4 (same padding): 61 -> MaxPool2: 61/4 = 15
        # B1 Out: 128 * 15 = 1920
        # 
        # B2_Conv1: (3000-400)/50 + 1 = 53 -> MaxPool1: 53/4 = 13
        # B2_Conv2-4 (same padding): 13 -> MaxPool2: 13/2 = 6
        # B2 Out: 128 * 6 = 768
        # Total CNN Features: 1920 + 768 = 2688
        cnn_out_features = 2688
        
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
        
        # Final Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size * 2, num_classes)
        )

    def forward(self, x):
        """
        x shape: (B, Seq_Len, C, L) e.g., (32, 20, 1, 3000)
        """
        # Save original shape
        B, Seq_Len, C, L = x.shape
        
        # Flatten B and Seq_Len to process through CNN epoch-wise
        x = x.view(B * Seq_Len, C, L)
        
        # Pass through CNN branches
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        
        # Flatten and Concatenate
        out1 = out1.view(out1.size(0), -1)
        out2 = out2.view(out2.size(0), -1)
        cnn_features = torch.cat((out1, out2), dim=1)  # Shape: (B * Seq_Len, 2688)
        cnn_features = self.dropout_cnn(cnn_features)
        
        # Reshape for Sequence Learning
        seq_features = cnn_features.view(B, Seq_Len, -1)  # Shape: (B, Seq_Len, 2688)
        
        # Process BiLSTM
        lstm_out, _ = self.bilstm(seq_features)  # Shape: (B, Seq_Len, 1024)
        
        # Process Shortcut Connection
        shortcut_out = self.shortcut(seq_features) # Shape: (B, Seq_Len, 1024)
        
        # Add residual/shortcut connection
        combined_features = lstm_out + shortcut_out
        
        # Final Classification
        # We classify each epoch in the sequence
        logits = self.classifier(combined_features)  # Shape: (B, Seq_Len, Num_Classes)
        
        return logits

# Testing the model output shape
if __name__ == "__main__":
    model = DeepSleepNet(in_channels=1, num_classes=5)
    # Dummy data: Batch=4, Seq=10, Channel=1, Length=3000
    dummy_x = torch.randn(4, 10, 1, 3000)
    out = model(dummy_x)
    print("Model Input:", dummy_x.shape)
    print("Model Output:", out.shape) # Expected: (4, 10, 5)
    print("Number of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
