import numpy as np
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Standard sine-cosine positional encoding for Transformer inputs.
    Introduces order information to the inputs.
    """
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: (Batch, Seq_Len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x

class CNNFrontEnd(nn.Module):
    """
    Lightweight 1D CNN front-end to process raw 1D EEG signal of shape (B, 1, 3000)
    and output a 2D feature grid of shape (B, T, F) matching the SleepTransformer expected input,
    where T = 29 (time steps) and F = 128 (feature dimension / d_model).
    """
    def __init__(self, in_channels=1, out_channels=128, T=29):
        super(CNNFrontEnd, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=50, stride=6, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(0.5),
            
            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, out_channels, kernel_size=8, stride=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.pool = nn.AdaptiveAvgPool1d(T)
        
    def forward(self, x):
        # x: (B, 1, 3000)
        feat = self.features(x)  # (B, out_channels, L_out)
        feat = self.pool(feat)   # (B, out_channels, T)
        feat = feat.transpose(1, 2)  # (B, T, out_channels)
        return feat

class AttentionPooler(nn.Module):
    """
    Softmax attention pooler to reduce sequence of time-step features to a single feature vector.
    Corresponds to Equations (12)-(14) in Phan et al. (2022).
    """
    def __init__(self, d_model, attention_size=64):
        super(AttentionPooler, self).__init__()
        self.Wa = nn.Linear(d_model, attention_size)
        self.ae = nn.Parameter(torch.randn(attention_size, 1))
        
    def forward(self, x):
        # x: (Batch, Seq_Len, d_model)
        # at = tanh(Wa * xt + ba)
        at = torch.tanh(self.Wa(x))  # (Batch, Seq_Len, attention_size)
        
        # score = at * ae
        scores = torch.matmul(at, self.ae).squeeze(-1)  # (Batch, Seq_Len)
        
        # alpha = softmax(scores)
        alpha = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (Batch, Seq_Len, 1)
        
        # x_pooled = sum(alpha_t * x_t)
        x_pooled = torch.sum(alpha * x, dim=1)  # (Batch, d_model)
        return x_pooled

class SleepTransformer(nn.Module):
    """
    SleepTransformer Model (Raw-Signal Variant) for Sleep Staging (Phan et al., 2022).
    Standardized to 1-step end-to-end sequence training.
    """
    def __init__(self, in_channels=1, num_classes=5, d_model=128, nhead=8, 
                 d_ff=1024, num_epoch_layers=4, num_sequence_layers=4, 
                 dropout=0.1, attention_size=64):
        super(SleepTransformer, self).__init__()
        
        self.d_model = d_model
        
        # 1. CNN Front-End to extract grid features from raw 1D signals
        self.cnn_frontend = CNNFrontEnd(in_channels=in_channels, out_channels=d_model, T=29)
        
        # 2. Positional Encoding for epoch and sequence transformers
        self.pos_encoder_epoch = PositionalEncoding(d_model=d_model)
        self.pos_encoder_seq = PositionalEncoding(d_model=d_model)
        
        # 3. Epoch Transformer (Intra-epoch processing)
        epoch_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff, 
            dropout=dropout, activation='relu', batch_first=True
        )
        self.epoch_transformer = nn.TransformerEncoder(epoch_encoder_layer, num_layers=num_epoch_layers)
        
        # 4. Epoch Attention Pooling (collapses 29 time frames to 1 vector)
        self.attention_pool = AttentionPooler(d_model=d_model, attention_size=attention_size)
        
        # 5. Sequence Transformer (Inter-epoch processing)
        seq_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff, 
            dropout=dropout, activation='relu', batch_first=True
        )
        self.seq_transformer = nn.TransformerEncoder(seq_encoder_layer, num_layers=num_sequence_layers)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes)
        )

    def _epoch_feature_extraction(self, x):
        """Processes (B, 1, 3000) epochs and returns (B, d_model) pooled representation"""
        # CNN Front-End
        cnn_features = self.cnn_frontend(x)  # (B, T=29, F=128)
        
        # Positional Encoding
        cnn_features = self.pos_encoder_epoch(cnn_features)
        
        # Epoch Transformer Encoder
        epoch_features = self.epoch_transformer(cnn_features)  # (B, T=29, F=128)
        
        # Attention Pooling
        pooled_features = self.attention_pool(epoch_features)  # (B, F=128)
        return pooled_features

    def forward(self, x, mask=None):
        # x: (Batch, Seq_Len, 1, 3000)
        B, Seq_Len, C, L = x.shape
        # Flatten sequence to process all epochs through the epoch encoder in parallel
        x_flat = x.view(B * Seq_Len, C, L)  # (B * Seq_Len, 1, 3000)
        
        pooled_features = self._epoch_feature_extraction(x_flat)  # (B * Seq_Len, d_model)
        
        # Reshape back to sequence
        seq_features = pooled_features.view(B, Seq_Len, self.d_model)  # (B, Seq_Len, d_model)
        
        # Positional Encoding for sequence
        seq_features = self.pos_encoder_seq(seq_features)
        
        # Sequence Transformer Encoder
        if mask is not None:
            seq_out = self.seq_transformer(seq_features, src_key_padding_mask=~mask.bool())  # (B, Seq_Len, d_model)
        else:
            seq_out = self.seq_transformer(seq_features)  # (B, Seq_Len, d_model)
        
        # Classifier
        logits = self.classifier(seq_out)  # (B, Seq_Len, num_classes)
        return logits
