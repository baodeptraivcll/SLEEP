import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sine-cosine positional encoding.
    """

    def __init__(self, d_model, max_len=512):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class AttentionPool(nn.Module):
    """
    Learnable attention pooling.
    """

    def __init__(self, d_model, hidden=64):
        super().__init__()
        self.proj = nn.Linear(d_model, hidden)
        self.score = nn.Linear(hidden, 1, bias=False)

    def forward(self, x, mask=None):
        scores = self.score(torch.tanh(self.proj(x))).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), -1e9)

        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        pooled = torch.sum(weights * x, dim=1)

        return pooled


class CNNFrontEnd(nn.Module):
    """
    Converts each raw EEG epoch into frame-level features.

    Input:
        x: (B, C, 3000)

    Output:
        features: (B, frame_count, d_model)
    """

    def __init__(self, in_channels=1, d_model=128, frame_count=29, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=50, stride=6, padding=24, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(dropout),

            nn.Conv1d(64, d_model, kernel_size=8, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),

            nn.Conv1d(d_model, d_model, kernel_size=8, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(frame_count),
        )

    def forward(self, x):
        features = self.net(x)
        features = features.transpose(1, 2)
        return features


class SleepTransformer(nn.Module):
    """
    SleepTransformer-style architecture.

    Input:
        x: (B, L, C, 3000)

    Output:
        logits: (B, L, num_classes)

    Core idea:
        CNN frame extractor -> intra-epoch Transformer -> attention pool
        -> inter-epoch Transformer -> classifier
    """

    def __init__(
        self,
        in_channels=1,
        num_classes=5,
        d_model=128,
        nhead=8,
        dim_feedforward=512,
        epoch_layers=2,
        sequence_layers=2,
        frame_count=29,
        dropout=0.1,
    ):
        super().__init__()

        self.d_model = d_model

        self.frontend = CNNFrontEnd(
            in_channels=in_channels,
            d_model=d_model,
            frame_count=frame_count,
            dropout=dropout,
        )

        self.epoch_pos = PositionalEncoding(d_model=d_model, max_len=frame_count + 8)
        self.sequence_pos = PositionalEncoding(d_model=d_model, max_len=512)

        epoch_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.epoch_encoder = nn.TransformerEncoder(
            epoch_layer,
            num_layers=epoch_layers,
        )

        self.epoch_pool = AttentionPool(d_model=d_model, hidden=d_model // 2)

        sequence_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.sequence_encoder = nn.TransformerEncoder(
            sequence_layer,
            num_layers=sequence_layers,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _extract_epoch_features(self, x):
        frame_features = self.frontend(x)
        frame_features = self.epoch_pos(frame_features)
        frame_features = self.epoch_encoder(frame_features)
        epoch_features = self.epoch_pool(frame_features)
        return epoch_features

    def forward(self, x, mask=None):
        if x.ndim != 4:
            raise ValueError(f"Expected x shape (B, L, C, T), got {tuple(x.shape)}")

        B, L, C, T = x.shape

        x = x.reshape(B * L, C, T)
        epoch_features = self._extract_epoch_features(x)
        seq_features = epoch_features.reshape(B, L, self.d_model)

        if mask is not None:
            seq_features = seq_features * mask.unsqueeze(-1).float()
            padding_mask = ~mask.bool()
        else:
            padding_mask = None

        seq_features = self.sequence_pos(seq_features)
        seq_features = self.sequence_encoder(
            seq_features,
            src_key_padding_mask=padding_mask,
        )

        logits = self.classifier(seq_features)

        return logits
