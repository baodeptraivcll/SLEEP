import torch
import torch.nn as nn


class DeepSleepNet(nn.Module):
    """
    DeepSleepNet-style architecture.

    Input:
        x: (B, L, C, 3000)

    Output:
        logits: (B, L, num_classes)

    Core idea:
        two CNN branches -> epoch features -> BiLSTM -> residual classifier
    """

    def __init__(
        self,
        in_channels=1,
        num_classes=5,
        base_channels=64,
        feature_dim=512,
        lstm_hidden=256,
        lstm_layers=2,
        dropout=0.5,
    ):
        super().__init__()

        self.feature_dim = feature_dim

        self.small_branch = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=50, stride=6, padding=24, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(dropout),

            nn.Conv1d(base_channels, base_channels * 2, kernel_size=8, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),

            nn.Conv1d(base_channels * 2, base_channels * 2, kernel_size=8, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),

            nn.Conv1d(base_channels * 2, base_channels * 2, kernel_size=8, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
        )

        self.large_branch = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=400, stride=50, padding=200, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(dropout),

            nn.Conv1d(base_channels, base_channels * 2, kernel_size=6, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),

            nn.Conv1d(base_channels * 2, base_channels * 2, kernel_size=6, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),

            nn.Conv1d(base_channels * 2, base_channels * 2, kernel_size=6, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
        )

        cnn_dim = base_channels * 4

        self.feature_projector = nn.Sequential(
            nn.Linear(cnn_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.bilstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.shortcut = nn.Linear(feature_dim, lstm_hidden * 2)

        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, num_classes),
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
        small = self.small_branch(x).squeeze(-1)
        large = self.large_branch(x).squeeze(-1)
        features = torch.cat([small, large], dim=1)
        features = self.feature_projector(features)
        return features

    def forward(self, x, mask=None):
        if x.ndim != 4:
            raise ValueError(f"Expected x shape (B, L, C, T), got {tuple(x.shape)}")

        B, L, C, T = x.shape

        x = x.reshape(B * L, C, T)
        epoch_features = self._extract_epoch_features(x)
        seq_features = epoch_features.reshape(B, L, self.feature_dim)

        if mask is not None:
            seq_features = seq_features * mask.unsqueeze(-1).float()

        lstm_out, _ = self.bilstm(seq_features)
        residual = self.shortcut(seq_features)

        features = lstm_out + residual
        logits = self.classifier(features)

        return logits
