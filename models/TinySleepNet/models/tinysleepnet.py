import torch
import torch.nn as nn


class TinySleepNet(nn.Module):
    """
    TinySleepNet-style compact architecture.

    Input:
        x: (B, L, C, 3000)

    Output:
        logits: (B, L, num_classes)

    Core idea:
        lightweight CNN -> compact LSTM -> classifier
    """

    def __init__(
        self,
        in_channels=1,
        num_classes=5,
        cnn_channels=128,
        lstm_hidden=128,
        dropout=0.5,
        pool_bins=16,
    ):
        super().__init__()

        self.cnn_channels = cnn_channels
        self.pool_bins = pool_bins
        self.feature_dim = cnn_channels * pool_bins

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, cnn_channels, kernel_size=50, stride=6, padding=24, bias=False),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(dropout),

            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=8, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(inplace=True),

            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=8, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(inplace=True),

            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=8, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(pool_bins),
        )

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_hidden),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes),
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
        features = self.cnn(x)
        features = features.reshape(features.size(0), -1)
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

        lstm_out, _ = self.lstm(seq_features)
        logits = self.classifier(lstm_out)

        return logits
