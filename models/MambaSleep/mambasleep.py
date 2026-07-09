import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNFrontEnd(nn.Module):
    """
    Compact CNN front-end for raw EEG epochs.

    Input:
        x: (B, C, 3000)

    Output:
        features: (B, d_model)
    """

    def __init__(self, in_channels=1, d_model=128, dropout=0.2, pool_bins=16):
        super().__init__()

        self.d_model = d_model
        self.pool_bins = pool_bins

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=50, stride=6, padding=24, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(inplace=True),
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(inplace=True),

            nn.Conv1d(128, 128, kernel_size=8, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(inplace=True),

            nn.AdaptiveAvgPool1d(pool_bins),
        )

        self.proj = nn.Sequential(
            nn.Linear(128 * pool_bins, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        features = self.cnn(x)
        features = features.reshape(features.size(0), -1)
        features = self.proj(features)
        return features


class MambaBlock(nn.Module):
    """
    Pure PyTorch Mamba-inspired sequence block.

    This does not require the external mamba-ssm package.
    It uses:
        LayerNorm -> gated projection -> depthwise temporal convolution
        -> diagonal selective SSM-style scan -> gated output projection.
    """

    def __init__(
        self,
        d_model=128,
        expansion=2,
        conv_kernel=3,
        dropout=0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.inner_dim = d_model * expansion

        self.norm = nn.LayerNorm(d_model)

        self.in_proj = nn.Linear(d_model, self.inner_dim * 2)

        self.depthwise_conv = nn.Conv1d(
            self.inner_dim,
            self.inner_dim,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2,
            groups=self.inner_dim,
            bias=True,
        )

        self.ssm_proj = nn.Linear(self.inner_dim, self.inner_dim * 3)

        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.inner_dim + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(self.inner_dim))

        self.out_proj = nn.Linear(self.inner_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def selective_scan(self, u, delta, b, c):
        """
        u, delta, b, c:
            (B, L, inner_dim)

        Returns:
            y: (B, L, inner_dim)
        """
        B, L, D = u.shape

        A = -torch.exp(self.A_log).view(1, 1, D)
        delta_A = torch.exp(torch.clamp(delta * A, min=-20.0, max=0.0))

        state = torch.zeros(B, D, device=u.device, dtype=u.dtype)
        outputs = []

        for t in range(L):
            alpha = delta_A[:, t, :]
            beta = 1.0 - alpha

            state = alpha * state + beta * torch.tanh(b[:, t, :])
            y_t = state * torch.sigmoid(c[:, t, :]) + self.D * u[:, t, :]

            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        return y

    def forward(self, x):
        residual = x

        x = self.norm(x)

        u, gate = self.in_proj(x).chunk(2, dim=-1)

        u = u.transpose(1, 2)
        u = self.depthwise_conv(u)
        u = u.transpose(1, 2)

        u = F.silu(u)

        delta, b, c = self.ssm_proj(u).chunk(3, dim=-1)
        delta = F.softplus(delta)

        y = self.selective_scan(u, delta, b, c)
        y = y * F.silu(gate)

        y = self.out_proj(y)
        y = self.dropout(y)

        return residual + y


class MambaSleep(nn.Module):
    """
    MambaSleep-style architecture.

    Input:
        x: (B, L, C, 3000)

    Output:
        logits: (B, L, num_classes)

    Core idea:
        CNN epoch encoder -> stacked Mamba-inspired SSM blocks -> classifier
    """

    def __init__(
        self,
        in_channels=1,
        num_classes=5,
        d_model=128,
        num_layers=4,
        expansion=2,
        dropout=0.1,
    ):
        super().__init__()

        self.d_model = d_model

        self.frontend = CNNFrontEnd(
            in_channels=in_channels,
            d_model=d_model,
            dropout=dropout,
            pool_bins=16,
        )

        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    expansion=expansion,
                    conv_kernel=3,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
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

    def forward(self, x, mask=None):
        if x.ndim != 4:
            raise ValueError(f"Expected x shape (B, L, C, T), got {tuple(x.shape)}")

        B, L, C, T = x.shape

        x = x.reshape(B * L, C, T)
        epoch_features = self.frontend(x)
        seq_features = epoch_features.reshape(B, L, self.d_model)

        if mask is not None:
            seq_features = seq_features * mask.unsqueeze(-1).float()

        for block in self.blocks:
            seq_features = block(seq_features)

            if mask is not None:
                seq_features = seq_features * mask.unsqueeze(-1).float()

        seq_features = self.final_norm(seq_features)
        logits = self.classifier(seq_features)

        return logits
