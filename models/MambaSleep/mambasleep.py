"""
Fast WaveMamba-style adaptation for the unified SLEEP benchmark.

Drop-in replacement for:
    models/MambaSleep/mambasleep.py

Interface (same as the other models in the repository):
    input  x      : (B, L, C, 3000)
    input  mask   : (B, L) or None
    output logits : (B, L, num_classes)

Design goals:
- Preserve the main WaveMamba ideas: db4 wavelet decomposition, five
  frequency branches, cross-band attention, and temporal Mamba-style blocks.
- Remove the expensive 6 Mamba layers inside every frequency branch.
- Use only PyTorch; no pywt, ptwt, or mamba_ssm dependency is required.
- Remain stable with CUDA AMP by evaluating the recurrent selective scan in
  float32 while returning tensors in the original dtype.

This is an adapted/lightweight WaveMamba-style model, not an exact
reproduction of the original WaveMamba paper implementation.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedDB4DWT(nn.Module):
    """GPU-friendly 4-level db4 analysis filter bank implemented in PyTorch.

    A 4-level decomposition produces five tensors:
        A4, D4, D3, D2, D1

    At 100 Hz these approximately correspond to:
        delta, theta, alpha, sigma/beta, gamma.

    The filters are fixed buffers, so no wavelet package is needed and the
    operation moves automatically with model.to(device).
    """

    def __init__(self, levels: int = 4) -> None:
        super().__init__()
        if levels != 4:
            raise ValueError("This adapted model expects levels=4 to create 5 bands.")
        self.levels = levels

        # Daubechies-4 decomposition filters. Conv1d performs correlation,
        # therefore the PyWavelets analysis coefficients are reversed here.
        dec_lo = [
            -0.010597401785069032,
            0.0328830116668852,
            0.030841381835560764,
            -0.18703481171888114,
            -0.027983769416859854,
            0.6308807679298587,
            0.7148465705529154,
            0.2303778133088965,
        ]
        dec_hi = [
            -0.2303778133088965,
            0.7148465705529154,
            -0.6308807679298587,
            -0.027983769416859854,
            0.18703481171888114,
            0.030841381835560764,
            -0.0328830116668852,
            -0.010597401785069032,
        ]

        low = torch.tensor(dec_lo[::-1], dtype=torch.float32).view(1, 1, -1)
        high = torch.tensor(dec_hi[::-1], dtype=torch.float32).view(1, 1, -1)
        self.register_buffer("low_filter", low, persistent=True)
        self.register_buffer("high_filter", high, persistent=True)

    @staticmethod
    def _safe_pad(x: torch.Tensor) -> torch.Tensor:
        # Asymmetric 7-sample padding keeps lengths close to exact halving.
        # Reflection is safe here because all sleep-epoch tensors are long.
        return F.pad(x, (3, 4), mode="reflect")

    def _analysis_step(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        channels = x.size(1)
        low = self.low_filter.repeat(channels, 1, 1)
        high = self.high_filter.repeat(channels, 1, 1)
        x = self._safe_pad(x)
        approx = F.conv1d(x, low, stride=2, groups=channels)
        detail = F.conv1d(x, high, stride=2, groups=channels)
        return approx, detail

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"DWT expects (N, C, T), got {tuple(x.shape)}")

        approx = x
        details: List[torch.Tensor] = []
        for _ in range(self.levels):
            approx, detail = self._analysis_step(approx)
            details.append(detail)

        # Low-to-high frequency order: A4, D4, D3, D2, D1.
        return [approx, details[3], details[2], details[1], details[0]]


class FrequencyBranch(nn.Module):
    """Very small CNN that converts one wavelet band into one token."""

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        hidden_channels: Sequence[int] = (24, 48),
        pool_bins: int = 4,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if len(hidden_channels) != 2:
            raise ValueError("hidden_channels must contain exactly two values.")

        c1, c2 = int(hidden_channels[0]), int(hidden_channels[1])
        if c1 % 6 != 0 or c2 % 8 != 0:
            raise ValueError("Default GroupNorm requires c1 % 6 == 0 and c2 % 8 == 0.")

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=7, stride=4, padding=3, bias=False),
            nn.GroupNorm(6, c1),
            nn.SiLU(inplace=True),
            nn.Conv1d(c1, c2, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, c2),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool1d(pool_bins),
        )
        self.project = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(c2 * pool_bins, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.encoder(x))


class CrossBandFusion(nn.Module):
    """Attention-based fusion of the five frequency tokens."""

    def __init__(
        self,
        d_model: int,
        num_bands: int = 5,
        num_heads: int = 4,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.band_embedding = nn.Parameter(torch.zeros(1, num_bands, d_model))
        nn.init.trunc_normal_(self.band_embedding, std=0.02)

        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.pool_score = nn.Linear(d_model, 1, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B*L, 5, d_model)
        tokens = tokens + self.band_embedding

        normed = self.norm1(tokens)
        attended, _ = self.attention(
            normed,
            normed,
            normed,
            need_weights=False,
        )
        tokens = tokens + attended
        tokens = tokens + self.ffn(self.norm2(tokens))

        # Learn which frequency branches matter for each epoch.
        weights = torch.softmax(self.pool_score(tokens).squeeze(-1), dim=1)
        return torch.sum(tokens * weights.unsqueeze(-1), dim=1)


class TemporalMambaBlock(nn.Module):
    """Compact pure-PyTorch Mamba-inspired temporal block.

    The scan is only over the epoch sequence (normally L=20), not over the raw
    3000 EEG samples and not inside each frequency branch. This is the main
    reason the adapted network is substantially faster than the original ZIP.
    """

    def __init__(
        self,
        d_model: int,
        expansion: int = 2,
        conv_kernel: int = 3,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if conv_kernel % 2 == 0:
            raise ValueError("conv_kernel must be odd.")

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

        # Stable negative diagonal state matrix A = -exp(A_log).
        init = torch.linspace(0.0, 3.0, self.inner_dim, dtype=torch.float32)
        self.A_log = nn.Parameter(init)
        self.D = nn.Parameter(torch.ones(self.inner_dim, dtype=torch.float32))

        self.out_proj = nn.Linear(self.inner_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_scale = nn.Parameter(torch.full((d_model,), 1e-2))

    def _selective_scan(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        # Float32 scan avoids fp16/bfloat16 recurrent underflow under AMP.
        original_dtype = u.dtype
        u32 = u.float()
        delta32 = delta.float()
        b32 = b.float()
        c32 = c.float()

        batch, length, width = u32.shape
        a = -torch.exp(self.A_log.float()).view(1, width)
        d = self.D.float().view(1, width)
        state = torch.zeros(batch, width, device=u.device, dtype=torch.float32)
        outputs = []

        for step in range(length):
            dt = torch.clamp(delta32[:, step, :], min=1e-4, max=1.0)
            alpha = torch.exp(torch.clamp(dt * a, min=-20.0, max=0.0))
            state = alpha * state + (1.0 - alpha) * torch.tanh(b32[:, step, :])
            y = state * torch.sigmoid(c32[:, step, :]) + d * u32[:, step, :]
            outputs.append(y)

        return torch.stack(outputs, dim=1).to(dtype=original_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        u, gate = self.in_proj(x).chunk(2, dim=-1)
        u = self.depthwise_conv(u.transpose(1, 2)).transpose(1, 2)
        u = F.silu(u)

        delta, b, c = self.ssm_proj(u).chunk(3, dim=-1)
        delta = F.softplus(delta)
        y = self._selective_scan(u, delta, b, c)
        y = y * F.silu(gate)
        y = self.dropout(self.out_proj(y))

        return residual + self.layer_scale * y


class MambaSleep(nn.Module):
    """Fast adapted WaveMamba-style network for the SLEEP repository.

    Architecture:
        raw EEG -> fixed db4 DWT -> 5 lightweight CNN branches
        -> cross-band attention -> 2 temporal Mamba-style blocks
        -> per-epoch classifier
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        d_model: int = 96,
        branch_channels: Sequence[int] = (24, 48),
        branch_pool_bins: int = 4,
        num_heads: int = 4,
        temporal_layers: int = 2,
        expansion: int = 2,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if temporal_layers < 1:
            raise ValueError("temporal_layers must be at least 1.")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.d_model = d_model
        self.num_bands = 5

        self.dwt = FixedDB4DWT(levels=4)
        self.branches = nn.ModuleList(
            [
                FrequencyBranch(
                    in_channels=in_channels,
                    d_model=d_model,
                    hidden_channels=branch_channels,
                    pool_bins=branch_pool_bins,
                    dropout=dropout,
                )
                for _ in range(self.num_bands)
            ]
        )
        self.fusion = CrossBandFusion(
            d_model=d_model,
            num_bands=self.num_bands,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.temporal_blocks = nn.ModuleList(
            [
                TemporalMambaBlock(
                    d_model=d_model,
                    expansion=expansion,
                    conv_kernel=3,
                    dropout=dropout,
                )
                for _ in range(temporal_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_classes),
        )

        self._init_trainable_weights()

    def _init_trainable_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Restore deliberately small residual scaling after generic init.
        for block in self.temporal_blocks:
            nn.init.constant_(block.layer_scale, 1e-2)

    @staticmethod
    def _validate_mask(mask: torch.Tensor, batch: int, length: int) -> torch.Tensor:
        if mask.shape != (batch, length):
            raise ValueError(
                f"Expected mask shape {(batch, length)}, got {tuple(mask.shape)}"
            )
        return mask.bool()

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x shape (B, L, C, T), got {tuple(x.shape)}")

        batch, length, channels, samples = x.shape
        if channels != self.in_channels:
            raise ValueError(
                f"Model was created with in_channels={self.in_channels}, "
                f"but received C={channels}."
            )
        if samples < 64:
            raise ValueError(f"EEG epoch is unexpectedly short: T={samples}")

        if mask is not None:
            mask = self._validate_mask(mask, batch, length)
            mask_f = mask.unsqueeze(-1).to(dtype=x.dtype)
        else:
            mask_f = None

        # Process every 30-s epoch independently in the spectral front-end.
        epochs = x.reshape(batch * length, channels, samples)
        bands = self.dwt(epochs)

        band_tokens = [branch(band) for branch, band in zip(self.branches, bands)]
        band_tokens = torch.stack(band_tokens, dim=1)  # (B*L, 5, d_model)
        epoch_features = self.fusion(band_tokens)
        sequence = epoch_features.reshape(batch, length, self.d_model)

        if mask_f is not None:
            sequence = sequence * mask_f

        for block in self.temporal_blocks:
            sequence = block(sequence)
            if mask_f is not None:
                sequence = sequence * mask_f

        sequence = self.final_norm(sequence)
        logits = self.classifier(sequence)

        # Masked logits are not used by the shared loss/evaluator, but zeroing
        # them makes padded behavior deterministic and easy to test.
        if mask_f is not None:
            logits = logits * mask_f

        return logits


# Backward compatibility with the user's older ZIP main.py.
WaveMamba = MambaSleep


if __name__ == "__main__":
    model = MambaSleep(in_channels=1, num_classes=5)
    dummy_x = torch.randn(2, 20, 1, 3000)
    dummy_mask = torch.ones(2, 20, dtype=torch.bool)
    dummy_mask[1, -3:] = False
    output = model(dummy_x, mask=dummy_mask)
    print("Output shape:", tuple(output.shape))
    print("Parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
