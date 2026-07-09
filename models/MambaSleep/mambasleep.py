import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNFrontEnd(nn.Module):
    """
    CNN front-end to extract features from each raw epoch (B*L, 1, 3000) -> (B*L, d_model)
    """
    def __init__(self, in_channels=1, out_channels=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=50, stride=6, padding=22), # 3000 -> 500
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8, padding=2), # 500 -> 63
            nn.Dropout(0.5),
            
            nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=3), # 63 -> 63
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=2) # 63 -> 16
        )
        self.fc = nn.Linear(128 * 16, out_channels)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        feat = self.dropout(feat)
        feat = self.fc(feat)
        return feat

class SelectiveSSM(nn.Module):
    """
    A pure PyTorch implementation of the Selective State Space (SSM) layer.
    """
    def __init__(self, d_model, d_state=16, d_inner=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_inner or d_model * 2
        
        # Projection layers
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # 1D Convolution along sequence dimension
        self.conv = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=3,
            padding=1,
            groups=self.d_inner
        )
        
        # SSM projections (selective scan inputs)
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        
        # Initialize A parameter
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # Initialize D parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
    def forward(self, x):
        # x: (B, L, d_model)
        B, L, D_model = x.shape
        
        # Project input
        projected = self.in_proj(x) # (B, L, 2 * d_inner)
        x_ssm, z = projected.chunk(2, dim=-1) # (B, L, d_inner), (B, L, d_inner)
        
        # Conv1d along sequence (L) dimension
        x_ssm = x_ssm.transpose(1, 2) # (B, d_inner, L)
        x_ssm = self.conv(x_ssm)
        x_ssm = F.silu(x_ssm)
        x_ssm = x_ssm.transpose(1, 2) # (B, L, d_inner)
        
        # Project to get dt, B, C
        x_proj_out = self.x_proj(x_ssm)
        dt, B_param, C_param = torch.split(x_proj_out, [1, self.d_state, self.d_state], dim=-1)
        
        # Compute selective dt
        dt = F.softplus(self.dt_proj(dt)) # (B, L, d_inner)
        
        # Discretization parameters
        A = -torch.exp(self.A_log)
        
        # Selective Scan (Iterate over sequence length L)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        
        for t in range(L):
            dt_t = dt[:, t, :].unsqueeze(-1) # (B, d_inner, 1)
            B_t = B_param[:, t, :].unsqueeze(1) # (B, 1, d_state)
            C_t = C_param[:, t, :].unsqueeze(1) # (B, 1, d_state)
            x_t = x_ssm[:, t, :].unsqueeze(-1) # (B, d_inner, 1)
            
            # dA_t = exp(dt_t * A) -> (B, d_inner, d_state)
            dA_t = torch.exp(dt_t * A.unsqueeze(0))
            # dB_t = dt_t * B_t -> (B, d_inner, d_state)
            dB_t = dt_t * B_t
            
            # Update state
            h = dA_t * h + dB_t * x_t
            
            # Compute output y_t = sum(C_t * h) -> (B, d_inner)
            y_t = torch.sum(h * C_t, dim=-1)
            ys.append(y_t)
            
        ys = torch.stack(ys, dim=1) # (B, L, d_inner)
        
        # Add D connection
        ys = ys + x_ssm * self.D.unsqueeze(0).unsqueeze(0)
        
        # Gate with z (residual gate)
        out = ys * F.silu(z)
        
        # Project out
        out = self.out_proj(out) # (B, L, d_model)
        return out

class MambaSleep(nn.Module):
    """
    MambaSleep Model (Mamba-based adapted raw EEG sequence model).
    Standardized to 1-step end-to-end sequence training.
    """
    def __init__(self, in_channels=1, num_classes=5, d_model=128, d_state=16):
        super().__init__()
        self.cnn_frontend = CNNFrontEnd(in_channels=in_channels, out_channels=d_model)
        self.ssm1 = SelectiveSSM(d_model=d_model, d_state=d_state)
        self.ssm2 = SelectiveSSM(d_model=d_model, d_state=d_state)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x, mask=None):
        # x shape: (B, L, 1, 3000)
        B, L, C, Length = x.shape
        x_flat = x.view(B * L, C, Length)
        
        feat_flat = self.cnn_frontend(x_flat) # (B * L, d_model)
        feat_seq = feat_flat.view(B, L, -1) # (B, L, d_model)
        
        # Apply selective SSM layers
        feat_seq = self.ssm1(feat_seq)
        feat_seq = self.ssm2(feat_seq)
        
        feat_seq = self.dropout(feat_seq)
        logits = self.classifier(feat_seq) # (B, L, 5)
        return logits
