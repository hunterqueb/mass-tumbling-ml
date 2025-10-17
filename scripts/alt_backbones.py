#!/usr/bin/env python3
# alt_backbones.py
# Three alternative sequence encoders sharing the same estimator head + API:
# 1) TransformerEncoderEstimator
# 2) TCNEstimator (dilated 1D CNN)
# 3) BiLSTMEstimator
#
# Usage with your existing trainer:
#   from alt_backbones import build_estimator
#   model = build_estimator(kind="transformer", d_model=192, n_layers=4)
#   trainer = InertiaTrainer(model, cfg)  # unchanged

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------ shared utilities ------------------

def _spd_from_head(h):
    """MLP -> Cholesky -> SPD I with trace=1. h: (B,D)"""
    head = getattr(_spd_from_head, "_head", None)
    if head is None or head[0] != h.size(-1):
        # bind a small MLP once per D
        D = h.size(-1)
        _spd_from_head._head = (D, nn.Sequential(
            nn.Linear(D, 128), nn.SiLU(),
            nn.Linear(128, 6)
        ).to(h.device))
        head = _spd_from_head._head
    mlp = head[1]
    p = mlp(h)  # (B,6)
    l11, l21, l22, l31, l32, l33 = p.split(1, dim=-1)
    l11 = F.softplus(l11) + 1e-4
    l22 = F.softplus(l22) + 1e-4
    l33 = F.softplus(l33) + 1e-4
    B = h.size(0)
    L = torch.zeros(B,3,3, device=h.device, dtype=h.dtype)
    L[:,0,0] = l11.squeeze(-1)
    L[:,1,0] = l21.squeeze(-1); L[:,1,1] = l22.squeeze(-1)
    L[:,2,0] = l31.squeeze(-1); L[:,2,1] = l32.squeeze(-1); L[:,2,2] = l33.squeeze(-1)
    I = L @ L.transpose(-1,-2)
    tr = torch.clamp(torch.diagonal(I, dim1=-2, dim2=-1).sum(-1, keepdim=True), 1e-8)
    return I / tr.unsqueeze(-1)

class _BaseEstimator(nn.Module):
    """Base class: packs [w,q]->proj->encoder->pool->SPD head."""
    def __init__(self, d_model=192, pool="mean"):
        super().__init__()
        self.input_dim = 7
        self.proj_in = nn.Linear(self.input_dim, d_model)
        self.d_model = d_model
        self.pool = pool

    def pool_tokens(self, z):
        if self.pool == "mean":
            return z.mean(dim=1)
        elif self.pool == "last":
            return z[:, -1, :]
        else:
            raise ValueError("pool must be 'mean' or 'last'")

    def head(self, h):
        return _spd_from_head(h)

    def forward(self, w, q):
        raise NotImplementedError

# ------------------ 1) Transformer encoder ------------------

class TransformerEncoderEstimator(_BaseEstimator):
    def __init__(self, d_model=192, n_layers=4, n_heads=6, d_ff=4*192, dropout=0.1, pool="mean"):
        super().__init__(d_model=d_model, pool=pool)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, w, q):
        x = torch.cat([w, q], dim=-1)        # (B,T,7)
        z = self.proj_in(x)                  # (B,T,D)
        z = self.encoder(z)                  # (B,T,D)
        h = self.pool_tokens(z)              # (B,D)
        I = self.head(h)                     # (B,3,3) SPD, trace=1
        return I, h, z

# ------------------ 2) Temporal Convolutional Network (TCN) ------------------

class Chomp1d(nn.Module):
    def __init__(self, chomp_size): super().__init__(); self.chomp_size = chomp_size
    def forward(self, x): return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad), nn.GELU(),
            nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad), nn.GELU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        out = self.net(x)
        res = self.downsample(x)
        return self.act(out + res)

class TCNEstimator(_BaseEstimator):
    def __init__(self, d_model=192, n_layers=6, kernel_size=5, dropout=0.1, pool="mean"):
        super().__init__(d_model=d_model, pool=pool)
        layers = []
        for i in range(n_layers):
            dilation = 2 ** i
            in_ch = d_model if i > 0 else d_model
            layers.append(TemporalBlock(in_ch, d_model, kernel_size, dilation, dropout))
        self.tcn = nn.Sequential(*layers)

    def forward(self, w, q):
        x = torch.cat([w, q], dim=-1)        # (B,T,7)
        z = self.proj_in(x)                  # (B,T,D)
        z1 = z.transpose(1, 2)               # (B,D,T)
        z1 = self.tcn(z1)                    # (B,D,T)
        z = z1.transpose(1, 2)               # (B,T,D)
        h = self.pool_tokens(z)
        I = self.head(h)
        return I, h, z

# ------------------ 3) BiLSTM ------------------

class BiLSTMEstimator(_BaseEstimator):
    def __init__(self, d_model=192, n_layers=2, dropout=0.1, pool="mean"):
        super().__init__(d_model=d_model, pool=pool)
        self.lstm = nn.LSTM(
            input_size=d_model, hidden_size=d_model//2, num_layers=n_layers,
            dropout=dropout, bidirectional=True, batch_first=True
        )
        self.proj_out = nn.Linear(d_model, d_model)

    def forward(self, w, q):
        x = torch.cat([w, q], dim=-1)        # (B,T,7)
        z = self.proj_in(x)                  # (B,T,D)
        z, _ = self.lstm(z)                  # (B,T,D) (bi=2 * D/2)
        z = self.proj_out(z)                 # (B,T,D)
        h = self.pool_tokens(z)
        I = self.head(h)
        return I, h, z

# ------------------ factory ------------------

def build_estimator(kind="transformer", **kwargs):
    kind = kind.lower()
    if kind == "transformer":
        return TransformerEncoderEstimator(**kwargs)
    if kind == "tcn":
        return TCNEstimator(**kwargs)
    if kind == "bilstm":
        return BiLSTMEstimator(**kwargs)
    raise ValueError(f"unknown kind: {kind}")
