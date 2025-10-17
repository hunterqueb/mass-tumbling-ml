#!/usr/bin/env python3
# inertia_mamba_selfsup.py
# Self-supervised identification of inertia (scale-free) using a Mamba backbone.
# Input per timestep: [omega(3), quaternion(4)]  (scalar-first quaternion)
# Output: SPD inertia matrix I with trace=1 (principal moment ratios + axes).
#
# Usage (synthetic demo):
#   python inertia_mamba_selfsup.py --epochs 30 --device cuda
#
# You need your Mamba implementation importable as:
#   from mamba_impl import Mamba, MambaConfig
# If it's in a package, adjust the import below or put this file next to it.

import math
import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- bring in your backbone ----
try:
    from qutils.ml.mamba import Mamba, MambaConfig   # adjust to your module path
except Exception as e:
    raise ImportError(
        "Cannot import Mamba/MambaConfig. Ensure your file exposes "
        "`Mamba` and `MambaConfig` as in the snippet you shared."
    ) from e

# ======================== math utils ========================

def normalize_quat(q: torch.Tensor) -> torch.Tensor:
    return q / (q.norm(dim=-1, keepdim=True) + 1e-12)

def quat_to_R(q: torch.Tensor) -> torch.Tensor:
    """q (...,4) scalar-first -> R (...,3,3)"""
    q = normalize_quat(q)
    w, x, y, z = q.unbind(-1)
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    R = torch.stack([
        torch.stack([1-2*(yy+zz), 2*(xy - wz),   2*(xz + wy)], dim=-1),
        torch.stack([2*(xy + wz), 1-2*(xx+zz),   2*(yz - wx)], dim=-1),
        torch.stack([2*(xz - wy), 2*(yz + wx),   1-2*(xx+yy)], dim=-1)
    ], dim=-2)
    return R

def gaussian_smooth_1d(x: torch.Tensor, sigma=2.0, k=9) -> torch.Tensor:
    """Depthwise 1D Gaussian smoothing over time. x: (B,T,D) -> (B,T,D)"""
    half = k//2
    t = torch.arange(-half, half+1, device=x.device, dtype=x.dtype)
    ker = torch.exp(-0.5*(t/sigma)**2); ker = ker/ker.sum()
    ker = ker.view(1,1,k)
    B,T,D = x.shape
    y = x.permute(0,2,1)  # (B,D,T)
    y = F.pad(y, (half,half), mode='replicate')
    y = F.conv1d(y, ker.repeat(D,1,1), groups=D)
    return y.permute(0,2,1)

def central_diff(x: torch.Tensor, dt: float) -> torch.Tensor:
    """Central differences over time. x: (B,T,D) -> (B,T,D)"""
    dx = torch.empty_like(x)
    dx[:,1:-1,:] = (x[:,2:,:] - x[:,:-2,:])/(2*dt)
    dx[:,:1,:]   = (x[:,1:2,:] - x[:,:1,:])/dt
    dx[:,-1:,:]  = (x[:,-1:,:] - x[:,-2:-1,:])/dt
    return dx

def as_scalar_dt(dt):
    # Accept float/int, tensor, list/tuple; return a Python float
    import torch
    if isinstance(dt, (float, int)):
        return float(dt)
    if isinstance(dt, (list, tuple)):
        return float(dt[0])
    if torch.is_tensor(dt):
        if dt.numel() == 1:
            return dt.item()
        return dt.reshape(-1)[0].item()
    return float(dt)


@torch.no_grad()
def _project_spd(I, min_eig=1e-6):
    """
    Batch project to SPD:
      1) symmetrize
      2) clamp eigenvalues to >= min_eig
      3) renormalize trace to 1
      4) replace non-finite with I/3
    I: (N,3,3)
    """
    I = I.clone()
    # sanitize
    finite = torch.isfinite(I).all(dim=(-2,-1), keepdim=True)
    I[~finite.expand_as(I)] = 0.0

    # symmetrize
    I = 0.5 * (I + I.transpose(-1,-2))

    # eig
    evals, evecs = torch.linalg.eigh(I.double())   # better conditioning in float64
    evals = evals.float(); evecs = evecs.float()

    # clamp
    evals = torch.clamp(evals, min=min_eig)

    # reconstruct
    I_spd = evecs @ torch.diag_embed(evals) @ evecs.transpose(-1,-2)

    # trace normalize
    tr = torch.diagonal(I_spd, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
    I_spd = I_spd / (tr + 1e-12)

    # final sanitize (if something still blew up)
    mask_bad = ~torch.isfinite(I_spd).all(dim=(-2,-1), keepdim=True)
    if mask_bad.any():
        eye = torch.eye(3, device=I_spd.device, dtype=I_spd.dtype).unsqueeze(0) / 3.0
        I_spd[mask_bad.expand_as(I_spd)] = eye.expand(mask_bad.sum(), 3, 3)

    return I_spd

@torch.no_grad()
def eig_ratio_axis_metrics(I_pred, I_true, min_eig=1e-6):
    """
    Safe eigenvalue ratio error and axis alignment.
    Both inputs projected to SPD with trace=1 before eig.
    """
    I_pred = _project_spd(I_pred, min_eig=min_eig)
    I_true = _project_spd(I_true, min_eig=min_eig)

    ep, Up = torch.linalg.eigh(I_pred.double())
    et, Ut = torch.linalg.eigh(I_true.double())
    ep = ep.float(); Up = Up.float(); et = et.float(); Ut = Ut.float()

    ep, _ = torch.sort(ep, dim=-1)
    et, _ = torch.sort(et, dim=-1)
    ratio_err = torch.linalg.norm(ep - et, dim=-1).mean().item()

    # greedy axis alignment score in |dot| space
    scores = []
    for i in range(I_pred.shape[0]):
        M = torch.abs(Up[i].T @ Ut[i])  # 3x3
        s = 0.0
        used_r = set(); used_c = set()
        for _ in range(3):
            r = torch.argmax(M.max(dim=1).values).item()
            c = torch.argmax(M[r]).item()
            s += float(M[r, c])
            M[r, :] = -1; M[:, c] = -1
        scores.append(s / 3.0)
    axis_score = float(np.mean(scores))
    return ratio_err, axis_score


# ======================== synthetic torque-free sim ========================

def sample_inertia(batch, device, min_ratio=0.05):
    # ensure eigenvalues not arbitrarily small (≥ min_ratio of trace)
    alphas = torch.tensor([2.0,2.0,2.0], device=device)
    g = torch.distributions.Gamma(alphas, torch.ones_like(alphas))
    eig = g.sample((batch,))
    eig = eig / eig.sum(dim=-1, keepdim=True)
    eig = torch.clamp(eig, min=min_ratio)
    eig = eig / eig.sum(dim=-1, keepdim=True)
    M = torch.randn(batch,3,3, device=device)
    Q,_ = torch.linalg.qr(M); det = torch.linalg.det(Q)
    Q[det<0,:,0] = -Q[det<0,:,0]
    return Q @ torch.diag_embed(eig) @ Q.transpose(-1,-2)

def omega_to_quatdot(q: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """qdot = 0.5 * q * [0, w] (Hamilton product, scalar-first)."""
    w0 = torch.zeros_like(w[..., :1])
    wq = torch.cat([w0, w], dim=-1)
    # Hamilton product q * wq:
    w1,x1,y1,z1 = q.unbind(-1)
    w2,x2,y2,z2 = wq.unbind(-1)
    return 0.5*torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=-1)

@torch.no_grad()
def simulate_torque_free(I: torch.Tensor, q0: torch.Tensor, w0: torch.Tensor, T: float, dt: float,
                         noise_std: float=0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Semi-implicit Euler for w, first-order for q. Returns q:(S,4), w:(S,3) with S=int(T/dt)."""
    device = I.device
    steps = int(T/dt)
    qs = torch.empty(steps, 4, device=device)
    ws = torch.empty(steps, 3, device=device)

    q = q0
    w = w0
    Iw = I @ w.unsqueeze(-1)
    for t in range(steps):
        qs[t] = q
        ws[t] = w
        # torque-free Euler: I wdot = (I w) x w
        torque_like = torch.cross(Iw.squeeze(-1), w, dim=-1)
        wdot = torch.linalg.solve(I, torque_like)
        w = w + dt*wdot
        Iw = I @ w.unsqueeze(-1)
        q = q + dt*omega_to_quatdot(q, w)
        q = normalize_quat(q)

    if noise_std > 0:
        ws = ws + noise_std*torch.randn_like(ws)
    return qs, ws

class TorqueFreeDataset(torch.utils.data.Dataset):
    def __init__(self, N=2048, T=4.0, dt=0.01, device="cpu",
                 w0_mag_range=(0.2, 2.0), noise_std=0.002):
        self.N=N; self.T=T; self.dt=dt; self.device=device
        self.steps = int(T/dt)
        self.I_true=[]; self.q=[]; self.w=[]
        for _ in range(N):
            I = sample_inertia(1, device=device)[0]
            axis = torch.randn(3, device=device); axis = axis/axis.norm()
            ang = torch.rand((), device=device)*2*math.pi
            q0 = torch.tensor([math.cos(ang/2), *(math.sin(ang/2)*axis)], device=device)
            mag = torch.empty((), device=device).uniform_(*w0_mag_range)
            v = torch.randn(3, device=device); v = v / (v.norm()+1e-9)
            w0 = mag * v
            q, w = simulate_torque_free(I, q0, w0, T, dt, noise_std=noise_std)
            self.I_true.append(I.cpu()); self.q.append(q.cpu()); self.w.append(w.cpu())
            print(f"Generated sample {len(self.I_true)}/{N}", end='\r')
        self.I_true = torch.stack(self.I_true)  # (N,3,3)
        self.q = torch.stack(self.q)            # (N,S,4)
        self.w = torch.stack(self.w)            # (N,S,3)
        
    def __len__(self): return self.N
    def __getitem__(self, i):
        return self.q[i], self.w[i], self.I_true[i], self.dt

# ======================== model ========================

class InertiaHead(nn.Module):
    """Sequence embedding -> SPD inertia with trace=1 (Cholesky)."""
    def __init__(self, d_in, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.SiLU(),
            nn.Linear(hidden, 6)
        )

    def forward(self, h):
        p = self.net(h)  # (B,6)
        l11, l21, l22, l31, l32, l33 = p.split(1, dim=-1)
        l11 = F.softplus(l11)+1e-4
        l22 = F.softplus(l22)+1e-4
        l33 = F.softplus(l33)+1e-4
        B = h.size(0)
        L = torch.zeros(B,3,3, device=h.device, dtype=h.dtype)
        L[:,0,0] = l11.squeeze(-1)
        L[:,1,0] = l21.squeeze(-1); L[:,1,1] = l22.squeeze(-1)
        L[:,2,0] = l31.squeeze(-1); L[:,2,1] = l32.squeeze(-1); L[:,2,2] = l33.squeeze(-1)
        I = L @ L.transpose(-1,-2)
        tr = torch.clamp(torch.diagonal(I, dim1=-2, dim2=-1).sum(-1, keepdim=True), 1e-8)
        return I / tr.unsqueeze(-1)  # trace=1

class InertiaMambaEstimator(nn.Module):
    """
    Backbone: Mamba. Input per timestep: [omega(3), quaternion(4)] => 7 dims.
    Pool tokens -> embedding -> Cholesky head -> I (trace=1).
    """
    def __init__(self, d_model=192, n_layers=6, d_state=16, expand=2, d_conv=4,
                 pool='mean'):
        super().__init__()
        self.input_dim = 7
        cfg = MambaConfig(
            d_model=d_model, n_layers=n_layers,
            d_state=d_state, expand_factor=expand, d_conv=d_conv,
            dt_rank='auto', dt_min=1e-3, dt_max=1e-1, dt_init='random',
            bias=False, conv_bias=True, pscan=True, classifer=False
        )
        self.backbone = Mamba(cfg)
        self.proj_in = nn.Linear(self.input_dim, d_model)
        self.pool = pool
        self.head = InertiaHead(d_model)

    def forward(self, w: torch.Tensor, q: torch.Tensor):
        """
        w: (B,T,3) body angular velocity
        q: (B,T,4) scalar-first quaternion
        Returns: I (B,3,3), embedding (B,D), tokens (B,T,D)
        """
        x = torch.cat([w, q], dim=-1)         # (B,T,7)
        z = self.proj_in(x)                   # (B,T,D)
        z = self.backbone(z)                  # (B,T,D)
        if self.pool == 'mean':
            h = z.mean(dim=1)
        elif self.pool == 'last':
            h = z[:,-1,:]
        else:
            raise ValueError("pool must be {'mean','last'}")
        I = self.head(h)                      # (B,3,3)
        return I, h, z
    
class InertiaEncoder(nn.Module):
    """
    Input: window of (omega_t, rotation matrix R_t)
    Output: Cholesky factors of SPD inertia, normalized to trace=1
    """
    def __init__(self, T, use_R=True, hidden=128):
        super().__init__()
        self.use_R = use_R
        d_in = 3  # omega
        if use_R: d_in += 9
        self.feat = nn.Sequential(
            nn.Conv1d(d_in, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # temporal pooling
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 6)  # L lower-tri entries: l11,l21,l22,l31,l32,l33
        )

    def forward(self, w, R):
        # w: (B,T,3), R: (B,T,3,3)
        B,T,_ = w.shape
        x = [w]
        if self.use_R:
            x.append(R.reshape(B,T,9))
        x = torch.cat(x, dim=-1).permute(0,2,1)  # (B,d,T)
        h = self.feat(x)                          # (B,64,1)
        pars = self.head(h)                       # (B,6)

        l11, l21, l22, l31, l32, l33 = pars.split(1, dim=-1)
        # Diagonals positive via softplus
        l11 = torch.nn.functional.softplus(l11) + 1e-4
        l22 = torch.nn.functional.softplus(l22) + 1e-4
        l33 = torch.nn.functional.softplus(l33) + 1e-4
        L = torch.zeros(B,3,3, device=w.device, dtype=w.dtype)
        L[:,0,0] = l11.squeeze(-1)
        L[:,1,0] = l21.squeeze(-1)
        L[:,1,1] = l22.squeeze(-1)
        L[:,2,0] = l31.squeeze(-1)
        L[:,2,1] = l32.squeeze(-1)
        L[:,2,2] = l33.squeeze(-1)
        I = L @ L.transpose(-1,-2)               # SPD
        # normalize scale: trace=1
        tr = torch.clamp(torch.diagonal(I, dim1=-2, dim2=-1).sum(-1, keepdim=True), 1e-8)
        I = I / tr.unsqueeze(-1)
        return I  # (B,3,3)
# ======================== physics self-supervised loss ========================

class PhysicsLoss(nn.Module):
    def __init__(self, lam_energy=0.1, lam_dyn=1.0, smooth_sigma=2.5, smooth_k=9):
        super().__init__()
        self.lamE = lam_energy
        self.lamD = lam_dyn
        self.sigma = smooth_sigma
        self.k = 9  # use 9 or 11

    def forward(self, q, w, I, dt):
        # cast to float64 for all physics; keep model in fp32
        q64 = q.double(); w64 = w.double(); I64 = I.double()
        dt = float(dt)

        # smoothing + diff
        w_s = gaussian_smooth_1d(w64, sigma=self.sigma, k=self.k)
        wdot = central_diff(w_s, dt)

        # bound magnitudes to kill outliers
        w_s   = torch.clamp(w_s,   min=-50.0, max=50.0)
        wdot  = torch.clamp(wdot,  min=-200.0, max=200.0)

        R = quat_to_R(q64)
        Iw = (I64[:,None,:,:] @ w_s[...,None]).squeeze(-1)

        Linert = (R @ Iw[...,None]).squeeze(-1)
        Lmean  = Linert.mean(dim=1, keepdim=True)
        loss_L = ((Linert - Lmean)**2).sum(dim=-1)

        E      = (w_s * Iw).sum(dim=-1) * 0.5
        # variance that ignores NaNs/Infs
        E = torch.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0)
        loss_E = E.var(dim=1, unbiased=False)

        dyn    = (I64[:,None,:,:] @ wdot[...,None]).squeeze(-1) + torch.cross(w_s, Iw, dim=-1)
        dyn    = torch.nan_to_num(dyn, nan=0.0, posinf=0.0, neginf=0.0)
        loss_D = (dyn**2).sum(dim=-1)

        # sanitize per-sequence mean, then mean over batch
        loss_L = torch.nan_to_num(loss_L.mean(dim=1), nan=0.0).mean()
        loss_E = torch.nan_to_num(loss_E,              nan=0.0).mean()
        loss_D = torch.nan_to_num(loss_D.mean(dim=1), nan=0.0).mean()

        loss = (loss_L + self.lamE*loss_E + self.lamD*loss_D).float()
        return loss, {'L_const': loss_L.float(), 'E_const': loss_E.float(), 'Euler': loss_D.float()}

# ======================== trainer ========================

@dataclass
class TrainCfg:
    lr: float = 3e-4
    wd: float = 1e-4
    lam_energy: float = 0.1
    lam_dyn: float = 1.0
    device: str = 'cuda'

class InertiaTrainer:
    def __init__(self, model: InertiaMambaEstimator, cfg: TrainCfg):
        self.model = model.to(cfg.device)
        self.loss_fn = PhysicsLoss(cfg.lam_energy, cfg.lam_dyn).to(cfg.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
        self.device = cfg.device

    def step(self, batch):
        q, w, _, dt = batch
        q = q.to(self.device); w = w.to(self.device)
        I, _, _ = self.model(w, q)
        dt_val = as_scalar_dt(dt)
        loss, terms = self.loss_fn(q, w, I, dt_val)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        return float(loss.detach()), {k: float(v) for k,v in terms.items()}
    @torch.no_grad()
    def infer(self, q: torch.Tensor, w: torch.Tensor):
        q = q.to(self.device); w = w.to(self.device)
        I, h, z = self.model(w, q)
        return I, h, z




def project_spd(I, eps=1e-8):
    """Symmetrize, floor eigenvalues, renormalize trace=1. Accepts np or torch (3x3)."""
    if torch.is_tensor(I):
        A = 0.5 * (I + I.T)
        w, V = torch.linalg.eigh(A.double())
        w = torch.clamp(w, min=eps).float()
        A = (V.float() @ torch.diag_embed(w) @ V.float().T)
        tr = torch.diagonal(A).sum()
        return (A / (tr + 1e-12)).float()
    else:
        A = 0.5 * (I + I.T)
        w, V = np.linalg.eigh(A.astype(np.float64))
        w = np.clip(w, eps, None)
        A = (V @ np.diag(w) @ V.T).astype(np.float64)
        tr = np.trace(A)
        return (A / (tr + 1e-12)).astype(np.float64)

def principal_inertia_comparison(I_pred, I_true):
    """
    Inputs: 3x3 inertia matrices (torch or np). Returns dict with:
      - evals_pred/true (sorted), abs/rel errors
      - axis assignment (greedy |dot|), per-axis cosines
      - frame misalignment angle (rad, deg)
      - rotation aligning pred principal frame to true
    """
    # Convert to numpy for simplicity
    if torch.is_tensor(I_pred): I_pred = I_pred.detach().cpu().numpy()
    if torch.is_tensor(I_true): I_true = I_true.detach().cpu().numpy()

    # SPD project (guards numerical junk) and enforce trace=1
    I_pred = project_spd(I_pred)
    I_true = project_spd(I_true)

    # Eigen-decompose and sort by ascending eigenvalue
    evals_p, evecs_p = np.linalg.eigh(I_pred)
    evals_t, evecs_t = np.linalg.eigh(I_true)
    idx_p = np.argsort(evals_p); idx_t = np.argsort(evals_t)
    evals_p = evals_p[idx_p]; U_p = evecs_p[:, idx_p]
    evals_t = evals_t[idx_t]; U_t = evecs_t[:, idx_t]

    # Greedy axis matching by |dot| (handles 3! permutations and sign flips approximately)
    M = np.abs(U_p.T @ U_t)  # 3x3
    # Greedy selection
    used_r, used_c = set(), set()
    match = [-1, -1, -1]  # pred axis i -> true axis match[i]
    for _ in range(3):
        i = int(np.argmax(M.max(axis=1)))          # best pred row
        j = int(np.argmax(M[i, :]))                # best true col
        while i in used_r:
            # pick next best row
            row_scores = M.max(axis=1)
            row_scores[list(used_r)] = -1
            i = int(np.argmax(row_scores))
            j = int(np.argmax(M[i, :]))
        while j in used_c:
            M[i, j] = -1
            j = int(np.argmax(M[i, :]))
        match[i] = j
        used_r.add(i); used_c.add(j)
        M[i, :] = -1; M[:, j] = -1

    # Reorder true frame to matched order and apply sign to maximize alignment
    U_t_m = np.zeros_like(U_t)
    cos_axes = np.zeros(3)
    for i in range(3):
        j = match[i]
        v = U_t[:, j]
        s = np.sign(np.dot(U_p[:, i], v))  # choose sign to maximize dot
        U_t_m[:, i] = s * v
        cos_axes[i] = np.abs(np.dot(U_p[:, i], v))

    # Rotation from pred-principal frame to true-principal frame
    # Columns are basis vectors; R = U_t_m * U_p^T
    R_pt = U_t_m @ U_p.T
    # Clamp trace to valid range for acos
    tr = np.trace(R_pt)
    tr = np.clip(tr, -1.0, 3.0)
    theta_rad = np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))
    theta_deg = np.degrees(theta_rad)

    # Eigenvalue errors (moments). Both traces are 1 by construction -> values are ratios.
    abs_err = np.abs(evals_p - evals_t)
    rel_err = abs_err / (np.maximum(evals_t, 1e-12))

    return {
        "evals_pred": evals_p,
        "evals_true": evals_t,
        "abs_err": abs_err,
        "rel_err": rel_err,
        "axis_cosines": cos_axes,        # per matched axis |cos(angle)|
        "axis_alignment_mean": float(cos_axes.mean()),
        "R_pred_to_true": R_pt,          # 3x3 rotation matrix
        "frame_angle_rad": float(theta_rad),
        "frame_angle_deg": float(theta_deg),
        "match_indices": match           # pred axis i corresponds to true axis match[i]
    }



# ======================== CLI demo ========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--T', type=float, default=4.0)
    ap.add_argument('--dt', type=float, default=0.01)
    ap.add_argument('--trainN', type=int, default=2000)
    ap.add_argument('--valN', type=int, default=300)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--lamE', type=float, default=0.1)
    ap.add_argument('--lamD', type=float, default=1.0)
    ap.add_argument('--noise', type=float, default=0.002)
    ap.add_argument('--dmodel', type=int, default=64)
    ap.add_argument('--layers', type=int, default=2)
    args = ap.parse_args()

    device = args.device if (args.device=='cpu' or torch.cuda.is_available()) else 'cpu'

    # datasets

    # if data/self-sup-data.npz does not exist, generate datasets and save

    from pathlib import Path

    file_path_train = Path("data/self-sup-train.pt")
    file_path_val = Path("data/self-sup-val.pt")
    if file_path_train.is_file() and file_path_val.is_file():
        train_set = torch.load("data/self-sup-train.pt",weights_only=False)
        val_set = torch.load("data/self-sup-val.pt",weights_only=False)

    else:    
        print("Generating datasets ...")
        print(" Training set:")
        train_set = TorqueFreeDataset(N=args.trainN, T=args.T, dt=args.dt, device=device, noise_std=args.noise)
        print()
        print(" Validation set:")
        val_set   = TorqueFreeDataset(N=args.valN,   T=args.T, dt=args.dt, device=device, noise_std=args.noise)
        torch.save(train_set,"data/self-sup-train.pt")
        torch.save(val_set,"data/self-sup-val.pt")

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader   = torch.utils.data.DataLoader(val_set,   batch_size=args.batch, shuffle=False)

    # model + trainer
    model_mamba = InertiaMambaEstimator(d_model=args.dmodel, n_layers=args.layers)

    from alt_backbones import build_estimator
    model_lstm = build_estimator(kind="bilstm", d_model=args.dmodel, n_layers=args.layers)

    tcfg = TrainCfg(lr=args.lr, wd=args.wd, lam_energy=args.lamE, lam_dyn=args.lamD, device=device)
    trainer_mamba = InertiaTrainer(model_mamba, tcfg)
    trainer_lstm = InertiaTrainer(model_lstm, tcfg)

    def train(model,trainer):
        # train
        for ep in range(1, args.epochs+1):
            model.train()
            tr_sum=0; n=0
            for batch in train_loader:
                loss, terms = trainer.step(batch)
                tr_sum += loss*batch[0].size(0); n += batch[0].size(0)
            tr_loss = tr_sum/n

            # eval
            model.eval()
            with torch.no_grad():
                vals=[]; Lc=[]; Ec=[]; Dc=[]
                Ipred=[]; Itrue=[]
                for q,w,I,dt in val_loader:
                    q=q.to(device); w=w.to(device); Itrue.append(I)
                    I_hat,_,_ = trainer.infer(q,w); Ipred.append(I_hat.cpu())
                    dt_val = as_scalar_dt(dt)
                    l,t = trainer.loss_fn(q,w,I_hat,dt_val)
                    vals.append(float(l)); Lc.append(float(t['L_const'])); Ec.append(float(t['E_const'])); Dc.append(float(t['Euler']))
                vloss = float(np.mean(vals))
                rerr, ascore = eig_ratio_axis_metrics(torch.cat(Ipred), torch.cat(Itrue))
            print(f"[{ep:02d}] train={tr_loss:.4e} | val={vloss:.4e} | L={np.mean(Lc):.2e} E={np.mean(Ec):.2e} Dyn={np.mean(Dc):.2e} "
                f"| eig-ratio-L2={rerr:.3e} axis-align={ascore:.3f}")

        # analyze one sample
        q,w,Itrue,_ = val_set[0]
        I_hat,_,_ = trainer.infer(q.unsqueeze(0), w.unsqueeze(0))

        I_pred_np = I_hat[0].cpu()
        I_true_np = Itrue.cpu()

    
        report = principal_inertia_comparison(I_pred_np, I_true_np)

        print("Pred I (trace=1):\n", np.array_str(np.array(I_pred_np), precision=4, suppress_small=True))
        print("True I (trace=1):\n", np.array_str(np.array(I_true_np), precision=4, suppress_small=True))
        print("\n--- Principal inertia comparison ---")
        print("eigs_pred:", np.array_str(report["evals_pred"], precision=6))
        print("eigs_true:", np.array_str(report["evals_true"], precision=6))
        print("abs_err  :", np.array_str(report["abs_err"], precision=6))
        print("rel_err  :", np.array_str(report["rel_err"], precision=6))
        print("axis |cos|:", np.array_str(report["axis_cosines"], precision=6), 
            "  mean=", f'{report["axis_alignment_mean"]:.4f}')
        print("frame misalignment: "
            f'{report["frame_angle_deg"]:.3f} deg  ({report["frame_angle_rad"]:.4f} rad)')
        print("axis match (pred i -> true j):", report["match_indices"])

    print("training mamba")
    train(model_mamba,trainer_mamba)
    print("training lstm")
    train(model_lstm,trainer_lstm)

if __name__ == "__main__":
    main()
