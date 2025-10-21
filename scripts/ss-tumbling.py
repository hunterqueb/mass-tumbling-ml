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

import matplotlib.pyplot as plt

from qutils.integrators import ode45
from qutils.ml.mamba import Mamba, MambaConfig 

# ======================== math utils ========================

def normalize_quat(q: torch.Tensor) -> torch.Tensor:
    return q / (q.norm(dim=-1, keepdim=True) + 1e-12)
# normalize quat np version
def normalize_quat_np(q: np.ndarray) -> np.ndarray:
    return q / (np.linalg.norm(q) + 1e-12)

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
    # Sanitize non-finite entries
    finite = torch.isfinite(I).all(dim=-1, keepdim=True).all(dim=-2, keepdim=True)
    if not finite.all():
        I = I.clone()
        I[~finite.expand_as(I)] = 0.0

    # Symmetrize
    I = 0.5 * (I + I.transpose(-1, -2))

    # Eigen-decompose
    evals, evecs = torch.linalg.eigh(I.double())
    evals = torch.clamp(evals, min=min_eig).float()
    I_spd = (evecs.float() @ torch.diag_embed(evals) @ evecs.float().transpose(-1, -2))

    # Normalize trace
    tr = torch.diagonal(I_spd, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
    I_spd = I_spd / (tr + 1e-12)
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

def sample_inertia(batch, device=None, min_ratio=0.05):
    # ensure eigenvalues not arbitrarily small (≥ min_ratio of trace)
    alphas = np.array([2.0, 2.0, 2.0])
    eig = np.random.gamma(alphas, 1.0, size=(batch, 3))
    eig = eig / eig.sum(axis=-1, keepdims=True)
    eig = np.clip(eig, min_ratio, None)
    eig = eig / eig.sum(axis=-1, keepdims=True)
    M = np.random.randn(batch, 3, 3)
    Q = np.empty_like(M)
    for i in range(batch):
        q, _ = np.linalg.qr(M[i])
        if np.linalg.det(q) < 0:
            q[:, 0] = -q[:, 0]
        Q[i] = q
    I = np.matmul(Q, np.matmul(np.expand_dims(np.diagflat(eig[0]), 0) if batch == 1 else np.array([np.diag(e) for e in eig]), Q.transpose(0, 2, 1)))
    return I

@torch.no_grad()
def omega_mat(w):
    wx, wy, wz = w
    return np.array([
        [0.0, -wx, -wy, -wz],
        [wx,  0.0,  wz, -wy],
        [wy, -wz,  0.0,  wx],
        [wz,  wy, -wx,  0.0]
    ], dtype=float)


@torch.no_grad()
def euler_rhs(t, x, I_body, Iinv_body, torque_fn):
    """
    State x = [q(4), w(3)], q Hamilton (scalar-first), body-frame angular velocity w.
    I_body constant in body frame (3x3 symmetric positive-definite).
    torque_fn(t, q, w) returns external torque in body frame (3,).
    """
    q = x[:4]
    w = x[4:]

    # Quaternion kinematics: qdot = 0.5 * Omega(w) * q
    qdot = 0.5 * omega_mat(w) @ q

    # Euler rotational dynamics in body frame: I wdot + w x (I w) = tau
    H = I_body @ w
    tau = torque_fn(t, q, w)
    wdot = Iinv_body @ (tau - np.cross(w, H))

    return np.hstack([qdot, wdot])


@torch.no_grad()
def simulate_torque_free(I, q0, w0, T: float, dt: float,device,
                         noise_std: float=0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    torque_fn=lambda t, q, w: np.zeros(3)
    
    steps = int(T/dt)

    Iinv_body = np.linalg.inv(I)

    x0 = np.hstack([normalize_quat_np(np.asarray(q0, dtype=float)), np.asarray(w0, dtype=float)])

    def rhs_renorm(t, x):
        # Drift control: renormalize q in-place every call to keep unit length
        q = x[:4]; w = x[4:]
        qn = q / max(1e-15, np.linalg.norm(q))
        # ----- noise injection on angular velocity -----
        if noise_std > 0.0:
            w = w + np.random.randn(3) * noise_std
        # ----------------------------------------------

        x_fixed = np.hstack([qn, w])
        return euler_rhs(t, x_fixed, I, Iinv_body, torque_fn)

    t,y = ode45(
        rhs_renorm,
        [0,T],
        x0,
        t_eval=np.linspace(0, T, steps),
    )

    # Final renormalization on output
    qs = y[:, :4]
    norms = np.linalg.norm(qs, axis=0)
    y[:, :4] = qs / norms

    ws = y[:, 4:]

    qs = torch.tensor(y[:, :4], device=device)
    ws = torch.tensor(ws, device=device)
    return qs, ws
    
class TorqueFreeDataset(torch.utils.data.Dataset):
    def __init__(self, N=2048, T=4.0, dt=0.01, device="cpu",
                 w0_mag_range=(0.2, 2.0), noise_std=0.002):
        self.N=N; self.T=T; self.dt=dt; self.device=device
        self.steps = int(T/dt)
        self.I_true=[]; self.q=[]; self.w=[]
        for _ in range(N):
            I = sample_inertia(1, device=device)[0]
            axis = np.random.randn(3); axis = axis/np.linalg.norm(axis)
            ang = np.random.rand()*2*math.pi
            q0 = np.array([math.cos(ang/2), *(math.sin(ang/2)*axis)])
            mag = np.random.randn(3) * (w0_mag_range[1]-w0_mag_range[0]) + w0_mag_range[0]
            v = np.random.randn(3); v = v / (np.linalg.norm(v)+1e-9)
            w0 = mag * v
            q, w = simulate_torque_free(I, q0, w0, T, dt, device=device, noise_std=noise_std)
            self.I_true.append(I); self.q.append(q); self.w.append(w)
            print(f"Generated sample {len(self.I_true)}/{N}", end='\r')
        self.I_true = torch.tensor(np.array(self.I_true),device=device,dtype=torch.float32)  # (N,3,3)
        self.q = torch.stack(self.q)             # (N,S,4)
        self.q = self.q.float()
        self.w = torch.stack(self.w)             # (N,S,3)
        self.w = self.w.float()

    def __len__(self): return self.N
    def __getitem__(self, i):
        return self.q[i], self.w[i], self.I_true[i], self.dt
    def convert_to_float64(self):
        self.I_true = self.I_true.double()
        self.q = self.q.double()
        self.w = self.w.double()

    def convert_to_float32(self):
        self.I_true = self.I_true.float()
        self.q = self.q.float()
        self.w = self.w.float()

    def to(self, device):
        self.I_true = self.I_true.to(device)
        self.q = self.q.to(device)
        self.w = self.w.to(device)
        self.device = device
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
    def __str__(self):
        return "mamba"


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

        loss_tau, _ = self.dynamics_losses(I64, w_s, wdot, mode="tau_residual")
        loss_D = loss_tau  # rename to keep your total-loss code unchanged

        # sanitize per-sequence mean, then mean over batch
        loss_L = torch.nan_to_num(loss_L.mean(dim=1), nan=0.0).mean()
        loss_E = torch.nan_to_num(loss_E,             nan=0.0).mean()
        loss_D = torch.nan_to_num(loss_D,             nan=0.0).mean()

        loss = (loss_L + self.lamE*loss_E + self.lamD*loss_D).float()
        return loss, {'L_const': loss_L.float(), 'E_const': loss_E.float(), 'Euler': loss_D.float()}

    def dynamics_losses(self,I64, w_s, wdot, tau=None, mode="wdot_residual"):
        """
        mode = 'tau_residual' -> || tau - (I wdot + w x (I w)) ||^2
        mode = 'wdot_residual' -> || wdot - I^{-1}(tau - w x (I w)) ||^2
        If tau is None, both reduce to the torque-free form.
        Returns: loss_D (scalar tensor), tau_pred (B,T,3) for diagnostics
        """
        # I w
        Iw = (I64[:, None, :, :] @ w_s[..., None]).squeeze(-1)  # (B,T,3)
        cross = torch.cross(w_s, Iw, dim=-1)                    # (B,T,3)

        if tau is None:
            # torque-free fallback
            tau = torch.zeros_like(cross)

        tau = tau.to(w_s.dtype)

        if mode == "tau_residual":
            # predict torque from measured wdot
            tau_pred = (I64[:, None, :, :] @ wdot[..., None]).squeeze(-1) + cross  # (B,T,3)
            resid = tau - tau_pred
            resid = torch.nan_to_num(resid, nan=0.0, posinf=0.0, neginf=0.0)
            loss_D = (resid**2).sum(dim=-1).mean()  # mean over time and batch
            return loss_D, tau_pred

        elif mode == "wdot_residual":
            # predict wdot from measured tau
            rhs = tau - cross                                         # (B,T,3)
            A = I64[:, None, :, :].expand(-1, rhs.size(1), -1, -1)    # (B,T,3,3)
            wdot_model = torch.linalg.solve(A, rhs[..., None]).squeeze(-1)  # (B,T,3)
            resid = wdot - wdot_model
            resid = torch.nan_to_num(resid, nan=0.0, posinf=0.0, neginf=0.0)
            loss_D = (resid**2).sum(dim=-1).mean()
            return loss_D, None

        else:
            raise ValueError("mode must be 'tau_residual' or 'wdot_residual'")

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
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--T', type=float, default=3.0)
    ap.add_argument('--dt', type=float, default=0.01)
    ap.add_argument('--trainN', type=int, default=2000)
    ap.add_argument('--valN', type=int, default=300)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--lamE', type=float, default=1.0)
    ap.add_argument('--lamD', type=float, default=0.1)
    ap.add_argument('--noise', type=float, default=0.002)
    ap.add_argument('--dmodel', type=int, default=64)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--force', action='store_true', help='Force dataset regeneration')
    ap.add_argument('--save',action='store_true', help='Save logs from training')
    args = ap.parse_args()

    if args.save:
        import os
        from datetime import datetime
        log_dir = "logs/selfsup_mamba_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs(log_dir, exist_ok=True)
        import sys
        import shutil
        # copy this script to log dir
        shutil.copy(__file__, os.path.join(log_dir, os.path.basename(__file__)))
        # redirect stdout to log file
        log_file = open(os.path.join(log_dir, "train_log.txt"), "w")
        class Tee(object):
            def __init__(self, *files):
                self.files = files
            def write(self, obj):
                for f in self.files:
                    f.write(obj)
                    f.flush()
            def flush(self):
                for f in self.files:
                    f.flush()
        sys.stdout = Tee(sys.stdout, log_file)

        # save args
        with open(os.path.join(log_dir, "args.txt"), "w") as f:
            for k,v in vars(args).items():
                f.write(f"{k}: {v}\n")

        print("Logging to", log_dir)

    from qutils.ml import getDevice
    device = getDevice()

    print("Using device:", device)
    # datasets

    # if data/self-sup-data.npz does not exist, generate datasets and save

    from pathlib import Path
 
    file_path_train = Path("data/self-sup-train_" + str(args.T) + ".pt")
    file_path_val = Path("data/self-sup-val_" + str(args.T) + ".pt")
    if file_path_train.is_file() and file_path_val.is_file() and not args.force:
        train_set = torch.load("data/self-sup-train_" + str(args.T) + ".pt",weights_only=False,map_location=device)
        val_set = torch.load("data/self-sup-val_" + str(args.T) + ".pt",weights_only=False,map_location=device)

    else:    
        print("Generating datasets ...")
        print(" Training set:")
        train_set = TorqueFreeDataset(N=args.trainN, T=args.T, dt=args.dt, device=device, noise_std=args.noise)
        print()
        print(" Validation set:")
        val_set   = TorqueFreeDataset(N=args.valN,   T=args.T, dt=args.dt, device=device, noise_std=args.noise)
        torch.save(train_set,"data/self-sup-train_" + str(args.T) + ".pt")
        torch.save(val_set,"data/self-sup-val_" + str(args.T) + ".pt")
        print()

    # plot a random sample from a set
    sample_idx = np.random.randint(0, len(val_set))
    q_sample, w_sample, I_sample, dt_sample = val_set[sample_idx]
    time_array = np.arange(0, args.T, args.dt)[:q_sample.shape[0]]
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(time_array, w_sample.cpu().numpy())
    plt.title('Angular Velocity (omega)')
    plt.xlabel('Time (s)')
    plt.ylabel('Omega (rad/s)')
    plt.legend(['omega_x', 'omega_y', 'omega_z'])
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(time_array, q_sample.cpu().numpy())
    plt.title('Orientation (Quaternion)')
    plt.xlabel('Time (s)')
    plt.ylabel('Quaternion Components')
    plt.legend(['q_w', 'q_x', 'q_y', 'q_z'])
    plt.grid()
    plt.tight_layout()


    train_set.convert_to_float32()
    val_set.convert_to_float32()
    train_set.to(device)
    val_set.to(device)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader   = torch.utils.data.DataLoader(val_set,   batch_size=args.batch, shuffle=False)

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
                    I_hat,_,_ = trainer.infer(q,w); Ipred.append(I_hat)
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

    def validate(model,trainer):

        modelStr = str(model)
        print("Validation of model:", modelStr)
        model.eval()

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

        # visualize predicted vs true inertia ellipsoids
        # define ellipsoid points
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))
        def plot_ellipsoid(I, ax, color='b', alpha=0.5, label=''):
            # eigen-decomposition
            evals, evecs = np.linalg.eigh(I)
            # radii are proportional to sqrt of eigenvalues
            rx, ry, rz = np.sqrt(evals)
            # transform unit sphere points
            ellipsoid_points = np.array([rx * x.flatten(), ry * y.flatten(), rz * z.flatten()])  # (3,N)
            ellipsoid_transformed = evecs @ ellipsoid_points  # (3,N)
            X = ellipsoid_transformed[0, :].reshape(x.shape)
            Y = ellipsoid_transformed[1, :].reshape(y.shape)
            Z = ellipsoid_transformed[2, :].reshape(z.shape)
            ax.plot_surface(X, Y, Z, color=color, alpha=alpha, label=label)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        plot_ellipsoid(I_true_np.numpy(), ax, color='g', alpha=0.5, label='True Inertia')
        plot_ellipsoid(I_pred_np.numpy(), ax, color='r', alpha=0.5, label='Predicted Inertia')
        ax.set_title('Inertia Ellipsoids')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.legend()
        plt.savefig(log_dir+"/inertia_ellipsoids_"+modelStr+".png")

        # using the inertia matrices to compute energy and angular momentum and plot the error over time
        I_pred_torch = I_hat[0].to(device)
        I_true_torch = Itrue.to(device)
        w_torch = w.to(device)
        q_torch = q.to(device)
        dt_val = as_scalar_dt(dt_sample)
        Iw_pred = (I_pred_torch @ w_torch.unsqueeze(-1)).squeeze(-1)
        Iw_true = (I_true_torch @ w_torch.unsqueeze(-1)).squeeze(-1)
        E_pred = 0.5 * (w_torch * Iw_pred).sum(-1)
        E_true = 0.5 * (w_torch * Iw_true).sum(-1)
        L_pred = (quat_to_R(q_torch) @ Iw_pred.unsqueeze(-1)).squeeze(-1)
        L_true = (quat_to_R(q_torch) @ Iw_true.unsqueeze(-1)).squeeze(-1)
        time_array = np.arange(0, args.T, args.dt)[:q.shape[0]]
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(time_array, E_pred.cpu().numpy(), label='Predicted Energy')
        plt.plot(time_array, E_true.cpu().numpy(), label='True Energy')
        plt.title('Rotational Kinetic Energy')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy (J)')
        plt.legend()
        plt.grid()
        plt.subplot(1, 2, 2)
        plt.plot(time_array, np.linalg.norm(L_pred.cpu().numpy(), axis=-1), label='Predicted |L|')
        plt.plot(time_array, np.linalg.norm(L_true.cpu().numpy(), axis=-1), label='True |L|')
        plt.title('Angular Momentum Magnitude')
        plt.xlabel('Time (s)')
        plt.ylabel('|L| (kg·m²/s)')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(log_dir+"/energy_angular_momentum_"+modelStr+".png")

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(time_array, np.abs(E_pred.cpu().numpy() - E_true.cpu().numpy()), label='Energy Error')
        plt.title('Energy Conservation Error')
        plt.xlabel('Time (s)')
        plt.ylabel('Absolute Error (J)')
        plt.legend()
        plt.grid()
        plt.subplot(1, 2, 2)
        plt.plot(time_array, np.linalg.norm(L_pred.cpu().numpy() - L_true.cpu().numpy(), axis=-1), label='Angular Momentum Error')
        plt.title('Angular Momentum Conservation Error')
        plt.xlabel('Time (s)')
        plt.ylabel('Absolute Error (kg·m²/s)')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(log_dir+"/conservation_errors_"+modelStr+".png")

    # model + trainer

    from alt_backbones import build_estimator
    model_lstm = build_estimator(kind="bilstm", d_model=args.dmodel, n_layers=args.layers).to(device)
    model_mamba = InertiaMambaEstimator(d_model=args.dmodel, n_layers=args.layers).to(device)
    model_transformer = build_estimator(kind="transformer", d_model=args.dmodel, n_layers=args.layers).to(device)
    model_tcn = build_estimator(kind="tcn", d_model=args.dmodel, n_layers=args.layers).to(device)

    tcfg = TrainCfg(lr=args.lr, wd=args.wd, lam_energy=args.lamE, lam_dyn=args.lamD, device=device)
    trainer_mamba = InertiaTrainer(model_mamba, tcfg)
    trainer_lstm = InertiaTrainer(model_lstm, tcfg)
    trainer_transformer = InertiaTrainer(model_transformer, tcfg)
    trainer_tcn = InertiaTrainer(model_tcn, tcfg)

    print("training lstm")
    train(model_lstm,trainer_lstm)
    validate(model_lstm,trainer_lstm)
    print("training mamba")
    train(model_mamba,trainer_mamba)
    validate(model_mamba,trainer_mamba)
    print("training transformer")
    train(model_transformer,trainer_transformer)
    validate(model_transformer,trainer_transformer)
    print("training tcn")
    train(model_tcn,trainer_tcn)
    validate(model_tcn,trainer_tcn)


    if not args.save:
        plt.show()

if __name__ == "__main__":
    main()
