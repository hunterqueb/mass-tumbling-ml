#!/usr/bin/env python3

import math
import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo

import matplotlib.pyplot as plt

from qutils.integrators import ode45
from qutils.tictoc import timer
from qutils.ml.mamba import Mamba, MambaConfig 
from qutils.ml import printModelParmSize
from scipy.integrate import solve_ivp

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

def quat_to_R_np(q):
    """
    numpy version of quat_to_R
    """
    w, x, y, z = q
    
    R = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])
    return R

def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to Euler angles (roll, pitch, yaw) in radians.
    q: (4,) scalar-first
    returns: (3,) roll, pitch, yaw
    """
    w, x, y, z = q

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * (np.pi / 2)  # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])

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
def central_diff_vec(x, dt: float):
    """
    x: (B, T, D) over time dimension 1
    returns dx/dt with same shape
    """
    dx = torch.empty_like(x)
    dx[:,1:-1] = (x[:,2:] - x[:,:-2]) / (2.0*dt)
    dx[:, :1]  = (x[:,1:2] - x[:, :1]) / dt
    dx[:, -1:] = (x[:, -1:] - x[:, -2:-1]) / dt
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

import torch
import torch.nn.functional as F

def central_diff_1d(x, dt):
    # x: (B,T)
    dx = torch.empty_like(x)
    dx[:,1:-1] = (x[:,2:] - x[:,:-2]) / (2*dt)
    dx[:, :1]  = (x[:,1:2] - x[:, :1]) / dt
    dx[:, -1:] = (x[:, -1:] - x[:, -2:-1]) / dt
    return dx

def sliding_var(x, win):
    # x: (B,T)
    B,T = x.shape
    if win >= T:
        m = x.mean(dim=1, keepdim=True)
        return ((x - m)**2).mean(dim=1)  # (B,)
    # conv-based windowed mean/var
    w = torch.ones(1,1,win, device=x.device, dtype=x.dtype) / win
    y = x.unsqueeze(1)                         # (B,1,T)
    mu = F.conv1d(y, w, padding=win//2)[:,0]   # (B,T)
    v  = F.conv1d((y - mu.unsqueeze(1))**2, w, padding=win//2)[:,0]  # (B,T)
    # return per-trajectory average of windowed variance
    return v.mean(dim=1)                        # (B,)

def non_dc_power(x):
    # x: (B,T) -> average power at non-DC frequencies via rFFT
    X = torch.fft.rfft(x - x.mean(dim=1, keepdim=True), dim=1)
    # drop DC (index 0). Normalize by T for scale invariance
    P = (X[:,1:].abs()**2)
    T = x.shape[1]
    return (P.sum(dim=1) / T)                  # (B,)
def rotx(angle_degrees):
    angle_radians = np.deg2rad(angle_degrees)
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)
    return np.array([
        [1, 0, 0],
        [0, cos_angle, -sin_angle],
        [0, sin_angle, cos_angle]
    ])
def roty(angle_degrees):
    angle_radians = np.deg2rad(angle_degrees)
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)
    return np.array([
        [cos_angle, 0, sin_angle],
        [0, 1, 0],
        [-sin_angle, 0, cos_angle]
    ])
def rotz(angle_degrees):
    angle_radians = np.deg2rad(angle_degrees)
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)
    return np.array([
        [cos_angle, -sin_angle, 0],
        [sin_angle, cos_angle, 0],
        [0, 0, 1]
    ])

def axang2quat(axis, ang):
    axis = np.asarray(axis, dtype=float).reshape(3)
    axis = axis / np.linalg.norm(axis)
    c = np.cos(ang / 2.0)
    s = np.sin(ang / 2.0)
    q = np.empty(4, dtype=float)
    q[0] = c
    q[1:] = s * axis
    return q


def vec(M):
    # Column-stacking like MATLAB's (:)
    M = np.asarray(M)
    return M.reshape(-1, order="F")

def random_inertia(principal = True):
    # --- 1. Random eigenvalues (positive) ---
    eig = np.random.rand(3)           # random positive values
    eig = eig / np.sum(eig)           # normalize or not, your choice
    # You can scale eig if you want a different trace or magnitude

    # --- 2. Random orthonormal matrix Q via QR decomposition ---
    M = np.random.randn(3,3)
    Q, R = np.linalg.qr(M)

    # Ensure det(Q) = +1 (proper rotation)
    if np.linalg.det(Q) < 0:
        Q[:,0] *= -1

    # --- 3. Construct SPD inertia ---
    J = Q @ np.diag(eig) @ Q.T

    if principal:
        # Return in principal axes (diagonal)
        J = np.diag(eig)
    return J

def dyn_partial_adapt_v2(t, x, J_fixed, S, J_true, K_R, K_Om, Gamma, eps_reg, dith,control_torque_log,control_torque_max):
    """
    x = [q(4), w(3), alpha(3)]
    """
    q = x[0:4].copy()
    w = x[4:7].copy()
    alpha = x[7:10].copy()

    # Normalize quaternion
    n_q = np.linalg.norm(q)
    q = q / max(n_q, 1e-12)

    # Build Jhat = J_fixed + sum_i alpha_i S_i
    Jt_hat = alpha[0] * S[:, :, 0] + alpha[1] * S[:, :, 1] + alpha[2] * S[:, :, 2]
    Jt_hat = 0.5 * (Jt_hat + Jt_hat.T)
    Jhat = J_fixed + Jt_hat

    # Small-angle error
    eR = 2.0 * q[1:4]

    # PD torque
    uPD = -K_R @ eR - K_Om @ w

    # Dither torque
    A_d = dith["A"]
    w_d = dith["w"]
    u_dith = A_d * np.array(
        [
            np.sin(w_d[0] * t),
            np.sin(w_d[1] * t + 0.7),
            np.sin(w_d[2] * t + 1.3),
        ],
        dtype=float,
    )

    # Control torque
    u = uPD + np.cross(w, Jhat @ w) + u_dith

    if control_torque_max is not None:
        # clamp u between specified control torque max
        u = np.clip(u, -control_torque_max, control_torque_max)

    # save control torque for diagnostics
    control_torque_log.append(u.copy())

    # True plant dynamics: J_true * wdot = u - w × (J_true w)
    rhs = u - np.cross(w, J_true @ w)
    wdot = np.linalg.solve(J_true, rhs)

    # Quaternion kinematics
    wx, wy, wz = w
    W = np.array(
        [
            [0.0, -wx, -wy, -wz],
            [wx, 0.0, wz, -wy],
            [wy, -wz, 0.0, wx],
            [wz, wy, -wx, 0.0],
        ],
        dtype=float,
    )
    qdot = 0.5 * (W @ q)

    # Adaptation
    # tau_err = uPD + u_dith - Jhat * wdot
    tau_err = (uPD + u_dith) - (Jhat @ wdot)

    # Phi = [-(S1*wdot), -(S2*wdot), -(S3*wdot)] (3x3)
    Phi = np.column_stack(
        [
            -(S[:, :, 0] @ wdot),
            -(S[:, :, 1] @ wdot),
            -(S[:, :, 2] @ wdot),
        ]
    )

    denom = np.trace(Phi.T @ Phi) + eps_reg
    alpha_dot_raw = -Gamma @ (Phi.T @ tau_err) / denom
    # --- projection to enforce alpha >= alpha_min > 0 ---
    alpha_min = 1e-2  # choose a physically meaningful lower bound

    alpha_dot = alpha_dot_raw.copy()
    for i in range(alpha_dot.size):
        # if we're at/below the bound and the update would push alpha lower, block it
        if alpha[i] <= alpha_min and alpha_dot[i] < 0.0:
            alpha_dot[i] = 0.0

    dx = np.concatenate([qdot, wdot, alpha_dot])
    return dx

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

def add_sensor_noise(qs_t, ws_t, gyro_std=0.002, att_std_deg=0.0, device='cpu'):
    # gyro noise: additive per-sample std (rad/s)
    ws_noisy = ws_t + torch.randn_like(ws_t) * gyro_std
    if att_std_deg <= 0:
        return qs_t, ws_noisy
    # small attitude noise: apply small random rotations to each q
    att_std = np.deg2rad(att_std_deg)
    B,T = 1, qs_t.shape[0]
    qs = qs_t.cpu().numpy()
    rng = np.random.default_rng()
    for k in range(T):
        axis = rng.standard_normal(3); axis /= np.linalg.norm(axis) + 1e-12
        ang = rng.normal(0.0, att_std)
        dq = np.array([np.cos(ang/2), *(np.sin(ang/2)*axis)], float)
        # quaternion multiply dq ⊗ q
        w1,x1,y1,z1 = dq; w2,x2,y2,z2 = qs[k]
        qs[k] = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], float)
        qs[k] /= np.linalg.norm(qs[k]) + 1e-12
    return torch.tensor(qs, device=device), ws_noisy

class TorqueFreeDataset(torch.utils.data.Dataset):
    def __init__(self, file=None,N=2048, T=3.0, dt=0.01, device="cpu",
                 w0_mag_range=(0.2, 2.0), noise_std=0.002):
        self.N=N; self.T=T; self.dt=dt; self.device=device
        self.steps = int(T/dt)
        if file is not None:
            # load from file
            data = np.load(file)
            self.device = device
            self.dt = data['dt']
            self.I_true = torch.tensor(data['I_true'],device=self.device,dtype=torch.float32)
            self.q = torch.tensor(data['q'],device=self.device,dtype=torch.float32)
            self.w = torch.tensor(data['w'],device=self.device,dtype=torch.float32)
            self.N = self.q.shape[0]
            self.I_true_unscaled = data["I_true_real"] if "I_true_real" in data else None
            self.shape = data["shape"] if "shape" in data else None

    def __len__(self): return self.N
    def __getitem__(self, i):
        return self.q[i], self.w[i], self.I_true[i], self.dt, self.I_true_unscaled[i] if hasattr(self, "I_true_unscaled") else None
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

    def saveDataset(self, file):
        np.savez_compressed(
            file,
            dt=self.dt,
            I_true=self.I_true.cpu().numpy(),
            q=self.q.cpu().numpy(),
            w=self.w.cpu().numpy()
        )
    def get_shape(self,i):
        if hasattr(self, "shape") and self.shape is not None:
            return self.shape[i]
        return None
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
    def __init__(self, d_model=192, n_layers=6, d_state=32, expand=1, d_conv=4,
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
    def __init__(self, lam_energy=0.1, lam_dyn=1.0, smooth_sigma=2.5, smooth_k=9,dynamics_mode="tau"):
        super().__init__()
        self.lamE = lam_energy
        self.lamD = lam_dyn
        self.sigma = smooth_sigma
        self.k = 9  # use 9 or 11
        self.dynamics_mode = dynamics_mode
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
        Ldev   = Linert - Lmean                          # (B,T,3)
        loss_L_const_seq = (Ldev**2).sum(dim=-1).mean(dim=1)  # (B,) mean over time
        loss_L_const = loss_L_const_seq.mean()           # scalar over batch

        Ldot   = central_diff(Linert, dt)               # (B,T,3)
        Ldot   = torch.clamp(Ldot, min=-500.0, max=500.0)

        loss_L_deriv_seq = (Ldot**2).sum(dim=-1).mean(dim=1)  # (B,)
        loss_L_deriv = loss_L_deriv_seq.mean()

        loss_L = loss_L_const + 0.5 * loss_L_deriv

        # E      = (w_s * Iw).sum(dim=-1) * 0.5
        # # variance that ignores NaNs/Infs
        # E = torch.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0)
        # loss_E = E.var(dim=1, unbiased=False)
        loss_E, E_terms = self.energy_losses(w_s.double(), I64, dt,
                                        lam_E=1.0, lam_win=0.5, lam_spec=0.2, lam_dEdt=0.5,
                                        win= max(11, (w_s.shape[1]//20)|1),  # ~5% of window, odd
                                        topk_frac=0.5)


        loss_tau, _ = self.dynamics_losses(I64, w_s, wdot)
        loss_D = loss_tau  # rename to keep your total-loss code unchanged

        loss_axis = self.axis_constancy_loss(q64, w_s, I64)

        # sanitize per-sequence mean, then mean over batch
        loss_L = torch.nan_to_num(loss_L,             nan=0.0).mean()
        loss_E = torch.nan_to_num(loss_E,             nan=0.0).mean()
        loss_D = torch.nan_to_num(loss_D,             nan=0.0).mean()
        loss_axis = torch.nan_to_num(loss_axis,       nan=0.0).mean()

        loss = (0.5 * loss_L + self.lamE*loss_E + self.lamD*loss_D + 0.1*loss_axis).float()
        return loss, {'L_const': loss_L.float(), 'E_const': loss_E.float(), 'Euler': loss_D.float(), 'Axis_const': loss_axis.float()}

    def dynamics_losses(self,I64, w_s, wdot, tau=None):
        """
        mode = 'tau' -> || tau - (I wdot + w x (I w)) ||^2
        mode = 'wdot' -> || wdot - I^{-1}(tau - w x (I w)) ||^2
        If tau is None, both reduce to the torque-free form.
        Returns: loss_D (scalar tensor), tau_pred (B,T,3) for diagnostics
        """
        mode=self.dynamics_mode
        # I w
        Iw = (I64[:, None, :, :] @ w_s[..., None]).squeeze(-1)  # (B,T,3)
        cross = torch.cross(w_s, Iw, dim=-1)                    # (B,T,3)

        if tau is None:
            # torque-free fallback
            tau = torch.zeros_like(cross)

        tau = tau.to(w_s.dtype)

        if mode == "tau":
            # predict torque from measured wdot
            tau_pred = (I64[:, None, :, :] @ wdot[..., None]).squeeze(-1) + cross  # (B,T,3)
            resid = tau - tau_pred
            resid = torch.nan_to_num(resid, nan=0.0, posinf=0.0, neginf=0.0)
            loss_D = (resid**2).sum(dim=-1).mean()  # mean over time and batch
            return loss_D, tau_pred

        elif mode == "wdot":
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
    
    def energy_losses(self,w, I, dt, lam_E=1.0, lam_win=1.0, lam_spec=0.2, lam_dEdt=0.5, win=51, topk_frac=0.5):
        """
        w: (B,T,3), I: (B,3,3)
        Returns: scalar loss_E and dict
        """

        # E(t) per batch
        w_s = gaussian_smooth_1d(w, sigma=self.sigma, k=self.k)

        Iw = (I[:,None,:,:] @ w_s[...,None]).squeeze(-1)   # (B,T,3)
        E  = 0.5 * (w_s * Iw).sum(dim=-1)                  # (B,T)

        # variance that ignores NaNs/Infs
        E = torch.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0)
        L_E = E.var(dim=1, unbiased=False)

        # 1) derivative penalty (unused) - use energy itself 
        dEdt = central_diff_1d(E, dt)
        L_dEdt = (dEdt**2).mean(dim=1)                     # (B,)

        # 2) windowed variance (captures oscillations even if mean is right)
        L_win = sliding_var(E, win=win)                  # (B,)

        # 3) spectral power off-DC (kills periodic energy ripple)
        L_spec = non_dc_power(E)                         # (B,)

        # robust aggregation: take top-k worst sequences (avoids cancellation)
        B = E.shape[0]
        k = max(1, int(topk_frac * B))
        def topk_mean(v):
            vals, _ = torch.topk(v, k=k, largest=True, sorted=False)
            return vals.mean()

        loss = (lam_E * topk_mean(L_E) +
                lam_win   * topk_mean(L_win)   +
                lam_spec  * topk_mean(L_spec)  +
                lam_dEdt  * topk_mean(L_dEdt)) 

        terms = {
            'E':  topk_mean(L_E).detach(),
            'E_winvar': topk_mean(L_win).detach(),
            'E_spec':   topk_mean(L_spec).detach(),
            'E_dEdt':   topk_mean(L_dEdt).detach()
        }
        return loss, terms
    
    def axis_constancy_loss(self,q, w, I, lam_dir=1.0, lam_mag=0.5, eps=1e-12):
        """
        q: (B,T,4) scalar-first, unit
        w: (B,T,3)
        I: (B,3,3) SPD
        returns: scalar loss
        """
        q = q.double(); w = w.double(); I = I.double()

        # Body->inertial rotation matrices, force (B,T,3,3)
        R = quat_to_R(q)                     # ensure your quat_to_R returns (B,T,3,3)
        if R.dim() == 3:                     # if it returns (T,3,3) for single batch, fix it
            R = R.unsqueeze(0).expand(q.size(0), -1, -1, -1)

        # I ω in body frame -> (B,T,3)
        Iw = torch.einsum('bij,btj->bti', I, w)       # (B,3,3) x (B,T,3)

        # Angular momentum in inertial frame H_I(t) = R(t) * (I ω)(t)
        H_I = torch.einsum('btij,btj->bti', R, Iw)    # (B,T,3)

        # Direction constancy: 1 - cos(angle with batch mean direction)
        Hm = H_I.mean(dim=1, keepdim=True)            # (B,1,3)
        Hm_norm = torch.linalg.norm(Hm, dim=-1, keepdim=True).clamp_min(eps)  # (B,1,1)
        HI_norm = torch.linalg.norm(H_I, dim=-1, keepdim=True).clamp_min(eps) # (B,T,1)
        dir_cos = (H_I * Hm).sum(dim=-1, keepdim=True) / (HI_norm * Hm_norm)  # (B,T,1)
        L_dir = (1.0 - dir_cos).mean()                # scalar

        # Magnitude constancy: variance over time of ||H_I||
        Hmag = HI_norm.squeeze(-1)                    # (B,T)
        L_mag = Hmag.var(dim=1, unbiased=False).mean()

        loss = lam_dir * L_dir + lam_mag * L_mag
        return loss# ======================== trainer ========================

@dataclass
class TrainCfg:
    lr: float = 3e-4
    wd: float = 1e-4
    lam_energy: float = 0.1
    lam_dyn: float = 1.0
    device: str = 'cuda'
    residual: str = 'tau'  # 'tau' or 'wdot'

class InertiaTrainer:
    def __init__(self, model: InertiaMambaEstimator, cfg: TrainCfg):
        self.model = model.to(cfg.device)
        self.loss_fn = PhysicsLoss(cfg.lam_energy, cfg.lam_dyn,dynamics_mode=cfg.residual).to(cfg.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
        self.device = cfg.device
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        self.opt,
        mode='min',             # or 'max' for accuracy
        factor=0.5,             # shrink LR by 50%
        patience=3             # wait for 3 epochs of no improvement
    )

    def step(self, batch):
        q, w, _, dt, _ = batch
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

def main(validate_OOD=False,plot=True):

    from qutils.ml import getDevice
    device = getDevice()

    # datasets
    # if data/self-sup-data.npz does not exist, generate datasets and save

    dataLoc = args.data + str(args.trainN) + "/self-sup-"

    from pathlib import Path
 
    file_path_val = Path(dataLoc+"val_" + str(args.T) + ".npz")
    if file_path_val.is_file() and not args.force:
        # load datasets
        if not validate_OOD:
            val_set = TorqueFreeDataset(file=dataLoc+"val_" + str(args.T) + ".npz",N=args.trainN, T=args.T, dt=args.dt, device=device, noise_std=args.noise)
        else:
            print("Validating on OOD dataset")
            val_set = TorqueFreeDataset(file=dataLoc+"OOD-val_" + str(args.T) + ".npz",N=args.trainN, T=args.T, dt=args.dt, device=device, noise_std=args.noise)
        print("Loaded datasets from", dataLoc)

    # plot a random sample from a set
    sample_idx = np.random.randint(0, len(val_set))
    q_sample, w_sample, I_sample, dt_sample, _ = val_set[sample_idx]
    time_array = np.arange(0, args.T, dt_sample)[:q_sample.shape[0]]

    val_set.convert_to_float32()
    val_set.to(device)

    val_loader   = torch.utils.data.DataLoader(val_set,   batch_size=args.batch, shuffle=False)

    def validate(model,trainer,infer_num=1):

        modelStr = str(model)
        print("Validation of model:", modelStr)
        inferenceTime = timer()
        model.eval()

        # # analyze one sample
        # q,w,Itrue,_,Itrue_unscaled = val_set[infer_num]
        # print(f"Analyzing sample #{infer_num} from validation set: {val_set.get_shape(infer_num)} ")
        # I_hat,_,_ = trainer.infer(q.unsqueeze(0), w.unsqueeze(0))

        # I_pred_np = I_hat[0].cpu()
        # I_true_np = Itrue.cpu()

        #  analyze entire validation set
        I_preds = []
        I_trues = []
        I_trues_unscaled = []
        q_all = []
        w_all = []
        for batch in val_loader:
            q,batch_w,Itrue,_,Itrue_unscaled = batch
            I_hat,_,_ = trainer.infer(q, batch_w)
            I_preds.append(I_hat.cpu())
            I_trues.append(Itrue.cpu())
            I_trues_unscaled.append(Itrue_unscaled)
            q_all.append(q.cpu())
            w_all.append(batch_w.cpu())

        inferenceTime.toc()
        return I_preds, I_trues_unscaled,q_all,w_all
    # model + trainer

    from alt_backbones import build_estimator
    model_lstm = build_estimator(kind="bilstm", d_model=args.dmodel, n_layers=args.layers).to(device)
    model_mamba = InertiaMambaEstimator(d_model=args.dmodel, n_layers=args.layers).to(device)
    model_transformer = build_estimator(kind="transformer", d_model=args.dmodel, n_layers=args.layers).to(device)
    model_tcn = build_estimator(kind="tcn", d_model=args.dmodel, n_layers=args.layers).to(device)

    tcfg = TrainCfg(lr=args.lr, wd=args.wd, lam_energy=args.lamE, lam_dyn=args.lamD, device=device,residual=args.residual)
    trainer_mamba = InertiaTrainer(model_mamba, tcfg)
    trainer_lstm = InertiaTrainer(model_lstm, tcfg)
    trainer_transformer = InertiaTrainer(model_transformer, tcfg)
    trainer_tcn = InertiaTrainer(model_tcn, tcfg)

    # get state dict folder and load if exists
    lstm_state_dict = torch.load(args.load + "model_lstm.pth",map_location=device)
    model_lstm.load_state_dict(lstm_state_dict)

    mamba_state_dict = torch.load(args.load + "model_mamba.pth",map_location=device)
    model_mamba.load_state_dict(mamba_state_dict,strict=False)
    
    transformer_state_dict = torch.load(args.load + "model_transformer.pth",map_location=device)
    model_transformer.load_state_dict(transformer_state_dict)
    
    tcn_state_dict = torch.load(args.load + "model_tcn.pth",map_location=device)
    model_tcn.load_state_dict(tcn_state_dict)

    inference_num = np.random.randint(0, len(val_set))

    qf = q_sample.cpu().numpy()[-1,:]
    wf = w_sample.cpu().numpy()[-1,:]

    I_pred_lstm, I_true,q_all,w_all = validate(model_lstm,trainer_lstm,inference_num)
    I_pred_mamba, _, _, _ = validate(model_mamba,trainer_mamba,inference_num)
    I_pred_transformer, _, _, _ = validate(model_transformer,trainer_transformer,inference_num)

    I_pred_tcn, _, _, _ = validate(model_tcn,trainer_tcn,inference_num)

    # stack batch results
    I_pred_lstm = torch.cat(I_pred_lstm, dim=0)
    I_pred_mamba = torch.cat(I_pred_mamba, dim=0)
    I_pred_transformer = torch.cat(I_pred_transformer, dim=0)
    I_pred_tcn = torch.cat(I_pred_tcn, dim=0)
    I_true = torch.cat(I_true, dim=0)
    q_all = torch.cat(q_all, dim=0)
    w_all = torch.cat(w_all, dim=0)

    return I_pred_lstm, I_pred_mamba, I_pred_transformer, I_pred_tcn, I_true,q_all,w_all
def demo_adapt_partial_target(J_fixed = np.diag([20.0, 25.0, 15.0]),    
                              Jt_est = np.diag([0.188065, 0.405968, 0.405968]),
                              Jt_diag_true = 100.0 * np.diag([0.146925, 0.417965, 0.435109]),
                              initialization = "random",
                              Gamma_scale = 10.0,
                              Kr = 30.0,
                              Kom = 10.0,
                              R_bt = np.eye(3),
                              q0 = None, w0 = None,
                              tf = 60.0,
                              control_torque_max = None,
                              frobErr = None,    
):
    np.random.seed(1110)

    # -------- Weaken feedback (so inertia matters) --------
    K_R = Kr * np.eye(3)
    K_Om = Kom * np.eye(3)

    # -------- Fast adaptation + normalization --------
    Gamma = Gamma_scale * np.eye(3)
    eps_reg = 1e-6

    # -------- Estimated target (principal frame) --------
    L0, Vt = np.linalg.eigh(0.5 * (Jt_est + Jt_est.T))  # eigenvalues, eigenvectors
    lambda0 = L0  # 1D array of eigenvalues

    # Rank-1 basis S_i = lambda0_i * (R v_i)(R v_i)^T  (in BODY frame)
    v1 = R_bt @ Vt[:, 0]
    v2 = R_bt @ Vt[:, 1]
    v3 = R_bt @ Vt[:, 2]
    S1 = lambda0[0] * np.outer(v1, v1)
    S2 = lambda0[1] * np.outer(v2, v2)
    S3 = lambda0[2] * np.outer(v3, v3)
    S = np.stack([S1, S2, S3], axis=2)  # (3,3,3)

    # -------- True docked inertia (plant) --------
    J_true = R_bt @ (Jt_diag_true) @ R_bt.T + J_fixed
    Jt_true = J_true - J_fixed

    # Best-fit alpha* in chosen basis (for plotting)
    A = np.column_stack([vec(S1), vec(S2), vec(S3)])  # (9,3)
    b = vec(Jt_true)                                  # (9,)
    alpha_star, *_ = np.linalg.lstsq(A, b, rcond=None)

    # -------- Multi-axis IC + dither torque --------

    if q0 is None:
        ang_deg = 25.0
        axis0 = np.array([1.0, 0.7, 0.4], dtype=float)
        axis0 = axis0 / np.linalg.norm(axis0)
        q0 = axang2quat(axis0, np.deg2rad(ang_deg))
        w0 = np.deg2rad(np.array([1.2, -0.8, 0.9], dtype=float))
    else:
        q0 = q0 / np.linalg.norm(q0)
        w0 = w0

    dith = {
        "A": 0,
        "w": 2.0 * np.pi * np.array([0.4, 0.7, 1.1], dtype=float),
    }
    alpha0 = 0 * np.array([1.0, 1.0, 1.0], dtype=float)

    x0 = np.concatenate([q0, w0, alpha0])
    control_torque_log = []

    # First run: "NN" init
    sol = solve_ivp(
        fun=lambda t, x: dyn_partial_adapt_v2(
            t, x, J_fixed, S, J_true, K_R, K_Om, Gamma, eps_reg, dith, control_torque_log, control_torque_max
        ),
        t_span=(0.0, tf),
        y0=x0,
        rtol=1e-12,
        atol=1e-12,
        dense_output=False,
    )

    t = sol.t
    x = sol.y.T  # (N,10)

    # Unpack
    q = x[:, 0:4].copy()
    for k in range(q.shape[0]):
        n_q = np.linalg.norm(q[k])
        q[k] /= max(n_q, 1e-12)
    w = x[:, 4:7]
    alpha = x[:, 7:10]

    N = t.size
    Jt_hat_diag = np.zeros((N, 3))
    J_hat_diag = np.zeros((N, 3))
    Frob_err = np.zeros(N)

    for k in range(N):
        Jt_hat_k = (
            alpha[k, 0] * S[:, :, 0]
            + alpha[k, 1] * S[:, :, 1]
            + alpha[k, 2] * S[:, :, 2]
        )
        Jt_hat_k = 0.5 * (Jt_hat_k + Jt_hat_k.T)
        Jt_hat_diag[k, :] = np.diag(Jt_hat_k)
        J_hat_diag[k, :] = np.diag(J_fixed + Jt_hat_k)
        Frob_err[k] = np.linalg.norm(Jt_hat_k - Jt_true, ord="fro")
    frobErr.append(Frob_err[-1])

    # print("Time when ||Jt_hat - Jt_true||_F < 1e-1 for " + initialization + " Initialization:", end=" ")
    # below_thresh_indices = np.where(Frob_err < 1e-1)[0]
    # if len(below_thresh_indices) > 0:
    #     first_below_index = below_thresh_indices[0]
    #     print(f"{t[first_below_index]:.2f} s")
    # else:
    #     print("Not reached within simulation time.")

    # # Store last Jt_hat from run
    # J_hat = Jt_hat_k.copy()

    # return J_hat, J_true, Jt_true

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--T', type=float, default=10.0)
    ap.add_argument('--dt', type=float, default=0.01)
    ap.add_argument('--trainN', type=int, default=2000)
    ap.add_argument('--valN', type=int, default=300)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--lamE', type=float, default=0.5)
    ap.add_argument('--lamD', type=float, default=1)
    ap.add_argument('--noise', type=float, default=0.002)
    ap.add_argument('--dmodel', type=int, default=64)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--force', action='store_true', help='Force dataset regeneration')
    ap.add_argument('--OOD', action='store_true', help='Validate on out-of-distribution dataset')
    ap.add_argument('--save',action='store_true', help='Save logs from training')
    ap.add_argument('--residual', type=str, default='tau', choices=['tau','wdot'])
    ap.add_argument('--data', type=str, default='data/shapes/', help='Path to dataset parent directory')
    ap.add_argument('--datagen', action='store_true', help='Only generate datasets and exit')
    ap.add_argument('--load', type=str, default="models/", help='Folder path to model checkpoint to load')
    args = ap.parse_args()

    if args.save:
        import os
        from datetime import datetime
        log_dir = "logs/selfsup_inference_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs(log_dir, exist_ok=True)
        import sys
        import shutil
        # copy this script to log dir
        shutil.copy(__file__, os.path.join(log_dir, os.path.basename(__file__)))
        # redirect stdout to log file
        log_file = open(os.path.join(log_dir, "inference_log.txt"), "w")
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
        with open(os.path.join(log_dir, "args.txt"), "w", encoding="utf-8") as f:
            for k,v in vars(args).items():
                f.write(f"{k}: {v}\n")

        print("Logging to", log_dir)


    I_pred_lstm_all, I_pred_mamba_all, I_pred_transformer_all, I_pred_tcn_all, I_true_all, q_all, w_all = main(validate_OOD=args.OOD,plot=False)
    
    frobErr_lstm = []
    frobErr_mamba = []
    frobErr_transformer = []
    frobErr_tcn = []
    frobErr_rand = []
    
    for i in range(5):
        I_pred_lstm = I_pred_lstm_all[i].cpu().numpy()
        I_pred_mamba = I_pred_mamba_all[i].cpu().numpy()
        I_pred_transformer = I_pred_transformer_all[i].cpu().numpy()
        I_pred_tcn = I_pred_tcn_all[i].cpu().numpy()
        I_true = I_true_all[i].cpu().numpy()
        qf_obs = q_all[i][-1].cpu().numpy()
        Rf_obs = quat_to_R_np(qf_obs)
        wf_obs = w_all[i][-1]
        J_fixed = [[1200,100,-200],[100,2200,300], [-200,300,3100]] # from spacecraft modeling attitude determination and control quaternion based approach pg 140
        Jt_est_rand = random_inertia()
        I_true= 10.0 * I_true
        init_euler_angle = quat_to_euler(qf_obs)

        control_torque_max = 60 * 5 # Nm - max slew rate of 1 deg/s per sec for 3200 kgm^2 chaser inertia is 60Nm


        qf_obs = qf_obs / np.linalg.norm(qf_obs)
        swapped = False
        # Enforce shortest-path: w >= 0
        if qf_obs[0] < 0.0:
            qf_obs = -qf_obs
            swapped = True
        theta_max = 2 * np.arctan2(np.sqrt(np.sum(qf_obs[1:4]**2)), qf_obs[0])  # radian
        
        theta_design = 0.52
        Kr = 0.6 * control_torque_max / theta_design
        
        damping_ratio = 0.5
        Kom = (2.0 * damping_ratio * np.sqrt(Kr * np.linalg.eigvals(J_fixed)))

        t_settling_no_sat = 8 * np.linalg.eigvals(J_fixed) / Kom
        t_settling_sat = 2 * np.sqrt(theta_max * np.linalg.eigvals(J_fixed) / control_torque_max)
        tf = 5 * 60

        T_theta = np.max(t_settling_no_sat)

        Gamma_scale = 5.0 / (T_theta)

        if swapped:
            qf_obs = -qf_obs

        # control_torque_max = None  # No saturation
        demo_adapt_partial_target(J_fixed = J_fixed, Jt_diag_true=I_true, Jt_est=I_pred_lstm,initialization="LSTM",Kom = Kom,Kr = Kr,Gamma_scale=Gamma_scale,R_bt=Rf_obs,q0=qf_obs,w0=wf_obs,tf=tf,control_torque_max=control_torque_max,frobErr=frobErr_lstm)

        demo_adapt_partial_target(J_fixed = J_fixed, Jt_diag_true=I_true, Jt_est=I_pred_mamba,initialization="Mamba",Kom = Kom,Kr = Kr,Gamma_scale=Gamma_scale,R_bt=Rf_obs,q0=qf_obs,w0=wf_obs,tf=tf,control_torque_max=control_torque_max,frobErr=frobErr_mamba)

        demo_adapt_partial_target(J_fixed = J_fixed, Jt_diag_true=I_true, Jt_est=I_pred_transformer,initialization="Transformer",Kom = Kom,Kr = Kr,Gamma_scale=Gamma_scale,R_bt=Rf_obs,q0=qf_obs,w0=wf_obs,tf=tf,control_torque_max=control_torque_max,frobErr=frobErr_transformer)

        demo_adapt_partial_target(J_fixed = J_fixed, Jt_diag_true=I_true, Jt_est=I_pred_tcn,initialization="TCN",Kom = Kom,Kr = Kr,Gamma_scale=Gamma_scale,R_bt=Rf_obs,q0=qf_obs,w0=wf_obs,tf=tf,control_torque_max=control_torque_max,frobErr=frobErr_tcn)

        demo_adapt_partial_target(J_fixed = J_fixed, Jt_diag_true=I_true, Jt_est=Jt_est_rand, initialization="random",Gamma_scale=Gamma_scale,Kom = Kom,Kr = Kr,R_bt=Rf_obs,q0=qf_obs,w0=wf_obs,tf=tf,control_torque_max=control_torque_max,frobErr=frobErr_rand)

    # plot frobErr
    plt.figure(figsize=(8,6))
    plt.semilogy(frobErr_lstm, label='LSTM')
    plt.semilogy(frobErr_mamba, label='Mamba')
    plt.semilogy(frobErr_transformer, label='Transformer')
    plt.semilogy(frobErr_tcn, label='TCN')
    plt.semilogy(frobErr_rand, label='Random Init')
    plt.xlabel('Validation Sample Index')
    plt.ylabel('Final Frobenius Error ||Jt_hat - Jt_true||_F')
    plt.title('Frobenius Error after Adaptive Control with Partial Inertia Knowledge')
    # plot mean lines
    plt.axhline(y=np.mean(frobErr_lstm), color='C0', linestyle='--', label='LSTM Mean')
    plt.axhline(y=np.mean(frobErr_mamba), color='C1', linestyle='--', label='Mamba Mean')
    plt.axhline(y=np.mean(frobErr_transformer), color='C2', linestyle='--', label='Transformer Mean')
    plt.axhline(y=np.mean(frobErr_tcn), color='C3', linestyle='--', label='TCN Mean')
    plt.axhline(y=np.mean(frobErr_rand), color='C4', linestyle='--', label='Random Init Mean')
    plt.legend()

    if not args.save:
        plt.show()
