import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

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


def dyn_partial_adapt_v2(t, x, J_fixed, S, J_true, K_R, K_Om, Gamma, eps_reg, dith):
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
    alpha_dot = -Gamma @ (Phi.T @ tau_err) / denom

    dx = np.concatenate([qdot, wdot, alpha_dot])
    return dx


def demo_adapt_partial_target(J_fixed = np.diag([20.0, 25.0, 15.0]),    
                              Jt_est = np.diag([0.188065, 0.405968, 0.405968]),
                              Jt_diag_true = 100.0 * np.diag([0.146925, 0.417965, 0.435109]),
                              ):
    np.random.seed(0)
    tf = 100.0    

    # -------- Weaken feedback (so inertia matters) --------
    K_R = 30.0 * np.eye(3)
    K_Om = 10.0 * np.eye(3)

    # -------- Fast adaptation + normalization --------
    Gamma = 10.0 * np.diag([0.3, 0.3, 0.3])
    eps_reg = 1e-6

    # -------- Estimated target (principal frame) --------
    L0, Vt = np.linalg.eigh(0.5 * (Jt_est + Jt_est.T))  # eigenvalues, eigenvectors
    lambda0 = L0  # 1D array of eigenvalues

    # Docked orientation (body<-target)
    R_bt = np.eye(3)
    # R_bt = rotx(4)@roty(2)@rotz(7)
    # R_bt = rotx(41)@roty(72)@rotz(67)

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
    ang_deg = 25.0
    axis0 = np.array([1.0, 0.7, 0.4], dtype=float)
    axis0 = axis0 / np.linalg.norm(axis0)
    q0 = axang2quat(axis0, np.deg2rad(ang_deg))
    w0 = np.deg2rad(np.array([1.2, -0.8, 0.9], dtype=float))

    dith = {
        "A": 0.0,
        "w": 2.0 * np.pi * np.array([0.4, 0.7, 1.1], dtype=float),
    }
    alpha0 = np.array([1.0, 1.0, 1.0], dtype=float)

    x0 = np.concatenate([q0, w0, alpha0])

    # First run: "NN" init
    sol = solve_ivp(
        fun=lambda t, x: dyn_partial_adapt_v2(
            t, x, J_fixed, S, J_true, K_R, K_Om, Gamma, eps_reg, dith
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

    dtheta = 2.0 * q[:, 1:4]

    # ---------- Plots: NN estimation ----------
    colors_true = ['C0', 'C1', 'C2']

    plt.figure("Attitude & Rates (NN Estimation)")
    plt.subplot(2, 1, 1)
    plt.plot(t, np.rad2deg(dtheta))
    plt.ylabel(r"$\delta\theta$ [deg]")
    plt.grid(True)
    plt.legend([r"$\delta\theta_x$", r"$\delta\theta_y$", r"$\delta\theta_z$"])
    plt.title("Attitude & Rates (NN Estimation)")

    plt.subplot(2, 1, 2)
    plt.plot(t, np.rad2deg(w))
    plt.xlabel("t [s]")
    plt.ylabel(r"$\omega$ [deg/s]")
    plt.grid(True)
    plt.legend([r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"])

    plt.figure("Target inertia (NN Estimation)")
    plt.plot(t, Jt_hat_diag)
    Jt_true_diag = np.diag(Jt_true)
    for j in range(3):
        plt.axhline(Jt_true_diag[j], linestyle=":",color = colors_true[j])
    plt.xlabel("t [s]")
    plt.ylabel(r"diag($J_{target}$) [kg m$^2$]")
    plt.grid(True)
    plt.legend(
        [
            r"$\hat J_{t,11}$",
            r"$\hat J_{t,22}$",
            r"$\hat J_{t,33}$",
            r"$J_{t,11}^{true}$",
            r"$J_{t,22}^{true}$",
            r"$J_{t,33}^{true}$",
        ]
    )
    plt.title("Target inertia (NN Estimation)")

    plt.figure("Total inertia diagonal (NN Estimation)")
    plt.plot(t, J_hat_diag)
    J_true_diag = np.diag(J_true)
    for j in range(3):
        plt.axhline(J_true_diag[j], linestyle=":",color=colors_true[j])
    plt.xlabel("t [s]")
    plt.ylabel(r"diag($J_{total}$) [kg m$^2$]")
    plt.grid(True)
    plt.legend(
        [
            r"$\hat J_{11}$",
            r"$\hat J_{22}$",
            r"$\hat J_{33}$",
            r"$J_{11}^{true}$",
            r"$J_{22}^{true}$",
            r"$J_{33}^{true}$",
        ]
    )
    plt.title("Total inertia diagonal (NN Estimation)")

    plt.figure("Alpha vs best-fit (NN Estimation)")
    plt.plot(t, alpha)
    for j in range(3):
        plt.axhline(alpha_star[j], linestyle=":",color=colors_true[j])
    plt.xlabel("t [s]")
    plt.ylabel(r"$\alpha_i$")
    plt.grid(True)
    plt.legend(
        [
            r"$\alpha_1$",
            r"$\alpha_2$",
            r"$\alpha_3$",
            r"$\alpha_1^\ast$",
            r"$\alpha_2^\ast$",
            r"$\alpha_3^\ast$",
        ]
    )
    plt.title("Alpha vs best-fit (Random Initialization)")

    plt.figure()
    plt.plot(t, Frob_err)
    plt.xlabel("t [s]")
    plt.ylabel(r"$\|J_{\hat{t}} - J_{t}^{true}\|_F$")
    plt.title("Frobenius norm of inertia estimation error")
    plt.grid(True)

    # Store last Jt_hat from NN run
    J_hat_NN = Jt_hat_k.copy()

    # ================== Second run: random J_target init ==================

    # Estimated target (principal frame): random diagonal
    Jt_est2 = np.diag(np.random.rand(3))
    L0, Vt = np.linalg.eigh(0.5 * (Jt_est + Jt_est.T))  # eigenvalues, eigenvectors

    L0_2,Vt2 = np.linalg.eigh(0.5 * (Jt_est2 + Jt_est2.T))
    lambda0_2 = np.diag(L0_2)

    v1 = R_bt @ Vt2[:, 0]
    v2 = R_bt @ Vt2[:, 1]
    v3 = R_bt @ Vt2[:, 2]
    S1 = lambda0_2[0] * np.outer(v1, v1)
    S2 = lambda0_2[1] * np.outer(v2, v2)
    S3 = lambda0_2[2] * np.outer(v3, v3)
    S = np.stack([S1, S2, S3], axis=2)

    # Best-fit alpha* in chosen basis (for plotting)
    A = np.column_stack([vec(S1), vec(S2), vec(S3)])
    b = vec(Jt_true)
    alpha_star, *_ = np.linalg.lstsq(A, b, rcond=None)

    # Weaken feedback (more) so inertia matters
    K_R = 0.3 * np.eye(3)
    K_Om = 0.1 * np.eye(3)

    # Fast adaptation again
    Gamma = 10.0 * np.diag([0.3, 0.3, 0.3])
    eps_reg = 1e-6

    ang_deg = 25.0
    axis0 = np.array([1.0, 0.7, 0.4], dtype=float)
    axis0 = axis0 / np.linalg.norm(axis0)
    q0 = axang2quat(axis0, np.deg2rad(ang_deg))
    w0 = np.deg2rad(np.array([1.2, -0.8, 0.9], dtype=float))
    alpha0 = np.array([1.0, 1.0, 1.0], dtype=float)

    x0 = np.concatenate([q0, w0, alpha0])

    # dith is kept as before but amplitude is effectively zero (no new setting)
    sol2 = solve_ivp(
        fun=lambda t, x: dyn_partial_adapt_v2(
            t, x, J_fixed, S, J_true, K_R, K_Om, Gamma, eps_reg, dith
        ),
        t_span=(0.0, tf),
        y0=x0,
        rtol=1e-12,
        atol=1e-12,
        dense_output=False,
    )

    t2 = sol2.t
    x2 = sol2.y.T

    q2 = x2[:, 0:4].copy()
    for k in range(q2.shape[0]):
        n_q = np.linalg.norm(q2[k])
        q2[k] /= max(n_q, 1e-12)
    w2 = x2[:, 4:7]
    alpha2 = x2[:, 7:10]

    N2 = t2.size
    Jt_hat_diag2 = np.zeros((N2, 3))
    J_hat_diag2 = np.zeros((N2, 3))
    Frob_err2 = np.zeros(N2)

    for k in range(N2):
        Jt_hat_k2 = (
            alpha2[k, 0] * S[:, :, 0]
            + alpha2[k, 1] * S[:, :, 1]
            + alpha2[k, 2] * S[:, :, 2]
        )
        Jt_hat_k2 = 0.5 * (Jt_hat_k2 + Jt_hat_k2.T)
        Jt_hat_diag2[k, :] = np.diag(Jt_hat_k2)
        J_hat_diag2[k, :] = np.diag(J_fixed + Jt_hat_k2)
        Frob_err2[k] = np.linalg.norm(Jt_hat_k2 - Jt_true, ord="fro")

    dtheta2 = 2.0 * q2[:, 1:4]

    # ---------- Plots: random initialization ----------
    plt.figure("Attitude & Rates (Random Initialization)")
    plt.subplot(2, 1, 1)
    plt.plot(t2, np.rad2deg(dtheta2))
    plt.ylabel(r"$\delta\theta$ [deg]")
    plt.grid(True)
    plt.legend([r"$\delta\theta_x$", r"$\delta\theta_y$", r"$\delta\theta_z$"])
    plt.title("Attitude & Rates (Random Initialization)")

    plt.subplot(2, 1, 2)
    plt.plot(t2, np.rad2deg(w2))
    plt.xlabel("t [s]")
    plt.ylabel(r"$\omega$ [deg/s]")
    plt.grid(True)
    plt.legend([r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"])

    plt.figure("Target inertia (Random Initialization)")
    plt.plot(t2, Jt_hat_diag2)
    Jt_true_diag = np.diag(Jt_true)
    for j in range(3):
        plt.axhline(Jt_true_diag[j], linestyle=":",color= colors_true[j])
    plt.xlabel("t [s]")
    plt.ylabel(r"diag($J_{target}$) [kg m$^2$]")
    plt.grid(True)
    plt.legend(
        [
            r"$\hat J_{t,11}$",
            r"$\hat J_{t,22}$",
            r"$\hat J_{t,33}$",
            r"$J_{t,11}^{true}$",
            r"$J_{t,22}^{true}$",
            r"$J_{t,33}^{true}$",
        ]
    )
    plt.title("Target inertia (Random Initialization)")

    plt.figure("Total inertia diagonal (Random Initialization)")
    plt.plot(t2, J_hat_diag2)
    J_true_diag = np.diag(J_true)
    for j in range(3):
        plt.axhline(J_true_diag[j], linestyle=":",color = colors_true[j])
    plt.xlabel("t [s]")
    plt.ylabel(r"diag($J_{total}$) [kg m$^2$]")
    plt.grid(True)
    plt.legend(
        [
            r"$\hat J_{11}$",
            r"$\hat J_{22}$",
            r"$\hat J_{33}$",
            r"$J_{11}^{true}$",
            r"$J_{22}^{true}$",
            r"$J_{33}^{true}$",
        ]
    )
    plt.title("Total inertia diagonal (Random Initialization)")

    plt.figure("Alpha vs best-fit (Random Initialization)")
    plt.plot(t2, alpha2)
    for j in range(3):
        plt.axhline(alpha_star[j], linestyle=":",color= colors_true[j])
    plt.xlabel("t [s]")
    plt.ylabel(r"$\alpha_i$")
    plt.grid(True)
    plt.legend(
        [
            r"$\alpha_1$",
            r"$\alpha_2$",
            r"$\alpha_3$",
            r"$\alpha_1^\ast$",
            r"$\alpha_2^\ast$",
            r"$\alpha_3^\ast$",
        ]
    )
    plt.title("Alpha vs best-fit (Random Initialization)")

    plt.figure()
    plt.plot(t2, Frob_err2)
    plt.xlabel("t [s]")
    plt.ylabel(r"$\|J_{\hat{t}} - J_{t}^{true}\|_F$")
    plt.title("Frobenius norm of inertia estimation error")
    plt.grid(True)

    # Last Jt_hat from random run
    J_hat_rand = Jt_hat_k2.copy()

    return J_hat_NN, J_hat_rand, J_true, Jt_true


if __name__ == "__main__":
    J_hat_NN, J_hat_rand, J_true, Jt_true = demo_adapt_partial_target()
    plt.show()
