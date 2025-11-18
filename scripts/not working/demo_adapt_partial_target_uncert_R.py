import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.linalg import expm

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

def skew(v):
    v = np.asarray(v).reshape(3,)
    return np.array([
        [0,     -v[2],  v[1]],
        [v[2],   0,    -v[0]],
        [-v[1],  v[0],  0   ]
    ])

def dyn_partial_adapt_R(t, x, J_fixed, Vt, lambda0, R_bt, J_true,
                        K_R, K_Om, Gamma, eps_reg, dith):
    """
    Python translation of MATLAB dyn_partial_adapt_R.
    All inputs must be numpy arrays.

    x = [q(4), w(3), alpha(3), theta(3)]
    """

    # Unpack ------------------------------------------------------------------
    q     = x[0:4]
    w     = x[4:7]
    alpha = x[7:10]
    theta = x[10:13]

    # Normalize quaternion
    nrm = np.linalg.norm(q)
    if nrm < 1e-12:
        nrm = 1e-12
    q = q / nrm

    # Orientation estimate R_est = exp(skew(theta)) * R_bt --------------------
    R_est = expm(skew(theta)) @ R_bt

    # Target inertia in principal frame --------------------------------------
    Jt_alpha = Vt @ np.diag(alpha * lambda0) @ Vt.T

    # Rotate into body frame
    Jt_body = R_est @ Jt_alpha @ R_est.T
    Jt_body = 0.5 * (Jt_body + Jt_body.T)

    # Total inertia estimate --------------------------------------------------
    Jhat = J_fixed + Jt_body

    # Small-angle error eR = 2*q_vec -----------------------------------------
    eR = 2.0 * q[1:4]

    # PD term -----------------------------------------------------------------
    uPD = -K_R @ eR - K_Om @ w

    # Dither torque -----------------------------------------------------------
    u_dith = dith["A"] * np.array([
        np.sin(dith["w"][0] * t),
        np.sin(dith["w"][1] * t + 0.7),
        np.sin(dith["w"][2] * t + 1.3)
    ])

    # Control torque: PD + gyro + dither -------------------------------------
    u = uPD + np.cross(w, Jhat @ w) + u_dith

    # True plant dynamics -----------------------------------------------------
    wdot = np.linalg.solve(J_true, (u - np.cross(w, J_true @ w)))

    # Quaternion kinematics qdot = 0.5 * W(q)*q -------------------------------
    wx, wy, wz = w
    W = np.array([
        [0,   -wx,  -wy,  -wz],
        [wx,   0,    wz,  -wy],
        [wy,  -wz,   0,    wx],
        [wz,   wy,  -wx,    0]
    ])
    qdot = 0.5 * (W @ q)

    # ---------------------- Adaptive Law ------------------------------------
    # torque prediction error (gyro terms cancel)
    tau_err = (uPD + u_dith) - (Jhat @ wdot)

    # Basis S_i (in current orientation) -------------------------------------
    v1 = R_est @ Vt[:, 0]
    v2 = R_est @ Vt[:, 1]
    v3 = R_est @ Vt[:, 2]

    S1 = lambda0[0] * np.outer(v1, v1)
    S2 = lambda0[1] * np.outer(v2, v2)
    S3 = lambda0[2] * np.outer(v3, v3)

    # Sensitivity T_k for theta ----------------------------------------------
    E1 = skew([1,0,0]);  E2 = skew([0,1,0]);  E3 = skew([0,0,1])
    T1 = E1 @ Jt_body - Jt_body @ E1
    T2 = E2 @ Jt_body - Jt_body @ E2
    T3 = E3 @ Jt_body - Jt_body @ E3

    # Regressor Phi = 3x6 -----------------------------------------------------
    Phi = np.column_stack([
        -(S1 @ wdot),
        -(S2 @ wdot),
        -(S3 @ wdot),
        -(T1 @ wdot),
        -(T2 @ wdot),
        -(T3 @ wdot)
    ])

    denom = np.trace(Phi.T @ Phi) + eps_reg

    # Parameter update
    param_dot = -Gamma @ (Phi.T @ tau_err) / denom

    # Split updates
    alpha_dot = param_dot[0:3]
    theta_dot = param_dot[3:6]

    # Full state derivative ---------------------------------------------------
    dx = np.concatenate([qdot, wdot, alpha_dot, theta_dot])

    return dx

def demo_adapt_partial_target_uncert_R(J_fixed = np.diag([20.0, 25.0, 15.0]),    
                              Jt_est = np.diag([0.188065, 0.405968, 0.405968]),
                              Jt_diag_true = 100.0 * np.diag([0.146925, 0.417965, 0.435109]),
                              initialization = "random",
                              Gamma_scale = 10.0,
                              Kr = 30.0,
                              Kom = 10.0,
                              R_bt = np.eye(3)

                              ):
    np.random.seed(1110)
    tf = 60.0    

    # -------- Weaken feedback (so inertia matters) --------
    K_R = Kr * np.eye(3)
    K_Om = Kom * np.eye(3)

    # -------- Fast adaptation + normalization --------
    Gamma = Gamma_scale * np.diag([0.3,0.3,0.3, 0.03,0.03,0.03])
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
    ang_deg = 25.0
    axis0 = np.array([1.0, 0.7, 0.4], dtype=float)
    axis0 = axis0 / np.linalg.norm(axis0)
    q0 = axang2quat(axis0, np.deg2rad(ang_deg))
    w0 = np.deg2rad(np.array([1.2, -0.8, 0.9], dtype=float))

    dith = {
        "A": 0,
        "w": 2.0 * np.pi * np.array([0.4, 0.7, 1.1], dtype=float),
    }
    alpha0 = np.array([1.0, 1.0, 1.0], dtype=float)
    theta0 = np.deg2rad(np.array([0,0,0], dtype=float))

    x0 = np.concatenate([q0, w0, alpha0,theta0])

    # First run: "NN" init
    sol = solve_ivp(
        fun=lambda t, x: dyn_partial_adapt_R(
            t, x, J_fixed, Vt, lambda0,R_bt, J_true, K_R, K_Om, Gamma, eps_reg, dith
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
    theta = x[:, 10:13]

    N = t.size
    Jt_hat_diag = np.zeros((N, 3))
    J_hat_diag = np.zeros((N, 3))
    Frob_err = np.zeros(N)

    for k in range(N):
        R_est = expm(skew(theta[k])) @ R_bt

        Jt_hat_alpha = (
            alpha[k, 0] * S[:, :, 0]
            + alpha[k, 1] * S[:, :, 1]
            + alpha[k, 2] * S[:, :, 2]
        )
        Jt_hat_k = R_est @ Jt_hat_alpha @ R_est.T
        Jt_hat_k = 0.5 * (Jt_hat_k + Jt_hat_k.T)
        Jt_hat_diag[k, :] = np.diag(Jt_hat_k)
        J_hat_diag[k, :] = np.diag(J_fixed + Jt_hat_k)
        Frob_err[k] = np.linalg.norm(Jt_hat_k - Jt_true, ord="fro")

    print("Time when ||Jt_hat - Jt_true||_F < 1e-1 for " + initialization + " Initialization:", end=" ")
    below_thresh_indices = np.where(Frob_err < 1e-1)[0]
    if len(below_thresh_indices) > 0:
        first_below_index = below_thresh_indices[0]
        print(f"{t[first_below_index]:.2f} s")
    else:
        print("Not reached within simulation time.")
    dtheta = 2.0 * q[:, 1:4]

    # ---------- Plots: NN estimation ----------
    colors_true = ['C0', 'C1', 'C2']

    plt.figure("Attitude & Rates (" + initialization + " Initialization)")
    plt.subplot(2, 1, 1)
    plt.plot(t, np.rad2deg(dtheta))
    plt.ylabel(r"$\delta\theta$ [deg]")
    plt.grid(True)
    plt.legend([r"$\delta\theta_x$", r"$\delta\theta_y$", r"$\delta\theta_z$"])
    plt.title("Attitude & Rates (" + initialization + " Initialization)")

    plt.subplot(2, 1, 2)
    plt.plot(t, np.rad2deg(w))
    plt.xlabel("t [s]")
    plt.ylabel(r"$\omega$ [deg/s]")
    plt.grid(True)
    plt.legend([r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"])

    plt.figure("Target inertia (" + initialization + " Initialization)")
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
    plt.title("Target inertia (" + initialization + " Initialization)")

    plt.figure("Total inertia diagonal (" + initialization + " Initialization)")
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
    plt.title("Total inertia diagonal (" + initialization + " Initialization)")

    plt.figure("Alpha vs best-fit (" + initialization + " Initialization)")
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
    plt.title("Alpha vs best-fit (" + initialization + " Initialization)")

    plt.figure()
    plt.plot(t, Frob_err)
    plt.xlabel("t [s]")
    plt.ylabel(r"$\|J_{\hat{t}} - J_{t}^{true}\|_F$")
    plt.title("Frobenius norm of inertia estimation error")
    plt.grid(True)

    # Store last Jt_hat from run
    J_hat = Jt_hat_k.copy()

    return J_hat, J_true, Jt_true


if __name__ == "__main__":
    J_fixed = [[1200,100,-200],[100,2200,300], [-200,300,3100]] # from spacecraft modeling attitude determination and control quaternion based approach pg 140
    Jt_diag_true = 1 * np.diag([1.135114, 2.450324, 2.450324])
    # the larger the target craft, the faster the NN adapts compared to rand initialization
    
    J_one_epoch = [0.31198,0.335318,0.352703]
    J_mamba = [0.125538,0.435155,0.439306]
    J_transformer = [0.298614,0.344211,0.357174]

    Jt_est = np.diag(J_transformer)
    Jt_est_rand = np.diag(np.random.rand(3))  

    R_bt = np.eye(3)
    # R_bt = rotx(4)@roty(2)@rotz(7)
    # R_bt = rotx(41)@roty(72)@rotz(67)

    J_hat_NN, J_true, Jt_true   = demo_adapt_partial_target_uncert_R(J_fixed = J_fixed, Jt_diag_true=Jt_diag_true, Jt_est=Jt_est,initialization="NN",Kom = 1000,Kr = 3000,R_bt=R_bt)
    J_hat_rand, J_true, Jt_true = demo_adapt_partial_target_uncert_R(J_fixed = J_fixed, Jt_diag_true=Jt_diag_true, Jt_est=Jt_est_rand, initialization="random",Gamma_scale=5.0,Kom = 1000,Kr = 3000,R_bt=R_bt)

    plt.show()
