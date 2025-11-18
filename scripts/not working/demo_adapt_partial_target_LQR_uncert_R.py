import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
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
def lqr(A, B, Q, R):
    """
    Continuous-time LQR: solve d/dt x = A x + B u, u = -K x
    Returns K, P, eigenvalues.
    """
    # Solve Riccati
    P = solve_continuous_are(A, B, Q, R)
    # K = R^-1 B^T P
    K = np.linalg.inv(R) @ (B.T @ P)
    # Closed-loop eigenvalues
    eigvals = np.linalg.eigvals(A - B @ K)
    return K, P, eigvals
def axis_angle_to_R(axis, angle):
    """Rodrigues rotation matrix for rotation of 'angle' about 'axis'."""
    axis = np.asarray(axis, dtype=float)
    axis /= (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    K = np.array([[0, -z, y],
                  [z, 0, -x],
                  [-y, x, 0]], dtype=float)
    I = np.eye(3)
    return I + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)

def dyn_partial_adapt_LQR(t, x, J_fixed, S, J_true, K, Gamma):
    """
    Dynamics for partial inertia adaptation with LQR and uncertain R.

    State x = [q(4); w(3); alpha(3)]
      q     : quaternion body->LVLH (4,)
      w     : body angular velocity (3,)
      alpha : target principal-moment scales (3,)

    J_fixed : (3,3) known chaser inertia
    S       : (3,3,3) basis matrices S_i for target inertia (built with R_hat)
    J_true  : (3,3) full true inertia (built with R_true)
    K       : (3,6) LQR gain, u_LQR = -K [delta_theta; w]
    Gamma   : (3,3) adaptation gain for alpha
    """
    q = x[0:4].copy()
    w = x[4:7].copy()
    alpha = x[7:10].copy()

    # Normalize quaternion
    nq = np.linalg.norm(q)
    if nq < 1e-12:
        q[:] = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        q /= nq

    # Estimated total inertia Jhat = J_fixed + sum_i alpha_i S_i
    Jhat = J_fixed.copy()
    for i in range(3):
        Jhat += alpha[i] * S[:, :, i]
    Jhat = 0.5 * (Jhat + Jhat.T)

    # Small-angle attitude error: delta_theta ≈ 2*q_vec
    eR = 2.0 * q[1:4]
    x_e = np.concatenate([eR, w])   # 6x1

    # LQR control
    u_LQR = -K @ x_e

    # Include gyroscopic term using Jhat so Jhat actually matters
    Jhatw = Jhat @ w
    u = u_LQR + np.cross(w, Jhatw)

    # True plant dynamics with J_true
    Jw_true = J_true @ w
    wdot = np.linalg.solve(J_true, u - np.cross(w, Jw_true))

    # Quaternion kinematics
    wx, wy, wz = w
    W = np.array([
        [0.0, -wx, -wy, -wz],
        [wx,  0.0,  wz, -wy],
        [wy, -wz,  0.0,  wx],
        [wz,  wy, -wx,  0.0]
    ])
    qdot = 0.5 * (W @ q)

    # --- Partial adaptation on target principal moments only ---
    # torque prediction error:
    # tau_err = u - (Jhat*wdot + w x (Jhat w)) = u_LQR - Jhat*wdot
    tau_err = u_LQR - (Jhat @ wdot)

    # Regressor: Y_i = -(S_i * wdot), alpha_dot = Gamma * Y^T * tau_err
    YTtau = np.zeros(3)
    for i in range(3):
        Si_wdot = S[:, :, i] @ wdot
        YTtau[i] = -Si_wdot.dot(tau_err)

    alpha_dot = -Gamma @ YTtau

    dx = np.zeros_like(x)
    dx[0:4] = qdot
    dx[4:7] = wdot
    dx[7:10] = alpha_dot
    return dx

def demo_adapt_partial_target_LQR(J_fixed = np.diag([20.0, 25.0, 15.0]),    
                              Jt_est = np.diag([0.188065, 0.405968, 0.405968]),
                              Jt_diag_true = 100.0 * np.diag([0.146925, 0.417965, 0.435109]),
                              initialization = "random",
                              Gamma_scale = 10.0,
                              R_true = np.eye(3),
                              q_att_weight = 1e3,
                              q_omega_weight = 10.0,
                              r_torque_weight = 1e-2
                              ):
    np.random.seed(0)

    # ----- Target inertia estimate in its own frame -----
    Jt_est = 0.5 * (Jt_est + Jt_est.T)

    # Principal axes + principal moments of estimated target inertia
    eigvals_est, Vt_est = np.linalg.eigh(Jt_est)
    lambda0 = eigvals_est.copy()  # 3x1 estimated principal moments

    # Estimated R_hat with some small error (e.g., 5 deg about z-axis)
    angle_err = np.deg2rad(5.0)
    axis_err = np.array([0.0, 0.0, 1.0])
    R_err = axis_angle_to_R(axis_err, angle_err)
    R_hat = R_true @ R_err   # controller uses R_hat; plant uses R_true

    # ----- Build true target inertia in body frame using R_true -----
    J_true_target = R_true @ Jt_diag_true @ R_true.T

    # Full true docked inertia
    J_true = J_fixed + J_true_target

    # ----- Build basis S_i using uncertain R_hat (controller's frame) -----
    # S_i = lambda0_i * (R_hat * v_i)(R_hat * v_i)^T
    v1 = R_hat @ Vt_est[:, 0]
    v2 = R_hat @ Vt_est[:, 1]
    v3 = R_hat @ Vt_est[:, 2]
    S1 = lambda0[0] * np.outer(v1, v1)
    S2 = lambda0[1] * np.outer(v2, v2)
    S3 = lambda0[2] * np.outer(v3, v3)
    S = np.stack([S1, S2, S3], axis=2)   # (3,3,3)

    # ----- Nominal inertia for LQR design -----
    # Use J_nom = J_fixed + estimated target from S with alpha = 1
    J_nom = J_fixed + (S1 + S2 + S3)

    # ----- LQR design on small-angle model -----
    # x = [delta_theta; omega]
    A = np.block([
        [np.zeros((3, 3)), np.eye(3)],
        [np.zeros((3, 3)), np.zeros((3, 3))]
    ])
    B = np.vstack([
        np.zeros((3, 3)),
        np.linalg.inv(J_nom)
    ])

    Q = np.block([
        [q_att_weight * np.eye(3), np.zeros((3, 3))],
        [np.zeros((3, 3)), q_omega_weight * np.eye(3)]
    ])
    R_lqr = r_torque_weight * np.eye(3)

    K, _, _ = lqr(A, B, Q, R_lqr)

    # ----- Adaptation gain -----
    Gamma = Gamma_scale * np.diag([1e-3, 1e-3, 1e-3])

    # ----- Initial state -----
    ang0 = np.deg2rad(20.0)
    axis0 = np.array([0.0, 1.0, 0.0])
    axis0 /= np.linalg.norm(axis0)
    q0 = np.empty(4)
    q0[0] = np.cos(ang0 / 2.0)
    q0[1:] = np.sin(ang0 / 2.0) * axis0

    w0 = np.deg2rad(np.array([1.0, -0.5, 0.8]))
    alpha0 = np.array([1.0, 1.0, 1.0])

    x0 = np.concatenate([q0, w0, alpha0])

    # ----- Integrate -----
    t_span = (0.0, 300.0)

    def ode_wrapper(t, x):
        return dyn_partial_adapt_LQR(t, x, J_fixed, S, J_true, K, Gamma)

    sol = solve_ivp(
        ode_wrapper,
        t_span,
        x0,
        method='RK45',
        rtol=1e-9,
        atol=1e-9,
    )

    t = sol.t
    X = sol.y.T  # (N, 10)

    q = X[:, 0:4].copy()
    w = X[:, 4:7].copy()
    alpha = X[:, 7:10].copy()

    # Normalize quaternions
    for k in range(q.shape[0]):
        nq = np.linalg.norm(q[k, :])
        if nq < 1e-12:
            q[k, :] = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            q[k, :] /= nq

    delta_theta = 2.0 * q[:, 1:4]

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
        Frob_err[k] = np.linalg.norm(Jt_hat_k - J_true_target, ord="fro")

    # ----- Plots -----
    plt.figure(figsize=(8, 9))

    plt.subplot(3, 1, 1)
    plt.plot(t, np.rad2deg(delta_theta))
    plt.ylabel(r'$\delta\theta$ [deg]')
    plt.legend([r'$\delta\theta_x$', r'$\delta\theta_y$', r'$\delta\theta_z$'])
    plt.grid(True)
    plt.title('Attitude error (LQR + partial J adaptation, uncertain R)')

    plt.subplot(3, 1, 2)
    plt.plot(t, np.rad2deg(w))
    plt.ylabel(r'$\omega$ [deg/s]')
    plt.legend([r'$\omega_x$', r'$\omega_y$', r'$\omega_z$'])
    plt.grid(True)
    plt.title('Body rates')

    plt.subplot(3, 1, 3)
    plt.plot(t, alpha)
    plt.xlabel('t [s]')
    plt.ylabel(r'$\alpha_i$')
    plt.legend([r'$\alpha_1$', r'$\alpha_2$', r'$\alpha_3$'])
    plt.grid(True)
    plt.title('Target principal-moment scales')

    plt.tight_layout()

    plt.figure()
    plt.plot(t, Frob_err)
    plt.xlabel("t [s]")
    plt.ylabel(r"$\|J_{\hat{t}} - J_{t}^{true}\|_F$")
    plt.title("Frobenius norm of inertia estimation error")
    plt.grid(True)


if __name__ == "__main__":
    J_fixed = [[1200,100,-200],[100,2200,300], [-200,300,3100]] # from spacecraft modeling attitude determination and control quaternion based approach pg 140
    Jt_diag_true = 100 * np.diag([1.135114, 2.450324, 2.450324])
    # the larger the target craft, the faster the NN adapts compared to rand initialization
    
    J_one_epoch = [0.31198,0.335318,0.352703]
    J_mamba = [0.125538,0.435155,0.439306]
    J_transformer = [0.298614,0.344211,0.357174]

    Jt_est = np.diag(J_transformer)
    Jt_est_rand = np.diag(np.random.rand(3))  

    R_bt = np.eye(3)
    # R_bt = rotx(4)@roty(2)@rotz(7)
    # R_bt = rotx(41)@roty(72)@rotz(67)

    demo_adapt_partial_target_LQR(J_fixed = J_fixed, Jt_diag_true=Jt_diag_true, Jt_est=Jt_est,initialization="NN",R_true=R_bt,Gamma_scale=100)
    demo_adapt_partial_target_LQR(J_fixed = J_fixed, Jt_diag_true=Jt_diag_true, Jt_est=Jt_est_rand, initialization="random",R_true=R_bt,Gamma_scale=10)
    plt.show()
