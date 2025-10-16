#!/usr/bin/env python3
# Rigid body tumbling: Euler dynamics + quaternion kinematics (Hamilton convention)

import numpy as np
from qutils.integrators import ode45
# ---------- Utilities ----------
def skew(v):
    x, y, z = v
    return np.array([[0, -z,  y],
                     [z,  0, -x],
                     [-y, x,  0]], dtype=float)

def omega_mat(w):
    wx, wy, wz = w
    return np.array([
        [0.0, -wx, -wy, -wz],
        [wx,  0.0,  wz, -wy],
        [wy, -wz,  0.0,  wx],
        [wz,  wy, -wx,  0.0]
    ], dtype=float)

def q_normalize(q):
    return q / np.linalg.norm(q)

def get_euler_angle_from_quaternion(q):
    """
    Convert quaternion (Hamilton convention) to 3-2-1 Euler angles (rad).
    q = [q0, q1, q2, q3] with q0 scalar.
    Returns (phi, theta, psi).
    """
    q0, q1, q2, q3 = q
    phi = np.arctan2(2*(q0*q1 + q2*q3), 1 - 2*(q1*q1 + q2*q2))
    theta = np.arcsin(np.clip(2*(q0*q2 - q3*q1), -1.0, 1.0))
    psi = np.arctan2(2*(q0*q3 + q1*q2), 1 - 2*(q2*q2 + q3*q3))
    return phi, theta, psi

# ---------- Dynamics ----------
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

# ---------- Integration wrapper ----------
def simulate_rigidbody(
    t_span,
    q0,
    w0,
    I_principal=(1.0, 2.0, 3.0),
    torque_fn=lambda t, q, w: np.zeros(3),
    rtol=1e-9,
    atol=1e-12,
):
    I_body = np.diag(I_principal)
    Iinv_body = np.diag(1.0 / np.asarray(I_principal, dtype=float))

    x0 = np.hstack([q_normalize(np.asarray(q0, dtype=float)), np.asarray(w0, dtype=float)])

    def rhs_renorm(t, x):
        # Drift control: renormalize q in-place every call to keep unit length
        q = x[:4]; w = x[4:]
        qn = q / max(1e-15, np.linalg.norm(q))
        x_fixed = np.hstack([qn, w])
        return euler_rhs(t, x_fixed, I_body, Iinv_body, torque_fn)

    t,y = ode45(
        rhs_renorm,
        t_span,
        x0,
        rtol=rtol,
        atol=atol,
        t_eval=np.linspace(t_span[0], t_span[1], 1000),
    )

    # Final renormalization on output
    qs = y[:, :4]
    norms = np.linalg.norm(qs, axis=0)
    y[:, :4] = qs / norms

    return t,y


# ---------- Example usage ----------
if __name__ == "__main__":

    t0 = 0
    tf = 30.0

    # Inertia with I1 < I2 < I3
    # I1, I2, I3 = 1.0, 2.0, 4.0
    I1, I2, I3 = 100.0, 200.0, 400.0
    I_body = np.diag([I1, I2, I3])

    # Initial attitude and angular velocity (body frame)
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    # Mostly about the intermediate axis (y), tiny x-perturbation to trigger flip
    w0 = np.array([0.02, 5.0, 0.0])  # rad/s


    # No external torque (tumbling)
    def zero_tau(t, q, w): return np.zeros(3)

    t,y = simulate_rigidbody(
        t_span=(t0, tf),
        q0=q0,
        w0=w0,
        I_principal=(I1, I2, I3),
        torque_fn=zero_tau,
        rtol=1e-9,
        atol=1e-12
    )

    q = y[:, :4]
    w = y[:, 4:]

    # Print a quick check: invariants under zero torque
    I = np.diag([I1, I2, I3])
    H = (I @ w.T).T
    T = 0.5 * np.sum(w * H, axis=1)

    print(f"||q|| deviation (max): {np.max(np.abs(np.linalg.norm(q, axis=1)-1)):.6e}")
    print(f"|H| (initial, final): {np.linalg.norm(H[0]):.6f}, {np.linalg.norm(H[-1]):.6f}")
    print(f"T  (initial, final): {T[0]:.6f}, {T[-1]:.6f}")

    # check euler angles
    euler_angles = np.zeros((t.size, 3))
    for i in range(t.size):
        euler_angles[i,:] = get_euler_angle_from_quaternion(q[i,:])

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(t, w[:, 0], label=r'$\omega_1$')
    plt.plot(t, w[:, 1], label=r'$\omega_2$')
    plt.plot(t, w[:, 2], label=r'$\omega_3$')
    plt.xlabel('t [s]')
    plt.ylabel('angular rate [rad/s]')
    plt.legend()
    plt.title('Torque-free rigid body (Euler)')

    plt.figure()
    plt.plot(t, q[:, 0], label='q0')
    plt.plot(t, q[:, 1], label='q1')
    plt.plot(t, q[:, 2], label='q2')
    plt.plot(t, q[:, 3], label='q3')
    plt.xlabel('t [s]')
    plt.ylabel('quaternion')
    plt.legend()
    plt.title('Attitude quaternion')

    plt.figure()
    plt.plot(t, euler_angles[:, 0]*180.0/np.pi, label=r'$\phi$')
    plt.plot(t, euler_angles[:, 1]*180.0/np.pi, label=r'$\theta$')
    plt.plot(t, euler_angles[:, 2]*180.0/np.pi, label=r'$\psi$')
    plt.xlabel('t [s]')
    plt.ylabel('Euler angles [deg]')
    plt.legend()
    plt.title('3-2-1 Euler angles')



    # ---- 3D tumbling visualization (matplotlib + quaternions) ----
    from matplotlib import animation
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    def quat_to_R(q):
        q0,q1,q2,q3 = q
        # Hamilton, body->inertial
        s = q0*q0 - q1*q1 - q2*q2 - q3*q3
        x2,y2,z2 = 2*q1*q1, 2*q2*q2, 2*q3*q3
        xy,xz,yz = 2*q1*q2, 2*q1*q3, 2*q2*q3
        wx,wy,wz = 2*q0*q1, 2*q0*q2, 2*q0*q3
        return np.array([
            [s + x2,   xy - wz,  xz + wy],
            [xy + wz,  s + y2,   yz - wx],
            [xz - wy,  yz + wx,  s + z2]
        ], dtype=float)

    def make_box_wireframe(a,b,c):
        # 8 vertices in body frame
        V = np.array([[ sx*a, sy*b, sz*c ]
                    for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)], dtype=float)
        # 12 edges as pairs of vertex indices
        def idx(sx,sy,sz): return (0 if sx<0 else 4) + (0 if sy<0 else 2) + (0 if sz<0 else 1)
        edges = []
        for sx in (-1,1):
            for sy in (-1,1):
                edges.append((idx(sx,sy,-1), idx(sx,sy, 1)))
        for sx in (-1,1):
            for sz in (-1,1):
                edges.append((idx(sx,-1,sz), idx(sx, 1,sz)))
        for sy in (-1,1):
            for sz in (-1,1):
                edges.append((idx(-1,sy,sz), idx( 1,sy,sz)))
        return V, np.array(edges, dtype=int)

    def segments_from_VE(V, E):
        return np.stack([V[E[:,0]], V[E[:,1]]], axis=1)  # (n_edges, 2, 3)

    def equalize_axes(ax, pts):
        mins = pts.min(0)
        maxs = pts.max(0)
        center = (mins + maxs)/2.0
        r = (maxs - mins).max()/2.0
        ax.set_xlim(center[0]-r, center[0]+r)
        ax.set_ylim(center[1]-r, center[1]+r)
        ax.set_zlim(center[2]-r, center[2]+r)

    # Choose a body shape. Use semi-axes inversely proportional to sqrt(I) (inertia ellipsoid, up to scale).
    I1, I2, I3 = np.diag(np.diag(np.diag(np.diag(np.diag(np.diag(np.array([I1, I2, I3])))))))  # keep user vars
    inv_sqrtI = 1.0/np.sqrt(np.array([I1, I2, I3], dtype=float))
    a,b,c = (0.5 * inv_sqrtI / inv_sqrtI.max())  # scaled nicely

    # Body wireframe and axes (in body frame)
    V_body, E = make_box_wireframe(a,b,c)
    seg_body0 = segments_from_VE(V_body, E)

    # Body axes triad in body frame
    axis_len = 0.8*max(a,b,c)
    ax_lines_body = np.array([
        [[0,0,0],[axis_len,0,0]],  # x
        [[0,0,0],[0,axis_len,0]],  # y
        [[0,0,0],[0,0,axis_len]],  # z
    ], dtype=float)

    # Precompute inertial angular momentum (for reference): H_I = R * (I_body * w_body)
    I_body = np.diag([I1, I2, I3])
    H_I0 = quat_to_R(q[0]) @ (I_body @ w[0])
    H_I0 = H_I0 / (np.linalg.norm(H_I0) + 1e-15) * (1.2*axis_len)

    fig = plt.figure()
    ax3d = fig.add_subplot(111, projection='3d')
    ax3d.set_box_aspect((1,1,1))
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.set_title('Tumbling rigid body')
    # remove numbered ticks
    ax3d.set_xticks([])
    ax3d.set_yticks([])
    ax3d.set_zticks([])

    # Line collections for box and axes
    lc_body = Line3DCollection(seg_body0, linewidths=1.5)
    ax3d.add_collection3d(lc_body)

    # Axes: three separate lines so we can update them easily
    axis_lines = [ax3d.plot([], [], [])[0] for _ in range(3)]

    # Inertial angular momentum arrow (fixed in I-frame)
    # H_line = ax3d.plot([0, H_I0[0]], [0, H_I0[1]], [0, H_I0[2]])[0]

    # Set initial limits
    all_pts0 = seg_body0.reshape(-1,3)
    equalize_axes(ax3d, np.vstack([all_pts0, np.array([H_I0])]))
    def set_axes_equal(ax, pts):
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        center = (mins + maxs) / 2.0
        radius = (maxs - mins).max() / 2.0
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)

    def update(frame_idx):
        R = quat_to_R(q[frame_idx])  # body->inertial
        V_I = (R @ V_body.T).T       # rotate vertices
        seg_I = segments_from_VE(V_I, E)
        lc_body.set_segments(seg_I)

        # update axis triads
        axes_I = (R @ ax_lines_body.reshape(-1,3).T).T.reshape(3,2,3)
        for k in range(3):
            p0, p1 = axes_I[k,0], axes_I[k,1]
            axis_lines[k].set_data([p0[0], p1[0]], [p0[1], p1[1]])
            axis_lines[k].set_3d_properties([p0[2], p1[2]])

        # ---- recenter the plot here ----
        pts = seg_I.reshape(-1, 3)
        set_axes_equal(ax3d, pts)

        return [lc_body, *axis_lines]

    ani = animation.FuncAnimation(fig, update, frames=len(t), interval=1000*(t[1]-t[0]), blit=False)
    plt.show()
