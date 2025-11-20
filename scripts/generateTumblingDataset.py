#!/usr/bin/env python3
# Rigid body tumbling dataset generation

import numpy as np
from qutils.integrators import ode45
# ---------- Utilities ----------
def add_sensor_noise(qs_t, ws_t, gyro_std=0.002, att_std_deg=0.0):
    rng = np.random.default_rng()
    ws_noisy = ws_t + rng.normal(0.0, gyro_std, ws_t.shape)

    if att_std_deg <= 0.0:
        return qs_t.copy(), ws_noisy

    att_std = np.deg2rad(att_std_deg)
    qs_noisy = qs_t.copy()
    T = qs_t.shape[0]

    for k in range(T):
        # Random small rotation
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis) + 1e-12
        ang = rng.normal(0.0, att_std)
        dq = np.array([np.cos(ang/2), *(np.sin(ang/2)*axis)], float)

        # Quaternion multiplication dq ⊗ q
        w1,x1,y1,z1 = dq
        w2,x2,y2,z2 = qs_noisy[k]
        qs_noisy[k] = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], float)

        # Normalize
        qs_noisy[k] /= np.linalg.norm(qs_noisy[k]) + 1e-12

    return qs_noisy, ws_noisy

def project_spd_and_normalize(I, min_eig=1e-12):
    # I: (3,3) possibly noisy; make SPD, then trace-normalize
    S = 0.5 * (I + I.T)
    w, V = np.linalg.eigh(S)
    w = np.clip(w, min_eig, None)
    S_spd = (V * w) @ V.T               # V @ diag(w) @ V.T with broadcasting
    return S_spd / np.trace(S_spd)

def rand_rotation_QR():
    A = np.random.randn(3,3)
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:,0] = -Q[:,0]
    return Q
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

def inertia_from_principal(I_principal,q):
    # I_principal: iterable of (I1, I2, I3) or a 3x3 diagonal matrix
    if np.array(I_principal).shape == (3,):
        D = np.diag(I_principal)
    else:
        D = np.diag(np.diag(I_principal))
    R = quat_to_R_np(q)
    return R @ D @ R.T  # SPD, same eigenvalues, random body axes from q

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
    I_body,
    torque_fn=lambda t, q, w: np.zeros(3),
    rtol=1e-9,
    atol=1e-12,
    t_eval=None,
):
    Iinv_body = np.linalg.inv(I)

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
        t_eval=t_eval,
    )

    # Final renormalization on output
    qs = y[:, :4]
    norms = np.linalg.norm(qs, axis=0)
    y[:, :4] = qs / norms

    return t,y


# ---------- Example usage ----------
if __name__ == "__main__":

    import argparse

    argparser = argparse.ArgumentParser(description="Simulate and generate dataset for tumbling rigid body dynamics.")
    argparser.add_argument("--systems", type=int, default=2000,
                           help="Number of systems to simulate (default: 2000).")
    argparser.add_argument('--validation', type=int, default=300,help="Number of validation sets to train (default 300).")
    argparser.add_argument("--T", type=float, default=3.0,
                           help="Final time for the simulation in seconds (default: 10.0 seconds).")
    argparser.add_argument("--hz", type=int, default=100,
                           help="Sampling frequency in Hz (default: 1 Hz).")
    argparser.add_argument("--output", type=str, default="data/shapes/",
                           help="Output parent directory for the dataset (default: data/shapes/).")
    argparser.add_argument('--noise', type=float, default=0.002)

    args = argparser.parse_args()

    np.random.seed(42)

    total_systems = args.systems + args.validation

    t0 = 0
    tf = args.T
    num_steps = int((tf - t0) * args.hz)

    t_eval = np.linspace(t0, tf, num_steps)

    # Inertia tensor in body frame (principal axes, kg*m^2)

    # pick from a set of actual shapes
    pickShape = np.random.choice(['box', 'ellipsoid', 'cylinder', 'cone'], size=total_systems)
    I_list = []
    q0_list = []
    w0_list = []
    q_list = []
    w_list = []
    I_list_real = []

    m_list = []
    for shape in pickShape:
        m = np.random.uniform(1.0, 10.0)
        
        q0 = np.random.randn(4)
        q0 /= np.linalg.norm(q0)
        q0_list.append(q0)
        # Random angular velocity
        w0 = np.random.uniform(0.2, 5.0, size=3)
        w0_list.append(w0)
        m_list.append(m)

        if shape == 'box':
        # box with side lengths (a,b,c)
            a = np.random.uniform(0.1, 2.0)
            b = np.random.uniform(0.1, 2.0)
            c = np.random.uniform(0.1, 2.0)
            I1 = (1/12) * m * (b**2 + c**2)
            I2 = (1/12) * m * (a**2 + c**2)
            I3 = (1/12) * m * (a**2 + b**2)
        elif shape == 'ellipsoid':
         # solid ellipsoid with semi-axis lengths (a,b,c)
            a = np.random.uniform(0.1, 2.0)
            b = np.random.uniform(0.1, 2.0)
            c = np.random.uniform(0.1, 2.0)
            I1 = (1/5) * m * (b**2 + c**2)
            I2 = (1/5) * m * (a**2 + c**2)
            I3 = (1/5) * m * (a**2 + b**2)
        elif shape == 'cylinder':
        # solid cylinder with radius r and height h
            r = np.random.uniform(0.1, 1.0)
            h = np.random.uniform(0.1, 3.0)
            I1 = (1/12) * m * (3*r**2 + h**2)
            I2 = I1
            I3 = (1/2) * m * r**2
        elif shape == 'cone':
        # right circular cone with base radius r and height h
            r = np.random.uniform(0.1, 1.0)
            h = np.random.uniform(0.1, 3.0)
            I1 = (3/5) * m * h**2 + (3/20) * m * r**2
            I2 = I1
            I3 = (3/10) * m * r**2
        elif shape == 'rod':
        # slender rod about center
            L = np.random.uniform(0.1, 3.0)
            I1 = (1/12) * m * L**2
            I2 = 0
            I3 = (1/12) * m * (L)**2
        I_principal = np.diag([I1, I2, I3]) # generate random principal inertia values from shape
        I_rand = inertia_from_principal(I_principal,q0) # random orientation
        I_rand = project_spd_and_normalize(I_rand) # ensure SPD and trace=1
        
        I_list.append(I_rand)
        I_list_real.append(I_principal)    
        # Initial attitude and angular velocity (body frame)

    I_true = I_list   

    dt = 1/args.hz

    for i in range(total_systems):
        I1, I2, I3 = I_list[i]
        q0 = q0_list[i]
        w0 = w0_list[i]
        I = I_list[i]

        t,y = simulate_rigidbody(
            t_span=(t0, tf),
            q0=q0,
            w0=w0,
            I_body=I,
            rtol=1e-9,
            atol=1e-12,
            t_eval=t_eval
        )

        q = y[:, :4]
        w = y[:, 4:]


        q,w = add_sensor_noise(q, w, gyro_std=args.noise, att_std_deg=args.noise)

        q_list.append(q)
        w_list.append(w)
        print(f"System {i+1}/{total_systems} simulated.", end='\r')
    print()
    # ---------- Save dataset ----------
    # ensure output directory exists
    import os
    os.makedirs(args.output+str(args.systems), exist_ok=True)
    os.makedirs(args.output+str(args.validation), exist_ok=True)
    print(f"\nSaving training dataset to {args.output} ...")
    # save q,w,I_true,m,dt
    filePath_train = args.output+str(args.systems) + "/" + "self-sup-" + "train_" + str(tf) + ".npz"
    filePath_val = args.output+str(args.systems) + "/" + "self-sup-" + "val_" + str(tf) + ".npz"
    np.savez_compressed(
        filePath_train,
        q=np.array(q_list[:args.systems]),
        w=np.array(w_list[:args.systems]),
        I_true=np.array(I_list[:args.systems]),
        m=np.array(m_list[:args.systems]),
        dt=dt,
        I_true_real=np.array(I_list_real[:args.systems]),
        shape=pickShape[:args.systems]
    )

    print(f"Saving validation dataset to {args.output} ...")
    np.savez_compressed(
        filePath_val,
        q=np.array(q_list[args.systems:]),
        w=np.array(w_list[args.systems:]),
        I_true=np.array(I_list[args.systems:]),
        m=np.array(m_list[args.systems:]),
        dt=dt,
        I_true_real=np.array(I_list_real[args.systems:]),
        shape=pickShape[args.systems:]

    )

    print("Done.")
