#!/usr/bin/env python3
# Rigid body tumbling dataset generation

import numpy as np
from qutils.integrators import ode45
# ---------- Utilities ----------
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
    t_eval=None,
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
    argparser.add_argument("--systems", type=int, default=10000,
                           help="Number of systems to simulate (default: 10000).")
    argparser.add_argument("--tfinal", type=float, default=60.0,
                           help="Final time for the simulation in seconds (default: 60.0 seconds).")
    argparser.add_argument("--hz", type=int, default=1,
                           help="Sampling frequency in Hz (default: 1 Hz).")
    argparser.add_argument("--output", type=str, default="data/tumbling_dataset.npz",
                           help="Output file name for the dataset (default: tumbling_dataset.npz).")
    args = argparser.parse_args()

    np.random.seed(42)

    t0 = 0
    tf = args.tfinal
    num_steps = int((tf - t0) * args.hz) + 1

    t_eval = np.linspace(t0, tf, num_steps)

    # Inertia tensor in body frame (principal axes, kg*m^2)

    # pick from a set of actual shapes
    pickShape = np.random.choice(['box', 'ellipsoid', 'cylinder', 'cone', 'rod'], size=args.systems)
    I_list = []
    q_list = []
    w_list = []
    for shape in pickShape:
        m = np.random.uniform(1.0, 10.0)

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
            I2 = I1
            I3 = (1/12) * m * (L)**2
        I_list.append([I1, I2, I3])

        # Initial attitude and angular velocity (body frame)
        q0 = np.random.randn(4)
        q0 /= np.linalg.norm(q0)
        q_list.append(q0)
        # Random angular velocity
        w0 = np.random.uniform(-5.0, 5.0, size=3)
        w_list.append(w0)

    X = np.zeros((args.systems, num_steps, 7)) # time series of (q,w)
    Y = I_list # inertia tensors
    H = np.zeros((args.systems, num_steps,1)) # angular momentum magnitude
   

    t_array = np.zeros((args.systems, num_steps)) # time series of time
    T = np.zeros((args.systems, num_steps,1)) # total energy 

    for i in range(args.systems):
        I1, I2, I3 = I_list[i]
        q0 = q_list[i]
        w0 = w_list[i]

        t,y = simulate_rigidbody(
            t_span=(t0, tf),
            q0=q0,
            w0=w0,
            I_principal=(I1, I2, I3),
            rtol=1e-9,
            atol=1e-12,
            t_eval=t_eval
        )

        q = y[:, :4]
        w = y[:, 4:]
        X[i,:,:] = y
        t_array[i,:] = t[:,0]
        I = np.diag([I1, I2, I3])
        H_i = (I @ w.T).T
        H[i,:,0] = np.linalg.norm(H_i, axis=1)
        T_i = 0.5 * np.sum(w * H_i, axis=1)
        T[i,:,0] = T_i
        # recalc euler_rhs for entire time series to save
        euler_rhs_series = np.zeros_like(y)
        for j in range(num_steps):
            euler_rhs_series[j,:] = euler_rhs(t[j], y[j,:], I, np.diag(1.0/np.array([I1,I2,I3])), lambda t,q,w: np.zeros(3))
        print(f"Simulated system {i+1}/{args.systems}", end='\r')
    
    print(f"\nSaving dataset to {args.output} ...")
    np.savez_compressed(args.output, X=X, Y=Y, t_array=t_array, H=H, T=T,shapes=pickShape,accelerations=euler_rhs_series)
    print("Done.")
