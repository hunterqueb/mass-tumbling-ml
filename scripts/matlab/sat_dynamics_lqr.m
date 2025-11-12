function dx = sat_dynamics_lqr(~, x, J, K)
% x = [q; w]
% q: 4x1 quaternion (body->LVLH)
% w: 3x1 body angular velocity [rad/s]

    q = x(1:4);
    w = x(5:7);

    % Normalize quaternion to avoid drift
    q = q / norm(q);

    % Desired frame = LVLH nadir frame => desired quaternion = identity
    % So attitude error quaternion = q_err = q
    q_err = q;

    % Small-angle error vector
    delta_theta = 2 * q_err(2:4);   % 3x1

    % LQR state
    x_e = [delta_theta; w];         % 6x1

    % Control torque
    u = -K * x_e;                   % 3x1

    % Rigid body dynamics: J * wdot = u - w × (Jw)
    Jw   = J * w;
    wdot = J \ (u - cross(w, Jw));

    % Quaternion kinematics: qdot = 0.5 * Omega(w) * q
    wx = w(1); wy = w(2); wz = w(3);
    W = [  0   -wx  -wy  -wz;
          wx    0    wz  -wy;
          wy   -wz   0    wx;
          wz    wy  -wx   0  ];
    qdot = 0.5 * W * q;

    dx = [qdot; wdot];
end
