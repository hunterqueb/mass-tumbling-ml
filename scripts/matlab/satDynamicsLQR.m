function dx = satDynamicsLQR(~, x, J, K)
% x = [q; w]
% q : 4x1 quaternion body->LVLH
% w : 3x1 body angular velocity [rad/s]
%
% J : 3x3 inertia matrix for current configuration
% K : 3x6 LQR gain (u = -K * [delta_theta; w])

    q = x(1:4);
    w = x(5:7);

    % Normalize quaternion
    q = q / max(norm(q), 1e-12);

    % Small-angle attitude error: delta_theta ≈ 2*q_vec (desired = identity)
    delta_theta = 2*q(2:4);

    % LQR state
    x_e = [delta_theta; w];

    % Control torque
    u = -K * x_e;

    % True rigid-body dynamics
    Jw   = J * w;
    wdot = J \ (u - cross(w, Jw));

    % Quaternion kinematics
    wx = w(1); wy = w(2); wz = w(3);
    W = [  0   -wx  -wy  -wz;
          wx    0    wz  -wy;
          wy   -wz   0    wx;
          wz    wy  -wx   0 ];
    qdot = 0.5 * W * q;

    dx = [qdot; wdot];
end
