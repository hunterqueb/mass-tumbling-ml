function dx = sat_dynamics_adaptive_eig(~, x, J_true, V0, lambda0, K_R, K_Omega, Gamma_alpha)
% x = [q; w; alpha]
% q      : 4x1 quaternion (body->LVLH)
% w      : 3x1 body angular velocity [rad/s]
% alpha  : 3x1 eigenvalue scale factors
%
% Jhat = V0 * diag(alpha_i * lambda0_i) * V0'

    q     = x(1:4);
    w     = x(5:7);
    alpha = x(8:10);

    % Normalize quaternion
    q = q / max(norm(q), 1e-12);

    % Build estimated inertia
    lambda_hat = alpha .* lambda0;         % 3x1
    Jhat       = V0 * diag(lambda_hat) * V0';

    % --- Error signals for pure regulation ---
    q_err   = q;                 % desired quaternion = identity
    e_R     = 2 * q_err(2:4);    % small-angle attitude error
    e_Omega = w;                 % desired Omega_d = 0

    % --- PD control (in body frame) ---
    u_PD = -K_R * e_R - K_Omega * e_Omega;

    % Model-based term with Jhat
    Jhatw    = Jhat * w;
    cori_hat = cross(w, Jhatw);
    wdot_hat = Jhat \ (u_PD - cori_hat);

    % Actual torque applied
    u = Jhat * wdot_hat + cori_hat;

    % --- True plant dynamics with J_true ---
    Jw_true = J_true * w;
    wdot    = J_true \ (u - cross(w, Jw_true));

    % Quaternion kinematics
    wx = w(1); wy = w(2); wz = w(3);
    W = [  0   -wx  -wy  -wz;
          wx    0    wz  -wy;
          wy   -wz   0    wx;
          wz    wy  -wx   0 ];
    qdot = 0.5 * W * q;

    % --- Adaptive law for alpha (per principal axis) ---
    % Work in principal-axis coordinates to make it less stupid
    w_tilde       = V0' * w;
    wdot_tilde    = V0' * wdot;
    wdot_hat_tilde = V0' * wdot_hat;

    % Error in predicted angular acceleration per axis
    e_wdot_tilde = wdot_hat_tilde - wdot_tilde;   % 3x1

    % Simple gradient-style update:
    %   alpha_dot_i ∝ e_wdot_i * w_tilde_i
    % This pushes eigenvalue scales to reduce accel mismatch along each principal axis.
    alpha_dot = Gamma_alpha * (e_wdot_tilde .* w_tilde);

    dx = [qdot;
          wdot;
          alpha_dot];
end
