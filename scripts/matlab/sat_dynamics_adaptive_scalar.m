function dx = sat_dynamics_adaptive_scalar(~, x, J_true, J0, K_R, K_Omega, gamma_alpha)
% x = [q; w; alpha]
% q     : 4x1 quaternion (body->LVLH nadir frame)
% w     : 3x1 body angular velocity [rad/s]
% alpha : scalar scaling of J0 so that Jhat = alpha * J0

    q     = x(1:4);
    w     = x(5:7);
    alpha = x(8);

    % Normalize quaternion
    q = q / max(norm(q), 1e-12);

    % Estimated inertia used by controller
    Jhat = alpha * J0;

    % --- Error signals (pure regulation to fixed nadir frame) ---
    % Desired quaternion = identity; so attitude error quaternion = q
    q_err   = q;
    e_R     = 2 * q_err(2:4);  % small-angle vector
    e_Omega = w;               % desired Omega_d = 0

    % --- PD control based on current estimate ---
    u_PD = -K_R * e_R - K_Omega * e_Omega;

    % Use estimated inertia in model-based term (computed-torque style)
    Jhatw    = Jhat * w;
    cori_hat = cross(w, Jhatw);
    wdot_hat = Jhat \ (u_PD - cori_hat);   % predicted angular accel given Jhat

    % Actual control torque sent to plant (use model-based term)
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

    % --- Adaptive law for alpha ---
    % Idea: drive predicted angular accel (wdot_hat) toward true wdot.
    % Cost: J = 0.5 * || wdot_hat - wdot ||^2
    % Simple gradient step: alpha_dot ∝ (wdot_hat - wdot)^T * (∂wdot_hat/∂alpha)
    % Approximate ∂wdot_hat/∂alpha ≈ -(1/alpha) * wdot_hat (since Jhat = alpha*J0).
    e_wdot    = wdot_hat - wdot;
    if alpha <= 1e-3
        alpha = 1e-3;   % prevent division by ~0 and sign flips
    end
    alpha_dot = gamma_alpha * (e_wdot' * wdot_hat) / alpha;

    dx = [qdot;
          wdot;
          alpha_dot];
end
