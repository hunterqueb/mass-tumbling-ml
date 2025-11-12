function dx = sat_dynamics_adaptive(~, x, J_true, K_R, K_Omega, Gamma_J)
% x = [q; w; Jhat_diag]
% q         : 4x1 quaternion (body->LVLH nadir frame)
% w         : 3x1 body angular velocity [rad/s]
% Jhat_diag : 3x1 estimates of diagonal inertia

    q = x(1:4);
    w = x(5:7);
    Jhat_diag = x(8:10);

    % Normalize q
    q = q / norm(q);

    % Rebuild estimated inertia as diagonal matrix
    Jhat = diag(Jhat_diag);

    % For pure nadir regulation, desired frame is fixed and aligned with LVLH,
    % desired angular velocity is zero.
    % So attitude error quaternion = q, rate error = w.
    q_err = q;
    e_R = 2 * q_err(2:4);   % small-angle approx
    e_Omega = w;            % since Omega_d = 0

    % Nominal PD torque (using K_R, K_Omega)
    u_PD = -K_R * e_R - K_Omega * e_Omega;

    % Use estimated inertia in the dynamics (computed-torque structure)
    % True dynamics: J_true * wdot = u - w x (J_true * w)
    % Controller only knows Jhat, so it computes wdot_hat ≈ Jhat^{-1}*(u_PD - w x (Jhat*w)).
    Jhatw = Jhat * w;
    cori_hat = cross(w, Jhatw);
    wdot_hat = Jhat \ (u_PD - cori_hat);

    % Use u = Jhat*wdot_hat + cross(w, Jhat*w) (i.e., “cancel” dynamics based on Jhat)
    u = Jhat * wdot_hat + cross(w, Jhatw);

    % Now propagate **true** plant with true J
    Jw_true = J_true * w;
    wdot = J_true \ (u - cross(w, Jw_true));

    % Quaternion kinematics: qdot = 0.5 * W(w) * q
    wx = w(1); wy = w(2); wz = w(3);
    W = [  0   -wx  -wy  -wz;
          wx    0    wz  -wy;
          wy   -wz   0    wx;
          wz    wy  -wx   0 ];
    qdot = 0.5 * W * q;

    % --- Adaptive law for Jhat_diag (simple Lyapunov-inspired rule) ---
    % You can do more sophisticated MRAC; this is a simple heuristic:
    %   d(Jhat_i)/dt = gamma_i * e_Omega_i * (u_i + e_Omega_i)
    % Idea: if error and acceleration disagree, adjust inertia estimate.
    gamma_vec = diag(Gamma_J);
    Jhat_dot = gamma_vec .* (e_Omega .* (u_PD + e_Omega));

    dx = [qdot;
          wdot;
          Jhat_dot];
end
