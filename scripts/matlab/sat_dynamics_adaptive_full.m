function dx = sat_dynamics_adaptive_full(~, x, J_true, K_R, K_Omega, Gamma_J)
% x = [q; w; Jhat(:)]
% q       : 4x1 quaternion (body->LVLH)
% w       : 3x1 body angular velocity [rad/s]
% Jhat(:) : 9x1 vec of 3x3 inertia estimate

    q      = x(1:4);
    w      = x(5:7);
    Jhat_v = x(8:16);

    % Rebuild Jhat and keep symmetric
    Jhat = reshape(Jhat_v, 3, 3);
    Jhat = (Jhat + Jhat')/2;

    % Normalize quaternion
    q = q / max(norm(q), 1e-12);

    % Nadir regulation: desired quat = identity, desired omega = 0
    q_err = q;
    e_R   = 2 * q_err(2:4);
    e_Omega = w;

    % --- Control law: PD + model-based coriolis term with Jhat ---
    u_PD = -K_R * e_R - K_Omega * e_Omega;
    Jhatw = Jhat * w;
    u = u_PD + cross(w, Jhatw);   % <-- Jhat enters here and DOES NOT cancel

    % --- True plant dynamics using J_true ---
    Jw_true = J_true * w;
    wdot    = J_true \ (u - cross(w, Jw_true));

    % Quaternion kinematics
    wx = w(1); wy = w(2); wz = w(3);
    W = [  0   -wx  -wy  -wz;
          wx    0    wz  -wy;
          wy   -wz   0    wx;
          wz    wy  -wx   0 ];
    qdot = 0.5 * W * q;

    % --- Predicted acceleration if Jhat were the true inertia ---
    wdot_hat = Jhat \ (u - cross(w, Jhatw));

    % Accel prediction error
    e_wdot = wdot_hat - wdot;

    % Full-matrix adaptation: Jdot_hat = -Gamma_J * sym(e_wdot * w')
    Jdot_hat = -Gamma_J * 0.5 * (e_wdot * w' + w * e_wdot');
    Jdot_hat = (Jdot_hat + Jdot_hat')/2;

    dx = zeros(size(x));
    dx(1:4)  = qdot;
    dx(5:7)  = wdot;
    dx(8:16) = reshape(Jdot_hat, 9, 1);
end
