function J = inertiaCost6D(x, t, omega_body, odeOpts)
% x = [r; I1; I2; I3]

    r  = x(1:3);
    I1 = x(4); I2 = x(5); I3 = x(6);

    if I1 <= 0 || I2 <= 0 || I3 <= 0
        J = 1e20;
        return;
    end

    % rotation principal -> body
    R = so3_exp(r);

    % initial condition in principal frame
    omega0_body = omega_body(1,:).';     % 3x1
    omega0_p    = R.' * omega0_body;     % principal frame

    % integrate torque-free dynamics in principal frame
    odefun = @(tt,omega) eulerTorqueFreePrincipal(tt,omega,I1,I2,I3);
    [~, omega_p_hat] = ode45(odefun, t, omega0_p, odeOpts);   % N x 3

    % back to body frame
    omega_body_hat = (R * omega_p_hat.').';                   % N x 3

    % residuals
    if size(omega_body_hat,1) ~= size(omega_body,1)
        J = 1e20;
        return;
    end

    res = omega_body_hat - omega_body;
    J   = sum(res(:).^2);
end
