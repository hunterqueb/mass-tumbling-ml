function [I_opt, J_opt, info] = estimateInertiaTorqueFree(t, omega_body, Q_est, I0)
% estimateInertiaTorqueFree
%   Identify principal moments of inertia [I1, I2, I3] from torque-free motion.
%
% Inputs
%   t          : Nx1 time vector (seconds, strictly increasing)
%   omega_body : Nx3 angular velocity measurements in the body frame [rad/s]
%   Q_est      : 3x3 SPD "shape" matrix with good eigenvectors (principal axes)
%   I0         : (optional) 3x1 initial guess for [I1; I2; I3]
%
% Outputs
%   I_opt : 3x1 optimal principal moments [I1; I2; I3]
%   J_opt : 3x3 reconstructed inertia matrix in the body frame
%   info  : struct with optimizer info (exitflag, fval, etc.)

    if size(t,2) ~= 1
        t = t(:);
    end
    if size(omega_body,2) ~= 3
        error('omega_body must be N x 3.');
    end
    if length(t) ~= size(omega_body,1)
        error('t and omega_body must have the same number of samples.');
    end
    if ~isequal(size(Q_est), [3 3])
        error('Q_est must be 3x3.');
    end

    % principal axes from Q_est (any SPD works; we only use eigenvectors)
    [V, D] = eig((Q_est + Q_est.')/2);
    if det(V) < 0
        V(:,1) = -V(:,1);
    end

    % Transform measured omega into principal frame
    omega_p_meas = (V' * omega_body.').';  % N x 3

    % Initial condition in principal frame
    omega_p0 = omega_p_meas(1,:).';

    % initial guess for [I1 I2 I3]
    if nargin < 4 || isempty(I0)
        lam = diag(D);
        scale_guess = 100;              % arbitrary scale; tweak if you want
        I0 = scale_guess * lam;
    else
        I0 = I0(:);
        if numel(I0) ~= 3
            error('I0 must be a 3x1 vector.');
        end
    end

    lb = 1e-6 * ones(3,1);
    ub = 1e6  * ones(3,1);

    odeOpts = odeset('RelTol', 1e-8, 'AbsTol', 1e-10);

    costFun = @(I) inertiaCost(I, t, omega_p0, omega_p_meas, odeOpts);

    opts = optimoptions('fmincon', ...
                        'Display', 'iter', ...
                        'Algorithm', 'interior-point', ...
                        'MaxFunctionEvaluations', 1e4, ...
                        'MaxIterations', 200);

    problem.objective = costFun;
    problem.x0        = I0;
    problem.lb        = lb;
    problem.ub        = ub;
    problem.solver    = 'fmincon';
    problem.options   = opts;

    [I_opt, fval, exitflag, output] = fmincon(problem);

    J_opt = V * diag(I_opt) * V.';

    info.fval     = fval;
    info.exitflag = exitflag;
    info.output   = output;
    info.V_axes   = V;
end

% ---------- helpers ----------

function J = inertiaCost(I, t, omega_p0, omega_p_meas, odeOpts)
    I1 = I(1); I2 = I(2); I3 = I(3);

    if I1 <= 0 || I2 <= 0 || I3 <= 0
        J = 1e20;
        return;
    end

    odefun = @(tt, omega) eulerTorqueFreePrincipal(tt, omega, I1, I2, I3);
    [~, omega_hat] = ode45(odefun, t, omega_p0, odeOpts);

    if size(omega_hat,1) ~= size(omega_p_meas,1)
        J = 1e20;
        return;
    end

    res = omega_hat - omega_p_meas;
    J = sum(res(:).^2);
end

