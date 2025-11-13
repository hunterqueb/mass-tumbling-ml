function [I_opt, R_opt, J_opt, info] = estimateInertiaTorqueFree6D(t, omega_body, J_init, I_init)
% estimateInertiaTorqueFree6D
%   Jointly estimate principal moments (I1,I2,I3) and principal axes R
%   from torque-free angular velocity measurements.
%
%   J = R * diag(I) * R'
%
% Inputs:
%   t          : Nx1 time vector (s), strictly increasing
%   omega_body : Nx3 angular velocity measurements in body frame [rad/s]
%   J_init     : 3x3 SPD initial guess for inertia in body frame (e.g. your J_est)
%                (used only to initialize R and I; can be [])
%   I_init     : optional 3x1 initial guess for principal moments [I1; I2; I3]
%
% Outputs:
%   I_opt : 3x1 estimated principal moments
%   R_opt : 3x3 estimated rotation (principal -> body)
%   J_opt : 3x3 estimated inertia in body frame
%   info  : struct with optimization details

    % ---- basic input handling ----
    if size(t,2) ~= 1
        t = t(:);
    end
    if size(omega_body,2) ~= 3
        error('omega_body must be N x 3');
    end
    if length(t) ~= size(omega_body,1)
        error('t and omega_body must have same number of samples');
    end

    % ---- initial R and I from J_init if provided ----
    if nargin < 3 || isempty(J_init)
        R0 = eye(3);
        if nargin < 4 || isempty(I_init)
            I0 = [1;1;1];
        else
            I0 = I_init(:);
        end
    else
        Jsym = (J_init + J_init.')/2;
        [V0,D0] = eig(Jsym);
        if det(V0) < 0
            V0(:,1) = -V0(:,1);
        end
        % principal -> body is R0 = V0
        R0 = V0;
        lam0 = diag(D0);
        if nargin < 4 || isempty(I_init)
            % keep relative shape, possibly rescale magnitudes if desired
            I0 = lam0;
        else
            I0 = I_init(:);
        end
    end

    if numel(I0) ~= 3
        error('I_init (if provided) must be 3x1');
    end

    % convert initial R0 to so(3) vector r0
    r0 = rotm_log(R0);

    % parameter vector x = [r; I1; I2; I3]
    x0 = [r0; I0(:)];

    % ---- bounds: I > 0, r unconstrained ----
    lb = [-inf; -inf; -inf;  1e-6; 1e-6; 1e-6];
    ub = [ inf;  inf;  inf;  1e6;  1e6;  1e6];

    % ---- ODE options ----
    odeOpts = odeset('RelTol',1e-8,'AbsTol',1e-10);

    % ---- cost wrapper ----
    % cost = sum ||omega_body(tk) - omega_body_hat(tk; x)||^2
    costFun = @(x) inertiaCost6D(x, t, omega_body, odeOpts);

    % ---- optimization ----
    opts = optimoptions('fmincon', ...
                        'Display', 'iter', ...
                        'Algorithm', 'interior-point', ...
                        'MaxFunctionEvaluations', 5e4, ...
                        'MaxIterations', 4000);

    problem.objective = costFun;
    problem.x0        = x0;
    problem.lb        = lb;
    problem.ub        = ub;
    problem.solver    = 'fmincon';
    problem.options   = opts;

    [x_opt, fval, exitflag, output] = fmincon(problem);

    r_opt = x_opt(1:3);
    I_opt = x_opt(4:6);

    R_opt = so3_exp(r_opt);
    J_opt = R_opt * diag(I_opt) * R_opt.';

    info.fval     = fval;
    info.exitflag = exitflag;
    info.output   = output;
end
