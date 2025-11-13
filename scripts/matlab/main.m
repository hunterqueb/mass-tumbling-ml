% ---------- given ----------
J_true = 100 *  [[ 0.3797 -0.0024 -0.0709]
 [-0.0024  0.4057 -0.0065]
 [-0.0709 -0.0065  0.2145]];

J_est  =   [[ 0.3817 -0.0018 -0.0924]
 [-0.0018  0.4347 -0.0086]
 [-0.0924 -0.0086  0.1836]];

% ---------- true principal moments / axes ----------
[Jt_sym, ~] = deal((J_true + J_true.')/2);
[V_true, D_true] = eig(Jt_sym);
if det(V_true) < 0
    V_true(:,1) = -V_true(:,1);
end
I_true = diag(D_true);   % [I1_true; I2_true; I3_true]

disp('True principal moments:');
disp(I_true.');

% ---------- generate synthetic torque-free data ----------
T_end = 200;                  % simulation time [s]
N     = 2001;                 % number of samples
t     = linspace(0, T_end, N).';

% pick some non-trivial initial angular velocity in body frame
omega0_body = [0.05; 0.12; 0.09];

% transform to true principal frame
omega0_p = V_true.' * omega0_body;

% integrate in true principal frame
odefun_true = @(tt, omega) eulerTorqueFreePrincipal(tt, omega, ...
                                                    I_true(1), I_true(2), I_true(3));
odeOpts = odeset('RelTol',1e-8,'AbsTol',1e-10);
[~, omega_p] = ode45(odefun_true, t, omega0_p);

% transform back to body frame (this is your "measured" data)
omega_body_meas = (V_true * omega_p.').';   % N x 3

% optionally add noise if you want realism
% omega_body_meas = omega_body_meas + 0.001*randn(size(omega_body_meas));

% ---------- run the estimator with Q_est = J_est ----------
% You can optionally give an initial guess; here use principal moments of J_est
[Jest_sym, ~] = deal((J_est + J_est.')/2);
[Vest, Dest] = eig(Jest_sym);
if det(Vest) < 0
    Vest(:,1) = -Vest(:,1);
end
I0_est = diag(Dest);   % initial guess for [I1; I2; I3]

[I_opt, J_opt, info] = estimateInertiaTorqueFree(t, omega_body_meas, J_est, I0_est);

disp('Estimated principal moments:');
disp(I_opt.');

disp('Estimated inertia matrix in body frame:');
disp(J_opt);

disp('Difference J_opt - J_true:');
disp(J_opt - J_true);
