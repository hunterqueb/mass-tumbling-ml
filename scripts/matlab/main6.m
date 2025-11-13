
clear;clc;

% True inertia (plant)
J_true = 100 * [[ 0.3797 -0.0024 -0.0709]
 [-0.0024  0.4057 -0.0065]
 [-0.0709 -0.0065  0.2145]];

% Your estimate (bad scale + misaligned eigenvalues)
J_est  =  [[ 0.3817 -0.0018 -0.0924]
 [-0.0018  0.4347 -0.0086]
 [-0.0924 -0.0086  0.1836]];

% J_est  =  rand(3);

T_end = 200;
N     = T_end * 10 + 1;
t     = linspace(0,T_end,N).';


% --- generate synthetic torque-free motion from J_true ---
Jsym = (J_true + J_true.')/2;
[V_true, D_true] = eig(Jsym);
if det(V_true) < 0, V_true(:,1) = -V_true(:,1); end
I_true = diag(D_true);

% initial angular velocity in body frame
omega0_body = [0.05; 0.12; 0.09];

% principal frame of J_true
omega0_p = V_true.' * omega0_body;


odeOpts = odeset('RelTol',1e-9,'AbsTol',1e-12);
odefun_true = @(tt,omega) eulerTorqueFreePrincipal(tt,omega, ...
                                                   I_true(1),I_true(2),I_true(3));
[~, omega_p_true] = ode45(odefun_true, t, omega0_p, odeOpts);

omega_body_meas = (V_true * omega_p_true.').';   % N x 3
% add noise if you want:
% omega_body_meas = omega_body_meas + 0.002*randn(size(omega_body_meas));

% --- run 6D estimator with J_est as initial guess ---
[I_opt, R_opt, J_opt, info] = estimateInertiaTorqueFree6D(t, omega_body_meas, J_est);

disp('True principal moments:');
disp(I_true.');

disp('Estimated principal moments:');
disp(I_opt.');

disp('True inertia matrix:');
disp(J_true);

disp('Estimated inertia matrix:');
disp(J_opt);

disp('Error J_opt - J_true:');
disp(J_opt - J_true);
