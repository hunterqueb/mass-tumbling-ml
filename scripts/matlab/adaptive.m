%% inertia_id_principal_frame.m
% Online identification of principal moments of inertia using RLS
% Model: tau_p = Y(omega_p, domega_p) * theta
% theta = [I1; I2; I3] in the (fixed) principal frame Q_est.

clear; clc;

%% USER INPUTS / HOOKS
% -------------------------------------------------------------------------
% Replace these with your own quantities from simulation/flight code.

% 1) Estimated principal frame from your offline estimator (3x3, orthonormal)
Q_est =  [[ 0.3868  0.0468  0.0886]
 [ 0.0468  0.3694 -0.1004]
 [ 0.0886 -0.1004  0.2439]];   % <-- replace with your estimated eigenvector matrix

% 2) Time step and number of samples
dt   = 0.01;      % sample period [s]
Tend = 20.0;      % total time [s]
N    = round(Tend/dt) + 1;

t = (0:N-1)' * dt;

% 3) Synthetic test true inertia (for demo); replace with your real plant
I_true = 100* [[ 0.3905  0.0567  0.1052]
 [ 0.0567  0.3803 -0.115 ]
 [ 0.1052 -0.115   0.2292]];   % principal moments in Q_est frame

% 4) Generate a simple excitation torque pattern (in body frame)
tau_b = zeros(N,3);
tau_b(:,1) = 0.1 * sin(0.5 * t);
tau_b(:,2) = 0.08 * sin(0.8 * t + 0.3);
tau_b(:,3) = 0.06 * sin(1.2 * t + 0.7);

% 5) Simulate angular dynamics to get omega_b (in body frame) for this demo.
%    In your code, you already have omega_b from sensors or integrator.
omega_b = zeros(N,3);
omega   = [0.1; 0.05; -0.02];  % initial angular rate in body frame

for k = 1:N-1
    tau_k = tau_b(k,:)';

    % Here we assume Q_est is the principal frame of the *true* inertia for demo.
    % In your application, Q_est is fixed and I_true is unknown.
    % Dynamics in body frame: I*dot(omega) + omega x (I*omega) = tau
    Iw    = I_true * omega;
    domega = I_true \ (tau_k - cross(omega, Iw));

    omega = omega + domega*dt;
    omega_b(k+1,:) = omega.';
end

% 6) Approximate dot(omega_b) using finite difference + simple filter
domega_b = zeros(N,3);
for k = 2:N-1
    domega_b(k,:) = (omega_b(k+1,:) - omega_b(k-1,:)) / (2*dt);
end
domega_b(1,:)   = (omega_b(2,:) - omega_b(1,:)) / dt;
domega_b(end,:) = (omega_b(end,:) - omega_b(end-1,:)) / dt;

% (You should use a better differentiator / filter in real use.)

%% TRANSFORM TO PRINCIPAL FRAME
% -------------------------------------------------------------------------
% principal frame quantities: omega_p, domega_p, tau_p
omega_p  = (Q_est.' * omega_b.').';     % N x 3
domega_p = (Q_est.' * domega_b.').';    % N x 3
tau_p    = (Q_est.' * tau_b.').';       % N x 3

%% RLS INITIALIZATION
% -------------------------------------------------------------------------
% Parameter vector theta = [I1; I2; I3]
theta_hat = zeros(3,1);
% If you have eigenvalues from estimator, use them as initial guess:
% theta_hat = scale_guess * lambda_est;  (3x1)

P0   = 1e4 * eye(3);  % large covariance => low initial confidence
P    = P0;
lam  = 0.995;         % forgetting factor in (0,1]; 1 = no forgetting

% Constraints to enforce SPD and physical range
I_min = 1e-3;
I_max = 1e3;

theta_hist = zeros(N,3);
theta_hist(1,:) = theta_hat.';

%% MAIN IDENTIFICATION LOOP
% -------------------------------------------------------------------------
for k = 1:N
    w1  = omega_p(k,1);
    w2  = omega_p(k,2);
    w3  = omega_p(k,3);
    dw1 = domega_p(k,1);
    dw2 = domega_p(k,2);
    dw3 = domega_p(k,3);

    % Build regressor Y(omega_p, domega_p) [3 x 3]
    % tau_p = Y * [I1;I2;I3]
    Y = [ dw1,     -w2*w3,   w2*w3;
          w3*w1,    dw2,    -w3*w1;
         -w1*w2,    w1*w2,   dw3   ];

    tau_k = tau_p(k,:).';   % 3x1

    % Predicted torque
    tau_hat = Y * theta_hat;

    % Prediction error
    e = tau_k - tau_hat;    % 3x1

    % RLS gain
    S = lam*eye(3) + Y*P*Y.';      % 3x3
    K = (P*Y.') / S;               % 3x3

    % Parameter update
    theta_hat = theta_hat + K*e;

    % Covariance update
    P = (P - K*Y*P) / lam;

    % Enforce parameter constraints I_min <= I_i <= I_max
    theta_hat = max(theta_hat, I_min);
    theta_hat = min(theta_hat, I_max);

    theta_hist(k,:) = theta_hat.';
end

%% RECONSTRUCT ESTIMATED INERTIA IN BODY FRAME
% -------------------------------------------------------------------------
% I_hat(k) = Q_est * diag(theta_hat(k,:)) * Q_est'
% For plotting, just show convergence of I1,I2,I3 vs true.
I_hat = Q_est * diag(theta_hat) * Q_est.';

true_principal = diag(I_true);  % [I1_true; I2_true; I3_true]

figure; hold on; grid on;
plot(t, theta_hist(:,1), 'LineWidth', 1.5);
plot(t, theta_hist(:,2), 'LineWidth', 1.5);
plot(t, theta_hist(:,3), 'LineWidth', 1.5);
yline(true_principal(1), '--', 'I1 true');
yline(true_principal(2), '--', 'I2 true');
yline(true_principal(3), '--', 'I3 true');
xlabel('Time [s]');
ylabel('Estimated principal moments');
legend({'I1 hat','I2 hat','I3 hat','I1 true','I2 true','I3 true'});
title('Principal inertia identification (RLS in estimated principal frame)');
