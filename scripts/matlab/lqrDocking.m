%% Discrete docking with LQR recomputation
clear; clc;

%% Pre-dock inertia (chaser alone)
J_chaser = diag([20, 25, 15]);  % example

%% Post-dock inertia (chaser + target, e.g. your docked J_true)
J_docked = 100 * [ 0.3466  0.0132  0.0233
       0.0132  0.3365 -0.0185
       0.0233 -0.0185  0.3169 ] ...
           + diag([20,25,15]);

%% LQR weights (same for both, but you can change them if you want)
q_att_weight    = 1e3;
q_omega_weight  = 10;
r_torque_weight = 1e-2;

%% Design LQR gains for each configuration
K_chaser = designLQRAttitude(J_chaser, q_att_weight, q_omega_weight, r_torque_weight);
K_docked = designLQRAttitude(J_docked, q_att_weight, q_omega_weight, r_torque_weight);

%% Initial attitude / rate
ang0  = deg2rad(15);
axis0 = [0; 1; 0]; axis0 = axis0 / norm(axis0);
q0    = [cos(ang0/2); axis0*sin(ang0/2)];
w0    = deg2rad([0.5; -0.2; 0.3]);

x0 = [q0; w0];

%% Time intervals
t0     = 0;
t_dock = 200;   % docking time [s]
tf     = 600;   % final time [s]

opts = odeset('RelTol',1e-9,'AbsTol',1e-9);

%% Phase 1: pre-dock (chaser only)
[t1, x1] = ode45(@(t,x) satDynamicsLQR(t, x, J_chaser, K_chaser), [t0 t_dock], x0, opts);

x_dock = x1(end,:).';   % state at docking instant

%% Phase 2: post-dock (chaser + target, new inertia and new K)
[t2, x2] = ode45(@(t,x) satDynamicsLQR(t, x, J_docked, K_docked), [t_dock tf], x_dock, opts);

%% Concatenate results
t  = [t1; t2];
x  = [x1; x2];

q = x(:,1:4);
w = x(:,5:7);

% Normalize quaternions
for k = 1:size(q,1)
    q(k,:) = q(k,:)/norm(q(k,:));
end

delta_theta = 2*q(:,2:4);

%% Plots
figure;
subplot(2,1,1);
plot(t, rad2deg(delta_theta));
xlabel('t [s]'); ylabel('\delta\theta [deg]');
legend('\delta\theta_x','\delta\theta_y','\delta\theta_z');
grid on; title('Attitude error (pre/post docking LQR)');

subplot(2,1,2);
plot(t, rad2deg(w));
xlabel('t [s]'); ylabel('\omega [deg/s]');
legend('\omega_x','\omega_y','\omega_z');
grid on; title('Body rates');
