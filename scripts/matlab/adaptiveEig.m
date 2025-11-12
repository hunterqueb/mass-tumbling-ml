% True inertia (plant)
J_true = 100 * [[ 0.3905  0.0567  0.1052]
 [ 0.0567  0.3803 -0.115 ]
 [ 0.1052 -0.115   0.2292]] + diag([20,25,15]);

% Your full estimate
J0 = [[ 0.3466  0.0132  0.0233]
 [ 0.0132  0.3365 -0.0185]
 [ 0.0233 -0.0185  0.3169]] + diag([20,25,15]);

% Eigendecomposition of J0 (principal axes estimate)
[V0, Lambda0] = eig(J0);      % Lambda0 ~ diag(lambda01, lambda02, lambda03)

lambda0 = diag(Lambda0);      % 3x1

% Initial eigenvalue scale factors (start at 1 => Jhat_0 = J0)
alpha0 = [1;1;1];

% Gains
K_R     = 5 * eye(3);
K_Omega = 2 * eye(3);

Gamma_alpha = diag([1e-3, 1e-3, 1e-3]);  % adaptation gains for each axis

% Initial attitude / rate
ang0  = deg2rad(10);
axis0 = [0; 1; 0]; axis0 = axis0 / norm(axis0);
q0    = [cos(ang0/2); axis0*sin(ang0/2)];
w0    = deg2rad([0.5; -0.2; 0.3]);

% Full state: [q; w; alpha]
x0 = [q0;
      w0;
      alpha0];

tspan = [0 500];

odefun = @(t,x) sat_dynamics_adaptive_eig(t,x,J_true,V0,lambda0,K_R,K_Omega,Gamma_alpha);

opts = odeset('RelTol',1e-9,'AbsTol',1e-9);
[t,x] = ode45(odefun, tspan, x0, opts);

q     = x(:,1:4);
w     = x(:,5:7);
alpha = x(:,8:10);   % 3 scales

% Normalize quaternions
for k = 1:size(q,1)
    q(k,:) = q(k,:)/norm(q(k,:));
end

delta_theta = 2*q(:,2:4);

figure;
subplot(3,1,1);
plot(t, rad2deg(delta_theta));
xlabel('t [s]'); ylabel('\delta\theta [deg]');
legend('\delta\theta_x','\delta\theta_y','\delta\theta_z'); grid on;

subplot(3,1,2);
plot(t, rad2deg(w));
xlabel('t [s]'); ylabel('\omega [deg/s]');
legend('\omega_x','\omega_y','\omega_z'); grid on;

subplot(3,1,3);
plot(t, alpha);
xlabel('t [s]'); ylabel('\alpha_i(t)');
legend('\alpha_1','\alpha_2','\alpha_3'); grid on;
