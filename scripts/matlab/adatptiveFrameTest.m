%% True inertia and base estimate
J_true = 100 * [[ 0.3905  0.0567  0.1052]
 [ 0.0567  0.3803 -0.115 ]
 [ 0.1052 -0.115   0.2292]] + diag([20,25,15]);  % real plant
J0     = [[ 0.3466  0.0132  0.0233]
 [ 0.0132  0.3365 -0.0185]
 [ 0.0233 -0.0185  0.3169]] + diag([20,25,15]);        % your principal-axes estimate (shape)

% True relation: J_true = alpha_true * J0  => alpha_true = 100
alpha_true = 100;

% Controller's initial scalar estimate (wrong)
alpha0 = 1.0;     % so initial Jhat = 1 * J0 = diag(1,1,1)

%% Gains
K_R     = 5 * eye(3);        % attitude gain
K_Omega = 2 * eye(3);        % rate gain

gamma_alpha = 1e-3;          % adaptation gain for alpha (tune)

%% Initial attitude / rate
ang0  = deg2rad(10);
axis0 = [0; 1; 0]; axis0 = axis0 / norm(axis0);
q0    = [cos(ang0/2); axis0*sin(ang0/2)];
w0    = deg2rad([0.5; -0.2; 0.3]);

% State: [q; w; alpha]
x0 = [q0;
      w0;
      alpha0];

tspan = [0 500];

odefun = @(t,x) sat_dynamics_adaptive_scalar(t,x,J_true,J0,K_R,K_Omega,gamma_alpha);

opts = odeset('RelTol',1e-9,'AbsTol',1e-9);
[t,x] = ode45(odefun, tspan, x0, opts);

q     = x(:,1:4);
w     = x(:,5:7);
alpha = x(:,8);

% Normalize quaternions
for k = 1:size(q,1)
    q(k,:) = q(k,:)/norm(q(k,:));
end

delta_theta = 2*q(:,2:4);   % small-angle error

figure;
subplot(3,1,1);
plot(t, rad2deg(delta_theta));
xlabel('t [s]'); ylabel('\delta\theta [deg]');
legend('\delta\theta_x','\delta\theta_y','\delta\theta_z');
grid on; title('Attitude error');

subplot(3,1,2);
plot(t, rad2deg(w));
xlabel('t [s]'); ylabel('\omega [deg/s]');
legend('\omega_x','\omega_y','\omega_z');
grid on; title('Body rates');

subplot(3,1,3);
plot(t, alpha);
xlabel('t [s]'); ylabel('\alpha(t)');
grid on; title('Inertia scale estimate (true = 100)');
