%% Adaptive attitude control for nadir-pointing spacecraft
clear; clc;

% True inertia (unknown to controller)
J_true = 100 * [[ 0.3905  0.0567  0.1052]
 [ 0.0567  0.3803 -0.115 ]
 [ 0.1052 -0.115   0.2292]] + diag([20,25,15]);   % kg·m^2
Jhat0 =  [[ 0.3466  0.0132  0.0233]
 [ 0.0132  0.3365 -0.0185]
 [ 0.0233 -0.0185  0.3169]] + diag([20,25,15]);    % 3x3

% Nominal gains (you tune these)
K_R     = 50 * eye(3);          % attitude error gain
K_Omega = 20 * eye(3);          % rate error gain

% Adaptation gains (tune)
Gamma_J = 5*eye(3);   % for each J_i

% Initial condition: 10 deg about y
ang0 = deg2rad(10);
axis0 = [0; 1; 0];
axis0 = axis0 / norm(axis0);
q0 = [cos(ang0/2);
      axis0*sin(ang0/2)];      % body->nadir frame

w0 = deg2rad([0.5; -0.2; 0.3]);

% Initial parameter estimates (wrong on purpose)
Jhat0_vec = diag(Jhat0);       % store just diagonal as 3x1

% Full state: [q; w; Jhat_diag]
x0 = [q0;
      w0;
      Jhat0_vec];

tspan = [0 500];

odefun = @(t,x) sat_dynamics_adaptive(t,x,J_true,K_R,K_Omega,Gamma_J);

opts = odeset('RelTol',1e-9,'AbsTol',1e-9);
[t,x] = ode45(odefun, tspan, x0, opts);

q     = x(:,1:4);
w     = x(:,5:7);
Jhatd = x(:,8:10);   % estimated diagonal elements

% Normalize quaternions
for k = 1:size(q,1)
    q(k,:) = q(k,:)/norm(q(k,:));
end

% Small-angle attitude error approx
delta_theta = 2*q(:,2:4);

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
plot(t, Jhatd);
xlabel('t [s]'); ylabel('\hat{J}_{ii} [kg m^2]');
legend('\hat{J}_{11}','\hat{J}_{22}','\hat{J}_{33}');
grid on; title('Estimated inertia diagonal');


J_err = abs(J_true - diag(Jhatd(end,:)))