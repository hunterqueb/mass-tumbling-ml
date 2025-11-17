%% Full-matrix adaptive inertia estimation for docked spacecraft (fixed control law)
clear; clc;

%% True inertia (plant) after docking
J_true = 100 * diag([0.146925 0.417965 0.435109]) ...
         %+ diag([20,25,15]);

%% Initial full inertia estimate J0
J0 =  diag([0.188065 0.405968 0.405968])
     %+ diag([20,25,15]);
J0 = (J0 + J0')/2;   % enforce symmetry

%% Control gains (tune)
K_R     = .3 * eye(3);   % attitude gain
K_Omega = .1 * eye(3);   % rate gain

%% Adaptation gain
Gamma_J = 5 * eye(3);  % diag gains for J update

%% Initial attitude / rate
ang0  = deg2rad(20);        % bigger error to actually see dynamics
axis0 = [0; 1; 0]; axis0 = axis0 / norm(axis0);
q0    = [cos(ang0/2); axis0*sin(ang0/2)];
w0    = deg2rad([1.0; -0.5; 0.8]);

%% Initial estimate Jhat(0) = J0
Jhat0     = J0;
Jhat0_vec = reshape(Jhat0, 9, 1);

%% Full state: [q; w; Jhat(:)]
x0 = [q0;
      w0;
      Jhat0_vec];

tspan = [0 300];

odefun = @(t,x) sat_dynamics_adaptive_full(t, x, J_true, K_R, K_Omega, Gamma_J);
opts   = odeset('RelTol',1e-9,'AbsTol',1e-9);

[t, x] = ode45(odefun, tspan, x0, opts);

%% Unpack
q     = x(:,1:4);
w     = x(:,5:7);
Jhatv = x(:,8:end);

N = size(t,1);
Jhat_diag = zeros(N,3);

for k = 1:N
    q(k,:) = q(k,:) ./ norm(q(k,:));
    Jhat_k = reshape(Jhatv(k,:).', 3, 3);
    Jhat_k = (Jhat_k + Jhat_k')/2;    % symmetrize
    Jhat_diag(k,:) = diag(Jhat_k).';
end

delta_theta = 2 * q(:,2:4);

%% Plots
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
plot(t, Jhat_diag);
xlabel('t [s]'); ylabel('diag(\hat{J}) [kg m^2]');
legend('\hat{J}_{11}','\hat{J}_{22}','\hat{J}_{33}');
grid on; title('Estimated inertia diagonal');
