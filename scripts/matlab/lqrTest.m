%% Nonlinear nadir-pointing LQR simulation
clear; clc;

%% Inertia
J   =  100 * [[ 0.3905  0.0567  0.1052]
 [ 0.0567  0.3803 -0.115 ]
 [ 0.1052 -0.115   0.2292]] + diag([20,25,15])

J_rand =  [[ 0.3466  0.0132  0.0233]
 [ 0.0132  0.3365 -0.0185]
 [ 0.0233 -0.0185  0.3169]] + diag([20,25,15])

%% Linearized A, B (same as before)
A = [zeros(3), eye(3);
     zeros(3), zeros(3)];
B = [zeros(3);
     inv(J)];

%% LQR weights (tune these)
q_att   = 1e3;
q_omega = 10;
Q = blkdiag(q_att*eye(3), q_omega*eye(3));
r_torque = 1e-2;
R = r_torque * eye(3);

%% LQR gain
[K, ~, e] = lqr(A,B,Q,R);
disp('Closed-loop eigenvalues (linearized):'); disp(e.');

%% Initial condition: small attitude error and some rate
% Small rotation of 10 deg about y-axis, for example
ang0 = deg2rad(10);
axis0 = [0;1;0]; axis0 = axis0 / norm(axis0);
q0 = [cos(ang0/2);
      axis0*sin(ang0/2)];          % body->LVLH quaternion

w0 = deg2rad([0.5; -0.2; 0.3]);    % initial body rates [rad/s]

x0 = [q0; w0];                     % 7x1 state

%% Time span
% tspan = [0 20];   % seconds, adjust as needed
tspan = linspace(0,10,400);
%% Integrate closed-loop nonlinear dynamics
odefun = @(t,x) sat_dynamics_lqr(t,x,J,K);

opts = odeset('RelTol',1e-9,'AbsTol',1e-9);
[t,x_tr] = ode45(odefun, tspan, x0, opts);

q = x_tr(:,1:4);        % Nx4
w = x_tr(:,5:7);        % Nx3

%% Enforce normalization (numerical drift)
for k = 1:size(q,1)
    q(k,:) = q(k,:) ./ norm(q(k,:));
end

%% Recover small-angle attitude error (for plotting)
delta_theta = 2 * q(:,2:4);    % small-angle approx

%% Plots
figure;
subplot(2,1,1);
plot(t, rad2deg(delta_theta));
xlabel('t [s]'); ylabel('\delta\theta [deg]');
legend('\delta\theta_x','\delta\theta_y','\delta\theta_z');
grid on; title('Attitude error (small-angle approx)');

subplot(2,1,2);
plot(t, rad2deg(w));
xlabel('t [s]'); ylabel('\omega [deg/s]');
legend('\omega_x','\omega_y','\omega_z');
grid on; title('Body rates');












%% Linearized A, B (same as before)

J = J_rand


A = [zeros(3), eye(3);
     zeros(3), zeros(3)];
B = [zeros(3);
     inv(J)];

%% LQR weights (tune these)
q_att   = 1e3;
q_omega = 10;
Q = 100 * blkdiag(q_att*eye(3), q_omega*eye(3));
r_torque = 1e-2;
R = 100 * r_torque * eye(3);

%% LQR gain
[K, ~, e] = lqr(A,B,Q,R);
disp('Closed-loop eigenvalues (linearized):'); disp(e.');

%% Initial condition: small attitude error and some rate
% Small rotation of 10 deg about y-axis, for example
ang0 = deg2rad(10);
axis0 = [0;1;0]; axis0 = axis0 / norm(axis0);
q0 = [cos(ang0/2);
      axis0*sin(ang0/2)];          % body->LVLH quaternion

w0 = deg2rad([0.5; -0.2; 0.3]);    % initial body rates [rad/s]

x0 = [q0; w0];                     % 7x1 state

%% Integrate closed-loop nonlinear dynamics
odefun = @(t,x) sat_dynamics_lqr(t,x,J,K);

opts = odeset('RelTol',1e-9,'AbsTol',1e-9);
[t,x_r] = ode45(odefun, tspan, x0, opts);

q = x_r(:,1:4);        % Nx4
w = x_r(:,5:7);        % Nx3

%% Enforce normalization (numerical drift)
for k = 1:size(q,1)
    q(k,:) = q(k,:) ./ norm(q(k,:));
end

%% Recover small-angle attitude error (for plotting)
delta_theta = 2 * q(:,2:4);    % small-angle approx

%% Plots
figure;
subplot(2,1,1);
plot(t, rad2deg(delta_theta));
xlabel('t [s]'); ylabel('\delta\theta [deg]');
legend('\delta\theta_x','\delta\theta_y','\delta\theta_z');
grid on; title('Attitude error (small-angle approx)');

subplot(2,1,2);
plot(t, rad2deg(w));
xlabel('t [s]'); ylabel('\omega [deg/s]');
legend('\omega_x','\omega_y','\omega_z');
grid on; title('Body rates');


err = x_tr - x_r;
q = err(:,1:4);
w = err(:,5:7);
delta_theta = 2 * q(:,2:4);

figure;
subplot(2,1,1);
plot(t, rad2deg(delta_theta));
xlabel('t [s]'); ylabel('\delta\theta [deg]');
legend('\delta\theta_x','\delta\theta_y','\delta\theta_z');
grid on; title('Attitude error bwtn different matricies');

subplot(2,1,2);
plot(t, rad2deg(w));
xlabel('t [s]'); ylabel('\omega [deg/s]');
legend('\omega_x','\omega_y','\omega_z');
grid on; title('Body rates bwtn different matricies');


