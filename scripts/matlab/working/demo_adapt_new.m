%% Adaptive inertia matrix identification for a free-tumbling rigid body
% Closed-loop attitude control + online inertia estimation.
% - True plant inertia: J_true
% - Initial estimate: J_hat(0) from a trace-1, frame-aligned estimator
% - Controller: PD attitude hold + excitation torques
% - Estimator: gradient adaptation on torque prediction error

clear; clc; close all;

rng(1); % reproducibility

%% True inertia (plant) - arbitrary SPD matrix
% You can replace this with your own J_true.
Q_rand = orth(randn(3,3));
eig_true = 100 * [0.188065 0.405968 0.405968];          % principal moments
J_true  = Q_rand * diag(eig_true) * Q_rand.';  % 3x3 SPD

%% "Estimator" inertia with trace 1 and ~5 deg frame misalignment (your prior)
% This block mimics your estimator output: normalized trace, slightly misaligned.
% In practice, replace J_est_normalized with your estimator.

% Normalize true inertia to trace 1
J_norm      = J_true / trace(J_true);

J_est_normalized = [[ 0.3817 -0.0018 -0.0924]
 [-0.0018  0.4347 -0.0086]
 [-0.0924 -0.0086  0.1836]];        % SPD, trace ≈ 1

% Deliberately pick wrong scale for initial estimate
scale_guess = 0.5 * trace(J_true);                  % wrong scale on purpose
J_hat0      = scale_guess * J_est_normalized;       % 3x3

% Pack initial estimate into parameter vector theta_hat
theta_hat0 = J2theta(J_hat0);  % [Jxx Jyy Jzz Jxy Jxz Jyz]^T

%% Controller and estimator gains
Kp = 5 * eye(3);       % attitude proportional gain
Kd = 3 * eye(3);       % angular rate damping

Gamma = 0.05 * eye(6); % adaptation gain for inertia parameters

% Dither (excitation) torque: small multi-frequency signal
u_dither_amp = 0.02;   % Nm
w_d1 = 0.3; w_d2 = 0.5; w_d3 = 0.7; % rad/s

%% Initial state
q0 = [1;0;0;0];        % initial attitude (quaternion) ~ identity
w0 = [0.0; 0.0; 0.0];  % initial angular velocity [rad/s]

x0 = [q0; w0; theta_hat0];

%% Simulation settings
tspan = [0 300];       % seconds

% Integrate dynamics + adaptation
[t, x] = ode45(@(t,x) ode_rigid_body_adaptive(t,x,J_true,Kp,Kd,Gamma,...
                                              u_dither_amp,[w_d1 w_d2 w_d3]), ...
               tspan, x0, odeset('RelTol',1e-8,'AbsTol',1e-9));

%% Post-processing: reconstruct J_hat(t) and errors
N  = numel(t);
Jhat_hist  = zeros(3,3,N);
Frob_err   = zeros(N,1);
diag_true  = diag(J_true);
diag_hat   = zeros(N,3);
misalign   = zeros(N,1); % principal axis misalignment

for k = 1:N
    theta_hat_k = x(k,8:13).';
    Jhat_k      = theta2J(theta_hat_k);
    Jhat_hist(:,:,k) = Jhat_k;
    Frob_err(k) = norm(Jhat_k - J_true, 'fro');
    diag_hat(k,:) = diag(Jhat_k).';
    % principal axis misalignment (first eigenvector)
    [Vt,~]  = eig(J_true);
    [Ve,~]  = eig(Jhat_k);
    v1_true = Vt(:,1);
    v1_hat  = Ve(:,1);
    ang     = acos( max(-1,min(1, v1_true.'*v1_hat )) );
    misalign(k) = rad2deg(ang);
end

%% Plots

figure;
subplot(3,1,1);
plot(t, repmat(diag_true.',N,1), '--', 'LineWidth',1.2); hold on;
plot(t, diag_hat, 'LineWidth',1.4);
xlabel('Time [s]');
ylabel('J_{ii} [kg m^2]');
legend({'J_{xx,true}','J_{yy,true}','J_{zz,true}', ...
        'J_{xx,hat}','J_{yy,hat}','J_{zz,hat}'}, 'Location','best');
title('Diagonal inertia elements: true vs estimate');
grid on;

subplot(3,1,2);
plot(t, Frob_err, 'LineWidth',1.4);
xlabel('Time [s]');
ylabel('\|J_{hat} - J_{true}\|_F');
title('Frobenius norm of inertia estimation error');
grid on;

subplot(3,1,3);
plot(t, misalign, 'LineWidth',1.4);
xlabel('Time [s]');
ylabel('Angle [deg]');
title('Misalignment of first principal axis (true vs estimate)');
grid on;

% Print final matrices
J_hat_final = theta2J(x(end,8:13).');
disp('True inertia J_true:');
disp(J_true);
disp('Initial estimated inertia J_hat0 (from trace-1 estimator, misaligned + wrong scale):');
disp(J_hat0);
disp('Final estimated inertia J_hat_final:');
disp(J_hat_final);
disp('Final Frobenius error:');
disp(Frob_err(end));

%% ---- Helper functions -------------------------------------------------

function dx = ode_rigid_body_adaptive(t,x,J_true,Kp,Kd,Gamma,u_dither_amp,wd)
    % State unpacking
    q = x(1:4);
    w = x(5:7);
    theta_hat = x(8:13);   % [Jxx Jyy Jzz Jxy Jxz Jyz]^T

    % Normalize quaternion to avoid drift
    q = q / norm(q + 1e-15);

    % Desired attitude and angular velocity (attitude hold at identity)
    q_d = [1;0;0;0];               % desired quaternion
    w_d = [0;0;0];                 % desired angular velocity

    % Attitude error (q_e = q_d*^{-1} ⊗ q)
    q_d_conj = [q_d(1); -q_d(2:4)];
    q_e = quat_mul(q_d_conj, q);
    if q_e(1) < 0
        q_e = -q_e;  % avoid unwinding
    end
    e_R = q_e(2:4);             % vector part
    e_w = w - w_d;              % angular rate error

    % PD control torque
    u_fb = -Kp * e_R - Kd * e_w;

    % Excitation torque (dither) for identifiability
    u_d = u_dither_amp * [sin(wd(1)*t);
                          sin(wd(2)*t);
                          sin(wd(3)*t)];
    u = u_fb + u_d;

    % True rotational dynamics
    % J_true * dw = u - w × (J_true * w)
    tau_coriolis = cross(w, J_true*w);
    dw = J_true \ (u - tau_coriolis);

    % Quaternion kinematics
    qdot = 0.5 * omega_mat(w) * q;

    % Inertia parameter adaptation (gradient on torque prediction error)
    % Model torque: tau_hat = Y(theta_hat) * theta_hat
    Y = build_regressor(w, dw);    % 3x6 regressor
    tau_hat = Y * theta_hat;       % predicted torque
    torque_err = u - tau_hat;      % 3x1

    theta_dot = Gamma * (Y.' * torque_err);  % 6x1

    % Pack derivatives
    dx = [qdot;
          dw;
          theta_dot];
end

function Y = build_regressor(w,dw)
    % Build 3x6 regressor Y such that:
    % tau = J*dw + w×(J w) = Y * theta
    % with theta = [Jxx Jyy Jzz Jxy Jxz Jyz]^T.
    Y = zeros(3,6);
    for k = 1:6
        E = basis_matrix(k);
        tau_k = E*dw + cross(w, E*w);
        Y(:,k) = tau_k;
    end
end

function E = basis_matrix(k)
    % Basis matrices for symmetric inertia:
    % theta = [Jxx Jyy Jzz Jxy Jxz Jyz]^T
    E = zeros(3,3);
    switch k
        case 1
            E(1,1) = 1;
        case 2
            E(2,2) = 1;
        case 3
            E(3,3) = 1;
        case 4
            E(1,2) = 1; E(2,1) = 1;
        case 5
            E(1,3) = 1; E(3,1) = 1;
        case 6
            E(2,3) = 1; E(3,2) = 1;
    end
end

function theta = J2theta(J)
    % Pack symmetric 3x3 matrix J into theta = [Jxx Jyy Jzz Jxy Jxz Jyz]^T
    theta = [J(1,1);
             J(2,2);
             J(3,3);
             J(1,2);
             J(1,3);
             J(2,3)];
end

function J = theta2J(theta)
    % Unpack theta = [Jxx Jyy Jzz Jxy Jxz Jyz]^T into symmetric 3x3 J
    Jxx = theta(1);
    Jyy = theta(2);
    Jzz = theta(3);
    Jxy = theta(4);
    Jxz = theta(5);
    Jyz = theta(6);

    J = [Jxx, Jxy, Jxz;
         Jxy, Jyy, Jyz;
         Jxz, Jyz, Jzz];
end

function S = skew(v)
    % Skew-symmetric matrix for cross product
    S = [  0   -v(3)  v(2);
          v(3)   0   -v(1);
         -v(2) v(1)   0  ];
end

function Om = omega_mat(w)
    % Quaternion kinematics matrix Ω(ω) such that qdot = 0.5*Ω(ω)*q
    wx = w(1); wy = w(2); wz = w(3);
    Om = [ 0   -wx  -wy  -wz;
           wx   0    wz  -wy;
           wy  -wz   0    wx;
           wz   wy  -wx   0 ];
end

function qc = quat_conj(q)
    qc = [q(1); -q(2:4)];
end

function q_prod = quat_mul(q1,q2)
    % Quaternion product q_prod = q1 ⊗ q2
    w1 = q1(1); v1 = q1(2:4);
    w2 = q2(1); v2 = q2(2:4);
    w  = w1*w2 - dot(v1,v2);
    v  = w1*v2 + w2*v1 + cross(v1,v2);
    q_prod = [w; v];
end
