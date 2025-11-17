function [J_hat_NN,J_hat_rand,J_true,Jt_true] = demo_adapt_partial_target()
  rng(0);
  tf = 100;
  % -------- Fixed (chaser) --------
  J_fixed = diag([20,25,15]);

  % -------- Weaken feedback (so inertia matters) --------
  K_R  = 30*eye(3);
  K_Om = 10*eye(3);

  % -------- Fast adaptation + normalization --------
  Gamma = 10 * diag([0.3,0.3,0.3]);      % much larger than before
  eps_reg = 1e-6;                   % regressor power floor


  % -------- Estimated target (principal frame) --------
  Jt_est  = diag([0.188065 0.405968 0.405968]);
  [Vt,L0] = eig((Jt_est+Jt_est')/2);      lambda0 = diag(L0);
  %R_bt = eye(3);                          % set your docked orientation if known
  R_bt = rotx(4)*roty(2)*rotz(7);                          % set your docked orientation if known
  %R_bt = rotx(41)*roty(72)*rotz(67);                          % set your docked orientation if known

  % Rank-1 basis S_i = lambda0_i * (R v_i)(R v_i)^T  (in BODY frame)
  v1=R_bt*Vt(:,1); v2=R_bt*Vt(:,2); v3=R_bt*Vt(:,3);  S1 = lambda0(1)*(v1*v1.'); S2 = lambda0(2)*(v2*v2.'); S3 = lambda0(3)*(v3*v3.');
  S  = cat(3,S1,S2,S3);

  % -------- True docked inertia (plant) --------
  J_true = 100 * diag([0.146925 0.417965 0.435109]) + J_fixed;
  Jt_true = J_true - J_fixed;

  % Best-fit alpha* in chosen basis (for plotting)
  A = [vec(S1), vec(S2), vec(S3)];  b = vec(Jt_true);
  alpha_star = A\b;

  % -------- Multi-axis IC + dither torque --------
  ang_deg = 25;
  axis0 = [1;0.7;0.4]; axis0=axis0/norm(axis0);
  q0 = axang2quat(axis0, deg2rad(ang_deg));
  w0 = deg2rad([1.2; -0.8; 0.9]);

  dith.A = 0.05;                    % Nm (keep small vs actuator limits)
  dith.w = 2*pi*[0.4, 0.7, 1.1];    % three distinct freqs [rad/s]
  alpha0 = [1;1;1];

  x0 = [q0; w0; alpha0];

  odef = @(t,x) dyn_partial_adapt_v2(t,x,J_fixed,S,J_true,K_R,K_Om,Gamma,eps_reg,dith);
  opts = odeset('RelTol',1e-12,'AbsTol',1e-12);
  [t,x] = ode45(odef,[0 tf],x0,opts);

  % Unpack
  q = x(:,1:4); for k=1:size(q,1), q(k,:)=q(k,:)/norm(q(k,:)); end
  w = x(:,5:7);
  alpha = x(:,8:10);

  % Build estimates
  N = numel(t);
  Jt_hat_diag = zeros(N,3);
  J_hat_diag  = zeros(N,3);
  Frob_err   = zeros(N,1);

  for k=1:N
      Jt_hat = alpha(k,1)*S(:,:,1) + alpha(k,2)*S(:,:,2) + alpha(k,3)*S(:,:,3);
      Jt_hat = (Jt_hat+Jt_hat')/2;
      Jt_hat_diag(k,:) = diag(Jt_hat).';
      J_hat_diag(k,:)  = diag(J_fixed + Jt_hat).';
      Frob_err(k) = norm(Jt_hat - Jt_true, 'fro');

  end

  dtheta = 2*q(:,2:4);

  % ---------- Plots ----------
  figure('Name','Attitude & Rates (NN Estimation)');
  subplot(2,1,1); plot(t,rad2deg(dtheta),'LineWidth',1.2);
  ylabel('\delta\theta [deg]'); grid on; legend('\delta\theta_x','\delta\theta_y','\delta\theta_z');
  title('Attitude & Rates (NN Estimation)');

  subplot(2,1,2); plot(t,rad2deg(w),'LineWidth',1.2);
  xlabel('t [s]'); ylabel('\omega [deg/s]'); grid on; legend('\omega_x','\omega_y','\omega_z');

  figure('Name','Target inertia (NN Estimation)');
  plot(t, Jt_hat_diag, 'LineWidth',1.4); hold on;
  Jt_true_diag = diag(Jt_true);
  yline(Jt_true_diag(1),':'); yline(Jt_true_diag(2),':'); yline(Jt_true_diag(3),':');
  xlabel('t [s]'); ylabel('diag(J_{target}) [kg m^2]'); grid on;
  legend('\hat J_{t,11}','\hat J_{t,22}','\hat J_{t,33}','J_{t,11}^{true}','J_{t,22}^{true}','J_{t,33}^{true}');
  title('Target inertia (NN Estimation)');

  figure('Name','Total inertia diagonal (NN Estimation)');
  plot(t, J_hat_diag, 'LineWidth',1.4); hold on;
  J_true_diag = diag(J_true);
  yline(J_true_diag(1),':'); yline(J_true_diag(2),':'); yline(J_true_diag(3),':');
  xlabel('t [s]'); ylabel('diag(J_{total}) [kg m^2]'); grid on;
  legend('\hat J_{11}','\hat J_{22}','\hat J_{33}','J_{11}^{true}','J_{22}^{true}','J_{33}^{true}');

  figure('Name','Alpha vs best-fit (NN Estimation)');
  plot(t,alpha,'LineWidth',1.4); hold on;
  yline(alpha_star(1),':'); yline(alpha_star(2),':'); yline(alpha_star(3),':');
  xlabel('t [s]'); ylabel('\alpha_i'); grid on;
  legend('\alpha_1','\alpha_2','\alpha_3','\alpha^\ast_1','\alpha^\ast_2','\alpha^\ast_3');
  title('Faster adaptation + PE → \alpha(t) → \alpha^\ast');

  figure()
  plot(t, Frob_err, 'LineWidth',1.4);
  xlabel('Time [s]');
  ylabel('\|J_{hat} - J_{true}\|_F');
  title('Frobenius norm of inertia estimation error');
  grid on;
  J_hat_NN = Jt_hat;
%% random guess of J_target

  % -------- Estimated target (principal frame) --------
  Jt_est  = diag(rand(3))';
  [Vt,L0] = eig((Jt_est+Jt_est')/2);      lambda0 = diag(L0);

  % Rank-1 basis S_i = lambda0_i * (R v_i)(R v_i)^T  (in BODY frame)
  v1=R_bt*Vt(:,1); v2=R_bt*Vt(:,2); v3=R_bt*Vt(:,3);
  S1 = lambda0(1)*(v1*v1.'); S2 = lambda0(2)*(v2*v2.'); S3 = lambda0(3)*(v3*v3.');
  S  = cat(3,S1,S2,S3);

  % -------- True docked inertia (plant) --------
  J_true = 100 * diag([0.146925 0.417965 0.435109]) + J_fixed;
  Jt_true = J_true - J_fixed;

  % Best-fit alpha* in chosen basis (for plotting)
  A = [vec(S1), vec(S2), vec(S3)];  b = vec(Jt_true);
  alpha_star = A\b;

  % -------- Weaken feedback (so inertia matters) --------
  K_R  = .3*eye(3);
  K_Om = .1*eye(3);

  % -------- Fast adaptation + normalization --------
  Gamma = 10*diag([0.3,0.3,0.3]);      % much larger than before
  eps_reg = 1e-6;                   % regressor power floor

  % -------- Multi-axis IC + dither torque --------
  ang_deg = 25;
  axis0 = [1;0.7;0.4]; axis0=axis0/norm(axis0);
  q0 = axang2quat(axis0, deg2rad(ang_deg));
  w0 = deg2rad([1.2; -0.8; 0.9]);

  dith.A = 0.05;                    % Nm (keep small vs actuator limits)
  dith.w = 2*pi*[0.4, 0.7, 1.1];    % three distinct freqs [rad/s]
  alpha0 = [1;1;1];

  x0 = [q0; w0; alpha0];

  odef = @(t,x) dyn_partial_adapt_v2(t,x,J_fixed,S,J_true,K_R,K_Om,Gamma,eps_reg,dith);
  opts = odeset('RelTol',1e-12,'AbsTol',1e-12);
  [t,x] = ode45(odef,[0 tf],x0,opts);

  % Unpack
  q = x(:,1:4); for k=1:size(q,1), q(k,:)=q(k,:)/norm(q(k,:)); end
  w = x(:,5:7);
  alpha = x(:,8:10);

  % Build estimates
  N = numel(t);
  Jt_hat_diag = zeros(N,3);
  J_hat_diag  = zeros(N,3);
  Frob_err   = zeros(N,1);

  for k=1:N
      Jt_hat = alpha(k,1)*S(:,:,1) + alpha(k,2)*S(:,:,2) + alpha(k,3)*S(:,:,3);
      Jt_hat = (Jt_hat+Jt_hat')/2;
      Jt_hat_diag(k,:) = diag(Jt_hat).';
      J_hat_diag(k,:)  = diag(J_fixed + Jt_hat).';
      Frob_err(k) = norm(Jt_hat - Jt_true, 'fro');

  end

  dtheta = 2*q(:,2:4);

  % ---------- Plots ----------
  figure('Name','Attitude & Rates (Random Initialization)');
  subplot(2,1,1); plot(t,rad2deg(dtheta),'LineWidth',1.2);
  ylabel('\delta\theta [deg]'); grid on; legend('\delta\theta_x','\delta\theta_y','\delta\theta_z');
  title('Attitude & Rates (Random Initialization)');

  subplot(2,1,2); plot(t,rad2deg(w),'LineWidth',1.2);
  xlabel('t [s]'); ylabel('\omega [deg/s]'); grid on; legend('\omega_x','\omega_y','\omega_z');

  figure('Name','Target inertia (Random Initialization)');
  plot(t, Jt_hat_diag, 'LineWidth',1.4); hold on;
  Jt_true_diag = diag(Jt_true);
  yline(Jt_true_diag(1),':'); yline(Jt_true_diag(2),':'); yline(Jt_true_diag(3),':');
  xlabel('t [s]'); ylabel('diag(J_{target}) [kg m^2]'); grid on;
  legend('\hat J_{t,11}','\hat J_{t,22}','\hat J_{t,33}','J_{t,11}^{true}','J_{t,22}^{true}','J_{t,33}^{true}');
  title('Target inertia (Random Initialization)');

  figure('Name','Total inertia diagonal (Random Initialization)');
  plot(t, J_hat_diag, 'LineWidth',1.4); hold on;
  J_true_diag = diag(J_true);
  yline(J_true_diag(1),':'); yline(J_true_diag(2),':'); yline(J_true_diag(3),':');
  xlabel('t [s]'); ylabel('diag(J_{total}) [kg m^2]'); grid on;
  legend('\hat J_{11}','\hat J_{22}','\hat J_{33}','J_{11}^{true}','J_{22}^{true}','J_{33}^{true}');

  figure('Name','Alpha vs best-fit (Random Initialization)');
  plot(t,alpha,'LineWidth',1.4); hold on;
  yline(alpha_star(1),':'); yline(alpha_star(2),':'); yline(alpha_star(3),':');
  xlabel('t [s]'); ylabel('\alpha_i'); grid on;
  legend('\alpha_1','\alpha_2','\alpha_3','\alpha^\ast_1','\alpha^\ast_2','\alpha^\ast_3');
  title('Faster adaptation + PE → \alpha(t) → \alpha^\ast');

  figure()
  plot(t, Frob_err, 'LineWidth',1.4);
  xlabel('Time [s]');
  ylabel('\|J_{hat} - J_{true}\|_F');
  title('Frobenius norm of inertia estimation error');
  grid on;
  J_hat_rand = Jt_hat;
end

function dx = dyn_partial_adapt_v2(t,x,J_fixed,S,J_true,K_R,K_Om,Gamma,eps_reg,dith)
  q = x(1:4);  w = x(5:7);  alpha = x(8:10);
  q = q / max(norm(q),1e-12);

  % Build Jhat = J_fixed + sum_i alpha_i S_i
  Jt_hat = alpha(1)*S(:,:,1) + alpha(2)*S(:,:,2) + alpha(3)*S(:,:,3);
  Jt_hat = (Jt_hat+Jt_hat')/2;
  Jhat = J_fixed + Jt_hat;

  % Small-angle error
  eR = 2*q(2:4);

  % PD
  uPD = -K_R*eR - K_Om*w;

  % Multi-axis dither (small)
  u_dith = dith.A * [sin(dith.w(1)*t);
                     sin(dith.w(2)*t + 0.7);
                     sin(dith.w(3)*t + 1.3)];

  % Control torque (Jhat affects gyro comp; no algebraic cancellation)
  u = uPD + cross(w, Jhat*w) + u_dith;

  % True plant dynamics
  wdot = J_true \ (u - cross(w, J_true*w));

  % Quaternion kinematics
  wx=w(1); wy=w(2); wz=w(3);
  W = [0 -wx -wy -wz; wx 0 wz -wy; wy -wz 0 wx; wz wy -wx 0];
  qdot = 0.5*W*q;

  % --- Adaptation on target principal moments only ---
  % tau_err = uPD + u_dith - Jhat*wdot  (cross terms cancel)
  tau_err = (uPD + u_dith) - Jhat*wdot;

  % Regressor Phi = [-(S1*wdot), -(S2*wdot), -(S3*wdot)]  (3x3)
  Phi = [-(S(:,:,1)*wdot), -(S(:,:,2)*wdot), -(S(:,:,3)*wdot)];

  % Normalized gradient: alpha_dot = Gamma * Phi' * tau_err / (trace(Phi'*Phi)+eps)
  denom = trace(Phi.'*Phi) + eps_reg;
  alpha_dot = -Gamma * (Phi.' * tau_err) / denom;

  dx = [qdot; wdot; alpha_dot];
end

% -------- helpers --------
function q = axang2quat(axis, ang)
  c = cos(ang/2); s = sin(ang/2); axis = axis(:)/norm(axis);
  q = [c; s*axis];
end
function v = vec(M), v = M(:); end
