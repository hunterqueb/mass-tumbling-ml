function demo_mrac_attitude_quat()
  clear; clc;
  tf = 200;

  % ===== Two-phase schedule =====
  t_id   = 0;      %#ok<NASGU> (kept for clarity)
  t_swap = 120;     % enable MRAC adaptation after 120 s


  % ===== True inertia (plant) =====
  J_true = 100 * [ 0.3905  0.0567  0.1052
                   0.0567  0.3803 -0.1150
                   0.1052 -0.1150  0.2292 ] + diag([20,25,15]);
  J_true = (J_true + J_true')/2;

  % ===== MRAC reference model (per-axis) =====
  zeta = 0.9; wn = 0.4;
  Am = [0 1; -wn^2 -2*zeta*wn];
  Bm = [0; 1];
  Q  = eye(2);
  P  = lyap(Am', Q);                   % A_m^T P + P A_m = -Q

  % ===== MRAC adaptation: make slow (so ID can't be “absorbed”) =====
  GammaK = 1e-2 * diag([1.0, 1.0]);   % << slower than before
  sigmaK = 0.05;                       % stronger leakage

  % ===== Inertia identifier: make fast + normalized =====
  gammaJ = 5e-1;                       % << 100x faster than before
  % normalized by ||wdot||^2 inside the ODE

  % ===== Dither for persistent excitation =====
  dith.A  = 0.10;                      % Nm (small vs actuator)
  dith.w  = 2*pi*[0.5, 0.8, 1.3];      % rad/s
  dith.ph = [0.0, 0.7, 1.1];           % phases


  % ===== Initial conditions (multi-axis excitation) =====
  ang0  = deg2rad(25);
  axis0 = [1;0.6;0.3]; axis0 = axis0/norm(axis0);
  q0    = axang2quat(axis0, ang0);     % body->LVLH
  w0    = deg2rad([0.8; -0.6; 0.7]);

  % Reference model initial state matches small-angle plant state
  dtheta0 = 2*q0(2:4);
  x_m0    = [dtheta0.'; w0.'];         % [th_m, thd_m] per axis stacked

  % MRAC parameters initial guesses (per axis): k=[k1;k2] (slow → start near 0)
  k0 = [0;0; 0;0; 0;0];

  % J identifier initial guess (very wrong → force movement)
  Jhat0 = [[ 0.3466  0.0132  0.0233]
     [ 0.0132  0.3365 -0.0185]
     [ 0.0233 -0.0185  0.3169]];                   % symmetric
  Jhat0 = 100 * eye(3);
  j0 = [Jhat0(1,1); Jhat0(2,2); Jhat0(3,3); Jhat0(1,2); Jhat0(1,3); Jhat0(2,3)];

  % Pack state: [ q(4); w(3); x_m(6); k(6); j(6) ]
  x0 = [q0; w0; x_m0(:); k0; j0];

  % Bundle params
  pars.Jtrue  = J_true;
  pars.Am     = Am; pars.Bm = Bm; pars.P = P;
  pars.GammaK = GammaK; pars.sigmaK = sigmaK;
  pars.gammaJ = gammaJ;
  pars.dith   = dith;
  pars.tswap  = t_swap;
  pars.floorJ = 1e-2;                  % eigenvalue floor (plotting/robustness)

  % Integrate
  tspan = [0 tf];
  opts  = odeset('RelTol',1e-9,'AbsTol',1e-9);
  [t,x] = ode45(@(t,x) dyn_mrac_quat_id_v2(t,x,pars), tspan, x0, opts);

  % ===== Unpack & post =====
  q = x(:,1:4); for k=1:size(q,1), q(k,:)=q(k,:)/norm(q(k,:)); end
  w = x(:,5:7);
  dtheta = 2*q(:,2:4);

  xm   = x(:,8:13);                    % [th_mx thd_mx th_my thd_my th_mz thd_mz]
  th_m = xm(:,1:2:end);
  td_m = xm(:,2:2:end);

  kh = x(:,14:19);                     % [k1x k2x k1y k2y k1z k2z]
  jh = x(:,20:25);                     % [j11 j22 j33 j12 j13 j23]

  N = numel(t);
  Jhat_diag = zeros(N,3);
  JtF = zeros(N,1);
  for i=1:N
      Ji = [jh(i,1) jh(i,4) jh(i,5);
            jh(i,4) jh(i,2) jh(i,6);
            jh(i,5) jh(i,6) jh(i,3)];
      Ji = (Ji + Ji')/2;
      [V,D] = eig(Ji); D = max(D, pars.floorJ*eye(3)); Ji = V*D*V';  % SPD floor (for plotting)
      Jhat_diag(i,:) = diag(Ji).';
      JtF(i) = norm(Ji - J_true, 'fro');
  end

  % ===== Plots =====
  figure('Name','Plant vs MRAC Reference');
  subplot(2,1,1);
  plot(t, rad2deg(dtheta),'LineWidth',1.2); hold on;
  plot(t, rad2deg(th_m),'--','LineWidth',1.1);
  grid on; ylabel('\theta [deg]');
  legend('\theta_x','\theta_y','\theta_z','\theta_{m,x}','\theta_{m,y}','\theta_{m,z}','Location','best');
  title('Small-angle plant vs MRAC model');

  subplot(2,1,2);
  plot(t, rad2deg(w),'LineWidth',1.2); hold on;
  plot(t, rad2deg(td_m),'--','LineWidth',1.1);
  grid on; xlabel('t [s]'); ylabel('\dot{\theta} [deg/s]');
  legend('\omega_x','\omega_y','\omega_z','\dot{\theta}_{m,x}','\dot{\theta}_{m,y}','\dot{\theta}_{m,z}','Location','best');

  figure('Name','Adaptive Gains (slow MRAC)');
  subplot(3,1,1); plot(t, kh(:,1:2),'LineWidth',1.2); grid on; ylabel('[k1_x,k2_x]');
  subplot(3,1,2); plot(t, kh(:,3:4),'LineWidth',1.2); grid on; ylabel('[k1_y,k2_y]');
  subplot(3,1,3); plot(t, kh(:,5:6),'LineWidth',1.2); grid on; ylabel('[k1_z,k2_z]'); xlabel('t [s]');

  figure('Name','Inertia Identification');
  subplot(2,1,1);
  plot(t, Jhat_diag, 'LineWidth', 1.4); hold on;
  yl = diag(J_true).';
  yline(yl(1),':'); yline(yl(2),':'); yline(yl(3),':');
  grid on; ylabel('diag(\hat J) [kg m^2]');
  legend('\hat J_{11}','\hat J_{22}','\hat J_{33}','J_{11}^{true}','J_{22}^{true}','J_{33}^{true}','Location','best');
  title('Identifier-only \hat J (fast, normalized, PE with dither)');

  subplot(2,1,2);
  plot(t, JtF, 'LineWidth', 1.4);
  grid on; xlabel('t [s]'); ylabel('||\hat J - J||_F');
  title('Frobenius norm error (should decrease once ID kicks in)');
end

% ================= Dynamics (MRAC + fast normalized J-ID + dither + phase swap) =================
function dx = dyn_mrac_quat_id_v2(t, x, p)
  % State: [ q(4); w(3); x_m(6); k(6); j(6) ]
  q  = x(1:4);
  w  = x(5:7);
  xm = x(8:13);
  k  = x(14:19);
  j  = x(20:25);

  % Normalize quaternion
  q = q / max(norm(q),1e-12);

  % Small-angle variables
  theta  = 2*q(2:4);
  thetad = w;

  % Two-phase: freeze MRAC gains before tswap
  freezeK = (t < p.tswap);

  % Dither torque
  u_dith = p.dith.A * [ sin(p.dith.w(1)*t + p.dith.ph(1));
                        sin(p.dith.w(2)*t + p.dith.ph(2));
                        sin(p.dith.w(3)*t + p.dith.ph(3)) ];

  % MRAC control (per axis) + dither (no Jhat in control → pure MRAC)
  u = [ k(1)*theta(1) + k(2)*thetad(1);
        k(3)*theta(2) + k(4)*thetad(2);
        k(5)*theta(3) + k(6)*thetad(3) ] + u_dith;

  % Plant dynamics (true J)
  J  = p.Jtrue;
  Jw = J*w;
  wdot = J \ (u - cross(w, Jw));

  % Quaternion kinematics
  wx=w(1); wy=w(2); wz=w(3);
  W = [  0   -wx  -wy  -wz;
        wx    0    wz  -wy;
        wy   -wz   0    wx;
        wz    wy  -wx   0 ];
  qdot = 0.5 * W * q;

  % Reference model (per axis): x_m_dot = Am x_m (r=0)
  Am = p.Am;
  xmdot = zeros(6,1);
  for i = 1:3
      idx = (2*i-1):(2*i);
      xmi = xm(idx);
      xmd = Am * xmi;
      xmdot(idx) = xmd;
  end

  % MRAC adaptation (slow; optionally frozen)
  P = p.P; Gamma = p.GammaK; sigma = p.sigmaK;
  kdot = zeros(6,1);
  for i = 1:3
      idx  = (2*i-1):(2*i);
      ki   = k(2*i-1:2*i);                 % [k1_i; k2_i]
      xi   = [theta(i); thetad(i)];
      xmi  = xm(idx);
      ei   = xi - xmi;
      s    = (ei.' * P * [0;1]);          % e' P Bm  (Bm=[0;1])
      kid  = -Gamma * (xi * s) - sigma * ki;
      if freezeK, kid(:) = 0; end         % phase 1: ID-only
      kdot(2*i-1:2*i) = kid;
  end

  % Identifier for J (full symmetric, fast + normalized)
  Jhat = [j(1) j(4) j(5);
          j(4) j(2) j(6);
          j(5) j(6) j(3)];
  Jhat = (Jhat + Jhat')/2;

  % Torque prediction error: tau_err = u - (Jhat*wdot + w x (Jhat w))
  tau_err = u - (Jhat*wdot + cross(w, Jhat*w));

  % Normalized gradient step (denominator prevents vanishing updates)
  den = (wdot.'*wdot) + 1e-6;
  Jhat_dot = -(p.gammaJ/den) * 0.5 * (tau_err*wdot' + wdot*tau_err');
  Jhat_dot = (Jhat_dot + Jhat_dot')/2;

  jdot = [ Jhat_dot(1,1);
           Jhat_dot(2,2);
           Jhat_dot(3,3);
           Jhat_dot(1,2);
           Jhat_dot(1,3);
           Jhat_dot(2,3) ];

  dx = [qdot; wdot; xmdot; kdot; jdot];
end

% ================= helpers =================
function q = axang2quat(axis, ang)
  axis = axis(:)/norm(axis);
  c = cos(ang/2); s = sin(ang/2);
  q = [c; s*axis];
end
