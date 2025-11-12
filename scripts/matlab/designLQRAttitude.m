function K = designLQRAttitude(J, q_att_weight, q_omega_weight, r_torque_weight)
% Design continuous-time LQR gain for small-angle attitude model
% x = [delta_theta; omega], u = torque
%
% J : 3x3 inertia (for the current configuration)
% q_att_weight    : scalar weight on attitude error
% q_omega_weight  : scalar weight on rates
% r_torque_weight : scalar weight on control effort

    A = [zeros(3), eye(3);
         zeros(3), zeros(3)];

    B = [zeros(3);
         inv(J)];

    Q = blkdiag(q_att_weight*eye(3), q_omega_weight*eye(3));
    R = r_torque_weight*eye(3);

    [K,~,~] = lqr(A,B,Q,R);   % K is 3x6, u = -K x

end
