function r = rotm_log(R)
% Logarithm map from rotation matrix R (SO(3)) to so(3) vector r

    % Ensure orthonormality symmetrically
    R = projectSO3(R);

    tr = trace(R);
    c  = (tr - 1) / 2;
    c  = max(min(c,1),-1);      % clamp for numerical errors
    theta = acos(c);

    if theta < 1e-12
        r = [0;0;0];
        return;
    end

    v = (1/(2*sin(theta))) * [ R(3,2) - R(2,3);
                               R(1,3) - R(3,1);
                               R(2,1) - R(1,2) ];
    r = theta * v;
end

