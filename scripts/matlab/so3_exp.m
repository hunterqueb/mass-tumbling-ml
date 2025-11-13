function R = so3_exp(r)
% Exponential map from so(3) vector r to rotation matrix R

    theta = norm(r);
    if theta < 1e-12
        % first-order approximation
        R = eye(3) + skew(r);
        return;
    end

    k = r / theta;
    K = skew(k);

    R = eye(3) + sin(theta)*K + (1 - cos(theta))*(K*K);
end



