function domega = eulerTorqueFreePrincipal(~, omega, I1, I2, I3)
    w1 = omega(1);
    w2 = omega(2);
    w3 = omega(3);

    domega = zeros(3,1);
    domega(1) = ((I2 - I3)/I1) * w2 * w3;
    domega(2) = ((I3 - I1)/I2) * w3 * w1;
    domega(3) = ((I1 - I2)/I3) * w1 * w2;
end
