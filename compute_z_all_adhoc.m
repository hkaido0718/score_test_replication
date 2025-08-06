function [z_delta, z_beta] = compute_z_all_adhoc(beta, x, delta, p)
    % Input:
%  beta:   structural parameter of interest (1 by 2 vector)
%  x:      Covariates (4 by 2) [1,1; 1,-1; -1,1; -1,-1].
%  delta:  nuisance parameter (2 by 1)
%
% Output:
%  z_delta: a 2 by 16 matrix with the following form
%                          x = [1,1]         |    x=[1,-1]        ...
%                 (0, 0) (1, 1) (1, 0) (0, 1)| (0, 0) ... (0, 1)
%           delta1                           |
%           delta2                           |
%  z_beta: a 2 by 16 matrix with the following form
%                          x = [1,1]         |    x=[1,-1]        ...
%                 (0, 0) (1, 1) (1, 0) (0, 1)| (0, 0) ... (0, 1)
%           beta1                            |
%           beta2                            |
    delta = delta';
    % All of the following variables are 4 by 2.
    xdelta = x.*delta;
    phi = normpdf(xdelta);
    Phi = normcdf(xdelta);
    phi_beta = normpdf(xdelta + beta);
    Phi_beta = normcdf(xdelta + beta);

    % For events (0, 0) and (1, 1):
    zbeta_00 = zeros(4, 2);
    zdelta_00 = -phi.*x./(1-Phi);
    zbeta_11 = phi_beta./Phi_beta;
    zdelta_11 = phi_beta.*x./Phi_beta;

    % For events (1, 0)
    % Now for events (1, 0) and (0, 1), define the conditions
    dim = size(x);
    % pre-allocation
    zdelta_1001 = zeros(2, 2*dim(1));
    zbeta_1001 = zeros(2, 2*dim(1));
    for i = 1:dim(1) % loop over covariate configurations
        q_01 = Phi(i,2)*(1-Phi(i,1))+(Phi(i,1)-Phi_beta(i,1))*(p*Phi(i,2)+(1-p)*Phi_beta(i,2));
        q_10 = (1-Phi(i,2))*Phi(i,1)+(Phi(i,2)-Phi_beta(i,2))*((1-p)*Phi(i,1)+p*Phi_beta(i,1));

        zdelta1_01 = ((1-p)*(Phi_beta(i,2)-Phi(i,2))*phi(i,1)*x(i,1))/q_01 - ((p*Phi(i,2)+(1-p)*Phi_beta(i,2))*phi_beta(i,1)*x(i,1))/q_01;
        zdelta2_01 = (1-(1-p)*Phi(i,1)-p*Phi_beta(i,1))*phi(i,2)*x(i,2)/q_01 - ((1-p)*(Phi_beta(i,1)-Phi(i,1))*phi_beta(i,2)*x(i,2))/q_01;
        zbeta1_01 = -(phi_beta(i,1)*(p*Phi(i,2)+(1-p)*Phi_beta(i,2)))/q_01;
        zbeta2_01 = ((1-p)*(Phi(i,1)-Phi_beta(i,1))*phi_beta(i,2))/q_01;

        zdelta1_10 = ((1-p*Phi(i,2)-(1-p)*Phi_beta(i,2))*phi(i,1)*x(i,1))/q_10 + (p*(Phi(i,2)-Phi_beta(i,2))*phi_beta(i,1)*x(i,1))/q_10;
        zdelta2_10 = ((2-p)*Phi(i,1)+p*Phi_beta(i,1))*phi(i,2)*x(i,2)/q_10 - (((1-p)*Phi(i,1)+p*Phi_beta(i,1))*phi_beta(i,2)*x(i,2))/q_10;
        zbeta1_10 = (p*phi_beta(i,1)*(Phi(i,2)-Phi_beta(i,2)))/q_10;
        zbeta2_10 = -(phi_beta(i,2)*((1-p)*Phi(i,1)+p*Phi_beta(i,1)))/q_10;
        % stacking up the results
        zdelta_1001(:, (2*i-1):(2*i)) = [zdelta1_10, zdelta1_01; zdelta2_10, zdelta2_01];
        zbeta_1001(:, (2*i-1):(2*i)) = [zbeta1_10, zbeta1_01; zbeta2_10, zbeta2_01];
    end

    % returning zs
    z_delta = zeros(2, 4*dim(1));
    z_beta = zeros(2, 4*dim(1));
    for i = 1:dim(1)
        z_delta(:,1+4*(i-1)) = zdelta_00(i,:);
        z_beta(:,1+4*(i-1)) = zbeta_00(i,:);
        z_delta(:,2+4*(i-1)) = zdelta_11(i,:);
        z_beta(:,2+4*(i-1)) = zbeta_11(i,:);
        z_delta(:,(3+4*(i-1)):(4*i)) = zdelta_1001(:,(2*i-1):(2*i));
        z_beta(:,(3+4*(i-1)):(4*i)) = zbeta_1001(:,(2*i-1):(2*i));
    end
end
