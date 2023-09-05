function [] = monte_carlo_main(ncores,varargin)
% 08/15/2023 Shuowen Chen and Hiroaki Kaido
%% Inputs %%
% ncores: # of CPUs used for parallelization
% step  : 'first_stage' 'second_stage_est', 'second_stage_true', or 'power'
% Choose one of them to decide which part of the program to debug.
%
% This file conducts Monte Carlo Simulations for project "Robust Test Score
% for Incomplete Models with Nuisance Parameters"
rng(123)
S = 1999; % number of simulations for each sample size
parpool(ncores)

% Size evaluation
covariate = [1, -1];
% true value of nuisance parameters
delta = [0.25, 0.25];
level = 0.05; % nominal size
N = [2500, 5000, 7500];
J = length(N);
X = [1,1; 1,-1; -1,1; -1,-1]; % combination of covariates
% pre-allocation to store g_delta statstics and sup test statistic
gn = zeros(2*length(N), S);
test = zeros(length(N), S);
outcome_com = zeros(16,S,length(N));
delta_hat_com1 = zeros(length(N),S);
delta_hat_com2 = zeros(length(N),S);
cv = zeros(length(N),S);
for j = 1:length(N) % loop over sample size
    n = N(j);
    % Call data_generation function to generate data
    [data_com, x1_com, x2_com] = data_generation([0, 0], covariate, ...
        delta, n, S, 'Complete');
    outcome_com(:,:,j) = counting(data_com, x1_com, x2_com);
    % loop over number of simulations
    gn_temp = zeros(2,S);
    parfor i = 1:S
        % Conduct Restricted MLE
        delta_hat_com = rmle([0,0], n, 'BFGS', data_com(:,i), x1_com(:,i), x2_com(:,i));
        delta_hat_com1(j,i) = delta_hat_com(1);
        delta_hat_com2(j,i) = delta_hat_com(2);
        % generate var-cov matrices and gn statistics (generated under the null beta=0)
        %[test(j,i),gn_temp(:,i), varI] = stat(delta_hat_com, X, outcome_com(:,i,j), n,lambda);
        [test(j,i),gn_temp(:,i), varI] = stat_ab(delta_hat_com, X, outcome_com(:,i,j), n);
        cv(j,i) = crt_ab(level,varI);
    end
    gn((2*j-1):(2*j),:) = gn_temp;
end
rej = test > cv;
size = sum(rej,2)./S;
disp(size)
filename = ['../Results/Matfiles/test_crt_S' num2str(S) '.mat'];
save(filename)





%% Power Evaluation
%% Random Number Generation %%
rng(123); % set seed
n = 7500;
X = [1,1; 1,-1; -1,1; -1,-1]; % combination of covariates
DGP = 'LFP';

% pre-define alternatives
h_alt  = -(eps:0.5:15)';
h_alt2 = -(eps:0.5:15)';
K = length(h_alt);
% pre-allocation to store sup test statistic for each selection mechanism
test = zeros(K, S);
cv   = zeros(K,S);
for k=1:K
    % Define true beta that generates the data (Change between beta_alt
    % and beta_alt2 in the data_generation function)
    beta_alt = h_alt(k,:)/sqrt(n);
    beta_alt2 = h_alt2(k,:)/sqrt(n);
    % Call data_generation function to generate data
    [data, x1, x2] = data_generation([beta_alt,beta_alt2], covariate, ...
        delta, n, S, DGP);
    outcome = counting(data, x1, x2);
    % loop over number of simulations
    parfor i = 1:S
        % Conduct Restricted MLE
        delta_hat = rmle([0,0], n, 'BFGS', data(:,i), x1(:,i), x2(:,i));
        % generate var-cov matrices and gn statistics (generated under the null beta=0)
        [test(k,i), ~,varI_iid] = stat_ab(delta_hat, X, outcome(:,i), n);
        cv(k,i) = crt_ab(level,varI_iid);
    end
end
date = datestr(now, 'yyyy_mm_dd_HH_MM_SS');
filename = ['../Results/Matfiles/test_power_DGP' DGP '_n' num2str(n) '_S' num2str(S) '_' date '.mat'];
save(filename)
end



% 5/21/2020 Shuowen Chen and Hiroaki Kaido
% Generates data
% When beta is 0 (null), model is complete, no selection needed, use
% 'Complete' instead
% When beta is not 0 (alternative), model features multiple equilibria,
% use one of the three selection mechanisms to determine the outcome:
% (1) iid, (2) non iid (nonergodic), and (3) LFP.
function [data, x1, x2] = data_generation(beta, covariate, delta, n, S, selection)
%  Input:
%   beta:       structural parameters of interest (1 by 2)
%   covariate:  covariates (1 by 2) we draw from: (1, -1)
%   delta:      coefficients of covariates (nuisance parameters) (1 by 2)
%   n:          sample size
%   S:          number of simulations
%   selection:  equilibrium selection mechanism
%  Output:
%   data:       the generated dataset (n by S). Each column denotes one
%               monte carlo simulation with sample size n.
%   x1:         sampled x for the first player (n by S)
%   x2:         sampled x for the second player (n by S)

% generate a sequence of x
x = randsample(covariate, n*2*S, true);
x1 = reshape(x(1:(n*S)), n, S);
x2 = reshape(x((n*S+1):length(x)), n, S);
% calculate xdelta
xdelta1 = x1*delta(1);
xdelta2 = x2*delta(2);

u1 = randn([n, S]);
u2 = randn([n, S]);

% Data generation under the null
if all(beta == 0) && strcmp(selection, 'Complete')
    data = NaN(n, S); % placeholder for data
    % Plug in the event values based on the model predictions
    data((u1 < -xdelta1) & (u2 < -xdelta2)) = 00;
    data((u1 > -xdelta1) & (u2 > -xdelta2)) = 11;
    data((u1 < -xdelta1) & (u2 > -xdelta2)) = 01;
    data((u1 > -xdelta1) & (u2 < -xdelta2)) = 10;
elseif all(beta == 0) && (strcmp(selection, 'Complete') == false)
    error('The model is complete under the null. Specify selection to be Complete.')
elseif ~all(beta == 0) && (strcmp(selection, 'Complete') == true)
    error('The model is incomplete given parameter specifications. Use IID, Non IID or LFP in selection')
end
% Data generation under the alternative
if strcmp(selection, 'IID')
    % iid selection mechanism, selects (1,0) out of {(1,0),(0,1)} if an
    % iid Bernoulli r.v vi takes 1, and (0,1) if vi takes 0.
    % Random numbers from standard normal distribution
    % Define the Bernoulli r.v: threshold value to create Bernoulli variables
    alpha = 0.05;
    vi = rand(n,S) <= alpha;
    % placeholder for data
    G_result = NaN(n, S);
    % Plug in the event values based on the model predictions
    G_result((u1 < -xdelta1) & (u2 < -xdelta2)) = 00;
    G_result((u1 > -xdelta1-beta(1)) & (u2 > -xdelta2-beta(2))) = 11;
    G_result((u1 < -xdelta1) & (u2 > -xdelta2)) = 01;
    G_result((-xdelta1 < u1) & (u1 < -xdelta1-beta(1)) & ...
        (u2 > -xdelta2-beta(2))) = 01;
    G_result((u1 > -xdelta1-beta(1)) & (u2 < -xdelta2-beta(2))) = 10;
    G_result((-xdelta1 < u1) & (u1 < -xdelta1-beta(1)) & ...
        (u2 < -xdelta2)) = 10;
    % Use vi to select outcome from multiple equilibria
    G_result((-xdelta1 < u1) & (u1 < -xdelta1-beta(1)) & ...
        (-xdelta2 < u2) & (u2 < -xdelta2-beta(2)) & vi) = 10;
    G_result((-xdelta1 < u1) & (u1 < -xdelta1-beta(1)) & ...
        (-xdelta2 < u2) & (u2 < xdelta2-beta(2)) & (~vi)) = 01;
    data = G_result;
end
if strcmp(selection, 'LFP')
    % generate data from a least favorable distribution
    % Draws an outcome sequence from Q_0 if beta = [0,0] and Q_1 o.w.
    data = NaN(n*S, 1);
    if all(beta == 0)
        % gen Q0 distributions for each obs
        [Q, ~] = get_LFP(beta, xdelta1, xdelta2);
    elseif all(beta < 0)
        % gen Q1 distributions for each obs
        [~, Q] = get_LFP(beta, xdelta1, xdelta2);
    end
    % cumulative probability from the Q dist for each observation.
    cdf_Q = cumsum(Q, 2);
    % Generate uniform distribution (probabilities) matrix
    unif_prob = rand(n*S, 1);
    % Generate iid samples from Q1
    data((unif_prob >= 0) & (unif_prob < cdf_Q(:,1))) = 00;
    data((unif_prob >= cdf_Q(:,1)) & (unif_prob < cdf_Q(:,2))) = 11;
    data((unif_prob >= cdf_Q(:,2)) & (unif_prob < cdf_Q(:,3))) = 10;
    data((unif_prob >= cdf_Q(:,3)) & (unif_prob < cdf_Q(:,4))) = 01;
    data = reshape(data, n, S);
end
end

% Define auxiliary functions.
function [Q0, Q1] = get_LFP(beta, xdelta1, xdelta2)
% Purpose: The function calculates the analytical LFP given beta and Xdelta.
% Inputs:
%   beta:  structural parameters of interest
%   xdelta1: n by S matrix
%   xdelta2: n by S matrix
% Outputs:
%   Q0:     Distribution under the null [q0_00, q0_11, q0_10, q0_01],
%           Dimension: nS by 4.
%   Q1:     Distribution under the alternative [q1_00, q1_11, q1_10, q1_01],
%           Dimension: nS by 4.

dim = size(xdelta1);
% sample size
n = dim(1);
% number of MC
S = dim(2);
% Each of the following four objects are n by S.
Phi1 = normcdf(xdelta1);
Phi2 = normcdf(xdelta2);
Phi_beta1 = normcdf(xdelta1 + beta(1));
Phi_beta2 = normcdf(xdelta2 + beta(2));
% Within a MC, Q0 is identical arocss all subcases.
% The Q0 below is a n*S by 4 matrix
Q0 = [reshape((1-Phi1).*(1-Phi2),n*S,1), reshape(Phi1.*Phi2,n*S,1),...
    reshape(Phi1.*(1-Phi2),n*S,1), reshape((1-Phi1).*Phi2,n*S,1)];
% Depending on the values of beta, we have three cases to
% consider. We have six conditions (two of which are repetitive) to
% consider the regions. Given each local alternative, we can evaluate
% the conditions and thus determine which subregion the given local
% alternative belongs to. First define the 5 variables used in the
% conditions. All 5 variables are n by S.
var1 = Phi1.*(1-Phi_beta2);
z1 = Phi1.*(1-Phi2);
z2 = Phi2.*(1-Phi1) + Phi1 + Phi2 - Phi1.*Phi2 - Phi_beta1.*Phi_beta2;
var2 = reshape((z1.*z2 - Phi2.*(1-Phi1).*z1)./(Phi1 + Phi2 - 2*Phi1.*Phi2), n*S, 1);
var3 = reshape(Phi1.*(1-Phi2), n*S, 1);
var4 = reshape(Phi_beta1.*Phi2, n*S, 1);
var5 = reshape(Phi_beta1.*Phi_beta2, n*S, 1);
var1 = reshape(var1, n*S, 1);
% for condition 1, instead of using ==, we need to take into consideration
% the floating-point arithmetic using a tolerance
tol = eps(1);
% Case1: Mixture over (1,0) and (0,1)
cond1 = (var1 - var2 > 0 | abs(var1 - var2) <= tol) & ...
    (var2 - var3 - var4 + var5 > 0 | abs(var2 - var3 - var4 + var5) <= tol);
% Case2: (0,1) chosen
cond2 = var4 + var3 - var5 - var2 > 0 & var1 - var3 - var4 + var5 > 0;
% Case 3: (1,0) chosen
cond3 = var1 - var2< 0 & var1 - var3 - var4 + var5 > 0;
% stack up the evaluations
evaluation = [cond1, cond2, cond3];
% Now based on the evaluation, calculate Q1 for each obs. Instead of
% evaluating each obs case-by-case (using if else), we first calculate Q1
% in each of the three subcases. This gives us three nS by 4 matrices. Then
% we dot multiply each of them with the corresponding column in the evaluation
% matrix and sum them up. This gives us the nS by 4 matrix of LFP for all obs
% (S Monte Carlos each with sample size n).

% First calculate Q1 in each case
% Q1 in case 1: Mixture over (1, 0) and (0, 1)
Q1_1 = [reshape((1-Phi1).*(1-Phi2), n*S, 1),...
    reshape(Phi_beta1.*Phi_beta2, n*S, 1),...
    reshape((z1.*z2-Phi2.*(1-Phi1).*z1)./(Phi1+Phi2-2*Phi1.*Phi2), n*S, 1),...
    reshape(Phi1+Phi2-Phi1.*Phi2-Phi_beta1.*Phi_beta2 - ...
    (z1.*z2-Phi2.*(1-Phi1).*z1)./(Phi1+Phi2-2*Phi1.*Phi2), n*S, 1)];
% Q1 in case 2: (0, 1) chosen
Q1_2 = [reshape((1-Phi1).*(1-Phi2), n*S, 1),...
    reshape(Phi_beta1.*Phi_beta2, n*S, 1), ...
    reshape((1-Phi2).*Phi1+Phi_beta1.*(Phi2-Phi_beta2), n*S, 1),...
    reshape((1-Phi_beta1).*Phi2, n*S, 1)];
% Q1 in case 3: (1, 0) chosen
Q1_3 = [reshape((1-Phi1).*(1-Phi2), n*S, 1),...
    reshape(Phi_beta1.*Phi_beta2, n*S, 1),...
    reshape(Phi1.*(1-Phi_beta2), n*S, 1),...
    reshape((1-Phi1).*Phi2+Phi_beta2.*(Phi1-Phi_beta1), n*S, 1)];
% Now get Q1 for each obs (nS by 4)
Q1 = evaluation(:,1).*Q1_1 + evaluation(:,2).*Q1_2 + evaluation(:,3).*Q1_3;
end

% 5/28/2020 Shuowen Chen and Hiroaki Kaido
% Calculates the restricted mle
% Under the restriction beta1 = beta2 = 0.
function [delta] = rmle(delta_initial, n, algorithm, data, x1, x2)
% Input:
%   beta:           structural parameters of interest under restriction (0,0)
%   delta_initial:  initial guess of coefficients of covariates (1 by 2)
%   n:              sample size
%   outcome:        the 16 by S matrix that stores the number of
%                   occurrences of each combination of events and covariate
%                   configurations
%   algorithm:      the algorithm for estimation: BFGS, LM-BFGS or BHHH
% Output:
%   delta:          estimates of parameters (1 by 2)

% Compute the outcome for each configuration combination of x and delta
outcome = counting(data, x1, x2);
if strcmp(algorithm, 'BHHH')
    % Acknowledgement:
    % tolerance level and statistics for convergence evaluation are taken from
    % Chapter 8, Train, Kenneth, Discrete Choice Methods with Simulation, 2009,
    % 2nd edition, Cambridge University Press.
    tol = 0.0001;
    stat = 1;
    % stepsize for BHHH algorithm
    lambda = 0.8;
    % beta under the restriction
    beta = [0,0];
    while stat > tol
        % Call compute_z_all to get zdelta, 2 by 16.
        [zdelta, ~] = compute_z_all(beta, [1,1; 1,-1; -1,1; -1,-1], delta_initial');
        zdelta_t = zdelta';
        % Compute sum of score functions (2 by 1)
        sum_l_dot = zdelta * outcome;
        % Compute outer product of score functions
        hess = zeros(2, 2);
        for i = 1:16
            score_sqr_i = outcome(i)*zdelta(:, i)*zdelta_t(i, :);
            hess = hess + score_sqr_i;
        end
        % BHHH update (NOTE delta_update is 2 by 1)
        delta_update = delta_initial' + lambda*(hess+log(n)/sqrt(n)*[1,0;0,1])\sum_l_dot;
        %delta_update = delta_initial' + lambda*hess\sum_l_dot;
        delta_initial = delta_update';
        % Update statistic m to evaluate convergence
        stat = sum_l_dot'*((hess+log(n)/sqrt(n)*[1,0;0,1])\sum_l_dot)/n;
        %stat = sum_l_dot'*(hess\sum_l_dot)/n;
    end
    delta = delta_initial;
elseif strcmp(algorithm, 'BFGS')
    % Objective function and gradient
    f = @(delta_est) obj(data, x1, x2, delta_est, outcome);
    % Setting Options
    % Once gradient is supplied, switch to trust region algorithm
    options = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true);
    %initial = [1;1];
    % Call fminunc
    delta = fminunc(f, delta_initial', options);
elseif strcmp(algorithm, 'LM-BFGS')
    % Now use the LM-BFGS algorithm
    fcn = @(nuisance) loglikelihood(data, x1, x2, nuisance);
    % Gradient
    grad = @(nuisance) score(nuisance, outcome);
    l  = zeros(2,1);    % lower bound
    u  = inf(2,1);         % upper bound
    fun = @(nuisance) fminunc_wrapper(nuisance, fcn, grad);
    % Request very high accuracy for this test:
    opts2 = struct( 'factr', 1e4, 'pgtol', 1e-8, 'm', 10);
    %opts2.printEvery = 100;
    % Run the algorithm:
    [delta, ~, ~] = lbfgsb(fun, l, u, opts2);
end
end


% 5/26/2020 Shuowen Chen and Hiroaki Kaido
% Computes the neg log likelihood function and gradient
% -- Inputs --
%  data:    market outcome (n by 1)
%  x1:      covariate for player 1 (n by 1)
%  x2:      covariate for player 2 (n by 1)
%  delta:   nuisance parameter (for estimation purpose): 2 by 1
%  outcome: frequencies of covariate and outcome combinations (16 by 1)
% -- Outputs --
%  f:       negative log likelihood function
%  g:       score function of f
function [f, g] = obj(data, x1, x2, delta, outcome)
% compute the PMF for each market given covariates and delta (n by 4):
% [q0_00, q0_11, q0_10, q0_01]
q0 = get_LFP_q0(x1, x2, delta');
% construct the negative log likelihood function
temp1 = zeros(size(data));
temp2 = zeros(size(data));
temp3 = zeros(size(data));
temp4 = zeros(size(data));
temp1(data == 0) = 1;
temp2(data == 11) = 1;
temp3(data == 10) = 1;
temp4(data == 1) = 1;
temp = [temp1, temp2, temp3, temp4];
% objective function
f = -sum(sum(temp .* q0));
% gradient required
if nargout > 1
    % Call compute_z_all to get zdelta, 2 by 16.
    [zdelta, ~] = compute_z_all([0, 0], [1,1; 1,-1; -1,1; -1,-1], delta);
    % Compute sum of score functions (2 by 1)
    g = zdelta * outcome;
    % Since we add negative sign to use minimization algorithm, adjust the
    % score accordingly
    g = -g;
end
end

function Q0 = get_LFP_q0(x1, x2, delta)
% Purpose: The function calculates the analytical LFP given and Xdelta.
% Inputs:
%   beta:  structural parameters of interest
%   x1:    covariate of player 1; n by 1 matrix
%   x2:    covariate of player 2; n by 1 matrix
%   delta: nuisance parameter; 1 by 2
% Outputs:
%   Q0:     Distribution under the null [q0_00, q0_11, q0_10, q0_01],
%           Dimension: nS by 4.
xdelta1 = x1*delta(1);
xdelta2 = x2*delta(2);
dim = size(xdelta1);
% sample size
n = dim(1);
% number of MC
S = dim(2);
% Each of the following four objects are n by S.
Phi1 = normcdf(xdelta1);
Phi2 = normcdf(xdelta2);
% Within a MC, Q0 is identical arocss all subcases.
% The Q0 below is a n*S by 4 matrix
Q0 = [reshape((1-Phi1).*(1-Phi2),n*S,1), reshape(Phi1.*Phi2,n*S,1),...
    reshape(Phi1.*(1-Phi2),n*S,1), reshape((1-Phi1).*Phi2,n*S,1)];
end

% for LM-BFGS
function f = loglikelihood(data, x1, x2, delta)
% delta should be 2 by 1
q0 = get_LFP(x1, x2, delta');
% construct the negative log likelihood function
temp1 = zeros(size(data));
temp2 = zeros(size(data));
temp3 = zeros(size(data));
temp4 = zeros(size(data));
temp1(data == 0) = 1;
temp2(data == 11) = 1;
temp3(data == 10) = 1;
temp4(data == 1) = 1;
temp = [temp1, temp2, temp3, temp4];
% objective function
f = -sum(sum(temp .* q0));
end

function g = score(delta, outcome)
% delta should be 2 by 1
% Call compute_z_all to get zdelta, 2 by 16.
[zdelta, ~] = compute_z_all([0, 0], [1,1; 1,-1; -1,1; -1,-1], delta);
% Compute sum of score functions (2 by 1)
g = zdelta * outcome;
% Since we add negative sign to use minimization algorithm, adjust the
% score accordingly
g = -g;
end

% 5/21/2020 Shuowen Chen and Hiroaki Kaido
% Computes the Tn test statistics without regularizations
function [Test, g_delta, varI] = stat_ab(delta_hat, x, occurrence, n)
% Inputs:
%   delta_hat:  first-step estimated delta (1 by 2)
%   x:          possible combination of covariates (k by 2)
%   occurrence: number of occurrences of each combination of covariate and
%               potential outcomes (4k by 1)
%   n:          sample size
% Outputs:
%   Test:       the sup test statistic
%   g_delta:    gn statistic
beta_null = [0, 0];
[~, zbeta] = compute_z_all(beta_null, x, delta_hat);
Cbeta = zbeta * occurrence / sqrt(n);
% Compute the information matrix, which is approximated by the outer
% product of the scores. NOTE: we split the information matrix into 4
% parts, each of which is 2 by 2.

% The following three blocks are each 4 by 16, where 16 stands for possible
% combinations of outcome s and covariate x.
betablock = [zbeta(1,:).*zbeta(1,:); zbeta(1,:).*zbeta(2,:);...
    zbeta(2,:).*zbeta(1,:); zbeta(2,:).*zbeta(2,:)];

% The following three blocks are 4 by 1
betablock = betablock * occurrence / n;

% Reshape to be 2 by 2
I_beta2 = reshape(betablock, 2, 2);
g_delta = Cbeta;
varI = I_beta2;
varI = get_sigmatilde(varI);

gtilde = varI^(-0.5)*g_delta;
Test = gtilde'*gtilde;
Vmin = @(x) quadform(varI,g_delta-x);
[~,Vstar] = fmincon(Vmin,[0;0],eye(2),zeros(2,1));
Test = Test - Vstar;
end

function SigmaTilde = get_sigmatilde(Sigma)
% Inputs:
% Sigma: Covariance matrix
% SigmaTilde: Regularized covariance matrix using Andrews & Barwick's
% method
sh_param = 0.05;
sigma = sqrt(diag(Sigma));
Omega  = Sigma./(sigma*sigma');
D = diag(diag(Sigma));
SigmaTilde = Sigma + max(sh_param-det(Omega),0).*D;
end

function t = quadform(A,g)
gtilde = A^(-0.5)*g;
t = gtilde'*gtilde;
end

% 5/28/2020 Shuowen Chen and Hiroaki Kaido
% The function computes the analytical z_delta and z_beta, an ingredient of
% the scores. Note the outputs are conditional on the value of x.
function [z_delta, z_beta] = compute_z_all(beta, x, delta)
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

% For events (0, 0) and (1, 1) no subcase consideration (all are 4 by 2)
zbeta_00 = zeros(4, 2);
zdelta_00 = -phi.*x./(1-Phi);
zbeta_11 = phi_beta./Phi_beta;
zdelta_11 = phi_beta.*x./Phi_beta;

% For events (1, 0) and (0, 1), for each configuration of Xdelta,
% there are three subcases to consider. We have six conditions
%(two of which are repetitive) to consider the regions.
% Given each local alternative, we can evaluate the conditions and thus
% determine which subregion the given local alternative belongs to.
% First define the 5 variables used in the conditions
var1 = Phi(:,1).*(1-Phi_beta(:,2));
z1 = Phi(:,1).*(1-Phi(:,2));
z2 = Phi(:,2).*(1-Phi(:,1)) + Phi(:,1) + Phi(:,2) - Phi(:,1).*Phi(:,2) - ...
    Phi_beta(:,1).*Phi_beta(:,2);
var2 = (z1.*z2 - Phi(:,2).*(1-Phi(:,1)).*z1)./(Phi(:,1) + Phi(:,2) - ...
    2*Phi(:,1).*Phi(:,2));
var3 = Phi(:,1).*(1-Phi(:,2));
var4 = Phi_beta(:,1).*Phi(:,2);
var5 = Phi_beta(:,1).*Phi_beta(:,2);

% Now for events (1, 0) and (0, 1), define the conditions
dim = size(x);
% pre-allocation
zdelta_1001 = zeros(2, 2*dim(1));
zbeta_1001 = zeros(2, 2*dim(1));
for i = 1:dim(1) % loop over covariate configurations
    % (0,1) chosen
    if var4(i) + var3(i) - var5(i) > var2(i) && var1(i) > var3(i) + var4(i) - var5(i)

        q1_10 = Phi(i,1)*(1-Phi(i,2))+Phi_beta(i,1)*Phi(i,2)-Phi_beta(i,1)*Phi_beta(i,2);

        zbeta1_10 = (Phi(i,2)*phi_beta(i,1)-Phi_beta(i,2)*phi_beta(i,1))/q1_10;

        zbeta2_10 = -Phi_beta(i,1)*phi_beta(i,2)/q1_10;

        zdelta1_10 = x(i,1) * (phi(i,1)*(1-Phi(i,2))+Phi(i,2)*phi_beta(i,1)-...
            Phi_beta(i,2)*phi_beta(i,1))/q1_10;

        zdelta2_10 = x(i,2) * (-phi(i,2)*Phi_beta(i,1)+Phi(i,1)*phi(i,2)-...
            Phi_beta(i,1)*phi_beta(i,2))/q1_10;

        zdelta1_01 = -x(i,1)*phi_beta(i,1)/(1-Phi_beta(i,1));
        zdelta2_01 = x(i,2)*phi(i,2)/Phi(i,2);

        zbeta1_01 = -phi_beta(i,1)/(1-Phi_beta(i,1));

        zbeta2_01 = 0;
        % (1,0) chosen
    elseif var1(i) < var2(i) && var1(i) - var3(i) - var4(i) + var5(i) > 0

        q1_01 = (1-Phi(i,1))*Phi(i,2) + Phi_beta(i,2)*(Phi(i,1)-Phi_beta(i,1));

        zdelta1_10 = x(i,1)*phi(i,1)/Phi(i,1);

        zdelta2_10 = -x(i,2)*phi_beta(i,2)/(1-Phi_beta(i,2));

        zbeta1_10 = 0;

        zbeta2_10 = -phi_beta(i,2)/(1-Phi_beta(i,2));

        zdelta1_01 = (-x(i,1)*Phi(i,2)*phi(i,1) + Phi_beta(i,2)*x(i,1)*(phi(i,1)-...
            phi_beta(i,1)))/q1_01;

        zdelta2_01 = (x(i,2)*(1-Phi(i,1))*phi(i,2) + x(i,2)*(Phi(i,1)-Phi_beta(i,1))...
            *phi_beta(i,2))/q1_01;

        zbeta1_01 = -Phi_beta(i,2)*phi_beta(i,1)/q1_01;

        zbeta2_01 = (Phi(i,1)-Phi_beta(i,1))*phi_beta(i,2)/q1_01;
    else
        % Mixture over (1,0) and (0,1)
        denominator1 = Phi(i,1)+Phi(i,2)-Phi(i,1)*Phi(i,2)-Phi_beta(i,1)*Phi_beta(i,2);

        denominator2 =  Phi(i,1)+Phi(i,2)-2*Phi(i,1)*Phi(i,2);

        zdelta1_10 = phi(i,1)*x(i,1)/Phi(i,1) + (phi(i,1)*x(i,1)-phi(i,1)*x(i,1)*Phi(i,2)-...
            phi_beta(i,1)*Phi_beta(i,2)*x(i,1))/denominator1 - (phi(i,1)*x(i,1)*(1-2*Phi(i,2)))/denominator2;

        zdelta2_10 = -phi(i,2)*x(i,2)/(1-Phi(i,2)) + (phi(i,2)*x(i,2)-phi(i,2)*Phi(i,1)*x(i,2)-...
            phi_beta(i,2)*Phi_beta(i,1)*x(i,2))/denominator1 - (phi(i,2)*x(i,2)*(1-2*Phi(i,1)))/denominator2;

        zbeta1_10 = -phi_beta(i,1)*Phi_beta(i,2)/denominator1;

        zbeta2_10 = -Phi_beta(i,1)*phi_beta(i,2)/denominator1;

        zdelta1_01 = -phi(i,1)*x(i,1)/(1-Phi(i,1)) + (phi(i,1)*x(i,1)-phi(i,1)*Phi(i,2)*x(i,1)-...
            phi_beta(i,1)*Phi_beta(i,2)*x(i,1))/denominator1 - (phi(i,1)*x(i,1)*(1-2*Phi(i,2)))/denominator2;

        zdelta2_01 = phi(i,2)*x(i,2)/Phi(i,2) + (phi(i,2)*x(i,2)-phi(i,2)*Phi(i,1)*x(i,2)-...
            phi_beta(i,2)*Phi_beta(i,1)*x(i,2))/denominator1 - (phi(i,2)*x(i,2)*(1-2*Phi(i,1)))/denominator2;

        zbeta1_01 = -phi_beta(i,1)*Phi_beta(i,2)/denominator1;

        zbeta2_01 = -phi_beta(i,2)*Phi_beta(i,1)/denominator1;
    end
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

% 11/16/2019 Shuowen Chen and Hiroaki Kaido
% Counts the number of occurrences for each of the possible combination of
% events and configurations of covariates.
% Four potential outcomes: (0, 0), (1, 1), (1, 0) and (0, 1);
% Four covariate combinations: (1, 1), (1, -1), (-1, 1) and (-1, -1).
function outcome = counting(data, x1, x2)
% Inputs:
%   data: simulated observations (n by S) by the data_generation function
%   x1:   simulated x1 (n by S) by the data_generation function
%   x2:   simulated x2 (n by S) by the data_generation function
% Outputs:
%   outcome: a 16 by S matrix that lists the number of occurrences for each
%            Monte Carlo simulation sequence
%   note: S denotes number of simulations for each sample sizetemp = size(data);
temp = size(data);
S = temp(2);
% placeholder
outcome = NaN(16, S);
outcome(1, :) = sum(data(:, :)==0 & x1(:, :)==1 & x2(:, :)==1);
outcome(2, :) = sum(data(:, :)==11 & x1(:, :)==1 & x2(:, :)==1);
outcome(3, :) = sum(data(:, :)==10 & x1(:, :)==1 & x2(:, :)==1);
outcome(4, :) = sum(data(:, :)==1 & x1(:, :)==1 & x2(:, :)==1);
outcome(5, :) = sum(data(:, :)==0 & x1(:, :)==1 & x2(:, :)==-1);
outcome(6, :) = sum(data(:, :)==11 & x1(:, :)==1 & x2(:, :)==-1);
outcome(7, :) = sum(data(:, :)==10 & x1(:, :)==1 & x2(:, :)==-1);
outcome(8, :) = sum(data(:, :)==1 & x1(:, :)==1 & x2(:, :)==-1);
outcome(9, :) = sum(data(:, :)==0 & x1(:, :)==-1 & x2(:, :)==1);
outcome(10, :) = sum(data(:, :)==11 & x1(:, :)==-1 & x2(:, :)==1);
outcome(11, :) = sum(data(:, :)==10 & x1(:, :)==-1 & x2(:, :)==1);
outcome(12, :) = sum(data(:, :)==1 & x1(:, :)==-1 & x2(:, :)==1);
outcome(13, :) = sum(data(:, :)==0 & x1(:, :)==-1 & x2(:, :)==-1);
outcome(14, :) = sum(data(:, :)==11 & x1(:, :)==-1 & x2(:, :)==-1);
outcome(15, :) = sum(data(:, :)==10 & x1(:, :)==-1 & x2(:, :)==-1);
outcome(16, :) = sum(data(:, :)==1 & x1(:, :)==-1 & x2(:, :)==-1);
end

% 3/10/2020 Hiroaki Kaido and Shuowen Chen
% The function calculates the critical value of the tn statistic
% Recall that the limiting distribution of tn is the sup of bivariate
% standard normal distribution
function crit = crt_ab(nominal,varI)
rng(pi); % set seed for random number generations
J = 2;    % # of components in the score
R = 5000; % # of draws
z = randn(J, R); % J by R vector of draws from standard normal
[LI,DI] = ldl(varI);
ztilde=LI*DI^(0.5)*z;
%     DI(DI<tol) = tol; % truncate negative values

first = sum(z.^2); % limiting distribution
second = zeros(1,R);
for r=1:R
    Vmin = @(x) quadform(varI,ztilde(:,r)-x);
    options = optimoptions('fmincon','Display','off');
    [~,Vstar] = fmincon(Vmin,[0;0],eye(2),zeros(2,1),[],[],[],[],[],options);
    second(r) = Vstar;
end
sim_stat = first-second;
crit = quantile(sim_stat, (1-nominal));
end

% 5/20/2020
% Shuowen Chen and Hiroaki Kaido
% A function that calculates the critical values of the simulated Tn
% statistics
function [real, l] = actualsize(test_matrix, N, cv)
% Outputs:
% real: the actual size
% l: (auxiliary output) length of sequence (some simulations end up producing NA values, 
%    so compute l to have a sense how severe the issue is)
real = zeros(length(N), 1); % actual size placeholder
l = zeros(length(N), 1);
for i = 1:length(N)
    test_seq = test_matrix(i, :);
    % this is to delete ill-behaved test statistics (due to var-cov matrix)
    % test_seq = test_seq(~isnan(test_seq) & imag(test_seq) == 0);
    test_seq = test_seq(~isnan(test_seq));
    % compute the actual size
    real(i) = sum(test_seq > cv)/length(test_seq);
    % compute the length of sequence
    l(i) = length(test_seq);
end
end
