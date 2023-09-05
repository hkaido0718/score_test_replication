function [] = Simulation_BCS(ncores,part)
% 5/10/2021 Shuowen Chen and Hiroaki Kaido
% Monte Carlo for comparison with Bugni, Canay and Shi (2017)
%% 0. Parameters
% seed for replication
rng(123);
parpool(ncores)
% In each sample we will draw X from this vector uniformly with replacement
covariate = [1, -1];
% true value of nuisance parameters
delta = [2, 1.5];
% number of simulations
S = 1999;
n = 7500;
% # of bootstrap draws
B = 500;
% shocks \xi in eq (2.8) pp. 5
W2_AA = norminv(rand(B,n));
% constant in the paper;
C_list = [1];
% GMS Thresholding parameter, preferred choice Eq. 4.4 in Andrews and Soares (2010).
kappa_list = C_list*sqrt(log(n));
% %% 1. DGP under the null
% [data_com, x1_com, x2_com] = data_generation([0, 0], covariate, delta, n, S, 'Complete');
%% 2. Implement Test MR in BCS (algorithm 2.1)
options = optimset('Display','off','Algorithm','interior-point'); % from BCS
% significance level
DGP = 'IID'; %'IID' or 'LFP'
alpha = 0.1;
% Placeholders
minQn_DR = NaN(size(kappa_list,1), B);
minQn_PR = NaN(size(kappa_list,1), B);
minQn_MR = NaN(size(kappa_list,1), B);
cn_MR  = NaN(1,size(kappa_list,1));

h_alt  = -(eps:0.5:15)';
h_alt2 = -(eps:0.5:15)';
K = length(h_alt);
Tn_MRsim = NaN(K,S);
cn_MRsim = NaN(K,S);
pbegin = (part-1)*4+1;
pend = part*4;
if part == 8
    pend = 31;
end
% Implementation starts here
for k = pbegin:pend
    beta_alt = h_alt(k,:)/sqrt(n);
    beta_alt2 = h_alt2(k,:)/sqrt(n);
    % Call data_generation function to generate data
    [data_com, x1_com, x2_com] = data_generation([beta_alt,beta_alt2], covariate, ...
        delta, n, S, DGP);
    for s = 1:S % loop over simulations
        disp(s)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Compute the test statstic
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Pick a large value as initial function value for the minimization
        min_value = 10^10; % from BCS
        % Multiple starting points using Latin Hypercube
        lb = delta - 1;
        ub = delta + 1;
        M = 49; % number of starting points using lhsscale
        start = [delta; lhsscale(M, 2, ub, lb)]; % 1st starting point is oracle
        % placeholder for minimization results
        min_outcomes = zeros(M+1, 3);
        
        % loop over starting points for minimization
        for initial = 1:(M + 1)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % (Line 20 in BCS Algorithm 2.1)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Instead of set minimizer we have point estimator of nuisance par
            try
                [delta_est,Qn_aux,bandera] =  fminunc(@(x) Qn_function_X(x, [0, 0], data_com(:,s), x1_com(:,s), x2_com(:,s), 'agg'), start(initial, :), options);
                % check whether minimization is successful and reduced value
                if Qn_aux  < min_value && bandera >= 1
                    Qn_minimizer = delta_est;
                    min_value = Qn_aux;
                end
            catch
                min_value = NaN;
                bandera = 0;
            end
            
            % if minimization is successful, collect minimizer and its value;
            if bandera >= 1
                min_outcomes(initial,:) = [min_value, Qn_minimizer];
            end
        end
        % return the test statstic
        minQn = min_value;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Test MR
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        kappa_type =1;
        parfor b = 1:B % loop over bootstrap draws
            % (1) DR test (equivalent to plugging in nuisance estimator and null value)
            % compute simulated DR criterion function
            minQn_DR(kappa_type, b) = Qn_MR_function_X(delta_est, [0,0], data_com(:,s), x1_com(:,s), x2_com(:,s), kappa_list(kappa_type), W2_AA(b,:), 1, 'agg');
            
            % (2) PR test
            [delta_aux,min_value_PR] =  fmincon(@(x) Qn_MR_function_X(x, [0,0], data_com(:,s), x1_com(:,s), x2_com(:,s), kappa_list(kappa_type), W2_AA(b,:), 2, 'agg'),...
                delta,[],[],[],[],...
                [1,1],[2.5,5],[],options);
            % compute simulated PR criterion function
            minQn_PR(kappa_type, b) = min_value_PR ;
            
            % (3) MR test
            minQn_MR(kappa_type, b) = min(minQn_DR(kappa_type, b), minQn_PR(kappa_type, b));
        end
        % Compute critical values
        cn_MR(kappa_type) = quantile(minQn_MR(kappa_type,:), 1-alpha);
        cn_MRsim(k,s) = cn_MR;
        Tn_MRsim(k,s) = minQn;
        %     accept_testMR = repmat(minQn,1,size(kappa_list,1)) <= cn_MR;
    end
end
filename = ['../Results/Matfiles/BCS_power_DGP' DGP '_n' num2str(n) '_S' num2str(S) 'part' num2str(part) '.mat'];
save(filename)

end
%%

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


% 6/17/2020 Shuowen Chen and Hiroaki Kaido
% Implements Latin Hypercube Sampling and rescale to produce initial 
% guesses for restricted MLE
function output = lhsscale(nsamples, nvars, ux, lx)
% Inputs:
%   nsamples: the number of points for each parameter
%   nvars:    the dimension of the parameter
%   ux:       the upper bound of the parameter space (1 by nvars)
%   lx:       the lower bounds of the parameter space (1 by nvars)
rng(123)
% draw points on a cube (Latin hypercube sampling)
% the nsample points will be from (0,1/n), (1/n, 2/n),...,(1-1/n,1), where
% n is shorthand notation of nsamples. Note the intervals can be randomly
% permutated
temp = lhsdesign(nsamples, nvars);  
% rescale to draw points on parameter space	
output = zeros(nsamples, nvars); 
for i = 1:nvars
    output(:, i) = repmat((ux(i)-lx(i)), nsamples, 1).*temp(:,i) + ...
        repmat(lx(i), nsamples, 1);
end
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   5/1/2021 Shuowen Chen and Hiroaki Kaido
%   This is taken from code by Bugni, Canay and Shi (QE) with some changes
%   (1) We have two type of moment conditions
%   (2) p, the number of moment inequalities, are now in the fcn body
%   (3) we use dataOp.m to produce studendized moments
%   (4) instead of naming theta_to_min and theta_H0, we directly input delta
%       and beta

%   Defines sample criterion function according to MMM in Eq. (4.2) of BCS
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function value = Qn_function_X(delta, beta, data, X1, X2, mtype)
%{
 - Inputs - 
   delta: nuisance parameter (to be estimated)
   beta:  strategic parameter 
   data:  data
   X1:    covariate for player 1
   X2:    covariate for player 2
   mtype: whether use aggregate or disaggregate moments

 - Outputs - 
   value: function value 
%}

% determines sample size;
n = size(data,1);

% the number of moment inequalities (which should appear first);
if strcmp(mtype, 'disagg')
    p = 8;
elseif strcmp(mtype, 'agg')
    p = 2;
end

% studentizes and averages the data (4 by S)
[mbar_std, ~, ~] = dataOp(delta, beta, data, 1, X1, X2, mtype); % No use for kappa, set to one.

% computes the sample criterion function;
value = S_function(sqrt(n)*mbar_std', p);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   5/1/2021 Shuowen Chen and Hiroaki Kaido
%   This is taken from code by Bugni, Canay and Shi (QE) with some changes
%   (1) We have two type of moment conditions
%   (2) p, the number of moment inequalities, are now in the fcn body
%   (3) k, total number of moments, are now in the fcn body
%   (3) we use dataOp.m to produce studendized moments
%   (4) instead of naming theta_to_min and theta_H0, we directly input delta
%       and beta
%   Defines the sample criterion function for DR or PR inference
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function value = Qn_MR_function_X(delta, beta, data, X1, X2, kappa, W2_AA, MR_type, moment_type)
%{
 - Inputs - 
   delta:       nuisance parameter (to be estimated)
   beta:        strategic parameter 
   data:        data
   X1:          covariate for player 1
   X2:          covariate for player 2
   kappa:       tuning parameter for GMS
   W2_AA:       a vector of random variables used to implement the (multiplier) bootstrap.
   MR_type:     the type of resampling, i.e., DR or PR
   moment_type: whether aggregate or disaggregate moments

 - Outputs - 
   value:       function value 
%}

% decide if the number of moments are
n = size(data,1); % sample size

% the number of moment inequalities (which should appear first) and total moments;
if strcmp(moment_type, 'disagg')
    p = 8;
    k = 16;
elseif strcmp(moment_type, 'agg')
    p = 2;
    k = 4;
end

[~, mData, xi] = dataOp(delta, beta, data, kappa, X1, X2, moment_type);

if MR_type == 1 % DR method;
    value = S_function(W2_AA*zscore(mData, 1)/sqrt(n) + repmat(phi_function(xi',p,k-p),size(W2_AA,1),1), p);
elseif MR_type == 2 % PR method;
    value = S_function(W2_AA*zscore(mData, 1)/sqrt(n) + repmat(xi',size(W2_AA,1),1), p);
end
end


% 5/7/2021 Shuowen Chen and Hiroaki Kaido
% Acknowledgement: based on code by Bugni, Canay and Shi (2017, QE)
% Data manipulation to produce sample moment conditions for the game in
% Chen and Kaido (2021)

% We consider two types of moments: 
% "agg":    aggregates the covariates across markets 4 moment (in)equalities
% "disagg": conditional on the value of covariates, has 16 ( because the 
% support of x is {(-1,-1), (-1,1), (1,-1), (1,1)} )

function [mbar_std, mData, xi] = dataOp(delta, beta, data, kappa, X1, X2, mtype)
%{
 -Inputs-
 delta:    nuisance parameter
 beta:     the parameter of interest
 data:     dataset (n by S)
 kappa:    tuning parameter
 X1:       covariate for player 1 (n by S)
 X2:       covariate for player 2 (n by S)
 mtype:    whether we use aggregate or disaggregate moments

 -Outputs- 
 Let k denote the total number of moments
 mbar_std: standardized sample moment inequalities and equalities (k by S)
 mData:    data for sample moments (n by k by S). It is used for
           constructing objective function for DR and PR tests.  
 xi:       slackness measure in GMS (k by S)
%}

% sample size (number of markets);
n = size(data, 1);
% number of simulations
S = 1;
% S = size(data, 2);

% Model predictions (n by S)
% (1) equalities
modelP00 = normcdf(-X1*delta(1)) .* normcdf(-X2*delta(2));
modelP11 = normcdf(X1*delta(1)+beta(1)) .* normcdf(X2*delta(2)+beta(2));
% (2) inequalities
% region of multiplicity
mul = (normcdf(-X1*delta(1)-beta(1)) - normcdf(-X1*delta(1))) .* (normcdf(-X2*delta(2)-beta(2)) - normcdf(-X2*delta(2)));
modelP10_ub = normcdf(X1*delta(1)) .* normcdf(-X2*delta(2)-beta(2));
modelP10_lb = modelP10_ub - mul;

if strcmp(mtype, 'disagg')
    % Construct aggregate sample moments
    dataP00x1 = zeros(n, S); % placeholders for outcome and covariate combo
    dataP00x2 = zeros(n, S);
    dataP00x3 = zeros(n, S);
    dataP00x4 = zeros(n, S); 
    dataP11x1 = zeros(n, S);
    dataP11x2 = zeros(n, S);
    dataP11x3 = zeros(n, S);
    dataP11x4 = zeros(n, S);
    % outcome (1,0) will be used twice for the inequalities
    dataP10x1 = zeros(n, S); 
    dataP10x2 = zeros(n, S);
    dataP10x3 = zeros(n, S);
    dataP10x4 = zeros(n, S);
    for s = 1:S % loop over each simulation
        % From the data, count frequencies of possible outcomes (0, 0), (1, 1)
        % and (1, 0)
        dataP00x1(data(:, s) == 0 & X1(:, s) == 1 & X2(:, s) == 1, s) = 1;
        dataP00x2(data(:, s) == 0 & X1(:, s) == 1 & X2(:, s) == -1, s) = 1;
        dataP00x3(data(:, s) == 0 & X1(:, s) == -1 & X2(:, s) == 1, s) = 1;
        dataP00x4(data(:, s) == 0 & X1(:, s) == -1 & X2(:, s) == -1, s) = 1;
        dataP11x1(data(:, s) == 11 & X1(:, s) == 1 & X2(:, s) == 1, s) = 1;
        dataP11x2(data(:, s) == 11 & X1(:, s) == 1 & X2(:, s) == -1, s) = 1;
        dataP11x3(data(:, s) == 11 & X1(:, s) == -1 & X2(:, s) == 1, s) = 1;
        dataP11x4(data(:, s) == 11 & X1(:, s) == -1 & X2(:, s) == -1, s) = 1;
        dataP10x1(data(:, s) == 10 & X1(:, s) == 1 & X2(:, s) == 1, s) = 1;
        dataP10x2(data(:, s) == 10 & X1(:, s) == 1 & X2(:, s) == -1, s) = 1;
        dataP10x3(data(:, s) == 10 & X1(:, s) == -1 & X2(:, s) == 1, s) = 1;
        dataP10x4(data(:, s) == 10 & X1(:, s) == -1 & X2(:, s) == -1, s) = 1;
    end
    
    % since the covariates come from uniform discrete distribution, each
    % configuration of covariate has probability 0.25
    pX = 0.25;
    
    % Define moment (in)equalities data (n by S).
    mData_1qx1 = dataP00x1 - modelP00*pX;
    mData_1qx2 = dataP00x2 - modelP00*pX;
    mData_1qx3 = dataP00x3 - modelP00*pX;
    mData_1qx4 = dataP00x4 - modelP00*pX;
    mData_2qx1 = dataP11x1 - modelP11*pX;
    mData_2qx2 = dataP11x2 - modelP11*pX;
    mData_2qx3 = dataP11x3 - modelP11*pX;
    mData_2qx4 = dataP11x4 - modelP11*pX;
    mData_3qx1 = dataP10x1 - modelP10_lb*pX;
    mData_3qx2 = dataP10x2 - modelP10_lb*pX;
    mData_3qx3 = dataP10x3 - modelP10_lb*pX;
    mData_3qx4 = dataP10x4 - modelP10_lb*pX;
    mData_4qx1 = modelP10_ub*pX - dataP10x1;
    mData_4qx2 = modelP10_ub*pX - dataP10x2;
    mData_4qx3 = modelP10_ub*pX - dataP10x3;
    mData_4qx4 = modelP10_ub*pX - dataP10x4;
    % Stack up moments for each simulation Note: inequalities should appear first
    mData = zeros(n, 16, S);
    mbar_std = zeros(16, S);
    xi = zeros(16, S);
    epsilon = 0.000001; % introduces this parameter to avoid division by zero
    for s = 1:S
        mData(:, :, s) = [mData_3qx1(:, s), mData_3qx2(:, s), mData_3qx3(:, s), mData_3qx4(:, s),...
            mData_4qx1(:, s), mData_4qx2(:, s), mData_4qx3(:, s), mData_4qx4(:, s),...
            mData_1qx1(:, s), mData_1qx2(:, s), mData_1qx3(:, s), mData_1qx4(:, s),...
            mData_2qx1(:, s), mData_2qx2(:, s), mData_2qx3(:, s), mData_2qx4(:, s)];
        % compute studentized sample averages of mData
        mbar_std(:, s)  = mean(mData(:, :, s))./(std(mData(:, :, s)) + epsilon);
        % Additional parameter needed in DR, PR, and MR inference
        xi(:, s) = (1/kappa)*sqrt(n)*mbar_std(:, s); % Slackness measure in GMS
    end
elseif strcmp(mtype, 'agg')
    % Construct aggregate sample moments
    dataP00 = zeros(n, S); % placeholders
    dataP11 = zeros(n, S);
    dataP10 = zeros(n, S);
    for s = 1:S % loop over each simulation
        % From the data, count possible outcomes 
        % (0, 0), (1, 1) and (1, 0)
        dataP00(data(:, s) == 0, s) = 1;
        dataP11(data(:, s) == 11, s) = 1;
        dataP10(data(:, s) == 10, s) = 1;
    end
    % Define moment (in)equalities data (n by S).
    mData_1q = dataP00 - modelP00;
    mData_2q = dataP11 - modelP11;
    mData_3q = dataP10 - modelP10_lb;
    mData_4q = modelP10_ub - dataP10;
    % Stack up moments for each simulation Note: inequalities should appear first
    mData = zeros(n, 4, S); % placeholders
    mbar_std = zeros(4, S);
    xi = zeros(4, S);
    epsilon = 0.000001; % introduces this parameter to avoid division by zero
    for s = 1:S
        mData(:, :, s) = [mData_3q(:, s), mData_4q(:, s), mData_1q(:, s), mData_2q(:, s)];
        % compute studentized sample averages of mData
        mbar_std(:, s)  = mean(mData(:, :, s))./(std(mData(:, :, s)) + epsilon);
        % Additional parameter needed in DR, PR, and MR inference
        xi(:, s) = (1/kappa)*sqrt(n)*mbar_std(:, s); % Slackness measure in GMS
    end
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Defines the S function according to MMM in Eq. (2.6);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Qn,chi] = S_function(m_std, p)
% m_std denotes the studentized sample moment conditions;
% p denotes the number of moment inequalities (which should appear first);

chi = 2; % degree of homogeneity of criterion function in MMM

% take negative part of inequalities;
m_std(:, 1:p) = min(m_std(:, 1:p), zeros(size(m_std, 1), p));

% sample criterion function;
Qn = sum(abs(m_std).^chi,2);

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   
%   Defines the GMS penalization function for DR inference as in Eq. (2.11)
%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function value = phi_function(xi,p,v)
% xi is the xi(1) vector in AS (2010) Eq. 4.5;
% p is the number of moment inequalities;
% v is the number of moment equalities;

% My definition of "infinity" (satisfies Inf*0 = 0 (as desired), while the built in "inf" has inf*0 = NaN)
Inf = 10^10;

% define GMS penalization;
value      = zeros(1, p+v); % zero for equalities and "close" inequalities;
value(1:p) = (xi(:, 1:p)>1).*Inf; % infinity for "sufficiently violated" inequalities;
end