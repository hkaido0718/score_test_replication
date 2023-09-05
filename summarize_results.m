
DGP = 'LFP  '; 
S=1999;
cn_BCS = zeros(31,S);
Tn_BCS = zeros(31,S);
n = 7500;

% Load results on the BCS test
for part = 1:8
    pbegin = (part-1)*4+1;
    pend = part*4;
    if part == 8
        pend = 31;
    end
    filename = ['../Results/Matfiles/BCS_power_DGP' DGP '_n7500_S1999part' num2str(part) '.mat'];
    load(filename,'*MRsim')
    Tn_BCS(pbegin:pend,:) = Tn_MRsim(pbegin:pend,:);
    cn_BCS(pbegin:pend,:) = cn_MRsim(pbegin:pend,:);
end

% Power of BCS test
power_BCS = sum(Tn_BCS>cn_BCS,2)/S;

% Power of Score test
filename = ['../Results/Matfiles/test_power_DGP' DGP '_n7500_S1999.mat'];
load(filename)
power_func = sum(test>cv,2)/S;
beta_alt_vec = -(eps:0.5:15)'./sqrt(n);


plot(beta_alt_vec,power_func,beta_alt_vec,power_BCS)
legend('Score Test','BCS Test')
xlabel('$\beta^{(j)},j=1,2$','Interpreter','latex')
fig_filename = ['../Figures/power_' DGP '.pdf'];
saveas(gcf,fig_filename)


