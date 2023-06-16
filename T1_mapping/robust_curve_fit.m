% load tvec for generating data

load('MAVI102_20151026_pre1.mat');
x = sort(tvec(2,:));

% for reproducing experiments

rng('default');
s = rng;

% define true parameters
f = waitbar(0, 'Starting');
n_trials = 65536;


for k = [800,1000,1200,1400,1600]
    
    for j = [20,30,40]
    calculated_t1 = zeros(1,n_trials);
    calculated_sd = zeros(1,n_trials);
    true_C = 300;
    true_k = 2;
    true_T1 = k;
    SNR = j;
    
    
    % give true regression lines
    
    true_regression_line = true_C.*(1 - true_k.*exp(-1.*x.*(true_k - 1)./true_T1));
    scale = true_C/SNR;
    % add for loop here
        for i = 1:n_trials
        
            y_perturb = true_regression_line + scale * randn(size(x));
            
            % now work on fitting
            
            [fitresult,gof,output] = createFit(x,y_perturb);
            coef = coeffvalues(fitresult);
            calculated_t1(i) = coef(3);
            mad = sort(abs(output.residuals));
            mad = mad(3:end);
            mad = median(mad)/0.6745;
            sd = sd_calculation(coef, x, mad);
            calculated_sd(i) = sd;
            waitbar(i/n_trials, f, sprintf('Progress: %d %%', floor(i/n_trials*100)));
            
        end
        save(['LS_',num2str(j),'_',num2str(k),'_',num2str(i),'_no_robust'],'calculated_sd','calculated_t1')
    end
end
plot(x,true_regression_line);

fit_line = coef(1).*(1 - coef(2).*exp(-1.*x.*(coef(2) - 1)./coef(3)));
hold on
plot(x,fit_line);
legend('true','fit')