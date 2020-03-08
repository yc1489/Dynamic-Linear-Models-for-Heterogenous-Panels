function [omega_7, fval_7, exitflag_7, output_7, lambda_7, hessian_7]=MAIV_opt_7(T1,K_7,sigma_etau_i,Sigma_u_i_i,Sigma_eta_i_i, hat_U_i_7,Gamma,j7,omega0,lq_7,uq_7)
 
% maximum likelihood routine. see the program transformed_ml.m
[omega_7, fval_7, exitflag_7, output_7, lambda_7,~,hessian_7] = fmincon(@likelihood,omega0,[],[],[ones(1,j7)],1,lq_7,uq_7,[]); 
 
    function [likelihood]=likelihood(omega)
 dim_omega = size(omega0,1);
    omega  = omega(1:dim_omega,1);
  
 sigma_mse=(sigma_etau_i^2)*((K_7'*omega)^2)/T1+Sigma_u_i_i*((omega'*hat_U_i_7*omega- Sigma_eta_i_i*(j7-2*K_7'*omega+omega'*Gamma*omega))/T1);
  
    
    
 likelihood = sigma_mse;
    
  
end
end