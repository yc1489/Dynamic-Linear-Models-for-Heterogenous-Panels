function [omega_5, fval_5, exitflag_5, output_5, lambda_5, hessian_5]=MAIV_opt_5(T1,K_5,sigma_etau_i,Sigma_u_i_i,Sigma_eta_i_i, hat_U_i_5,Gamma,j,omega0,lq_5,uq_5)
 
% maximum likelihood routine. see the program transformed_ml.m
 

[omega_5, fval_5, exitflag_5, output_5, lambda_5,~,hessian_5] = fmincon(@likelihood,omega0,[],[],[ones(1,j)],1,lq_5,uq_5,[]); 
 
    function [likelihood]=likelihood(omega)
 dim_omega = size(omega0,1);
    omega  = omega(1:dim_omega,1);
 
    
    
 sigma_mse=(sigma_etau_i^2)*((K_5'*omega)^2)/T1+Sigma_u_i_i*((omega'*hat_U_i_5*omega- Sigma_eta_i_i*(j-2*K_5'*omega+omega'*Gamma*omega))/T1);
  
    
    
 likelihood = sigma_mse;
    
    
    
  
end
end
