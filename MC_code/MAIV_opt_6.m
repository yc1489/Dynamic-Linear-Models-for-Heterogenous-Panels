function [omega_6, fval_6, exitflag_6, output_6, lambda_6, hessian_6]=MAIV_opt_6(T1,K_6,sigma_etau_i,Sigma_u_i_i,Sigma_eta_i_i, hat_U_i_6,Gamma,j6,omega0,lq_6,uq_6)
 
% maximum likelihood routine. see the program transformed_ml.m
 

[omega_6, fval_6, exitflag_6, output_6, lambda_6,~,hessian_6] = fmincon(@likelihood,omega0,[],[],[ones(1,j6)],1,lq_6,uq_6,[]); 
 
    function [likelihood]=likelihood(omega)
 dim_omega = size(omega0,1);
    omega  = omega(1:dim_omega,1);
 
    
    
 sigma_mse=(sigma_etau_i^2)*((K_6'*omega)^2)/T1+Sigma_u_i_i*((omega'*hat_U_i_6*omega- Sigma_eta_i_i*(j6-2*K_6'*omega+omega'*Gamma*omega))/T1);
  
    
    
 likelihood = sigma_mse;
    
    
    
  
end
end
