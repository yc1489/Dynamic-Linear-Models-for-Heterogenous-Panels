function [omega_3, fval_3, exitflag_3, output_3, lambda_3, hessian_3]=MAIV_opt_3(T1,K_3,sigma_etau_i,Sigma_u_i_i,Sigma_eta_i_i, hat_U_i,Gamma,j3,omega0,lq_3,uq_3)
 
% maximum likelihood routine. see the program transformed_ml.m
 

[omega_3, fval_3, exitflag_3, output_3, lambda_3,~,hessian_3] = fmincon(@likelihood,omega0,[],[],[ones(1,3)],1,lq_3,uq_3,[]); 
 
    function [likelihood]=likelihood(omega)
 dim_omega = size(omega0,1);
    omega  = omega(1:dim_omega,1);
 
    
    
 sigma_mse=(sigma_etau_i^2)*((K_3'*omega)^2)/T1+Sigma_u_i_i*((omega'*hat_U_i*omega- Sigma_eta_i_i*(2-2*K_3'*omega+omega'*Gamma*omega))/T1);
  
    
    
 likelihood = sigma_mse;
    
    
    
  
end
end
