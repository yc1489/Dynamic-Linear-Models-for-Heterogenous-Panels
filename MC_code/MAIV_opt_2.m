function [omega_2, fval_2, exitflag_2, output_2, lambda_2, hessian_2]=MAIV_opt_2(T1,K_2,sigma_etau_i,Sigma_u_i_i,Sigma_eta_i_i, hat_U_i,Gamma,j2,omega0,lq_2,uq_2)
 
% maximum likelihood routine. see the program transformed_ml.m
 

[omega_2, fval_2, exitflag_2, output_2, lambda_2,~,hessian_2] = fmincon(@likelihood,omega0,[],[],[ones(1,2)],1,lq_2,uq_2,[]); 
 
    function [likelihood]=likelihood(omega)
 dim_omega = size(omega0,1);
    omega  = omega(1:dim_omega,1);
 
    
    
 sigma_mse=(sigma_etau_i^2)*((K_2'*omega)^2)/T1+Sigma_u_i_i*((omega'*hat_U_i*omega- Sigma_eta_i_i*(2-2*K_2'*omega+omega'*Gamma*omega))/T1);
  
    
    
 likelihood = sigma_mse;
    
    
    
  
end
end
