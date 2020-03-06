function [omega_5, fval_5, exitflag_5, output_5, lambda_5, hessian_5]=MAIV_opt_nores_5(T1,K_5,sigma_etau_i,Sigma_u_i_i,Sigma_eta_i_i, hat_U_i_5,Gamma_5, Sigmau_5,sigmaue_5,A1,A2, H2,etai5,d,j5,omega0,lq_5,uq_5)
 
% maximum likelihood routine. see the program transformed_ml.m
 

[omega_5, fval_5, exitflag_5, output_5, lambda_5,~,hessian_5] = fmincon(@likelihood,omega0,[],[],[ones(1,j5)],1,lq_5,uq_5,[]); 
 
    function [likelihood]=likelihood(omega)
 dim_omega = size(omega0,1);
    omega  = omega(1:dim_omega,1);
 
    
 
 B_N=2*(Sigma_u_i_i*Sigmau_5+d*real(sigmaue_5*sigmaue_5')+A1/T1+A2/T1);
    
 sigma_mse=(sigma_etau_i^2)*((K_5'*omega)^2)/T1+(Sigma_u_i_i*Sigma_u_i_i+sigma_etau_i^2)*(omega'*Gamma_5*omega/T1)-(K_5'*omega/T1)*etai5'* pinv(H2)*B_N*pinv(H2)*etai5+Sigma_u_i_i*((omega'*hat_U_i_5*omega- Sigma_eta_i_i*(j5-2*K_5'*omega+omega'*Gamma_5*omega))/T1);
  
    
    
 likelihood = sigma_mse;
    
    
    
  
end
end
