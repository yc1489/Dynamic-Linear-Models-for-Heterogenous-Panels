function [omega_4, fval_4, exitflag_4, output_4, lambda_4, hessian_4]=MAIV_opt_nores_4(T1,K_4,sigma_etau_i,Sigma_u_i_i,Sigma_eta_i_i, hat_U_i_4,Gamma_4, Sigmau_4,sigmaue_4,A1,A2, H2,etai4,d,j4,omega0,lq_4,uq_4)
 
% maximum likelihood routine. see the program transformed_ml.m
 

[omega_4, fval_4, exitflag_4, output_4, lambda_4,~,hessian_4] = fmincon(@likelihood,omega0,[],[],[ones(1,j4)],1,lq_4,uq_4,[]); 
 
    function [likelihood]=likelihood(omega)
 dim_omega = size(omega0,1);
    omega  = omega(1:dim_omega,1);
 
    
 
 B_N=2*(Sigma_u_i_i*Sigmau_4+d*real(sigmaue_4*sigmaue_4')+A1/T1+A2/T1);
    
 sigma_mse=(sigma_etau_i^2)*((K_4'*omega)^2)/T1+(Sigma_u_i_i*Sigma_u_i_i+sigma_etau_i^2)*(omega'*Gamma_4*omega/T1)-(K_4'*omega/T1)*etai4'* pinv(H2)*B_N*pinv(H2)*etai4+Sigma_u_i_i*((omega'*hat_U_i_4*omega- Sigma_eta_i_i*(j4-2*K_4'*omega+omega'*Gamma_4*omega))/T1);
  
    
    
 likelihood = sigma_mse;
    
    
    
  
end
end
