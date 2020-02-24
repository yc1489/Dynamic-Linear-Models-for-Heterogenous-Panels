function [omega,fval,exitflag,output,lambda,hessian]=MAIV_opt(T1,K,sigma_etau_i,Sigma_u_i_i,Sigma_eta_i_i, hat_U_i_4,Gamma,j4,omega0,lq,uq)
 
% maximum likelihood routine. see the program transformed_ml.m
 

[omega,fval,exitflag,output,lambda,~,hessian] = fmincon(@likelihood,omega0,[],[],[ones(1,j4)],1,lq,uq,[]); 
 
    function [likelihood]=likelihood(omega)
 dim_omega = size(omega0,1);
    omega  = omega(1:dim_omega,1);
 
    
    
 sigma_mse=(sigma_etau_i^2)*((K'*omega)^2)/T1+Sigma_u_i_i*((omega'*hat_U_i_4*omega- Sigma_eta_i_i*(j4-2*K'*omega+omega'*Gamma*omega))/T1);
  
    
    
 likelihood = sigma_mse;
    
    
    
  
end
end
