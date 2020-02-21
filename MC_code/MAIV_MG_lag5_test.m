rep = 50;  
list_T = [25 50 100 200]; 
list_N = [25 50 100 200];  
list_phi= [0.5 ]; 
b1=3;
rho_b=0.4;
rho_b1=0.5;
tao=0.5;
v_theta=-0.2;
b=[b1];
k=1;   % number of regressors
j=4; % lag of regressor
Dis_T=50;
%%%%%%%%%%%%%%%%%%%%%%%%
ini_bias_mean_phi_1=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
ini_bias_mean_beta1_1=zeros(size(list_T,2), size(list_N,2), size(list_phi,2)); 

ini_std_phi_1=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));  
ini_std_beta1_1=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));  

ini_rmse_phi_1=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
ini_rmse_beta1_1=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));

ini_mae_phi_1=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
ini_mae_beta1_1=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));


%%%%%%%%%%%%%%%%%%%%%%%%%%%
ini_bias_mean_phi_2=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
ini_bias_mean_beta1_2=zeros(size(list_T,2), size(list_N,2), size(list_phi,2)); 

ini_std_phi_2=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));  
ini_std_beta1_2=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));  

ini_rmse_phi_2=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
ini_rmse_beta1_2=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));

ini_mae_phi_2=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
ini_mae_beta1_2=zeros(size(list_T,2), size(list_N,2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%
ini_bias_mean_phi_3=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
ini_bias_mean_beta1_3=zeros(size(list_T,2), size(list_N,2), size(list_phi,2)); 

ini_std_phi_3=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));  
ini_std_beta1_3=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));  

ini_rmse_phi_3=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
ini_rmse_beta1_3=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));

ini_mae_phi_3=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
ini_mae_beta1_3=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%
ini_bias_mean_phi_4=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
ini_bias_mean_beta1_4=zeros(size(list_T,2), size(list_N,2), size(list_phi,2)); 

ini_std_phi_4=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));  
ini_std_beta1_4=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));  

ini_rmse_phi_4=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
ini_rmse_beta1_4=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));

ini_mae_phi_4=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
ini_mae_beta1_4=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

opt_bias_mean_phi_2=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
opt_bias_mean_beta1_2=zeros(size(list_T,2), size(list_N,2), size(list_phi,2)); 

opt_std_phi_2=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));  
opt_std_beta1_2=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));


opt_rmse_phi_2=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
opt_rmse_beta1_2=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));

opt_mae_phi_2=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
opt_mae_beta1_2=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
opt_bias_mean_phi_3=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
opt_bias_mean_beta1_3=zeros(size(list_T,2), size(list_N,2), size(list_phi,2)); 

opt_std_phi_3=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));  
opt_std_beta1_3=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));

opt_rmse_phi_3=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
opt_rmse_beta1_3=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));

opt_mae_phi_3=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
opt_mae_beta1_3=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
opt_bias_mean_phi_4=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
opt_bias_mean_beta1_4=zeros(size(list_T,2), size(list_N,2), size(list_phi,2)); 

opt_std_phi_4=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));  
opt_std_beta1_4=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));

opt_rmse_phi_4=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
opt_rmse_beta1_4=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));

opt_mae_phi_4=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
opt_mae_beta1_4=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for idx_phi=1:size(list_phi,2)     
phi= list_phi(idx_phi);  
  b1=3;
 
for idx_T=1:size(list_T,2)    
T0 = list_T(idx_T);             
T1=T0-1; 
for idx_N=1:size(list_N,2)   
N = list_N(idx_N);

TT= (T0+1+j)+Dis_T;    


ini_IVMG_1=zeros(1+k,rep);   % 1+k by rep
ini_IVMG_2=zeros(1+k,rep);   % 1+k by rep
ini_IVMG_3=zeros(1+k,rep);   % 1+k by rep
ini_IVMG_4=zeros(1+k,rep);   % 1+k by rep


opt_IVMG_2=zeros(1+k,rep);   % 1+k by rep
opt_IVMG_3=zeros(1+k,rep);   % 1+k by rep
opt_IVMG_4=zeros(1+k,rep);   % 1+k by rep

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
randn('state', 12345678) ;
rand('state', 1234567) ;
   RandStream.setGlobalStream (RandStream('mcg16807','seed',34));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sml=1;         
while sml<=rep
%parfor sml=1:rep
y=zeros(N,TT);
x=zeros(k,N,TT);  
eta_rho_i=-0.2+(0.4)*rand([1,N]); % 1 by N
phi_i= phi+  eta_rho_i;

beta_i=b'*ones(1,N)+sqrt(1-rho_b^(2))*ones(k,1)*eta_rho_i; % k by N
eta_i=normrnd(0,1,[N,1]);

v_x=normrnd(0,1,[N,TT]);  % N by TT


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

y(:,1)=zeros(N,1);  
x(:,:,1)=zeros(k,N,1); 
for t=2:TT
    for ii=1:N 
x(:,ii,t)=rho_b1*x(:,ii,t-1)+tao*eta_i(ii,1)+ v_theta*v_x(ii,t-1)+normrnd(0,1);
y(ii,t)=phi_i(:,ii)*y(ii,t-1)+beta_i(k,ii)*x(:,ii,t)+eta_i(ii,1)+v_x(ii,t);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
F_11=zeros(1,T1);
for tt=1:T1
F_11(:,tt)=sqrt((T0-tt)/(T0-tt+1));
end

F_1=diag(F_11);

F_2=zeros(T1,T0);
for bt=1:T0
 A=T1:-1:bt;   
 B=-1*A.^-1;  
 C=diag(B);
F_2= F_2+[zeros(T1,bt) [C;zeros(bt-1,T0-bt)]] ;
end
F_2=[diag(ones(1,T1)) zeros(T1,1)]+F_2;
F=  F_1*F_2;   % T0-1 by T0


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

y_NT=y(:,TT-T0:TT); % dicard first 50 time series N by T0+1 
y_NT1=F*y_NT(:,1:T0)';  % y_(i,-1); T1 by N
y_NT2=F*y_NT(:,2:T0+1)';  % y_(i,T)  ; T1 by N

x_NT=x(:,:,TT-T0-j:TT); %  dicard first 50 time series for x ; k by N by T0+1+j


x_NT1=zeros(N, T0);
x_NT1_1=zeros(N, T0);

x_NT1_2=zeros(N, T0);

x_NT1_3=zeros(N, T0);

x_NT1_4=zeros(N, T0);



for it=1:T0
x_NT1(:,it)=x_NT(1,:,it+j+1);   % x_(i,-1) ; N by T0 
x_NT1_1(:,it)=x_NT(1,:,it+j);   % x_(i,-1) ; N by T0 
x_NT1_2(:,it)=x_NT(1,:,it+(j-1));   % x_(i,-2) ; N by T0 
x_NT1_3(:,it)=x_NT(1,:,it+(j-2));   % x_(i,-3) ; N by T0 
x_NT1_4(:,it)=x_NT(1,:,it+(j-3));   % x_(i,-4) ; N by T0
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
W_it=zeros(N,T1,1+k);  % N by T1 by 1+k  
W_it(:,:,1)=y_NT1';  %  y_{i,-1}; N by T1   
W_it(:,:,2)=(F*x_NT1')'; %  x_(i,T); N by T1  
%W_it(:,:,2)=x_NT1;  %  x_(i,T); N by T1  

Z_it_1=zeros(N,T1,2*k);           % N by T1 by 2k
Z_it_1(:,:,1)=(F*x_NT1')';   % N by T1 
Z_it_1(:,:,2)=(F*x_NT1_1')';  % N by T1 


Z_it_2=zeros(N,T1,3*k);           % N by T1 by 3k
Z_it_2(:,:,1)=(F*x_NT1')';  % N by T1 
Z_it_2(:,:,2)=(F*x_NT1_1')';  % N by T1  ; lag 1
Z_it_2(:,:,3)=(F*x_NT1_2')';   % N by T1    ; lag 2

Z_it_3=zeros(N,T1,4*k);           % N by T1  by 4k
Z_it_3(:,:,1)=(F*x_NT1')';  % N by T1 
Z_it_3(:,:,2)=(F*x_NT1_1')';   % N by T1   ; lag 1
Z_it_3(:,:,3)=(F*x_NT1_2')';   % N by T1     ; lag 2
Z_it_3(:,:,4)=(F*x_NT1_3')';  % N by T1     ; lag 3


Z_it_4=zeros(N,T1,5*k);           % N by T1  by 5k
Z_it_4(:,:,1)=(F*x_NT1')';  % N by T1  
Z_it_4(:,:,2)=(F*x_NT1_1')';  % N by T1  ; lag 1
Z_it_4(:,:,3)=(F*x_NT1_2')';    % N by T1     ; lag 2
Z_it_4(:,:,4)=(F*x_NT1_3')';  % N by T1    ; lag 3
Z_it_4(:,:,5)=(F*x_NT1_4')';  % N by T1     ; lag 4





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
D=zeros(T1,1+k,N);    
Z_1=zeros(T1,2*k,N);   % lag 1
Z_2=zeros(T1,3*k,N);  % lag 2
Z_3=zeros(T1,4*k,N);  % lag 3
Z_4=zeros(T1,5*k,N);  % lag 4

for iti=1:N
  D(:,:,iti)=[W_it(iti,:,1)', W_it(iti,:,2)']; % T1 by 1+k
  Z_1(:,:,iti)=[Z_it_1(iti,:,1)', Z_it_1(iti,:,2)'];  %  T1 by 2k by N  lag 1
  Z_2(:,:,iti)=[Z_it_2(iti,:,1)', Z_it_2(iti,:,2)', Z_it_2(iti,:,3)'];  %  T1 by 3k by N lag 2  
  Z_3(:,:,iti)=[Z_it_3(iti,:,1)', Z_it_3(iti,:,2)', Z_it_3(iti,:,3)', Z_it_3(iti,:,4)'];  %  T1 by 3k by N lag 3
  Z_4(:,:,iti)=[Z_it_4(iti,:,1)', Z_it_4(iti,:,2)', Z_it_4(iti,:,3)', Z_it_4(iti,:,4)', Z_it_4(iti,:,5)'];  %  T1 by 3k by N lag 4
end    


P_i_1=zeros(T1,T1,N);
P_i_2=zeros(T1,T1,N);
P_i_3=zeros(T1,T1,N);
P_i_4=zeros(T1,T1,N);

for p=1:N
  P_i_1(:,:,p)=  Z_1(:,:,p)*inv(Z_1(:,:,p)'*Z_1(:,:,p))*Z_1(:,:,p)' ;% T1 by T1   
  P_i_2(:,:,p)=  Z_2(:,:,p)*inv(Z_2(:,:,p)'*Z_2(:,:,p))*Z_2(:,:,p)' ;% T1 by T1   
  P_i_3(:,:,p)=  Z_3(:,:,p)*inv(Z_3(:,:,p)'*Z_3(:,:,p))*Z_3(:,:,p)' ;% T1 by T1 
  P_i_4(:,:,p)=  Z_4(:,:,p)*inv(Z_4(:,:,p)'*Z_4(:,:,p))*Z_4(:,:,p)' ;% T1 by T1   
end

H=zeros(1+k,1+k,N);
for hi=1:N
H(:,:,hi)=D(:,:,hi)'*D(:,:,hi);
end


ini_theta_IV_1=zeros(1+k,N); 
for iii1=1:N
ini_theta_IV_1(:,iii1)= inv(D(:,:,iii1)'*P_i_1(:,:,iii1)*D(:,:,iii1))*D(:,:,iii1)'*P_i_1(:,:,iii1)*y_NT2(:,iii1);
end

ini_theta_IV_2=zeros(1+k,N); 
for iii2=1:N
ini_theta_IV_2(:,iii2)= inv(D(:,:,iii2)'*P_i_2(:,:,iii2)*D(:,:,iii2))*D(:,:,iii2)'*P_i_2(:,:,iii2)*y_NT2(:,iii2);
end

ini_theta_IV_3=zeros(1+k,N); 
for iii3=1:N
ini_theta_IV_3(:,iii3)= inv(D(:,:,iii3)'*P_i_3(:,:,iii3)*D(:,:,iii3))*D(:,:,iii3)'*P_i_3(:,:,iii3)*y_NT2(:,iii3);
end

ini_theta_IV_4=zeros(1+k,N); 
for iii4=1:N
ini_theta_IV_4(:,iii4)= inv(D(:,:,iii4)'*P_i_4(:,:,iii4)*D(:,:,iii4))*D(:,:,iii4)'*P_i_4(:,:,iii4)*y_NT2(:,iii4);
end




hat_u=zeros(T1,N);
for hat_i=1:N
hat_u(:,hat_i)= y_NT2(:,hat_i)- D(:,:,hat_i)*ini_theta_IV_1(:,hat_i);  % T1 by N; preliminary residual 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
V_2=zeros(T1,1+k,N);
for V_i_2=1:N
   V_2(:,:,V_i_2)= D(:,:,V_i_2)- Z_2(:,:,V_i_2)*inv(Z_2(:,:,V_i_2)'*Z_2(:,:,V_i_2))*Z_2(:,:,V_i_2)'*D(:,:,V_i_2);  %T1 by 1+k ; first stage residual
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
V_3=zeros(T1,1+k,N);
for V_i_3=1:N
   V_3(:,:,V_i_3)= D(:,:,V_i_3)- Z_3(:,:,V_i_3)*inv(Z_3(:,:,V_i_3)'*Z_3(:,:,V_i_3))*Z_3(:,:,V_i_3)'*D(:,:,V_i_3);  %T1 by 1+k ; first stage residual
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
V=zeros(T1,1+k,N);
for V_i=1:N
   V(:,:,V_i)= D(:,:,V_i)- Z_4(:,:,V_i)*inv(Z_4(:,:,V_i)'*Z_4(:,:,V_i))*Z_4(:,:,V_i)'*D(:,:,V_i);  %T1 by 1+k ; first stage residual
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

eta_i_2=rand(1+k,N);
hat_v_eta_2=zeros(T1,1,N);
for v_i_2=1:N
    hat_v_eta_2(:,:,v_i_2)=  V(:,:,v_i_2)*inv(H(:,:,v_i_2))*eta_i_2(:,v_i_2);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
eta_i_3=rand(1+k,N);
hat_v_eta_3=zeros(T1,1,N);
for v_i_3=1:N
    hat_v_eta_3(:,:,v_i_3)=  V(:,:,v_i_3)*inv(H(:,:,v_i_3))*eta_i_3(:,v_i_3);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
eta_i=rand(1+k,N);
hat_v_eta=zeros(T1,1,N);
for v_i=1:N
    hat_v_eta(:,:,v_i)=  V(:,:,v_i)*inv(H(:,:,v_i))*eta_i(:,v_i);
end
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
J=1+j;
Sigma_u_i=zeros(1,N);
Sigma_eta_i=zeros(1,N);
sigma_etau=zeros(1,N);
V_eta_1=zeros(T1,N);   %
V_eta_2=zeros(T1,N);  %
V_eta_3=zeros(T1,N);   %
V_eta_4=zeros(T1,N);  %
hat_U=zeros(j,j,N);
for sig=1:N
 Sigma_u_i(:,sig)=hat_u(:,sig)'*hat_u(:,sig)/T1;
 Sigma_eta_i(:,sig)= hat_v_eta(:,:,sig)'*hat_v_eta(:,:,sig)/T1;
 sigma_etau(:,sig)= hat_v_eta(:,:,sig)'*hat_u(:,sig)/T1;
 V_eta_1(:,sig)=(P_i_4(:,:,sig)-P_i_1(:,:,sig))* D(:,:,sig)*inv(H(:,:,sig))*eta_i(:,sig);   % T1 by 1
 V_eta_2(:,sig)=(P_i_4(:,:,sig)-P_i_2(:,:,sig))* D(:,:,sig)*inv(H(:,:,sig))*eta_i(:,sig);
 V_eta_3(:,sig)=(P_i_4(:,:,sig)-P_i_3(:,:,sig))* D(:,:,sig)*inv(H(:,:,sig))*eta_i(:,sig);
 V_eta_4(:,sig)=(P_i_4(:,:,sig)-P_i_4(:,:,sig))* D(:,:,sig)*inv(H(:,:,sig))*eta_i(:,sig);
 hat_U2(:,:,sig)=[V_eta_1(:,sig),V_eta_2(:,sig)]'*[ V_eta_1(:,sig),V_eta_2(:,sig)];
 hat_U3(:,:,sig)=[V_eta_1(:,sig),V_eta_2(:,sig),V_eta_3(:,sig)]'*[ V_eta_1(:,sig),V_eta_2(:,sig),V_eta_3(:,sig)];
 hat_U(:,:,sig)=[V_eta_1(:,sig),V_eta_2(:,sig),V_eta_3(:,sig),V_eta_4(:,sig)]'*[ V_eta_1(:,sig),V_eta_2(:,sig),V_eta_3(:,sig),V_eta_4(:,sig)];


end

Gamma_2=[1,1;1,2];
Gamma_3=[1,1,1;1,2,2;1,2,3];

Gamma=[1,1,1,1;1,2,2,2;1,2,3,3;1,2,3,4];
K_2=ones(2,1);
K_3=ones(3,1);
K=ones(j,1);
j2=2;
 j3=3;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
omega_IV_i_2=zeros(2,N);
for ome_i=1:N
    omega_ini_2 = rand(2,1);
    lq_2 = [zeros(2,1)];
    uq_2 = [ones(2,1)];
    
    sigma_etau_i=sigma_etau(:,ome_i); 
    Sigma_u_i_i=Sigma_u_i(:,ome_i);
    Sigma_eta_i_i=Sigma_eta_i(:,ome_i);
    hat_U_i=hat_U2(:,:,ome_i);    
    [omega_2, fval_2, exitflag_2, output_2, lambda_2, hessian_2] = MAIV_opt_2(T1,K_2,sigma_etau_i,Sigma_u_i_i,Sigma_eta_i_i, hat_U_i,Gamma_2,j2,omega_ini_2,lq_2,uq_2);
 omega_IV_i_2(:,ome_i)=omega_2;
 
end


omega_IV_i_3=zeros(3,N);
for ome_i=1:N
    omega_ini_3 = rand(3,1);
    lq_3 = [zeros(3,1)];
    uq_3 = [ones(3,1)];
   
    sigma_etau_i=sigma_etau(:,ome_i); 
    Sigma_u_i_i=Sigma_u_i(:,ome_i);
    Sigma_eta_i_i=Sigma_eta_i(:,ome_i);
    hat_U_i=hat_U3(:,:,ome_i);    
    [omega_3, fval_3, exitflag_3, output_3, lambda_3, hessian_3] = MAIV_opt_3(T1,K_3,sigma_etau_i,Sigma_u_i_i,Sigma_eta_i_i, hat_U_i,Gamma_3,j3,omega_ini_3,lq_3,uq_3);
 omega_IV_i_3(:,ome_i)=omega_3;
 
end




omega_IV_i=zeros(j,N);
for ome_i=1:N
    omega_ini = rand(j,1);
    lq = [zeros(j,1)];
    uq = [ones(j,1)];
    sigma_etau_i=sigma_etau(:,ome_i); 
    Sigma_u_i_i=Sigma_u_i(:,ome_i);
    Sigma_eta_i_i=Sigma_eta_i(:,ome_i);
    hat_U_i=hat_U(:,:,ome_i);    
    [omega, fval, exitflag, output, lambda, hessian] = MAIV_opt(T1,K,sigma_etau_i,Sigma_u_i_i,Sigma_eta_i_i, hat_U_i,Gamma,j,omega_ini,lq,uq);
 omega_IV_i(:,ome_i)=omega;
 
end

P_2=zeros(T1,T1,N);
for pi_2=1:N
P_2(:,:,pi_2)= omega_IV_i_2(1,pi_2)*P_i_1(:,:,pi_2)+omega_IV_i_2(2,pi_2)*P_i_2(:,:,pi_2);
end

P_3=zeros(T1,T1,N);
for pi_3=1:N
P_3(:,:,pi_3)= omega_IV_i_3(1,pi_3)*P_i_1(:,:,pi_3)+omega_IV_i_3(2,pi_3)*P_i_2(:,:,pi_3)+omega_IV_i_3(3,pi_3)*P_i_3(:,:,pi_3);
end

P_4=zeros(T1,T1,N);
for pi=1:N
P_4(:,:,pi)= omega_IV_i(1,pi)*P_i_1(:,:,pi)+omega_IV_i(2,pi)*P_i_2(:,:,pi)+omega_IV_i(3,pi)*P_i_3(:,:,pi)+omega_IV_i(4,pi)*P_i_4(:,:,pi) ;
end

opt_theta_IV_i_2=zeros(1+k,N);
for ome_i_2=1:N
opt_theta_IV_i_2(:,ome_i_2)= inv(D(:,:,ome_i_2)'*P_2(:,:,ome_i_2)*D(:,:,ome_i_2))*D(:,:,ome_i_2)'*P_2(:,:,ome_i_2)*y_NT2(:,ome_i_2) ;
end

opt_theta_IV_i_3=zeros(1+k,N);
for ome_i_3=1:N
opt_theta_IV_i_3(:,ome_i_3)= inv(D(:,:,ome_i_3)'*P_3(:,:,ome_i_3)*D(:,:,ome_i_3))*D(:,:,ome_i_3)'*P_3(:,:,ome_i_3)*y_NT2(:,ome_i_3) ;
end


opt_theta_IV_i=zeros(1+k,N);
for ome_i=1:N
opt_theta_IV_i(:,ome_i)= inv(D(:,:,ome_i)'*P_4(:,:,ome_i)*D(:,:,ome_i))*D(:,:,ome_i)'*P_4(:,:,ome_i)*y_NT2(:,ome_i) ;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ini_IVMG_1(:,sml)=nanmean(ini_theta_IV_1,2); 
ini_IVMG_2(:,sml)=nanmean(ini_theta_IV_2,2); 
ini_IVMG_3(:,sml)=nanmean(ini_theta_IV_3,2); 
ini_IVMG_4(:,sml)=nanmean(ini_theta_IV_4,2); 


opt_IVMG_2(:,sml)=nanmean(opt_theta_IV_i_2,2); % LS_MG
opt_IVMG_3(:,sml)=nanmean(opt_theta_IV_i_3,2); % LS_MG
opt_IVMG_4(:,sml)=nanmean(opt_theta_IV_i,2); % LS_MG


sml=sml+1;
end


ini_mean_phi_1= nanmean(ini_IVMG_1(1,:));
ini_mean_beta1_1= nanmean(ini_IVMG_1(2,:));

ini_mean_phi_2= nanmean(ini_IVMG_2(1,:));
ini_mean_beta1_2= nanmean(ini_IVMG_2(2,:));

ini_mean_phi_3= nanmean(ini_IVMG_3(1,:));
ini_mean_beta1_3= nanmean(ini_IVMG_3(2,:));

ini_mean_phi_4= nanmean(ini_IVMG_4(1,:));
ini_mean_beta1_4= nanmean(ini_IVMG_4(2,:));


opt_mean_phi_2= nanmean(opt_IVMG_2(1,:));
opt_mean_beta1_2= nanmean(opt_IVMG_2(2,:));

opt_mean_phi_3= nanmean(opt_IVMG_3(1,:));
opt_mean_beta1_3= nanmean(opt_IVMG_3(2,:));


opt_mean_phi= nanmean(opt_IVMG_4(1,:));
opt_mean_beta1= nanmean(opt_IVMG_4(2,:));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ini_bias_mean_phi_1(idx_T, idx_N, idx_phi) = ini_mean_phi_1 - phi;
ini_bias_mean_beta1_1(idx_T, idx_N, idx_phi)=ini_mean_beta1_1 - b1;

ini_bias_mean_phi_2(idx_T, idx_N, idx_phi) = ini_mean_phi_2 - phi;
ini_bias_mean_beta1_2(idx_T, idx_N, idx_phi)=ini_mean_beta1_2 - b1;

ini_bias_mean_phi_3(idx_T, idx_N, idx_phi) = ini_mean_phi_3 - phi;
ini_bias_mean_beta1_3(idx_T, idx_N, idx_phi)=ini_mean_beta1_3 - b1;

ini_bias_mean_phi_4(idx_T, idx_N, idx_phi) = ini_mean_phi_4 - phi;
ini_bias_mean_beta1_4(idx_T, idx_N, idx_phi)=ini_mean_beta1_4 - b1;

opt_bias_mean_phi_2(idx_T, idx_N, idx_phi) =opt_mean_phi_2 - phi;
opt_bias_mean_beta1_2(idx_T, idx_N, idx_phi)=opt_mean_beta1_2 - b1;

opt_bias_mean_phi_3(idx_T, idx_N, idx_phi) =opt_mean_phi_3 - phi;
opt_bias_mean_beta1_3(idx_T, idx_N, idx_phi)=opt_mean_beta1_3 - b1;

opt_bias_mean_phi_4(idx_T, idx_N, idx_phi) =opt_mean_phi - phi;
opt_bias_mean_beta1_4(idx_T, idx_N, idx_phi)=opt_mean_beta1 - b1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


ini_std_phi_1(idx_T, idx_N, idx_phi) = nanstd(ini_IVMG_1(1,:));
ini_std_beta1_1(idx_T, idx_N, idx_phi) = nanstd(ini_IVMG_1(2,:));

ini_std_phi_2(idx_T, idx_N, idx_phi) = nanstd(ini_IVMG_2(1,:));
ini_std_beta1_2(idx_T, idx_N, idx_phi) = nanstd(ini_IVMG_2(2,:));

ini_std_phi_3(idx_T, idx_N, idx_phi) = nanstd(ini_IVMG_3(1,:));
ini_std_beta1_3(idx_T, idx_N, idx_phi) = nanstd(ini_IVMG_3(2,:));

ini_std_phi_4(idx_T, idx_N, idx_phi) = nanstd(ini_IVMG_4(1,:));
ini_std_beta1_4(idx_T, idx_N, idx_phi) = nanstd(ini_IVMG_4(2,:));



opt_std_phi_2(idx_T, idx_N, idx_phi) = nanstd(opt_IVMG_2(1,:));
opt_std_beta1_2(idx_T, idx_N, idx_phi) = nanstd(opt_IVMG_2(2,:));

opt_std_phi_3(idx_T, idx_N, idx_phi) = nanstd(opt_IVMG_3(1,:));
opt_std_beta1_3(idx_T, idx_N, idx_phi) = nanstd(opt_IVMG_3(2,:));


opt_std_phi_4(idx_T, idx_N, idx_phi) = nanstd(opt_IVMG_4(1,:));
opt_std_beta1_4(idx_T, idx_N, idx_phi) = nanstd(opt_IVMG_4(2,:));



ini_rmse_phi_1(idx_T, idx_N,idx_phi) = sqrt( nanmean( (ini_IVMG_1(1,:)-phi).^2) ); 
ini_rmse_beta1_1(idx_T, idx_N, idx_phi) = sqrt( nanmean( (ini_IVMG_1(2,:)-b1).^2) ); 

ini_rmse_phi_2(idx_T, idx_N,idx_phi) = sqrt( nanmean( (ini_IVMG_2(1,:)-phi).^2) ); 
ini_rmse_beta1_2(idx_T, idx_N, idx_phi) = sqrt( nanmean( (ini_IVMG_2(2,:)-b1).^2) ); 

ini_rmse_phi_3(idx_T, idx_N,idx_phi) = sqrt( nanmean( (ini_IVMG_3(1,:)-phi).^2) ); 
ini_rmse_beta1_3(idx_T, idx_N, idx_phi) = sqrt( nanmean( (ini_IVMG_3(2,:)-b1).^2) ); 

ini_rmse_phi_4(idx_T, idx_N,idx_phi) = sqrt( nanmean( (ini_IVMG_4(1,:)-phi).^2) ); 
ini_rmse_beta1_4(idx_T, idx_N, idx_phi) = sqrt( nanmean( (ini_IVMG_4(2,:)-b1).^2) ); 


opt_rmse_phi_2(idx_T, idx_N,idx_phi) = sqrt( nanmean( (opt_IVMG_2(1,:)-phi).^2) ); 
opt_rmse_beta1_2(idx_T, idx_N, idx_phi) = sqrt( nanmean( (opt_IVMG_2(2,:)-b1).^2) ); 

opt_rmse_phi_3(idx_T, idx_N,idx_phi) = sqrt( nanmean( (opt_IVMG_3(1,:)-phi).^2) ); 
opt_rmse_beta1_3(idx_T, idx_N, idx_phi) = sqrt( nanmean( (opt_IVMG_3(2,:)-b1).^2) ); 

opt_rmse_phi_4(idx_T, idx_N,idx_phi) = sqrt( nanmean( (opt_IVMG_4(1,:)-phi).^2) ); 
opt_rmse_beta1_4(idx_T, idx_N, idx_phi) = sqrt( nanmean( (opt_IVMG_4(2,:)-b1).^2) ); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ini_mae_phi_1(idx_T, idx_N,idx_phi) = median((ini_IVMG_1(1,:)-phi).^2);
ini_mae_beta1_1(idx_T, idx_N, idx_phi) = median((ini_IVMG_1(2,:)-b1).^2);


ini_mae_phi_2(idx_T, idx_N,idx_phi) = median((ini_IVMG_2(1,:)-phi).^2);
ini_mae_beta1_2(idx_T, idx_N, idx_phi) = median((ini_IVMG_2(2,:)-b1).^2);

ini_mae_phi_3(idx_T, idx_N,idx_phi) = median((ini_IVMG_3(1,:)-phi).^2);
ini_mae_beta1_3(idx_T, idx_N, idx_phi) = median((ini_IVMG_3(2,:)-b1).^2);

ini_mae_phi_4(idx_T, idx_N,idx_phi) = median((ini_IVMG_4(1,:)-phi).^2);
ini_mae_beta1_4(idx_T, idx_N, idx_phi) = median((ini_IVMG_4(2,:)-b1).^2);


opt_mae_phi_2(idx_T, idx_N,idx_phi) = median((opt_IVMG_2(1,:)-phi).^2);
opt_mae_beta1_2(idx_T, idx_N, idx_phi) = median((opt_IVMG_2(2,:)-b1).^2);

opt_mae_phi_3(idx_T, idx_N,idx_phi) = median((opt_IVMG_3(1,:)-phi).^2);
opt_mae_beta1_3(idx_T, idx_N, idx_phi) = median((opt_IVMG_3(2,:)-b1).^2);


opt_mae_phi_4(idx_T, idx_N,idx_phi) = median((opt_IVMG_4(1,:)-phi).^2);
opt_mae_beta1_4(idx_T, idx_N, idx_phi) = median((opt_IVMG_4(2,:)-b1).^2);



end
end
end

