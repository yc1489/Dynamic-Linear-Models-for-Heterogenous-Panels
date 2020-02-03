rep = 100;  
list_T = [25 100]; 
list_N = [25 100];  
list_phi= [0.5]; 
list_power = [0.00 0.10]; % for power computation
b1=3;
rho_b=0.4;
rho_b1=0.5;
tao=0.5;
v_theta=-0.2;
b=[b1];
k=1;   % number of regressors
j=1; % lag of regressor
J=1+j;
Dis_T=50;
%%%%%%%%%%%%%%%%%%%%%%%%%%%
bias_IVMG2_phi=zeros(size(list_T,2), size(list_N,2));
bias_IVMG2_beta1=zeros(size(list_T,2), size(list_N,2)); 

std_IVMG2_phi=zeros(size(list_T,2), size(list_N,2));  
std_IVMG2_beta1=zeros(size(list_T,2), size(list_N,2));  

rmse_IVMG2_phi=zeros(size(list_T,2), size(list_N,2));
rmse_IVMG2_beta1=zeros(size(list_T,2), size(list_N,2));
size_IVMG2_phi=zeros(size(list_T,2), size(list_N,2));  
power_IVMG2_phi=zeros(size(list_T,2), size(list_N,2));  
size_IVMG2_beta=zeros(size(list_T,2), size(list_N,2));  
power_IVMG2_beta=zeros(size(list_T,2), size(list_N,2));  
%%%%%%%%%%%%%%%%%%%%%%%%
bias_IVMG3_phi=zeros(size(list_T,2), size(list_N,2));
bias_IVMG3_beta1=zeros(size(list_T,2), size(list_N,2)); 

std_IVMG3_phi=zeros(size(list_T,2), size(list_N,2));  
std_IVMG3_beta1=zeros(size(list_T,2), size(list_N,2));  

rmse_IVMG3_phi=zeros(size(list_T,2), size(list_N,2));
rmse_IVMG3_beta1=zeros(size(list_T,2), size(list_N,2));

size_IVMG3_phi=zeros(size(list_T,2), size(list_N,2));  
power_IVMG3_phi=zeros(size(list_T,2), size(list_N,2));  
size_IVMG3_beta=zeros(size(list_T,2), size(list_N,2));  
power_IVMG3_beta=zeros(size(list_T,2), size(list_N,2));  


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

optMG_bias_phi=zeros(size(list_T,2), size(list_N,2));
optMG_bias_beta1=zeros(size(list_T,2), size(list_N,2)); 

optMG_std_phi=zeros(size(list_T,2), size(list_N,2));  
optMG_std_beta1=zeros(size(list_T,2), size(list_N,2));

optMG_rmse_phi=zeros(size(list_T,2), size(list_N,2));
optMG_rmse_beta1=zeros(size(list_T,2), size(list_N,2));

size_optMG_phi=zeros(size(list_T,2), size(list_N,2));  
power_optMG_phi=zeros(size(list_T,2), size(list_N,2));  
size_optMG_beta=zeros(size(list_T,2), size(list_N,2));  
power_optMG_beta=zeros(size(list_T,2), size(list_N,2));  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bias_IV2_phi=zeros(size(list_T,2), size(list_N,2));
bias_IV2_beta1=zeros(size(list_T,2), size(list_N,2)); 

std_IV2_phi=zeros(size(list_T,2), size(list_N,2));  
std_IV2_beta1=zeros(size(list_T,2), size(list_N,2));  

rmse_IV2_phi=zeros(size(list_T,2), size(list_N,2));
rmse_IV2_beta1=zeros(size(list_T,2), size(list_N,2));
%%%%%%%%%%%%%%%%%%%%%%%%
bias_IV3_phi=zeros(size(list_T,2), size(list_N,2));
bias_IV3_beta1=zeros(size(list_T,2), size(list_N,2)); 

std_IV3_phi=zeros(size(list_T,2), size(list_N,2));  
std_IV3_beta1=zeros(size(list_T,2), size(list_N,2));  

rmse_IV3_phi=zeros(size(list_T,2), size(list_N,2));
rmse_IV3_beta1=zeros(size(list_T,2), size(list_N,2));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
opt_bias_phi=zeros(size(list_T,2), size(list_N,2));
opt_bias_beta1=zeros(size(list_T,2), size(list_N,2)); 

opt_std_phi=zeros(size(list_T,2), size(list_N,2));  
opt_std_beta1=zeros(size(list_T,2), size(list_N,2));

opt_rmse_phi=zeros(size(list_T,2), size(list_N,2));
opt_rmse_beta1=zeros(size(list_T,2), size(list_N,2));
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

bias_IV2=zeros(1+k,rep);
bias_IV3=zeros(1+k,rep);
bias_opt_IV=zeros(1+k,rep);
opt_IVMG=zeros(1+k,rep);   % 1+k by rep
ini_IVMG=zeros(1+k,rep);   % 1+k by rep
ini_IVMG2=zeros(1+k,rep);   % 1+k by rep

se_IVMG1=zeros(1+k,1+k,rep);
se_IVMG2=zeros(1+k,1+k,rep);
se_opt_IVMG=zeros(1+k,1+k,rep);
size_power_IVMG1_phi= zeros(size(list_power,2),rep);
size_power_IVMG1_beta= zeros(size(list_power,2),rep);
size_power_IVMG2_phi= zeros(size(list_power,2),rep);
size_power_IVMG2_beta= zeros(size(list_power,2),rep);
size_power_optIVMG_phi= zeros(size(list_power,2),rep);
size_power_optIVMG_beta= zeros(size(list_power,2),rep);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
se_IV1=zeros(1+k,1+k,rep);
se_IV2=zeros(1+k,1+k,rep);
se_opt_IV=zeros(1+k,1+k,rep);
size_power_IV1_phi= zeros(size(list_power,2),rep);
size_power_IV1_beta= zeros(size(list_power,2),rep);
size_power_IV2_phi= zeros(size(list_power,2),rep);
size_power_IV2_beta= zeros(size(list_power,2),rep);
size_power_optIV_phi= zeros(size(list_power,2),rep);
size_power_optIV_beta= zeros(size(list_power,2),rep);

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
theta_i= [phi_i; beta_i]; %(1+k) by N
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


for it=1:T0
x_NT1(:,it)=x_NT(1,:,it+j+1);   % x_(i,:) ; N by T0 
x_NT1_1(:,it)=x_NT(1,:,it+j);   % x_(i,-1) ; N by T0 
x_NT1_2(:,it)=x_NT(1,:,it+(j-1));   % x_(i,-2) ; N by T0 
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
W_it=zeros(N,T1,1+k);  % N by T1 by 1+k  
W_it(:,:,1)=y_NT1';  %  y_{i,-1}; N by T1   
W_it(:,:,2)=(F*x_NT1')'; %  x_(i,T); N by T1  
%W_it(:,:,2)=x_NT1;  %  x_(i,T); N by T1  

Z_it_1=zeros(N,T1,2*k);           % N by T1 by 2k
Z_it_1(:,:,1)=x_NT1(:,1:T1);   % N by T1 
Z_it_1(:,:,2)=x_NT1_1(:,1:T1);   % N by T1 


Z_it_2=zeros(N,T1,3*k);           % N by T1 by 3k
Z_it_2(:,:,1)=x_NT1(:,1:T1);   % N by T1 
Z_it_2(:,:,2)=x_NT1_1(:,1:T1);   % N by T1  ; lag 1
Z_it_2(:,:,3)=x_NT1_2(:,1:T1);   % N by T1    ; lag 2



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
D=zeros(T1,1+k,N);    
Z_1=zeros(T1,2*k,N);   % lag 1
Z_2=zeros(T1,3*k,N);  % lag 2
for iti=1:N
  D(:,:,iti)=[W_it(iti,:,1)', W_it(iti,:,2)']; % T1 by 1+k
  Z_1(:,:,iti)=[Z_it_1(iti,:,1)', Z_it_1(iti,:,2)'];  %  T1 by 2k by N  lag 1
  Z_2(:,:,iti)=[Z_it_2(iti,:,1)', Z_it_2(iti,:,2)', Z_it_2(iti,:,3)'];  %  T1 by 3k by N lag 2  
end    


P_i_1=zeros(T1,T1,N);
P_i_2=zeros(T1,T1,N);

for p=1:N
  P_i_1(:,:,p)=  Z_1(:,:,p)*inv(Z_1(:,:,p)'*Z_1(:,:,p))*Z_1(:,:,p)' ;% T1 by T1   
  P_i_2(:,:,p)=  Z_2(:,:,p)*inv(Z_2(:,:,p)'*Z_2(:,:,p))*Z_2(:,:,p)' ;% T1 by T1     
end

H=zeros(1+k,1+k,N);
for hi=1:N
H(:,:,hi)=D(:,:,hi)'*D(:,:,hi);
end



ini_theta_IV=zeros(1+k,N); 
for iii=1:N
%ini_theta_IV(:,iii)=(((Z_5(:,:,iii)'*D(:,:,iii))' /T1)*((Z_5(:,:,iii)'*Z_5(:,:,iii))/T1)^(-1)*((Z_5(:,:,iii)'*D(:,:,iii)) /T1))^(-1)*(((Z_5(:,:,iii)'*D(:,:,iii))' /T1)*((Z_5(:,:,iii)'*Z_5(:,:,iii))/T1)^(-1)*((Z_5(:,:,iii)'*y_NT2(:,iii))/T1));
ini_theta_IV(:,iii)= inv(D(:,:,iii)'*P_i_1(:,:,iii)*D(:,:,iii))*D(:,:,iii)'*P_i_1(:,:,iii)*y_NT2(:,iii);
end

ini_theta_IV_1=zeros(1+k,N); 
for iii1=1:N
ini_theta_IV_1(:,iii1)= inv(D(:,:,iii1)'*P_i_2(:,:,iii1)*D(:,:,iii1))*D(:,:,iii1)'*P_i_2(:,:,iii1)*y_NT2(:,iii1);
end





hat_u=zeros(T1,N);
for hat_i=1:N
hat_u(:,hat_i)= y_NT2(:,hat_i)- D(:,:,hat_i)*ini_theta_IV(:,hat_i);  % T1 by N; preliminary residual 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
V=zeros(T1,1+k,N);
for V_i=1:N
   V(:,:,V_i)= D(:,:,V_i)- Z_2(:,:,V_i)*inv(Z_2(:,:,V_i)'*Z_2(:,:,V_i))*Z_2(:,:,V_i)'*D(:,:,V_i);  %T1 by 1+k ; first stage residual
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
eta_i=rand(1+k,N);
hat_v_eta=zeros(T1,1,N);
for v_i=1:N
    hat_v_eta(:,:,v_i)=  V(:,:,v_i)*inv(H(:,:,v_i))*eta_i(:,v_i);
end
 

Sigma_u_i=zeros(1,N);
Sigma_eta_i=zeros(1,N);
sigma_etau=zeros(1,N);
V_eta_1=zeros(T1,N);   %
V_eta_2=zeros(T1,N);  %
hat_U=zeros(J,J,N);
for sig=1:N
 Sigma_u_i(:,sig)=hat_u(:,sig)'*hat_u(:,sig)/T1;
 Sigma_eta_i(:,sig)= hat_v_eta(:,:,sig)'*hat_v_eta(:,:,sig)/T1;
 sigma_etau(:,sig)= hat_v_eta(:,:,sig)'*hat_u(:,sig)/T1;
 V_eta_1(:,sig)=(P_i_2(:,:,sig)-P_i_1(:,:,sig))* D(:,:,sig)*inv(H(:,:,sig))*eta_i(:,sig);   % T1 by 1
 V_eta_2(:,sig)=(P_i_2(:,:,sig)-P_i_2(:,:,sig))* D(:,:,sig)*inv(H(:,:,sig))*eta_i(:,sig);
 hat_U(:,:,sig)=[V_eta_1(:,sig),V_eta_2(:,sig)]'*[ V_eta_1(:,sig),V_eta_2(:,sig)];
end


   Gamma=[1, 1 ; 1,2];
   
K=ones(J,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
omega_IV_i=zeros(J,N);

for ome_i=1:N
    omega_ini = rand(J,1);
    lq = [zeros(J,1)];
    uq = [ones(J,1)];
    sigma_etau_i=sigma_etau(:,ome_i); 
    Sigma_u_i_i=Sigma_u_i(:,ome_i);
    Sigma_eta_i_i=Sigma_eta_i(:,ome_i);
    hat_U_i=hat_U(:,:,ome_i);
    
    [omega, fval, exitflag, output, lambda, hessian] = MAIV_opt(T1,K,sigma_etau_i,Sigma_u_i_i,Sigma_eta_i_i, hat_U_i,Gamma,J,omega_ini,lq,uq);

 omega_IV_i(:,ome_i)=omega;
 
end

P_i=zeros(T1,T1,N);
for pi=1:N
P_i(:,:,pi)= omega_IV_i(1,pi)*P_i_1(:,:,pi)+omega_IV_i(2,pi)*P_i_2(:,:,pi) ;
end

opt_theta_IV_i=zeros(1+k,N);
for ome_i=1:N
opt_theta_IV_i(:,ome_i)= inv(D(:,:,ome_i)'*P_i(:,:,ome_i)*D(:,:,ome_i))*D(:,:,ome_i)'*P_i(:,:,ome_i)*y_NT2(:,ome_i) ;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bias_IV2(:,sml)= nanmean(ini_theta_IV-theta_i,2);
bias_IV3(:,sml)=nanmean(ini_theta_IV_1-theta_i,2);
bias_opt_IV(:,sml)=nanmean(opt_theta_IV_i-theta_i,2);

se_IV1(:,:,sml)=ini_theta_IV*ini_theta_IV'/(N-1);
se_IV2(:,:,sml)= ini_theta_IV_1*ini_theta_IV_1'/(N-1);
se_opt_IV(:,:,sml)=  opt_theta_IV_i*opt_theta_IV_i'/(N-1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ini_IVMG(:,sml)=nanmean(ini_theta_IV,2);  % mean group estimator
ini_IVMG2(:,sml)=nanmean(ini_theta_IV_1,2); 
opt_IVMG(:,sml)=nanmean(opt_theta_IV_i,2); 

se_IVMG1(:,:,sml)= ((ini_theta_IV-ini_IVMG(:,sml)*ones(1,N))*(ini_theta_IV-ini_IVMG(:,sml)*ones(1,N))')/(N-1);
se_IVMG2(:,:,sml)= ((ini_theta_IV_1-ini_IVMG2(:,sml)*ones(1,N))*(ini_theta_IV_1-ini_IVMG2(:,sml)*ones(1,N))')/(N-1);
se_opt_IVMG(:,:,sml)=  ((opt_theta_IV_i-opt_IVMG(:,sml)*ones(1,N))*(opt_theta_IV_i-opt_IVMG(:,sml)*ones(1,N))')/(N-1);



for idx_power=1:size(list_power,2)  % do test for power and size of test, power=P(reject H_0 | H_1 is true ), size=P(reject H_0 | H_0 is true) 
d = list_power(idx_power);

if abs(ini_IVMG(1,sml) - (phi-d))/se_IVMG1(1,1,sml)> 1.96; 
        size_power_IVMG1_phi(idx_power, sml)=1;  end

if abs(ini_IVMG(2,sml)-(b-d))/se_IVMG1(2,2,sml)> 1.96; 
        size_power_IVMG1_beta(idx_power, sml)=1;  end
    
if abs(ini_IVMG2(1,sml) - (phi-d))/se_IVMG2(1,1,sml)> 1.96; 
        size_power_IVMG2_phi(idx_power, sml)=1;  end

if abs(ini_IVMG2(2,sml)- (b-d))/se_IVMG2(2,2,sml)> 1.96; 
        size_power_IVMG2_beta(idx_power, sml)=1;  end
    
if abs(opt_IVMG(1,sml) - (phi-d))/se_opt_IVMG(1,1,sml)> 1.96; 
       size_power_optIVMG_phi(idx_power, sml)=1;  end

if abs(opt_IVMG(2,sml)- (b-d))/se_opt_IVMG(2,2,sml)> 1.96; 
       size_power_optIVMG_beta(idx_power, sml)=1;  end    
        
end    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for idx_power=1:size(list_power,2)  % do test for power and size of test, power=P(reject H_0 | H_1 is true ), size=P(reject H_0 | H_0 is true) 
d = list_power(idx_power);

if abs(ini_IVMG(1,sml) - (phi-d))/se_IVMG1(1,1,sml)> 1.96; 
        size_power_IVMG1_phi(idx_power, sml)=1;  end

if abs(ini_IVMG(2,sml)-(b-d))/se_IVMG1(2,2,sml)> 1.96; 
        size_power_IVMG1_beta(idx_power, sml)=1;  end
    
if abs(ini_IVMG2(1,sml) - (phi-d))/se_IVMG2(1,1,sml)> 1.96; 
        size_power_IVMG2_phi(idx_power, sml)=1;  end

if abs(ini_IVMG2(2,sml)- (b-d))/se_IVMG2(2,2,sml)> 1.96; 
        size_power_IVMG2_beta(idx_power, sml)=1;  end
    
if abs(opt_IVMG(1,sml) - (phi-d))/se_opt_IVMG(1,1,sml)> 1.96; 
       size_power_optIVMG_phi(idx_power, sml)=1;  end

if abs(opt_IVMG(2,sml)- (b-d))/se_opt_IVMG(2,2,sml)> 1.96; 
       size_power_optIVMG_beta(idx_power, sml)=1;  end    
        
end    





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






sml=sml+1;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

bias_IV2_phi(idx_T, idx_N)= nanmean(bias_IV2(1,:),2);
bias_IV2_beta1(idx_T, idx_N)= nanmean(bias_IV2(2,:),2);

bias_IV3_phi(idx_T, idx_N)= nanmean(bias_IV3(1,:),2);
bias_IV3_beta1(idx_T, idx_N)= nanmean(bias_IV3(2,:),2);

opt_bias_phi(idx_T, idx_N)= nanmean(bias_opt_IV(1,:),2);
opt_bias_beta1(idx_T, idx_N)= nanmean(bias_opt_IV(2,:),2);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bias_IVMG2_phi(idx_T, idx_N) = nanmean(ini_IVMG(1,:) - phi*ones(1,rep));
bias_IVMG2_beta1(idx_T, idx_N)=nanmean(ini_IVMG(2,:) - b1*ones(1,rep));

bias_IVMG3_phi(idx_T, idx_N) = nanmean(ini_IVMG2(1,:)- phi*ones(1,rep));
bias_IVMG3_beta1(idx_T, idx_N)=nanmean(ini_IVMG2(2,:)- b1*ones(1,rep));

optMG_bias_phi(idx_T, idx_N) =nanmean(opt_IVMG(1,:)- phi*ones(1,rep));
optMG_bias_beta1(idx_T, idx_N)=nanmean(opt_IVMG(2,:)- b1*ones(1,rep));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

std_IVMG2_phi(idx_T, idx_N) = nanstd(ini_IVMG(1,:));
std_IVMG2_beta1(idx_T, idx_N) = nanstd(ini_IVMG(2,:));

std_IVMG3_phi(idx_T, idx_N) = nanstd(ini_IVMG2(1,:));
std_IVMG3_beta1(idx_T, idx_N) = nanstd(ini_IVMG2(2,:));

opt_std_phi(idx_T, idx_N) = nanstd(opt_IVMG(1,:));
opt_std_beta1(idx_T, idx_N) = nanstd(opt_IVMG(2,:));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rmse_IV2_phi(idx_T, idx_N) = sqrt( nanmean( (bias_IV2(1,:)).^2) ); 
rmse_IV2_beta1(idx_T, idx_N) = sqrt( nanmean( (bias_IV2(2,:)).^2) ); 

rmse_IV3_phi(idx_T, idx_N) = sqrt( nanmean( (bias_IV3(1,:)).^2) ); 
rmse_IV3_beta1(idx_T, idx_N) = sqrt( nanmean( (bias_IV3(1,:)).^2) ); 

opt_rmse_phi(idx_T, idx_N) = sqrt( nanmean( (bias_opt_IV(1,:)).^2) ); 
opt_rmse_beta1(idx_T, idx_N) = sqrt( nanmean( (bias_opt_IV(2,:)).^2) ); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rmse_IVMG2_phi(idx_T, idx_N) = sqrt( nanmean( (ini_IVMG(1,:)-phi).^2) ); 
rmse_IVMG2_beta1(idx_T, idx_N) = sqrt( nanmean( (ini_IVMG(2,:)-b1).^2) ); 

rmse_IVMG3_phi(idx_T, idx_N) = sqrt( nanmean( (ini_IVMG2(1,:)-phi).^2) ); 
rmse_IVMG3_beta1(idx_T, idx_N) = sqrt( nanmean( (ini_IVMG2(2,:)-b1).^2) ); 

optMG_rmse_phi(idx_T, idx_N) = sqrt( nanmean( (opt_IVMG(1,:)-phi).^2) ); 
optMG_rmse_beta1(idx_T, idx_N) = sqrt( nanmean( (opt_IVMG(2,:)-b1).^2) ); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
size_power_IVMG2phi= nanmean(size_power_IVMG1_phi,2); 
 size_IVMG2_phi(idx_T, idx_N, idx_gam) = size_power_IVMG2phi(1,1)';                        
power_IVMG2_phi(idx_T, idx_N, idx_gam) = size_power_IVMG2phi(2,1)'; 

 size_power_IVMG2beta= nanmean(size_power_IVMG1_beta,2); 
 size_IVMG2_beta(idx_T, idx_N, idx_gam) = size_power_IVMG2beta(1,1)';                        
power_IVMG2_beta(idx_T, idx_N, idx_gam) = size_power_IVMG2beta(2,1)'; 
                            
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
size_power_IVMG3phi= nanmean(size_power_IVMG2_phi,2); 
 size_IVMG3_phi(idx_T, idx_N, idx_gam) = size_power_IVMG3phi(1,1)';                        
power_IVMG3_phi(idx_T, idx_N, idx_gam) = size_power_IVMG3phi(2,1)'; 

size_power_IVMG3beta= nanmean(size_power_IVMG2_beta,2); 
size_IVMG3_beta(idx_T, idx_N, idx_gam) = size_power_IVMG3beta(1,1)';                        
power_IVMG3_beta(idx_T, idx_N, idx_gam) = size_power_IVMG3beta(2,1)'; 
                            
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
size_power_optIVMGphi= nanmean(size_power_optIVMG_phi,2); 
size_optMG_phi(idx_T, idx_N, idx_gam) =size_power_optIVMGphi(1,1)';                        
power_optMG_phi(idx_T, idx_N, idx_gam) = size_power_optIVMGphi(2,1)'; 

size_power_optIVMGbeta= nanmean(size_power_optIVMG_beta,2); 
size_optMG_beta(idx_T, idx_N, idx_gam) = size_power_optIVMGbeta(1,1)';                        
power_optMG_beta(idx_T, idx_N, idx_gam) = size_power_optIVMGbeta(2,1)'; 
                            
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



 


end
end
end

