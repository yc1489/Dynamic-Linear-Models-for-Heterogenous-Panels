tic
rep = 500;  
list_T = [25]; 
list_N = [25];  
list_phi= [0.8 ]; 
b1=3;
b2=1;
b=[b1,b2]; % 1 by k 
rho=0; % lag of x ARDL(0,1) 
pi_u=0.75; % 0.25 or 0.75 
SNR=4;
Mu1=1;
Mu2=-0.5;
A=0.5;
rho_gamma_1s=0; %  0 independent factor loading; 0.5 correlated loading
k=2;   % number of regressors
m_x=2; % number of factors of regressor
my=1;
m_y=m_x+my; % number of factors of y
xi_es=(pi_u/(1-pi_u))*m_y;

rho_mu=0.5;
rho_v=0.5;
rho_b=0.4;
xi_ev=xi_es*(SNR-(rho_v^(2)/(1-rho_v^(2))))*((b1^(2)+b2^(2))/(1-rho_v^(2)))^(-1);



j=7; % lag of regressor
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
opt_bias_mean_phi_5=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
opt_bias_mean_beta1_5=zeros(size(list_T,2), size(list_N,2), size(list_phi,2)); 

opt_std_phi_5=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));  
opt_std_beta1_5=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));

opt_rmse_phi_5=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
opt_rmse_beta1_5=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));

opt_mae_phi_5=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
opt_mae_beta1_5=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
opt_bias_mean_phi_6=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
opt_bias_mean_beta1_6=zeros(size(list_T,2), size(list_N,2), size(list_phi,2)); 

opt_std_phi_6=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));  
opt_std_beta1_6=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));

opt_rmse_phi_6=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
opt_rmse_beta1_6=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));

opt_mae_phi_6=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
opt_mae_beta1_6=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
opt_bias_mean_phi_7=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
opt_bias_mean_beta1_7=zeros(size(list_T,2), size(list_N,2), size(list_phi,2)); 

opt_std_phi_7=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));  
opt_std_beta1_7=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));

opt_rmse_phi_7=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
opt_rmse_beta1_7=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));

opt_mae_phi_7=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
opt_mae_beta1_7=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



for idx_phi=1:size(list_phi,2)     
phi= list_phi(idx_phi);  
  
for idx_T=1:size(list_T,2)    
T0 = list_T(idx_T);             
T1=T0-1; 
for idx_N=1:size(list_N,2)   
N = list_N(idx_N);

TT= (T0+1+j)+Dis_T;    
ini_IVMG_1=zeros(1+k,rep);   % 1+k by rep


opt_IVMG_2=zeros(1+k,rep);   % 1+k by rep
opt_IVMG_3=zeros(1+k,rep);   % 1+k by rep
opt_IVMG_4=zeros(1+k,rep);   % 1+k by rep
opt_IVMG_5=zeros(1+k,rep);   % 1+k by rep
opt_IVMG_6=zeros(1+k,rep);   % 1+k by rep
opt_IVMG_7=zeros(1+k,rep);   % 1+k by rep
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
omega_IV_i_2=zeros(2,N,rep);
omega_IV_i_3=zeros(3,N,rep);
omega_IV_i_4=zeros(4,N,rep);
omega_IV_i_5=zeros(5,N,rep);
omega_IV_i_6=zeros(6,N,rep);
omega_IV_i_7=zeros(7,N,rep);

 fval2=zeros(N,rep);
 fval3=zeros(N,rep);
  fval4=zeros(N,rep);
   fval5=zeros(N,rep);
fval6=zeros(N,rep);
fval7=zeros(N,rep);





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
randn('state', 12345678) ;
rand('state', 1234567) ;
   RandStream.setGlobalStream (RandStream('mcg16807','seed',34));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sml=1;         
while sml<=rep
%parfor sml=1:rep


eta_rho_i=-0.2+(0.4)*rand([1,N]); % 1 by N heterogeneous_y U(-0.2,0.2)
eta_rho_bi=-sqrt(3)+2*sqrt(3)*rand([k,N]);


phi_i= phi+  eta_rho_i;
sig_bar_w=0.5+rand;
v_x=zeros(k,N,TT);
v_x(:,:,1)=zeros(k,N,1);  
for ttt=2:TT
    for iiii=1:N
%v_x(:,iiii,ttt)=0.5*v_x(:,iiii,ttt-1)+sqrt((1-0.5^(2)))*normrnd(0,sqrt(xi_ev*sig_bar_w),[k,1]); % k by N by TT
v_x(:,iiii,ttt)=0.5*v_x(:,iiii,ttt-1)+sqrt((1-0.5^(2)))*normrnd(0,sqrt(xi_ev),[k,1]); % k by N by TT

    end 
end
bar_v_i=zeros(k,N);
for tttt=1:TT
bar_v_i=bar_v_i+(v_x(:,:,tttt).^2);
end
bar_v_i=bar_v_i/TT; % k by N
bar_v=sum(bar_v_i,2)/N; % k by 1
diff_bar_v=bar_v_i-bar_v*ones(1,N); % k by N
mean_sqr_diff_bar_v=sqrt((sum((diff_bar_v.^2),2)/N)); % k by 1
Xi_b=zeros(k,N);
for kk=1:k
    for iiiii=1:N
Xi_b(kk,iiiii)= (diff_bar_v(kk,iiiii))/(mean_sqr_diff_bar_v(kk));  % k by N
    end
end

beta_i=b'*ones(1,N)+(sqrt(0.4^(2)/12)*rho_b*Xi_b+sqrt(1-rho_b^(2))*eta_rho_bi); % k by N

theta_i=[phi_i;beta_i];   % 1+k by N  


a_i=zeros(N,1);
for a =1:N
a_i(a,:)=A+normrnd(0,(1-phi_i(:,a))); % individual effect for y; N by 1
end

mu1_i=zeros(N,1);
mu2_i=zeros(N,1);
for mu=1:N
mu1_i(mu,:)=Mu1+rho_mu*(a_i(mu,:)-A)+sqrt(1-rho_mu^(2))*normrnd(0,(1-phi_i(:,mu))); % interactive effect for x; N by 1
mu2_i(mu,:)=Mu2+rho_mu*(a_i(mu,:)-A)+sqrt(1-rho_mu^(2))*normrnd(0,(1-phi_i(:,mu))); % interactive effect for x; N by 1
end
mu_i=[mu1_i,mu2_i]; % N by k
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55
fy=zeros(m_y,TT);  % creat a space for saving data factor
fy(:,1)=zeros(m_y,1);   % setting the initial factor
for t=2:TT
   fy(:,t)= 0.5*fy(:,t-1)+sqrt(1-0.5^2)*normrnd(0,1,[m_y,1]); % m_y by TT 
end 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%5555
Gamma0=[0.25, 0.25, -1; 0.5,-1,0.25; 0.5, 0, 0];

Gamma_i=zeros(m_y,m_y,N);
for g=1:N
gamma_1= normrnd(0,1);
gamma_2= normrnd(0,1);
gamma_3= normrnd(0,1);
gamma_11= rho_gamma_1s*gamma_3 +sqrt(1-rho_gamma_1s^2)*normrnd(0,1) ;
gamma_12= rho_gamma_1s*gamma_3 +sqrt(1-rho_gamma_1s^2)*normrnd(0,1) ;
gamma_21= 0.5*gamma_1+sqrt(1-0.5^2)*normrnd(0,1) ;
gamma_22= 0.5*gamma_2+sqrt(1-0.5^2)*normrnd(0,1) ;
Gamma_i(:,:,g)=Gamma0+[gamma_1, gamma_11,gamma_21;gamma_2, gamma_12, gamma_22; gamma_3, 0, 0];
%Gamma_i(:,:,g)=[gamma_1, gamma_11,gamma_21;gamma_2, gamma_12, gamma_22; gamma_3, 0, 0];
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fx=fy(1:m_x,:);  % m_x by TT

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
eta_y=zeros(1,m_y,N);
eta_x=zeros(m_x,m_x,N);
y=zeros(N,TT);
x=zeros(k,N,TT);  
y(:,1)=zeros(N,1);  
x(:,:,1)=zeros(k,N,1);  
 esiption=zeros(N,TT);
for tt=2:TT
    for ii=1:N
     esiption(ii,tt)=(sqrt(xi_es)*sqrt((chi2rnd(2)/2)*(tt/TT))*(chi2rnd(1)-1))/sqrt(2);   
   eta_x(:,:,ii)= [Gamma_i(1,2,ii),Gamma_i(1,3,ii);Gamma_i(2,2,ii),Gamma_i(2,3,ii)];     
  
 x(:,ii,tt)=mu_i(ii,:)'+rho*(x(:,ii,tt-1))+eta_x(:,:,ii)'*fx(:,tt)+v_x(:,ii,tt)+ esiption(ii,tt-1);     % k by N by TT
 
   eta_y(:,:,ii)=[Gamma_i(1,1,ii),Gamma_i(2,1,ii),Gamma_i(3,1,ii)];

  y(ii,tt)= a_i(ii,:)+y(ii,tt-1)*phi_i(:,ii)+x(:,ii,tt)'*beta_i(:,ii)+eta_y(:,:,ii)*fy(:,tt)+ esiption(ii,tt);    % N by TT    
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
F_11=zeros(1,T1);
for tt=1:T1
F_11(:,tt)=sqrt((T0-tt)/(T0-tt+1));
end

F_1=diag(F_11);

F_2=zeros(T1,T0);
for bt=1:T0
 A2=T1:-1:bt;   
 B=-1*A2.^-1;  
 C=diag(B);
F_2= F_2+[zeros(T1,bt) [C;zeros(bt-1,T0-bt)]] ;
end
F_2=[diag(ones(1,T1)) zeros(T1,1)]+F_2;
F=  F_1*F_2;   % T0-1 by T0

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y_NT=y(:,TT-T0:TT); % dicard first 50 time series N by T0+1 
y_NT1=F*y_NT(:,1:T0)';  % y_(i,-1); T by N
y_NT2=F*y_NT(:,2:T0+1)';  % y_(i,T)  ; T by N

fx1=fx(:,TT-T0+2:TT); % m by T0; F_(x)
fx2=fx(:,TT-T0+1:TT-1); % m by T0; F_(x,-1)
fx3=fx(:,TT-T0:TT-2); % m by T0; F_(x,-2)
fx4=fx(:,TT-T0-1:TT-3); % m by T0; F_(x,-3)
fx5=fx(:,TT-T0-2:TT-4); % m by T0; F_(x,-4)
fx6=fx(:,TT-T0-3:TT-5); % m by T0; F_(x,-5)
fx7=fx(:,TT-T0-4:TT-6); % m by T0; F_(x,-5)
fx8=fx(:,TT-T0-5:TT-7); % m by T0; F_(x,-5)

MF_x1=eye(T1)-fx1'*((fx1*fx1')^(-1))*fx1;  % MF_x T0 by T0
MF_x2=eye(T1)-fx2'*((fx2*fx2')^(-1))*fx2;   %MF_x,-1 T0 by T0
MF_x3=eye(T1)-fx3'*((fx3*fx3')^(-1))*fx3;   %MF_x,-2 T0 by T0
MF_x4=eye(T1)-fx4'*((fx4*fx4')^(-1))*fx4;   %MF_x,-3 T0 by T0
MF_x5=eye(T1)-fx5'*((fx5*fx5')^(-1))*fx5;   %MF_x,-4 T0 by T0
MF_x6=eye(T1)-fx6'*((fx6*fx6')^(-1))*fx6;   %MF_x,-5 T0 by T0
MF_x7=eye(T1)-fx7'*((fx7*fx7')^(-1))*fx7;   %MF_x,-5 T0 by T0
MF_x8=eye(T1)-fx8'*((fx8*fx8')^(-1))*fx8;   %MF_x,-5 T0 by T0

x_NT=x(:,:,TT-T0-j:TT); %  dicard first 50 time series for x ; k by N by T0+1+j

x_NT1=zeros(N, T0);
x_NT2=zeros(N, T0);
x_NT1_1=zeros(N, T0);
x_NT2_1=zeros(N, T0);
x_NT1_2=zeros(N, T0);
x_NT2_2=zeros(N, T0);
x_NT1_3=zeros(N, T0);
x_NT2_3=zeros(N, T0);
x_NT1_4=zeros(N, T0);
x_NT2_4=zeros(N, T0);
x_NT1_5=zeros(N, T0);
x_NT2_5=zeros(N, T0);
x_NT1_6=zeros(N, T0);
x_NT2_6=zeros(N, T0);
x_NT1_7=zeros(N, T0);
x_NT2_7=zeros(N, T0);
x_NT1_8=zeros(N, T0);
x_NT2_8=zeros(N, T0);

for it=1:T0
x_NT1(:,it)=x_NT(1,:,it+j+1);   % x_(i,) ; N by T0 
x_NT2(:,it)=x_NT(2,:,it+j+1);   % x_(i,) ; N by T0 
x_NT1_1(:,it)=x_NT(1,:,it+j);   % x_(i,-1) ; N by T0 
x_NT2_1(:,it)=x_NT(2,:,it+j);   % x_(i,-1) ; N by T0 
x_NT1_2(:,it)=x_NT(1,:,it+(j-1));   % x_(i,-2) ; N by T0 
x_NT2_2(:,it)=x_NT(2,:,it+(j-1));   % x_(i,-2) ; N by T0 
x_NT1_3(:,it)=x_NT(1,:,it+(j-2));   % x_(i,-3) ; N by T0 
x_NT2_3(:,it)=x_NT(2,:,it+(j-2));   % x_(i,-3) ; N by T0 
x_NT1_4(:,it)=x_NT(1,:,it+(j-3));   % x_(i,-4) ; N by T0
x_NT2_4(:,it)=x_NT(2,:,it+(j-3));   % x_(i,-4) ; N by T0
x_NT1_5(:,it)=x_NT(1,:,it+(j-4));   % x_(i,-5) ; N by T0
x_NT2_5(:,it)=x_NT(2,:,it+(j-4));   % x_(i,-5) ; N by T0
x_NT1_6(:,it)=x_NT(1,:,it+(j-5));   % x_(i,-6) ; N by T0
x_NT2_6(:,it)=x_NT(2,:,it+(j-5));   % x_(i,-6) ; N by T0
x_NT1_7(:,it)=x_NT(1,:,it+(j-6));   % x_(i,-7) ; N by T0
x_NT2_7(:,it)=x_NT(2,:,it+(j-6));   % x_(i,-7) ; N by T0
x_NT1_8(:,it)=x_NT(1,:,it+(j-7));   % x_(i,-7) ; N by T0
x_NT2_8(:,it)=x_NT(2,:,it+(j-7));   % x_(i,-7) ; N by T0
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
W_it=zeros(N,T1,1+k);  % N by T by 1+k  
W_it(:,:,1)=y_NT1';  %  y_{i,-1}; N by T   
W_it(:,:,2)=(F*x_NT1')'; %  x_(i,T); N by T  
W_it(:,:,3)=(F*x_NT2')'; %  x_(i,T); N by T  

Z_it_1=zeros(N,T1,2*k);    % N by T by 2k
Z_it_1(:,:,1)=(MF_x1*F*x_NT1')';   % N by T 
Z_it_1(:,:,2)=(MF_x1*F*x_NT2')';    % N by T 
Z_it_1(:,:,3)=(MF_x2*F*x_NT1_1')';  % N by T 
Z_it_1(:,:,4)=(MF_x2*F*x_NT2_1')';  % N by T 

Z_it_2=zeros(N,T1,1*k);           % N by T 
Z_it_2(:,:,1)=(MF_x3*F*x_NT1_2')';  % N by T  
Z_it_2(:,:,2)=(MF_x3*F*x_NT2_2')';  % N by T  

Z_it_3=zeros(N,T1,1*k);           % N by T 
Z_it_3(:,:,1)=(MF_x4*F*x_NT1_3')';  % N by T   
Z_it_3(:,:,2)=(MF_x4*F*x_NT2_3')';  % N by T   

Z_it_4=zeros(N,T1,1*k);           % N by T
Z_it_4(:,:,1)=(MF_x5*F*x_NT1_4')';  % N by T
Z_it_4(:,:,2)=(MF_x5*F*x_NT2_4')';  % N by T

Z_it_5=zeros(N,T1,1*k);           % N by T  by 5k
Z_it_5(:,:,1)=(MF_x6*F*x_NT1_5')';  % N by T  
Z_it_5(:,:,2)=(MF_x6*F*x_NT2_5')';  % N by T 

Z_it_6=zeros(N,T1,1*k);           % N by T  by 5k
Z_it_6(:,:,1)=(MF_x7*F*x_NT1_6')';  % N by T  
Z_it_6(:,:,2)=(MF_x7*F*x_NT2_6')';  % N by T 


Z_it_7=zeros(N,T1,1*k);           % N by T  by 5k
Z_it_7(:,:,1)=(MF_x8*F*x_NT1_7')';  % N by T  
Z_it_7(:,:,2)=(MF_x8*F*x_NT2_7')';  % N by T 




Z_it_M=zeros(N,T1,(j+1)*k);           % N by T  by 5k
Z_it_M(:,:,1)=Z_it_1(:,:,1);  % N by T  
Z_it_M(:,:,2)=Z_it_1(:,:,2);  % N by T  ; lag 1
Z_it_M(:,:,3)=Z_it_1(:,:,3);    % N by T     ; lag 2
Z_it_M(:,:,4)=Z_it_1(:,:,4);  % N by T    ; lag 3
Z_it_M(:,:,5)=Z_it_2(:,:,1);  % N by T     ; lag 4
Z_it_M(:,:,6)=Z_it_2(:,:,2);  % N by T     ; lag 5
Z_it_M(:,:,7)=Z_it_3(:,:,1);  % N by T     ; lag 5
Z_it_M(:,:,8)=Z_it_3(:,:,2);  % N by T     ; lag 5
Z_it_M(:,:,9)=Z_it_4(:,:,1);  % N by T     ; lag 5
Z_it_M(:,:,10)=Z_it_4(:,:,2);  % N by T     ; lag 5
Z_it_M(:,:,11)=Z_it_5(:,:,1);  % N by T     ; lag 5
Z_it_M(:,:,12)=Z_it_5(:,:,2);  % N by T     ; lag 5
Z_it_M(:,:,13)=Z_it_6(:,:,1);  % N by T     ; lag 5
Z_it_M(:,:,14)=Z_it_6(:,:,2);  % N by T     ; lag 5
Z_it_M(:,:,15)=Z_it_7(:,:,1);  % N by T     ; lag 5
Z_it_M(:,:,16)=Z_it_7(:,:,2);  % N by T     ; lag 5




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
D=zeros(T1,1+k,N);    
Z_1=zeros(T1,2*k,N);   % lag 1
Z_2=zeros(T1,1*k,N);  % lag 2
Z_3=zeros(T1,1*k,N);  % lag 3
Z_4=zeros(T1,1*k,N);  % lag 4
Z_5=zeros(T1,1*k,N);  % lag 4
Z_6=zeros(T1,1*k,N);  % lag 4
Z_7=zeros(T1,1*k,N);  % lag 4

Z_M2=zeros(T1,3*k,N);  % lag 4
Z_M3=zeros(T1,4*k,N);  % lag 4
Z_M4=zeros(T1,5*k,N);  % lag 4
Z_M5=zeros(T1,6*k,N);  % lag 4
Z_M6=zeros(T1,7*k,N);  % lag 4
Z_M7=zeros(T1,8*k,N);  % lag 4

for iti=1:N
  D(:,:,iti)=[W_it(iti,:,1)', W_it(iti,:,2)', W_it(iti,:,3)']; % T by 1+k
  Z_1(:,:,iti)=[Z_it_1(iti,:,1)', Z_it_1(iti,:,2)', Z_it_1(iti,:,3)', Z_it_1(iti,:,4)'];  %  T by 2k by N  lag 1
  Z_2(:,:,iti)= [Z_it_2(iti,:,1)',Z_it_2(iti,:,2)'];  %  T by k by N lag 2  
  Z_3(:,:,iti)=[Z_it_3(iti,:,1)',Z_it_3(iti,:,2)'];  %  T by k by N lag 3
  Z_4(:,:,iti)= [Z_it_4(iti,:,1)',Z_it_4(iti,:,2)'];  %  T by  k by N lag 4
   Z_5(:,:,iti)= [Z_it_5(iti,:,1)',Z_it_5(iti,:,2)'];  %  T by k by N lag 4
   Z_6(:,:,iti)= [Z_it_6(iti,:,1)',Z_it_6(iti,:,2)'];  %  T by k by N lag 4
   Z_7(:,:,iti)= [Z_it_7(iti,:,1)',Z_it_7(iti,:,2)'];  %  T by k by N lag 4
   
  Z_M2(:,:,iti)=[Z_it_M(iti,:,1)', Z_it_M(iti,:,2)', Z_it_M(iti,:,3)', Z_it_M(iti,:,4)', Z_it_M(iti,:,5)', Z_it_M(iti,:,6)'];  %  T1 by 3k by N lag 4 
  Z_M3(:,:,iti)=[Z_it_M(iti,:,1)', Z_it_M(iti,:,2)', Z_it_M(iti,:,3)', Z_it_M(iti,:,4)', Z_it_M(iti,:,5)', Z_it_M(iti,:,6)', Z_it_M(iti,:,7)', Z_it_M(iti,:,8)'];  %  T1 by 4k by N lag 4
  Z_M4(:,:,iti)=[Z_it_M(iti,:,1)', Z_it_M(iti,:,2)', Z_it_M(iti,:,3)', Z_it_M(iti,:,4)', Z_it_M(iti,:,5)', Z_it_M(iti,:,6)', Z_it_M(iti,:,7)', Z_it_M(iti,:,8)', Z_it_M(iti,:,9)', Z_it_M(iti,:,10)'];  %  T1 by 3k by N lag 4
  Z_M5(:,:,iti)=[Z_it_M(iti,:,1)', Z_it_M(iti,:,2)', Z_it_M(iti,:,3)', Z_it_M(iti,:,4)', Z_it_M(iti,:,5)', Z_it_M(iti,:,6)', Z_it_M(iti,:,7)', Z_it_M(iti,:,8)', Z_it_M(iti,:,9)', Z_it_M(iti,:,10)', Z_it_M(iti,:,11)', Z_it_M(iti,:,12)'];  %  T1 by 3k by N lag 4
 Z_M6(:,:,iti)=[Z_it_M(iti,:,1)', Z_it_M(iti,:,2)', Z_it_M(iti,:,3)', Z_it_M(iti,:,4)', Z_it_M(iti,:,5)', Z_it_M(iti,:,6)', Z_it_M(iti,:,7)', Z_it_M(iti,:,8)', Z_it_M(iti,:,9)', Z_it_M(iti,:,10)', Z_it_M(iti,:,11)', Z_it_M(iti,:,12)', Z_it_M(iti,:,13)', Z_it_M(iti,:,14)'];  %  T1 by 3k by N lag 4
Z_M7(:,:,iti)=[Z_it_M(iti,:,1)', Z_it_M(iti,:,2)', Z_it_M(iti,:,3)', Z_it_M(iti,:,4)', Z_it_M(iti,:,5)', Z_it_M(iti,:,6)', Z_it_M(iti,:,7)', Z_it_M(iti,:,8)', Z_it_M(iti,:,9)', Z_it_M(iti,:,10)', Z_it_M(iti,:,11)', Z_it_M(iti,:,12)', Z_it_M(iti,:,13)', Z_it_M(iti,:,14)',Z_it_M(iti,:,15)', Z_it_M(iti,:,16)'];  %  T1 by 3k by N lag 4;  %  T1 by 3k by N lag 4
end    


P_i_1=zeros(T1,T1,N);
P_i_2=zeros(T1,T1,N);
P_i_3=zeros(T1,T1,N);
P_i_4=zeros(T1,T1,N);
P_i_5=zeros(T1,T1,N);
P_i_6=zeros(T1,T1,N);
P_i_7=zeros(T1,T1,N);

 P_i_M2=zeros(T1,T1,N);
 P_i_M3=zeros(T1,T1,N);
 P_i_M4=zeros(T1,T1,N);
 P_i_M5=zeros(T1,T1,N);
 P_i_M6=zeros(T1,T1,N);
 P_i_M7=zeros(T1,T1,N);
for p=1:N
  P_i_1(:,:,p)=  Z_1(:,:,p)*pinv(Z_1(:,:,p)'*MF_x1*Z_1(:,:,p))*Z_1(:,:,p)' ;% T by T   
  P_i_2(:,:,p)=  Z_2(:,:,p)*pinv(Z_2(:,:,p)'*MF_x1*Z_2(:,:,p))*Z_2(:,:,p)' ;% T by T   
  P_i_3(:,:,p)=  Z_3(:,:,p)*pinv(Z_3(:,:,p)'*MF_x1*Z_3(:,:,p))*Z_3(:,:,p)' ;% T by T 
  P_i_4(:,:,p)=  Z_4(:,:,p)*pinv(Z_4(:,:,p)'*MF_x1*Z_4(:,:,p))*Z_4(:,:,p)' ;% T by T   
 P_i_5(:,:,p)=  Z_5(:,:,p)*pinv(Z_5(:,:,p)'*MF_x1*Z_5(:,:,p))*Z_5(:,:,p)' ;% T by T  
 P_i_6(:,:,p)=  Z_6(:,:,p)*pinv(Z_6(:,:,p)'*MF_x1*Z_6(:,:,p))*Z_6(:,:,p)' ;% T by T  
 P_i_7(:,:,p)=  Z_7(:,:,p)*pinv(Z_7(:,:,p)'*MF_x1*Z_7(:,:,p))*Z_7(:,:,p)' ;% T by T  
 
 
  P_i_M2(:,:,p)=  Z_M2(:,:,p)*pinv(Z_M2(:,:,p)'*MF_x1*Z_M2(:,:,p))*Z_M2(:,:,p)' ;% T by T  
  P_i_M3(:,:,p)=  Z_M3(:,:,p)*pinv(Z_M3(:,:,p)'*MF_x1*Z_M3(:,:,p))*Z_M3(:,:,p)' ;% T by T  
  P_i_M4(:,:,p)=  Z_M4(:,:,p)*pinv(Z_M4(:,:,p)'*MF_x1*Z_M4(:,:,p))*Z_M4(:,:,p)' ;% T by T  
 P_i_M5(:,:,p)=  Z_M5(:,:,p)*pinv(Z_M5(:,:,p)'*MF_x1*Z_M5(:,:,p))*Z_M5(:,:,p)' ;% T by T  
 P_i_M6(:,:,p)=  Z_M6(:,:,p)*pinv(Z_M6(:,:,p)'*MF_x1*Z_M6(:,:,p))*Z_M6(:,:,p)' ;% T by T  
 P_i_M7(:,:,p)=  Z_M7(:,:,p)*pinv(Z_M7(:,:,p)'*MF_x1*Z_M7(:,:,p))*Z_M7(:,:,p)' ;% T by T  
end

first_stage=zeros((1+1)*k,1+k,N);
for fir=1:N
    first_stage(:,:,fir)=pinv( Z_1(:,:,fir)'* Z_1(:,:,fir))*Z_1(:,:,fir)'*D(:,:,fir) ; % (j+1)k by (1+k)
end

V_2=zeros(T1,1+k,N);
for V_i_2=1:N
%  V_2(:,:,V_i_2)= D(:,:,V_i_2)- Z_1(:,:,V_i_2)*pinv(Z_1(:,:,V_i_2)'*Z_1(:,:,V_i_2))*Z_1(:,:,V_i_2)'*D(:,:,V_i_2);  %T by 1+k ; first stage residual
 V_2(:,:,V_i_2)= D(:,:,V_i_2)- Z_1(:,:,V_i_2)*first_stage(:,:,V_i_2);   % T1 by 1+k by N
end

%H1=zeros(T1,1+k,N);

H=zeros(1+k,1+k,N);

for hi=1:N
%H1(:,:,hi)= [Z_1(:,1,hi), Z_1(:,2,hi),(Z_1(:,3,hi)+Z_1(:,4,hi))/2 ];  
%H1(:,:,hi)= [Z_1(:,1,hi), Z_1(:,2,hi),Z_1(:,4,hi) ];  
%H(:,:,hi)= Z_1(:,:,hi)'* Z_1(:,:,hi)/T0; % 2k by 2k
H(:,:,hi)= first_stage(:,:,hi)'* first_stage(:,:,hi)/T1; % 1+k by 1+k
end 

ini_theta_IV_1=zeros(1+k,N); 
for iii1=1:N
ini_theta_IV_1(:,iii1)= pinv(D(:,:,iii1)'*MF_x1*P_i_1(:,:,iii1)*MF_x1*D(:,:,iii1))*D(:,:,iii1)'*MF_x1*P_i_1(:,:,iii1)*MF_x1*y_NT2(:,iii1);
end

%fist_stage_theta=zeros(2,N); 
%for iii2=1:N
%fist_stage_theta(:,iii2)= pinv(H2(:,:,iii2)'*H2(:,:,iii2))*H2(:,:,iii1)'*y_NT1(:,iii2);
%end


hat_u=zeros(T1,N);
for hat_i=1:N
hat_u(:,hat_i)= y_NT2(:,hat_i)- D(:,:,hat_i)*ini_theta_IV_1(:,hat_i);  % T by N; preliminary residual 
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
V_2=zeros(T1,1+k,N);
for V_i_2=1:N
  V_2(:,:,V_i_2)= D(:,:,V_i_2)- Z_1(:,:,V_i_2)*pinv(Z_1(:,:,V_i_2)'*Z_1(:,:,V_i_2))*Z_1(:,:,V_i_2)'*D(:,:,V_i_2);  %T by 1+k ; first stage residual
end


etai=rand(1+k,1)*ones(1,N);
hat_v_eta_2=zeros(T1,1,N);
for v_i_2=1:N
    hat_v_eta_2(:,:,v_i_2)=  V_2(:,:,v_i_2)*pinv(H(:,:,v_i_2))*etai(:,v_i_2);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Sigma_u_i=zeros(1,N);
Sigma_eta_i=zeros(1,N);
sigma_etau=zeros(1,N);

V_eta_M2_1=zeros(T1,N);   %
V_eta_M2_2=zeros(T1,N);  %

V_eta_M3_1=zeros(T1,N);   %
V_eta_M3_2=zeros(T1,N);  %
V_eta_M3_3=zeros(T1,N);   %



V_eta_M4_1=zeros(T1,N);   %
V_eta_M4_2=zeros(T1,N);  %
V_eta_M4_3=zeros(T1,N);   %
V_eta_M4_4=zeros(T1,N);  %

V_eta_M5_1=zeros(T1,N);   %
V_eta_M5_2=zeros(T1,N);  %
V_eta_M5_3=zeros(T1,N);   %
V_eta_M5_4=zeros(T1,N);  %
V_eta_M5_5=zeros(T1,N);  %

V_eta_M6_1=zeros(T1,N);   %
V_eta_M6_2=zeros(T1,N);  %
V_eta_M6_3=zeros(T1,N);   %
V_eta_M6_4=zeros(T1,N);  %
V_eta_M6_5=zeros(T1,N);  %
V_eta_M6_6=zeros(T1,N);  %

V_eta_M7_1=zeros(T1,N);   %
V_eta_M7_2=zeros(T1,N);  %
V_eta_M7_3=zeros(T1,N);   %
V_eta_M7_4=zeros(T1,N);  %
V_eta_M7_5=zeros(T1,N);  %
V_eta_M7_6=zeros(T1,N);  %
V_eta_M7_7=zeros(T1,N);  %

hat_U2=zeros(2,2,N);
hat_U3=zeros(3,3,N);
hat_U4=zeros(4,4,N);
hat_U5=zeros(5,5,N);
hat_U6=zeros(6,6,N);
hat_U7=zeros(j,j,N);

for sig=1:N
 Sigma_u_i(:,sig)=hat_u(:,sig)'*hat_u(:,sig)/T1;
 Sigma_eta_i(:,sig)= hat_v_eta_2(:,:,sig)'*hat_v_eta_2(:,:,sig)/T1;
 sigma_etau(:,sig)= hat_v_eta_2(:,:,sig)'*hat_u(:,sig)/T1;
 
  V_eta_M2_1(:,sig)=(P_i_M2(:,:,sig)-P_i_1(:,:,sig))* D(:,:,sig)*pinv(H(:,:,sig))*etai(:,sig);   % T1 by 1
 V_eta_M2_2(:,sig)=(P_i_M2(:,:,sig)-P_i_2(:,:,sig))* D(:,:,sig)*pinv(H(:,:,sig))*etai(:,sig);
 
 V_eta_M3_1(:,sig)=(P_i_M3(:,:,sig)-P_i_1(:,:,sig))* D(:,:,sig)*pinv(H(:,:,sig))*etai(:,sig);   % T1 by 1
 V_eta_M3_2(:,sig)=(P_i_M3(:,:,sig)-P_i_2(:,:,sig))* D(:,:,sig)*pinv(H(:,:,sig))*etai(:,sig);
 V_eta_M3_3(:,sig)=(P_i_M3(:,:,sig)-P_i_3(:,:,sig))* D(:,:,sig)*pinv(H(:,:,sig))*etai(:,sig);
 
 V_eta_M4_1(:,sig)=(P_i_M4(:,:,sig)-P_i_1(:,:,sig))* D(:,:,sig)*pinv(H(:,:,sig))*etai(:,sig);   % T1 by 1
 V_eta_M4_2(:,sig)=(P_i_M4(:,:,sig)-P_i_2(:,:,sig))* D(:,:,sig)*pinv(H(:,:,sig))*etai(:,sig);
 V_eta_M4_3(:,sig)=(P_i_M4(:,:,sig)-P_i_3(:,:,sig))* D(:,:,sig)*pinv(H(:,:,sig))*etai(:,sig);
 V_eta_M4_4(:,sig)=(P_i_M4(:,:,sig)-P_i_4(:,:,sig))* D(:,:,sig)*pinv(H(:,:,sig))*etai(:,sig);
 
 V_eta_M5_1(:,sig)=(P_i_M5(:,:,sig)-P_i_1(:,:,sig))* D(:,:,sig)*pinv(H(:,:,sig))*etai(:,sig);   % T1 by 1
 V_eta_M5_2(:,sig)=(P_i_M5(:,:,sig)-P_i_2(:,:,sig))* D(:,:,sig)*pinv(H(:,:,sig))*etai(:,sig);
 V_eta_M5_3(:,sig)=(P_i_M5(:,:,sig)-P_i_3(:,:,sig))* D(:,:,sig)*pinv(H(:,:,sig))*etai(:,sig);
 V_eta_M5_4(:,sig)=(P_i_M5(:,:,sig)-P_i_4(:,:,sig))* D(:,:,sig)*pinv(H(:,:,sig))*etai(:,sig);
 V_eta_M5_5(:,sig)=(P_i_M5(:,:,sig)-P_i_5(:,:,sig))* D(:,:,sig)*pinv(H(:,:,sig))*etai(:,sig);
 
 
 V_eta_M6_1(:,sig)=(P_i_M6(:,:,sig)-P_i_1(:,:,sig))* D(:,:,sig)*pinv(H(:,:,sig))*etai(:,sig);   % T1 by 1
 V_eta_M6_2(:,sig)=(P_i_M6(:,:,sig)-P_i_2(:,:,sig))* D(:,:,sig)*pinv(H(:,:,sig))*etai(:,sig);
 V_eta_M6_3(:,sig)=(P_i_M6(:,:,sig)-P_i_3(:,:,sig))* D(:,:,sig)*pinv(H(:,:,sig))*etai(:,sig);
 V_eta_M6_4(:,sig)=(P_i_M6(:,:,sig)-P_i_4(:,:,sig))* D(:,:,sig)*pinv(H(:,:,sig))*etai(:,sig);
 V_eta_M6_5(:,sig)=(P_i_M6(:,:,sig)-P_i_5(:,:,sig))* D(:,:,sig)*pinv(H(:,:,sig))*etai(:,sig);
  V_eta_M6_6(:,sig)=(P_i_M6(:,:,sig)-P_i_6(:,:,sig))* D(:,:,sig)*pinv(H(:,:,sig))*etai(:,sig);
 

  V_eta_M7_1(:,sig)=(P_i_M7(:,:,sig)-P_i_1(:,:,sig))* D(:,:,sig)*pinv(H(:,:,sig))*etai(:,sig);   % T1 by 1
 V_eta_M7_2(:,sig)=(P_i_M7(:,:,sig)-P_i_2(:,:,sig))* D(:,:,sig)*pinv(H(:,:,sig))*etai(:,sig);
 V_eta_M7_3(:,sig)=(P_i_M7(:,:,sig)-P_i_3(:,:,sig))* D(:,:,sig)*pinv(H(:,:,sig))*etai(:,sig);
 V_eta_M7_4(:,sig)=(P_i_M7(:,:,sig)-P_i_4(:,:,sig))* D(:,:,sig)*pinv(H(:,:,sig))*etai(:,sig);
 V_eta_M7_5(:,sig)=(P_i_M7(:,:,sig)-P_i_5(:,:,sig))* D(:,:,sig)*pinv(H(:,:,sig))*etai(:,sig);
  V_eta_M7_6(:,sig)=(P_i_M7(:,:,sig)-P_i_6(:,:,sig))* D(:,:,sig)*pinv(H(:,:,sig))*etai(:,sig);
   V_eta_M7_7(:,sig)=(P_i_M7(:,:,sig)-P_i_7(:,:,sig))* D(:,:,sig)*pinv(H(:,:,sig))*etai(:,sig);
  
 hat_U2(:,:,sig)=[V_eta_M2_1(:,sig),V_eta_M2_2(:,sig)]'*[ V_eta_M2_1(:,sig),V_eta_M2_2(:,sig)];
 hat_U3(:,:,sig)=[V_eta_M3_1(:,sig),V_eta_M3_2(:,sig),V_eta_M3_3(:,sig)]'*[ V_eta_M3_1(:,sig),V_eta_M3_2(:,sig),V_eta_M3_3(:,sig)];
 hat_U4(:,:,sig)=[V_eta_M4_1(:,sig),V_eta_M4_2(:,sig),V_eta_M4_3(:,sig),V_eta_M4_4(:,sig)]'*[ V_eta_M4_1(:,sig),V_eta_M4_2(:,sig),V_eta_M4_3(:,sig),V_eta_M4_4(:,sig)];
 hat_U5(:,:,sig)=[V_eta_M5_1(:,sig),V_eta_M5_2(:,sig),V_eta_M5_3(:,sig),V_eta_M5_4(:,sig),V_eta_M5_5(:,sig)]'*[ V_eta_M5_1(:,sig),V_eta_M5_2(:,sig),V_eta_M5_3(:,sig),V_eta_M5_4(:,sig),V_eta_M5_5(:,sig)];
 hat_U6(:,:,sig)=[V_eta_M6_1(:,sig),V_eta_M6_2(:,sig),V_eta_M6_3(:,sig),V_eta_M6_4(:,sig),V_eta_M6_5(:,sig),V_eta_M6_6(:,sig)]'*[ V_eta_M6_1(:,sig),V_eta_M6_2(:,sig),V_eta_M6_3(:,sig),V_eta_M6_4(:,sig),V_eta_M6_5(:,sig),V_eta_M6_6(:,sig)];
 hat_U7(:,:,sig)=[V_eta_M7_1(:,sig),V_eta_M7_2(:,sig),V_eta_M7_3(:,sig),V_eta_M7_4(:,sig),V_eta_M7_5(:,sig),V_eta_M7_6(:,sig),V_eta_M7_7(:,sig)]'*[ V_eta_M7_1(:,sig),V_eta_M7_2(:,sig),V_eta_M7_3(:,sig),V_eta_M7_4(:,sig),V_eta_M7_5(:,sig),V_eta_M7_6(:,sig),V_eta_M7_7(:,sig)];

end

Gamma_2=[1,1;1,2];
Gamma_3=[1,1,1;1,2,2;1,2,3];
Gamma_4=[1,1,1,1;1,2,2,2;1,2,3,3;1,2,3,4];
Gamma_5=[1,1,1,1,1;1,2,2,2,2;1,2,3,3,3;1,2,3,4,4;1,2,3,4,5];
Gamma_6=[1,1,1,1,1,1;1,2,2,2,2,2;1,2,3,3,3,3;1,2,3,4,4,4;1,2,3,4,5,5;1,2,3,4,5,6];
Gamma_7=[1,1,1,1,1,1,1;1,2,2,2,2,2,2;1,2,3,3,3,3,3;1,2,3,4,4,4,4;1,2,3,4,5,5,5;1,2,3,4,5,6,6;1,2,3,4,5,6,7];

K_2=ones(2,1);
K_3=ones(3,1);
K_4=ones(4,1);
K_5=ones(5,1);
K_6=ones(6,1);
K_7=ones(j,1);


j2=2;
 j3=3;
 j4=4;
 j5=5;
 j6=6;
 j7=7;
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for ome_i_2=1:N
    omega_ini_2 = rand(j2,1);
    lq_2 = [zeros(j2,1)];
    uq_2 = [ones(j2,1)];
    
    sigma_etau_i=sigma_etau(:,ome_i_2); 
    Sigma_u_i_i=Sigma_u_i(:,ome_i_2);
    Sigma_eta_i_i=Sigma_eta_i(:,ome_i_2);
    hat_U_i_2=hat_U2(:,:,ome_i_2);    
    [omega_2, fval_2, exitflag_2, output_2, lambda_2, hessian_2] = MAIV_opt_2(T1,K_2,sigma_etau_i,Sigma_u_i_i,Sigma_eta_i_i, hat_U_i_2,Gamma_2,j2,omega_ini_2,lq_2,uq_2);
 omega_IV_i_2(:,ome_i_2,sml)=omega_2;
 fval2(ome_i_2,sml)=fval_2;
end


for ome_i_3=1:N
    omega_ini_3 = rand(j3,1);
    lq_3 = [zeros(j3,1)];
    uq_3 = [ones(j3,1)];
   
    sigma_etau_i=sigma_etau(:,ome_i_3); 
    Sigma_u_i_i=Sigma_u_i(:,ome_i_3);
    Sigma_eta_i_i=Sigma_eta_i(:,ome_i_3);
    hat_U_i_3=hat_U3(:,:,ome_i_3);    
    [omega_3, fval_3, exitflag_3, output_3, lambda_3, hessian_3] = MAIV_opt_3(T1,K_3,sigma_etau_i,Sigma_u_i_i,Sigma_eta_i_i, hat_U_i_3,Gamma_3,j3,omega_ini_3,lq_3,uq_3);
 omega_IV_i_3(:,ome_i_3,sml)=omega_3;
 fval3(ome_i_3,sml)=fval_3;
end


for ome_i=1:N
    omega_ini = rand(j4,1);
    lq = [zeros(j4,1)];
    uq = [ones(j4,1)];
    sigma_etau_i=sigma_etau(:,ome_i); 
    Sigma_u_i_i=Sigma_u_i(:,ome_i);
    Sigma_eta_i_i=Sigma_eta_i(:,ome_i);
    hat_U_i_4=hat_U4(:,:,ome_i);    
    [omega, fval, exitflag, output, lambda, hessian] = MAIV_opt(T1,K_4,sigma_etau_i,Sigma_u_i_i,Sigma_eta_i_i, hat_U_i_4,Gamma_4,j4,omega_ini,lq,uq);
 omega_IV_i_4(:,ome_i,sml)=omega;
 fval4(ome_i,sml)=fval;
end




for ome_i_5=1:N
    omega_ini_5 = rand(j5,1);
    lq_5 = [zeros(j5,1)];
    uq_5 = [ones(j5,1)];
    sigma_etau_i=sigma_etau(:,ome_i_5); 
    Sigma_u_i_i=Sigma_u_i(:,ome_i_5);
    Sigma_eta_i_i=Sigma_eta_i(:,ome_i_5);
    hat_U_i_5=hat_U5(:,:,ome_i_5);    
    [omega_5, fval_5, exitflag_5, output_5, lambda_5, hessian_5] = MAIV_opt_5(T1,K_5,sigma_etau_i,Sigma_u_i_i,Sigma_eta_i_i, hat_U_i_5,Gamma_5,j5,omega_ini_5,lq_5,uq_5);
 omega_IV_i_5(:,ome_i_5,sml)=omega_5;
 fval5(ome_i_5,sml)=fval_5;
end

for ome_i_6=1:N
    omega_ini_6 = rand(j6,1);
    lq_6 = [zeros(j6,1)];
    uq_6 = [ones(j6,1)];
    sigma_etau_i=sigma_etau(:,ome_i_6); 
    Sigma_u_i_i=Sigma_u_i(:,ome_i_6);
    Sigma_eta_i_i=Sigma_eta_i(:,ome_i_6);
    hat_U_i_6=hat_U6(:,:,ome_i_6);    
    [omega_6, fval_6, exitflag_6, output_6, lambda_6, hessian_6] = MAIV_opt_6(T1,K_6,sigma_etau_i,Sigma_u_i_i,Sigma_eta_i_i, hat_U_i_6,Gamma_6,j6,omega_ini_6,lq_6,uq_6);
 omega_IV_i_6(:,ome_i_6,sml)=omega_6;
 fval6(ome_i_6,sml)=fval_6;
end


for ome_i_7=1:N
    omega_ini_7 = rand(j7,1);
    lq_7 = [zeros(j7,1)];
    uq_7 = [ones(j7,1)];
    sigma_etau_i=sigma_etau(:,ome_i_7); 
    Sigma_u_i_i=Sigma_u_i(:,ome_i_7);
    Sigma_eta_i_i=Sigma_eta_i(:,ome_i_7);
    hat_U_i_7=hat_U7(:,:,ome_i_7);    
    [omega_7, fval_7, exitflag_7, output_7, lambda_7, hessian_7] = MAIV_opt_7(T1,K_7,sigma_etau_i,Sigma_u_i_i,Sigma_eta_i_i, hat_U_i_7,Gamma_7,j7,omega_ini_7,lq_7,uq_7);
 omega_IV_i_7(:,ome_i_7,sml)=omega_7;
 fval7(ome_i_7,sml)=fval_7;
end





P_2=zeros(T1,T1,N);
for pi_2=1:N
P_2(:,:,pi_2)= omega_IV_i_2(1,pi_2,sml)*P_i_1(:,:,pi_2)+omega_IV_i_2(2,pi_2,sml)*P_i_2(:,:,pi_2);
end

P_3=zeros(T1,T1,N);
for pi_3=1:N
P_3(:,:,pi_3)= omega_IV_i_3(1,pi_3,sml)*P_i_1(:,:,pi_3)+omega_IV_i_3(2,pi_3,sml)*P_i_2(:,:,pi_3)+omega_IV_i_3(3,pi_3,sml)*P_i_3(:,:,pi_3);
end

P_4=zeros(T1,T1,N);
for pi=1:N
P_4(:,:,pi)= omega_IV_i_4(1,pi,sml)*P_i_1(:,:,pi)+omega_IV_i_4(2,pi,sml)*P_i_2(:,:,pi)+omega_IV_i_4(3,pi,sml)*P_i_3(:,:,pi)+omega_IV_i_4(4,pi,sml)*P_i_4(:,:,pi) ;
end


P_5=zeros(T1,T1,N);
for pi_5=1:N
P_5(:,:,pi_5)= omega_IV_i_5(1,pi_5,sml)*P_i_1(:,:,pi_5)+omega_IV_i_5(2,pi_5,sml)*P_i_2(:,:,pi_5)+omega_IV_i_5(3,pi_5,sml)*P_i_3(:,:,pi_5)+omega_IV_i_5(4,pi_5,sml)*P_i_4(:,:,pi_5)+omega_IV_i_5(5,pi_5,sml).*P_i_5(:,:,pi_5) ;
end

P_6=zeros(T1,T1,N);
for pi_6=1:N
P_6(:,:,pi_6)= omega_IV_i_6(1,pi_6,sml)*P_i_1(:,:,pi_6)+omega_IV_i_6(2,pi_6,sml)*P_i_2(:,:,pi_6)+omega_IV_i_6(3,pi_6,sml)*P_i_3(:,:,pi_6)+omega_IV_i_6(4,pi_6,sml)*P_i_4(:,:,pi_6)+omega_IV_i_6(5,pi_5,sml).*P_i_5(:,:,pi_6)+omega_IV_i_6(6,pi_5,sml).*P_i_6(:,:,pi_6) ;
end

P_7=zeros(T1,T1,N);
for pi_7=1:N
P_7(:,:,pi_7)= omega_IV_i_7(1,pi_7,sml)*P_i_1(:,:,pi_7)+omega_IV_i_7(2,pi_7,sml)*P_i_2(:,:,pi_7)+omega_IV_i_7(3,pi_7,sml)*P_i_3(:,:,pi_7)+omega_IV_i_7(4,pi_7,sml)*P_i_4(:,:,pi_7)+omega_IV_i_7(5,pi_7,sml).*P_i_5(:,:,pi_7)+omega_IV_i_7(6,pi_7,sml).*P_i_6(:,:,pi_7)+omega_IV_i_7(7,pi_7,sml).*P_i_7(:,:,pi_7) ;
end


opt_theta_IV_i_2=zeros(1+k,N);
for ome__2=1:N
opt_theta_IV_i_2(:,ome__2)= pinv(D(:,:,ome__2)'*MF_x1*P_2(:,:,ome__2)*MF_x1*D(:,:,ome__2))*D(:,:,ome__2)'*MF_x1*P_2(:,:,ome__2)*MF_x1*y_NT2(:,ome__2) ;
end

opt_theta_IV_i_3=zeros(1+k,N);
for ome__3=1:N
opt_theta_IV_i_3(:,ome__3)= pinv(D(:,:,ome__3)'*MF_x1*P_3(:,:,ome__3)*MF_x1*D(:,:,ome__3))*D(:,:,ome__3)'*MF_x1*P_3(:,:,ome__3)*MF_x1*y_NT2(:,ome__3) ;
end


opt_theta_IV_i_4=zeros(1+k,N);
for ome__4=1:N
opt_theta_IV_i_4(:,ome__4)= pinv(D(:,:,ome__4)'*MF_x1*P_4(:,:,ome__4)*MF_x1*D(:,:,ome__4))*D(:,:,ome__4)'*MF_x1*P_4(:,:,ome__4)*MF_x1*y_NT2(:,ome__4) ;
end


opt_theta_IV_i_5=zeros(1+k,N);
for ome__5=1:N
opt_theta_IV_i_5(:,ome__5)= pinv(D(:,:,ome__5)'*MF_x1*P_5(:,:,ome__5)*MF_x1*D(:,:,ome__5))*D(:,:,ome__5)'*MF_x1*P_5(:,:,ome__5)*MF_x1*y_NT2(:,ome__5) ;
end

opt_theta_IV_i_6=zeros(1+k,N);
for ome__6=1:N
opt_theta_IV_i_6(:,ome__6)= pinv(D(:,:,ome__6)'*MF_x1*P_6(:,:,ome__6)*MF_x1*D(:,:,ome__6))*D(:,:,ome__6)'*MF_x1*P_6(:,:,ome__6)*MF_x1*y_NT2(:,ome__6) ;
end

opt_theta_IV_i_7=zeros(1+k,N);
for ome__7=1:N
opt_theta_IV_i_7(:,ome__7)= pinv(D(:,:,ome__7)'*MF_x1*P_7(:,:,ome__7)*MF_x1*D(:,:,ome__7))*D(:,:,ome__7)'*MF_x1*P_7(:,:,ome__7)*MF_x1*y_NT2(:,ome__7) ;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ini_IVMG_1(:,sml)=nanmean(ini_theta_IV_1,2); 

opt_IVMG_2(:,sml)=nanmean(opt_theta_IV_i_2,2); % LS_MG
opt_IVMG_3(:,sml)=nanmean(opt_theta_IV_i_3,2); % LS_MG
opt_IVMG_4(:,sml)=nanmean(opt_theta_IV_i_4,2); % LS_MG
opt_IVMG_5(:,sml)=nanmean(opt_theta_IV_i_5,2); % LS_MG
opt_IVMG_6(:,sml)=nanmean(opt_theta_IV_i_6,2); % LS_MG
opt_IVMG_7(:,sml)=nanmean(opt_theta_IV_i_7,2); % LS_MG

sml=sml+1;
end

ome_2=zeros(2,N);
ome_3=zeros(3,N);
ome_4=zeros(4,N);
ome_5=zeros(5,N);
ome_6=zeros(6,N);
ome_7=zeros(7,N);
fva2=zeros(N,1);
fva3=zeros(N,1);
fva4=zeros(N,1);
fva5=zeros(N,1);
fva6=zeros(N,1);
fva7=zeros(N,1);
for rep1=1:rep
ome_2=ome_2+omega_IV_i_2(:,:,rep1);
ome_3=ome_3+omega_IV_i_3(:,:,rep1);
ome_4=ome_4+omega_IV_i_4(:,:,rep1);
ome_5=ome_5+omega_IV_i_5(:,:,rep1);
ome_6=ome_6+omega_IV_i_6(:,:,rep1);
ome_7=ome_7+omega_IV_i_7(:,:,rep1);

fva2= fva2 +fval2(:,rep1);
fva3=fva3 +fval3(:,rep1);
fva4=fva4 +fval4(:,rep1);
fva5=fva5 +fval5(:,rep1);
fva6=fva6 +fval6(:,rep1);
fva7=fva7 +fval7(:,rep1);
end
ome2=nanmean(ome_2,2)/rep;
ome3=nanmean(ome_3,2)/rep;
ome4=nanmean(ome_4,2)/rep;
ome5=nanmean(ome_5,2)/rep;
ome6=nanmean(ome_6,2)/rep;
ome7=nanmean(ome_7,2)/rep;
fv2=nanmean(fva2,1);
fv3=nanmean(fva3,1);
fv4=nanmean(fva4,1);
fv5=nanmean(fva5,1);
fv6=nanmean(fva6,1);
fv7=nanmean(fva7,1);


ini_mean_phi_1= nanmean(ini_IVMG_1(1,:));
ini_mean_beta1_1= nanmean(ini_IVMG_1(2,:));

opt_mean_phi_2= nanmean(opt_IVMG_2(1,:));
opt_mean_beta1_2= nanmean(opt_IVMG_2(2,:));

opt_mean_phi_3= nanmean(opt_IVMG_3(1,:));
opt_mean_beta1_3= nanmean(opt_IVMG_3(2,:));


opt_mean_phi_4= nanmean(opt_IVMG_4(1,:));
opt_mean_beta1_4= nanmean(opt_IVMG_4(2,:));

opt_mean_phi_5= nanmean(opt_IVMG_5(1,:));
opt_mean_beta1_5= nanmean(opt_IVMG_5(2,:));

opt_mean_phi_6= nanmean(opt_IVMG_6(1,:));
opt_mean_beta1_6= nanmean(opt_IVMG_6(2,:));

opt_mean_phi_7= nanmean(opt_IVMG_7(1,:));
opt_mean_beta1_7= nanmean(opt_IVMG_7(2,:));



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ini_bias_mean_phi_1(idx_T, idx_N, idx_phi) = ini_mean_phi_1 - phi;
ini_bias_mean_beta1_1(idx_T, idx_N, idx_phi)=ini_mean_beta1_1 - b1;


opt_bias_mean_phi_2(idx_T, idx_N, idx_phi) =opt_mean_phi_2 - phi;
opt_bias_mean_beta1_2(idx_T, idx_N, idx_phi)=opt_mean_beta1_2 - b1;

opt_bias_mean_phi_3(idx_T, idx_N, idx_phi) =opt_mean_phi_3 - phi;
opt_bias_mean_beta1_3(idx_T, idx_N, idx_phi)=opt_mean_beta1_3 - b1;

opt_bias_mean_phi_4(idx_T, idx_N, idx_phi) =opt_mean_phi_4 - phi;
opt_bias_mean_beta1_4(idx_T, idx_N, idx_phi)=opt_mean_beta1_4 - b1;


opt_bias_mean_phi_5(idx_T, idx_N, idx_phi) =opt_mean_phi_5 - phi;
opt_bias_mean_beta1_5(idx_T, idx_N, idx_phi)=opt_mean_beta1_5 - b1;

opt_bias_mean_phi_6(idx_T, idx_N, idx_phi) =opt_mean_phi_6 - phi;
opt_bias_mean_beta1_6(idx_T, idx_N, idx_phi)=opt_mean_beta1_6 - b1;

opt_bias_mean_phi_7(idx_T, idx_N, idx_phi) =opt_mean_phi_7 - phi;
opt_bias_mean_beta1_7(idx_T, idx_N, idx_phi)=opt_mean_beta1_7 - b1;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


ini_std_phi_1(idx_T, idx_N, idx_phi) = nanstd(ini_IVMG_1(1,:));
ini_std_beta1_1(idx_T, idx_N, idx_phi) = nanstd(ini_IVMG_1(2,:));

opt_std_phi_2(idx_T, idx_N, idx_phi) = nanstd(opt_IVMG_2(1,:));
opt_std_beta1_2(idx_T, idx_N, idx_phi) = nanstd(opt_IVMG_2(2,:));

opt_std_phi_3(idx_T, idx_N, idx_phi) = nanstd(opt_IVMG_3(1,:));
opt_std_beta1_3(idx_T, idx_N, idx_phi) = nanstd(opt_IVMG_3(2,:));


opt_std_phi_4(idx_T, idx_N, idx_phi) = nanstd(opt_IVMG_4(1,:));
opt_std_beta1_4(idx_T, idx_N, idx_phi) = nanstd(opt_IVMG_4(2,:));

opt_std_phi_5(idx_T, idx_N, idx_phi) = nanstd(opt_IVMG_5(1,:));
opt_std_beta1_5(idx_T, idx_N, idx_phi) = nanstd(opt_IVMG_5(2,:));

opt_std_phi_6(idx_T, idx_N, idx_phi) = nanstd(opt_IVMG_6(1,:));
opt_std_beta1_6(idx_T, idx_N, idx_phi) = nanstd(opt_IVMG_6(2,:));

opt_std_phi_7(idx_T, idx_N, idx_phi) = nanstd(opt_IVMG_7(1,:));
opt_std_beta1_7(idx_T, idx_N, idx_phi) = nanstd(opt_IVMG_7(2,:));



ini_rmse_phi_1(idx_T, idx_N,idx_phi) = sqrt( nanmean( (ini_IVMG_1(1,:)-phi).^2) ); 
ini_rmse_beta1_1(idx_T, idx_N, idx_phi) = sqrt( nanmean( (ini_IVMG_1(2,:)-b1).^2) ); 


opt_rmse_phi_2(idx_T, idx_N,idx_phi) = sqrt( nanmean( (opt_IVMG_2(1,:)-phi).^2) ); 
opt_rmse_beta1_2(idx_T, idx_N, idx_phi) = sqrt( nanmean( (opt_IVMG_2(2,:)-b1).^2) ); 

opt_rmse_phi_3(idx_T, idx_N,idx_phi) = sqrt( nanmean( (opt_IVMG_3(1,:)-phi).^2) ); 
opt_rmse_beta1_3(idx_T, idx_N, idx_phi) = sqrt( nanmean( (opt_IVMG_3(2,:)-b1).^2) ); 

opt_rmse_phi_4(idx_T, idx_N,idx_phi) = sqrt( nanmean( (opt_IVMG_4(1,:)-phi).^2) ); 
opt_rmse_beta1_4(idx_T, idx_N, idx_phi) = sqrt( nanmean( (opt_IVMG_4(2,:)-b1).^2) ); 

opt_rmse_phi_5(idx_T, idx_N,idx_phi) = sqrt( nanmean( (opt_IVMG_5(1,:)-phi).^2) ); 
opt_rmse_beta1_5(idx_T, idx_N, idx_phi) = sqrt( nanmean( (opt_IVMG_5(2,:)-b1).^2) ); 

opt_rmse_phi_6(idx_T, idx_N,idx_phi) = sqrt( nanmean( (opt_IVMG_6(1,:)-phi).^2) ); 
opt_rmse_beta1_6(idx_T, idx_N, idx_phi) = sqrt( nanmean( (opt_IVMG_6(2,:)-b1).^2) ); 

opt_rmse_phi_7(idx_T, idx_N,idx_phi) = sqrt( nanmean( (opt_IVMG_7(1,:)-phi).^2) ); 
opt_rmse_beta1_7(idx_T, idx_N, idx_phi) = sqrt( nanmean( (opt_IVMG_7(2,:)-b1).^2) ); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ini_mae_phi_1(idx_T, idx_N,idx_phi) = median((ini_IVMG_1(1,:)-phi).^2);
ini_mae_beta1_1(idx_T, idx_N, idx_phi) = median((ini_IVMG_1(2,:)-b1).^2);

opt_mae_phi_2(idx_T, idx_N,idx_phi) = median((opt_IVMG_2(1,:)-phi).^2);
opt_mae_beta1_2(idx_T, idx_N, idx_phi) = median((opt_IVMG_2(2,:)-b1).^2);

opt_mae_phi_3(idx_T, idx_N,idx_phi) = median((opt_IVMG_3(1,:)-phi).^2);
opt_mae_beta1_3(idx_T, idx_N, idx_phi) = median((opt_IVMG_3(2,:)-b1).^2);


opt_mae_phi_4(idx_T, idx_N,idx_phi) = median((opt_IVMG_4(1,:)-phi).^2);
opt_mae_beta1_4(idx_T, idx_N, idx_phi) = median((opt_IVMG_4(2,:)-b1).^2);

opt_mae_phi_5(idx_T, idx_N,idx_phi) = median((opt_IVMG_5(1,:)-phi).^2);
opt_mae_beta1_5(idx_T, idx_N, idx_phi) = median((opt_IVMG_5(2,:)-b1).^2);

opt_mae_phi_6(idx_T, idx_N,idx_phi) = median((opt_IVMG_6(1,:)-phi).^2);
opt_mae_beta1_6(idx_T, idx_N, idx_phi) = median((opt_IVMG_6(2,:)-b1).^2);

opt_mae_phi_7(idx_T, idx_N,idx_phi) = median((opt_IVMG_7(1,:)-phi).^2);
opt_mae_beta1_7(idx_T, idx_N, idx_phi) = median((opt_IVMG_7(2,:)-b1).^2);


end
end
end
toc
