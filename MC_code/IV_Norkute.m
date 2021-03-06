rep = 250;  
list_T = [25 ]; 
list_N = [25 ];  
list_phi= [0.5]; 
b1=3;
b2=1;
b=[b1,b2];
rho=[0];
pi_u=0.75;
SNR=4;
Mu1=1;
Mu2=-0.5;
A=0.5;
rho_gamma_1s=0.5;
k=2;   % number of regressors
m_x=2; % number of factors of regressor
my=1;
m_y=m_x+my; % number of factors of y
xi_es=sqrt((pi_u/(1-pi_u))*m_y);

rho_mu=0.5;
rho_v=0.5;
rho_b=0.4;
xi_ev=xi_es^(2)*(SNR-(rho_v^(2)/(1-rho_v^(2))))*((b1^(2)+b2^(2))/(1-rho_v^(2)))^(-1);

bias_mean_phi=zeros(size(list_T,2), size(list_N,2), size(list_phi,2)); 
bias_mean_beta1=zeros(size(list_T,2), size(list_N,2)); 
bias_mean_beta2=zeros(size(list_T,2), size(list_N,2)); 
std_phi=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));  
std_beta1=zeros(size(list_T,2), size(list_N,2));  
std_beta2=zeros(size(list_T,2), size(list_N,2));  
rmse_phi=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
rmse_beta1=zeros(size(list_T,2), size(list_N,2));
rmse_beta2=zeros(size(list_T,2), size(list_N,2));

for idx_phi=1:size(list_phi,2)     
phi= list_phi(idx_phi);  
 
  
for idx_T=1:size(list_T,2)    
T0 = list_T(idx_T);             
 
for idx_N=1:size(list_N,2)   
N = list_N(idx_N);
Dis_T=50;  % discard first 50 time series
TT= (T0+1)+Dis_T;    
IVs_MG=zeros(1+k,rep);   % 1+k by rep 
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

eta_rho_i=-0.25+(0.5)*rand([1,N]); % 1 by N heterogeneous_y
eta_rho_bi=-sqrt(3)+2*sqrt(3)*rand([1,N]);
phi_i= phi+eta_rho_i;  % phi heterogeneous; 1 by N
v_x=zeros(k,N,TT);
v_x(:,:,1)=zeros(k,N,1);  
for ttt=2:TT
    for iiii=1:N
v_x(:,iiii,ttt)=0.5*v_x(:,iiii,ttt-1)+sqrt((1-0.5^(2)))*normrnd(0,xi_ev,[k,1]); % k by N by TT
    end 
end
bar_v_i=zeros(k,N);
for tttt=1:TT
bar_v_i=bar_v_i+v_x(:,:,tttt);
end
bar_v_i=bar_v_i*(TT)^(-1); % k by N
bar_v=sum(bar_v_i,2)*(N)^(-1); % k by 1
diff_bar_v=bar_v_i-bar_v*ones(1,N); % k by N
mean_sqr_diff_bar_v=sqrt( (sum((diff_bar_v.^(2)),2)*N^(-1))); % k by 1
Xi_b=zeros(k,N);
for kk=1:k
    for iiiii=1:N
Xi_b(kk,iiiii)= (diff_bar_v(kk,iiiii))/(mean_sqr_diff_bar_v(kk));  % k by N
    end
end



beta_i=b'*ones(1,N)+(sqrt(0.4^(2)/12)*rho_b*Xi_b+sqrt(1-rho_b^(2))*ones(k,1)*eta_rho_bi); % k by N

theta_i=[phi_i;beta_i];   % 1+k by N  




a_i=zeros(N,1);
for a=1:N
a_i(a,:)=A+normrnd(0,(1-phi_i(:,a))^(2)); % interactive effect for y; N by 1
end

mu1_i=zeros(N,1);
mu2_i=zeros(N,1);
for mu=1:N
mu1_i(mu,:)=Mu1+rho_mu*a_i(mu,:)+sqrt(1-rho_mu^(2))*normrnd(0,(1-phi_i(:,mu))^(2)); % interactive effect for x; N by 1
mu2_i(mu,:)=Mu2+rho_mu*a_i(mu,:)+sqrt(1-rho_mu^(2))*normrnd(0,(1-phi_i(:,mu))^(2)); % interactive effect for x; N by 1
end
mu_i=[mu1_i,mu2_i]; % N by k

fy=zeros(m_y,TT);  % creat a space for saving data factor
fy(:,1)=zeros(m_y,1);   % setting the initial factor
for t=2:TT
   fy(:,t)= 0.5*fy(:,t-1)+sqrt(1-0.5^2)*normrnd(0,1,[m_y,1]); % m_y by TT 
end 
%fy1=fy(:,TT-T0+1:TT); % m_y by T0; F_y


Gamma0=[0.25, 0.25, -1; 0.5,-1,0.25; 0.5, 0, 0];

Gamma_i=zeros(m_y,m_y,N);
for g=1:N
gamma_1= normrnd(0,1);
gamma_2= normrnd(0,1);
gamma_3= normrnd(0,1);
gamma_11= rho_gamma_1s*gamma_3 +sqrt(1-rho_gamma_1s^(2))*normrnd(0,1) ;
gamma_12= rho_gamma_1s*gamma_3 +sqrt(1-rho_gamma_1s^(2))*normrnd(0,1) ;
gamma_21= 0.5*normrnd(0,1)+sqrt(1-0.5^(2))*normrnd(0,1) ;
gamma_22= 0.5*normrnd(0,1)+sqrt(1-0.5^(2))*normrnd(0,1) ;
Gamma_i(:,:,g)=Gamma0+[gamma_1, gamma_11,gamma_21;gamma_2, gamma_12, gamma_22; gamma_3, 0, 0];
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fx=fy(1:m_x,:);  % m_x by TT

fx1=fx(:,TT-T0:TT-1); % m by T0; F_(x,-1)
fx2=fx(:,TT-T0+1:TT); % m by T0; F_(x)
MF_x1=eye(T0)-fx1'*((fx1*fx1')^(-1))*fx1;  % MF_x,-1 T0 by T0
MF_x2=eye(T0)-fx2'*((fx2*fx2')^(-1))*fx2;   %MF_x T0 by T0


eta_y=zeros(1,m_y,N);
eta_x=zeros(m_x,m_x,N);
y(:,1)=zeros(N,1);  
x(:,:,1)=zeros(k,N,1);  

for tt=2:TT
    for ii=1:N
        
   eta_x(:,:,ii)= [Gamma_i(1,2,ii),Gamma_i(1,3,ii);Gamma_i(2,2,ii),Gamma_i(2,3,ii)];     
   
 x(:,ii,tt)=mu_i(ii,:)'+rho*(x(:,ii,tt-1))+eta_x(:,:,ii)*fx(:,tt)+v_x(:,ii,tt);     % k by N by TT
 
   eta_y(:,:,ii)=[Gamma_i(1,1,ii),Gamma_i(2,1,ii),Gamma_i(3,1,ii)];
 
  y(ii,tt)= a_i(ii,:)+([y(ii,tt-1),x(:,ii,tt)'])*theta_i(:,ii)+eta_y(:,:,ii)*fy(:,tt)+(xi_es*sqrt((chi2rnd(2)/2)*(tt/TT))*(chi2rnd(1)-1))/sqrt(2);    % N by TT    
    end
end

  


y_NT=y(:,TT-T0:TT); % dicard first 50 time series for y ; N by T0+1 
y_NT1=y_NT(:,1:T0);    % y_(i,-1) ; N by T0
y_NT2=y_NT(:,2:T0+1);   % y_(i,T) N by T0


x_NT=x(:,:,TT-T0:TT); %  dicard first 50 time series for x ; k by N by T0+1

x_NT1=zeros(N, T0);
x_NT2=zeros(N, T0);
x_NT1_1=zeros(N, T0);
x_NT2_1=zeros(N, T0);
for it=1:T0
x_NT1(:,it)=x_NT(1,:,it+1);   % x_(i,-1) ; N by T0 
x_NT2(:,it)=x_NT(2,:,it+1);   % x_(i, T); N by T0
x_NT1_1(:,it)=x_NT(1,:,it);   % x_(i,-1) ; N by T0
x_NT2_1(:,it)=x_NT(2,:,it);   % x_(i, T);  N by T0
end

W_it=zeros(N,T0,1+k);  % N by T0 by 1+k  
W_it(:,:,1)=y_NT1;  %  y_{i,-1}; N by T0   
W_it(:,:,2)=x_NT1; %  x_(i,T); N by T0  
W_it(:,:,3)=x_NT2; %  x_(i,T); N by T0  
 
Z_it=zeros(N,T0,2*k);           % N by T0 by 2k
Z_it(:,:,1)=x_NT1*MF_x2;   % N by T0 
Z_it(:,:,2)=x_NT2*MF_x2;      % N by T0 
Z_it(:,:,3)=x_NT1_1*MF_x1;   % N by T0 
Z_it(:,:,4)=x_NT2_1*MF_x1;      % N by T0 


theta_IV=zeros(1+k,N); 
W=zeros(T0,1+k,N);
Z=zeros(T0,2*k,N);
for iii=1:N
W(:,:,iii)=[W_it(iii,:,1)', W_it(iii,:,2)',W_it(iii,:,3)']; % T0 by 1+k
Z(:,:,iii)=[Z_it(iii,:,1)', Z_it(iii,:,2)', Z_it(iii,:,3)',Z_it(iii,:,4)'];  %  T0 by 2k
theta_IV(:,iii)=(((Z(:,:,iii)'*MF_x2*W(:,:,iii))' /T0)*((Z(:,:,iii)'*MF_x2*Z(:,:,iii))/T0)^(-1)*((Z(:,:,iii)'*MF_x2*W(:,:,iii)) /T0))^(-1)*(((Z(:,:,iii)'*MF_x2*W(:,:,iii))' /T0)*((Z(:,:,iii)'*MF_x2*Z(:,:,iii))/T0)^(-1)*((Z(:,:,iii)'*MF_x2*y_NT2(iii,:)')/T0));

end

IVs_MG(:,sml)=nanmean(theta_IV,2); % LS_MG

sml=sml+1;
end

mean_phi= nanmean(IVs_MG(1,:));
mean_beta1= nanmean(IVs_MG(2,:));
mean_beta2= nanmean(IVs_MG(3,:));
bias_mean_phi(idx_T, idx_N, idx_phi) = mean_phi - phi;
bias_mean_beta1(idx_T, idx_N)=mean_beta1 - b1;
bias_mean_beta2(idx_T, idx_N)=mean_beta2 - b2;


std_phi(idx_T, idx_N, idx_phi) = nanstd(IVs_MG(1,:));
std_beta1(idx_T, idx_N) = nanstd(IVs_MG(2,:));
std_beta2(idx_T, idx_N) = nanstd(IVs_MG(3,:));


rmse_phi(idx_T, idx_N,idx_phi) = sqrt( nanmean( (IVs_MG(1,:)-phi).^2) ); 
rmse_beta1(idx_T, idx_N) = sqrt( nanmean( (IVs_MG(2,:)-b1).^2) ); 
rmse_beta2(idx_T, idx_N) = sqrt( nanmean( (IVs_MG(3,:)-b2).^2) );

end
end
end
filename = 'IVMG_rho_gamma_1s0.5.mat';
save(filename)