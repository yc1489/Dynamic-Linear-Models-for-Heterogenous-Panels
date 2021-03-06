tic
rep = 500;  
list_T = [ 25 50]; 
list_N = [25  50];  
list_phi= [0.5]; 
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

j=1; % lag of regressor
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sec_bias_mean_phi_1=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
sec_bias_mean_beta1_1=zeros(size(list_T,2), size(list_N,2), size(list_phi,2)); 

sec_std_phi_1=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));  
sec_std_beta1_1=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));  

sec_rmse_phi_1=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
sec_rmse_beta1_1=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));

sec_mae_phi_1=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
sec_mae_beta1_1=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));



for idx_phi=1:size(list_phi,2)     
phi= list_phi(idx_phi);  
  
for idx_T=1:size(list_T,2)    
T0 = list_T(idx_T);             
T1=T0-1; 
for idx_N=1:size(list_N,2)   
N = list_N(idx_N);

TT= (T0+1+j)+Dis_T;    
ini_IVMG_1=zeros(1+k,rep);   % 1+k by rep
sec_IVMG_1=zeros(1+k,rep);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
randn('state', 12345678) ;
rand('state', 1234567) ;
   RandStream.setGlobalStream (RandStream('mcg16807','seed',34));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sml=1;         
while sml<=rep
eta_rho_i=-0.2+(0.4)*rand([1,N]); % 1 by N heterogeneous_y U(-0.2,0.2)
eta_rho_bi=-sqrt(3)+2*sqrt(3)*rand([1,N]);

phi_i= phi+  eta_rho_i;
sig_bar_w=0.5+rand(N,1);
v_x=zeros(k,N,TT);
v_x(:,:,1)=zeros(k,N,1);  
for ttt=2:TT
    for iiii=1:N
v_x(:,iiii,ttt)=0.5*v_x(:,iiii,ttt-1)+sqrt((1-0.5^(2)))*normrnd(0,sqrt(xi_ev*sig_bar_w(iiii,1)),[k,1]); % k by N by TT
%v_x(:,iiii,ttt)=0.5*v_x(:,iiii,ttt-1)+sqrt((1-0.5^(2)))*normrnd(0,sqrt(xi_ev),[k,1]); % k by N by TT
    end 
end
bar_v_i=zeros(k,N);
for tttt=TT-T0+1:TT
bar_v_i=bar_v_i+(v_x(:,:,tttt).^2);
end
bar_v_i=bar_v_i/T0; % k by N
bar_v=sum(bar_v_i,2)/N; % k by 1
diff_bar_v=bar_v_i-bar_v*ones(1,N); % k by N
mean_sqr_diff_bar_v=sqrt( (sum((diff_bar_v.^2),2)/N)); % k by 1
Xi_b=zeros(k,N);
for kk=1:k
    for iiiii=1:N
Xi_b(kk,iiiii)= (diff_bar_v(kk,iiiii))/(mean_sqr_diff_bar_v(kk));  % k by N
    end
end

 beta_i=b'*ones(1,N)+(sqrt((0.4^2)/12)*rho_b*Xi_b+ones(k,1)*sqrt(1-rho_b^(2))*eta_rho_bi); % k by N
%beta_i=b'*ones(1,N);



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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55
%fy=zeros(m_y,TT);  % creat a space for saving data factor
%fy(:,1)=zeros(m_y,1);   % setting the initial factor
%for t=2:TT
%   fy(:,t)= 0.5*fy(:,t-1)+sqrt(1-0.5^2)*normrnd(0,1,[m_y,1]); % m_y by TT 
%end 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%5555
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
eta_y=zeros(1,m_y,N);
eta_x=zeros(m_x,m_x,N);
y=zeros(N,TT);
x=zeros(k,N,TT);  
y(:,1)=zeros(N,1);  
x(:,:,1)=zeros(k,N,1);  

fy=zeros(m_y,TT);  % creat a space for saving data factor
fy(:,1)=zeros(m_y,1);   % setting the initial factor
for tt=2:TT
     fy(:,tt)= 0.5*fy(:,tt-1)+sqrt(1-0.5^2)*normrnd(0,1,[m_y,1]); % m_y by TT 
    for ii=1:N
        
   eta_x(:,:,ii)= [Gamma_i(1,2,ii),Gamma_i(1,3,ii);Gamma_i(2,2,ii),Gamma_i(2,3,ii)];     
 %
 x(:,ii,tt)= mu_i(ii,:)'+eta_x(:,:,ii)'*fy(1:m_x,tt)+v_x(:,ii,tt);     % k by N by TT
 
   eta_y(:,:,ii)=[Gamma_i(1,1,ii),Gamma_i(2,1,ii),Gamma_i(3,1,ii)]; % 1 by m_y
% 
  y(ii,tt)= a_i(ii,:)+y(ii,tt-1)*phi_i(:,ii)+x(:,ii,tt)'*beta_i(:,ii)+eta_y(:,:,ii)*fy(:,tt)+(sqrt(xi_es)*sqrt((chi2rnd(2)/2)*(tt/TT))*(chi2rnd(1)-1))/sqrt(2);    % N by TT    
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
 A2=T1:-1:bt;   
 B=-1*A2.^-1;  
 C=diag(B);
F_2= F_2+[zeros(T1,bt) [C;zeros(bt-1,T0-bt)]] ;
end
F_2=[diag(ones(1,T1)) zeros(T1,1)]+F_2;
F=  F_1*F_2;   % T0-1 by T0

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fx=fy(1:m_x, :);  % m_x by TT
y_NT=y(:,TT-T0:TT); % dicard first 50 time series N by T0+1 

%bar_yi=sum(y_nt,2)/(T0+1);  % N by 1
%bar_yt=sum(y_nt,1)/N;     %  1 by T0+1+j
%bar_y=sum(bar_yi,1)/N;    %  1 by 1   

%y_NT=y_nt-bar_yi*ones(1,T0+1)-ones(N,1)*bar_yt+bar_y*ones(N,T0+1);



y_NT1=F*y_NT(:,1:T0)';  % y_(i,-1); T by N
y_NT2=F*y_NT(:,2:T0+1)';  % y_(i,T)  ; T by N



%fy1=fy(:,TT-T0+1:TT);   % m by T0; F_y
%fx1=fx(:,TT-T0+1:TT); % m by T0; F_(x)
%fx2=fx(:,TT-T0:TT-1); % m by T0; F_(x,-1)
%fx3=fx(:,TT-T0-1:TT-2); % m by T0; F_(x,-2)

fx1=fx(:,TT-T0+2:TT); % m by T0; F_(x)
fx2=fx(:,TT-T0+1:TT-1); % m by T0; F_(x,-1)
fx3=fx(:,TT-T0:TT-2); % m by T0; F_(x,-2)

%MF_y=eye(T0)-fy1'*((fy1*fy1')^(-1))*fy1;  % MF_y T0 by T0
MF_x1=eye(T1)-fx1'*((fx1*fx1')^(-1))*fx1;  % MF_x T0 by T0
MF_x2=eye(T1)-fx2'*((fx2*fx2')^(-1))*fx2;   %MF_x,-1 T0 by T0
MF_x3=eye(T1)-fx3'*((fx3*fx3')^(-1))*fx3;   %MF_x,-2 T0 by T0


x_NT=x(:,:,TT-T0-j:TT); %  dicard first 50 time series for x ; k by N by T0+1+j
%sum_xt=zeros(k,N);
%for bi=1:T0+1+j
%sum_xt=sum_xt+x_NT(:,:,bi);  
%end
%bar_xi=sum_xt/(T0+1+j);    % k by N
%sum_xi=zeros(k, T0+1+j);
%for bt=1:N
%sum_xi=sum_xi+x_NT(:,bt,:); 
%end
%bar_xt=sum_xi/N;    % k by T0+1+j
%bar_x=sum(bar_xi,2)/N;  % 1 by 1 





x_NT1=zeros(N, T0);
x_NT2=zeros(N, T0);
x_NT1_1=zeros(N, T0);
x_NT2_1=zeros(N, T0);
x_NT1_2=zeros(N, T0);
x_NT2_2=zeros(N, T0);

for it=1:T0
x_NT1(:,it)=x_NT(1,:,it+j+1);   % x_(i,) ; N by T0 
x_NT2(:,it)=x_NT(2,:,it+j+1);   % x_(i,) ; N by T0 
x_NT1_1(:,it)=x_NT(1,:,it+j);   % x_(i,-1) ; N by T0 
x_NT2_1(:,it)=x_NT(2,:,it+j);   % x_(i,-1) ; N by T0 
x_NT1_2(:,it)=x_NT(1,:,it+(j-1));   % x_(i,-2) ; N by T0 
x_NT2_2(:,it)=x_NT(2,:,it+(j-1));   % x_(i,-2) ; N by T0 
end
%x_NT1= x_NT1-bar_xi(1,:)'-bar_xt(1,3:T0+1+j)+bar_x(1,:);
%x_NT2=x_NT2-bar_xi(2,:)'-bar_xt(2,3:T0+1+j)+bar_x(2,:);
%x_NT1_1=x_NT1_1-bar_xi(1,:)'-bar_xt(1,2:T0+j)+bar_x(1,:);
%x_NT2_1=x_NT2_1-bar_xi(2,:)'-bar_xt(2,2:T0+j)+bar_x(2,:);
%x_NT1_2=x_NT1_2-bar_xi(1,:)'-bar_xt(1,1:T0)+bar_x(1,:);
%x_NT2_2=x_NT2_2-bar_xi(2,:)'-bar_xt(2,1:T0)+bar_x(2,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
D=zeros(T1,1+k,N);    
Z_1=zeros(T1,2*k,N);   % lag 1
Z_2=zeros(T1,3*k,N);  % lag 2
for iti=1:N
D(:,:,iti)=[y_NT1(:,iti), F*x_NT1(iti,:)' ,F*x_NT2(iti,:)'  ];   %T by 1+k
Z_1(:,:,iti)=[MF_x1*F*x_NT1(iti,:)', MF_x1*F*x_NT2(iti,:)',MF_x2*F*x_NT1_1(iti,:)' ,MF_x2*F*x_NT2_1(iti,:)'  ] ;  % T by 2k   
Z_2(:,:,iti)=[MF_x1*F*x_NT1(iti,:)', MF_x1*F*x_NT2(iti,:)',MF_x2*F*x_NT1_1(iti,:)' ,MF_x2*F*x_NT2_1(iti,:)',MF_x3*F*x_NT1_2(iti,:)',MF_x3*F*x_NT2_2(iti,:)'] ; 
end


ini_theta_IV_1=zeros(1+k,N); 
for iii1=1:N
ini_theta_IV_1(:,iii1)= pinv((D(:,:,iii1)'*MF_x1* Z_1(:,:,iii1)/T0)*pinv(Z_1(:,:,iii1)'*MF_x1*Z_1(:,:,iii1)/T0)*(Z_1(:,:,iii1)'*MF_x1*D(:,:,iii1)/T0))*(D(:,:,iii1)'*MF_x1*Z_1(:,:,iii1)/T0)*pinv(Z_1(:,:,iii1)'*MF_x1*Z_1(:,:,iii1)/T0)*(Z_1(:,:,iii1)'*MF_x1*y_NT2(:,iii1)/T0);
end

sec_theta_IV_1=zeros(1+k,N); 
for iii2=1:N
sec_theta_IV_1(:,iii2)= pinv(D(:,:,iii2)'*MF_x1* Z_2(:,:,iii2)*pinv(Z_2(:,:,iii2)'*MF_x1*Z_2(:,:,iii2))*Z_2(:,:,iii2)'*MF_x1*D(:,:,iii2))*D(:,:,iii2)'*MF_x1*Z_2(:,:,iii2)*pinv(Z_2(:,:,iii2)'*MF_x1*Z_2(:,:,iii2))*Z_2(:,:,iii2)'*MF_x1*y_NT2(:,iii2);
end

 
   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ini_IVMG_1(:,sml)=nanmean(ini_theta_IV_1,2); 
sec_IVMG_1(:,sml)=nanmean(sec_theta_IV_1,2); 

sml=sml+1;
end


ini_bias_mean_phi_1(idx_T, idx_N, idx_phi)= nanmean(ini_IVMG_1(1,:)- phi,2);
ini_bias_mean_beta1_1(idx_T, idx_N, idx_phi)= nanmean(ini_IVMG_1(2,:)- b1,2);

sec_bias_mean_phi_1(idx_T, idx_N, idx_phi) = nanmean(sec_IVMG_1(1,:)- phi,2);
sec_bias_mean_beta1_1(idx_T, idx_N, idx_phi)= nanmean(sec_IVMG_1(2,:)- b1,2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






ini_std_phi_1(idx_T, idx_N, idx_phi) = nanstd(ini_IVMG_1(1,:));
ini_std_beta1_1(idx_T, idx_N, idx_phi) = nanstd(ini_IVMG_1(2,:));

sec_std_phi_1(idx_T, idx_N, idx_phi) = nanstd(sec_IVMG_1(1,:));
sec_std_beta1_1(idx_T, idx_N, idx_phi) = nanstd(sec_IVMG_1(2,:));



ini_rmse_phi_1(idx_T, idx_N,idx_phi) = sqrt( nanmean( (ini_IVMG_1(1,:)-phi).^2) ); 
ini_rmse_beta1_1(idx_T, idx_N, idx_phi) = sqrt( nanmean( (ini_IVMG_1(2,:)-b1).^2) ); 

sec_rmse_phi_1(idx_T, idx_N,idx_phi) = sqrt( nanmean( (sec_IVMG_1(1,:)-phi).^2) ); 
sec_rmse_beta1_1(idx_T, idx_N, idx_phi) = sqrt( nanmean( (sec_IVMG_1(2,:)-b1).^2) ); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ini_mae_phi_1(idx_T, idx_N,idx_phi) = median((ini_IVMG_1(1,:)-phi).^2);
ini_mae_beta1_1(idx_T, idx_N, idx_phi) = median((ini_IVMG_1(2,:)-b1).^2);

sec_mae_phi_1(idx_T, idx_N,idx_phi) = median((sec_IVMG_1(1,:)-phi).^2);
sec_mae_beta1_1(idx_T, idx_N, idx_phi) = median((sec_IVMG_1(2,:)-b1).^2);

end
end
end
toc