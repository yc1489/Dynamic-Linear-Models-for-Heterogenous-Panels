rep = 2000;  
list_T = [25 50 100 200]; 
list_N = [25 50 100 200];  
list_phi= [0.5]; 
rho=0;
b1=3;
b2=1;
rho_b=0.4;
b=[b1,b2];
k=2;   % number of regressors
bias_mean_phi=zeros(size(list_T,2), size(list_N,2), size(list_phi,2)); bias_mean_beta1=zeros(size(list_T,2), size(list_N,2)); 
bias_mean_beta2=zeros(size(list_T,2), size(list_N,2)); 
std_phi=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));  
std_beta1=zeros(size(list_T,2), size(list_N,2));  
std_beta2=zeros(size(list_T,2), size(list_N,2));  
rmse_phi=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
rmse_beta1=zeros(size(list_T,2), size(list_N,2));
rmse_beta2=zeros(size(list_T,2), size(list_N,2));


for idx_phi=1:size(list_phi,2)     
phi= list_phi(idx_phi);  
  b1=3;
  b2=1;
for idx_T=1:size(list_T,2)    
T0 = list_T(idx_T);             
 
for idx_N=1:size(list_N,2)   
N = list_N(idx_N);
Dis_T=50;

TT= (T0+1)+Dis_T;    

IVMG=zeros(1+k,rep);   % 1+k by rep

sml=1;         
while sml<=rep
%parfor sml=1:rep
y=zeros(N,TT);
x=zeros(k,N,TT);

eta_rho_i=-0.25+(0.5)*rand([1,N]); % 1 by N
phi_i= phi+  eta_rho_i;

beta_i=b'*ones(1,N)+sqrt(1-rho_b^(2))*ones(k,1)*eta_rho_i; % k by N

theta_i=[phi_i;beta_i];   % 1+k by N
v_x=zeros(k,N,TT);
v_x(:,:,1)=zeros(k,N,1);  
for ttt=2:TT
    for iiii=1:N
v_x(:,iiii,ttt)=0.5*v_x(:,iiii,ttt-1)+sqrt((1-0.5^(2)))*(0.5+rand(k,1)); % k by N by TT
    end 
end




y(:,1)=zeros(N,1);  
x(:,:,1)=zeros(k,N,1); 
for tt=2:TT
    for ii=1:N
 x(:,ii,tt)=rho*(x(:,ii,tt-1))+v_x(:,ii,tt);     % k by N by TT
 
  y(ii,tt)= ([y(ii,tt-1),x(:,ii,tt)'])*theta_i(:,ii)+normrnd(0,1);    % N by TT
    end
end


y_NT=y(:,TT-T0:TT); % dicard first 50 time series N by T0+1 
y_NT1=y_NT(:,1:T0);  % y_(i,-1); N by T0
y_NT2=y_NT(:,2:T0+1);  % y_(i,T)  ; N by T0

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
Z_it(:,:,1)=x_NT1;   % N by T0 
Z_it(:,:,2)=x_NT2;      % N by T0 
Z_it(:,:,3)=x_NT1_1;   % N by T0 
Z_it(:,:,4)=x_NT2_1;      % N by T0 

theta_IV=zeros(1+k,N); 
W=zeros(T0,1+k,N);
Z=zeros(T0,2*k,N);
for iii=1:N
W(:,:,iii)=[W_it(iii,:,1)', W_it(iii,:,2)',W_it(iii,:,3)']; % T0 by 1+k
Z(:,:,iii)=[Z_it(iii,:,1)', Z_it(iii,:,2)', Z_it(iii,:,3)',Z_it(iii,:,4)'];  %  T0 by 2k
theta_IV(:,iii)=(((Z(:,:,iii)'*W(:,:,iii))' /T0)*((Z(:,:,iii)'*Z(:,:,iii))/T0)^(-1)*((Z(:,:,iii)'*W(:,:,iii)) /T0))^(-1)*(((Z(:,:,iii)'*W(:,:,iii))' /T0)*((Z(:,:,iii)'*Z(:,:,iii))/T0)^(-1)*((Z(:,:,iii)'*y_NT2(iii,:)')/T0));

end

IVMG(:,sml)=nanmean(theta_IV,2); % LS_MG

sml=sml+1;
end

mean_phi= nanmean(IVMG(1,:));
mean_beta1= nanmean(IVMG(2,:));
mean_beta2= nanmean(IVMG(3,:));

bias_mean_phi(idx_T, idx_N, idx_phi) = mean_phi - phi;
bias_mean_beta1(idx_T, idx_N )=mean_beta1 - b1;
bias_mean_beta2(idx_T, idx_N)=mean_beta2 - b2;

std_phi(idx_T, idx_N, idx_phi) = nanstd(IVMG(1,:));
std_beta1(idx_T, idx_N) = nanstd(IVMG(2,:));
std_beta2(idx_T, idx_N) = nanstd(IVMG(3,:));

rmse_phi(idx_T, idx_N,idx_phi) = sqrt( nanmean( (IVMG(1,:)-phi).^2) ); 
rmse_beta1(idx_T, idx_N) = sqrt( nanmean( (IVMG(2,:)-b1).^2) ); 
rmse_beta2(idx_T, idx_N) = sqrt( nanmean( (IVMG(3,:)-b2).^2) ); 


end
end
end
filename = 'IVMG1.mat';
save(filename)



