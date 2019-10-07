rep = 1000;  
list_T = [25 50 100 200]; 
list_N = [25 50 100 200];  
list_phi= [0.25]; 
rho=[0];
k=1;   % number of regressors
m_x=2; % number of factors of regressor
m_y=2; % number of factors of y
bias_mean_phi=zeros(size(list_T,2), size(list_N,2), size(list_phi,2)); 
bias_mean_beta=zeros(size(list_T,2), size(list_N,2)); 
std_phi=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));  
std_beta=zeros(size(list_T,2), size(list_N,2));  
rmse_phi=zeros(size(list_T,2), size(list_N,2), size(list_phi,2));
rmse_beta=zeros(size(list_T,2), size(list_N,2));

for idx_phi=1:size(list_phi,2)     
phi= list_phi(idx_phi);  
  b=1-phi; 
for idx_T=1:size(list_T,2)    
T0 = list_T(idx_T);             
 
for idx_N=1:size(list_N,2)   
N = list_N(idx_N);
Dis_T=50;

TT= (T0+1)+Dis_T;    


LS_MG=zeros(1+k,rep);   % 1+k by rep
sml=1;         
%while sml<=rep
parfor sml=1:rep
y=zeros(N,TT);
x=zeros(N,TT);
%phi_i= phi+  (0.8)*rand([1,N]);
phi_i= phi*ones(1,N);
beta_i=b+0.5+(0.5)*rand([1,N]); 
theta_i=[phi_i;beta_i];   % 1+k by N
eta_y=normrnd(0,1/m_y,[m_y,N]);  % factor loading; N by m
fy=zeros(m_y,TT);  % creat a space for saving data factor
fy(:,1)=zeros(m_y,1);   % setting the initial factor
for t=2:TT
   fy(:,t)= 0.5*fy(:,t-1)+sqrt(1-0.5^2)*normrnd(0,1,[m_y,1]); % m by TT
end 

eta_x=normrnd(0,1/m_x,[m_x,N]);  % factor loading; N by m
fx=zeros(m_x,TT);  % creat a space for saving data factor
fx(:,1)=zeros(m_x,1);   % setting the initial factor
for t=2:TT
   fx(:,t)= 0.5*fx(:,t-1)+sqrt(1-0.5^2)*normrnd(0,1,[m_x,1]); % m by TT
end 


y(:,1)=zeros(N,1);  
x(:,1)=zeros(N,1);  
for tt=2:TT
    for ii=1:N
   x(ii,tt)=rho*(x(ii,tt-1))+eta_x(:,ii)'*fx(:,tt)+normrnd(0,1);     % N by TT
 %x(ii,tt)=normrnd(0,1);     % N by TT
  y(ii,tt)= ([y(ii,tt-1),x(ii,tt)])*theta_i(:,ii)+eta_y(:,ii)'*fy(:,tt)+normrnd(0,1);    % N by TT
    end
end
%for tt=2:TT
% for ii=1:N
% x(ii,tt)=0.5*(x(ii,tt-1))+normrnd(0,1);     % N by TT
% y(ii,tt)= phi_i(1,ii)*y(ii,tt-1)+beta_i(1,ii)*x(ii,tt) +normrnd(0,1);    % N by TT end
%end
%end

y_NT=y(:,TT-T0:TT); % dicard first 50 time series N by T0+1 
y_NT1=y_NT(:,1:T0);
x_NT=x(:,TT-T0+1:TT); %  N by T0
W_it=zeros(N,T0,1+k);  % N by T0 by 1+k 

W_it(:,:,1)=y_NT1;  %  y_{i,-1}; N by T0 
W_it(:,:,2)=x_NT; %  N by T0


theta_LS=zeros(1+k,N);
W=zeros(1+k,T0,N);
for iii=1:N
W(:,:,iii)=[W_it(iii,:,1);W_it(iii,:,2)]; % 1+k by TT
theta_LS(:,iii)= ((W(:,:,iii)*W(:,:,iii)')^(-1))*(W(:,:,iii)*y_NT(iii,2:T0+1)');
end
LS_MG(:,sml)=nanmean(theta_LS,2); % LS_MG

%sml=sml+1;
end

mean_phi= nanmean(LS_MG(1,:));
mean_beta= nanmean(LS_MG(2,:));
bias_mean_phi(idx_T, idx_N, idx_phi) = mean_phi - phi;
bias_mean_beta(idx_T, idx_N,size(b,2) )=mean_beta - b;

std_phi(idx_T, idx_N, idx_phi) = nanstd(LS_MG(1,:));
std_phi(idx_T, idx_N, size(b,2)) = nanstd(LS_MG(2,:));
rmse_phi(idx_T, idx_N,idx_phi) = sqrt( nanmean( (LS_MG(1,:)-phi).^2) ); 
rmse_beta(idx_T, idx_N) = sqrt( nanmean( (LS_MG(2,:)-b).^2) ); 

end
end
end
filename = 'LS_MG_f2.mat';
save(filename)

