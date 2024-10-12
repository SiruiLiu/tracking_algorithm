%% 雷达参数设置
c = 3e8;                    % 光速
fc= 77e9;                   % 雷达工作频率 载频
lambda = c/fc;              % 波长
B = 3.8e8;                  % 发射信号带宽
Tchirp = 3.8e-5;            % 扫频时间
slope = B / Tchirp;         % 调频斜率
Nd = 256;                   % 每帧chirp数量 
Nr = 1140;                  % 每个chirp中ADC采样点数
RX_num = 64;                % 接收天线的数量
d0 = lambda/2;              % 接收天线间的距离
%% 相关测量参数计算
Tframe = Nd*Tchirp;                                     % 每个frame的持续时间
fs = Nr/Tchirp;                                         % ADC采样频率
fi_max= fs/2;                                           % 中频信号最大频率，根据香农采样定理计算
d_max = fi_max*c/(2*slope);                             % ADC限制的最大测量距离
d_res = (1/Tchirp)/slope*c/2;                           % 距离分辨率
v_max = lambda/(4*Tchirp);                              % 最大不模糊速度
v_res = ((1/Tframe)*2*pi*Tchirp*lambda/(4*pi))/Tchirp;  % 速度分辨率
%% 目标参数设置
r0 = [140;140];                         % 目标径向距离矩阵
v0 = [10;10];                           % 目标径向速度矩阵
rcs = [0.8;0.8];                        % 目标雷达截面积矩阵
theta = [pi/4;pi/3];                    % 目标角度矩阵
obj_list=[];   
%% 发射波形、回波、中频信号的生成
t = linspace(0,Tchirp,Nr);                      % 时间数组
St = cos(2*pi*(fc*t + (slope*t.^2)/2));         % 发射信号 实数信号
final_mix_data = zeros(Nd,length(t),RX_num);    % 最终的混频信号
for i = 1:RX_num    % 为每个接收天线生成回波信号
    St_data = zeros(Nd,length(t));     % 一个frame的发射信号，预分配内存
    Sr_data = zeros(Nd,length(t));     % 一个frame的回波信号，预分配内存
    Mix_data = zeros(Nd,length(t));    % 一个frame的中频信号，预分配内存
    for j =1:Nd     % 为frame生成每个chirp的回波信号和中频信号
        r= r0 + v0*(t+Tchirp*(j-1));                        % 距离更新
        td = 2*r/c+repmat(sin(theta)*d0*i/c,1,length(t));   % 延迟时间
        Sr = zeros(1,length(t))+wgn(1,length(t),20);                            % 一个chirp的回波信号
        for k = 1:length(r0)
            Sr = Sr+rcs(k)*cos(2*pi*(fc*(t-td(k,:)) + (slope*(t-td(k,:)).^2)/2)); %接收信号 实数信号
        end
        Mix = St.*Sr;               % 一个chirp上的混频
        St_data(j,:) = St;          % 将每个chirp的发射信号记录到相应的frame数据上
        Sr_data(j,:) = Sr;          % 将每个chirp的回波信号记录到相应的frame数据上
        Mix_data(j,:) = Mix;        % 将每个chirp的中频信号记录到相应的frame数据上
    end
    final_mix_data(:,:,i) = Mix_data;   % 将每个frame的中频信号记录到相应的天线上
end
%% 2D-FFT处理
FFT_2D = zeros(Nd,Nr/2,RX_num);                 % 每个天线中频信号的距离、速度维2D-FFT结果矩阵
for i = 1:RX_num                                % 对每个天线的中频信号做2D-FFT处理，并记录结果
   sig_fft2 = fft2(final_mix_data(:,:,i));
   sig_fft2 = sig_fft2(1:Nd,1:Nr/2);
   sig_fft2 = fftshift(sig_fft2,1);
   FFT_2D(:,:,i)=sig_fft2;
end
%% 距离速度CFAR恒虚警检测
data=sum(abs(FFT_2D),3)/3;
% 参考单元和保护单元数量
T1 = 4;
T2 = 8;
G1 = 2;
G2 = 4;
Training_num=(2*T1+2*G1+1)*(2*T2+2*G2+1)-(2*G1+1)*(2*G2+1);
P_fa = 1e-5;    % 虚警概率
alpha = Training_num*(P_fa^(-1/Training_num) - 1);  
% 估计噪声转化为卷积操作
fliter = ones(2*T1+2*G1+1,2*T2+2*G2+1);     
fliter(T1+1:T1+2*G1+1,T2+1:T2+2*G2+1)=0;
fliter = fliter/Training_num;
% 估计噪声功率，循环填充，即使用翻转法处理边界数据
Pn= imfilter(data,fliter,'circular','same'); 
Pn = alpha*Pn;  % 检测门限
CFAR_2D = data-Pn;
% 2D CFAR检测结果
CFAR_2D(CFAR_2D>0)=1;   
CFAR_2D(CFAR_2D<=0)=0;
[rows, cols]=find(CFAR_2D>0);
% 存储2D-CFAR检测结果的列表
temp_obj_list=[];
for i = 1:length(rows)
    row = rows(i);
    col = cols(i);
    obj = [(row-Nd/2-1)*v_res (col-1)*d_res];
    temp_obj_list= [temp_obj_list;obj];
end
%% 距离速度检测结果可视化
x = (-Nd/2:Nd/2-1)*v_res;
y = (0:Nr/2-1)*d_res;
figure;
mesh(y,x,data);
ylabel('y速度 m/s'); xlabel('x距离 m'); 
title('距离、速度2D-FFT');
% CFAR检测结果
figure
mesh(y,x,CFAR_2D)
ylabel('y速度 m/s'); xlabel('x距离 m'); 
title('速度、距离CFAR检测结果');
%% 角度维的FFT
angle_fft_data = [];
N = 256;
for i =1:length(rows)
    row = rows(i);
    col = cols(i);
    angle_fft_data = [angle_fft_data;FFT_2D(row,col,:)];
end
angle_fft_data = squeeze(angle_fft_data);
if length(rows)>1   
    angle_fft = fft(angle_fft_data,N,2);
    angle_fft = fftshift(angle_fft,2);
    angle_fft = abs(angle_fft);
else
    angle_fft_data = fliplr(angle_fft_data');
    angle_fft = fft(angle_fft_data,N);
    angle_fft = fftshift(angle_fft);
    angle_fft = abs(angle_fft);
end
%% 角度检测结果可视化
x = asin((-N/2:N/2-1)/N*lambda/d0)*360/(2*pi);
for i =1:length(rows)
    figure;
    plot(x,angle_fft(i,:))
    xlabel('角度 °'); 
    title('角度维FFT');
end
%% CFAR恒虚警检测+峰值检测
% 参考单元和保护单元数量
T = 4;
G = 4;
Training_num=2*T;
P_fa = 0.01;    %虚警概率
alpha = Training_num*(P_fa^(-1/Training_num) - 1);  

fliter = ones(1,2*T+2*G+1);     %估计噪声转化为卷积操作
fliter(T+1:T+2*G+1)=0;
fliter = fliter/Training_num;
num = length(rows);
for i = 1:num
    Pn= imfilter(angle_fft(i,:),fliter,'circular','same'); % 估计噪声功率，循环填充，即使用翻转法处理边界数据
    Pn = alpha*Pn;  %检测门限
    CFAR_1D = angle_fft(i,:)-Pn;
    [rows, cols]=find(CFAR_1D>0);
    CFAR_1D(CFAR_1D>0)=1;   %2D CFAR检测结果
    CFAR_1D(CFAR_1D<=0)=0;
    peak=islocalmax(angle_fft(i,:));
    CFAR_1D = peak.*CFAR_1D;
    angle=asin((find(CFAR_1D==1)-N/2-1)/N*lambda/d0)*360/(2*pi);
    if num>1
        for j = 1:length(angle)
            obj = [temp_obj_list(i,1) temp_obj_list(i,2) angle(j)];
            obj_list = [obj_list;obj];
        end
    else
        for j = 1:length(angle)
            obj = [temp_obj_list(1) temp_obj_list(1) angle(j)];
            obj_list = [obj_list;obj];
        end
    end
    figure
    plot(x,CFAR_1D)
    title('角度域CFAR检测+峰值检测结果')
end