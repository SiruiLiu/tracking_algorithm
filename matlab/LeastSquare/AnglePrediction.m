%% ------------------------------------------------------------------------
%         角度环仿真 
%  ------------------------------------------------------------------------
clear;
close all;
clc;

%% ------------------------------------------------------------------------
% 创建目标角度曲线并添加测量误差
% 模拟曲线直接采用正弦曲线
% 采样周期为20ms
% 采样点数为128点
% 角度在70度附近呈正弦分布，正弦幅值不超过3°
% 伺服定位误差不超过0.05， 可设置
%  ------------------------------------------------------------------------
Period = 20e-3; %采样周期
ServePeriod = Period*0.5;%伺服脉冲周期只有测量周期的一半
Fs=1/Period;    %采样率
N = 128;        %采样点数
Duration = N*Period; %总时间
TrackWinWidth = 6;   %跟踪窗口宽度，宽度可设置
t = Period*8:Period:Duration+Period*7; %从第二个周期开始，避免出现0
Fc = 0.7;
CentralAng = 70; %角度设为70，该值仿真时可调
AngAmp = 15;	 %角度范围在70 ± 15范围内。
Angle = AngAmp*sin(2*pi*Fc.*t); 
Angle = Angle + CentralAng; %产生角度分布曲线
GuardTime = 13e-3;
PosErr = 0.05;

% 添加随机误差
AmpErr = 3;
Err = (-AmpErr - AmpErr)*rand(1, N) + AmpErr;
AngDetected = Angle+Err;
figure(1000);
o_curve = plot(Angle); hold on;
detected_curve = plot(AngDetected, 'r'); hold off;
ylim([50, 100])
legend('目标飞行角度曲线', '实际探测角度曲线（含噪）')

%% -----------------------------------------------------------------------
% 线性预测仿真（分段曲线拟合后预测）
% -- 一次函数拟合
PredictedData = LinearFunc_Fitting(t, AngDetected, TrackWinWidth);
figure(1001);
plot(AngDetected); hold on;
plot(PredictedData, 'r'); hold off;

% -- 二次函数拟合
PredictedData = QuadFunc_Fitting(t, AngDetected, TrackWinWidth);
figure(1002);
plot(AngDetected); hold on;
plot(PredictedData, 'r'); hold off;

% -- 三次函数拟合
PredictedData = CubicFunc_Fitting(t, AngDetected, TrackWinWidth);
figure(1003);
plot(AngDetected); hold on;
plot(PredictedData, 'r'); hold off;

%% -----------------------------------------------------------------------
% 闭环仿真
% 预测结果添加一个误差作为伺服读数, 伺服读数和测量值有对应时间，对伺服读数和测量值进行
% 拟合，拟合结果根据伺服脉冲周期对后续位置进行预测。
% 伺服角读数不能少于窗宽-1

%一次函数拟合
PredictedData = ClosedLoop_Sim(t, AngDetected, TrackWinWidth, CentralAng, ...
                               GuardTime, ServePeriod, PosErr, 'Linear');
figure(1004);
plot(PredictedData);
title('一次函数拟合跟踪结果');

PredictedData = ClosedLoop_Sim(t, AngDetected, TrackWinWidth, CentralAng, ...
                               GuardTime, ServePeriod, PosErr, 'Quard');
figure(1005);
plot(PredictedData);
title('二次函数拟合跟踪结果');
PredictedData = ClosedLoop_Sim(t, AngDetected, TrackWinWidth, CentralAng, ...
                               GuardTime, ServePeriod, PosErr, 'Cubic');
figure(1006);
plot(PredictedData);
title('三次函数拟合跟踪结果');