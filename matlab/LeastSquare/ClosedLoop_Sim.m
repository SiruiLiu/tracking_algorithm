%% -------------------------------------------------------------
% 闭环仿真，连续N个点（2≤N≤5）的罗盘读数加一个测量值放入拟合队列，用拟合结果
% 预测下一个位置，该位置添加一个较小的随机误差，作为新的罗盘读数，进入队列。
% @t 每个测量值对应的采集时间
% @AngleData 测量角度数据
% @WinWidth 跟踪窗宽度
% @InitialAng 伺服初始角度
% @GuardTime 保护时间，测量位置与最近一次上报罗盘数据的时间差。
% @Error 伺服定位误差
% @Period 伺服脉冲周期
function [PredictedData] = ClosedLoop_Sim(t, DetectedData, WinWidth, InitialAng, ... 
										  GuardTime, Period, Error, Method)
	TrackWin = ones(1, WinWidth+1); %+1用于存测得的角度
	TimeArray = zeros(1, WinWidth+1); %时间队列，前WinWidth个数据对应罗盘角度，+1对应测得的角度
	TrackWin = TrackWin.*InitialAng; %开始时罗盘角固定不动。
	PredictedData = zeros(1, length(DetectedData));
	PosErr = (-Error - Error)*rand(1, length(DetectedData)) + Error;

	for i=1:length(DetectedData)
		TrackWin(WinWidth + 1) = DetectedData(i);
		TimeArray(WinWidth + 1) = t(i);
		TimeTmp = zeros(1, WinWidth);
		for j = 1:WinWidth 
			TimeTmp(j) = TimeArray(WinWidth + 1) - GuardTime - j*Period;
		end
		TimeArray(1:WinWidth) = fliplr(TimeTmp);

		switch Method 
		case 'Linear' 
			PredictedData(i) = LinearFitting(TimeArray, TrackWin, Period);
		case 'Quard' 
			PredictedData(i) = QuardFitting(TimeArray, TrackWin, Period);
		case 'Cubic' 
			PredictedData(i) = CubicFitting(TimeArray, TrackWin, Period);
		end
		TrackWin(1:WinWidth) = [TrackWin(2:WinWidth), PredictedData(i)+PosErr(i)];
	end
end

%% -------最小二乘法
% 线性跟踪
function [PredictData] = LinearFitting(TimeArray, TrackData, Period)
	Array = [TimeArray.', ones(length(TimeArray), 1)];
	Coe = (Array.'*Array)^-1*Array.'*TrackData.';
	%因为最后一个时间是测量到角度的时间，所以用倒数第二个时间+周期作为新的罗盘角位置。
	PredictData = (TimeArray(length(TimeArray)-1)+Period)*Coe(1)+Coe(2); 
end

% 二次函数拟合预测
function [PredictData] = QuardFitting(TimeArray, TrackData, Period)
	Array = [(TimeArray.').^2, TimeArray.', ones(length(TimeArray), 1)];
	Coe = (Array.'*Array)^-1*Array.'*TrackData.';
	PredictData = (TimeArray(length(TimeArray)-1)+Period).^2*Coe(1)+ ...
	              (TimeArray(length(TimeArray)-1)+Period)*Coe(2)+Coe(3); 
end

%三次函数拟合预测
function [PredictData] = CubicFitting(TimeArray, TrackData, Period)
	Array = [(TimeArray.').^3, (TimeArray.').^2, TimeArray.', ones(length(TimeArray), 1)];
	Coe = (Array.'*Array)^-1*Array.'*TrackData.';
	PredictData = (TimeArray(length(TimeArray)-1)+Period).^3*Coe(1)+...
				  (TimeArray(length(TimeArray)-1)+Period).^2*Coe(2)+...
	              (TimeArray(length(TimeArray)-1)+Period)*Coe(3)+Coe(4); 
end

%Kalman预测跟踪