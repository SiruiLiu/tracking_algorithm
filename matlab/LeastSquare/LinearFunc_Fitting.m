%% -----------------------------------------------------------
% Least square method, 最小二乘拟合。
% @t， 时间，对应每个角度数据的采集时间
% @AngleData, 角度数据
% @TrackWinWidth, 跟踪窗宽度
%  -----------------------------------------------------------
function [PredictedData] = LinearFunc_Fitting(t, AngleData, TrackWinWidth)
	PredictedData = zeros(1, length(AngleData));
	PredictedData(1:TrackWinWidth) = AngleData(1:TrackWinWidth);
	for i = 1:length(AngleData)-TrackWinWidth+1
		X = t(i:i+TrackWinWidth-1);
		TrackWindow = AngleData(i:i+TrackWinWidth-1).';
		Array = [X(1,:).', ones(length(TrackWindow), 1)];
		
		Coe = (Array.'*Array)^-1*Array.'*TrackWindow;
		PredictedData(i+TrackWinWidth-1) = t(i+TrackWinWidth-1)*Coe(1)+Coe(2);
	end
end

