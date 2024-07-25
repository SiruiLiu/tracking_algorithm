clearvars;
close all;
clc;

figure();
% 参数
t = 0:0.01:4 * pi;

% 初始条件
t0 = 0;
x0 = cos(t0);
y0 = sin(t0);
z0 = t0 * 2;

% 初速度 (可调节)
v0 = 1;

% 时间步长和总时间
dt = 0.01;
T = 4 * pi;

% 初始化轨迹
xt = x0;
yt = y0;
zt = z0;

% 运动模拟
for t = t0:dt:T
    % 速度
    v_xt = -sin(t);
    v_yt = cos(t);
    v_zt = 2;

    % 位置更新
    xt = xt + v_xt * v0 * dt;
    yt = yt + v_yt * v0 * dt;
    zt = zt + v_zt * v0 * dt;

end

hold off;
