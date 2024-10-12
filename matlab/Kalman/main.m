% A target tracking simulation program, this is the main script that
% defines the parameters
clear;
close all;
clc;
time_durationo = 200; % duration, secons
target_vel = 1; % velocity, m/s
d_t = 0.1; % time slice

initial_pos = [13, -26, 2]; % Our device's initial position
vel = 6.0; % Our device's velocity

CovarianceMatVal = 1;
ProcessVar = 0.3;
ObservingVar = 0.5;

useAcc = true;
useRandomAcc = false;

[t_traj, obsv_traj] = target_trajectory(d_t, target_vel, time_durationo);
figure(1);
plot3(t_traj(1, :), t_traj(2, :), t_traj(3, :));
grid on;
hold on;
plot3(obsv_traj(1, :), obsv_traj(2, :), obsv_traj(3, :));
plot3(initial_pos(1), initial_pos(2), initial_pos(3), 'ro');

device_traj = tracking(initial_pos, vel, d_t, obsv_traj, ...
    CovarianceMatVal, ProcessVar, ObservingVar, ...
    useAcc, useRandomAcc);

plot3(device_traj(1, :), device_traj(2, :), device_traj(3, :), "r*");
legend('目标轨迹', '观测轨迹', '追踪器初始位置', '追踪轨迹');
title('追踪仿真');
