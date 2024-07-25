function [trajectory, trajectory_n] = target_trajectory(d_t, vel, period)
    %TARGET_ Summary of this function goes here
    %   Detailed explanation goes here
    t0 = 0;
    t = t0:d_t:period - d_t;
    len = round(period / d_t);
    trajectory = zeros(3, len);
    trajectory_n = zeros(3, len);
    trajectory(:, 1) = [3.3 * cos(t0); t0 * 4; 1.2 * sin(t0)];

    for i = 2:1:len
        vel_dir_x = 3.3 * cos(t(i));
        vel_dir_y = 4;
        vel_dir_z = 1.2 * sin(t(i));

        trajectory(:, i) = [next_pos(trajectory(1, i - 1), vel_dir_x, vel, d_t);
                            next_pos(trajectory(2, i - 1), vel_dir_y, vel, d_t);
                            next_pos(trajectory(3, i - 1), vel_dir_z, vel, d_t)];
    end

    for i = 1:3
        trajectory_n(i, :) = awgn(trajectory(i, :), 30, "measured");
    end

end

function [p] = next_pos(current_pos, vel_dir, vel, d_t)
    p = current_pos + vel_dir * vel * d_t;
end
