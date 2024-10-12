function [device_traj] = tracking(initial_pos, vel, d_t, observed_pos, ...
        CovarianceMatVal, ProcessVar, ...
        ObservingVar, useAcc, useRandomAcc)

    if (useAcc)
        %% With accelerate
        A = [1, 0, 0, d_t, 0, 0, 0.5 * d_t ^ 2, 0, 0; ...
                 0, 1, 0, 0, d_t, 0, 0, 0.5 * d_t ^ 2, 0; ...
                 0, 0, 1, 0, 0, d_t, 0, 0, 0.5 * d_t ^ 2; ...
                 0, 0, 0, 1, 0, 0, d_t, 0, 0; ...
                 0, 0, 0, 0, 1, 0, 0, d_t, 0; ...
                 0, 0, 0, 0, 0, 1, 0, 0, d_t; ...
                 0, 0, 0, 0, 0, 0, 1 0, 0; ...
                 0, 0, 0, 0, 0, 0, 0, 1, 0; ...
                 0, 0, 0, 0, 0, 0, 0, 0, 1]; % -- State transforming matrix
        H = [1, 0, 0, 0, 0, 0, 0, 0, 0;
             0, 1, 0, 0, 0, 0, 0, 0, 0;
             0, 0, 1, 0, 0, 0, 0, 0, 0];
        ProcessVarMat = eye(9) * ProcessVar;
        ObservVarMat = eye(3) * ObservingVar;
        initial_states = [cal_relative_pos(initial_pos.', observed_pos(:, 1)); 0; 0; 0; 0; 0; 0];
        CovarianceMatrix = eye(9) * CovarianceMatVal;
    else
        %% Without accelerate
        A = [1, 0, 0, d_t, 0, 0; ...
                 0, 1, 0, 0, d_t, 0; ...
                 0, 0, 1, 0, 0, d_t; ...
                 0, 0, 0, 1, 0, 0; ...
                 0, 0, 0, 0, 1, 0; ...
                 0, 0, 0, 0, 0, 1];
        H = [1, 0, 0, 0, 0, 0;
             0, 1, 0, 0, 0, 0;
             0, 0, 1, 0, 0, 0];

        ProcessVarMat = eye(6) * ProcessVar;
        ObservVarMat = eye(3) * ObservingVar;
        initial_states = [cal_relative_pos(initial_pos.', observed_pos(:, 1)); 0; 0; 0];
        CovarianceMatrix = eye(6) * CovarianceMatVal;
    end

    States = initial_states;
    device_pos = initial_pos.';
    device_traj = zeros(3, size(observed_pos, 2));
    device_traj(:, 1) = States(1:3);

    for i = 2:size(observed_pos, 2)
        r_pos = cal_relative_pos(device_pos, observed_pos(:, i));

        [States, P] = EKF(States, r_pos, d_t, ...
            A, CovarianceMatrix, H, ...
                ProcessVarMat, ObservVarMat, useRandomAcc);

        CovarianceMatrix = P;
        relative_pos = States(1:3);
        dir = relative_pos / norm(relative_pos);
        device_pos = device_pos + dir * vel * d_t;
        device_traj(:, i) = device_pos;
    end

end

function [r_pos] = cal_relative_pos(pos, target_pos)
    r_pos = target_pos - pos;
end
