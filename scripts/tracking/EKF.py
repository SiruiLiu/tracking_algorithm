import numpy as np
import matplotlib.pyplot as plt

# 初始状态
x = np.array([[0], [0], [10], [np.pi / 4], [0.1]])  # [x, y, v, theta, omega]
P = np.eye(5)  # 初始协方差矩阵
Q = np.eye(5) * 0.1  # 过程噪声协方差矩阵
R = np.eye(2) * 0.1  # 观测噪声协方差矩阵

# 时间步长
dt = 1.0


# 非线性状态转移函数
def f(x, dt):
    x_pos = x[0, 0]
    y_pos = x[1, 0]
    v = x[2, 0]
    theta = x[3, 0]
    omega = x[4, 0]

    if np.abs(omega) > 1e-5:
        x_pos_new = x_pos + v / omega * (np.sin(theta + omega*dt) - np.sin(theta))
        y_pos_new = y_pos + v / omega * (np.cos(theta) - np.cos(theta + omega*dt))
    else:
        x_pos_new = x_pos + v * dt * np.cos(theta)
        y_pos_new = y_pos + v * dt * np.sin(theta)

    theta_new = theta + omega*dt

    return np.array([[x_pos_new], [y_pos_new], [v], [theta_new], [omega]])


# 雅可比矩阵
def F_jacobian(x, dt):
    v = x[2, 0]
    theta = x[3, 0]
    omega = x[4, 0]

    if np.abs(omega) > 1e-5:
        F = np.array([[
            1, 0, (np.sin(theta + omega*dt) - np.sin(theta)) / omega,
            v * (np.cos(theta + omega*dt) - np.cos(theta)) / omega,
            v * (np.sin(theta) - np.sin(theta + omega*dt)) / omega**2 + v * dt * np.cos(theta + omega*dt) / omega
        ],
                      [
                          0, 1, (np.cos(theta) - np.cos(theta + omega*dt)) / omega,
                          v * (np.sin(theta + omega*dt) - np.sin(theta)) / omega,
                          v * (np.cos(theta + omega*dt) - np.cos(theta)) / omega**2 +
                          v * dt * np.sin(theta + omega*dt) / omega
                      ], [0, 0, 1, 0, 0], [0, 0, 0, 1, dt], [0, 0, 0, 0, 1]])
    else:
        F = np.array([[1, 0, dt * np.cos(theta), -v * dt * np.sin(theta), 0],
                      [0, 1, dt * np.sin(theta), v * dt * np.cos(theta), 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, dt],
                      [0, 0, 0, 0, 1]])

    return F


# 观测函数
def h(x):
    return np.array([[x[0, 0]], [x[1, 0]]])


# 观测矩阵
def H_jacobian(x):
    return np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])


# 扩展卡尔曼滤波主循环
num_steps = 50
x_hist = np.zeros((5, num_steps))
z_hist = np.zeros((2, num_steps))

for i in range(num_steps):
    # 预测步骤
    x = f(x, dt)
    print(x.shape)
    F = F_jacobian(x, dt)
    P = F @ P @ F.T + Q

    # 生成模拟观测值
    z = h(x) + np.sqrt(R) @ np.random.randn(2, 1)

    # 更新步骤
    H = H_jacobian(x)
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    x = x + K @ (z - h(x))
    P = (np.eye(5) - K@H) @ P

    # 保存历史记录
    x_hist[:, i] = x.flatten()
    z_hist[:, i] = z.flatten()

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(x_hist[0, :], x_hist[1, :], label='Estimated trajectory')
plt.scatter(z_hist[0, :], z_hist[1, :], color='red', marker='x', label='Measurements')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('CTRV EKF Tracking')
plt.legend()
plt.grid(True)
plt.show()

print("Final estimated state:\n", x)
print("Final estimated covariance:\n", P)
