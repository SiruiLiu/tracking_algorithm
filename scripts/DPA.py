import numpy as np
import matplotlib.pyplot as plt

# 假设一些初始目标状态
target_pos = np.array([1000, 2000])  # 目标初始位置
target_vel = np.array([5, -3])  # 目标初始速度
dt = 1.0  # 时间间隔

# 假设一些测量数据（包括目标和杂波）
measurements = np.array([
    [1005, 1997],
    [1010, 1994],
    [950, 2050],  # 杂波
    [1015, 1991]
])

# 计算预测位置
predicted_pos = target_pos + target_vel*dt


# 计算关联概率
def calc_association_prob(predicted_pos, measurements, noise_cov):
    probs = []
    for z in measurements:
        diff = z - predicted_pos
        exponent = -0.5 * np.dot(diff.T, np.linalg.inv(noise_cov).dot(diff))
        probs.append(np.exp(exponent))
    probs = np.array(probs)
    probs /= probs.sum()  # 归一化为概率
    return probs


# 噪声协方差矩阵
noise_cov = np.array([[25, 0], [0, 25]])

# 计算关联概率
probs = calc_association_prob(predicted_pos, measurements, noise_cov)

# 使用关联概率更新目标位置
updated_pos = np.sum(probs[:, np.newaxis] * measurements, axis=0)

print("预测位置:", predicted_pos)
print("更新后位置:", updated_pos)

# 绘制结果
plt.scatter(measurements[:, 0], measurements[:, 1], c='r', label='测量数据')
plt.scatter(target_pos[0], target_pos[1], c='g', label='真实位置')
plt.scatter(predicted_pos[0], predicted_pos[1], c='b', label='预测位置')
plt.scatter(updated_pos[0], updated_pos[1], c='y', label='更新后位置')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('PDA Algorithm for Radar Tracking')
plt.show()
