import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# 获取系统中所有字体
font_list = fm.fontManager.ttflist
# 筛选包含简体中文的字体
# 强制添加字体路径
font_path = 'E:\\Project\\Airport\\ttf\\wqy-zenhei.ttf'
fm.fontManager.addfont(font_path)
# 设置字体
plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
# 列出所有被 matplotlib 加载的字体
loaded_fonts = fm.fontManager.ttflist
# 常量和参数
initial_dist = 20e3  # 初始距离 (单位: 米)
c = 299_792_458.0  # 光速 (单位: 米/秒)
vel = 340.0 * 2  # 目标速度 (单位: 米/秒)
dt = 20.0e-3  # 时间间隔 (单位: 秒)
f0 = 10.0e9  # 中心频率 (单位: Hz)
delta_fi = np.array([8550000, 8550600, 8553600, 8571600, 8679600])  # 一次频差 (单位: Hz)
d_delta_fi = np.diff(delta_fi)  # 二次频差 (单位: Hz)
fi = delta_fi + f0  # 每个频率f_i (单位: Hz)
lambda_f0 = c / f0  # 中心频率波长 (单位: 米)
lambda_fi = c / fi  # 每个f_i的波长 (单位: 米)
duration = 50  # 仿真点数

def phase_unwrap(phi):
    return np.mod(phi+np.pi, 2*np.pi)-np.pi

# 目标距离计算
tar_dist = np.zeros(duration)
for i in range(duration):
    tar_dist[i] = initial_dist + i * dt * vel  # 随时间变化的距离

measured_dist =-np.zeros(duration)
measured_dist[:20] = tar_dist[:20] + np.random.uniform(-400, 400, size=(1, duration-30))
measured_dist[20:] = tar_dist[20:] + np.random.uniform(-150, 150, size=(1, duration-20))
print(np.size(measured_dist))

# 计算理论接收信号的相位
# 对于中心频率f0
frac_part_0, _ = np.modf(2 * tar_dist / lambda_f0)  # 信号传播的双程距离与中心波长的比值
phi_0 = frac_part_0 * 2 * np.pi  # 对应的相位（范围: 0到2π）

# 对于每个fi的相位
frac_part_i, _ = np.modf(2 * tar_dist[:5] / lambda_fi)  # 只计算前5个频率的相位
phi_i = frac_part_i * 2 * np.pi

# 计算一次相差 (delta_phi)
delta_phi = phi_i - phi_0[:5]  # 每个频率与中心频率的相位差
delta_phi = np.mod(delta_phi + np.pi, 2 * np.pi) - np.pi  # 将相位差约束在 (-π, π) 之间
print("一次相位差 (delta_phi):", delta_phi)

# 计算二次相差 (d_delta_phi)
d_delta_phi = np.diff(delta_phi)  # 相位差的一阶差分，得到二次相位差
print("二次相位差 (d_delta_phi):", d_delta_phi)

# 计算等效波长
lambda_equivalent = c / d_delta_fi  # 使用二次频差计算等效波长
print("等效波长 (lambda_equivalent):", lambda_equivalent)

# 理论上根据二次频差计算的二次相位差
phi_equivalent = 2 * tar_dist[1] / lambda_equivalent[0] * 2 * np.pi
print("二次频差对应的理论相位差 (phi_equivalent):", phi_equivalent)

# 第一次距离解算
dd_phi = phase_unwrap(d_delta_phi[0])
phi_comp = phase_unwrap(delta_fi[0]/d_delta_fi[0]*vel*dt)
R = (dd_phi+phi_comp)*c/(4*np.pi*d_delta_fi[0])
print(R)

measured_dist[0] = R

for i in range(1, 4):
    dd_phi = phase_unwrap(d_delta_phi[i])
    phi_comp = phase_unwrap(delta_fi[i]/d_delta_fi[i]*vel*dt)
    r_pred = R + vel*dt
    dd_phi_ideal = 4*np.pi*r_pred*d_delta_fi[i]/c
    N = np.floor(dd_phi_ideal/(2*np.pi))
    print(N)
    R = (dd_phi+phi_comp+2*np.pi*N)*c/(4*np.pi*d_delta_fi[i])
    print(R)
    if i == 1:
        measured_dist[i] = R


lambda_equivalent_1 = c/delta_fi
print(lambda_equivalent_1)
frac_part, _ = np.modf(tar_dist[:5], lambda_equivalent_1)
delta_phi_1 = 2*np.pi*frac_part
print(delta_phi)

# 绘制测距结果
fig, ax = plt.subplots()
ax.plot(tar_dist, 'b', label='实际距离', marker='o')
ax.plot(measured_dist, 'r', label='测量距离', marker='x')
ax.legend()
plt.savefig("./images/ekf_img/dist.jpg")
plt.show()
