import numpy as np
from matplotlib import pyplot as plt

cz_width = 20e3
cz_length = 40e3
dir = 0
p_res = 20  # 分辨率，即两点之间的间隔为20米
point_num = 100  # 弧线的点的个数


def rotate(x, y, angle):
    x_rotated = x * np.cos(angle) - y * np.sin(angle)
    y_rotated = x * np.sin(angle) + y * np.cos(angle)
    return x_rotated, y_rotated


def create_airport_clear_zone(width, length, dir, res, point_num):
    half_width = cz_width * 0.5
    half_length = cz_length * 0.5
    radius = half_width
    strip_dir_rad = dir / 180 * np.pi

    top_line_x = np.linspace(-half_length, half_length, round(cz_length / p_res) + 1)
    top_line_y = np.ones(top_line_x.shape) * half_width
    bottom_line_x = top_line_x
    bottom_line_y = -1 * top_line_y

    radians = np.linspace(-0.5 * np.pi, 0.5 * np.pi, point_num)
    curve1_x = np.array([radius * np.cos(radian) + half_length for radian in radians])
    curve1_y = np.array([radius * np.sin(radian) for radian in radians])
    curve2_x = -1 * curve1_x
    curve2_y = curve1_y

    top_line_x, top_line_y = rotate(top_line_x, top_line_y, strip_dir_rad)
    bottom_line_x, bottom_line_y = rotate(bottom_line_x, bottom_line_y, strip_dir_rad)
    curve1_x, curve1_y = rotate(curve1_x, curve1_y, strip_dir_rad)
    curve2_x, curve2_y = rotate(curve2_x, curve2_y, strip_dir_rad)

    return top_line_x, top_line_y, bottom_line_x, bottom_line_x, bottom_line_y, curve1_x, curve1_y, curve2_x, curve2_y


top_line_x, top_line_y, bottom_line_x, bottom_line_x, bottom_line_y, curve1_x, curve1_y, curve2_x, curve2_y = create_airport_clear_zone(
    cz_width, cz_length, dir, p_res, point_num
)

print(curve1_x[0], curve1_y[0], curve1_x[-1], curve1_y[-1])

fig, ax = plt.subplots()
ax.plot(top_line_x, top_line_y, label='Top Line')
ax.plot(bottom_line_x, bottom_line_y, label='Bottom Line')

# 绘制弧线
ax.plot(curve1_x, curve1_y, label='Curve 1')
ax.plot(curve2_x, curve2_y, label='Curve 2')

# 设置图例
ax.legend()

plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Plot with Straight Lines and Curves')
plt.show()
