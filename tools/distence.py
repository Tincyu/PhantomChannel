import matplotlib.pyplot as plt
import numpy as np
import statistics
from scipy.interpolate import make_interp_spline
import matplotlib
matplotlib.rc("font",family='Times New Roman')

# 距离（米）作为横坐标
positions = ['1m', '3m', '5m','10m', '20m']
x = np.arange(len(positions))

# BER 和 Reception Rate 数据
ber_list_d = [0.0016395, 0.002635, 0.004635, 0.006635, 0.0108]
loss_list_d = [0.9875, 0.968, 0.96, 0.936, 0.8714]

ber_list_office = [0.002845, 0.005506, 0.007635, 0.00943, 0.02939]
loss_list_office = [0.968, 0.928, 0.904, 0.85, 0.6894]

# 创建图形和双 y 轴
fig, ax1 = plt.subplots(figsize=(9, 5))

# ===== 左侧 y 轴（BER） =====
# ax1.set_xlabel('Distance (m)', fontsize=14)
# ax1.set_ylabel('BER', color='black', fontsize=14)
ax1.set_xlabel('Distance (m)', fontsize=20)
ax1.set_ylabel('BER', color='black', fontsize=20)
# ax1.plot(x, ber_list_d, marker='o', linestyle='-', color='#9bbf8aff', label='BER:Outdoor')
# ax1.plot(x, ber_list_office, marker='D', linestyle='--', color='#82afdaff', label='BER:Office')
ax1.plot(x, ber_list_d, marker='o', linestyle='-', color="#0752f3ff", label='BER: Outdoor', markersize=12)
ax1.plot(x, ber_list_office, marker='D', linestyle='--', color="#f34612ff", label='BER: Office', markersize=12)
ax1.tick_params(axis='y', labelcolor='black')
# ax1.set_ylim(0, max(max(ber_list_d), max(ber_list_office)) * 1.2)
ax1.set_ylim(0, 0.07)
ax1.tick_params(axis='both', direction='in')  # 刻度线朝内
lines1, labels1 = ax1.get_legend_handles_labels()
ax1.legend(lines1, labels1, loc='center left', bbox_to_anchor=(0.0, 0.3),fontsize=20)

# ===== 右侧 y 轴（Reception Rate） =====
ax2 = ax1.twinx()
# ax2.set_ylabel('PSR', color='black', fontsize=14)
ax2.set_ylabel('PSR', color='black', fontsize=20)
# ax2.plot(x, loss_list_d, marker='s', linestyle='-', color='#9bbf8aff', label='PSR:Outdoor')
# ax2.plot(x, loss_list_office, marker='^', linestyle='--', color='#82afdaff', label='PSR:Office')
ax2.plot(x, loss_list_d, marker='s', linestyle='-', color='#0752f3ff', label='PSR: Outdoor', markersize=12)
ax2.plot(x, loss_list_office, marker='^', linestyle='--', color='#f34612ff', label='PSR: Office', markersize=12)
ax2.tick_params(axis='y', labelcolor='black')
ax2.set_ylim(0, 1.05)
ax2.tick_params(axis='both', direction='in')  # 刻度线朝内
# 设置 x 轴标签
plt.xticks(x, positions, fontsize=20)

ax1.tick_params(axis='x', labelsize=20)   # x轴刻度大小
ax1.tick_params(axis='y', labelsize=20)   # 左y轴刻度大小
ax2.tick_params(axis='y', labelsize=20)  

lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines2, labels2, loc='center left', bbox_to_anchor=(0.0, 0.7), fontsize=20)

plt.tight_layout()
# plt.savefig("figure.pdf", format="pdf", bbox_inches="tight")
plt.show()

# 4个位置，每个位置3次测量（示例：丢包率或BER）
# 行为位置，列为测量次数
# data = np.array([
#     [0.005176, 0.013158, 0.002315],   # 位置1
#     [0.061508, 0.0, 0.006349],   # 位置2
#     [0.00085, 0.002976,0.002315],   # 位置3
#     [0.00, 0.001082, 0.000916]    # 位置4
# ])

# # 位置
# data = np.array([
#     [0.005176,0.013158,0.002315],   # A
#     [0.00085,0.002976,0.002315],#  B
#     [0.005176,0.013158,0.002315] # C
# ])

# # 计算每个位置的平均值和方差
# means = np.mean(data, axis=1)
# variances = np.var(data, axis=1, ddof=1)  # 样本方差
# std_devs = np.sqrt(variances)  # 也可以用标准差作为误差条

# # 横坐标标签
# positions = ['A', 'B', 'C']
# x = np.arange(len(positions))

# # 绘制柱状图（带误差条）
# plt.figure(figsize=(6, 4))
# plt.bar(x, means, yerr=variances, capsize=3, color='saddlebrown', edgecolor='black')
# # plt.plot(x, means, marker='o', label='BER', color='blue')

# # 添加标签和标题
# plt.xticks(x, positions)
# plt.ylabel('Rate')
# plt.title('Average Rate with Variance at Different Positions')
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.ylim(0, 0.02) 
# plt.tight_layout()
# plt.show()

# ----------------------------------------------
# 位置柱状图
# positions = ['A', 'B', 'C']

# # 每个位置对应的 BER 和丢包率（示例数据）
# ber_list = [0.006845, 0.002506, 0.002143]
# loss_list = [0.01, 0.03, 0.06, 0.12]  # 丢包率（0~1）

# # 设置柱状图的位置
# x = np.arange(len(positions))
# width = 0.35  # 两组柱子的宽度间距

# # 创建图形
# plt.figure(figsize=(8, 5))
# plt.bar(x - width/2, ber_list, width, label='BER', color='saddlebrown')
# # plt.bar(x + width/2, loss_list, width, label='Packet Loss Rate', color='salmon')

# # 添加标签与标题
# plt.xlabel('Position')
# plt.ylabel('Rate')
# plt.title('BER and Packet Loss Rate at Different Positions')
# plt.xticks(x, positions)
# plt.legend()
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.ylim(0, 0.1) 

# plt.tight_layout()
# plt.show()

# data = [0.005176, 0.013158,0.002315]  # 示例：丢包率或BER等
# var = statistics.variance(data)  # 样本方差
# print(f"方差：{var}")

