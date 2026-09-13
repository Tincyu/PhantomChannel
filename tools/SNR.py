import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib

# ===== 全局字体（与上一张一致）=====
matplotlib.rc("font", family="Times New Roman")

# ===== 数据 =====
orig_snr = np.array([20, 14, 10, 8, 6, 5, 3, 2, 1, 0])[::-1]

ber_group1 = np.array([0.014368,0.048645,0.035057,0.102969,0.197797,0.246983,0.296983,0.350369,0.352011,0.370074])[::-1]
ber_group2 = np.array([0.028439,0.03,0.044048,0.085119,0.168651,0.225952,0.292619,0.332976,0.343006,0.379167])[::-1]
ber_group3 = np.array([0.038793,0.049138,0.073276,0.137069,0.153213,0.253352,0.305172,0.315318,0.371767,0.381466])[::-1]
ber_group4 = np.array([0.025,0.052083,0.0625,0.142361,0.204545,0.291667,0.320582,0.377874,0.384698,0.425493])[::-1]

interp_snr = np.arange(0, 21, 1)

f1 = interp1d(orig_snr, ber_group1)
f2 = interp1d(orig_snr, ber_group2)
f3 = interp1d(orig_snr, ber_group3)
f4 = interp1d(orig_snr, ber_group4)

interp_ber1 = f1(interp_snr)
interp_ber2 = f2(interp_snr)
interp_ber3 = f3(interp_snr)
interp_ber4 = f4(interp_snr)

# ===== 画布大小（与上一张完全一致）=====
fig, ax = plt.subplots(figsize=(9,5))

# ===== 曲线 =====
ax.plot(interp_snr, interp_ber1, 'o-', label='Adv. state Embedding', markersize=14)
ax.plot(interp_snr, interp_ber2, 's--', label='Conn. state Embedding', markersize=14)
# ax.plot(interp_snr, interp_ber3, '^-', label='CTE-Like', markersize=12)
ax.plot(interp_snr, interp_ber4, 'd:', label='PIP Embedding', markersize=14)

# ===== 坐标范围 =====
ax.set_ylim(0, 0.5)
ax.set_xlim(0, 20)
ax.set_xticks(np.arange(0, 21, 1))

# ===== 字体大小（与上一张一致）=====
ax.set_xlabel('SNR (dB)', fontsize=20)
ax.set_ylabel('BER', fontsize=20)

ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='both', direction='in')

# ===== 图例（与上一张一致）=====
ax.legend(fontsize=20)

plt.tight_layout()
plt.show()
