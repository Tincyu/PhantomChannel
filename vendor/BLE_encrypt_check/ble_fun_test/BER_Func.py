
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin
from numpy.lib.stride_tricks import as_strided

def decision(freq_dev, sps):
    center_indices = np.arange(sps//2, len(freq_dev), sps);  
    center_freqs = freq_dev[center_indices]
    bits = (center_freqs >0.0).astype(int)  # 假设 >0 为 1，<0 为 0
    return bits

def bits_to_hex_buffer(bits):

    bits = list(bits)  # 确保为列表
    ble_pkt_buffer = ""
    
    for i in range(0, len(bits), 4):
        nibble = bits[i:i+4]
        if len(nibble) < 4:
            nibble = nibble + [0] * (4 - len(nibble))  # 补零
        nibble.reverse()  # 翻转4个bit的顺序
        val = int("".join(str(b) for b in nibble), 2)
        ble_pkt_buffer += format(val, 'x')  # 小写 hex 字符
    
    return ble_pkt_buffer

# Swap bits of a 8-bit value
def swap_bits(value):
    return (value * 0x0202020202  & 0x010884422010) % 1023

# (De)Whiten data based on BLE channel
def dewhitening(data, channel):
  ret = []
  lfsr = swap_bits(channel) | 2

  for d in data:
    d = swap_bits(d)
    for i in 128, 64, 32, 16, 8, 4, 2, 1:
      if lfsr & 0x80:
        lfsr ^= 0x11
        d ^= i

      lfsr <<= 1
      i >>=1
    ret.append(swap_bits(d))

  return ret

def signal_threshold(iq_sample, threshold, min_len):
    amp = np.abs(iq_sample)

    # 找到幅值大于阈值的位置
    above_th = amp > threshold

    # 利用 np.diff 找上升沿和下降沿
    edges = np.diff(above_th.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1

    # 处理边界
    if above_th[0]:
        starts = np.r_[0, starts]
    if above_th[-1]:
        ends = np.r_[ends, len(above_th)]

    # 过滤过短段
    segments = [(s, e) for s, e in zip(starts, ends) if e - s >= min_len]

    # 将每段信号保存到一个数组中
    signal_segments = [iq_sample[s:e] for s, e in segments]
    # print(f"总共有 {len(signal_segments)} 段信号")

    return signal_segments

def gfsk_demodulate(iq_signal, gain):
 
    # 1) 计算瞬时相位
    phase = np.angle(iq_signal)
    # 2) 展开相位，消除 2π 跳变
    phase_unwrapped = np.unwrap(phase)
    # 3) 相位差分
    dphase = np.diff(phase_unwrapped)   
    # 4) 频偏解调
    demod = gain * dphase
    return demod

def result_match(freq_dev, iq_len):

    #test 2M parser:
    samples_per_bit = 1  # 每比特采样点数 
    # 判决为比特
    bits_2M = decision(freq_dev, samples_per_bit)
    access_address_2M, pkt_len_2M, min_match_score_2M = Parser_pkt(bits_2M,'2M', iq_len)


    #test 1M parser:
    samples_per_bit = 2 # 每比特采样点数 
    # 判决为比特
    bits_1M = decision(freq_dev, samples_per_bit)
    access_address_1M, pkt_len_1M, min_match_score_1M = Parser_pkt(bits_1M,'1M', iq_len)

# 这里的采样率写死了，如果变的话，需要传入采样率。目前是4e6
def Parser_pkt(bits, ble_mode, iq_len):
    
    match_error = []

    ble_mod_rate = 2e6  if ble_mode == '2M' else 1e6  
    samples_per_bit = int(2e6 / ble_mod_rate)  
    tail_sample = 1*samples_per_bit*4 

    offset = 1 # 找到的 preamble 对应索引
    srate = 1    # 每 symbol 一个采样
    lap, valid = extract_lap_from_bitstream(bits, offset, srate)

    if lap:
        print("********lap :",hex(lap))

    if valid:
        print(f"经典蓝牙 LAP = 0x{lap:06X}")
    else:
        print("Sync Word 校验失败，不是经典蓝牙")

    if match_error:
        access_address_match, pkt_len_match, min_match_socre = min(match_error, key=lambda x: x[2])
        return access_address_match, pkt_len_match, min_match_socre
    else:
        return None, None, None


# BCH(64,30) 解码 for Bluetooth Sync Word

def bits_to_int(bits):
    """LSB->MSB bits """
    val = 0
    for i, b in enumerate(bits):
        val |= int(b) << i
    return val

def calculate_rssi_dbm(iq_samples, calibration_offset=0.0):

    # 计算每个样本的瞬时功率 (|I + jQ|^2 = I^2 + Q^2)
    instantaneous_power = np.abs(iq_samples) ** 2
    
    # 计算平均功率 (线性值)
    avg_power = np.mean(instantaneous_power)
    
    # 转换为dBFS (相对于满量程的分贝值)
    power_dbfs = 10 * np.log10(avg_power)
    
    # 应用校准偏移得到RSSI (dBm)
    rssi_dbm = power_dbfs + calibration_offset
    
    return rssi_dbm

def generate_pn_sequence():
    """生成经典蓝牙 64-bit PN 序列 LSB->MSB"""
    reg = 0x7FFFFFFFFFFFFFFF  # 初始状态 63-bit 全1
    pn = []
    for _ in range(63):
        pn.append(reg & 1)
        new_bit = ((reg >> 0) ^ (reg >> 1) ^ (reg >> 3) ^ (reg >> 4)) & 1
        reg = (reg >> 1) | (new_bit << 62)
    pn.append(0)  # 扩展到64位
    return pn

def compute_remainder(value, g):
    """二进制多项式除法余式计算。"""
    remainder = value
    g_len = g.bit_length()
    while remainder.bit_length() >= g_len:
        shift = remainder.bit_length() - g_len
        remainder ^= g << shift
    return remainder

def build_bluetooth_sync_word(lap, bch_poly=None):
    """根据 LAP 生成经典蓝牙 64-bit sync word，bit0 为传输顺序中的第一位。"""
    if bch_poly is None:
        # Bluetooth BCH(64,30) expurgated block code:
        # g(D) = (1 + D) * g'(D), g'(D)=0x37CD0EB67
        bch_poly = 0o260534236651

    pn = 0x83848D96BBCC54FC
    lap &= 0xFFFFFF
    barker = 0x13 if ((lap >> 23) & 1) else 0x2c
    x = (barker << 24) | lap

    # 信息字段先和 PN 的 information covering 部分异或；BCH 系统码
    # parity = (xtilde * D^34) mod g(D)。
    xtilde = (pn >> 34) ^ x
    parity = compute_remainder(xtilde << 34, bch_poly)
    codeword = parity | (xtilde << 34)
    return codeword ^ pn

def extract_lap_from_bitstream(binbuf, offset, srate, bch_poly=None, pn_seq=None):
    """
    从捕获 bit 流中提取 LAP 并验证经典蓝牙 Sync Word

    参数:
        binbuf : list/array of 0/1, 捕获 bit 流
        offset : int, preamble 或 sync word 开始索引
        srate  : 每 symbol 采样数
        bch_poly: BCH 生成多项式 (整数)，默认 Bluetooth g(D)=0o260534236651
        pn_seq : 保留兼容旧调用，当前使用规范固定 PN 常量
    返回:
        lap_int : LAP (整数)
        valid   : 校验是否通过（True=经典蓝牙）
    """
    if bch_poly is None:
        bch_poly = 0o260534236651  # Bluetooth BCH(64,30) g(D)

    # 1. 提取 Barker (6-bit)
    barker_bits = binbuf[offset + 62*srate : offset + 62*srate + 6]
    barker = bits_to_int(barker_bits)
    # print("barker is:",bin(barker))

    if barker not in (0x13, 0x2c):
        return None, False

    # 2. 提取 LAP (24-bit)
    lap_bits = binbuf[offset + 38*srate : offset + 38*srate + 24]
    lap = bits_to_int(lap_bits)
    # print("lap is:",bin(lap))
    # print("lap is:",lap_bits)

    # 3. 提取 BCH code (34-bit)
    code_bits = binbuf[offset + 4*srate : offset + 4*srate + 34]
    code = bits_to_int(code_bits)


    # 4. 根据 LAP MSB 重算 Barker
    lap_msb = (lap >> 23) & 1
    barker_true = 0x13 if lap_msb else 0x2c

    # 5. 重算 64-bit sync word
    awfinal = build_bluetooth_sync_word(lap, bch_poly)

    # 捕获的 AW
    aw_captured = (barker << 58) | (lap << 34) | code
    # print("awfinal is:",hex(awfinal),"aw_captured is :",hex(aw_captured))

    valid = (awfinal == aw_captured)
    return lap, valid

def find_sync_start(bits_1M):

    patterns = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]

    n = len(bits_1M)

    for i in range(n - 3):
        window = bits_1M[i:i+4]
        if window == patterns[0] or window == patterns[1]:
            return i

    return -1
