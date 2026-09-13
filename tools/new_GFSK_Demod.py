import numpy as np
from scipy.signal.windows import gaussian
from scipy import signal
import matplotlib.pyplot as plt
import os
import glob
import struct

# === 参数设置 ===
sample_rate = 4e6        # 采样率（Hz）
freq_dev    = 250e3      # BLE GFSK ±250kHz
ble_mod_rate = 2e6 #2Mbps
samples_per_bit = int(sample_rate / ble_mod_rate)  # 每比特采样点数 

real_addr_lack_p = []
# GFSK解调增益
gain = sample_rate / (2 * np.pi * freq_dev)


# IQ 文件目录
# data_dir = "./iq_captures_reduce"
data_dir = "./iq_captures"
pattern = os.path.join(data_dir, "iq_packet_00*.npy")

# BER write dir
result_dir = "./BER/result/5.txt"

# 遍历所有 .npy 文件
file_list = sorted(glob.glob(pattern))
if not file_list:
    print(f"未找到文件: {pattern}")
    exit(1)

def reverse_bytes_every_two(b: bytes) -> bytes:
    if len(b) % 2 != 0:
        b += b'\x00'  # 补零

    reversed_pairs = []
    for i in range(0, len(b), 2):
        pair = b[i:i+2]
        reversed_pairs.extend([pair[1], pair[0]])  # 交换两个字节顺序
    
    return bytes(reversed_pairs)

def write_to_txt(input_str: str, length: int, filename):
    truncated_str = input_str[:length*2]
    
    bytes_list = [truncated_str[i:i+2] for i in range(0, len(truncated_str), 2)]
    
    # 按指定长度分组
    grouped_lines = [bytes_list[i:i+length] for i in range(0, len(bytes_list), length)]
    
    with open(filename, 'a') as f:
        for group in grouped_lines:
            line = ' '.join(group)
            f.write(line + '\n')

def write_ble_packet_to_pcap(bits_hex_str, read_more, filename):
    # """
    # 将给定 BLE 包（字符串形式）写入 pcap 文件

    # 参数:
    #     bits_hex_str: str，十六进制字符串，每两个字符为一个字节，例如 '55557ad948b450d8...'
    #     read_more: 判断是否读取额外字节的数据
    #     filename: str，输出的 pcap 文件名
    # """
    # 处理输入，转换成字节序列
    # if len(bits_hex_str) % 2 != 0:
    #     bits_hex_str = "a"+bits_hex_str
    # for i in range(0,3):
    #     if addr_start == i:
    #         bits_hex_str = "a"*(4-i)+bits_hex_str
    print("pkt bits_hex_str is ",bits_hex_str)
    byte_list = [bits_hex_str[i:i+2] for i in range(0, len(bits_hex_str), 2)]
    reversed_byte_strs = [b[1] + b[0] for b in byte_list if len(b) == 2]
    data_init = bytes(int(b, 16) for b in reversed_byte_strs)
    # print("翻转 data_init is ",data_init.hex())
    data = data_init[:]
    print("pkt data is ",data.hex())
    if read_more:
        write_to_txt(data.hex(), 25, result_dir)

    # 提取 Access Address
    access_address = data[:4]
    print("Access Address:", access_address.hex())

    # 提取 Header（2 bytes）
    header = data[4:6]

    # 获取 Payload 长度（Header 第一个字节的低6位）
    header_byte0 = header[1] if read_more==0 else 16
    payload_len = header_byte0 & 0x3F
    print("Payload Length:", payload_len)

    # 提取 Payload
    payload_start = 6
    payload = data[payload_start:payload_start + payload_len]

    # 提取 CRC（3 bytes）
    crc_start = payload_start + payload_len
    crc = data[crc_start:crc_start + 3]

    # 构造 BLE 报文
    ble_packet = access_address + header + payload + crc

    # 检查是否第一次写入
    file_exists = os.path.exists(filename)
    mode = "ab"
    ref_aa = int.from_bytes(access_address, byteorder='little')  # 注意是小端

    with open(filename, mode) as f:
        if not file_exists:
            # 写入 pcap 全局头，linktype = 256（LINKTYPE_BLUETOOTH_LE_LL）
            f.write(struct.pack(
                "<IHHIIII",
                0xa1b2c3d4,   # magic number
                2, 4,         # version major, minor
                0, 0,         # timezone, sigfigs
                65535,        # snaplen
                256           # LINKTYPE_BLUETOOTH_LE_LL
            ))

        # 每个包都要构造 pseudo_header
        pseudo_header = struct.pack(
            "<BBBBIB",
            1,                 # channel
            0xff,               # signal power
            0xff,               # noise power
            0,                  # offensive flag
            ref_aa,             # reference access address
            1                   # CRC OK
        )

        # 加上伪头的完整数据包
        packet = pseudo_header + ble_packet

        # 写入 pcap 包头（timestamp、长度）
        ts_sec, ts_usec = 0, 0
        incl_len = len(packet)
        orig_len = incl_len
        f.write(struct.pack("<IIII", ts_sec, ts_usec, incl_len, orig_len))
        f.write(packet)

    print(f"[+] BLE packet written to '{filename}' ({len(ble_packet)} bytes)")


def gfsk_modulate(bits, sps=8, bt=0.5, freq_dev=250e3, fs=4e6):
    # """
    # bits: 比特列表，如 [1,0,1,0...]
    # sps: 每比特的采样数（samples per symbol）
    # bt: 高斯滤波器带宽时间积
    # freq_dev: 频率偏移（Hz）
    # fs: 采样率（Hz）
    # """
    N = len(bits)
    bit_vals = np.array(bits) * 2 - 1  # [1, -1, 1, -1,...]
    data_upsampled = np.repeat(bit_vals, sps)

    # 高斯滤波器
    span = 4  # 滤波器跨度（单位bit）
    n = span * sps
    t = np.linspace(-span/2, span/2, n)
    h = gaussian(n, std=sps * bt)
    h /= np.sum(h)
    filtered = signal.convolve(data_upsampled, h, mode='same')

    # 积分成相位（单位弧度）
    dt = 1/fs
    freq = freq_dev * filtered  # Hz
    phase = 2 * np.pi * np.cumsum(freq) * dt

    # 生成复数IQ信号
    iq = np.exp(1j * phase)
    return iq

def gfsk_demodulate(iq_signal):
 
    # 1) 计算瞬时相位
    phase = np.angle(iq_signal)
    # 2) 展开相位，消除 2π 跳变
    phase_unwrapped = np.unwrap(phase)
    # 3) 相位差分
    dphase = np.diff(phase_unwrapped)   
    # 4) 频偏解调
    demod = gain * dphase
    return demod


# --------互相关匹配
# rx_iq 接收到的样本 npy file
# template_iq 本地生成信号


def estimate_preamble_position(iq_samples, template_iq, fs):
    # """
    # 估计前导码（如 0xAA 0xAA）在接收 IQ 数据中的起始位置

    # 参数:
    #     iq_samples: 接收到的 IQ 样本（复数 numpy 数组）
    #     template_iq: 本地生成的 GFSK 模板 IQ 信号
    #     fs: 采样率（单位：Hz）eg.fs=1e6

    # 返回:
    #     est_index: 精确估计的前导码起始位置（浮点型，可含小数表示亚采样精度）
    #     peak_value: 最大相关值
    #     peak_phase: 最大相关点的相位（用于估计精度）
    # """

    # 1. 互相关（template 需取共轭反转）
    corr = signal.correlate(iq_samples, template_iq.conj(), mode='valid')

    # 2. 寻找最大相关值的位置（初始估计）
    peak_index = np.argmax(np.abs(corr))
    peak_value = np.abs(corr[peak_index])
    peak_phase = np.angle(corr[peak_index])

    # 3. 细化估计：使用相关值前后做抛物线插值或相位微调
    # 这里使用简单的相位微调（可选更复杂拟合）
    # 相位差 Δφ 推测时间偏移 Δt ≈ φ / (2πf)
    symbol_duration = len(template_iq) / fs
    phase_offset = peak_phase / (2 * np.pi)
    time_offset = phase_offset * symbol_duration

    # 4. 返回最终位置（亚采样精度）
    est_index = peak_index - time_offset * fs

    return est_index, peak_value, peak_phase

def preamble_pos_eval(samples_per_bit):
    #估计起始的preamble位置
    #暂态延迟占一部分
    # 参数:
    #     samples_per_bit: 每几位表示1bit

    # 返回:
    #     pos: 可能的起始位置，跳过部分暂态延迟的sample
    return 3*(8*samples_per_bit)+3*samples_per_bit - 8

# 假设每bit占sps个采样点，我们在每bit中取中心采样点的频偏判断0/1
def decision(freq_dev, sps):
    center_indices = np.arange(sps//2, len(freq_dev), sps)
    center_freqs = freq_dev[center_indices]
    threshold = np.mean(center_freqs)
    #2M模式4Mbps采样率的最佳经验阈值为0.36
    #2M模式8Mbps采样率的最佳经验阈值为0.36
    # print("threshold is ",threshold)
    bits = (center_freqs > 0.35).astype(int)  # 假设 >0 为 1，<0 为 0
    return bits

# 生成前导码比特
bits = [0,1,0,1,0,1,0,1]  # 55 55
fs = sample_rate         # 采样率5MHz
sps = 2 if samples_per_bit == 2 else 4       # 每bit 2个采样点
# sps = 4
bt = 0.5         # 高斯滤波器带宽时间积

# # 生成本地参考信号
template_iq = gfsk_modulate(bits, sps=sps, bt=bt, freq_dev=freq_dev, fs=fs)
# print(template_iq)

# demod_iq = gfsk_demodulate(template_iq)
# print(demod_iq)

# plt.plot(np.real(template_iq[:200]), label='I')
# plt.plot(np.imag(template_iq[:200]), label='Q')
# plt.title("GFSK Modulated IQ of 0xAA 0xAA")
# plt.legend()
# plt.grid(True)
# plt.show()
def find_ble_preamble(bit_sequence, rate):
    # """
    # 查找BLE前导码起始位置

    # 参数:
    #     bit_sequence: list[int] or np.ndarray，包含0和1的比特序列
    #     rate: str，'1M' 或 '2M'，代表BLE速率模式

    # 返回:
    #     int or None，最佳匹配的起始位置（start），若无匹配则返回 None
    # """
    bit_sequence = np.array(bit_sequence)
    
    if rate == '1M':
        pattern_len = 8
        patterns = {
            '0x55': np.array([0,1]*4),
            '0xAA': np.array([1,0]*4)
        }
    elif rate == '2M':
        pattern_len = 16
        patterns = {
            '0x5555': np.array([0,1]*8),
            '0xAAAA': np.array([1,0]*8)
        }
    else:
        raise ValueError("速率只能为 '1M' 或 '2M'")

    results = []

    for i in range(len(bit_sequence) - pattern_len):
        sub = bit_sequence[i:i+pattern_len]
        if i + pattern_len >= len(bit_sequence):
            break
        next_bit = bit_sequence[i + pattern_len]

        for name, pattern in patterns.items():
            errors = int(np.sum(sub != pattern))
            if next_bit != sub[-1]:  # 访问地址第1位需等于前导码最后1位
                results.append((i, errors))

    if not results:
        return 0
    
    # 按误码数排序，返回最佳匹配的 start
    return sorted(results, key=lambda x: x[1])[0][0]

def rotate_right_1bit(hex_str: str) -> str:
    # """将输入的1字节十六进制字符串循环右移1位，返回十六进制字符串"""
  
    x = int(hex_str, 16) & 0xFF
    # 循环右移1位
    result = ((x >> 1) | (x << 7)) & 0xFF
    # 转回十六进制字符串
    return f"{result:02x}"

def bits_to_hex_buffer(bits):
    # """
    # 将比特序列每4位翻转后转为16进制字符拼接，返回字符串

    # 参数:
    #     bits: list[int] or np.ndarray

    # 返回:
    #     str，16进制编码后的结果
    # """
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

def count_mismatched_bits_with_tolerance(A, B, error=4):
    # """
    # 计算两个4字节十六进制列表之间的错误得分。
    
    # 参数:
    #     A, B: list[str]，如 ['7e', 'bd', 'ee', 'ba']
    #     error: int，最大允许的比特误差（仅用于完全无字节匹配时）

    # 返回:
    #     float，错误得分（越低表示匹配越好，越高表示不匹配）
    # """
    if len(A) != 4 or len(B) != 4:
        if len(A) != 4:
            A = A.append('00')
        # raise ValueError("必须是两个长度为4的十六进制字节列表")

    exact_match = 0
    mismatch_bit_sum = 0

    for a_hex, b_hex in zip(A, B):
        a = int(a_hex, 16)
        b = int(b_hex, 16)
        if a == b:
            exact_match += 1
        else:
            mismatch_bit_sum += bin(a ^ b).count('1')

    # 全匹配，最小错误得分
    if exact_match == 4:
        return 0.0  

    # 如果至少有一个字节匹配，按匹配数量给予分数（0~3），越少越差
    if exact_match > 0:
        return 4 - exact_match + (mismatch_bit_sum / error)  # 越少匹配字节，得分越高

    # 如果完全没有匹配，检查 bit-level 错误是否可接受
    if mismatch_bit_sum > error:
        return 10.0  # 误差太大，置信度极低，赋予最大惩罚
    else:
        return 5.0 + (mismatch_bit_sum / error)  # 在 5.0~6.0 之间浮动

for filepath in file_list:

    found = False
    
    # 加载复数 IQ 数据
    iq_samples = np.load(filepath)  # dtype=np.complex64
    preamble_pos_init = preamble_pos_eval(samples_per_bit)
    est_index, peak_value, peak_phase = estimate_preamble_position(iq_samples[preamble_pos_init:preamble_pos_init+32*(samples_per_bit)], template_iq, sample_rate)
    print(est_index, peak_value, peak_phase)

    if samples_per_bit == 2:
        ble_rate = 2
        
        start_index = int(np.round(est_index)) + preamble_pos_init
        print("start_index is :",start_index)
        if(start_index > 54*samples_per_bit/2 + 4): #理论上nordic在2Mbps下，采样率4Mhz下，暂态延迟为54个采样点左右
            start_index = int(52*samples_per_bit/2)
    else:
        ble_rate = 1
        start_index = int(np.round(est_index))
        print("start_index is :",start_index)
        start_index = 54

    iq_signal = iq_samples[start_index:]

    # 相位补偿
    iq_signal = iq_signal * np.exp(-1j * peak_phase)

    # 解调为频偏
    freq_dev = gfsk_demodulate(iq_signal)

    # 判决为比特
    sps = 2  if samples_per_bit == 2  else 4 # 每比特2个采样点
    bits = decision(freq_dev, sps)
    print(bits,len(bits),type(bits))

    preamle_start = find_ble_preamble(bits[:ble_rate*8+8],'2M') #2M下2字节，16bit,8作为冗余长度
    print("preamle_start pos is ", preamle_start )

    #以地址作为判断因素 ,通常只需要第一次判断即可
    # 假的地址addr_fake= addr3+addr2+addr1+addr0
    # 真的地址addr_real= addr0+addr1+addr2+addr3
    pkt_buff = []
    match_error = [] # 每次偏移中，匹配错误得分，分越低，匹配度越高
    match_error_r = [] #包不完整时，用于计算不完整包的得分
    for i in range(0,4): 
        if ble_rate == 2:
            
            #bits_to_hex_buffer(bits[preamle_start-i:])
            #if(count_mismatched_bits(A, B):)
            # print(bits_to_hex_buffer(bits[preamle_start+i:]))
            buffer = bits_to_hex_buffer(bits[preamle_start+i:])
            print(buffer)
            for j in range(0,6): #最多偏移6个字节
                fake_addr = [rotate_right_1bit(buffer[j+6:j+8]),
                                  rotate_right_1bit(buffer[j+4:j+6]),
                                  rotate_right_1bit(buffer[j+2:j+4]),
                                  rotate_right_1bit(buffer[j:j+2])]

                # print("fake_addr is ",fake_addr)
                k = j + 12+2* ble_rate     #k : real addr pos 
                real_addr = [buffer[k:k+2],buffer[k+2:k+4],buffer[k+4:k+6],buffer[k+6:k+8]]
                # print("real_addr is ",real_addr)
                score = count_mismatched_bits_with_tolerance(fake_addr, real_addr,error=4)
                match_error.append((i, j, score))
                # print(f"[i={i}, j={j}] fake_addr={fake_addr}, real_addr={real_addr}, score={score:.2f}")
        else:

            buffer = bits_to_hex_buffer(bits[preamle_start+i:])
            print(buffer)
            for j in range(0,6): #最多偏移6个字节
                fake_addr = [rotate_right_1bit(buffer[j+6:j+8]),
                                  rotate_right_1bit(buffer[j+4:j+6]),
                                  rotate_right_1bit(buffer[j+2:j+4]),
                                  rotate_right_1bit(buffer[j:j+2])]

                # print("fake_addr is ",fake_addr)
                k = j + 12+2* ble_rate     #k : real addr pos 
                real_addr = [buffer[k:k+2],buffer[k+2:k+4],buffer[k+4:k+6],buffer[k+6:k+8]]
                # print("real_addr is ",real_addr)
                score = count_mismatched_bits_with_tolerance(fake_addr, real_addr,error=4)
                match_error.append((i, j, score))

    # 找到得分最低的一组(i, j)
    if match_error:
        best_i, best_j, best_score = min(match_error, key=lambda x: x[2])
        # fake_addr_pos = best_j+1 if best_j % 2 != 0 else best_j
        fake_addr_pos = best_j
        real_addr_pos = fake_addr_pos + 12+2* ble_rate

        #real_addr_pos = real_addr_pos+1 if best_j % 2 != 0 else real_addr_pos 
        print(f"\nfound it ! 最佳匹配位置: fadd_p={fake_addr_pos}, radd_p={real_addr_pos}，得分={best_score:.2f}")
        pkt_buff = bits_to_hex_buffer(bits[preamle_start+best_i:])
        if best_score < 5.0:#默认可接受的范围
            real_addr_lack_p = [pkt_buff[real_addr_pos:real_addr_pos+2],pkt_buff[real_addr_pos+2:real_addr_pos+4],
                        pkt_buff[real_addr_pos+4:real_addr_pos+6],pkt_buff[real_addr_pos+6:real_addr_pos+8]]
            # print(" real_addr_lack_p is ----:",real_addr_lack_p)
            # if len(pkt_buff) % 2 != 0 :
            #     pkt_buff = "5"+pkt_buff
            # parser pkt
            print("read file:",filepath)
            # print("write pkt_buff is :",pkt_buff)
            write_ble_packet_to_pcap(pkt_buff[fake_addr_pos:], 0, "./pcap/fake_output.pcap")
            write_ble_packet_to_pcap(pkt_buff[real_addr_pos:], 1, "./pcap/real_output.pcap")

        #地址匹配差距太大，可能原因之一是样本缺失，前面一些sample没有采集到，直接匹配地址段
        else:
            if len(pkt_buff) % 2 != 0 :
                pkt_buff = pkt_buff+"0"
            byte_data = bytes.fromhex(pkt_buff)
            # Step 2: 每个字节转成8位二进制字符串，并拼接起来
            bit_string = ''.join(f'{byte:08b}' for byte in byte_data)
            # Step 3: 转为 numpy.ndarray，元素为0或1
            bit_array = np.array(list(bit_string), dtype=np.uint8)

            real_preamble_index = find_ble_preamble(bit_array[:],'2M') 
            # print(" real_addr_index is :",real_preamble_index)
            for i in range(0,4):

                buffer = bits_to_hex_buffer(bits[real_preamble_index+i:])
                # print(" buffer is :",buffer)
                for j in range(2,6):
                    real_addr_p = [buffer[j:j+2],buffer[j+2:j+4],buffer[j+4:j+6],buffer[j+6:j+8]]
                    score = count_mismatched_bits_with_tolerance(real_addr_p, real_addr_lack_p,error=4)

                    match_error_r.append((i, j, score))
                    # print(f"[i={i}, j={j}] real_addr_p={real_addr_p}, real_addr_lack_p={real_addr_lack_p}, score={score:.2f}")

            best_i, best_j, best_score = min(match_error_r, key=lambda x: x[2])
            addr_pos = best_j+1 if best_j % 2 != 0 else best_j
            print(f"\nfound it 部分缺失 ! 最佳匹配位置: radd_p={addr_pos}，得分={best_score:.2f}")
            pkt_buff = bits_to_hex_buffer(bits[real_preamble_index+best_i:])
            if len(pkt_buff) % 2 != 0 :
                pkt_buff = "5"+pkt_buff
            # print(" pkt_buff len is :",len(pkt_buff))
            # parser pkt
            print("read file:",filepath)
            # print("write pkt_buff is :",pkt_buff)
            write_ble_packet_to_pcap(pkt_buff[addr_pos:], 1, "./pcap/real_output.pcap")

    else:
        print("[!] 未找到任何匹配") 



    
                   
    

                    

  
    


    