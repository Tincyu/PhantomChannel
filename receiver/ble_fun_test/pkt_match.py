import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin,find_peaks
from numpy.lib.stride_tricks import as_strided


ADV_ACCESS_ADDRESS_RAW = "D6BE898E"
ADV_PDU_TYPES = {
    0x0: "ADV_IND",
    0x1: "ADV_DIRECT_IND",
    0x2: "ADV_NONCONN_IND",
    0x3: "SCAN_REQ",
    0x4: "SCAN_RSP",
    0x5: "CONNECT_IND",
    0x6: "ADV_SCAN_IND",
    0x7: "ADV_EXT_IND",
}


def format_hex_bytes(byte_values):
    return "".join(f"{value:02X}" for value in byte_values)


def format_ble_address(byte_values):
    if len(byte_values) < 6:
        return ""
    # BLE addresses are transmitted least-significant octet first on air.
    return "".join(f"{value:02X}" for value in reversed(byte_values[:6]))


def address_type_label(is_random):
    return "random" if is_random else "public"


def parse_adv_payload(access_address_raw, channel, header_flag, payload):
    info = {
        "ble_pdu_type": "",
        "advertiser_address": "",
        "advertiser_address_type": "",
        "peer_address": "",
        "peer_address_type": "",
        "ble_device_address": "",
    }
    if channel not in (37, 38, 39):
        return info
    if hamming_distance_hex(access_address_raw, ADV_ACCESS_ADDRESS_RAW) > 2:
        return info

    pdu_type = header_flag & 0x0F
    txaddr_random = bool((header_flag >> 6) & 0x01)
    rxaddr_random = bool((header_flag >> 7) & 0x01)
    info["ble_pdu_type"] = ADV_PDU_TYPES.get(pdu_type, f"ADV_TYPE_{pdu_type}")

    if pdu_type in (0x0, 0x1, 0x2, 0x4, 0x6) and len(payload) >= 6:
        info["advertiser_address"] = format_ble_address(payload[0:6])
        info["advertiser_address_type"] = address_type_label(txaddr_random)
        info["ble_device_address"] = info["advertiser_address"]

    if pdu_type == 0x1 and len(payload) >= 12:
        info["peer_address"] = format_ble_address(payload[6:12])
        info["peer_address_type"] = address_type_label(rxaddr_random)
    elif pdu_type == 0x3 and len(payload) >= 12:
        info["peer_address"] = format_ble_address(payload[0:6])
        info["peer_address_type"] = address_type_label(txaddr_random)
        info["advertiser_address"] = format_ble_address(payload[6:12])
        info["advertiser_address_type"] = address_type_label(rxaddr_random)
        info["ble_device_address"] = info["advertiser_address"]
    elif pdu_type == 0x5 and len(payload) >= 12:
        info["peer_address"] = format_ble_address(payload[0:6])
        info["peer_address_type"] = address_type_label(txaddr_random)
        info["advertiser_address"] = format_ble_address(payload[6:12])
        info["advertiser_address_type"] = address_type_label(rxaddr_random)
        info["ble_device_address"] = info["advertiser_address"]

    if not info["ble_device_address"] and info["advertiser_address"]:
        info["ble_device_address"] = info["advertiser_address"]
    return info


def valid_ble_data_header(header_flag, payload_len):
    """Conservative BLE data-channel header sanity check."""
    llid = header_flag & 0b11
    high_bits = (header_flag >> 5) & 0b111
    if llid == 0b00:
        return False
    if high_bits not in (0b000, 0b001):
        return False
    if payload_len < 0 or payload_len > 251:
        return False
    if llid in (0b10, 0b11) and payload_len == 0:
        return False
    return True


def valid_ble_connection_access_address(access_address_raw):
    """Apply the BLE uncoded-PHY connection Access Address constraints.

    ``access_address_raw`` is the parser's over-the-air byte order (for example,
    the advertising AA is ``D6BE898E``).  This is a blind protocol-validity
    check: it does not use a known/learned connection Access Address.
    """
    normalized = str(access_address_raw).strip().replace("0x", "").replace("0X", "")
    if len(normalized) != 8:
        return False
    try:
        raw_bytes = bytes.fromhex(normalized)
    except ValueError:
        return False

    # A connection AA must differ from the advertising AA by more than one bit.
    if hamming_distance_hex(normalized, ADV_ACCESS_ADDRESS_RAW) <= 1:
        return False
    if len(set(raw_bytes)) == 1:
        return False

    # The parser renders octets in transmission order. BLE transmits the least
    # significant octet first, so recover the numeric AA before testing MS bits.
    value = int.from_bytes(raw_bytes, byteorder="little", signed=False)
    bits_lsb_first = [(value >> bit) & 1 for bit in range(32)]
    transitions = sum(
        lhs != rhs for lhs, rhs in zip(bits_lsb_first, bits_lsb_first[1:])
    )
    if transitions > 24:
        return False

    longest_run = 1
    current_run = 1
    for lhs, rhs in zip(bits_lsb_first, bits_lsb_first[1:]):
        if lhs == rhs:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 1
    if longest_run > 6:
        return False

    most_significant_six = [(value >> bit) & 1 for bit in range(31, 25, -1)]
    if sum(
        lhs != rhs
        for lhs, rhs in zip(most_significant_six, most_significant_six[1:])
    ) < 2:
        return False
    return True


def decision(freq_dev, sps):
    center_indices = np.arange(sps//2, len(freq_dev), sps);  
    center_freqs = freq_dev[center_indices]
    # threshold = np.mean(center_freqs)
    bits = (center_freqs >0.0).astype(int)  # 假设 >0 为 1，<0 为 0
    return bits

def decision_p(freq_dev, sps):
    # 1. 查找局部极值点
    def find_local_extrema(signal):
        # 找到局部最大值和最小值
        peaks, _ = find_peaks(signal)
        troughs, _ = find_peaks(-signal)
        return np.concatenate([peaks, troughs])

    # 2. 初始化偏移量和比特列表
    offset = 0
    bits = []

    # 3. 遍历信号进行极值点查找，直到找到第一个极值点
    while offset + sps // 2 < len(freq_dev):
        # 提取信号窗口
        window = freq_dev[offset:offset + sps]
        draw_freq_dev(freq_dev)
        # 查找窗口中的局部极值点
        extrema_indices = find_local_extrema(window)
        
        if len(extrema_indices) > 0:
            # 4. 获取第一个极值点作为偏移基准
            current_offset = extrema_indices[0] + offset
            center_freq = freq_dev[current_offset]
            
            # 5. 判断比特：如果中心频率大于0，则为1，否则为0
            bit = 1 if center_freq > 0.0 else 0
            bits.append(bit)

            # 6. 找到第一个极值点后跳出循环
            break
        
        # 7. 更新偏移量，跳到下一个符号的位置
        offset += sps

    return np.array(bits)

def decision_m(freq_dev, sps):
    # 1. 计算每个符号窗口的均值
    center_indices = np.arange(sps//2, len(freq_dev), sps)
    bits = []
    
    for i in center_indices:
        # 获取每个符号窗口
        window = freq_dev[i - sps//2 : i + sps//2]
        
        # 计算该窗口的均值
        mean_value = np.mean(window)
        
        # 2. 使用均值进行比特判定：如果均值大于0，则为1，否则为0
        bit = 1 if mean_value > 0.0 else 0
        bits.append(bit)
    
    return np.array(bits)

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


# 24-bit CRC function
def crc(data, length, init):
  ret = [(init >> 16) & 0xff, (init >> 8) & 0xff, init & 0xff]

  for d in data[:length]:
    for v in range(8):
      t = (ret[0] >> 7) & 1

      ret[0] <<= 1
      if ret[1] & 0x80:
        ret[0] |= 1

      ret[1] <<= 1
      if ret[2] & 0x80:
        ret[1] |= 1

      ret[2] <<= 1

      if d & 1 != t:
        ret[2] ^= 0x5b
        ret[1] ^= 0x06

      d >>= 1

  ret[0] = swap_bits((ret[0] & 0xFF))
  ret[1] = swap_bits((ret[1] & 0xFF))
  ret[2] = swap_bits((ret[2] & 0xFF))

  return ret

def ble_crc_capture_fields(header_payload_crc, payload_len):
  """Return dewhitened PDU/CRC fields without assuming a CRCInit."""
  try:
    payload_len = int(payload_len)
  except (TypeError, ValueError):
    return {
      "dewhitened_pdu_hex": "",
      "captured_crc_hex": "",
      "crc_capture_status": "invalid_length",
    }
  if payload_len < 0 or payload_len > 255:
    return {
      "dewhitened_pdu_hex": "",
      "captured_crc_hex": "",
      "crc_capture_status": "invalid_length",
    }
  pdu_end = 2 + payload_len
  available_pdu_end = min(pdu_end, len(header_payload_crc))
  fields = {
    "dewhitened_pdu_hex": format_hex_bytes(header_payload_crc[:available_pdu_end]),
    "captured_crc_hex": "",
    "crc_capture_status": "truncated",
  }
  if len(header_payload_crc) >= pdu_end + 3:
    fields["captured_crc_hex"] = format_hex_bytes(
      header_payload_crc[pdu_end:pdu_end + 3]
    )
    fields["crc_capture_status"] = "ok"
  return fields

def signal_threshold(iq_sample, threshold, min_len):
    if len(iq_sample) == 0:
        return []  # 没有数据，直接返回空列表
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

def signal_mean_threshold(iq_sample, min_len):
    amp = np.abs(iq_sample)

    # 先粗略分段（这里用全局均值作为初步阈值）
    global_th = np.mean(amp)
    above_th = amp > global_th
    edges = np.diff(above_th.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1

    if above_th[0]:
        starts = np.r_[0, starts]
    if above_th[-1]:
        ends = np.r_[ends, len(above_th)]

    # 对每个候选段再用“该段均值”细分
    signal_segments = []
    for s, e in zip(starts, ends):
        seg = iq_sample[s:e]
        if len(seg) < min_len:
            continue

        seg_amp = np.abs(seg)
        local_th = np.mean(seg_amp)   # 本段的阈值
        mask = seg_amp > local_th
        sub_edges = np.diff(mask.astype(int))
        sub_starts = np.where(sub_edges == 1)[0] + 1
        sub_ends = np.where(sub_edges == -1)[0] + 1

        if mask[0]:
            sub_starts = np.r_[0, sub_starts]
        if mask[-1]:
            sub_ends = np.r_[sub_ends, len(mask)]

        for ss, ee in zip(sub_starts, sub_ends):
            if ee - ss >= min_len:
                signal_segments.append(seg[ss:ee])

    print(f"总共有 {len(signal_segments)} 段信号")
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

def estimate_cfo_hz_from_iq(iq_signal, sample_rate, trim_percentile=2.0, max_iter=20):
    """Estimate GFSK carrier frequency offset from a packet IQ segment."""
    if len(iq_signal) < 8:
        return None

    phase = np.unwrap(np.angle(iq_signal))
    inst_freq = np.diff(phase) * sample_rate / (2 * np.pi)
    inst_freq = inst_freq[np.isfinite(inst_freq)]
    if inst_freq.size < 8:
        return None

    low, high = np.percentile(inst_freq, [trim_percentile, 100.0 - trim_percentile])
    values = inst_freq[(inst_freq >= low) & (inst_freq <= high)]
    if values.size < 8:
        return None

    centers = np.percentile(values, [25.0, 75.0]).astype(float)
    if centers[0] == centers[1]:
        return float(centers[0])

    for _ in range(max_iter):
        distances = np.abs(values[:, None] - centers[None, :])
        labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()
        for idx in range(2):
            cluster = values[labels == idx]
            if cluster.size:
                new_centers[idx] = np.median(cluster)
        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    labels = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
    counts = [np.sum(labels == idx) for idx in range(2)]
    if min(counts) < max(4, int(values.size * 0.05)):
        return None

    centers = np.sort(centers)
    return float((centers[0] + centers[1]) / 2.0)

def normalize_access_address_set(access_addresses):
    if not access_addresses:
        return None
    normalized = set()
    for item in access_addresses:
        value = str(item).strip().replace("0x", "").replace("0X", "").upper()
        if value:
            normalized.add(value.zfill(8))
    return normalized or None


def detect_access_address(bits, ble_mode):
    for bit_offset in range(0, 4):
        buffer = bits_to_hex_buffer(bits[bit_offset:])
        pre_pos, detected_mode = find_patterns(buffer, ble_mode)
        if pre_pos == -1:
            continue
        if detected_mode == '2M':
            preamble_len = 2
        elif detected_mode == '1M':
            preamble_len = 1
        else:
            continue
        ble_pkt_buffer = hex_str_list_to_int_list(reverse_bytes(buffer[pre_pos:]))
        if len(ble_pkt_buffer) < preamble_len + 4:
            continue
        return format_hex_bytes(ble_pkt_buffer[preamble_len:preamble_len + 4])
    return None


def learn_access_address(result, learned_access_addresses=None, learned_ble_modes=None):
    if not result:
        return
    access_address = result.get("access_address")
    mode = result.get("mode")
    if not access_address or access_address == "unknow" or not mode:
        return
    normalized = str(access_address).strip().replace("0x", "").replace("0X", "").upper()
    if not normalized:
        return
    normalized = normalized.zfill(8)[-8:]
    if learned_access_addresses is not None:
        learned_access_addresses.add(normalized)
    if learned_ble_modes is not None:
        learned_ble_modes[normalized] = mode


def result_match(
    freq_dev,
    iq_len,
    chan,
    rssi,
    known_access_addresses=None,
    learned_access_addresses=None,
    learned_ble_modes=None,
):
    known_access_addresses = normalize_access_address_set(known_access_addresses)
    learned_access_address_store = learned_access_addresses
    learned_access_addresses = normalize_access_address_set(learned_access_addresses)

    bits_by_mode = {}
    results_by_mode = {}

    def bits_for_mode(mode):
        if mode not in bits_by_mode:
            samples_per_bit = 2 if mode == '2M' else 4
            bits_by_mode[mode] = decision(freq_dev, samples_per_bit)
        return bits_by_mode[mode]

    def parse_mode(mode):
        if mode not in results_by_mode:
            results_by_mode[mode] = Parser_pkt(
                bits_for_mode(mode),
                mode,
                iq_len,
                chan,
                known_access_addresses,
            )
        return results_by_mode[mode]

    if learned_access_addresses:
        mode_order = ['2M', '1M']
        if learned_ble_modes:
            mode_counts = {
                mode: list(learned_ble_modes.values()).count(mode)
                for mode in mode_order
            }
            mode_order = sorted(mode_order, key=lambda mode: mode_counts[mode], reverse=True)
        for mode in mode_order:
            result = parse_mode(mode)
            if not result:
                continue
            access_address = result.get("access_address")
            normalized = str(access_address).replace("0x", "").replace("0X", "").upper()
            if normalized in learned_access_addresses:
                best_match = result | {'mode': mode, 'rssi': rssi}
                learn_access_address(best_match, learned_access_address_store, learned_ble_modes)
                return best_match

    parse_mode('2M')
    parse_mode('1M')

    result_2M = results_by_mode.get('2M')
    result_1M = results_by_mode.get('1M')

    min_match_score_2M = None if result_2M is None else result_2M["score"]
    min_match_score_1M = None if result_1M is None else result_1M["score"]


    # 决策逻辑
    if min_match_score_1M is None and min_match_score_2M is None:
         best_match = {
            'access_address': 'unknow',
            'pkt_len': 'unknow',
            'mode': '2M',
            'rssi': '-120',
            'score':100,
            'ble_pdu_type': '',
            'advertiser_address': '',
            'advertiser_address_type': '',
            'peer_address': '',
            'peer_address_type': '',
            'ble_device_address': '',
        }
    elif min_match_score_1M is None:
        best_match = result_2M | {'mode': '2M', 'rssi': rssi}
    elif min_match_score_2M is None:
        best_match = result_1M | {'mode': '1M', 'rssi': rssi}
    else:
        # 两个都非 None，取最小 score 的一组
        if min_match_score_2M <= min_match_score_1M:
            best_match = result_2M | {'mode': '2M', 'rssi': rssi}
        else:
            best_match = result_1M | {'mode': '1M', 'rssi': rssi}

    # 输出
    if best_match:
        learn_access_address(best_match, learned_access_address_store, learned_ble_modes)
        return best_match
    else:
        return None


# 这里的采样率写死了，如果变的话，需要传入采样率。目前是4e6
def Parser_pkt(bits, ble_mode, iq_len, chan, known_access_addresses=None):
    
    match_error = []
    non_CTE_pkt = False
    CTE_pkt = False

    # print("bit is:",bits)
    ble_mod_rate = 2e6  if ble_mode == '2M' else 1e6     # 速率模式
    samples_per_bit = int(4e6 / ble_mod_rate)  # 每比特采样点数 
    tail_sample = 1*samples_per_bit*4 #尾部样本点

    for i in range(0,4):
        access_address = []
        crc_cal =[]
        preamble_len = 0

        buffer = bits_to_hex_buffer(bits[i:])
        # print("buffer_offset:",buffer)

        # 查找前导码位置
        pre_pos, ble_mode = find_patterns(buffer,ble_mode)
        # print("pre_pos is ", pre_pos)     

        if ble_mode == '2M':
            preamble_len = 2
        elif ble_mode == '1M':
            preamble_len = 1
        else:
            preamble_len = 0

        # if pre_pos == -1 or pre_pos > (len(buffer)/2)-1:
        #     continue
        # elif (pre_pos+preamble_len+9*2) >(len(buffer)-pre_pos)+2:
        #     continue
        
        if pre_pos == -1 :
            continue
        # elif (pre_pos+preamble_len+9*2) >(len(buffer)-pre_pos)+2:
        #     continue
        
        ble_pkt_buffer = hex_str_list_to_int_list((reverse_bytes(buffer[pre_pos:])))
        # print("ble_pkt_buffer is ", ble_pkt_buffer)

        if (len(ble_pkt_buffer) < (10+preamble_len)) :
            continue

        preamble = ble_pkt_buffer[0:preamble_len]
        # print("preamble is:",hex(preamble[0]))

        AAddress = ble_pkt_buffer[preamble_len:preamble_len+4]
        for i in range(0,4):
            access_address.append(hex(AAddress[i]))
        access_address_raw = format_hex_bytes(AAddress)
        if known_access_addresses is not None and access_address_raw not in known_access_addresses:
            continue
        if chan in (37, 38, 39) and hamming_distance_hex(access_address_raw, ADV_ACCESS_ADDRESS_RAW) > 2:
            continue
        if chan not in (37, 38, 39) and not valid_ble_connection_access_address(access_address_raw):
            continue
        # print("AAddress is:",access_address)

        len_exclude_payload = (1+4+2+3)      # Preamble + Access address + header + crc

        # 计算phy长度，计算误差
        pkt_payload_phy_len = (iq_len-pre_pos*samples_per_bit*4-(len_exclude_payload)*8*samples_per_bit - tail_sample)/(8*samples_per_bit)
        # print("pkt_payload_phy_len is:",pkt_payload_phy_len)

        #  header + payload + crc + covert
        header_payload_crc = dewhitening(ble_pkt_buffer[preamble_len+4:], chan)
        # header_payload_crc = ble_pkt_buffer[preamble_len+4:]
        # print(header_payload_crc)
        payload_len = header_payload_crc[1]
        header_flag = header_payload_crc[0]
        bit_flag = (header_flag >> 5) & 0b111

        # print("header_flag bit is :", bit_flag)
        # print("header_flag is :", bin(header_flag))
        # print("payload_len is :", hex(payload_len))

        if chan not in (37, 38, 39) and not valid_ble_data_header(header_flag, payload_len):
            continue

        #   目前没有返回payload，需要的话添加到返回值
        payload = header_payload_crc[2:2+payload_len]
        # print(len(payload))
        pdu_len = 2 + payload_len
        whitened_pdu = ble_pkt_buffer[preamble_len+4:preamble_len+4+pdu_len]

        # cal crc

        crc_cal = crc(header_payload_crc[ : 2 + payload_len], payload_len + 2, 0x97a38e)
            
        #   2 + payload_len + 3 = header(2B) + payload_len + CRC(3B)
        #   剩余部分即为Covert 长度
        covert_len = len(header_payload_crc) -( 2 + payload_len + 3 )
        if(covert_len > 0):
            covert_data = header_payload_crc[-covert_len:]
            del covert_data[-1] #尾波信号，非GFSK
        else:
            covert_data = []
        
        # 隐蔽接收
        # match_socre = abs(pkt_payload_phy_len-payload_len-3)

        match_socre = abs(pkt_payload_phy_len-payload_len)


        if chan != 37 and chan != 38 and chan != 39:
            if bit_flag == 0b000:
                non_CTE_pkt = True
            elif bit_flag == 0b001:
                CTE_pkt = True
            else:
                match_socre += 5

 
        # print("match_socre is:", match_socre)
        # if(match_socre>2):
        #     continue
        # if(match_socre < 5.0 ):
        # for i in range(0,len(payload)):
        #     payload_list.append(hex(payload[i]))
        # print("payload_list is :", payload_list)
            # print("covert_data len is :", len(covert_data),"covert_data is :", covert_data)
            # crc_cap = header_payload_crc[2+payload_len:2+payload_len+3]
            # print("crc_cal is :", crc_cal)
            # print("crc_cap is :", crc_cap)
            # print("header_flag is :", bin(header_flag))

        pkt_len = payload_len + len_exclude_payload #字节

        adv_info = parse_adv_payload(access_address_raw, chan, header_flag, payload)
        crc_fields = ble_crc_capture_fields(header_payload_crc, payload_len)
        match_error.append(
            {
                "access_address": access_address,
                "pkt_len": pkt_len - 10,
                "score": match_socre,
                "ble_pdu_type": adv_info["ble_pdu_type"],
                "whitened_pdu_hex": format_hex_bytes(whitened_pdu),
                **crc_fields,
                "advertiser_address": adv_info["advertiser_address"],
                "advertiser_address_type": adv_info["advertiser_address_type"],
                "peer_address": adv_info["peer_address"],
                "peer_address_type": adv_info["peer_address_type"],
                "ble_device_address": adv_info["ble_device_address"],
            }
        )

    if match_error:
        return min(match_error, key=lambda item: item["score"])
    else:
        return None


def find_patterns(s,ble_mode):
    high_priority = ['aaaa', '5555']
    low_priority = ['aa', '55']

    if ble_mode == '2M':
    # 高优先级精确匹配
        for pattern in high_priority:
            pos = s.find(pattern)
            if pos != -1:
                return pos, '2M'
        
        # 高优先级模糊匹配（汉明距离≤1）
        for pattern in high_priority:
            pos = fuzzy_find(s, pattern, tolerance=1)
            if pos != -1:
                return pos, '2M'
    elif ble_mode == '1M':
        # 低优先级精确匹配
        for pattern in low_priority:
            pos = s.find(pattern)
            if pos != -1:
                return pos, '1M'
        
        # 低优先级精确匹配（汉明距离≤1）
        for pattern in low_priority:
            pos = fuzzy_find(s, pattern, tolerance=1)
            if pos != -1:
                return pos, '1M'

    return -1, '0M'

def fuzzy_find(s, pattern, tolerance=1):
    pattern_len = len(pattern)
    for i in range(len(s) - pattern_len + 1):
        window = s[i:i+pattern_len]
        if hamming_distance_hex(window, pattern) <= tolerance:
            return i
    return -1

def hamming_distance_hex(str1, str2):

    b1 = int(str1, 16)
    b2 = int(str2, 16)
    return bin(b1 ^ b2).count('1')

def reverse_bytes(s):
    result = [byte[::-1] for byte in [s[i:i+2] for i in range(0, len(s), 2)]]
    return result

def hex_str_list_to_int_list(hex_list):
    return [int(b, 16) for b in hex_list]


# f_MHz中心频率
def ble_channel_from_freq(channel):

    channel = (channel-2402) / 2
    if(channel ==0 ):
        idx = 37
    elif(channel < 12):
        idx = channel - 1
    elif(channel ==12):
        idx = 38    
    elif(channel < 39):
        idx = channel - 2
    else:
        idx = 39
    return int(idx)

def calculate_rssi(iq_samples, calibration_offset=0.0, bandwidth=2e6):

    # 计算每个样本的瞬时功率 (|I + jQ|^2 = I^2 + Q^2)
    instantaneous_power = np.abs(iq_samples) ** 2
    
    # 计算平均功率 (线性值)
    avg_power = np.mean(instantaneous_power)
    
    # 转换为dBFS (相对于满量程的分贝值)
    power_dbfs = 10 * np.log10(avg_power)
    
    # 应用校准偏移得到RSSI (dBm)
    rssi_dbm = power_dbfs + calibration_offset
    
    return rssi_dbm, avg_power

def compute_rssi_db(iq_samples, gain_offset_db=-30):

    power_linear = np.abs(iq_samples)**2  # I² + Q²
    power_avg = np.mean(power_linear)     # 平均功率
    rssi_db = 10 * np.log10(power_avg) + gain_offset_db
    return rssi_db

def rbit(value: int) -> int:
    """Reverse bits of a 32-bit integer"""
    result = 0
    for i in range(32):
        # 取出 value 的最低位，加到 result 的最高位
        result = (result << 1) | (value & 1)
        value >>= 1
    return result

def draw_sub_signal_time(iq_sample, offset):

    plt.figure(figsize=(10,5))
    # plt.plot(np.real(subSig_decim), labe
    # l="I")
    # plt.plot(np.imag(subSig_decim), label="Q")
    plt.plot(np.abs(iq_sample))
    plt.xlabel("Sample Index")
    plt.ylabel("Am")
    plt.title(f"time wave{offset/1e6:.1f} MHz")
    plt.legend()
    plt.grid(True)
    plt.show()


def draw_iq_data(iq_sample, offset):

    plt.figure(figsize=(10,5))
    plt.plot(np.real(iq_sample), label="I")
    plt.plot(np.imag(iq_sample), label="Q")
    # plt.plot(np.abs(iq_sample))
    plt.xlabel("Sample Index")
    plt.ylabel("Am")
    plt.title(f"time wave{offset/1e6:.1f} MHz")
    plt.legend()
    plt.grid(True)
    plt.show()


def draw_sub_signal_fft(iq_sample, offset, fs_new, sub_bw):

    subN = 2**14
    subSpectrum = np.fft.fftshift(np.fft.fft(iq_sample[:subN], subN)) / subN
    subFreq = np.fft.fftshift(np.fft.fftfreq(subN, d=1/fs_new ))
    plt.figure(figsize=(8,4))
    plt.plot(subFreq/1e6, 20*np.log10(np.abs(subSpectrum)))
    plt.xlabel("freq(MHz)")
    plt.xlim(-4,4)
    plt.ylabel("AmdB)")
    plt.title(f"子带频谱 (中心偏移 {offset/1e6:.1f} MHz, 带宽 {sub_bw/1e6:.1f} MHz)")
    plt.grid(True)
    plt.show()


def draw_signal_time(iq_sample):

    plt.figure(figsize=(10,5))
    plt.plot(np.abs(iq_sample))
    plt.xlabel("Sample Index")
    plt.ylabel("Am")
    plt.title(f"total")
    plt.legend()
    plt.grid(True)
    plt.show()


def draw_freq_dev(freq_dev):

    plt.figure(figsize=(10,5))
    plt.plot(freq_dev)
    plt.xlabel("Sample Index")
    plt.ylabel("freq_dev")
    plt.title(f"total")
    plt.legend()
    plt.grid(True)
    plt.show()
