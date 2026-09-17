import sys
from pathlib import Path

import numpy as np
from scipy.signal import firwin, lfilter

from .io_utils import sample_index_to_s, sample_index_to_us


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BLE_FUN_TEST_DIR = PROJECT_ROOT / "ble_fun_test"
for path in (PROJECT_ROOT, BLE_FUN_TEST_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from BER_Func import (  # noqa: E402
    build_bluetooth_sync_word,
    calculate_rssi_dbm,
    decision as br_decision,
    extract_lap_from_bitstream,
    gfsk_demodulate as br_gfsk_demodulate,
)
from pkt_match import (  # noqa: E402
    ble_channel_from_freq,
    compute_rssi_db,
    estimate_cfo_hz_from_iq,
    gfsk_demodulate as ble_gfsk_demodulate,
    result_match,
)


BR_SYNC_POWERS = np.left_shift(np.uint64(1), np.arange(64, dtype=np.uint64))


def normalize_hex_id(value, digits):
    text = str(value).strip().replace("0x", "").replace("0X", "").upper()
    if not text:
        return ""
    try:
        return f"{int(text, 16):0{digits}X}"[-digits:]
    except ValueError:
        return text.zfill(digits)[-digits:]


def normalize_hex_id_set(values, digits):
    if not values:
        return None
    normalized = {normalize_hex_id(value, digits) for value in values}
    normalized.discard("")
    return normalized or None


def unpack_segment(segment_item, fallback_index):
    if len(segment_item) >= 4:
        start, end, segment, segment_index = segment_item[:4]
        return start, end, segment, int(segment_index)
    start, end, segment = segment_item
    return start, end, segment, fallback_index


def signal_threshold_with_index(iq_sample, threshold, min_len):
    if len(iq_sample) == 0:
        return []
    amp = np.abs(iq_sample)
    above_th = amp > threshold
    edges = np.diff(above_th.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if above_th[0]:
        starts = np.r_[0, starts]
    if above_th[-1]:
        ends = np.r_[ends, len(above_th)]
    return [(int(s), int(e), iq_sample[s:e]) for s, e in zip(starts, ends) if e - s >= min_len]


def access_address_to_str(value):
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, int):
                parts.append(f"{item:02X}")
            else:
                parts.append(str(item).replace("0x", "").replace("0X", "").zfill(2).upper())
        return "0x" + "".join(parts)
    return str(value)


def parse_ble_packets(
    iq,
    sample_rate,
    center_freq,
    freq_dev,
    threshold,
    segment_min_len,
    score_threshold,
    known_access_addresses=None,
    learned_access_addresses=None,
    learned_ble_modes=None,
):
    gain = sample_rate / (2 * np.pi * freq_dev * 8)
    channel = ble_channel_from_freq(center_freq / 1e6)
    segments = signal_threshold_with_index(iq, threshold, segment_min_len)
    return parse_ble_packet_segments(
        segments,
        sample_rate,
        center_freq,
        freq_dev,
        score_threshold,
        gain,
        channel,
        known_access_addresses=known_access_addresses,
        learned_access_addresses=learned_access_addresses,
        learned_ble_modes=learned_ble_modes,
    )


def parse_ble_packet_segments(
    segments,
    sample_rate,
    center_freq,
    freq_dev,
    score_threshold,
    gain=None,
    channel=None,
    segment_index_offset=0,
    known_access_addresses=None,
    learned_access_addresses=None,
    learned_ble_modes=None,
):
    if gain is None:
        gain = sample_rate / (2 * np.pi * freq_dev * 8)
    if channel is None:
        channel = ble_channel_from_freq(center_freq / 1e6)
    packets = []

    for seg_idx, segment_item in enumerate(segments):
        start, _end, segment, segment_index = unpack_segment(
            segment_item, segment_index_offset + seg_idx
        )
        demod = ble_gfsk_demodulate(segment, gain)
        rssi = compute_rssi_db(segment)
        result = result_match(
            demod,
            len(segment),
            channel,
            rssi,
            known_access_addresses=known_access_addresses,
            learned_access_addresses=learned_access_addresses,
            learned_ble_modes=learned_ble_modes,
        )
        if not result:
            continue
        try:
            score = float(result.get("score", 100.0))
        except (TypeError, ValueError):
            score = 100.0
        if score >= score_threshold:
            continue

        aa = access_address_to_str(result.get("access_address", ""))
        cfo_hz = estimate_cfo_hz_from_iq(segment, sample_rate)
        ble_pdu_type = result.get("ble_pdu_type", "")
        advertiser_address = result.get("advertiser_address", "")
        packet_type = "BLE_ADV" if ble_pdu_type else "BLE_CONN"
        direction_hint = "advertising" if ble_pdu_type else "connection"
        packets.append(
            {
                "packet_type": packet_type,
                "sample_index": start,
                "timestamp_us": f"{sample_index_to_us(start, sample_rate):.3f}",
                "timestamp_s": f"{sample_index_to_s(start, sample_rate):.9f}",
                "access_address": aa,
                "ble_pdu_type": ble_pdu_type,
                "whitened_pdu_hex": result.get("whitened_pdu_hex", ""),
                "dewhitened_pdu_hex": result.get("dewhitened_pdu_hex", ""),
                "captured_crc_hex": result.get("captured_crc_hex", ""),
                "post_crc_hex": result.get("post_crc_hex", ""),
                "crc_and_post_crc_hex": result.get("crc_and_post_crc_hex", ""),
                "crc_capture_status": result.get("crc_capture_status", ""),
                "advertiser_address": advertiser_address,
                "advertiser_address_type": result.get("advertiser_address_type", ""),
                "peer_address": result.get("peer_address", ""),
                "peer_address_type": result.get("peer_address_type", ""),
                "ble_device_address": result.get("ble_device_address", ""),
                "channel": channel,
                "center_freq_desc": f"{center_freq / 1e6:.3f} MHz",
                "crc_ok": "",
                "payload_len": result.get("pkt_len", ""),
                "rssi": int(rssi) if np.isfinite(rssi) else "",
                "cfo_hz": cfo_hz if cfo_hz is not None else "",
                "confidence_score": f"{100.0 - score:.3f}",
                "raw_offset_info": f"segment={segment_index};segment_start_sample={start};score={score:.3f}",
                "direction_hint": direction_hint,
            }
        )
    return packets


def majority_vote_fec(raw_header_bits):
    header = 0
    for i in range(0, 54, 3):
        bit = 1 if np.sum(raw_header_bits[i:i + 3]) >= 2 else 0
        header |= bit << (i // 3)
    return header


def check_hec(header_dewhitened, uap):
    lfsr = uap
    for i in range(10):
        data_in = (header_dewhitened >> i) & 0x1
        lfsr_out = (lfsr >> 7) & 0x1
        lfsr_in = lfsr_out ^ data_in
        lfsr_adder = (
            (lfsr_in << 7)
            | (lfsr_in << 5)
            | (lfsr_in << 2)
            | (lfsr_in << 1)
            | (lfsr_in << 0)
        )
        lfsr = (lfsr << 1) & 0xFF
        lfsr = lfsr ^ lfsr_adder
    for kk in range(8):
        bit_rx = (header_dewhitened >> (10 + kk)) & 0x1
        bit_tx = (lfsr >> (7 - kk)) & 0x1
        if bit_rx != bit_tx:
            return False
    return True


def get_packet_type_info(type_val):
    type_map = {
        0: ("NULL", 0),
        1: ("POLL", 0),
        2: ("FHS", 0),
        3: ("DM1", 1),
        4: ("DH1", 1),
        5: ("HV1_or_reserved_acl", -1),
        6: ("HV2_or_reserved_acl", -1),
        7: ("HV3_or_reserved_acl", -1),
        8: ("DV", 1),
        9: ("AUX1", 1),
        10: ("DM3", 2),
        11: ("DH3", 2),
        12: ("reserved_or_EV4", -1),
        13: ("reserved_or_EV5", -1),
        14: ("DM5", 2),
        15: ("DH5", 2),
    }
    return type_map.get(type_val, (f"Unknown_{type_val}", 0))


def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def fec23_air_bits(info_bits):
    return ceil_div(info_bits * 3, 2)


def compute_br_air_total_bytes(type_val, payload_len):
    access_code_bits = 72
    header_raw_bits = 54
    payload_len = int(payload_len)
    payload_bits = None
    if type_val in (0, 1):
        payload_bits = 0
    elif type_val == 2:
        payload_bits = 240
    elif type_val == 3 and 0 <= payload_len <= 17:
        payload_bits = fec23_air_bits((1 + payload_len + 2) * 8)
    elif type_val == 4 and 0 <= payload_len <= 27:
        payload_bits = (1 + payload_len + 2) * 8
    elif type_val in (5, 6, 7):
        payload_bits = 240
    elif type_val == 8 and 0 <= payload_len <= 9:
        payload_bits = 80 + fec23_air_bits((1 + payload_len + 2) * 8)
    elif type_val == 9 and 0 <= payload_len <= 29:
        payload_bits = (1 + payload_len) * 8
    elif type_val == 10 and 0 <= payload_len <= 121:
        payload_bits = fec23_air_bits((2 + payload_len + 2) * 8)
    elif type_val == 11 and 0 <= payload_len <= 183:
        payload_bits = (2 + payload_len + 2) * 8
    elif type_val == 14 and 0 <= payload_len <= 224:
        payload_bits = fec23_air_bits((2 + payload_len + 2) * 8)
    elif type_val == 15 and 0 <= payload_len <= 339:
        payload_bits = (2 + payload_len + 2) * 8
    if payload_bits is None:
        return None
    return ceil_div(access_code_bits + header_raw_bits + payload_bits, 8)


class LockedUAPCache:
    """Per-channel locked UAP snapshot cache.

    Persists locked UAP values across chunks so that a fresh PatientSniffer
    can start with the previously-locked UAP instead of searching all 256 UAP
    values from scratch.  Keyed by BT channel number (0-78 for BR/EDR).
    """

    def __init__(self):
        self._cache = {}  # channel -> uap

    def get(self, channel):
        return self._cache.get(channel)

    def set(self, channel, uap):
        self._cache[channel] = uap

    def clear(self, channel=None):
        if channel is None:
            self._cache.clear()
        else:
            self._cache.pop(channel, None)

    def __bool__(self):
        return True


class PatientSniffer:
    def __init__(self, uap_cache=None, channel=None):
        self.locked_uap = None
        self.miss_count = 0
        self.MAX_MISS = 10
        self._uap_cache = uap_cache
        self._channel = channel

        if self._uap_cache is not None and self._channel is not None:
            cached = self._uap_cache.get(self._channel)
            if cached is not None:
                self.locked_uap = cached

    def _update_cache(self):
        if self._uap_cache is not None and self._channel is not None:
            if self.locked_uap is not None:
                self._uap_cache.set(self._channel, self.locked_uap)
            else:
                self._uap_cache.clear(self._channel)

    def process(self, raw_bits):
        candidates = decode_br_header_candidates(
            raw_bits,
            [self.locked_uap] if self.locked_uap is not None else range(256),
            stop_after_first=True,
        )
        result = self.process_candidates(candidates, len(raw_bits))

        # Re-lock logic: if we have a locked UAP but no candidate matched,
        # increment miss counter.  After MAX_MISS consecutive misses the
        # lock is dropped and a full UAP search is attempted.
        if self.locked_uap is not None and result.get("status") == "Searching":
            self.miss_count += 1
            if self.miss_count >= self.MAX_MISS:
                self.locked_uap = None
                self.miss_count = 0
                self._update_cache()
                # Retry with full UAP search for the current segment
                candidates = decode_br_header_candidates(
                    raw_bits, range(256), stop_after_first=True,
                )
                result = self.process_candidates(candidates, len(raw_bits))

        self._update_cache()
        return result

    def process_candidates(self, candidates, raw_bits_len):
        access_code_len = 72
        header_raw_len = 54
        if raw_bits_len < access_code_len + header_raw_len:
            return {"status": "Too_Short"}

        if self.locked_uap is None:
            best_res = candidates[0] if candidates else None
        else:
            best_res = next(
                (candidate for candidate in candidates if candidate["uap"] == self.locked_uap),
                None,
            )

        if best_res is not None:
            status = "New_Lock" if self.locked_uap is None else "Locked"
            self.locked_uap = best_res["uap"]
            self.miss_count = 0
            return self.decode_payload(best_res, status)
        return {"status": "Searching"}

    def decode_payload(self, res, status):
        header = res["header"]
        type_val = (header >> 3) & 0xF
        type_name, h_mode = get_packet_type_info(type_val)
        lfsr = res["lfsr"]
        dewhitened = []
        for bit in res["p_raw"]:
            w_out = (lfsr >> 6) & 0x1
            lfsr = ((lfsr << 1) & 0x7F) ^ (w_out | (w_out << 4))
            dewhitened.append(bit ^ w_out)
        length = 0
        if h_mode == 1 and len(dewhitened) >= 8:
            byte0 = sum(dewhitened[i] << i for i in range(8))
            length = (byte0 >> 3) & 0x1F
        elif h_mode == 2 and len(dewhitened) >= 16:
            short0 = sum(dewhitened[i] << i for i in range(16))
            length = (short0 >> 3) & 0x1FF
        elif h_mode == -1:
            length = {5: 10, 6: 20, 7: 30}.get(type_val, 0)
        return {
            "status": status,
            "uap": res["uap"],
            "clk": res["clk"],
            "type": type_name,
            "type_val": type_val,
            "len": length,
            "total_bytes": compute_br_air_total_bytes(type_val, length),
        }


def decode_br_header_candidates(raw_bits, candidate_uaps=range(256), stop_after_first=False):
    access_code_len = 72
    header_raw_len = 54
    if len(raw_bits) < access_code_len + header_raw_len:
        return []

    raw_header_bits = raw_bits[access_code_len:access_code_len + header_raw_len]
    header_fec = majority_vote_fec(raw_header_bits)
    payload_bits_raw = raw_bits[access_code_len + header_raw_len:]
    candidates = []
    for uap in candidate_uaps:
        for clk in range(64):
            whitener = (clk & 0x3F) | 0x40
            header_dewhiten = header_fec
            temp_whitener = whitener
            for i in range(18):
                w_out = (temp_whitener >> 6) & 0x1
                temp_whitener = ((temp_whitener << 1) & 0x7F) ^ (w_out | (w_out << 4))
                header_dewhiten ^= w_out << i
            if check_hec(header_dewhiten, uap):
                candidates.append(
                    {
                        "uap": uap,
                        "clk": clk,
                        "header": header_dewhiten,
                        "lfsr": temp_whitener,
                        "p_raw": payload_bits_raw,
                    }
                )
                break
        if stop_after_first and candidates:
            break
    return candidates


def known_lap_sync_words(known_laps):
    if isinstance(known_laps, dict):
        return known_laps
    normalized = normalize_hex_id_set(known_laps, 6)
    if not normalized:
        return None
    return {
        build_bluetooth_sync_word(int(lap, 16)): int(lap, 16)
        for lap in normalized
    }


def learn_bredr_lap(lap_hex, learned_laps=None, learned_lap_sync_words=None):
    normalized = normalize_hex_id(lap_hex, 6)
    if not normalized:
        return
    if learned_laps is not None:
        learned_laps.add(normalized)
    if learned_lap_sync_words is not None:
        lap_value = int(normalized, 16)
        learned_lap_sync_words[build_bluetooth_sync_word(lap_value)] = lap_value


def find_known_lap_access_code(bits, known_laps):
    sync_words = known_lap_sync_words(known_laps)
    if not sync_words:
        return -1, None, False
    bits = np.asarray(bits)
    if bits.size < 72:
        return -1, None, False
    preamble_a = (
        (bits[:-71] == 1)
        & (bits[1:-70] == 0)
        & (bits[2:-69] == 1)
        & (bits[3:-68] == 0)
    )
    preamble_b = (
        (bits[:-71] == 0)
        & (bits[1:-70] == 1)
        & (bits[2:-69] == 0)
        & (bits[3:-68] == 1)
    )
    offsets = np.flatnonzero(preamble_a | preamble_b)
    if offsets.size == 0:
        return -1, None, False
    captures = np.correlate(bits[4:].astype(np.uint64), BR_SYNC_POWERS, mode="valid")
    for offset, captured in zip(offsets, captures[offsets]):
        lap_value = sync_words.get(int(captured))
        if lap_value is not None:
            return offset, lap_value, True
    return -1, None, False


def find_valid_access_code(bits, known_laps=None):
    if known_laps:
        return find_known_lap_access_code(bits, known_laps)
    bits = np.asarray(bits)
    if bits.size < 72:
        return -1, None, False
    for offset in range(bits.size - 71):
        window = bits[offset:offset + 4]
        if not (np.array_equal(window, [1, 0, 1, 0]) or np.array_equal(window, [0, 1, 0, 1])):
            continue
        lap_value, valid_result = extract_lap_from_bitstream(bits, offset, srate=1)
        if valid_result:
            return offset, lap_value, True
    return -1, None, False


def estimate_br_cfo_hz(iq_samples, bits, bit_offset, samples_per_bit, sample_rate, num_bits=72):
    bits = np.asarray(bits)
    if len(iq_samples) < 2 or bits.size == 0:
        return None
    phase = np.unwrap(np.angle(iq_samples))
    inst_freq = np.diff(phase) * sample_rate / (2 * np.pi)
    bit_start = int(bit_offset)
    bit_end = min(bit_start + num_bits, bits.size)
    bit_indices = np.arange(bit_start, bit_end)
    freq_indices = bit_indices * samples_per_bit + (samples_per_bit // 2)
    valid = freq_indices < inst_freq.size
    if not np.any(valid):
        return None
    bit_values = bits[bit_indices[valid]]
    freq_values = inst_freq[freq_indices[valid]]
    freq_zero = freq_values[bit_values == 0]
    freq_one = freq_values[bit_values == 1]
    if freq_zero.size < 3 or freq_one.size < 3:
        return None
    return float((np.median(freq_zero) + np.median(freq_one)) / 2.0)


def parse_btclassic_packets(
    iq,
    sample_rate,
    center_freq,
    freq_dev,
    threshold,
    segment_min_len,
    cutoff,
    known_laps=None,
    known_lap_fast_path=False,
    learned_laps=None,
    learned_lap_sync_words=None,
    uap_cache=None,
):
    gain = sample_rate / (2 * np.pi * freq_dev * 8)
    samples_per_bit = int(round(sample_rate / 1e6))
    lpf = firwin(100, cutoff, window=("kaiser", 8.0), fs=sample_rate)
    segments = signal_threshold_with_index(iq, threshold, segment_min_len)
    return parse_btclassic_packet_segments(
        segments,
        sample_rate,
        center_freq,
        freq_dev,
        cutoff,
        gain,
        samples_per_bit,
        lpf,
        known_laps=known_laps,
        known_lap_fast_path=known_lap_fast_path,
        learned_laps=learned_laps,
        learned_lap_sync_words=learned_lap_sync_words,
        uap_cache=uap_cache,
    )


def parse_btclassic_packet_segments(
    segments,
    sample_rate,
    center_freq,
    freq_dev,
    cutoff,
    gain=None,
    samples_per_bit=None,
    lpf=None,
    segment_index_offset=0,
    known_laps=None,
    known_lap_fast_path=False,
    learned_laps=None,
    learned_lap_sync_words=None,
    uap_cache=None,
):
    if gain is None:
        gain = sample_rate / (2 * np.pi * freq_dev * 8)
    if samples_per_bit is None:
        samples_per_bit = int(round(sample_rate / 1e6))
    if lpf is None:
        lpf = firwin(100, cutoff, window=("kaiser", 8.0), fs=sample_rate)
    normalized_known_laps = normalize_hex_id_set(known_laps, 6)
    known_lap_words = known_lap_sync_words(known_laps) if known_lap_fast_path else None
    if learned_lap_sync_words is None and learned_laps is not None:
        learned_lap_sync_words = known_lap_sync_words(learned_laps) or {}
    channel = int(round((center_freq / 1e6) - 2402))
    sniffer = PatientSniffer(uap_cache=uap_cache, channel=channel)
    packets = []

    for seg_idx, segment_item in enumerate(segments):
        start, _end, segment, segment_index = unpack_segment(
            segment_item, segment_index_offset + seg_idx
        )
        filtered = lfilter(lpf, 1.0, segment)
        demod = br_gfsk_demodulate(filtered, gain)
        bits_1m = br_decision(demod, samples_per_bit)
        offset, lap_value, valid_result = find_valid_access_code(
            bits_1m,
            learned_lap_sync_words if learned_lap_sync_words else None,
        )
        if offset == -1 or not valid_result:
            offset, lap_value, valid_result = find_valid_access_code(bits_1m, known_lap_words)
        if offset == -1 or not valid_result:
            continue
        lap_hex = f"{lap_value:06X}" if isinstance(lap_value, int) else ""
        if normalized_known_laps is not None and lap_hex not in normalized_known_laps:
            continue
        learn_bredr_lap(lap_hex, learned_laps, learned_lap_sync_words)

        sample_index = int(start + offset * samples_per_bit)
        raw_bits = bits_1m[offset:]
        res = sniffer.process(raw_bits)
        rssi = calculate_rssi_dbm(filtered, -60)
        cfo_hz = estimate_br_cfo_hz(filtered, bits_1m, offset, samples_per_bit, sample_rate)
        packets.append(
            {
                "packet_type": "BT_CLASSIC",
                "sample_index": sample_index,
                "timestamp_us": f"{sample_index_to_us(sample_index, sample_rate):.3f}",
                "timestamp_s": f"{sample_index_to_s(sample_index, sample_rate):.9f}",
                "lap": lap_hex,
                "uap": res.get("uap", ""),
                "nap": "",
                "bdaddr": "",
                "channel": "",
                "center_freq_desc": f"{center_freq / 1e6:.3f} MHz",
                "packet_header_info": (
                    f"segment={segment_index};type={res.get('type', '')};len={res.get('len', '')};"
                    f"total_bytes={res.get('total_bytes', '')};status={res.get('status', '')}"
                ),
                "hec_ok": bool(res.get("uap", "") != ""),
                "crc_ok": "",
                "rssi": int(rssi) if np.isfinite(rssi) else "",
                "cfo_hz": cfo_hz if cfo_hz is not None else "",
            }
        )
    return packets


def build_btclassic_packet_candidates(
    segments,
    sample_rate,
    center_freq,
    freq_dev,
    cutoff,
    gain=None,
    samples_per_bit=None,
    lpf=None,
    segment_index_offset=0,
    known_laps=None,
    known_lap_fast_path=False,
):
    if gain is None:
        gain = sample_rate / (2 * np.pi * freq_dev * 8)
    if samples_per_bit is None:
        samples_per_bit = int(round(sample_rate / 1e6))
    if lpf is None:
        lpf = firwin(100, cutoff, window=("kaiser", 8.0), fs=sample_rate)
    normalized_known_laps = normalize_hex_id_set(known_laps, 6)
    known_lap_words = known_lap_sync_words(known_laps) if known_lap_fast_path else None
    candidates = []
    for seg_idx, segment_item in enumerate(segments):
        start, _end, segment, segment_index = unpack_segment(
            segment_item, segment_index_offset + seg_idx
        )
        filtered = lfilter(lpf, 1.0, segment)
        demod = br_gfsk_demodulate(filtered, gain)
        bits_1m = br_decision(demod, samples_per_bit)
        offset, lap_value, valid_result = find_valid_access_code(bits_1m, known_lap_words)
        if offset == -1 or not valid_result:
            continue
        lap_hex = f"{lap_value:06X}" if isinstance(lap_value, int) else ""
        if normalized_known_laps is not None and lap_hex not in normalized_known_laps:
            continue

        sample_index = int(start + offset * samples_per_bit)
        raw_bits = bits_1m[offset:]
        rssi = calculate_rssi_dbm(filtered, -60)
        cfo_hz = estimate_br_cfo_hz(filtered, bits_1m, offset, samples_per_bit, sample_rate)
        candidates.append(
            {
                "packet_type": "BT_CLASSIC",
                "sample_index": sample_index,
                "timestamp_us": f"{sample_index_to_us(sample_index, sample_rate):.3f}",
                "timestamp_s": f"{sample_index_to_s(sample_index, sample_rate):.9f}",
                "lap": lap_hex,
                "uap": "",
                "nap": "",
                "bdaddr": "",
                "channel": "",
                "center_freq_desc": f"{center_freq / 1e6:.3f} MHz",
                "packet_header_info": "",
                "hec_ok": False,
                "crc_ok": "",
                "rssi": int(rssi) if np.isfinite(rssi) else "",
                "cfo_hz": cfo_hz if cfo_hz is not None else "",
                "_segment_index": segment_index,
                "_sniffer_raw_bits_len": len(raw_bits),
                "_sniffer_candidates": decode_br_header_candidates(raw_bits),
            }
        )
    return candidates


def finalize_btclassic_packet_candidates(candidates, sample_rate=None, center_freq=None, uap_cache=None):
    channel = int(round((center_freq / 1e6) - 2402)) if center_freq is not None else None
    sniffer = PatientSniffer(uap_cache=uap_cache, channel=channel)
    if candidates and all(bool(candidate.get("_sniffer_native_candidates", False)) for candidate in candidates):
        try:
            from .native_backend import load_native_module

            native_module, _ = load_native_module()
            if native_module is not None:
                candidate_batches = [candidate.get("_sniffer_candidates", []) for candidate in candidates]
                raw_bits_lens = np.asarray(
                    [candidate.get("_sniffer_raw_bits_len", 0) for candidate in candidates],
                    dtype=np.int64,
                )
                decisions = native_module.process_br_header_candidate_sequence(
                    candidate_batches,
                    raw_bits_lens,
                    sniffer.locked_uap,
                    sniffer.miss_count,
                    sniffer.MAX_MISS,
                    False,
                )
                packets = []
                for candidate, res in zip(candidates, decisions):
                    candidate.pop("_sniffer_candidates", None)
                    candidate.pop("_sniffer_raw_bits_len", None)
                    candidate.pop("_sniffer_native_candidates", None)
                    segment_index = candidate.pop("_segment_index")
                    candidate["uap"] = res.get("uap", "")
                    candidate["packet_header_info"] = (
                        f"segment={segment_index};type={res.get('type', '')};len={res.get('len', '')};"
                        f"total_bytes={res.get('total_bytes', '')};status={res.get('status', '')}"
                    )
                    candidate["hec_ok"] = bool(res.get("uap", "") != "")
                    packets.append(candidate)
                if decisions:
                    last = decisions[-1]
                    sniffer.locked_uap = last.get("locked_uap")
                    sniffer.miss_count = int(last.get("miss_count", 0))
                sniffer._update_cache()
                return packets
        except Exception:
            pass

    native_module = None
    native_checked = False
    packets = []
    for candidate in candidates:
        sniffer_candidates = candidate.pop("_sniffer_candidates", [])
        raw_bits_len = candidate.pop("_sniffer_raw_bits_len", 0)
        use_native_finalizer = bool(candidate.pop("_sniffer_native_candidates", False))
        res = None
        if use_native_finalizer:
            if not native_checked:
                try:
                    from .native_backend import load_native_module

                    native_module, _ = load_native_module()
                except Exception:
                    native_module = None
                native_checked = True
            if native_module is not None:
                try:
                    res = native_module.process_br_header_candidates(
                        sniffer_candidates,
                        raw_bits_len,
                        sniffer.locked_uap,
                        sniffer.miss_count,
                        sniffer.MAX_MISS,
                        False,
                    )
                    sniffer.locked_uap = res.get("locked_uap")
                    sniffer.miss_count = int(res.get("miss_count", 0))
                    sniffer._update_cache()
                except Exception:
                    res = None
        if res is None:
            res = sniffer.process_candidates(sniffer_candidates, raw_bits_len)
            if sniffer.locked_uap is not None and res.get("status") == "Searching":
                sniffer.miss_count += 1
                if sniffer.miss_count >= sniffer.MAX_MISS:
                    sniffer.locked_uap = None
                    sniffer.miss_count = 0
                    sniffer._update_cache()
            else:
                sniffer._update_cache()
        segment_index = candidate.pop("_segment_index")
        candidate["uap"] = res.get("uap", "")
        candidate["packet_header_info"] = (
            f"segment={segment_index};type={res.get('type', '')};len={res.get('len', '')};"
            f"total_bytes={res.get('total_bytes', '')};status={res.get('status', '')}"
        )
        candidate["hec_ok"] = bool(res.get("uap", "") != "")
        packets.append(candidate)
    return packets
