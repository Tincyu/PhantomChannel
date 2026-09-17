import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.signal import firwin, lfilter

from .io_utils import sample_index_to_s, sample_index_to_us


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BLE_FUN_TEST_DIR = PROJECT_ROOT / "ble_fun_test"
for path in (PROJECT_ROOT, BLE_FUN_TEST_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from BER_Func import calculate_rssi_dbm  # noqa: E402
from pkt_match import estimate_cfo_hz_from_iq  # noqa: E402


def load_native_module():
    try:
        import bt_native  # type: ignore
    except ImportError as exc:
        return None, exc
    return bt_native, None


def native_status():
    module, error = load_native_module()
    if module is None:
        return {
            "available": False,
            "version": "",
            "error": str(error),
        }
    version_func = getattr(module, "version", None)
    version = version_func() if callable(version_func) else "unknown"
    return {
        "available": True,
        "version": version,
        "error": "",
    }


def validate_native_parser_backend(ble_backend, bredr_backend):
    if ble_backend == "python" and bredr_backend == "python":
        return None

    module, error = load_native_module()
    if module is None:
        raise RuntimeError(
            "C++ parser backend was requested, but bt_native is not importable. "
            "Build the native extension and add its build directory to PYTHONPATH. "
            f"Import error: {error}"
        )

    return module


def _access_address_to_str(value):
    if isinstance(value, list):
        return "0x" + "".join(
            str(item).replace("0x", "").replace("0X", "").zfill(2).upper()
            for item in value
        )
    return str(value)


def _ble_channel_from_freq(freq_mhz):
    channel = (freq_mhz - 2402.0) / 2.0
    if channel == 0:
        return 37
    if channel < 12:
        return int(channel - 1)
    if channel == 12:
        return 38
    if channel < 39:
        return int(channel - 2)
    return 39


def _unpack_segment(segment_item, fallback_index):
    if len(segment_item) >= 4:
        start, end, segment, segment_index = segment_item[:4]
    else:
        start, end, segment = segment_item
        segment_index = fallback_index
    if len(segment_item) >= 6:
        core_start, core_end = segment_item[4:6]
    else:
        core_start, core_end = start, end
    return (
        int(start),
        int(end),
        segment,
        int(segment_index),
        int(core_start),
        int(core_end),
    )


def _is_host_segment_batch(segments):
    return all(
        hasattr(segments, name)
        for name in (
            "buffers",
            "buffer_indices",
            "starts",
            "ends",
            "offsets",
            "lengths",
            "segment_indices",
        )
    )


def _add_optional_timing(timings, key, seconds):
    if timings is not None:
        timings[key] = timings.get(key, 0.0) + seconds


def _native_input_from_legacy_segments(segments, segment_index_offset=0, timing_context=None):
    stage_start = perf_counter()
    unpacked = [
        _unpack_segment(segment_item, segment_index_offset + seg_idx)
        for seg_idx, segment_item in enumerate(segments)
    ]
    if not unpacked:
        _add_optional_timing(timing_context, "segment_input_build_s", perf_counter() - stage_start)
        return None

    offsets = np.empty(len(unpacked), dtype=np.int64)
    lengths = np.empty(len(unpacked), dtype=np.int64)
    score_lengths = np.empty(len(unpacked), dtype=np.int64)
    sample_indices = np.empty(len(unpacked), dtype=np.int64)
    segment_indices = np.empty(len(unpacked), dtype=np.int64)
    parts = []
    cursor = 0
    for idx, (start, _end, segment, segment_index, _core_start, core_end) in enumerate(unpacked):
        segment_array = np.ascontiguousarray(segment, dtype=np.complex64)
        offsets[idx] = cursor
        lengths[idx] = int(segment_array.size)
        score_lengths[idx] = max(0, min(int(segment_array.size), int(core_end) - int(start)))
        sample_indices[idx] = int(start)
        segment_indices[idx] = int(segment_index)
        parts.append(segment_array)
        cursor += int(segment_array.size)

    samples = np.concatenate(parts) if parts else np.empty(0, dtype=np.complex64)
    _add_optional_timing(timing_context, "segment_input_build_s", perf_counter() - stage_start)
    return {
        "mode": "legacy",
        "unpacked": unpacked,
        "samples": samples,
        "offsets": offsets,
        "lengths": lengths,
        "score_lengths": score_lengths,
        "sample_indices": sample_indices,
        "segment_indices": segment_indices,
    }


def _native_input_from_compact_segments(segments, timing_context=None):
    stage_start = perf_counter()
    if len(segments) == 0:
        _add_optional_timing(timing_context, "segment_input_build_s", perf_counter() - stage_start)
        return None
    buffers = [np.ascontiguousarray(buffer, dtype=np.complex64) for buffer in segments.buffers]
    _add_optional_timing(timing_context, "segment_input_build_s", perf_counter() - stage_start)
    return {
        "mode": "compact",
        "buffers": buffers,
        "buffer_indices": np.ascontiguousarray(segments.buffer_indices, dtype=np.int64),
        "offsets": np.ascontiguousarray(segments.offsets, dtype=np.int64),
        "lengths": np.ascontiguousarray(segments.lengths, dtype=np.int64),
        "score_lengths": np.ascontiguousarray(
            segments.core_ends - segments.starts, dtype=np.int64
        ),
        "sample_indices": np.ascontiguousarray(segments.starts, dtype=np.int64),
        "segment_indices": np.ascontiguousarray(segments.segment_indices, dtype=np.int64),
        "batch": segments,
    }


def _native_input_from_segments(segments, segment_index_offset=0, timing_context=None):
    if _is_host_segment_batch(segments):
        return _native_input_from_compact_segments(segments, timing_context=timing_context)
    return _native_input_from_legacy_segments(
        segments,
        segment_index_offset=segment_index_offset,
        timing_context=timing_context,
    )


def _segment_view_by_start_from_native_input(native_input, sample_index):
    if native_input["mode"] == "compact":
        starts = native_input["sample_indices"]
        matches = np.where(starts == int(sample_index))[0]
        if matches.size == 0:
            raise KeyError(sample_index)
        index = int(matches[0])
        batch = native_input["batch"]
        return np.ascontiguousarray(batch.segment_view(index), dtype=np.complex64)
    for start, _end, segment, _idx, _core_start, _core_end in native_input["unpacked"]:
        if int(start) == int(sample_index):
            return np.ascontiguousarray(segment, dtype=np.complex64)
    raise KeyError(sample_index)


def _segment_view_by_index_from_native_input(native_input, segment_index):
    index = int(segment_index)
    if native_input["mode"] == "compact":
        return np.ascontiguousarray(native_input["batch"].segment_view(index), dtype=np.complex64)
    return np.ascontiguousarray(native_input["unpacked"][index][2], dtype=np.complex64)


def parse_ble_packet_segments_native(
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
    thread_count=1,
    timing_context=None,
):
    if known_access_addresses or learned_access_addresses or learned_ble_modes:
        raise NotImplementedError(
            "bt_native BLE parser does not support known/learned AA fast paths yet."
        )

    module, error = load_native_module()
    if module is None:
        raise RuntimeError(
            "bt_native is not importable. Build native extension and add build-native "
            f"to PYTHONPATH. Import error: {error}"
        )
    if gain is None:
        gain = sample_rate / (2 * np.pi * freq_dev * 8)
    if channel is None:
        channel = _ble_channel_from_freq(center_freq / 1e6)

    native_input = _native_input_from_segments(
        segments,
        segment_index_offset=segment_index_offset,
        timing_context=timing_context,
    )
    if native_input is None:
        return []

    if native_input["mode"] == "compact":
        native_packets = module.parse_ble_segment_buffers_complex64(
            native_input["buffers"],
            native_input["buffer_indices"],
            native_input["offsets"],
            native_input["lengths"],
            native_input["score_lengths"],
            native_input["sample_indices"],
            native_input["segment_indices"],
            float(gain),
            float(score_threshold),
            int(channel),
            int(thread_count),
        )
    else:
        native_packets = module.parse_ble_segments_complex64(
            native_input["samples"],
            native_input["offsets"],
            native_input["lengths"],
            native_input["score_lengths"],
            native_input["sample_indices"],
            float(gain),
            float(score_threshold),
            int(channel),
            int(thread_count),
        )

    packets = []
    cfo_by_segment_index = {}
    for result in native_packets:
        start = int(result["sample_index"])
        result_segment_index = int(result.get("segment_index", 0))
        if result_segment_index not in cfo_by_segment_index:
            segment = _segment_view_by_index_from_native_input(
                native_input, result_segment_index
            )
            cfo_by_segment_index[result_segment_index] = estimate_cfo_hz_from_iq(
                segment, sample_rate
            )
        score = float(result.get("score", 100.0))
        aa = _access_address_to_str(result.get("access_address", ""))
        cfo_hz = cfo_by_segment_index[result_segment_index]
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
                "rssi": int(result["rssi"]) if np.isfinite(float(result["rssi"])) else "",
                "cfo_hz": cfo_hz if cfo_hz is not None else "",
                "confidence_score": f"{100.0 - score:.3f}",
                "raw_offset_info": (
                    f"segment={int(result.get('segment_index', 0))};"
                    f"segment_start_sample={start};score={score:.3f}"
                ),
                "direction_hint": direction_hint,
            }
        )
    return packets


def _estimate_br_cfo_hz(iq_samples, bits, bit_offset, samples_per_bit, sample_rate, num_bits=72):
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


def build_btclassic_packet_candidates_native(
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
    thread_count=1,
    batch_candidates=False,
    timing_context=None,
):
    if known_lap_fast_path:
        raise NotImplementedError(
            "bt_native BR/EDR hybrid backend does not support known LAP fast path yet."
        )

    module, error = load_native_module()
    if module is None:
        raise RuntimeError(
            "bt_native is not importable. Build native extension and add build-native "
            f"to PYTHONPATH. Import error: {error}"
        )
    if gain is None:
        gain = sample_rate / (2 * np.pi * freq_dev * 8)
    if samples_per_bit is None:
        samples_per_bit = int(round(sample_rate / 1e6))
    if lpf is None:
        lpf = firwin(100, cutoff, window=("kaiser", 8.0), fs=sample_rate)

    normalized_known_laps = None
    if known_laps:
        normalized_known_laps = {
            str(item).strip().replace("0x", "").replace("0X", "").upper().zfill(6)[-6:]
            for item in known_laps
        }
        normalized_known_laps.discard("")
        normalized_known_laps = normalized_known_laps or None

    if not batch_candidates:
        if _is_host_segment_batch(segments):
            segments = segments.to_segments()
        candidates = []
        for seg_idx, segment_item in enumerate(segments):
            start, _end, segment, segment_index, _core_start, _core_end = _unpack_segment(
                segment_item, segment_index_offset + seg_idx
            )
            filtered = lfilter(lpf, 1.0, segment)
            filtered_native = np.ascontiguousarray(filtered, dtype=np.complex64)
            demod = module.gfsk_demodulate_complex64(filtered_native, float(gain))
            bits_1m = np.asarray(
                module.decision_bits_float64(demod, int(samples_per_bit)),
                dtype=np.uint8,
            )
            access = module.find_br_access_code_bits(bits_1m)
            if int(access["offset"]) == -1 or not bool(access["valid"]):
                continue
            lap_value = access["lap"]
            lap_hex = f"{int(lap_value):06X}" if lap_value is not None else ""
            if normalized_known_laps is not None and lap_hex not in normalized_known_laps:
                continue

            offset = int(access["offset"])
            sample_index = int(start + offset * samples_per_bit)
            raw_bits = np.ascontiguousarray(bits_1m[offset:], dtype=np.uint8)
            native_header_candidates = module.decode_br_header_candidates_bits(
                raw_bits,
                np.arange(256, dtype=np.int32),
                False,
            )
            p_raw = raw_bits[72 + 54:].astype(int).tolist()
            sniffer_candidates = [
                {
                    "uap": int(item["uap"]),
                    "clk": int(item["clk"]),
                    "header": int(item["header"]),
                    "lfsr": int(item["lfsr"]),
                    "payload_bit_count": int(item["payload_bit_count"]),
                    "payload_prefix_bits": int(item["payload_prefix_bits"]),
                    "payload_prefix_count": int(item["payload_prefix_count"]),
                    "p_raw": p_raw,
                }
                for item in native_header_candidates
            ]
            rssi = calculate_rssi_dbm(filtered, -60)
            cfo_hz = _estimate_br_cfo_hz(
                filtered,
                bits_1m,
                offset,
                samples_per_bit,
                sample_rate,
            )
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
                    "_sniffer_candidates": sniffer_candidates,
                    "_sniffer_native_candidates": True,
                }
            )
        return candidates

    native_input = _native_input_from_segments(
        segments,
        segment_index_offset=segment_index_offset,
        timing_context=timing_context,
    )
    if native_input is None:
        return []

    if native_input["mode"] == "compact":
        native_candidates = module.build_br_packet_candidates_buffers_complex64(
            native_input["buffers"],
            native_input["buffer_indices"],
            native_input["offsets"],
            native_input["lengths"],
            native_input["sample_indices"],
            native_input["segment_indices"],
            np.ascontiguousarray(lpf, dtype=np.float64),
            float(gain),
            int(samples_per_bit),
            float(sample_rate),
            int(thread_count),
        )
    else:
        native_candidates = module.build_br_packet_candidates_complex64(
            native_input["samples"],
            native_input["offsets"],
            native_input["lengths"],
            native_input["sample_indices"],
            native_input["segment_indices"],
            np.ascontiguousarray(lpf, dtype=np.float64),
            float(gain),
            int(samples_per_bit),
            float(sample_rate),
            int(thread_count),
        )

    candidates = []
    for native_candidate in native_candidates:
        lap_hex = f"{int(native_candidate['lap']):06X}"
        if normalized_known_laps is not None and lap_hex not in normalized_known_laps:
            continue

        sample_index = int(native_candidate["sample_index"])
        p_raw = [int(bit) for bit in native_candidate["payload_bits"]]
        sniffer_candidates = [
            {
                "uap": int(item["uap"]),
                "clk": int(item["clk"]),
                "header": int(item["header"]),
                "lfsr": int(item["lfsr"]),
                "payload_bit_count": int(item["payload_bit_count"]),
                "payload_prefix_bits": int(item["payload_prefix_bits"]),
                "payload_prefix_count": int(item["payload_prefix_count"]),
                "p_raw": p_raw,
            }
            for item in native_candidate["header_candidates"]
        ]
        rssi = float(native_candidate["rssi"])
        cfo_hz = native_candidate["cfo_hz"]
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
                "_segment_index": int(native_candidate["segment_index"]),
                "_sniffer_raw_bits_len": int(native_candidate["raw_bits_len"]),
                "_sniffer_candidates": sniffer_candidates,
                "_sniffer_native_candidates": True,
            }
        )
    return candidates


def parse_btclassic_packet_segments_native_finalized(
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
    thread_count=1,
    uap_cache=None,
    timing_context=None,
):
    """Build BR/EDR candidates and run UAP/header state in C++.

    This fast path avoids materializing per-segment header candidate dictionaries
    in Python.  It intentionally falls back for known-LAP modes until the native
    wrapper supports the exact same filtering semantics.
    """
    if known_laps or known_lap_fast_path:
        raise NotImplementedError(
            "bt_native finalized BR/EDR path does not support known LAP fast path yet."
        )

    module, error = load_native_module()
    if module is None:
        raise RuntimeError(
            "bt_native is not importable. Build native extension and add build-native "
            f"to PYTHONPATH. Import error: {error}"
        )
    if gain is None:
        gain = sample_rate / (2 * np.pi * freq_dev * 8)
    if samples_per_bit is None:
        samples_per_bit = int(round(sample_rate / 1e6))
    if lpf is None:
        lpf = firwin(100, cutoff, window=("kaiser", 8.0), fs=sample_rate)

    native_input = _native_input_from_segments(
        segments,
        segment_index_offset=segment_index_offset,
        timing_context=timing_context,
    )
    if native_input is None:
        return []

    channel = int(round((center_freq / 1e6) - 2402)) if center_freq is not None else None
    locked_uap = uap_cache.get(channel) if uap_cache is not None and channel is not None else None
    if native_input["mode"] == "compact":
        native_result = module.build_and_process_br_packets_buffers_complex64(
            native_input["buffers"],
            native_input["buffer_indices"],
            native_input["offsets"],
            native_input["lengths"],
            native_input["sample_indices"],
            native_input["segment_indices"],
            np.ascontiguousarray(lpf, dtype=np.float64),
            float(gain),
            int(samples_per_bit),
            float(sample_rate),
            locked_uap,
            0,
            10,
            False,
            int(thread_count),
        )
    else:
        native_result = module.build_and_process_br_packets_complex64(
            native_input["samples"],
            native_input["offsets"],
            native_input["lengths"],
            native_input["sample_indices"],
            native_input["segment_indices"],
            np.ascontiguousarray(lpf, dtype=np.float64),
            float(gain),
            int(samples_per_bit),
            float(sample_rate),
            locked_uap,
            0,
            10,
            False,
            int(thread_count),
        )

    new_locked_uap = native_result.get("locked_uap")
    if uap_cache is not None and channel is not None:
        if new_locked_uap is None:
            uap_cache.clear(channel)
        else:
            uap_cache.set(channel, int(new_locked_uap))

    packets = []
    for item in native_result.get("packets", []):
        sample_index = int(item["sample_index"])
        lap_hex = f"{int(item['lap']):06X}"
        rssi = float(item["rssi"])
        cfo_hz = item.get("cfo_hz")
        segment_index = int(item["segment_index"])
        packets.append(
            {
                "packet_type": "BT_CLASSIC",
                "sample_index": sample_index,
                "timestamp_us": f"{sample_index_to_us(sample_index, sample_rate):.3f}",
                "timestamp_s": f"{sample_index_to_s(sample_index, sample_rate):.9f}",
                "lap": lap_hex,
                "uap": item.get("uap", ""),
                "nap": "",
                "bdaddr": "",
                "channel": "",
                "center_freq_desc": f"{center_freq / 1e6:.3f} MHz",
                "packet_header_info": (
                    f"segment={segment_index};type={item.get('type', '')};"
                    f"len={item.get('len', '')};total_bytes={item.get('total_bytes', '')};"
                    f"status={item.get('status', '')}"
                ),
                "hec_ok": bool(item.get("hec_ok", False)),
                "crc_ok": "",
                "rssi": int(rssi) if np.isfinite(rssi) else "",
                "cfo_hz": cfo_hz if cfo_hz is not None else "",
            }
        )
    return packets
