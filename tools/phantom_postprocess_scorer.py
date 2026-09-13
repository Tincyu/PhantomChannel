#!/usr/bin/env python3
"""Phantom-aware duration rescorer for BLE parser candidates.

This tool intentionally lives in PhantomChannel instead of BLE_encrypt_check.
It reads the existing parser CSV outputs plus the IQ file, then compares each
matched BLE candidate against two physical-length hypotheses:

* standard BLE: preamble + access address + header + payload + CRC
* PhantomChannel: standard BLE + RTT-reported post-CRC covert tail
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import match_rtt_sdr_results as matcher  # noqa: E402


OUTPUT_FIELDS = [
    "run_id",
    "seq",
    "channel",
    "sdr_channel",
    "covert_marker_hex",
    "tx_covert_hex",
    "tx_covert_data_hex",
    "covert_exact_match",
    "covert_data_exact_match",
    "standard_pdu_observed",
    "match_method",
    "wideband_sample_index",
    "frequency_hz",
    "normal_pdu_len",
    "air_extra_len",
    "standard_expected_us",
    "phantom_expected_us",
    "measured_duration_us",
    "measured_extra_us",
    "standard_abs_error_us",
    "phantom_abs_error_us",
    "duration_support",
    "duration_margin_us",
    "duration_confidence",
    "extracted_post_crc_hex",
    "extracted_frame_hex",
    "extracted_seq",
    "extracted_payload_hex",
    "extracted_payload_marker_ok",
    "extracted_data_hex",
    "extracted_integrity_ok",
    "extracted_exact_match",
    "extracted_data_exact_match",
    "extractor_known_bit_errors",
    "extractor_known_bit_error_rate",
    "extractor_bit_start_sample",
    "extractor_polarity",
    "extractor_threshold",
    "extractor_candidate_index",
    "extractor_candidate_wideband_sample_index",
    "extractor_candidate_timestamp_us",
    "extractor_candidate_pdu_hex",
    "predicted_wideband_sample_index",
    "predicted_extracted_post_crc_hex",
    "predicted_extracted_frame_hex",
    "predicted_extracted_seq",
    "predicted_extracted_payload_hex",
    "predicted_extracted_payload_marker_ok",
    "predicted_extracted_data_hex",
    "predicted_extracted_integrity_ok",
    "predicted_extracted_exact_match",
    "predicted_extracted_data_exact_match",
    "predicted_known_bit_errors",
    "predicted_bit_start_sample",
    "threshold",
    "noise_floor",
    "peak_power",
    "notes",
]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def float_or_none(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def int_or_none(value: Any) -> int | None:
    parsed = matcher.int_or_empty(value)
    return parsed if isinstance(parsed, int) else None


def marker_ok(payload_hex: str, marker_hex: str) -> str:
    return matcher.payload_marker_ok(payload_hex, marker_hex)


def data_hex(payload_hex: str, marker_hex: str) -> str:
    return matcher.payload_data_hex(payload_hex, marker_hex)


def ble_1m_airtime_us(payload_len_bytes: int, extra_len_bytes: int = 0) -> float:
    # 1M PHY, uncoded: one bit per microsecond.
    total_bytes = 1 + 4 + 2 + payload_len_bytes + 3 + extra_len_bytes
    return float(total_bytes * 8)


def swap_bits(value: int) -> int:
    return int((value * 0x0202020202 & 0x010884422010) % 1023)


def ble_whiten(data: bytes, channel: int) -> bytes:
    output = bytearray()
    lfsr = swap_bits(channel) | 2
    for byte in data:
        value = swap_bits(byte)
        for mask in (128, 64, 32, 16, 8, 4, 2, 1):
            if lfsr & 0x80:
                lfsr ^= 0x11
                value ^= mask
            lfsr = (lfsr << 1) & 0xFF
        output.append(swap_bits(value))
    return bytes(output)


def bytes_to_lsb_bits(data: bytes) -> np.ndarray:
    return np.asarray([(byte >> bit) & 1 for byte in data for bit in range(8)], dtype=np.uint8)


def lsb_bits_to_bytes(bits: np.ndarray) -> bytes:
    out = bytearray()
    usable = (len(bits) // 8) * 8
    for offset in range(0, usable, 8):
        value = 0
        for bit in range(8):
            value |= int(bits[offset + bit]) << bit
        out.append(value)
    return bytes(out)


def gfsk_demodulate(samples: np.ndarray) -> np.ndarray:
    if samples.size < 2:
        return np.empty(0, dtype=np.float32)
    phase = np.unwrap(np.angle(samples))
    return np.diff(phase).astype(np.float32)


def access_address_bytes(text: Any) -> bytes:
    normalized = matcher.normalize_hex(text)
    if len(normalized) != 8:
        raise ValueError(f"expected 4-byte access address, got {text!r}")
    return bytes.fromhex(normalized)


def known_air_bits(access_address: bytes, dewhitened_pdu_crc: bytes, channel: int) -> np.ndarray:
    air = bytes([0xAA]) + access_address + ble_whiten(dewhitened_pdu_crc, channel)
    return bytes_to_lsb_bits(air)


def dewhiten_tail_from_whitened_bits(
    tail_bits: np.ndarray,
    known_dewhitened_pdu_crc: bytes,
    channel: int,
) -> bytes:
    tail_whitened = lsb_bits_to_bytes(tail_bits)
    known_whitened = ble_whiten(known_dewhitened_pdu_crc, channel)
    dewhitened = ble_whiten(known_whitened + tail_whitened, channel)
    return dewhitened[len(known_dewhitened_pdu_crc):]


def extract_post_crc_from_channel_samples(
    channel_samples: np.ndarray,
    sample_rate_hz: float,
    packet_local_start: int,
    access_address: bytes,
    dewhitened_pdu_crc: bytes,
    channel: int,
    tail_len_bytes: int,
    lowpass_hz: float,
    search_samples: int,
) -> dict[str, Any]:
    if tail_len_bytes <= 0:
        return {"notes": "empty_tail_len"}

    taps = signal.firwin(129, lowpass_hz, fs=sample_rate_hz)
    filtered = signal.lfilter(taps, [1.0], channel_samples)
    demod = gfsk_demodulate(filtered)
    samples_per_bit = max(1, int(round(sample_rate_hz / 1_000_000.0)))
    known_bits = known_air_bits(access_address, dewhitened_pdu_crc, channel)
    tail_bit_count = tail_len_bytes * 8
    total_bits = len(known_bits) + tail_bit_count

    best: dict[str, Any] | None = None
    nominal_start = packet_local_start
    for start in range(nominal_start - search_samples, nominal_start + search_samples + 1):
        if start < 0:
            continue
        indices = start + np.arange(total_bits) * samples_per_bit + (samples_per_bit // 2)
        if indices[-1] >= len(demod):
            continue
        sampled = demod[indices]
        known_sampled = sampled[:len(known_bits)]
        if known_sampled.size != len(known_bits):
            continue
        for polarity in (1.0, -1.0):
            adjusted = polarity * known_sampled
            one_values = adjusted[known_bits == 1]
            zero_values = adjusted[known_bits == 0]
            if one_values.size == 0 or zero_values.size == 0:
                continue
            threshold = float((np.median(one_values) + np.median(zero_values)) / 2.0)
            decided_known = (adjusted > threshold).astype(np.uint8)
            errors = int(np.count_nonzero(decided_known != known_bits))
            candidate = {
                "errors": errors,
                "ber": errors / len(known_bits),
                "start": start,
                "polarity": polarity,
                "threshold": threshold,
            }
            if best is None or (candidate["errors"], abs(start - nominal_start)) < (
                best["errors"],
                abs(best["start"] - nominal_start),
            ):
                best = candidate

    if best is None:
        return {"notes": "no_valid_bit_alignment"}

    indices = best["start"] + np.arange(total_bits) * samples_per_bit + (samples_per_bit // 2)
    sampled = best["polarity"] * demod[indices]
    decided = (sampled > best["threshold"]).astype(np.uint8)
    tail_bits = decided[len(known_bits):]
    tail = dewhiten_tail_from_whitened_bits(tail_bits, dewhitened_pdu_crc, channel)
    frame = matcher.phantom_frame_from_hex(tail.hex())
    return {
        "extracted_post_crc_hex": tail.hex(),
        "extracted_frame_hex": frame["frame_hex"] if frame else "",
        "extracted_seq": frame["seq"] if frame else "",
        "extracted_payload_hex": frame["payload"] if frame else "",
        "extracted_integrity_ok": frame["integrity_ok"] if frame else "",
        "extractor_known_bit_errors": best["errors"],
        "extractor_known_bit_error_rate": f"{best['ber']:.9f}",
        "extractor_bit_start_sample": best["start"],
        "extractor_polarity": int(best["polarity"]),
        "extractor_threshold": best["threshold"],
        "notes": "",
    }


def _sample_demod_linear(
    demod: np.ndarray,
    bit_start: float,
    bit_positions: np.ndarray,
    samples_per_bit: float,
) -> np.ndarray | None:
    """Sample a demodulated waveform at fractional symbol positions."""

    indices = bit_start + (bit_positions * samples_per_bit) + (samples_per_bit / 2.0)
    if indices.size == 0 or indices[0] < 0 or indices[-1] >= len(demod) - 1:
        return None
    lower = np.floor(indices).astype(np.int64)
    fraction = (indices - lower).astype(np.float32)
    return (demod[lower] * (1.0 - fraction)) + (demod[lower + 1] * fraction)


def extract_post_crc_hypotheses_from_channel_samples(
    channel_samples: np.ndarray,
    sample_rate_hz: float,
    packet_local_start: int,
    access_address: bytes,
    dewhitened_pdu_crc: bytes,
    channel: int,
    tail_len_bytes: int,
    lowpass_hz_values: tuple[float, ...] = (700_000.0, 900_000.0, 1_100_000.0),
    samples_per_bit_values: tuple[float, ...] = (99.7, 100.0, 100.3),
    search_samples: int = 200,
    max_hypotheses: int = 16,
) -> dict[str, Any]:
    """Decode a tail while retaining a small list of local demod hypotheses.

    This is an opt-in investigation API.  The existing single-hypothesis
    extractor above remains the default path.  Hypotheses are selected using
    only known air bits and the Phantom frame's own format/integrity byte;
    RTT payload/sequence ground truth is deliberately not used here.
    """

    if tail_len_bytes <= 0:
        return {"notes": "empty_tail_len", "hypotheses": []}
    if not channel_samples.size:
        return {"notes": "empty_channel_samples", "hypotheses": []}

    known_bits = known_air_bits(access_address, dewhitened_pdu_crc, channel)
    tail_bit_count = tail_len_bytes * 8
    standard_prefix_bits = len(known_bits)
    total_bits = standard_prefix_bits + tail_bit_count
    nominal_samples_per_bit = sample_rate_hz / 1_000_000.0
    nominal_start = float(packet_local_start)
    start_values = range(
        max(0, packet_local_start - search_samples),
        packet_local_start + search_samples + 1,
    )
    candidates: list[dict[str, Any]] = []

    for lowpass_hz in lowpass_hz_values:
        taps = signal.firwin(129, lowpass_hz, fs=sample_rate_hz)
        filtered = signal.lfilter(taps, [1.0], channel_samples)
        demod = gfsk_demodulate(filtered)
        known_positions = np.arange(standard_prefix_bits, dtype=np.int64)
        for samples_per_bit in samples_per_bit_values:
            # Keep the configured grid near the actual sample rate.  This
            # prevents a caller from accidentally requesting an invalid tail
            # hypothesis that consumes the entire surrounding window.
            if abs(samples_per_bit - nominal_samples_per_bit) > 1.0:
                continue
            for start in start_values:
                sampled = _sample_demod_linear(
                    demod,
                    float(start),
                    known_positions,
                    samples_per_bit,
                )
                if sampled is None:
                    continue
                for polarity in (1.0, -1.0):
                    adjusted = polarity * sampled
                    one_values = adjusted[known_bits == 1]
                    zero_values = adjusted[known_bits == 0]
                    if one_values.size == 0 or zero_values.size == 0:
                        continue
                    threshold = float((np.median(one_values) + np.median(zero_values)) / 2.0)
                    decided_known = (adjusted > threshold).astype(np.uint8)
                    errors = int(np.count_nonzero(decided_known != known_bits))
                    spread = float(np.median(one_values) - np.median(zero_values))
                    candidates.append(
                        {
                            "lowpass_hz": lowpass_hz,
                            "samples_per_bit": samples_per_bit,
                            "start": start,
                            "polarity": int(polarity),
                            "threshold": threshold,
                            "known_bit_errors": errors,
                            "known_bit_error_rate": errors / max(1, len(known_bits)),
                            "known_bit_spread": spread,
                        }
                    )

    if not candidates:
        return {"notes": "no_valid_hypothesis", "hypotheses": []}

    def known_rank(item: dict[str, Any]) -> tuple[Any, ...]:
        return (
            item["known_bit_errors"],
            abs(item["start"] - nominal_start),
            -item["known_bit_spread"],
        )

    # Known-prefix scoring is cheap and is used to bound the expensive full
    # tail materialization.  The final ranking below still lets a valid frame
    # beat a marginally better prefix-only alignment.
    candidates.sort(key=known_rank)
    decode_candidates = candidates[:max(64, max_hypotheses * 8)]
    decoded: list[dict[str, Any]] = []
    full_tail_positions = standard_prefix_bits + np.arange(tail_bit_count, dtype=np.int64)
    demod_by_lowpass: dict[float, np.ndarray] = {}
    for item in decode_candidates:
        lowpass_hz = float(item["lowpass_hz"])
        demod = demod_by_lowpass.get(lowpass_hz)
        if demod is None:
            taps = signal.firwin(129, lowpass_hz, fs=sample_rate_hz)
            filtered = signal.lfilter(taps, [1.0], channel_samples)
            demod = gfsk_demodulate(filtered)
            demod_by_lowpass[lowpass_hz] = demod
        full_sampled = _sample_demod_linear(
            demod,
            float(item["start"]),
            full_tail_positions,
            float(item["samples_per_bit"]),
        )
        if full_sampled is None:
            continue
        tail_bits = (item["polarity"] * full_sampled > item["threshold"]).astype(np.uint8)
        tail = dewhiten_tail_from_whitened_bits(tail_bits, dewhitened_pdu_crc, channel)
        frame = matcher.phantom_frame_from_hex(tail.hex())
        frame_integrity_ok = frame.get("integrity_ok", "") if frame else ""
        frame_len_matches = bool(
            frame
            and frame.get("frame_offset_bytes") == 0
            and len(frame.get("frame_hex", "")) // 2 == tail_len_bytes
        )
        decoded.append(
            {
                **item,
                "extracted_post_crc_hex": tail.hex(),
                "extracted_frame_hex": frame.get("frame_hex", "") if frame else "",
                "extracted_seq": frame.get("seq", "") if frame else "",
                "extracted_payload_hex": frame.get("payload", "") if frame else "",
                "extracted_integrity_ok": frame_integrity_ok,
                "extracted_frame_len_bytes": len(frame.get("frame_hex", "")) // 2 if frame else "",
                "frame_len_matches": int(frame_len_matches),
            }
        )

    if not decoded:
        return {"notes": "no_decodable_hypothesis", "hypotheses": []}

    def rank(item: dict[str, Any]) -> tuple[Any, ...]:
        # Integrity and an exact tail length are protocol evidence.  RTT
        # sequence/payload are intentionally absent from this ordering.
        return (
            0 if item["extracted_integrity_ok"] == "1" else 1,
            0 if item["frame_len_matches"] else 1,
            item["known_bit_errors"],
            abs(item["start"] - nominal_start),
            -item["known_bit_spread"],
        )

    decoded.sort(key=rank)
    selected = decoded[:max(1, max_hypotheses)]
    winner = dict(selected[0])
    winner["notes"] = ""
    winner["hypotheses"] = selected
    winner["hypothesis_count"] = len(candidates)
    return winner


def extract_post_crc_hypotheses_from_iq(
    iq_path: Path,
    sample_rate_hz: float,
    center_frequency_hz: float,
    packet_start_sample: int,
    packet_frequency_hz: float,
    access_address_text: str,
    dewhitened_pdu_hex: str,
    captured_crc_hex: str,
    channel: int,
    tail_len_bytes: int,
    lowpass_hz_values: tuple[float, ...] = (700_000.0, 900_000.0, 1_100_000.0),
    samples_per_bit_values: tuple[float, ...] = (99.7, 100.0, 100.3),
    cfo_offset_hz_values: tuple[float, ...] = (-100_000.0, 0.0, 100_000.0),
    pre_margin_us: float = 20.0,
    expected_phantom_us: float = 5_000.0,
    post_margin_us: float = 200.0,
    search_us: float = 2.0,
    max_hypotheses: int = 16,
) -> dict[str, Any]:
    """Read one IQ neighborhood and run the local multi-hypothesis decoder."""

    try:
        access_address = access_address_bytes(access_address_text)
        dewhitened_pdu_crc = bytes.fromhex(
            matcher.normalize_hex(dewhitened_pdu_hex) + matcher.normalize_hex(captured_crc_hex)
        )
    except ValueError as exc:
        return {"notes": f"invalid_known_prefix:{exc}", "hypotheses": []}

    pre_samples = int(round(sample_rate_hz * pre_margin_us / 1_000_000.0))
    expected_samples = int(round(sample_rate_hz * expected_phantom_us / 1_000_000.0))
    post_samples = int(round(sample_rate_hz * post_margin_us / 1_000_000.0))
    start_sample = max(0, packet_start_sample - pre_samples)
    packet_local_start = packet_start_sample - start_sample
    samples = read_iq_window(
        iq_path,
        start_sample,
        packet_local_start + expected_samples + post_samples,
    )
    if samples.size == 0:
        return {"notes": "iq_window_empty", "hypotheses": []}

    cfo_values = cfo_offset_hz_values or (0.0,)
    merged: list[dict[str, Any]] = []
    for cfo_offset_hz in cfo_values:
        total_offset_hz = (packet_frequency_hz - center_frequency_hz) + cfo_offset_hz
        n = np.arange(samples.size, dtype=np.float32)
        shifted = samples * np.exp((-2j * np.pi * total_offset_hz / sample_rate_hz) * n)
        result = extract_post_crc_hypotheses_from_channel_samples(
            shifted,
            sample_rate_hz=sample_rate_hz,
            packet_local_start=packet_local_start,
            access_address=access_address,
            dewhitened_pdu_crc=dewhitened_pdu_crc,
            channel=channel,
            tail_len_bytes=tail_len_bytes,
            lowpass_hz_values=lowpass_hz_values,
            samples_per_bit_values=samples_per_bit_values,
            search_samples=int(round(sample_rate_hz * search_us / 1_000_000.0)),
            max_hypotheses=max_hypotheses,
        )
        for hypothesis in result.get("hypotheses", []):
            merged.append({"cfo_offset_hz": cfo_offset_hz, **hypothesis})

    if not merged:
        return {"notes": "no_valid_hypothesis", "hypotheses": []}

    def merged_rank(item: dict[str, Any]) -> tuple[Any, ...]:
        return (
            0 if item["extracted_integrity_ok"] == "1" else 1,
            0 if item["frame_len_matches"] else 1,
            item["known_bit_errors"],
            abs(item["start"] - packet_local_start),
            -item["known_bit_spread"],
        )

    merged.sort(key=merged_rank)
    selected = merged[:max(1, max_hypotheses)]
    winner = dict(selected[0])
    winner["bit_start_sample"] = start_sample + winner["start"]
    winner["hypotheses"] = selected
    winner["hypothesis_count"] = len(merged)
    winner["notes"] = ""
    return winner


def read_iq_window(iq_path: Path, start_sample: int, sample_count: int) -> np.ndarray:
    if start_sample < 0:
        raise ValueError("start_sample must be non-negative")
    if sample_count <= 0:
        return np.empty(0, dtype=np.complex64)

    raw = np.memmap(iq_path, dtype="<i2", mode="r")
    complex_samples = raw.size // 2
    if start_sample >= complex_samples:
        return np.empty(0, dtype=np.complex64)
    sample_count = min(sample_count, complex_samples - start_sample)
    interleaved = np.asarray(raw[start_sample * 2:(start_sample + sample_count) * 2], dtype=np.float32)
    return interleaved[0::2] + (1j * interleaved[1::2])


def smoothed_channel_power(
    samples: np.ndarray,
    sample_rate_hz: float,
    frequency_offset_hz: float,
    lowpass_hz: float,
    smooth_us: float,
) -> np.ndarray:
    if samples.size == 0:
        return np.empty(0, dtype=np.float32)

    n = np.arange(samples.size, dtype=np.float32)
    shifted = samples * np.exp((-2j * np.pi * frequency_offset_hz / sample_rate_hz) * n)
    taps = signal.firwin(129, lowpass_hz, fs=sample_rate_hz)
    baseband = signal.lfilter(taps, [1.0], shifted)
    power = np.abs(baseband) ** 2
    smooth_samples = max(1, int(round(sample_rate_hz * smooth_us / 1_000_000.0)))
    kernel = np.ones(smooth_samples, dtype=np.float32) / float(smooth_samples)
    return np.convolve(power, kernel, mode="same").astype(np.float32)


def fill_short_false_gaps(mask: np.ndarray, max_gap_samples: int) -> np.ndarray:
    if mask.size == 0 or max_gap_samples <= 0:
        return mask
    filled = mask.copy()
    false_start: int | None = None
    for index, value in enumerate(mask):
        if value:
            if false_start is not None and index - false_start <= max_gap_samples:
                filled[false_start:index] = True
            false_start = None
        elif false_start is None:
            false_start = index
    return filled


def estimate_burst_duration_us(
    iq_path: Path,
    sample_rate_hz: float,
    center_frequency_hz: float,
    packet_start_sample: int,
    packet_frequency_hz: float,
    expected_phantom_us: float,
    pre_margin_us: float,
    post_margin_us: float,
    lowpass_hz: float,
    smooth_us: float,
    threshold_sigma: float,
    min_threshold_ratio: float,
) -> dict[str, Any]:
    pre_samples = int(round(sample_rate_hz * pre_margin_us / 1_000_000.0))
    expected_samples = int(round(sample_rate_hz * expected_phantom_us / 1_000_000.0))
    post_samples = int(round(sample_rate_hz * post_margin_us / 1_000_000.0))
    start_sample = max(0, packet_start_sample - pre_samples)
    packet_local_start = packet_start_sample - start_sample
    sample_count = packet_local_start + expected_samples + post_samples

    samples = read_iq_window(iq_path, start_sample, sample_count)
    if samples.size == 0 or packet_local_start >= samples.size:
        return {"measured_duration_us": "", "notes": "iq_window_empty"}

    power = smoothed_channel_power(
        samples,
        sample_rate_hz=sample_rate_hz,
        frequency_offset_hz=packet_frequency_hz - center_frequency_hz,
        lowpass_hz=lowpass_hz,
        smooth_us=smooth_us,
    )
    noise_region = power[:max(1, min(packet_local_start, pre_samples))]
    if noise_region.size < 20:
        return {"measured_duration_us": "", "notes": "insufficient_noise_region"}

    noise_floor = float(np.median(noise_region))
    mad = float(np.median(np.abs(noise_region - noise_floor)))
    sigma = 1.4826 * mad
    threshold = noise_floor + max(threshold_sigma * sigma, noise_floor * min_threshold_ratio)

    search_end = min(power.size, packet_local_start + expected_samples + post_samples)
    search = power[packet_local_start:search_end]
    if search.size == 0:
        return {"measured_duration_us": "", "notes": "empty_search_region"}

    above = search > threshold
    gap_samples = int(round(sample_rate_hz * 2.0 / 1_000_000.0))
    above = fill_short_false_gaps(above, gap_samples)
    indices = np.flatnonzero(above)
    if indices.size == 0:
        return {
            "measured_duration_us": "",
            "threshold": threshold,
            "noise_floor": noise_floor,
            "peak_power": float(np.max(search)),
            "notes": "no_energy_above_threshold",
        }

    end_offset = int(indices[-1]) + 1
    measured_duration_us = end_offset * 1_000_000.0 / sample_rate_hz
    peak_power = float(np.max(search))
    duration_confidence = 10.0 * math.log10((peak_power + 1e-12) / (threshold + 1e-12))
    notes = "duration_hits_search_end" if end_offset >= search.size - 1 else ""
    return {
        "measured_duration_us": measured_duration_us,
        "threshold": threshold,
        "noise_floor": noise_floor,
        "peak_power": peak_power,
        "duration_confidence": duration_confidence,
        "notes": notes,
    }


def extract_post_crc_from_iq(
    iq_path: Path,
    sample_rate_hz: float,
    center_frequency_hz: float,
    packet_start_sample: int,
    packet_frequency_hz: float,
    access_address_text: str,
    dewhitened_pdu_hex: str,
    captured_crc_hex: str,
    channel: int,
    tail_len_bytes: int,
    lowpass_hz: float,
    pre_margin_us: float,
    expected_phantom_us: float,
    post_margin_us: float,
) -> dict[str, Any]:
    try:
        access_address = access_address_bytes(access_address_text)
        dewhitened_pdu_crc = bytes.fromhex(
            matcher.normalize_hex(dewhitened_pdu_hex) + matcher.normalize_hex(captured_crc_hex)
        )
    except ValueError as exc:
        return {"notes": f"invalid_known_prefix:{exc}"}

    pre_samples = int(round(sample_rate_hz * pre_margin_us / 1_000_000.0))
    expected_samples = int(round(sample_rate_hz * expected_phantom_us / 1_000_000.0))
    post_samples = int(round(sample_rate_hz * post_margin_us / 1_000_000.0))
    start_sample = max(0, packet_start_sample - pre_samples)
    packet_local_start = packet_start_sample - start_sample
    samples = read_iq_window(iq_path, start_sample, packet_local_start + expected_samples + post_samples)
    if samples.size == 0:
        return {"notes": "iq_window_empty"}

    n = np.arange(samples.size, dtype=np.float32)
    shifted = samples * np.exp((-2j * np.pi * (packet_frequency_hz - center_frequency_hz) / sample_rate_hz) * n)
    search_samples = int(round(sample_rate_hz * 2.0 / 1_000_000.0))
    return extract_post_crc_from_channel_samples(
        shifted,
        sample_rate_hz=sample_rate_hz,
        packet_local_start=packet_local_start,
        access_address=access_address,
        dewhitened_pdu_crc=dewhitened_pdu_crc,
        channel=channel,
        tail_len_bytes=tail_len_bytes,
        lowpass_hz=lowpass_hz,
        search_samples=search_samples,
    )


def scan_sdr_tail_candidates(
    sdr_rows: list[dict[str, str]],
    iq_path: Path,
    sample_rate_hz: float,
    center_frequency_hz: float,
    tx_match: dict[str, Any],
    frequency_hz: float,
    channel_i: int,
    normal_len: int,
    extra_len: int,
    lowpass_hz: float,
    pre_margin_us: float,
    expected_phantom_us: float,
    post_margin_us: float,
) -> dict[str, Any]:
    target_aa = matcher.normalize_aa(tx_match.get("parser_access_address"))
    target_channel = matcher.normalize_seq(tx_match.get("channel"))
    best: dict[str, Any] | None = None
    for index, candidate in enumerate(sdr_rows):
        try:
            candidate_aa = matcher.normalize_aa(matcher.first_present(candidate, "access_address"))
        except ValueError:
            continue
        if candidate_aa != target_aa:
            continue
        if matcher.normalize_seq(matcher.first_present(candidate, "channel")) != target_channel:
            continue
        if matcher.normalize_seq(matcher.first_present(candidate, "packet_type")) == "BLE_ADV":
            continue
        payload_len = int_or_none(candidate.get("payload_len"))
        dewhitened_pdu_hex = matcher.first_present(candidate, "dewhitened_pdu_hex")
        captured_crc_hex = matcher.first_present(candidate, "captured_crc_hex")
        if payload_len != normal_len:
            continue
        if len(matcher.normalize_hex(dewhitened_pdu_hex)) != (normal_len + 2) * 2:
            continue
        if len(matcher.normalize_hex(captured_crc_hex)) != 6:
            continue
        packet_start = int_or_none(candidate.get("wideband_sample_index"))
        if packet_start is None:
            continue
        extraction = extract_post_crc_from_iq(
            iq_path,
            sample_rate_hz=sample_rate_hz,
            center_frequency_hz=center_frequency_hz,
            packet_start_sample=packet_start,
            packet_frequency_hz=frequency_hz,
            access_address_text=matcher.first_present(candidate, "access_address"),
            dewhitened_pdu_hex=dewhitened_pdu_hex,
            captured_crc_hex=captured_crc_hex,
            channel=channel_i,
            tail_len_bytes=extra_len,
            lowpass_hz=lowpass_hz,
            pre_margin_us=pre_margin_us,
            expected_phantom_us=expected_phantom_us,
            post_margin_us=post_margin_us,
        )
        marker_hex = tx_match.get("covert_marker_hex", "")
        extracted_payload = str(extraction.get("extracted_payload_hex") or "")
        extracted_data = data_hex(extracted_payload, marker_hex) if extracted_payload else ""
        extracted_exact = bool(
            extracted_payload
            and extracted_payload == tx_match["tx_covert_hex"]
            and matcher.normalize_seq(extraction.get("extracted_seq")) == matcher.normalize_seq(tx_match["seq"])
            and extraction.get("extracted_integrity_ok") == "1"
        )
        extracted_data_exact = bool(
            extracted_data
            and extracted_data == tx_match.get("tx_covert_data_hex", "")
            and matcher.normalize_seq(extraction.get("extracted_seq")) == matcher.normalize_seq(tx_match["seq"])
            and extraction.get("extracted_integrity_ok") == "1"
        )
        known_errors = int_or_none(extraction.get("extractor_known_bit_errors"))
        rank = (
            0 if extracted_exact else 1,
            0 if marker_ok(extracted_payload, marker_hex) in ("", "1") else 1,
            0 if extracted_data_exact else 1,
            known_errors if known_errors is not None else 999999,
            abs((int_or_none(candidate.get("wideband_sample_index")) or 0) - (int_or_none(tx_match.get("sdr_wideband_sample_index")) or 0)),
        )
        enriched = {
            **extraction,
            "extractor_candidate_index": index,
            "extractor_candidate_wideband_sample_index": matcher.first_present(candidate, "wideband_sample_index"),
            "extractor_candidate_timestamp_us": matcher.first_present(candidate, "timestamp_us", "hw_timestamp_us"),
            "extractor_candidate_pdu_hex": matcher.normalize_hex(dewhitened_pdu_hex),
            "extracted_payload_marker_ok": marker_ok(extracted_payload, marker_hex),
            "extracted_data_hex": extracted_data,
            "extracted_data_exact_match": int(extracted_data_exact),
            "_rank": rank,
        }
        if best is None or rank < best["_rank"]:
            best = enriched
        if extracted_exact:
            return enriched
    if best is None:
        return {"notes": "no_same_aa_channel_len_candidate"}
    best.pop("_rank", None)
    return best


def extract_pc_frame_near_predicted_start(
    iq_path: Path,
    sample_rate_hz: float,
    center_frequency_hz: float,
    predicted_start_sample: int,
    packet_frequency_hz: float,
    access_address_text: str,
    channel: int,
    normal_len: int,
    tail_len_bytes: int,
    lowpass_hz: float,
    search_radius_us: float,
) -> dict[str, Any]:
    try:
        access_address = access_address_bytes(access_address_text)
    except ValueError as exc:
        return {"notes": f"invalid_access_address:{exc}"}

    samples_per_bit = max(1, int(round(sample_rate_hz / 1_000_000.0)))
    search_radius_samples = int(round(sample_rate_hz * search_radius_us / 1_000_000.0))
    standard_prefix_bits = (1 + 4 + 2 + normal_len + 3) * 8
    tail_bits_count = tail_len_bytes * 8
    total_bits = standard_prefix_bits + tail_bits_count
    post_samples = total_bits * samples_per_bit + int(round(sample_rate_hz * 60.0 / 1_000_000.0))
    start_sample = max(0, predicted_start_sample - search_radius_samples)
    packet_local_prediction = predicted_start_sample - start_sample
    samples = read_iq_window(iq_path, start_sample, (2 * search_radius_samples) + post_samples)
    if samples.size == 0:
        return {"notes": "prediction_iq_window_empty"}

    n = np.arange(samples.size, dtype=np.float32)
    shifted = samples * np.exp((-2j * np.pi * (packet_frequency_hz - center_frequency_hz) / sample_rate_hz) * n)
    taps = signal.firwin(129, lowpass_hz, fs=sample_rate_hz)
    filtered = signal.lfilter(taps, [1.0], shifted)
    demod = gfsk_demodulate(filtered)

    sync_bits = bytes_to_lsb_bits(bytes([0xAA]) + access_address)
    best_sync: dict[str, Any] | None = None
    coarse_step = max(1, samples_per_bit // 4)
    search_begin = max(0, packet_local_prediction - search_radius_samples)
    search_end = min(len(demod) - (total_bits * samples_per_bit) - 1, packet_local_prediction + search_radius_samples)
    if search_end <= search_begin:
        return {"notes": "prediction_search_window_empty"}

    def score_start(bit_start: int) -> dict[str, Any] | None:
        indices = bit_start + np.arange(len(sync_bits)) * samples_per_bit + (samples_per_bit // 2)
        if indices[-1] >= len(demod):
            return None
        sampled = demod[indices]
        best_local = None
        for polarity in (1.0, -1.0):
            adjusted = polarity * sampled
            one_values = adjusted[sync_bits == 1]
            zero_values = adjusted[sync_bits == 0]
            if one_values.size == 0 or zero_values.size == 0:
                continue
            threshold = float((np.median(one_values) + np.median(zero_values)) / 2.0)
            decided = (adjusted > threshold).astype(np.uint8)
            errors = int(np.count_nonzero(decided != sync_bits))
            item = {
                "errors": errors,
                "start": bit_start,
                "polarity": polarity,
                "threshold": threshold,
            }
            if best_local is None or item["errors"] < best_local["errors"]:
                best_local = item
        return best_local

    for bit_start in range(search_begin, search_end, coarse_step):
        candidate = score_start(bit_start)
        if candidate is None:
            continue
        if best_sync is None or (candidate["errors"], abs(candidate["start"] - packet_local_prediction)) < (
            best_sync["errors"],
            abs(best_sync["start"] - packet_local_prediction),
        ):
            best_sync = candidate

    if best_sync is None:
        return {"notes": "prediction_no_sync"}

    refine_begin = max(search_begin, best_sync["start"] - samples_per_bit)
    refine_end = min(search_end, best_sync["start"] + samples_per_bit + 1)
    for bit_start in range(refine_begin, refine_end):
        candidate = score_start(bit_start)
        if candidate is None:
            continue
        if (candidate["errors"], abs(candidate["start"] - packet_local_prediction)) < (
            best_sync["errors"],
            abs(best_sync["start"] - packet_local_prediction),
        ):
            best_sync = candidate

    tail_start_bit = standard_prefix_bits
    indices = (
        best_sync["start"]
        + (tail_start_bit + np.arange(tail_bits_count)) * samples_per_bit
        + (samples_per_bit // 2)
    )
    if indices[-1] >= len(demod):
        return {"notes": "prediction_tail_out_of_window"}
    sampled = best_sync["polarity"] * demod[indices]
    tail_whitened_bits = (sampled > best_sync["threshold"]).astype(np.uint8)
    dummy_prefix = bytes(2 + normal_len + 3)
    tail = dewhiten_tail_from_whitened_bits(tail_whitened_bits, dummy_prefix, channel)
    frame = matcher.phantom_frame_from_hex(tail.hex())
    return {
        "predicted_extracted_post_crc_hex": tail.hex(),
        "predicted_extracted_frame_hex": frame["frame_hex"] if frame else "",
        "predicted_extracted_seq": frame["seq"] if frame else "",
        "predicted_extracted_payload_hex": frame["payload"] if frame else "",
        "predicted_extracted_integrity_ok": frame["integrity_ok"] if frame else "",
        "predicted_known_bit_errors": best_sync["errors"],
        "predicted_bit_start_sample": start_sample + best_sync["start"],
        "notes": "",
    }


def classify_duration(
    measured_us: float | None,
    standard_expected_us: float,
    phantom_expected_us: float,
    tolerance_us: float,
) -> dict[str, Any]:
    if measured_us is None:
        return {
            "standard_abs_error_us": "",
            "phantom_abs_error_us": "",
            "duration_support": "unknown",
            "duration_margin_us": "",
        }

    standard_error = abs(measured_us - standard_expected_us)
    phantom_error = abs(measured_us - phantom_expected_us)
    margin = standard_error - phantom_error
    if phantom_error <= tolerance_us and margin > 0:
        support = "phantom"
    elif standard_error <= tolerance_us and margin < 0:
        support = "standard"
    elif phantom_error < standard_error:
        support = "phantom_weak"
    elif standard_error < phantom_error:
        support = "standard_weak"
    else:
        support = "ambiguous"
    return {
        "standard_abs_error_us": standard_error,
        "phantom_abs_error_us": phantom_error,
        "duration_support": support,
        "duration_margin_us": margin,
    }


def rescore_run(
    run_root: Path,
    output_dir: Path,
    tolerance_us: float,
    pre_margin_us: float,
    post_margin_us: float,
    lowpass_hz: float,
    smooth_us: float,
    threshold_sigma: float,
    min_threshold_ratio: float,
    prediction_search_radius_us: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ground_truth = run_root / "ground_truth" / "rtt_ground_truth.csv"
    ll_truth = run_root / "ground_truth" / "rtt_ll_tx.csv"
    ble_packets = run_root / "sdr" / "ble_packets.csv"
    metadata_path = run_root / "iq" / "metadata.json"
    iq_path = run_root / "iq" / "capture.sc16"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sample_rate_hz = float(metadata.get("actual_sample_rate_sps") or metadata["sample_rate_sps"])
    center_frequency_hz = float(metadata.get("actual_center_frequency_hz") or metadata["center_frequency_hz"])
    metadata.setdefault("analysis_bandwidth_hz", metadata.get("actual_sample_rate_sps", sample_rate_hz))

    rtt_rows = matcher.read_csv(ground_truth)
    ll_rows = matcher.read_csv(ll_truth)
    sdr_rows = matcher.read_csv(ble_packets)
    matches, _unmatched_rtt, _unmatched_sdr, match_summary = matcher.score_phantom_run(
        rtt_rows,
        ll_rows,
        sdr_rows,
        metadata,
    )
    anchor_pairs = [
        (int_or_none(row.get("seq")), int_or_none(row.get("sdr_wideband_sample_index")))
        for row in matches
        if str(row.get("covert_exact_match")) == "1"
    ]
    anchor_pairs = [
        (seq, sample)
        for seq, sample in anchor_pairs
        if seq is not None and sample is not None
    ]
    prediction_slope = 0.0
    prediction_intercept = 0.0
    if len(anchor_pairs) >= 2:
        seq_values = np.asarray([item[0] for item in anchor_pairs], dtype=np.float64)
        sample_values = np.asarray([item[1] for item in anchor_pairs], dtype=np.float64)
        prediction_slope, prediction_intercept = np.polyfit(seq_values, sample_values, 1)

    rows: list[dict[str, Any]] = []
    for match in matches:
        if not match["tx_covert_hex"] or not match["standard_pdu_observed"]:
            continue
        if str(match.get("tx_in_analysis_band", "")) == "0":
            continue

        packet_start = int_or_none(match.get("sdr_wideband_sample_index"))
        channel = match.get("channel")
        frequency_hz = matcher.ble_data_channel_frequency_hz(channel)
        normal_len = int_or_none(match.get("normal_pdu_len"))
        extra_len = int_or_none(match.get("air_extra_len"))
        notes: list[str] = []
        if packet_start is None:
            notes.append("missing_wideband_sample_index")
        if frequency_hz is None:
            notes.append("missing_frequency")
        if normal_len is None:
            notes.append("missing_normal_pdu_len")
        if extra_len is None:
            notes.append("missing_air_extra_len")

        standard_expected_us = ble_1m_airtime_us(normal_len or 0, 0)
        phantom_expected_us = ble_1m_airtime_us(normal_len or 0, extra_len or 0)
        estimate: dict[str, Any] = {"measured_duration_us": "", "notes": ";".join(notes)}
        if not notes:
            estimate = estimate_burst_duration_us(
                iq_path,
                sample_rate_hz=sample_rate_hz,
                center_frequency_hz=center_frequency_hz,
                packet_start_sample=packet_start,
                packet_frequency_hz=float(frequency_hz),
                expected_phantom_us=phantom_expected_us,
                pre_margin_us=pre_margin_us,
                post_margin_us=post_margin_us,
                lowpass_hz=lowpass_hz,
                smooth_us=smooth_us,
                threshold_sigma=threshold_sigma,
                min_threshold_ratio=min_threshold_ratio,
            )

        measured = float_or_none(estimate.get("measured_duration_us"))
        classification = classify_duration(measured, standard_expected_us, phantom_expected_us, tolerance_us)
        extraction: dict[str, Any] = {}
        duration_support = str(classification["duration_support"])
        if (
            not notes
            and str(match["covert_exact_match"]) != "1"
            and duration_support.startswith("phantom")
        ):
            channel_i = int_or_none(channel)
            if channel_i is None:
                extraction = {"notes": "missing_integer_channel"}
            else:
                extraction = scan_sdr_tail_candidates(
                    sdr_rows,
                    iq_path,
                    sample_rate_hz=sample_rate_hz,
                    center_frequency_hz=center_frequency_hz,
                    tx_match=match,
                    frequency_hz=float(frequency_hz),
                    channel_i=channel_i,
                    normal_len=normal_len or 0,
                    extra_len=extra_len or 0,
                    lowpass_hz=lowpass_hz,
                    pre_margin_us=pre_margin_us,
                    expected_phantom_us=phantom_expected_us,
                    post_margin_us=post_margin_us,
                )
        marker_hex = match.get("covert_marker_hex", "")
        extracted_payload = str(extraction.get("extracted_payload_hex") or "")
        extracted_data = data_hex(extracted_payload, marker_hex) if extracted_payload else ""
        extracted_exact = bool(
            extracted_payload
            and extracted_payload == match["tx_covert_hex"]
            and matcher.normalize_seq(extraction.get("extracted_seq")) == matcher.normalize_seq(match["seq"])
            and extraction.get("extracted_integrity_ok") == "1"
        )
        extracted_data_exact = bool(
            extracted_data
            and extracted_data == match.get("tx_covert_data_hex", "")
            and matcher.normalize_seq(extraction.get("extracted_seq")) == matcher.normalize_seq(match["seq"])
            and extraction.get("extracted_integrity_ok") == "1"
        )
        prediction: dict[str, Any] = {}
        predicted_exact = False
        seq_i = int_or_none(match.get("seq"))
        if (
            not extracted_exact
            and len(anchor_pairs) >= 2
            and seq_i is not None
            and not notes
            and str(match["covert_exact_match"]) != "1"
            and duration_support.startswith("phantom")
        ):
            channel_i = int_or_none(channel)
            if channel_i is not None:
                predicted_start = int(round((prediction_slope * seq_i) + prediction_intercept))
                prediction = extract_pc_frame_near_predicted_start(
                    iq_path,
                    sample_rate_hz=sample_rate_hz,
                    center_frequency_hz=center_frequency_hz,
                    predicted_start_sample=predicted_start,
                    packet_frequency_hz=float(frequency_hz),
                    access_address_text=match["parser_access_address"],
                    channel=channel_i,
                    normal_len=normal_len or 0,
                    tail_len_bytes=extra_len or 0,
                    lowpass_hz=lowpass_hz,
                    search_radius_us=prediction_search_radius_us,
                )
                prediction["predicted_wideband_sample_index"] = predicted_start
                predicted_payload = str(prediction.get("predicted_extracted_payload_hex") or "")
                predicted_data = data_hex(predicted_payload, marker_hex) if predicted_payload else ""
                predicted_exact = bool(
                    predicted_payload
                    and predicted_payload == match["tx_covert_hex"]
                    and matcher.normalize_seq(prediction.get("predicted_extracted_seq")) == matcher.normalize_seq(match["seq"])
                    and prediction.get("predicted_extracted_integrity_ok") == "1"
                )
                predicted_data_exact = bool(
                    predicted_data
                    and predicted_data == match.get("tx_covert_data_hex", "")
                    and matcher.normalize_seq(prediction.get("predicted_extracted_seq")) == matcher.normalize_seq(match["seq"])
                    and prediction.get("predicted_extracted_integrity_ok") == "1"
                )
                prediction["predicted_extracted_payload_marker_ok"] = marker_ok(predicted_payload, marker_hex)
                prediction["predicted_extracted_data_hex"] = predicted_data
                prediction["predicted_extracted_data_exact_match"] = int(predicted_data_exact)
        note_parts = [
            part for part in (estimate.get("notes", ""), extraction.get("notes", ""), prediction.get("notes", ""))
            if part
        ]
        row = {
            "run_id": match["run_id"],
            "seq": match["seq"],
            "channel": channel,
            "sdr_channel": match["sdr_channel"],
            "covert_marker_hex": marker_hex,
            "tx_covert_hex": match["tx_covert_hex"],
            "tx_covert_data_hex": match.get("tx_covert_data_hex", ""),
            "covert_exact_match": match["covert_exact_match"],
            "covert_data_exact_match": match.get("covert_data_exact_match", ""),
            "standard_pdu_observed": match["standard_pdu_observed"],
            "match_method": match["match_method"],
            "wideband_sample_index": match["sdr_wideband_sample_index"],
            "frequency_hz": frequency_hz or "",
            "normal_pdu_len": match["normal_pdu_len"],
            "air_extra_len": match["air_extra_len"],
            "standard_expected_us": standard_expected_us,
            "phantom_expected_us": phantom_expected_us,
            "measured_duration_us": estimate.get("measured_duration_us", ""),
            "measured_extra_us": (measured - standard_expected_us) if measured is not None else "",
            "threshold": estimate.get("threshold", ""),
            "noise_floor": estimate.get("noise_floor", ""),
            "peak_power": estimate.get("peak_power", ""),
            "duration_confidence": estimate.get("duration_confidence", ""),
            "notes": ";".join(note_parts),
            "extracted_post_crc_hex": extraction.get("extracted_post_crc_hex", ""),
            "extracted_frame_hex": extraction.get("extracted_frame_hex", ""),
            "extracted_seq": extraction.get("extracted_seq", ""),
            "extracted_payload_hex": extraction.get("extracted_payload_hex", ""),
            "extracted_payload_marker_ok": extraction.get("extracted_payload_marker_ok", marker_ok(extracted_payload, marker_hex)),
            "extracted_data_hex": extraction.get("extracted_data_hex", extracted_data),
            "extracted_integrity_ok": extraction.get("extracted_integrity_ok", ""),
            "extracted_exact_match": int(extracted_exact),
            "extracted_data_exact_match": int(extracted_data_exact),
            "extractor_known_bit_errors": extraction.get("extractor_known_bit_errors", ""),
            "extractor_known_bit_error_rate": extraction.get("extractor_known_bit_error_rate", ""),
            "extractor_bit_start_sample": extraction.get("extractor_bit_start_sample", ""),
            "extractor_polarity": extraction.get("extractor_polarity", ""),
            "extractor_threshold": extraction.get("extractor_threshold", ""),
            "extractor_candidate_index": extraction.get("extractor_candidate_index", ""),
            "extractor_candidate_wideband_sample_index": extraction.get("extractor_candidate_wideband_sample_index", ""),
            "extractor_candidate_timestamp_us": extraction.get("extractor_candidate_timestamp_us", ""),
            "extractor_candidate_pdu_hex": extraction.get("extractor_candidate_pdu_hex", ""),
            "predicted_wideband_sample_index": prediction.get("predicted_wideband_sample_index", ""),
            "predicted_extracted_post_crc_hex": prediction.get("predicted_extracted_post_crc_hex", ""),
            "predicted_extracted_frame_hex": prediction.get("predicted_extracted_frame_hex", ""),
            "predicted_extracted_seq": prediction.get("predicted_extracted_seq", ""),
            "predicted_extracted_payload_hex": prediction.get("predicted_extracted_payload_hex", ""),
            "predicted_extracted_payload_marker_ok": prediction.get("predicted_extracted_payload_marker_ok", ""),
            "predicted_extracted_data_hex": prediction.get("predicted_extracted_data_hex", ""),
            "predicted_extracted_integrity_ok": prediction.get("predicted_extracted_integrity_ok", ""),
            "predicted_extracted_exact_match": int(predicted_exact),
            "predicted_extracted_data_exact_match": prediction.get("predicted_extracted_data_exact_match", ""),
            "predicted_known_bit_errors": prediction.get("predicted_known_bit_errors", ""),
            "predicted_bit_start_sample": prediction.get("predicted_bit_start_sample", ""),
            **classification,
        }
        rows.append(row)

    exact_rows = [row for row in rows if str(row["covert_exact_match"]) == "1"]
    standard_only_rows = [row for row in rows if str(row["covert_exact_match"]) != "1"]
    phantom_supported = [row for row in rows if str(row["duration_support"]).startswith("phantom")]
    standard_only_phantom_supported = [
        row for row in standard_only_rows
        if str(row["duration_support"]).startswith("phantom")
    ]
    tail_extracted_exact = [
        row for row in standard_only_phantom_supported
        if str(row.get("extracted_exact_match", "")) == "1"
    ]
    predicted_tail_extracted_exact = [
        row for row in standard_only_phantom_supported
        if str(row.get("predicted_extracted_exact_match", "")) == "1"
    ]
    recovered_standard_only = {
        row["seq"]
        for row in tail_extracted_exact + predicted_tail_extracted_exact
    }
    summary = {
        "schema_version": 1,
        "run_id": run_root.name,
        "input_match_summary": match_summary,
        "rescored_matched_in_band_payload_packets": len(rows),
        "exact_packets_rescored": len(exact_rows),
        "standard_only_packets_rescored": len(standard_only_rows),
        "phantom_duration_supported_packets": len(phantom_supported),
        "standard_only_phantom_duration_supported_packets": len(standard_only_phantom_supported),
        "standard_only_tail_extracted_exact_packets": len(tail_extracted_exact),
        "standard_only_predicted_tail_extracted_exact_packets": len(predicted_tail_extracted_exact),
        "standard_only_any_tail_extracted_exact_packets": len(recovered_standard_only),
        "combined_exact_packets_after_tail_extraction": len(exact_rows) + len(recovered_standard_only),
        "combined_exact_packet_recovery_rate_after_tail_extraction": (
            (len(exact_rows) + len(recovered_standard_only)) / len(rows) if rows else 0.0
        ),
        "prediction_anchor_packets": len(anchor_pairs),
        "prediction_sample_slope_per_seq": prediction_slope,
        "prediction_sample_intercept": prediction_intercept,
        "prediction_search_radius_us": prediction_search_radius_us,
        "phantom_duration_support_rate": len(phantom_supported) / len(rows) if rows else 0.0,
        "standard_only_phantom_duration_support_rate": (
            len(standard_only_phantom_supported) / len(standard_only_rows) if standard_only_rows else 0.0
        ),
        "tolerance_us": tolerance_us,
        "pre_margin_us": pre_margin_us,
        "post_margin_us": post_margin_us,
        "lowpass_hz": lowpass_hz,
        "smooth_us": smooth_us,
        "threshold_sigma": threshold_sigma,
        "min_threshold_ratio": min_threshold_ratio,
        "notes": [
            "This is a PhantomChannel-local post-processing scorer; it does not modify BLE_encrypt_check.",
            "It tests whether matched BLE candidates physically look closer to standard BLE duration or standard+post-CRC Phantom duration.",
            "Parser-exported post_crc_hex remains the primary recovery path.",
            "For standard-only Phantom-supported candidates, the tool also attempts a local IQ tail extraction of the PC frame.",
        ],
    }
    return rows, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tolerance-us", type=float, default=18.0)
    parser.add_argument("--pre-margin-us", type=float, default=40.0)
    parser.add_argument("--post-margin-us", type=float, default=80.0)
    parser.add_argument("--lowpass-hz", type=float, default=900_000.0)
    parser.add_argument("--smooth-us", type=float, default=1.5)
    parser.add_argument("--threshold-sigma", type=float, default=8.0)
    parser.add_argument("--min-threshold-ratio", type=float, default=2.0)
    parser.add_argument("--prediction-search-radius-us", type=float, default=30_000.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_root = args.run_root.resolve()
    output_dir = args.output_dir or (run_root / "results" / "phantom_duration_rescore")
    rows, summary = rescore_run(
        run_root=run_root,
        output_dir=output_dir,
        tolerance_us=args.tolerance_us,
        pre_margin_us=args.pre_margin_us,
        post_margin_us=args.post_margin_us,
        lowpass_hz=args.lowpass_hz,
        smooth_us=args.smooth_us,
        threshold_sigma=args.threshold_sigma,
        min_threshold_ratio=args.min_threshold_ratio,
        prediction_search_radius_us=args.prediction_search_radius_us,
    )
    write_csv(output_dir / "duration_rescore.csv", rows, OUTPUT_FIELDS)
    write_json(output_dir / "duration_rescore_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
