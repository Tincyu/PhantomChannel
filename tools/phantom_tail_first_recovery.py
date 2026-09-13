#!/usr/bin/env python3
"""RTT-guided, parser-independent Phantom post-CRC recovery.

This investigation tool deliberately starts from IQ windows and a physical
long-burst inventory.  It does not require a parser row's decoded PDU, CRC, or
header length for the configured-length path.  RTT sequence/payload fields are
loaded only for the separate audit output; they never select a demodulation
hypothesis.

The current implementation is intentionally opt-in and does not replace the
existing scorer or modify BLE_encrypt_check.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import ndimage, signal

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import match_rtt_sdr_results as legacy  # noqa: E402
import match_rtt_sdr_v2 as matching  # noqa: E402
import phantom_postprocess_scorer as postprocess  # noqa: E402
from build_recovery_ledger import ble_guard_snapshot, sha256_file  # noqa: E402


TARGET_AA = "e888866a"
TAIL_BYTES = 237
EXPECTED_PAYLOAD_LENGTH = 231


_CUDA_DSP = None


def load_cuda_dsp() -> dict[str, Any]:
    """Load the CUDA DSP helpers without changing BLE_encrypt_check."""
    global _CUDA_DSP
    if _CUDA_DSP is not None:
        return _CUDA_DSP
    ble_root = Path("/path/to/BLE_encrypt_check")
    experiment_root = ble_root / "experiment"
    if str(experiment_root) not in sys.path:
        sys.path.insert(0, str(experiment_root))
    from bt_pipeline import cuda_backend  # type: ignore
    from bt_pipeline.pfb_channelizer import (  # type: ignore
        apply_cleanup_lpf,
        design_cleanup_lpf,
    )

    cp = cuda_backend.require_cuda()
    _CUDA_DSP = {
        "cp": cp,
        "cuda_backend": cuda_backend,
        "apply_cleanup_lpf": apply_cleanup_lpf,
        "design_cleanup_lpf": design_cleanup_lpf,
    }
    return _CUDA_DSP


BURST_FIELDS = [
    "burst_id",
    "channel",
    "frequency_hz",
    "start_sample",
    "end_sample",
    "duration_us",
    "predicted_sample_nearest",
    "nearest_tx_attempt_id",
    "tx_window_hits",
    "energy_threshold",
    "noise_floor",
    "peak_power",
    "snr_db",
    "long_burst",
    "window_clipped",
    "detector_notes",
]


BLIND_FIELDS = [
    "run_id",
    "variant",
    "aa_tolerance",
    "burst_id",
    "channel",
    "frequency_hz",
    "burst_start_sample",
    "burst_end_sample",
    "burst_duration_us",
    "sync_start_sample",
    "aa_hamming_distance",
    "aa_soft_score",
    "aa_soft_margin",
    "candidate_payload_length",
    "tail_offset_bits",
    "tail_end_sample",
    "frame_hex",
    "frame_seq",
    "frame_payload_hex",
    "frame_payload_len",
    "frame_integrity_ok",
    "frame_len_matches",
    "pc_structure_score",
    "soft_likelihood",
    "tail_threshold_mode",
    "samples_per_bit",
    "timing_drift_ppm",
    "polarity",
    "cfo_offset_hz",
    "lowpass_hz",
    "hypothesis_rank",
    "consensus_support",
    "notes",
]


AUDIT_FIELDS = [
    "run_id",
    "variant",
    "aa_tolerance",
    "attempt_index",
    "tx_attempt_id",
    "unique_notification_id",
    "seq",
    "channel",
    "predicted_sample",
    "in_iq_window",
    "target_long_burst_observed",
    "burst_id",
    "burst_start_sample",
    "burst_duration_us",
    "tail_candidate",
    "valid_phantom_frame",
    "blind_frame_seq",
    "blind_payload_len",
    "blind_integrity_ok",
    "payload_exact",
    "data_exact",
    "physical_start_residual_samples",
    "aa_hamming_distance",
    "aa_soft_score",
    "consensus_support",
    "notes",
]


RETRANSMISSION_FIELDS = [
    "variant",
    "aa_tolerance",
    "seq",
    "attempt_count",
    "assigned_count",
    "exact_count",
    "merged_best_exact",
    "burst_ids",
    "attempt_indices",
]


FUNNEL_FIELDS = [
    "variant",
    "aa_tolerance",
    "tx_windows_evaluated",
    "target_long_burst_observed",
    "tail_candidates",
    "valid_phantom_frames",
    "payload_exact",
    "data_exact",
    "false_positive_frame_count",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def int_or_none(value: Any) -> int | None:
    parsed = legacy.int_or_empty(value)
    return parsed if isinstance(parsed, int) else None


def float_or_none(value: Any) -> float | None:
    parsed = legacy.float_or_none(value)
    return parsed if parsed is not None else None


def normalize_hex(value: Any) -> str:
    try:
        return legacy.normalize_hex(value)
    except (TypeError, ValueError):
        return ""


def parse_grid(text: str, cast=float) -> tuple[Any, ...]:
    values: list[Any] = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(cast(item))
    if not values:
        raise ValueError(f"empty grid: {text!r}")
    return tuple(values)


@lru_cache(maxsize=4096)
def whitening_stream(length_bytes: int, channel: int) -> bytes:
    return postprocess.ble_whiten(bytes(max(0, length_bytes)), channel)


def lsb_bits_to_bytes_fast(bits: np.ndarray) -> bytes:
    usable = (len(bits) // 8) * 8
    if usable <= 0:
        return b""
    packed = np.packbits(
        np.asarray(bits[:usable], dtype=np.uint8).reshape(-1, 8),
        axis=1,
        bitorder="little",
    )
    return packed[:, 0].tobytes()


def dewhiten_direct_tail(raw_tail_bits: np.ndarray, channel: int, prefix_bytes: int) -> bytes:
    raw_tail = lsb_bits_to_bytes_fast(raw_tail_bits)
    stream = whitening_stream(prefix_bytes + len(raw_tail), channel)[prefix_bytes:]
    return bytes(left ^ right for left, right in zip(raw_tail, stream))


@lru_cache(maxsize=4096)
def expected_structure_bits(channel: int, prefix_bytes: int, tail_len_bytes: int) -> np.ndarray:
    reference = bytearray(b"PC\x00\x00")
    reference.append(max(0, tail_len_bytes - 6))
    reference.extend(bytes(max(0, tail_len_bytes - len(reference))))
    expected_raw = postprocess.ble_whiten(bytes(prefix_bytes) + bytes(reference), channel)[prefix_bytes:]
    return np.unpackbits(
        np.frombuffer(expected_raw, dtype=np.uint8),
        bitorder="little",
    ).astype(np.uint8, copy=False)


def sample_linear(demod: np.ndarray, start: float, positions: np.ndarray, samples_per_bit: float) -> np.ndarray | None:
    indices = start + positions.astype(np.float64) * samples_per_bit + samples_per_bit / 2.0
    if indices.size == 0 or indices[0] < 0 or indices[-1] >= len(demod) - 1:
        return None
    lower = np.floor(indices).astype(np.int64)
    fraction = (indices - lower).astype(np.float32)
    return demod[lower] * (1.0 - fraction) + demod[lower + 1] * fraction


def signed_threshold(values: np.ndarray, expected_bits: np.ndarray) -> tuple[float, np.ndarray, float] | None:
    one = values[expected_bits == 1]
    zero = values[expected_bits == 0]
    if one.size == 0 or zero.size == 0:
        return None
    threshold = float((np.median(one) + np.median(zero)) / 2.0)
    decided = (values > threshold).astype(np.uint8)
    spread = float(np.median(one) - np.median(zero))
    return threshold, decided, spread


def cuda_coarse_sync_candidates(
    demod: np.ndarray,
    *,
    begin: int,
    end: int,
    samples_per_bit: float,
    expected_bits: np.ndarray,
    coarse_step: int,
    max_candidates: int,
    cuda_device: int = 0,
) -> list[dict[str, Any]]:
    """Find coarse AA starts on CUDA and return only small host candidates."""
    dsp = load_cuda_dsp()
    cp = dsp["cp"]
    cuda_backend = dsp["cuda_backend"]
    coarse_rows: list[dict[str, Any]] = []
    chunk_size = 32_768
    with cuda_backend.use_device(cuda_device):
        demod_gpu = cp.asarray(demod, dtype=cp.float32)
        expected_gpu = cp.asarray(expected_bits, dtype=cp.uint8)
        positions_gpu = cp.arange(len(expected_bits), dtype=cp.float64)
        starts_host = np.arange(begin, end + 1, coarse_step, dtype=np.int64)
        for offset in range(0, len(starts_host), chunk_size):
            starts_host_chunk = starts_host[offset : offset + chunk_size]
            starts_gpu = cp.asarray(starts_host_chunk, dtype=cp.float64)
            indices = (
                starts_gpu[:, None]
                + positions_gpu[None, :] * float(samples_per_bit)
                + float(samples_per_bit) / 2.0
            )
            lower = cp.floor(indices).astype(cp.int64)
            valid = lower[:, -1] < demod_gpu.size - 1
            if not bool(cp.any(valid).item()):
                continue
            valid_starts = starts_gpu[valid]
            lower = lower[valid]
            indices = indices[valid]
            fraction = (indices - lower).astype(cp.float32)
            sampled = demod_gpu[lower] * (1.0 - fraction) + demod_gpu[lower + 1] * fraction
            expected_one = expected_gpu == 1
            expected_zero = ~expected_one
            expected_sign = cp.where(expected_one, 1.0, -1.0)
            for polarity in (1.0, -1.0):
                adjusted = polarity * sampled
                one = adjusted[:, expected_one]
                zero = adjusted[:, expected_zero]
                threshold = (cp.median(one, axis=1) + cp.median(zero, axis=1)) / 2.0
                decided = adjusted > threshold[:, None]
                hamming = cp.count_nonzero(decided != expected_gpu[None, :], axis=1)
                centered = adjusted - threshold[:, None]
                soft_score = cp.mean(expected_sign[None, :] * centered, axis=1) / (
                    cp.std(centered, axis=1) + 1e-9
                )
                soft_margin = cp.mean(cp.abs(centered), axis=1) / (
                    cp.std(centered, axis=1) + 1e-9
                )
                keep_count = min(16, int(hamming.size))
                if keep_count <= 0:
                    continue
                keep = cp.argpartition(hamming, keep_count - 1)[:keep_count]
                for index in cp.asnumpy(keep):
                    i = int(index)
                    coarse_rows.append(
                        {
                            "start": int(round(float(valid_starts[i].item()))),
                            "samples_per_bit": float(samples_per_bit),
                            "polarity": int(polarity),
                            "threshold": float(threshold[i].item()),
                            "aa_hamming_distance": int(hamming[i].item()),
                            "aa_soft_score": float(soft_score[i].item()),
                            "aa_soft_margin": float(soft_margin[i].item()),
                            "known_spread": float((cp.median(one[i]) - cp.median(zero[i])).item()),
                        }
                    )
    return coarse_rows


def sync_hypotheses(
    demod: np.ndarray,
    nominal_start: int,
    access_address: bytes,
    *,
    samples_per_bit_values: tuple[float, ...],
    search_samples: int,
    max_hamming: int,
    max_candidates: int = 32,
    use_cuda: bool = False,
    cuda_device: int = 0,
) -> list[dict[str, Any]]:
    expected_bits = postprocess.bytes_to_lsb_bits(bytes([0xAA]) + access_address)
    positions = np.arange(len(expected_bits), dtype=np.int64)
    candidates: list[dict[str, Any]] = []

    def score_at(demod_values: np.ndarray, start: int, samples_per_bit: float) -> list[dict[str, Any]]:
        sampled = sample_linear(demod_values, float(start), positions, float(samples_per_bit))
        if sampled is None:
            return []
        local: list[dict[str, Any]] = []
        for polarity in (1.0, -1.0):
            adjusted = polarity * sampled
            threshold_info = signed_threshold(adjusted, expected_bits)
            if threshold_info is None:
                continue
            threshold, decided, spread = threshold_info
            hamming = int(np.count_nonzero(decided != expected_bits))
            centered = adjusted - threshold
            expected_sign = np.where(expected_bits == 1, 1.0, -1.0)
            soft_score = float(np.mean(expected_sign * centered) / (np.std(centered) + 1e-9))
            soft_margin = float(np.mean(np.abs(centered)) / (np.std(centered) + 1e-9))
            local.append(
                {
                    "start": start,
                    "samples_per_bit": float(samples_per_bit),
                    "polarity": int(polarity),
                    "threshold": threshold,
                    "aa_hamming_distance": hamming,
                    "aa_soft_score": soft_score,
                    "aa_soft_margin": soft_margin,
                    "known_spread": spread,
                }
            )
        return local

    for samples_per_bit in samples_per_bit_values:
        begin = max(0, int(round(nominal_start - search_samples)))
        end = min(len(demod) - 2, int(round(nominal_start + search_samples)))
        search_width = end - begin
        if search_width <= 5_000:
            # A one-sample grid is intentional for short searches.
            for start in range(begin, end + 1):
                candidates.extend(score_at(demod, start, float(samples_per_bit)))
        else:
            # Long physical bursts can contain several adjacent BLE events.
            # Vectorize a coarse scan, then refine only its strongest starts;
            # scanning every sample with a Python loop is prohibitively slow.
            coarse_step = max(10, int(round(samples_per_bit / 2.0)))
            coarse_starts = np.arange(begin, end + 1, coarse_step, dtype=np.int64)
            if use_cuda:
                coarse = cuda_coarse_sync_candidates(
                    demod,
                    begin=begin,
                    end=end,
                    samples_per_bit=float(samples_per_bit),
                    expected_bits=expected_bits,
                    coarse_step=coarse_step,
                    max_candidates=max_candidates,
                    cuda_device=cuda_device,
                )
            else:
                coarse = []
                chunk_size = 20_000
                for offset in range(0, len(coarse_starts), chunk_size):
                    starts = coarse_starts[offset:offset + chunk_size]
                    indices = starts[:, None].astype(np.float64) + positions[None, :] * float(samples_per_bit) + float(samples_per_bit) / 2.0
                    lower = np.floor(indices).astype(np.int64)
                    valid = (lower[:, -1] < len(demod) - 1)
                    if not np.any(valid):
                        continue
                    lower = lower[valid]
                    fraction = (indices[valid] - lower).astype(np.float32)
                    sampled = demod[lower] * (1.0 - fraction) + demod[lower + 1] * fraction
                    for polarity in (1.0, -1.0):
                        adjusted = polarity * sampled
                        one = adjusted[:, expected_bits == 1]
                        zero = adjusted[:, expected_bits == 0]
                        threshold = (np.median(one, axis=1) + np.median(zero, axis=1)) / 2.0
                        decided = adjusted > threshold[:, None]
                        hamming = np.count_nonzero(decided != expected_bits[None, :], axis=1)
                        centered = adjusted - threshold[:, None]
                        expected_sign = np.where(expected_bits == 1, 1.0, -1.0)
                        soft_score = np.mean(expected_sign[None, :] * centered, axis=1) / (np.std(centered, axis=1) + 1e-9)
                        soft_margin = np.mean(np.abs(centered), axis=1) / (np.std(centered, axis=1) + 1e-9)
                        keep_count = min(12, len(hamming))
                        keep = np.argpartition(hamming, keep_count - 1)[:keep_count]
                        for index in keep:
                            coarse.append(
                                {
                                    "start": int(starts[valid][index]),
                                    "samples_per_bit": float(samples_per_bit),
                                    "polarity": int(polarity),
                                    "threshold": float(threshold[index]),
                                    "aa_hamming_distance": int(hamming[index]),
                                    "aa_soft_score": float(soft_score[index]),
                                    "aa_soft_margin": float(soft_margin[index]),
                                    "known_spread": float(np.median(one[index]) - np.median(zero[index])),
                                }
                            )
            coarse.sort(key=lambda row: (row["aa_hamming_distance"], -row["aa_soft_score"]))
            for item in coarse[:max_candidates * 2]:
                candidates.extend(
                    score_at(
                        demod,
                        max(begin, item["start"] - coarse_step),
                        float(samples_per_bit),
                    )
                )
                for start in range(
                    max(begin, item["start"] - coarse_step),
                    min(end, item["start"] + coarse_step) + 1,
                ):
                    candidates.extend(score_at(demod, start, float(samples_per_bit)))
    candidates.sort(
        key=lambda row: (
            row["aa_hamming_distance"],
            -row["aa_soft_score"],
            -row["aa_soft_margin"],
            abs(row["start"] - nominal_start),
        )
    )
    filtered = [row for row in candidates if row["aa_hamming_distance"] <= max_hamming]
    return (filtered or candidates)[:max_candidates]


def segment_thresholds(values: np.ndarray, segment_bits: int = 128) -> np.ndarray:
    thresholds = np.empty(len(values), dtype=np.float32)
    for start in range(0, len(values), segment_bits):
        end = min(len(values), start + segment_bits)
        segment = values[start:end]
        if len(segment) < 4:
            threshold = float(np.median(values))
        else:
            low = float(np.percentile(segment, 25.0))
            high = float(np.percentile(segment, 75.0))
            for _ in range(5):
                midpoint = (low + high) / 2.0
                low_values = segment[segment <= midpoint]
                high_values = segment[segment > midpoint]
                if low_values.size:
                    low = float(np.mean(low_values))
                if high_values.size:
                    high = float(np.mean(high_values))
            threshold = (low + high) / 2.0
        thresholds[start:end] = threshold
    return thresholds


def frame_structure_score(
    raw_tail_bits: np.ndarray,
    adjusted_tail: np.ndarray,
    thresholds: np.ndarray,
    *,
    channel: int,
    prefix_bytes: int,
    tail_len_bytes: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    decided = (adjusted_tail > thresholds).astype(np.uint8)
    tail = dewhiten_direct_tail(decided, channel, prefix_bytes)
    frame = legacy.phantom_frame_from_hex(tail.hex())
    if frame is None:
        return None, {
            "frame_hex": "",
            "frame_seq": "",
            "frame_payload_hex": "",
            "frame_payload_len": "",
            "frame_integrity_ok": "",
            "frame_len_matches": 0,
        }
    frame_len = int_or_none(frame.get("len"))
    frame_len_matches = bool(
        frame.get("frame_offset_bytes") == 0
        and len(frame.get("frame_hex", "")) // 2 == tail_len_bytes
        and frame_len == tail_len_bytes - 6
    )
    # PC and the length field are known without knowing the covert payload or
    # sequence.  Construct a zero-seq/zero-payload reference solely to score
    # these structural bits after the whitening offset.
    expected_bits = expected_structure_bits(channel, prefix_bytes, tail_len_bytes)
    known_positions = np.concatenate(
        [np.arange(16, dtype=np.int64), np.arange(32, 40, dtype=np.int64)]
    )
    available = min(len(known_positions), len(adjusted_tail))
    known_positions = known_positions[:available]
    expected_known = expected_bits[known_positions]
    actual_known = decided[known_positions]
    structure_errors = int(np.count_nonzero(actual_known != expected_known))
    structure_values = expected_sign_values(expected_known) * adjusted_tail[known_positions]
    structure_score = float(np.mean(structure_values) / (np.std(adjusted_tail) + 1e-9))
    return frame, {
        "frame_hex": frame.get("frame_hex", ""),
        "frame_seq": frame.get("seq", ""),
        "frame_payload_hex": frame.get("payload", ""),
        "frame_payload_len": frame_len if frame_len is not None else "",
        "frame_integrity_ok": frame.get("integrity_ok", ""),
        "frame_len_matches": int(frame_len_matches),
        "pc_structure_score": structure_score,
        "pc_structure_errors": structure_errors,
    }


def expected_sign_values(bits: np.ndarray) -> np.ndarray:
    return np.where(bits == 1, 1.0, -1.0)


def decode_hypothesis(
    demod: np.ndarray,
    sync: dict[str, Any],
    *,
    channel: int,
    payload_length: int,
    tail_len_bytes: int,
    timing_drift_ppm: float,
    threshold_mode: str,
    cfo_offset_hz: float,
    lowpass_hz: float,
) -> dict[str, Any]:
    prefix_bytes = 2 + payload_length + 3
    prefix_bits = (1 + 4 + prefix_bytes) * 8
    tail_bits_count = tail_len_bytes * 8
    positions = np.arange(tail_bits_count, dtype=np.float64)
    samples_per_bit = float(sync["samples_per_bit"])
    drift_factor = 1.0 + (timing_drift_ppm * 1e-6)
    tail_start = float(sync["start"]) + prefix_bits * samples_per_bit
    tail_positions = tail_start + positions * samples_per_bit * drift_factor + samples_per_bit / 2.0
    if tail_positions[-1] >= len(demod) - 1:
        return {"notes": "tail_out_of_window", "candidate_payload_length": payload_length}
    lower = np.floor(tail_positions).astype(np.int64)
    fraction = (tail_positions - lower).astype(np.float32)
    sampled = demod[lower] * (1.0 - fraction) + demod[lower + 1] * fraction
    adjusted = float(sync["polarity"]) * sampled
    global_threshold = float(sync["threshold"])
    thresholds = (
        np.full(len(adjusted), global_threshold, dtype=np.float32)
        if threshold_mode == "global"
        else segment_thresholds(adjusted)
    )
    frame, structure = frame_structure_score(
        np.empty(0, dtype=np.uint8),
        adjusted,
        thresholds,
        channel=channel,
        prefix_bytes=prefix_bytes,
        tail_len_bytes=tail_len_bytes,
    )
    centered = adjusted - thresholds
    soft_likelihood = float(np.mean(np.abs(centered)) / (np.std(adjusted) + 1e-9))
    return {
        **sync,
        **structure,
        "candidate_payload_length": payload_length,
        "tail_offset_bits": prefix_bits,
        "tail_end_sample": float(tail_positions[-1]),
        "timing_drift_ppm": timing_drift_ppm,
        "tail_threshold_mode": threshold_mode,
        "cfo_offset_hz": cfo_offset_hz,
        "lowpass_hz": lowpass_hz,
        "soft_likelihood": soft_likelihood,
        "notes": "" if frame is not None else "no_pc_frame",
    }


def candidate_rank(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        0 if row.get("frame_len_matches") else 1,
        -float(row.get("pc_structure_score") or -999.0),
        -float(row.get("soft_likelihood") or -999.0),
        int(row.get("aa_hamming_distance") or 999),
        -float(row.get("aa_soft_score") or -999.0),
        0 if row.get("frame_integrity_ok") == "1" else 1,
    )


def candidate_consensus_key(row: dict[str, Any]) -> tuple[str, ...]:
    """Key independent threshold/drift candidates by their decoded frame."""
    return (
        str(row.get("frame_seq", "")),
        normalize_hex(row.get("frame_payload_hex", "")),
        str(row.get("frame_payload_len", "")),
        str(row.get("frame_integrity_ok", "")),
    )


def apply_candidate_consensus(rows: list[dict[str, Any]]) -> None:
    support: dict[tuple[str, ...], int] = defaultdict(int)
    for row in rows:
        if row.get("frame_hex") and row.get("frame_len_matches"):
            support[candidate_consensus_key(row)] += 1
    for row in rows:
        row["consensus_support"] = support.get(candidate_consensus_key(row), 0)


def consensus_rank(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        0 if row.get("frame_len_matches") else 1,
        -int(row.get("consensus_support") or 0),
        *candidate_rank(row),
    )


def filter_and_demodulate(
    samples: np.ndarray,
    *,
    sample_rate_hz: float,
    center_frequency_hz: float,
    packet_frequency_hz: float,
    cfo_offset_hz: float,
    lowpass_hz: float,
    use_cuda: bool = False,
    cuda_device: int = 0,
    cuda_fir_backend: str = "kernel_multi_float",
) -> np.ndarray:
    if use_cuda:
        dsp = load_cuda_dsp()
        cp = dsp["cp"]
        cuda_backend = dsp["cuda_backend"]
        apply_cleanup_lpf = dsp["apply_cleanup_lpf"]
        design_cleanup_lpf = dsp["design_cleanup_lpf"]
        with cuda_backend.use_device(cuda_device):
            samples_gpu = cp.asarray(samples, dtype=cp.complex64)
            n = cp.arange(samples_gpu.size, dtype=cp.float32)
            offset_hz = (packet_frequency_hz - center_frequency_hz) + cfo_offset_hz
            shifted = samples_gpu * cp.exp(
                (-2j * cp.pi * offset_hz / sample_rate_hz) * n
            )
            taps = design_cleanup_lpf(sample_rate_hz, lowpass_hz, 129)
            filtered_gpu = apply_cleanup_lpf(
                shifted,
                taps,
                use_cuda=True,
                device_id=cuda_device,
                return_host=False,
                cuda_fir_backend=cuda_fir_backend,
            )
            phase = cp.unwrap(cp.angle(filtered_gpu))
            demod_gpu = cp.diff(phase).astype(cp.float32, copy=False)
            return cp.asnumpy(demod_gpu)
    n = np.arange(samples.size, dtype=np.float32)
    offset_hz = (packet_frequency_hz - center_frequency_hz) + cfo_offset_hz
    shifted = samples * np.exp((-2j * np.pi * offset_hz / sample_rate_hz) * n)
    taps = signal.firwin(129, lowpass_hz, fs=sample_rate_hz)
    return postprocess.gfsk_demodulate(signal.lfilter(taps, [1.0], shifted))


def detect_bursts_in_window(
    iq_path: Path,
    *,
    sample_rate_hz: float,
    center_frequency_hz: float,
    predicted_sample: int,
    packet_frequency_hz: float,
    search_radius_us: float,
    lowpass_hz: float,
    min_duration_us: float,
) -> list[dict[str, Any]]:
    radius = int(round(sample_rate_hz * search_radius_us / 1_000_000.0))
    expected_samples = int(round(sample_rate_hz * 2_200.0 / 1_000_000.0))
    start_sample = max(0, predicted_sample - radius)
    local_prediction = predicted_sample - start_sample
    sample_count = min(
        int(round(sample_rate_hz * 2.0 * search_radius_us / 1_000_000.0)) + expected_samples,
        2_000_000,
    )
    samples = postprocess.read_iq_window(iq_path, start_sample, sample_count)
    if samples.size < 1000:
        return []
    n = np.arange(samples.size, dtype=np.float32)
    offset_hz = packet_frequency_hz - center_frequency_hz
    shifted = samples * np.exp((-2j * np.pi * offset_hz / sample_rate_hz) * n)
    taps = signal.firwin(65, lowpass_hz, fs=sample_rate_hz)
    filtered = signal.lfilter(taps, [1.0], shifted)
    power = np.abs(filtered).astype(np.float32) ** 2
    smooth_size = max(1, int(round(sample_rate_hz * 2.0 / 1_000_000.0)))
    smooth = ndimage.uniform_filter1d(power, size=smooth_size, mode="nearest")
    noise_count = max(20, len(smooth) // 4)
    noise = np.partition(smooth, noise_count - 1)[:noise_count]
    noise_floor = float(np.median(noise))
    mad = float(np.median(np.abs(noise - noise_floor)))
    sigma = 1.4826 * mad
    threshold = noise_floor + max(8.0 * sigma, noise_floor * 2.0)
    above = postprocess.fill_short_false_gaps(
        smooth > threshold,
        max(1, int(round(sample_rate_hz * 5.0 / 1_000_000.0))),
    )
    changes = np.diff(np.concatenate(([False], above, [False])).astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    rows: list[dict[str, Any]] = []
    min_samples = int(round(sample_rate_hz * min_duration_us / 1_000_000.0))
    for local_start, local_end in zip(starts, ends):
        if local_end - local_start < min_samples:
            continue
        absolute_start = start_sample + int(local_start)
        absolute_end = start_sample + int(local_end)
        overlap = abs(absolute_start - predicted_sample) <= radius
        if not overlap:
            continue
        segment = smooth[local_start:local_end]
        peak = float(np.max(segment)) if segment.size else 0.0
        duration_us = (absolute_end - absolute_start) * 1_000_000.0 / sample_rate_hz
        rows.append(
            {
                "start_sample": absolute_start,
                "end_sample": absolute_end,
                "duration_us": duration_us,
                "predicted_sample": predicted_sample,
                "energy_threshold": threshold,
                "noise_floor": noise_floor,
                "peak_power": peak,
                "snr_db": 10.0 * math.log10((peak + 1e-12) / (noise_floor + 1e-12)),
                "long_burst": int(duration_us >= 1_900.0),
                "window_clipped": int(local_start == 0 or local_end == len(above)),
                "detector_notes": "" if not (local_start == 0 or local_end == len(above)) else "window_edge",
            }
        )
    rows.sort(key=lambda row: abs(row["start_sample"] - predicted_sample))
    return rows


def cluster_bursts(rows: list[dict[str, Any]], dedupe_samples: int) -> list[dict[str, Any]]:
    by_channel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_channel[str(row["channel"])].append(row)
    output: list[dict[str, Any]] = []
    counter = 0
    for channel, channel_rows in by_channel.items():
        channel_rows.sort(key=lambda row: (row["start_sample"], row.get("predicted_sample", 0)))
        current: dict[str, Any] | None = None
        evidence: list[dict[str, Any]] = []
        for row in channel_rows:
            if current is None or row["start_sample"] - current["start_sample"] > dedupe_samples:
                if current is not None:
                    output.append(finalize_burst(current, evidence, counter))
                    counter += 1
                current = dict(row)
                evidence = [row]
            else:
                evidence.append(row)
                current["start_sample"] = min(current["start_sample"], row["start_sample"])
                current["end_sample"] = max(current["end_sample"], row["end_sample"])
                current["duration_us"] = (current["end_sample"] - current["start_sample"]) * 1_000_000.0 / 100_000_000.0
                current["peak_power"] = max(current["peak_power"], row["peak_power"])
        if current is not None:
            output.append(finalize_burst(current, evidence, counter))
            counter += 1
    return sorted(output, key=lambda row: row["start_sample"])


def finalize_burst(row: dict[str, Any], evidence: list[dict[str, Any]], counter: int) -> dict[str, Any]:
    nearest = min(evidence, key=lambda item: abs(item["start_sample"] - item["predicted_sample"]))
    return {
        **row,
        "burst_id": f"burst:{counter}",
        "predicted_sample_nearest": nearest["predicted_sample"],
        "nearest_tx_attempt_id": nearest.get("tx_attempt_id", ""),
        "tx_window_hits": len({item.get("tx_attempt_id", "") for item in evidence}),
        "frequency_hz": nearest.get("frequency_hz", ""),
    }


def load_attempts(run_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ground_truth = read_csv(run_root / "ground_truth" / "rtt_ground_truth.csv")
    ll_rows = read_csv(run_root / "ground_truth" / "rtt_ll_tx.csv")
    old_matches = read_csv(run_root / "results" / "rtt_sdr_matches.csv")
    metadata = json.loads((run_root / "iq" / "metadata.json").read_text(encoding="utf-8"))
    attempts = matching._attempt_rows(ground_truth, ll_rows)
    alignment = matching._fit_affine(matching._old_exact_anchors(old_matches))
    sample_rate = float(metadata.get("actual_sample_rate_sps") or metadata["sample_rate_sps"])
    total_samples = int(metadata.get("samples", 0))
    gate_samples = max(
        sample_rate * 250.0 / 1_000_000.0,
        alignment["residual_p95_samples"] * 1.25 + sample_rate * 200.0 / 1_000_000.0,
        alignment["residual_max_samples"] * 1.05 + sample_rate * 200.0 / 1_000_000.0,
    )
    for attempt in attempts:
        seq = float_or_none(attempt.get("seq"))
        predicted = alignment["slope_samples_per_seq"] * seq + alignment["intercept_samples"] if seq is not None else None
        attempt["predicted_sample"] = predicted
        attempt["in_iq_window"] = bool(predicted is not None and 0 <= predicted < total_samples)
    return attempts, {
        "metadata": metadata,
        "alignment": alignment,
        "sample_rate_sps": sample_rate,
        "total_samples": total_samples,
        "gate_samples": gate_samples,
        "gate_us": gate_samples * 1_000_000.0 / sample_rate,
    }


def parser_length_map(run_root: Path) -> list[dict[str, str]]:
    return read_csv(run_root / "sdr" / "ble_packets.csv")


def nearest_parser_length(parser_rows: list[dict[str, str]], channel: str, sample: int) -> int | None:
    candidates = []
    for row in parser_rows:
        if str(row.get("channel", "")) != str(channel):
            continue
        row_sample = int_or_none(row.get("wideband_sample_index"))
        length = int_or_none(row.get("payload_len"))
        if row_sample is None or length is None:
            continue
        candidates.append((abs(row_sample - sample), length))
    if not candidates:
        return None
    return min(candidates)[1]


def decode_burst(
    iq_path: Path,
    *,
    burst: dict[str, Any],
    sample_rate_hz: float,
    center_frequency_hz: float,
    channel: int,
    access_address: bytes,
    variant: str,
    payload_lengths: tuple[int, ...],
    aa_tolerances: tuple[int, ...],
    lowpass_values: tuple[float, ...],
    cfo_values: tuple[float, ...],
    samples_per_bit_values: tuple[float, ...],
    drift_values: tuple[float, ...],
    tail_threshold_modes: tuple[str, ...],
    tail_len_bytes: int,
    use_cuda: bool = False,
    use_cuda_sync: bool = False,
    cuda_device: int = 0,
    cuda_fir_backend: str = "kernel_multi_float",
    max_sync_candidates: int = 32,
) -> list[dict[str, Any]]:
    start_sample = max(0, int(burst["start_sample"]) - int(round(sample_rate_hz * 25.0 / 1_000_000.0)))
    local_burst_start = int(burst["start_sample"]) - start_sample
    burst_span = max(2_500, int(burst["end_sample"]) - int(burst["start_sample"]) + 250)
    sample_count = burst_span + local_burst_start
    samples = postprocess.read_iq_window(iq_path, start_sample, sample_count)
    if samples.size == 0:
        return []
    all_results: list[dict[str, Any]] = []
    for lowpass_hz in lowpass_values:
        for cfo_offset_hz in cfo_values:
            demod = filter_and_demodulate(
                samples,
                sample_rate_hz=sample_rate_hz,
                center_frequency_hz=center_frequency_hz,
                packet_frequency_hz=float(burst["frequency_hz"]),
                cfo_offset_hz=cfo_offset_hz,
                lowpass_hz=lowpass_hz,
                use_cuda=use_cuda,
                cuda_device=cuda_device,
                cuda_fir_backend=cuda_fir_backend,
            )
            predicted_local = int(burst.get("predicted_sample_nearest") or burst["start_sample"]) - start_sample
            search_radius = max(
                10,
                abs(predicted_local - local_burst_start),
                abs((local_burst_start + burst_span) - predicted_local),
            )
            syncs = sync_hypotheses(
                demod,
                predicted_local,
                access_address,
                samples_per_bit_values=samples_per_bit_values,
                search_samples=search_radius,
                max_hamming=max(aa_tolerances),
                max_candidates=max_sync_candidates,
                use_cuda=use_cuda_sync,
                cuda_device=cuda_device,
            )
            for sync in syncs:
                for payload_length in payload_lengths:
                    for drift_ppm in drift_values:
                        for threshold_mode in tail_threshold_modes:
                            result = decode_hypothesis(
                                demod,
                                sync,
                                channel=channel,
                                payload_length=payload_length,
                                tail_len_bytes=tail_len_bytes,
                                timing_drift_ppm=drift_ppm,
                                threshold_mode=threshold_mode,
                                cfo_offset_hz=cfo_offset_hz,
                                lowpass_hz=lowpass_hz,
                            )
                            if result.get("notes") == "tail_out_of_window":
                                continue
                            result.update(
                                {
                                    "run_id": "",
                                    "variant": variant,
                                    "burst_id": burst["burst_id"],
                                    "channel": channel,
                                    "frequency_hz": burst["frequency_hz"],
                                    "burst_start_sample": burst["start_sample"],
                                    "burst_end_sample": burst["end_sample"],
                                    "burst_duration_us": burst["duration_us"],
                                    "sync_start_sample": start_sample + sync["start"],
                                    "hypothesis_rank": 0,
                                }
                            )
                            all_results.append(result)
    all_results.sort(key=candidate_rank)
    for rank, result in enumerate(all_results):
        result["hypothesis_rank"] = rank
    apply_candidate_consensus(all_results)
    selected: list[dict[str, Any]] = []
    for tolerance in aa_tolerances:
        valid = [row for row in all_results if row.get("aa_hamming_distance", 999) <= tolerance]
        if valid:
            selected.append({**min(valid, key=consensus_rank), "aa_tolerance": tolerance})
    return selected


def assignment_for_tx(
    attempts: list[dict[str, Any]],
    bursts: list[dict[str, Any]],
    decoded: list[dict[str, Any]],
    *,
    tolerance: int,
) -> dict[int, dict[str, Any]]:
    tx_rows = [row for row in attempts if row.get("in_iq_window")]
    decoded_by_burst: dict[str, dict[str, Any]] = {}
    for row in decoded:
        if int(row.get("aa_tolerance", tolerance)) != tolerance:
            continue
        current = decoded_by_burst.get(row["burst_id"])
        if current is None or candidate_rank(row) < candidate_rank(current):
            decoded_by_burst[row["burst_id"]] = row
    candidates = []
    for burst in bursts:
        decoded_row = decoded_by_burst.get(burst["burst_id"])
        if decoded_row is None:
            continue
        candidates.append({**decoded_row, "sample": float(burst["start_sample"])})
    # Physical bursts are globally unique.  The monotonic greedy pass keeps
    # duplicate burst consumption out of this first version; parser rows are
    # never used as the assignment key.
    assignments: dict[int, dict[str, Any]] = {}
    used: set[str] = set()
    for attempt in tx_rows:
        nearby = [
            row for row in candidates
            if str(row["channel"]) == str(int_or_none(attempt.get("channel")))
            and abs(row["sample"] - attempt["predicted_sample"]) <= attempts[0]["_gate_samples"]
            and row["burst_id"] not in used
        ]
        if not nearby:
            continue
        chosen = min(
            nearby,
            key=lambda row: (
                0 if row.get("frame_len_matches") else 1,
                0 if row.get("frame_integrity_ok") == "1" else 1,
                abs(row["sample"] - attempt["predicted_sample"]),
            ),
        )
        used.add(chosen["burst_id"])
        assignments[int(attempt["attempt_index"])] = chosen
    return assignments


def blind_row(result: dict[str, Any], run_id: str, tolerance: int) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "variant": result.get("variant", ""),
        "aa_tolerance": tolerance,
        **{field: result.get(field, "") for field in BLIND_FIELDS if field not in {"run_id", "variant", "aa_tolerance"}},
    }


def audit_rows(
    attempts: list[dict[str, Any]],
    assignments: dict[int, dict[str, Any]],
    *,
    run_id: str,
    variant: str,
    tolerance: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for attempt in attempts:
        if not attempt.get("in_iq_window"):
            continue
        result = assignments.get(int(attempt["attempt_index"]))
        tx_payload = normalize_hex(attempt.get("payload", ""))
        decoded_payload = normalize_hex(result.get("frame_payload_hex", "")) if result else ""
        decoded_seq = legacy.normalize_seq(result.get("frame_seq", "")) if result else ""
        seq = legacy.normalize_seq(attempt.get("seq"))
        integrity = bool(result and result.get("frame_integrity_ok") == "1")
        exact = bool(integrity and decoded_seq == seq and decoded_payload == tx_payload)
        marker = normalize_hex(attempt.get("marker", ""))
        tx_data = normalize_hex(attempt.get("data_payload", ""))
        decoded_data = legacy.payload_data_hex(decoded_payload, marker) if decoded_payload else ""
        data_exact = bool(exact and decoded_data == tx_data)
        rows.append(
            {
                "run_id": run_id,
                "variant": variant,
                "aa_tolerance": tolerance,
                "attempt_index": attempt["attempt_index"],
                "tx_attempt_id": attempt["tx_attempt_id"],
                "unique_notification_id": attempt["unique_notification_id"],
                "seq": seq,
                "channel": attempt.get("channel", ""),
                "predicted_sample": attempt.get("predicted_sample", ""),
                "in_iq_window": int(attempt.get("in_iq_window", False)),
                "target_long_burst_observed": int(bool(result)),
                "burst_id": result.get("burst_id", "") if result else "",
                "burst_start_sample": result.get("burst_start_sample", "") if result else "",
                "burst_duration_us": result.get("burst_duration_us", "") if result else "",
                "tail_candidate": int(bool(result and result.get("frame_hex"))),
                "valid_phantom_frame": int(bool(result and result.get("frame_len_matches"))),
                "blind_frame_seq": result.get("frame_seq", "") if result else "",
                "blind_payload_len": result.get("frame_payload_len", "") if result else "",
                "blind_integrity_ok": result.get("frame_integrity_ok", "") if result else "",
                "payload_exact": int(exact),
                "data_exact": int(data_exact),
                "physical_start_residual_samples": (
                    result.get("burst_start_sample", 0) - attempt.get("predicted_sample", 0) if result else ""
                ),
                "aa_hamming_distance": result.get("aa_hamming_distance", "") if result else "",
                "aa_soft_score": result.get("aa_soft_score", "") if result else "",
                "consensus_support": result.get("consensus_support", "") if result else "",
                "notes": "" if result else "no_long_burst_assigned",
            }
        )
    return rows


def retransmission_merge_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Post-selection audit for duplicate seq/notification retransmissions."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("variant", "")), str(row.get("aa_tolerance", "")), str(row.get("seq", "")))].append(row)
    merged: list[dict[str, Any]] = []
    for (variant, tolerance, seq), group in groups.items():
        if len(group) < 2:
            continue
        assigned = [row for row in group if row.get("burst_id")]
        exact_count = sum(int(row.get("data_exact", 0)) for row in group)
        merged.append(
            {
                "variant": variant,
                "aa_tolerance": tolerance,
                "seq": seq,
                "attempt_count": len(group),
                "assigned_count": len(assigned),
                "exact_count": exact_count,
                "merged_best_exact": int(exact_count > 0),
                "burst_ids": ";".join(str(row.get("burst_id", "")) for row in assigned),
                "attempt_indices": ";".join(str(row.get("attempt_index", "")) for row in group),
            }
        )
    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--burst-inventory", type=Path, default=None, help="reuse an existing independent burst_inventory.csv")
    parser.add_argument("--variants", default="configured_length_8", help="comma list: parser_length,configured_length_8,enumerated_length")
    parser.add_argument("--aa-tolerances", default="0,1,2,3")
    parser.add_argument("--lowpass-hz", default="900000")
    parser.add_argument("--cfo-offset-hz", default="0")
    parser.add_argument("--samples-per-bit", default="100")
    parser.add_argument("--timing-drift-ppm", default="0")
    parser.add_argument("--tail-threshold-modes", default="global,segment")
    parser.add_argument("--use-cuda", action="store_true", help="use PhantomChannel's CUDA environment for FIR and demodulation")
    parser.add_argument("--cuda-sync", action="store_true", help="also use CUDA for coarse AA synchronization (small per-burst batches may be slower)")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--cuda-fir-backend", default="kernel_multi_float")
    parser.add_argument("--max-sync-candidates", type=int, default=32)
    parser.add_argument("--search-radius-us", type=float, default=0.0)
    parser.add_argument("--min-burst-duration-us", type=float, default=1500.0)
    parser.add_argument("--max-tx", type=int, default=0)
    parser.add_argument("--dedupe-samples", type=int, default=2000)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    before_guard = ble_guard_snapshot()
    attempts, timing = load_attempts(run_root)
    in_window = [row for row in attempts if row.get("in_iq_window")]
    if args.max_tx > 0:
        selected_indices = {row["attempt_index"] for row in in_window[:args.max_tx]}
    else:
        selected_indices = {row["attempt_index"] for row in in_window}
    for row in attempts:
        row["_gate_samples"] = timing["gate_samples"]
    metadata = timing["metadata"]
    sample_rate_hz = timing["sample_rate_sps"]
    center_frequency_hz = float(metadata.get("actual_center_frequency_hz") or metadata["center_frequency_hz"])
    iq_path = run_root / "iq" / "capture.sc16"
    access_address = bytes.fromhex(TARGET_AA)
    search_radius_us = args.search_radius_us or timing["gate_us"]
    parser_rows = parser_length_map(run_root)
    cuda_status = None
    if args.use_cuda or args.cuda_sync:
        cuda_status = load_cuda_dsp()["cuda_backend"].describe_cuda_status()

    started = time.monotonic()
    detector_by_attempt: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if args.burst_inventory:
        bursts = read_csv(args.burst_inventory.resolve())
        for burst in bursts:
            for field in ("start_sample", "end_sample", "predicted_sample_nearest", "tx_window_hits"):
                if burst.get(field, "") != "":
                    burst[field] = int(float(burst[field]))
            for field in ("duration_us", "frequency_hz", "energy_threshold", "noise_floor", "peak_power", "snr_db"):
                if burst.get(field, "") != "":
                    burst[field] = float(burst[field])
    else:
        detector_rows: list[dict[str, Any]] = []
        for attempt in in_window:
            if attempt["attempt_index"] not in selected_indices:
                continue
            channel_i = int_or_none(attempt.get("channel"))
            predicted = int(round(attempt["predicted_sample"]))
            frequency_hz = legacy.ble_data_channel_frequency_hz(str(channel_i)) if channel_i is not None else None
            if frequency_hz is None:
                continue
            found = detect_bursts_in_window(
                iq_path,
                sample_rate_hz=sample_rate_hz,
                center_frequency_hz=center_frequency_hz,
                predicted_sample=predicted,
                packet_frequency_hz=frequency_hz,
                search_radius_us=search_radius_us,
                lowpass_hz=1_500_000.0,
                min_duration_us=args.min_burst_duration_us,
            )
            for row in found:
                row.update(
                    {
                        "channel": channel_i,
                        "frequency_hz": frequency_hz,
                        "tx_attempt_id": attempt["tx_attempt_id"],
                    }
                )
                detector_rows.append(row)
                detector_by_attempt[attempt["attempt_index"]].append(row)
        bursts = cluster_bursts(detector_rows, args.dedupe_samples)
    if args.max_tx > 0 and args.burst_inventory:
        selected_attempt_ids = {
            str(row["tx_attempt_id"])
            for row in in_window[:args.max_tx]
        }
        bursts = [
            row for row in bursts
            if str(row.get("nearest_tx_attempt_id", "")) in selected_attempt_ids
        ]
    burst_by_id = {row["burst_id"]: row for row in bursts}
    for row in bursts:
        row.setdefault("frequency_hz", legacy.ble_data_channel_frequency_hz(str(row.get("channel", ""))) or "")

    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    aa_tolerances = parse_grid(args.aa_tolerances, int)
    lowpass_values = parse_grid(args.lowpass_hz, float)
    cfo_values = parse_grid(args.cfo_offset_hz, float)
    samples_per_bit_values = parse_grid(args.samples_per_bit, float)
    drift_values = parse_grid(args.timing_drift_ppm, float)
    threshold_modes = parse_grid(args.tail_threshold_modes, str)
    decoded_rows: list[dict[str, Any]] = []
    decode_bursts = [row for row in bursts if str(row.get("long_burst", "1")) == "1"]
    for burst in decode_bursts:
        channel_i = int_or_none(burst.get("channel"))
        if channel_i is None:
            continue
        length_variants: dict[str, tuple[int, ...]] = {}
        if "configured_length_8" in variants:
            length_variants["configured_length_8"] = (8,)
        if "parser_length" in variants:
            parser_length = nearest_parser_length(parser_rows, str(channel_i), int(burst["start_sample"]))
            length_variants["parser_length"] = (parser_length if parser_length is not None else 8,)
        if "enumerated_length" in variants:
            # Full BLE data-channel PDU length enumeration.  The tail decoder
            # still selects only on physical frame structure and integrity;
            # RTT/parser payload fields are not part of this search.
            length_variants["enumerated_length"] = tuple(range(0, 252))
        for variant, lengths in length_variants.items():
            decoded_rows.extend(
                decode_burst(
                    iq_path,
                    burst=burst,
                    sample_rate_hz=sample_rate_hz,
                    center_frequency_hz=center_frequency_hz,
                    channel=channel_i,
                    access_address=access_address,
                    variant=variant,
                    payload_lengths=lengths,
                    aa_tolerances=aa_tolerances,
                    lowpass_values=lowpass_values,
                    cfo_values=cfo_values,
                    samples_per_bit_values=samples_per_bit_values,
                    drift_values=drift_values,
                    tail_threshold_modes=threshold_modes,
                    tail_len_bytes=TAIL_BYTES,
                    use_cuda=args.use_cuda,
                    use_cuda_sync=args.cuda_sync,
                    cuda_device=args.cuda_device,
                    cuda_fir_backend=args.cuda_fir_backend,
                    max_sync_candidates=args.max_sync_candidates,
                )
            )
    run_id = run_root.name
    blind_rows = [blind_row(row, run_id, int(row.get("aa_tolerance", max(aa_tolerances)))) for row in decoded_rows]
    write_csv(output_dir / "burst_inventory.csv", bursts, BURST_FIELDS)
    write_csv(output_dir / "blind_tail_candidates.csv", blind_rows, BLIND_FIELDS)

    audit_rows_all: list[dict[str, Any]] = []
    ablation: list[dict[str, Any]] = []
    for variant in variants:
        variant_rows = [row for row in decoded_rows if row.get("variant") == variant]
        for tolerance in aa_tolerances:
            assignments = assignment_for_tx(attempts, bursts, variant_rows, tolerance=tolerance)
            audit = audit_rows(attempts, assignments, run_id=run_id, variant=variant, tolerance=tolerance)
            audit_rows_all.extend(audit)
            window_audit = [row for row in audit if row["in_iq_window"]]
            ablation.append(
                {
                    "variant": variant,
                    "uses_parser_aa": False,
                    "aa_tolerance": tolerance,
                    "uses_parser_header_pdu_crc": variant == "parser_length",
                    "length_source": variant,
                    "tx_windows_evaluated": len(window_audit),
                    "target_long_burst_observed": sum(int(row["target_long_burst_observed"]) for row in window_audit),
                    "tail_candidates": sum(int(row["tail_candidate"]) for row in window_audit),
                    "valid_phantom_frames": sum(int(row["valid_phantom_frame"]) for row in window_audit),
                    "payload_exact": sum(int(row["payload_exact"]) for row in window_audit),
                    "data_exact": sum(int(row["data_exact"]) for row in window_audit),
                    "false_positive_frame_count": sum(
                        int(row["valid_phantom_frame"]) - int(row["payload_exact"]) for row in window_audit
                    ),
                }
            )
    write_csv(output_dir / "rtt_guided_audit.csv", audit_rows_all, AUDIT_FIELDS)
    write_csv(
        output_dir / "retransmission_merge.csv",
        retransmission_merge_rows(audit_rows_all),
        RETRANSMISSION_FIELDS,
    )
    write_csv(output_dir / "packet_funnel.csv", ablation, FUNNEL_FIELDS)
    write_json(output_dir / "ablation_summary.json", {
        "schema_version": 1,
        "rows": ablation,
        "detector_elapsed_s": time.monotonic() - started,
        "detector_windows_requested": len(selected_indices),
        "detector_windows_with_any_burst": (
            sum(bool(detector_by_attempt.get(index)) for index in selected_indices)
            if not args.burst_inventory
            else "reused_inventory"
        ),
        "burst_count": len(bursts),
        "decoded_candidate_count": len(decoded_rows),
        "notes": [
            "Tail-first candidate selection does not read RTT seq or payload.",
            "configured_length_8 uses only the experiment's configured 8-byte standard PDU length.",
            "parser_length uses parser payload_len only for its explicit control variant.",
            "enumerated_length covers the complete BLE payload-length domain 0..251.",
            "The physical detector uses IQ energy and channel frequency, not parser rows.",
        ],
    })
    after_guard = ble_guard_snapshot()
    input_files = {}
    for name, path in {
        "iq": iq_path,
        "iq_metadata": run_root / "iq" / "metadata.json",
        "rtt_ground_truth": run_root / "ground_truth" / "rtt_ground_truth.csv",
        "rtt_ll_tx": run_root / "ground_truth" / "rtt_ll_tx.csv",
        "old_matches": run_root / "results" / "rtt_sdr_matches.csv",
        "parser_rows_control_only": run_root / "sdr" / "ble_packets.csv",
    }.items():
        item = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            stat = path.stat()
            item.update({"size_bytes": stat.st_size, "sha256": sha256_file(path)})
        input_files[name] = item
    manifest = {
        "schema_version": 1,
        "analysis_id": "phase2_tail_first_v1",
        "run_root": str(run_root),
        "output_dir": str(output_dir),
        "target_access_address": TARGET_AA,
        "access_address_source": "bootstrap_from_frozen_exact_parser_observations",
        "timing": timing,
        "parameters": {
            "variants": variants,
            "aa_tolerances": aa_tolerances,
            "lowpass_hz": lowpass_values,
            "cfo_offset_hz": cfo_values,
            "samples_per_bit": samples_per_bit_values,
            "timing_drift_ppm": drift_values,
            "tail_threshold_modes": threshold_modes,
            "tail_length_bytes": TAIL_BYTES,
            "configured_payload_length": 8,
            "min_burst_duration_us": args.min_burst_duration_us,
            "search_radius_us": search_radius_us,
            "max_tx": args.max_tx,
            "use_cuda": args.use_cuda,
            "cuda_sync": args.cuda_sync,
            "cuda_device": args.cuda_device,
            "cuda_fir_backend": args.cuda_fir_backend,
            "max_sync_candidates": args.max_sync_candidates,
        },
        "cuda_status": cuda_status,
        "input_files": input_files,
        "ble_encrypt_check_guard": {
            "before": before_guard,
            "after": after_guard,
            "unchanged": before_guard == after_guard,
            "external_project_touched_by_this_tool": False,
        },
        "blind_selection": {
            "uses_rtt_payload": False,
            "uses_rtt_seq_for_hypothesis_selection": False,
            "uses_rtt_seq_for_posthoc_retransmission_merge": True,
            "uses_parser_pdu_crc_for_configured_length": False,
            "uses_rtt_channel_and_predicted_sample_for_diagnostic_window": True,
            "diagnostic_only": True,
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "ble_encrypt_check_guard.json", manifest["ble_encrypt_check_guard"])
    write_json(output_dir / "validation_report.json", {
        "analysis_id": "phase2_tail_first_v1",
        "diagnostic_only": True,
        "funnel_window_count": len(selected_indices),
        "burst_count": len(bursts),
        "funnel_closure_available": False,
        "ble_encrypt_check_unchanged": before_guard == after_guard,
        "frozen_main_result_untouched": True,
        "notes": [
            "This first tail-first pass is a diagnostic inventory and is not a blind end-to-end rate result.",
            "The configured-length path bypasses parser PDU/CRC bytes; parser_length is an explicit control.",
            "F0-F7 closure will be updated after physical burst_id matching is validated.",
        ],
    })
    return {
        "output_dir": str(output_dir),
        "windows": len(selected_indices),
        "burst_count": len(bursts),
        "decoded_candidates": len(decoded_rows),
        "ablation": ablation,
        "ble_encrypt_check_unchanged": before_guard == after_guard,
        "elapsed_s": time.monotonic() - started,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
