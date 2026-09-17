#!/usr/bin/env python3
"""Replay existing X310 IQ through an auditable BPL/CAF ablation harness.

The physical-burst denominator is produced locally from wideband IQ power.  No
parser CSV, RTT row, payload pattern, CRC result, or post-hoc reference is read
until after the four modes have generated their candidate rows.  The external
The PhantomChannel receiver source is treated as a read-only input; it
is never imported or invoked by this harness.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from tools.bpl_sync import BPLConfig, config_from_mapping, localize_preamble  # noqa: E402
from portable_paths import BLE_ROOT  # noqa: E402


SC16_BYTES_PER_COMPLEX_SAMPLE = 4
BLE_ENTRYPOINT = BLE_ROOT / "experiment/bt_40m_pfb_realtime.py"
MODES = ("baseline", "bpl_only", "caf_only", "bpl_caf")
BPL_MODES = {"bpl_only", "bpl_caf"}
CAF_MODES = {"caf_only", "bpl_caf"}
STAGE_ORDER = (
    "S0_PHYSICAL_BURST",
    "S1_BPL_ACCEPTED",
    "S2_SYMBOLS_AVAILABLE",
    "S3_AA_RECOVERED",
    "S4_LENGTH_VALID",
    "S5_CRC_OR_STRUCTURE_VALID",
    "S6_CAF_ACCEPTED",
    "S7_TARGET_EXACT",
)


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Iterable[str] | None = None) -> None:
    materialized = list(rows)
    names: list[str] = []
    seen: set[str] = set()
    for name in fieldnames or []:
        if name not in seen:
            names.append(name)
            seen.add(name)
    for row in materialized:
        for name in row:
            if name not in seen:
                names.append(name)
                seen.add(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def build_window_manifest(
    inventory: list[dict[str, Any]],
    run_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expand run-level source inventory into exact immutable crop records."""

    by_run = {str(row["run_id"]): row for row in inventory}
    output: list[dict[str, Any]] = []
    for item in run_summaries:
        summary = item.get("summary", item)
        source = by_run[str(summary["run_id"])]
        sample_offset = int(summary["sample_offset"])
        sample_count = int(summary["sample_count"])
        output.append(
            {
                "run_id": summary["run_id"],
                "condition": summary["condition"],
                "window": summary["window"],
                "source_root": source["source_root"],
                "iq_path": source["iq_path"],
                "metadata_path": source["metadata_path"],
                "iq_sha256": source["iq_sha256"],
                "actual_sample_rate_sps": source["actual_sample_rate_sps"],
                "sample_offset": sample_offset,
                "sample_count": sample_count,
                "byte_offset": sample_offset * SC16_BYTES_PER_COMPLEX_SAMPLE,
                "byte_count": sample_count * SC16_BYTES_PER_COMPLEX_SAMPLE,
            }
        )
    return output


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def git_snapshot(root: Path) -> dict[str, Any]:
    if not (root / ".git").exists():
        return {"root": str(root), "git_dir_present": False, "head": None, "status": ["NO_GIT_DIR"]}
    head = command_output(["git", "rev-parse", "HEAD"], root)
    status_text = command_output(["git", "status", "--porcelain=v1"], root) or ""
    return {
        "root": str(root),
        "git_dir_present": True,
        "head": head,
        "status": status_text.splitlines(),
    }


def guard_snapshot() -> dict[str, Any]:
    return {
        "phantomchannel": git_snapshot(PROJECT_ROOT),
        "receiver": {
            **git_snapshot(BLE_ROOT),
            "entrypoint": str(BLE_ENTRYPOINT),
            "entrypoint_sha256": sha256_file(BLE_ENTRYPOINT) if BLE_ENTRYPOINT.exists() else None,
        },
    }


def guard_unchanged(before: dict[str, Any], after: dict[str, Any]) -> bool:
    return before.get("receiver") == after.get("receiver")


def load_source_metadata(run: dict[str, Any]) -> tuple[Path, dict[str, Any], Path]:
    root = Path(str(run["path"])).expanduser().resolve()
    iq_path = root / "iq/capture.sc16"
    metadata_path = root / "iq/metadata.json"
    if not iq_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"missing IQ or metadata for {root}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return root, metadata, iq_path


def build_inventory(config: dict[str, Any], *, hash_sources: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in config.get("source_runs", []):
        root, metadata, iq_path = load_source_metadata(run)
        metadata_path = root / "iq/metadata.json"
        expected_bytes = int(metadata.get("samples", 0)) * SC16_BYTES_PER_COMPLEX_SAMPLE
        actual_bytes = iq_path.stat().st_size
        overflow = metadata.get("overflow_text_matches", [])
        row = {
            "run_id": run.get("run_id", root.name),
            "condition": run.get("condition", ""),
            "source_root": str(root),
            "iq_path": str(iq_path),
            "metadata_path": str(metadata_path),
            "device_model": metadata.get("device_model"),
            "center_frequency_hz": metadata.get("actual_center_frequency_hz"),
            "gain_db": metadata.get("actual_gain_db"),
            "actual_sample_rate_sps": metadata.get("actual_sample_rate_sps"),
            "bandwidth_hz": metadata.get("actual_rx_bandwidth_hz"),
            "sample_count": metadata.get("samples"),
            "received_duration_seconds": metadata.get("received_duration_seconds"),
            "iq_bytes": actual_bytes,
            "expected_iq_bytes": expected_bytes,
            "file_size_matches_metadata": actual_bytes == expected_bytes,
            "overflow_text_matches": json.dumps(overflow, separators=(",", ":")),
            "metadata_sha256": sha256_file(metadata_path),
            "iq_sha256": sha256_file(iq_path) if hash_sources else "NOT_COMPUTED",
        }
        rows.append(row)
    return rows


def read_iq(memmap: np.memmap, start: int, count: int) -> np.ndarray:
    start = max(0, int(start))
    end = min(len(memmap), start + max(0, int(count)))
    values = np.asarray(memmap[start:end], dtype=np.float32)
    if values.size == 0:
        return np.empty(0, dtype=np.complex64)
    return (values[:, 0] + 1j * values[:, 1]).astype(np.complex64)


def detect_physical_bursts(
    iq_path: Path,
    *,
    sample_offset: int,
    sample_count: int,
    sample_rate_sps: float,
    detector: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Detect wideband energy bursts without any parser-derived candidate."""

    total_samples = iq_path.stat().st_size // SC16_BYTES_PER_COMPLEX_SAMPLE
    memmap = np.memmap(iq_path, dtype="<i2", mode="r", shape=(total_samples, 2))
    block_samples = int(detector.get("block_samples", 10_000))
    block_means: list[float] = []
    cursor = int(sample_offset)
    end_sample = min(total_samples, cursor + int(sample_count))
    while cursor < end_sample:
        block_end = min(end_sample, cursor + block_samples)
        values = np.asarray(memmap[cursor:block_end], dtype=np.float32)
        power = values[:, 0] ** 2 + values[:, 1] ** 2
        block_means.append(float(np.mean(power)) if power.size else 0.0)
        cursor = block_end
    if not block_means:
        return [], {"baseline_power": None, "threshold_power": None, "active_blocks": 0, "block_count": 0}
    baseline = float(np.median(np.asarray(block_means)))
    threshold = baseline * float(detector.get("threshold_ratio", 4.0))
    active = np.asarray(block_means) >= threshold
    active_indices = np.flatnonzero(active)
    merge_gap = int(detector.get("merge_gap_blocks", 2))
    min_blocks = int(detector.get("minimum_blocks", 1))
    pre_blocks = int(detector.get("pre_blocks", 2))
    post_blocks = int(detector.get("post_blocks", 2))
    intervals: list[tuple[int, int]] = []
    if active_indices.size:
        first = int(active_indices[0])
        previous = first
        for index in active_indices[1:]:
            index = int(index)
            if index - previous <= merge_gap + 1:
                previous = index
                continue
            intervals.append((first, previous))
            first = previous = index
        intervals.append((first, previous))
    bursts: list[dict[str, Any]] = []
    for burst_index, (first, last) in enumerate(intervals):
        if last - first + 1 < min_blocks:
            continue
        start = max(int(sample_offset), int(sample_offset) + (first - pre_blocks) * block_samples)
        end = min(end_sample, int(sample_offset) + (last + 1 + post_blocks) * block_samples)
        bursts.append(
            {
                "burst_id": f"burst_{burst_index:06d}",
                "burst_index": burst_index,
                "physical_burst_start_sample": start,
                "physical_burst_end_sample": end,
                "physical_burst_duration_us": (end - start) * 1_000_000.0 / sample_rate_sps,
                "peak_block_power": float(max(block_means[first : last + 1])),
                "baseline_power": baseline,
                "threshold_power": threshold,
            }
        )
    summary = {
        "baseline_power": baseline,
        "threshold_power": threshold,
        "active_blocks": int(active_indices.size),
        "block_count": len(block_means),
        "physical_burst_count": len(bursts),
        "sample_offset": int(sample_offset),
        "sample_count": int(end_sample - sample_offset),
        "detector_source": "wideband_IQ_power_only; parser_rows_not_read",
    }
    return bursts, summary


def estimate_frequency_offset(
    memmap: np.memmap,
    burst: dict[str, Any],
    *,
    sample_rate_sps: float,
    nfft: int,
    bandwidth_hz: float,
) -> float:
    start = int(burst["physical_burst_start_sample"])
    end = int(burst["physical_burst_end_sample"])
    count = min(max(1024, end - start), nfft)
    raw = read_iq(memmap, start, count)
    if raw.size < 16:
        return 0.0
    window = np.hanning(raw.size).astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft(raw * window, n=nfft))
    frequencies = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate_sps))
    valid = np.abs(frequencies) <= bandwidth_hz / 2.0
    if not np.any(valid):
        return 0.0
    magnitudes = np.abs(spectrum)
    indices = np.flatnonzero(valid)
    return float(frequencies[indices[int(np.argmax(magnitudes[indices]))]])


def narrowband_burst(
    memmap: np.memmap,
    burst: dict[str, Any],
    *,
    sample_rate_sps: float,
    subband_sample_rate_sps: float,
    frequency_offset_hz: float,
    pre_margin_us: float = 20.0,
    post_margin_us: float = 40.0,
) -> tuple[np.ndarray, int]:
    pre = int(round(pre_margin_us * sample_rate_sps / 1_000_000.0))
    post = int(round(post_margin_us * sample_rate_sps / 1_000_000.0))
    raw_start = max(0, int(burst["physical_burst_start_sample"]) - pre)
    raw_end = min(len(memmap), int(burst["physical_burst_end_sample"]) + post)
    raw = read_iq(memmap, raw_start, raw_end - raw_start)
    if raw.size == 0:
        return np.empty(0, dtype=np.complex64), raw_start
    time = np.arange(raw.size, dtype=np.float64) / sample_rate_sps
    mixed = raw * np.exp(-2j * np.pi * float(frequency_offset_hz) * time).astype(np.complex64)
    ratio = sample_rate_sps / subband_sample_rate_sps
    if abs(ratio - round(ratio)) > 1e-6:
        raise ValueError("wideband/subband sample-rate ratio must be integral for this harness")
    decimation = int(round(ratio))
    narrow = signal.resample_poly(mixed, up=1, down=decimation, window=("kaiser", 8.6)).astype(np.complex64)
    narrow -= np.mean(narrow)
    return narrow, raw_start


def apply_bpl_compensation(
    iq: np.ndarray,
    bpl: dict[str, Any],
    *,
    sample_rate_sps: float,
) -> np.ndarray:
    """Apply the BPL residual CFO and packet phase normalization."""

    signal_iq = np.asarray(iq, dtype=np.complex64)
    indices = np.arange(signal_iq.size, dtype=np.float64)
    cfo = float(bpl.get("coarse_cfo_hz", 0.0) or 0.0)
    phase = float(bpl.get("phase_at_peak_rad", 0.0) or 0.0)
    rotation = np.exp(-2j * np.pi * cfo * indices / float(sample_rate_sps) - 1j * phase)
    return (signal_iq * rotation).astype(np.complex64)


def demodulate_bits(iq: np.ndarray, start: int, *, samples_per_symbol: int, max_symbols: int) -> tuple[list[int], float]:
    if start < 1 or start >= iq.size - 2:
        return [], 0.0
    phase_difference = np.angle(iq[1:] * np.conj(iq[:-1])).astype(np.float32)
    if phase_difference.size < samples_per_symbol:
        return [], 0.0
    kernel = np.ones(samples_per_symbol, dtype=np.float32) / float(samples_per_symbol)
    frequency = np.convolve(phase_difference, kernel, mode="same")
    positions = start + np.arange(max_symbols, dtype=np.int64) * samples_per_symbol
    positions = positions[positions < frequency.size]
    if positions.size == 0:
        return [], 0.0
    values = frequency[positions]
    bits = [int(value >= 0.0) for value in values]
    confidence = float(np.median(np.abs(values))) if values.size else 0.0
    return bits, confidence


def valid_access_address(value: int) -> bool:
    if value in {0, 0xFFFFFFFF}:
        return False
    bits = [(value >> index) & 1 for index in range(32)]
    transitions = sum(bits[index] != bits[index - 1] for index in range(1, len(bits)))
    return 2 <= transitions <= 24


def decode_candidate(
    iq: np.ndarray,
    *,
    candidate_start: int,
    burst_start_narrow: int,
    burst_end_narrow: int,
    samples_per_symbol: int,
    phase_offset_samples: int,
) -> dict[str, Any]:
    bits, symbol_confidence = demodulate_bits(
        iq,
        candidate_start + phase_offset_samples,
        samples_per_symbol=samples_per_symbol,
        max_symbols=8 + 32 + 16 + 251 * 8 + 24,
    )
    result: dict[str, Any] = {
        "candidate_start_sample": candidate_start + phase_offset_samples,
        "phase_offset_samples": phase_offset_samples,
        "symbols_available": bool(len(bits) >= 8 + 32 + 16),
        "symbol_count": len(bits),
        "symbol_confidence": symbol_confidence,
        "aa_value": "",
        "aa_recovered": False,
        "length_bytes": "",
        "length_valid": False,
        "structure_valid": False,
        "caf_accepted": False,
        "expected_end_sample": "",
        "stage": "S1_BPL_ACCEPTED",
        "failure_code": "D1_INSUFFICIENT_SYMBOLS",
        "candidate_score": 0,
        "bits": "".join(str(bit) for bit in bits),
    }
    if not result["symbols_available"]:
        return result
    aa_start = 8
    aa_value = sum(bits[aa_start + index] << index for index in range(32))
    aa_ok = valid_access_address(aa_value)
    result["aa_value"] = f"{aa_value:08X}"
    result["aa_recovered"] = aa_ok
    result["stage"] = "S3_AA_RECOVERED" if aa_ok else "S2_SYMBOLS_AVAILABLE"
    result["failure_code"] = "D3_LENGTH_INVALID" if aa_ok else "D2_AA_INVALID"
    if not aa_ok:
        return result
    header_start = aa_start + 32
    header = bits[header_start : header_start + 16]
    length = sum(header[8 + index] << index for index in range(6))
    result["length_bytes"] = length
    expected_symbols = (8 + 32 + 2 + length + 3) * 8
    expected_end = int(result["candidate_start_sample"]) + expected_symbols * samples_per_symbol
    length_ok = 0 <= length <= 251 and expected_end <= len(iq)
    result["length_valid"] = length_ok
    result["expected_end_sample"] = expected_end
    result["stage"] = "S4_LENGTH_VALID" if length_ok else "S3_AA_RECOVERED"
    if not length_ok:
        return result
    structure_ok = expected_end <= burst_end_narrow and int(result["candidate_start_sample"]) >= burst_start_narrow
    result["structure_valid"] = structure_ok
    result["stage"] = "S5_CRC_OR_STRUCTURE_VALID" if structure_ok else "S4_LENGTH_VALID"
    result["failure_code"] = "PASS" if structure_ok else "C1_PHY_LL_LENGTH_MISMATCH"
    if structure_ok:
        result["caf_accepted"] = True
        result["stage"] = "S6_CAF_ACCEPTED"
    result["candidate_score"] = int(aa_ok) + 2 * int(length_ok) + 3 * int(structure_ok)
    return result


def choose_winner(candidates: list[dict[str, Any]], *, mode: str) -> dict[str, Any] | None:
    if not candidates:
        return None
    ordered = sorted(
        candidates,
        key=lambda row: (
            -int(row.get("candidate_score", 0)),
            -float(row.get("symbol_confidence", 0.0)),
            abs(int(row.get("phase_offset_samples", 0))),
            int(row.get("candidate_start_sample", 0)),
        ),
    )
    winner = dict(ordered[0])
    winner["winner_policy"] = "max_structure_score_then_symbol_confidence_then_smallest_offset"
    winner["mode"] = mode
    return winner


def run_burst_modes(
    burst: dict[str, Any],
    *,
    memmap: np.memmap,
    config: dict[str, Any],
    bpl_config: BPLConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    sample_rate = float(config["sample_rate_sps"])
    subband_rate = float(config["subband_sample_rate_sps"])
    detector = config["physical_detector"]
    fft_samples = int(detector.get("frequency_fft_samples", 262144))
    frequency_offset = estimate_frequency_offset(
        memmap,
        burst,
        sample_rate_sps=sample_rate,
        nfft=fft_samples,
        bandwidth_hz=float(config.get("bandwidth_hz", 80e6)),
    )
    narrow, raw_start = narrowband_burst(
        memmap,
        burst,
        sample_rate_sps=sample_rate,
        subband_sample_rate_sps=subband_rate,
        frequency_offset_hz=frequency_offset,
    )
    raw_to_narrow = sample_rate / subband_rate
    burst_start_narrow = int(round((int(burst["physical_burst_start_sample"]) - raw_start) / raw_to_narrow))
    burst_end_narrow = int(round((int(burst["physical_burst_end_sample"]) - raw_start) / raw_to_narrow))
    pre_margin_samples = int(round(20.0 * subband_rate / 1_000_000.0))
    fixed_start = pre_margin_samples
    bpl_search_start = max(0, burst_start_narrow - int(bpl_config.search_guard_samples))
    bpl_search_end = min(narrow.size, burst_start_narrow + int(bpl_config.search_tail_samples))
    bpl_started = time.perf_counter()
    bpl = localize_preamble(
        narrow,
        bpl_config,
        search_start_sample=bpl_search_start,
        search_end_sample=bpl_search_end,
    )
    bpl_runtime_ms = (time.perf_counter() - bpl_started) * 1000.0
    bpl_compensated = apply_bpl_compensation(
        narrow,
        bpl,
        sample_rate_sps=subband_rate,
    )
    bpl_row = {
        "burst_id": burst["burst_id"],
        "frequency_offset_hz": frequency_offset,
        "narrowband_raw_start_sample": raw_start,
        "burst_start_narrow_sample": burst_start_narrow,
        "burst_end_narrow_sample": burst_end_narrow,
        "search_start_sample": bpl_search_start,
        "search_end_sample": bpl_search_end,
        "bpl_localization_runtime_ms": bpl_runtime_ms,
        "packet_start_minus_physical_burst_start_samples": (
            float(bpl["packet_start_sample"]) - burst_start_narrow
            if bpl.get("packet_start_sample") is not None
            else None
        ),
        **bpl,
    }
    caf_offsets = [int(value) for value in config.get("caf_offset_samples", list(range(-8, 9)))]
    candidates: list[dict[str, Any]] = []
    stage_events: list[dict[str, Any]] = []
    for mode in MODES:
        if mode == "baseline":
            starts = [fixed_start]
            offsets = [0]
        elif mode == "bpl_only":
            starts = [int(round(float(bpl["packet_start_sample"]))) if bpl.get("accepted") else -1]
            offsets = [0]
        elif mode == "caf_only":
            starts = [fixed_start]
            offsets = caf_offsets
        else:
            starts = [int(round(float(bpl["packet_start_sample"]))) if bpl.get("accepted") else -1]
            offsets = caf_offsets
        mode_candidates: list[dict[str, Any]] = []
        if starts[0] < 0:
            mode_candidates.append(
                {
                    "mode": mode,
                    "burst_id": burst["burst_id"],
                    "candidate_start_sample": -1,
                    "phase_offset_samples": 0,
                    "stage": "S0_PHYSICAL_BURST",
                    "failure_code": str(bpl.get("failure_code", "B4_START_OUT_OF_RANGE")),
                    "candidate_score": 0,
                    "symbols_available": False,
                    "aa_recovered": False,
                    "length_valid": False,
                    "structure_valid": False,
                    "caf_accepted": False,
                    "symbol_confidence": 0.0,
                }
            )
        else:
            for offset in offsets:
                decoded = decode_candidate(
                    bpl_compensated if mode in BPL_MODES else narrow,
                    candidate_start=starts[0],
                    burst_start_narrow=burst_start_narrow,
                    burst_end_narrow=burst_end_narrow,
                    samples_per_symbol=bpl_config.samples_per_symbol,
                    phase_offset_samples=offset,
                )
                if mode not in BPL_MODES:
                    # These paths begin from the fixed burst guard, not a BPL
                    # gate.  Preserve the common downstream stage names while
                    # making the entry stage explicit in raw stage_events.
                    if decoded.get("stage") == "S1_BPL_ACCEPTED":
                        decoded["stage"] = (
                            "S2_SYMBOLS_AVAILABLE"
                            if decoded.get("symbols_available")
                            else "S0_PHYSICAL_BURST"
                        )
                # CAF acceptance is a mode switch, not a property that may
                # leak from the common structural decoder into baseline or
                # BPL-only.  Those modes stop at the structure gate; CAF
                # modes are allowed to expose S6.
                if mode not in CAF_MODES:
                    decoded["caf_accepted"] = False
                    if decoded.get("stage") == "S6_CAF_ACCEPTED":
                        decoded["stage"] = "S5_CRC_OR_STRUCTURE_VALID"
                row = {
                    "mode": mode,
                    "burst_id": burst["burst_id"],
                    "frequency_offset_hz": frequency_offset,
                    **decoded,
                }
                if mode in BPL_MODES:
                    row.update(
                        {
                            "bpl_accepted": bool(bpl.get("accepted")),
                            "bpl_start_sample": bpl.get("packet_start_sample"),
                            "bpl_coarse_cfo_hz": bpl.get("coarse_cfo_hz"),
                            "bpl_phase_at_peak_rad": bpl.get("phase_at_peak_rad"),
                            "bpl_cfo_phase_compensation": True,
                        }
                    )
                mode_candidates.append(row)
                candidates.append(row)
        winner = choose_winner(mode_candidates, mode=mode)
        if winner is None:
            continue
        winner["physical_burst_start_sample"] = burst["physical_burst_start_sample"]
        winner["physical_burst_end_sample"] = burst["physical_burst_end_sample"]
        winner["narrowband_raw_start_sample"] = raw_start
        winner["subband_sample_rate_sps"] = subband_rate
        if mode in BPL_MODES:
            winner["bpl_normalized_peak"] = bpl.get("normalized_peak")
            winner["bpl_peak_to_second_ratio"] = bpl.get("peak_to_second_ratio")
            winner["bpl_peak_width_samples"] = bpl.get("peak_width_samples")
            winner["bpl_competing_peak_count"] = bpl.get("competing_peak_count")
        stage_events.append(winner)
    return bpl_row, candidates, stage_events


def parser_reference_rows(root: Path) -> list[dict[str, str]]:
    path = root / "results/parser_candidate_packets.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def add_posthoc_reference(
    stage_rows: list[dict[str, Any]],
    root: Path,
    sample_rate_sps: float,
    subband_sample_rate_sps: float,
) -> None:
    """Add compatibility-only parser comparison after mode selection."""

    reference = []
    for row in parser_reference_rows(root):
        try:
            sample = int(float(row.get("sample_index", "")))
        except ValueError:
            continue
        reference.append((sample, row))
    reference.sort(key=lambda item: item[0])
    for row in stage_rows:
        if not reference:
            row["posthoc_parser_match"] = False
            row["posthoc_target_exact"] = ""
            continue
        try:
            raw_start = int(
                round(
                    float(row.get("narrowband_raw_start_sample"))
                    + float(row.get("candidate_start_sample")) * sample_rate_sps / subband_sample_rate_sps
                )
            )
        except (TypeError, ValueError):
            raw_start = int(row.get("physical_burst_start_sample", 0))
        nearest = min(reference, key=lambda item: abs(item[0] - raw_start))
        distance = abs(nearest[0] - raw_start)
        matched = distance <= int(round(100.0 * sample_rate_sps / 1_000_000.0))
        row["posthoc_parser_match"] = matched
        row["posthoc_parser_sample_distance"] = distance
        row["posthoc_target_exact"] = nearest[1].get("pattern_exact", "") if matched else ""
        row["posthoc_integrity_ok"] = nearest[1].get("integrity_ok", "") if matched else ""
        # S7 is downstream of the CAF stage in the declared funnel.  Keep
        # compatibility exactness as a separate audit field for non-CAF modes;
        # only a CAF-enabled winner can advance its mode stage to S7.
        if (
            matched
            and row.get("mode") in {"caf_only", "bpl_caf"}
            and row.get("stage") == "S6_CAF_ACCEPTED"
            and str(row["posthoc_target_exact"]).strip().lower() in {"1", "true", "yes"}
        ):
            row["stage"] = "S7_TARGET_EXACT"


def mode_metric_rows(stage_rows: list[dict[str, Any]], *, run: dict[str, Any], window_label: str) -> list[dict[str, Any]]:
    total = len({str(row["burst_id"]) for row in stage_rows})
    rows: list[dict[str, Any]] = []
    for mode in MODES:
        selected = [row for row in stage_rows if row.get("mode") == mode]
        count = len(selected)
        def n(key: str) -> int:
            return sum(bool(row.get(key)) for row in selected)
        bpl_count = n("bpl_accepted") if mode in BPL_MODES else 0
        caf_candidate_count = bpl_count if mode == "bpl_caf" else count if mode == "caf_only" else 0
        rows.append(
            {
                "run_id": run["run_id"],
                "condition": run["condition"],
                "window": window_label,
                "mode": mode,
                "physical_bursts": total,
                "mode_rows": count,
                "bpl_accepted": n("bpl_accepted") if mode != "baseline" and mode != "caf_only" else "",
                "symbols_available": n("symbols_available"),
                "aa_recovered": n("aa_recovered"),
                "length_valid": n("length_valid"),
                "structure_valid": n("structure_valid"),
                "caf_accepted": n("caf_accepted") if mode in {"caf_only", "bpl_caf"} else "",
                "caf_candidate_count": caf_candidate_count if mode in CAF_MODES else "",
                "posthoc_target_exact": sum(str(row.get("posthoc_target_exact", "")) in {"1", "true", "True"} for row in selected),
                "physical_to_mode_rate": count / total if total else None,
                "aa_conditional_on_mode": n("aa_recovered") / count if count else None,
                "caf_conditional_on_candidates": n("caf_accepted") / caf_candidate_count if caf_candidate_count else "",
            }
        )
    return rows


def run_one_window(
    run: dict[str, Any],
    *,
    window: tuple[float, float],
    config: dict[str, Any],
    bpl_config: BPLConfig,
    output: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    root, metadata, iq_path = load_source_metadata(run)
    sample_rate = float(metadata.get("actual_sample_rate_sps") or config["sample_rate_sps"])
    start = int(round(float(window[0]) * sample_rate))
    count = int(round((float(window[1]) - float(window[0])) * sample_rate))
    total_samples = iq_path.stat().st_size // SC16_BYTES_PER_COMPLEX_SAMPLE
    count = min(count, max(0, total_samples - start))
    bursts, physical_summary = detect_physical_bursts(
        iq_path,
        sample_offset=start,
        sample_count=count,
        sample_rate_sps=sample_rate,
        detector=config["physical_detector"],
    )
    memmap = np.memmap(iq_path, dtype="<i2", mode="r", shape=(total_samples, 2))
    bpl_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    for burst in bursts:
        bpl_row, candidates, stages = run_burst_modes(
            burst,
            memmap=memmap,
            config=config,
            bpl_config=bpl_config,
        )
        bpl_rows.append({"run_id": run["run_id"], "condition": run["condition"], "window": f"{window[0]}-{window[1]}", **bpl_row})
        for row in candidates:
            candidate_rows.append({"run_id": run["run_id"], "condition": run["condition"], "window": f"{window[0]}-{window[1]}", **row})
        for row in stages:
            stage_rows.append({"run_id": run["run_id"], "condition": run["condition"], "window": f"{window[0]}-{window[1]}", **row})
    # This is intentionally after the four-mode candidate generation.  It is a
    # compatibility audit only and cannot alter a winner or a denominator.
    add_posthoc_reference(stage_rows, root, sample_rate, float(config["subband_sample_rate_sps"]))
    metrics = mode_metric_rows(stage_rows, run=run, window_label=f"{window[0]}-{window[1]}")
    summary = {
        "run_id": run["run_id"],
        "condition": run["condition"],
        "window": f"{window[0]}-{window[1]}",
        "sample_offset": start,
        "sample_count": count,
        "physical_summary": physical_summary,
        "bpl_rows": len(bpl_rows),
        "candidate_rows": len(candidate_rows),
        "stage_rows": len(stage_rows),
        "posthoc_reference_used_after_mode_generation": True,
    }
    output.mkdir(parents=True, exist_ok=True)
    return bpl_rows, candidate_rows, stage_rows, {"summary": summary, "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/bpl_caf_ablation.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase-derived-timing", action="store_true", help="refuse: experimental branch is not allowed in main results")
    parser.add_argument("--window-set", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--no-source-hash", action="store_true")
    parser.add_argument(
        "--reuse-inventory",
        type=Path,
        default=None,
        help="reuse previously recorded source hashes from an inventory CSV without rereading IQ for hashing",
    )
    args = parser.parse_args()
    if args.phase_derived_timing:
        parser.error("phase-derived timing is not enabled; use peak interpolation only")
    config = read_yaml(args.config)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    before_guard = guard_snapshot()
    write_json(output / "receiver_guard.json", {"before": before_guard})
    inventory = build_inventory(config, hash_sources=not args.no_source_hash and args.reuse_inventory is None)
    source_hashes_reused = False
    if args.reuse_inventory is not None:
        previous_rows = {
            row.get("run_id", ""): row
            for row in read_csv_rows(args.reuse_inventory.expanduser().resolve())
        }
        for row in inventory:
            previous = previous_rows.get(row["run_id"])
            if previous is None or previous.get("iq_sha256") in {None, "", "NOT_COMPUTED"}:
                raise RuntimeError(f"reusable inventory lacks IQ hash for {row['run_id']}")
            if previous.get("iq_path") != row.get("iq_path") or previous.get("iq_bytes") != str(row.get("iq_bytes")):
                raise RuntimeError(f"reusable inventory does not match current source for {row['run_id']}")
            row["iq_sha256"] = previous["iq_sha256"]
            source_hashes_reused = True
    write_csv(output / "dataset_inventory.csv", inventory)
    write_json(output / "manifest.json", {
        "analysis_id": config.get("analysis_id"),
        "config_path": str(args.config.expanduser().resolve()),
        "config_sha256": sha256_file(args.config.expanduser().resolve()),
        "window_set": args.window_set,
        "source_runs": [row["run_id"] for row in inventory],
        "source_hashes_computed": any(row.get("iq_sha256") not in {None, "", "NOT_COMPUTED"} for row in inventory),
        "source_hashes_reused": source_hashes_reused,
        "bpl_module": str(PROJECT_ROOT / "tools/bpl_sync.py"),
        "implementation_hashes": {
            "bpl_sync": sha256_file(PROJECT_ROOT / "tools/bpl_sync.py"),
            "run_bpl_caf_ablation": sha256_file(PROJECT_ROOT / "tools/run_bpl_caf_ablation.py"),
        },
        "candidate_denominator": "local wideband IQ power detector; parser rows not used before mode generation",
        "channelizer": "single-channel DDC followed by scipy resample_poly anti-alias decimation; no external parser import",
        "run_split_policy": "post_hoc_run_level; two repetitions per condition are insufficient for a three-way split; no condition-specific evaluation retuning",
        "development_windows_seconds": config.get("development_windows_seconds", []),
        "formal_windows_seconds": config.get("formal_windows_seconds", []),
        "parameter_freeze": {
            "source": "configs/bpl_caf_ablation.yaml",
            "evaluation_retuning": False,
            "bpl": config.get("bpl", {}),
            "physical_detector": config.get("physical_detector", {}),
            "caf_offset_samples": config.get("caf_offset_samples", []),
        },
        "phase_derived_timing": False,
    })
    bpl_config = config_from_mapping(config["bpl"])
    windows = config["development_windows_seconds"] if args.window_set == "smoke" else config["formal_windows_seconds"]
    runs = list(config.get("source_runs", []))
    if args.max_runs > 0:
        runs = runs[: args.max_runs]
    all_bpl: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    all_stages: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    for run in runs:
        for window in windows:
            bpl_rows, candidate_rows, stage_rows, result = run_one_window(
                run,
                window=(float(window[0]), float(window[1])),
                config=config,
                bpl_config=bpl_config,
                output=output,
            )
            all_bpl.extend(bpl_rows)
            all_candidates.extend(candidate_rows)
            all_stages.extend(stage_rows)
            all_metrics.extend(result["metrics"])
            run_summaries.append(result["summary"])
            print(json.dumps(result["summary"], ensure_ascii=False))
    write_csv(output / "bpl_candidates.csv", all_bpl)
    write_csv(output / "candidate_events.csv", all_candidates)
    write_csv(output / "stage_events.csv", all_stages)
    write_csv(output / "per_run_metrics.csv", all_metrics)
    window_manifest = build_window_manifest(inventory, run_summaries)
    write_csv(output / "window_manifest.csv", window_manifest)
    write_json(output / "run_summaries.json", run_summaries)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(
        {
            "window_manifest": str(output / "window_manifest.csv"),
            "window_manifest_rows": len(window_manifest),
            "formal_replay_total_seconds": sum(
                float(row["sample_count"]) / float(row["actual_sample_rate_sps"])
                for row in window_manifest
            ),
        }
    )
    write_json(output / "manifest.json", manifest)
    after_guard = guard_snapshot()
    guard_result = {
        "before": before_guard,
        "after": after_guard,
        "unchanged": guard_unchanged(before_guard, after_guard),
        "status": "PASS" if guard_unchanged(before_guard, after_guard) else "FAIL_EXTERNAL_DEPENDENCY_CHANGED",
    }
    write_json(output / "receiver_guard.json", guard_result)
    if not guard_result["unchanged"]:
        raise RuntimeError("PhantomChannel receiver read-only guard changed during experiment")
    write_json(output / "run_status.json", {
        "status": "completed",
        "window_set": args.window_set,
        "run_count": len(runs),
        "window_count": len(windows),
        "physical_bpl_candidate_count": len(all_bpl),
        "candidate_event_count": len(all_candidates),
        "stage_event_count": len(all_stages),
        "metrics_row_count": len(all_metrics),
        "guard": guard_result,
    })
    print(json.dumps({"status": "completed", "output_dir": str(output), "physical_bpl_candidates": len(all_bpl), "stage_events": len(all_stages)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
