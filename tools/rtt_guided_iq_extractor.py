#!/usr/bin/env python3
"""RTT-guided PhantomChannel IQ extractor.

This tool uses RTT LL TX ground truth as the packet list and only uses SDR
parser matches as optional timing anchors. Per-packet extraction is performed
directly from IQ around the RTT-predicted sample index.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import match_rtt_sdr_results as matcher  # noqa: E402
import phantom_postprocess_scorer as postprocess  # noqa: E402


OUTPUT_FIELDS = [
    "run_id",
    "seq",
    "channel",
    "tx_frequency_hz",
    "tx_in_analysis_band",
    "access_address",
    "parser_access_address",
    "normal_pdu_len",
    "air_extra_len",
    "tx_covert_hex",
    "tx_covert_data_hex",
    "predicted_wideband_sample_index",
    "search_radius_us",
    "extracted_post_crc_hex",
    "extracted_frame_hex",
    "extracted_seq",
    "extracted_payload_hex",
    "extracted_payload_marker_ok",
    "extracted_data_hex",
    "extracted_integrity_ok",
    "extracted_exact_match",
    "extracted_data_exact_match",
    "known_bit_errors",
    "known_bit_error_rate",
    "sync_tail_score_errors",
    "sync_tail_score_ber",
    "polarity",
    "threshold",
    "bit_start_sample",
    "samples_per_bit",
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


def int_or_none(value: Any) -> int | None:
    parsed = matcher.int_or_empty(value)
    return parsed if isinstance(parsed, int) else None


def float_or_none(value: Any) -> float | None:
    return matcher.float_or_none(value)


def load_metadata(run_root: Path, run_id: str, distance_m: str, analysis_bandwidth_hz: str) -> dict[str, Any]:
    metadata_path = run_root / "iq" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["run_id"] = run_id or run_root.name
    if distance_m:
        metadata["distance_m"] = distance_m
    metadata["analysis_bandwidth_hz"] = (
        float(analysis_bandwidth_hz)
        if analysis_bandwidth_hz
        else float(metadata.get("actual_sample_rate_sps") or metadata.get("sample_rate_sps"))
    )
    return metadata


def timing_anchors_from_matches(path: Path) -> list[tuple[int, int]]:
    if not path.is_file():
        return []
    anchors: list[tuple[int, int]] = []
    for row in matcher.read_csv(path):
        if str(row.get("covert_exact_match", "")) != "1":
            continue
        seq = int_or_none(row.get("seq"))
        sample = int_or_none(row.get("sdr_wideband_sample_index"))
        if seq is not None and sample is not None:
            anchors.append((seq, sample))
    return anchors


def fit_seq_to_sample(anchors: list[tuple[int, int]]) -> tuple[float, float]:
    if len(anchors) < 2:
        raise ValueError("at least two exact timing anchors are required")
    seq_values = np.asarray([item[0] for item in anchors], dtype=np.float64)
    sample_values = np.asarray([item[1] for item in anchors], dtype=np.float64)
    slope, intercept = np.polyfit(seq_values, sample_values, 1)
    return float(slope), float(intercept)


def choose_search_radius_us(slope_samples_per_seq: float, sample_rate_hz: float, requested_us: float | None) -> float:
    if requested_us is not None:
        return requested_us
    interval_us = abs(slope_samples_per_seq) * 1_000_000.0 / sample_rate_hz
    if interval_us <= 0:
        return 8_000.0
    return max(2_000.0, min(30_000.0, interval_us * 0.35))


def active_tx_window_from_sparse_seq_timestamps(tx_rows: list[dict[str, Any]]) -> tuple[float, float, str]:
    seq_values = [
        seq
        for seq in (int_or_none(row.get("seq")) for row in tx_rows if row.get("payload"))
        if seq is not None
    ]
    timestamped = sorted(
        (seq, timestamp)
        for row in tx_rows
        if row.get("payload")
        for seq, timestamp in [(int_or_none(row.get("seq")), float_or_none(row.get("timestamp_us")))]
        if seq is not None and timestamp is not None
    )
    if len(seq_values) < 2 or len(timestamped) < 2:
        active_duration_s, active_interval_s = matcher.active_tx_window_seconds(tx_rows)
        return active_duration_s, active_interval_s, "timestamp_median_interval"

    per_seq_intervals = [
        (right_time - left_time) / (right_seq - left_seq)
        for (left_seq, left_time), (right_seq, right_time) in zip(timestamped, timestamped[1:])
        if right_seq > left_seq and right_time > left_time
    ]
    if not per_seq_intervals:
        active_duration_s, active_interval_s = matcher.active_tx_window_seconds(tx_rows)
        return active_duration_s, active_interval_s, "timestamp_median_interval"

    interval_us = float(np.median(np.asarray(per_seq_intervals, dtype=np.float64)))
    active_seq_span = (max(seq_values) - min(seq_values)) + 1
    return (active_seq_span * interval_us) / 1_000_000.0, interval_us / 1_000_000.0, "seq_normalized_sparse_rtt_timestamp"


def phantom_frame_bytes(seq: int, payload_hex: str) -> bytes:
    payload = bytes.fromhex(matcher.normalize_hex(payload_hex))
    if len(payload) > 255:
        raise ValueError(f"covert payload too long for PC frame: {len(payload)}")
    frame = bytearray(b"PC")
    frame.extend((seq & 0xFF, (seq >> 8) & 0xFF, len(payload)))
    frame.extend(payload)
    check = 0
    for byte in frame:
        check ^= byte
    frame.append(check)
    return bytes(frame)


def _sample_demod_bits(
    demod: np.ndarray,
    bit_start: float,
    bit_positions: np.ndarray,
    samples_per_bit: float,
) -> np.ndarray | None:
    indices = bit_start + (bit_positions * samples_per_bit) + (samples_per_bit // 2)
    if indices.size == 0 or indices[0] < 0 or indices[-1] >= len(demod) - 1:
        return None
    lower = np.floor(indices).astype(np.int64)
    fraction = (indices - lower).astype(np.float32)
    return (demod[lower] * (1.0 - fraction)) + (demod[lower + 1] * fraction)


def _best_threshold_score(sampled: np.ndarray, expected_bits: np.ndarray) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for polarity in (1.0, -1.0):
        adjusted = polarity * sampled
        one_values = adjusted[expected_bits == 1]
        zero_values = adjusted[expected_bits == 0]
        if one_values.size == 0 or zero_values.size == 0:
            continue
        threshold = float((np.median(one_values) + np.median(zero_values)) / 2.0)
        decided = (adjusted > threshold).astype(np.uint8)
        errors = int(np.count_nonzero(decided != expected_bits))
        item = {
            "errors": errors,
            "ber": errors / len(expected_bits),
            "polarity": polarity,
            "threshold": threshold,
        }
        if best is None or item["errors"] < best["errors"]:
            best = item
    return best


def extract_known_pc_frame_near_predicted_start(
    *,
    iq_path: Path,
    sample_rate_hz: float,
    center_frequency_hz: float,
    predicted_start_sample: int,
    packet_frequency_hz: float,
    access_address_text: str,
    channel: int,
    normal_len: int,
    expected_frame: bytes,
    lowpass_hz: float,
    search_radius_us: float,
) -> dict[str, Any]:
    try:
        access_address = postprocess.access_address_bytes(access_address_text)
    except ValueError as exc:
        return {"notes": f"invalid_access_address:{exc}"}

    nominal_samples_per_bit = max(1, int(round(sample_rate_hz / 1_000_000.0)))
    search_radius_samples = int(round(sample_rate_hz * search_radius_us / 1_000_000.0))
    standard_prefix_bits = (1 + 4 + 2 + normal_len + 3) * 8
    tail_bits_count = len(expected_frame) * 8
    total_bits = standard_prefix_bits + tail_bits_count
    post_samples = total_bits * nominal_samples_per_bit + int(round(sample_rate_hz * 80.0 / 1_000_000.0))
    start_sample = max(0, predicted_start_sample - search_radius_samples)
    packet_local_prediction = predicted_start_sample - start_sample
    samples = postprocess.read_iq_window(iq_path, start_sample, (2 * search_radius_samples) + post_samples)
    if samples.size == 0:
        return {"notes": "prediction_iq_window_empty"}

    n = np.arange(samples.size, dtype=np.float32)
    shifted = samples * np.exp((-2j * np.pi * (packet_frequency_hz - center_frequency_hz) / sample_rate_hz) * n)
    taps = postprocess.signal.firwin(129, lowpass_hz, fs=sample_rate_hz)
    filtered = postprocess.signal.lfilter(taps, [1.0], shifted)
    demod = postprocess.gfsk_demodulate(filtered)

    sync_bits = postprocess.bytes_to_lsb_bits(bytes([0xAA]) + access_address)
    dummy_prefix = bytes(2 + normal_len + 3)
    expected_tail_air_bytes = postprocess.ble_whiten(dummy_prefix + expected_frame, channel)[len(dummy_prefix):]
    expected_tail_bits = postprocess.bytes_to_lsb_bits(expected_tail_air_bytes)

    # Score the sync plus a representative subset of the known post-CRC tail.
    # The tail subset includes the PC prefix, early payload, and final checksum so
    # a long payload cannot pass simply by matching its first few bytes.
    first_tail_bits = min(len(expected_tail_bits), 40 * 8)
    last_tail_bits = min(len(expected_tail_bits), 12 * 8)
    tail_positions = list(range(first_tail_bits))
    last_start = max(first_tail_bits, len(expected_tail_bits) - last_tail_bits)
    tail_positions.extend(range(last_start, len(expected_tail_bits)))
    tail_positions_np = np.asarray(sorted(set(tail_positions)), dtype=np.int64)
    bit_positions = np.concatenate(
        [
            np.arange(len(sync_bits), dtype=np.int64),
            standard_prefix_bits + tail_positions_np,
        ]
    )
    expected_bits = np.concatenate([sync_bits, expected_tail_bits[tail_positions_np]])

    search_begin = max(0, packet_local_prediction - search_radius_samples)
    search_end = min(len(demod) - (total_bits * nominal_samples_per_bit) - 1, packet_local_prediction + search_radius_samples)
    if search_end <= search_begin:
        return {"notes": "prediction_search_window_empty"}

    best: dict[str, Any] | None = None

    def consider(bit_start: float, samples_per_bit: float) -> None:
        nonlocal best
        sampled = _sample_demod_bits(demod, bit_start, bit_positions, samples_per_bit)
        if sampled is None:
            return
        score = _best_threshold_score(sampled, expected_bits)
        if score is None:
            return
        candidate = {**score, "start": bit_start, "samples_per_bit": samples_per_bit}
        rank = (candidate["errors"], abs(candidate["start"] - packet_local_prediction))
        if best is None or rank < (best["errors"], abs(best["start"] - packet_local_prediction)):
            best = candidate

    coarse_step = max(1, nominal_samples_per_bit)
    for bit_start in range(search_begin, search_end + 1, coarse_step):
        consider(float(bit_start), float(nominal_samples_per_bit))

    if best is None:
        return {"notes": "known_prediction_no_candidate"}

    refine_begin = max(search_begin, int(round(best["start"])) - nominal_samples_per_bit)
    refine_end = min(search_end, int(round(best["start"])) + nominal_samples_per_bit)
    bit_periods = np.linspace(nominal_samples_per_bit * 0.9990, nominal_samples_per_bit * 1.0010, 41)
    for bit_start in range(refine_begin, refine_end + 1):
        for bit_period in bit_periods:
            consider(float(bit_start), float(bit_period))

    if best is None:
        return {"notes": "known_prediction_no_refined_candidate"}

    full_positions = standard_prefix_bits + np.arange(tail_bits_count, dtype=np.int64)
    sampled = _sample_demod_bits(demod, best["start"], full_positions, best["samples_per_bit"])
    if sampled is None:
        return {"notes": "known_prediction_tail_out_of_window"}
    adjusted = best["polarity"] * sampled
    tail_whitened_bits = (adjusted > best["threshold"]).astype(np.uint8)
    tail = postprocess.dewhiten_tail_from_whitened_bits(tail_whitened_bits, dummy_prefix, channel)
    frame = matcher.phantom_frame_from_hex(tail.hex())
    expected_bits_plain = postprocess.bytes_to_lsb_bits(expected_frame)
    decoded_bits_plain = postprocess.bytes_to_lsb_bits(tail[:len(expected_frame)])
    full_errors = int(np.count_nonzero(decoded_bits_plain != expected_bits_plain))
    return {
        "predicted_extracted_post_crc_hex": tail.hex(),
        "predicted_extracted_frame_hex": frame["frame_hex"] if frame else "",
        "predicted_extracted_seq": frame["seq"] if frame else "",
        "predicted_extracted_payload_hex": frame["payload"] if frame else "",
        "predicted_extracted_integrity_ok": frame["integrity_ok"] if frame else "",
        "predicted_known_bit_errors": full_errors,
        "predicted_known_bit_error_rate": f"{full_errors / max(1, len(expected_bits_plain)):.9f}",
        "predicted_bit_start_sample": start_sample + best["start"],
        "predicted_samples_per_bit": f"{best['samples_per_bit']:.6f}",
        "predicted_sync_tail_score_errors": best["errors"],
        "predicted_sync_tail_score_ber": f"{best['ber']:.9f}",
        "predicted_polarity": int(best["polarity"]),
        "predicted_threshold": best["threshold"],
        "notes": "",
    }


def extract_rows(
    run_root: Path,
    output_dir: Path,
    run_id: str,
    distance_m: str,
    analysis_bandwidth_hz: str,
    search_radius_us: float | None,
    lowpass_hz: float,
    anchors_path: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata = load_metadata(run_root, run_id, distance_m, analysis_bandwidth_hz)
    sample_rate_hz = float(metadata.get("actual_sample_rate_sps") or metadata["sample_rate_sps"])
    center_frequency_hz = float(metadata.get("actual_center_frequency_hz") or metadata["center_frequency_hz"])
    iq_path = run_root / "iq" / "capture.sc16"
    rtt_ground_truth = matcher.read_csv(run_root / "ground_truth" / "rtt_ground_truth.csv")
    rtt_ll = matcher.read_csv(run_root / "ground_truth" / "rtt_ll_tx.csv")
    tx_rows = matcher.joined_rtt_tx_rows(rtt_ground_truth, rtt_ll)

    anchor_file = anchors_path or (run_root / "results" / "rtt_sdr_matches.csv")
    anchors = timing_anchors_from_matches(anchor_file)
    slope, intercept = fit_seq_to_sample(anchors)
    effective_search_radius_us = choose_search_radius_us(slope, sample_rate_hz, search_radius_us)

    rows: list[dict[str, Any]] = []
    for tx in tx_rows:
        if not tx.get("payload"):
            continue
        frequency_hz, in_band = matcher.channel_in_analysis_band(tx.get("channel"), metadata)
        if in_band is False:
            continue
        seq = int_or_none(tx.get("seq"))
        channel_i = int_or_none(tx.get("channel"))
        normal_len = int_or_none(tx.get("normal_pdu_len"))
        tail_len = int_or_none(tx.get("air_extra_len"))
        notes: list[str] = []
        if seq is None:
            notes.append("missing_seq")
        if channel_i is None:
            notes.append("missing_channel")
        if frequency_hz is None:
            notes.append("missing_frequency")
        if normal_len is None:
            notes.append("missing_normal_pdu_len")
        if tail_len is None:
            notes.append("missing_air_extra_len")

        predicted = int(round((slope * seq) + intercept)) if seq is not None else ""
        extraction: dict[str, Any] = {}
        if not notes:
            try:
                expected_frame = phantom_frame_bytes(seq, tx["payload"])
            except ValueError as exc:
                notes.append(f"invalid_expected_frame:{exc}")
                expected_frame = b""
            if expected_frame and len(expected_frame) != tail_len:
                notes.append(f"tail_len_mismatch:expected_frame={len(expected_frame)}")
            if expected_frame:
                extraction = extract_known_pc_frame_near_predicted_start(
                    iq_path=iq_path,
                    sample_rate_hz=sample_rate_hz,
                    center_frequency_hz=center_frequency_hz,
                    predicted_start_sample=int(predicted),
                    packet_frequency_hz=float(frequency_hz),
                    access_address_text=tx["parser_access_address"],
                    channel=channel_i,
                    normal_len=normal_len,
                    expected_frame=expected_frame,
                    lowpass_hz=lowpass_hz,
                    search_radius_us=effective_search_radius_us,
                )
            if extraction.get("notes"):
                notes.append(str(extraction["notes"]))

        marker_hex = tx.get("marker", "")
        payload = str(extraction.get("predicted_extracted_payload_hex") or "")
        data_hex = matcher.payload_data_hex(payload, marker_hex) if payload else ""
        exact = bool(
            payload
            and payload == tx["payload"]
            and matcher.normalize_seq(extraction.get("predicted_extracted_seq")) == matcher.normalize_seq(tx["seq"])
            and extraction.get("predicted_extracted_integrity_ok") == "1"
        )
        data_exact = bool(
            data_hex
            and data_hex == tx.get("data_payload", "")
            and matcher.normalize_seq(extraction.get("predicted_extracted_seq")) == matcher.normalize_seq(tx["seq"])
            and extraction.get("predicted_extracted_integrity_ok") == "1"
        )
        rows.append(
            {
                "run_id": tx.get("run_id") or metadata["run_id"],
                "seq": tx.get("seq", ""),
                "channel": tx.get("channel", ""),
                "tx_frequency_hz": frequency_hz or "",
                "tx_in_analysis_band": "" if in_band is None else int(in_band),
                "access_address": tx.get("access_address", ""),
                "parser_access_address": tx.get("parser_access_address", ""),
                "normal_pdu_len": tx.get("normal_pdu_len", ""),
                "air_extra_len": tx.get("air_extra_len", ""),
                "tx_covert_hex": tx.get("payload", ""),
                "tx_covert_data_hex": tx.get("data_payload", ""),
                "predicted_wideband_sample_index": predicted,
                "search_radius_us": effective_search_radius_us,
                "extracted_post_crc_hex": extraction.get("predicted_extracted_post_crc_hex", ""),
                "extracted_frame_hex": extraction.get("predicted_extracted_frame_hex", ""),
                "extracted_seq": extraction.get("predicted_extracted_seq", ""),
                "extracted_payload_hex": payload,
                "extracted_payload_marker_ok": matcher.payload_marker_ok(payload, marker_hex) if payload else "",
                "extracted_data_hex": data_hex,
                "extracted_integrity_ok": extraction.get("predicted_extracted_integrity_ok", ""),
                "extracted_exact_match": int(exact),
                "extracted_data_exact_match": int(data_exact),
                "known_bit_errors": extraction.get("predicted_known_bit_errors", ""),
                "known_bit_error_rate": extraction.get("predicted_known_bit_error_rate", ""),
                "sync_tail_score_errors": extraction.get("predicted_sync_tail_score_errors", ""),
                "sync_tail_score_ber": extraction.get("predicted_sync_tail_score_ber", ""),
                "polarity": extraction.get("predicted_polarity", ""),
                "threshold": extraction.get("predicted_threshold", ""),
                "bit_start_sample": extraction.get("predicted_bit_start_sample", ""),
                "samples_per_bit": extraction.get("predicted_samples_per_bit", ""),
                "notes": ";".join(notes),
            }
        )

    total = len(rows)
    exact_rows = [row for row in rows if str(row["extracted_exact_match"]) == "1"]
    data_exact_rows = [row for row in rows if str(row["extracted_data_exact_match"]) == "1"]
    tx_bytes = sum(int_or_none(row["air_extra_len"]) or 0 for row in rows)
    tx_payload_bytes = sum(int_or_none(row["normal_pdu_len"]) or 0 for row in rows)
    tx_data_bytes = sum((len(row["tx_covert_data_hex"]) // 2) for row in rows)
    recovered_payload_bytes = sum((len(row["extracted_payload_hex"]) // 2) for row in exact_rows)
    recovered_data_bytes = sum((len(row["extracted_data_hex"]) // 2) for row in data_exact_rows)
    capture_duration_s = matcher.capture_duration_seconds(metadata)
    active_duration_s, active_interval_s, active_window_method = active_tx_window_from_sparse_seq_timestamps(tx_rows)
    rate_duration_s = active_duration_s if active_duration_s > 0 else capture_duration_s
    summary = {
        "schema_version": 1,
        "run_id": metadata["run_id"],
        "mode": "rtt_guided_direct_iq",
        "anchor_file": str(anchor_file),
        "timing_anchor_count": len(anchors),
        "seq_to_sample_slope": slope,
        "seq_to_sample_intercept": intercept,
        "sample_rate_hz": sample_rate_hz,
        "center_frequency_hz": center_frequency_hz,
        "analysis_bandwidth_hz": metadata.get("analysis_bandwidth_hz", ""),
        "search_radius_us": effective_search_radius_us,
        "in_band_packets_attempted": total,
        "exact_packets": len(exact_rows),
        "exact_packet_recovery_rate": len(exact_rows) / total if total else 0.0,
        "data_exact_packets": len(data_exact_rows),
        "data_exact_packet_recovery_rate": len(data_exact_rows) / total if total else 0.0,
        "tx_post_crc_bytes_in_attempted_packets": tx_bytes,
        "tx_normal_payload_bytes_in_attempted_packets": tx_payload_bytes,
        "tx_covert_data_bytes_in_attempted_packets": tx_data_bytes,
        "recovered_payload_bytes": recovered_payload_bytes,
        "recovered_data_bytes": recovered_data_bytes,
        "capture_duration_s": capture_duration_s,
        "active_tx_duration_s": active_duration_s,
        "active_tx_interval_s": active_interval_s,
        "active_tx_window_method": active_window_method,
        "direct_iq_theoretical_bps_in_analysis_band": (
            tx_bytes * 8.0 / active_duration_s
        ) if active_duration_s > 0 else 0.0,
        "direct_iq_theoretical_data_bps_in_analysis_band": (
            tx_data_bytes * 8.0 / active_duration_s
        ) if active_duration_s > 0 else 0.0,
        "direct_iq_recovered_bps_in_analysis_band": (
            recovered_payload_bytes * 8.0 / rate_duration_s
        ) if rate_duration_s > 0 else 0.0,
        "direct_iq_recovered_data_bps_in_analysis_band": (
            recovered_data_bytes * 8.0 / rate_duration_s
        ) if rate_duration_s > 0 else 0.0,
        "direct_iq_capture_window_recovered_data_bps_in_analysis_band": (
            recovered_data_bytes * 8.0 / capture_duration_s
        ) if capture_duration_s > 0 else 0.0,
        "valid": bool(total and anchors),
        "notes": [
            "Per-packet extraction is guided by RTT LL rows, not BLE parser candidates.",
            "SDR parser exact matches are used only to calibrate the seq-to-IQ-sample time model.",
            "The extractor scores IQ directly with the RTT access address plus the expected post-CRC PC frame.",
        ],
    }
    return rows, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--distance-m", default="")
    parser.add_argument("--analysis-bandwidth-hz", default="")
    parser.add_argument("--anchors", type=Path, default=None, help="CSV with exact SDR matches for timing calibration.")
    parser.add_argument("--search-radius-us", type=float, default=None)
    parser.add_argument("--lowpass-hz", type=float, default=900_000.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_root = args.run_root.resolve()
    output_dir = args.output_dir or (run_root / "results" / "rtt_guided_iq")
    rows, summary = extract_rows(
        run_root=run_root,
        output_dir=output_dir,
        run_id=args.run_id,
        distance_m=args.distance_m,
        analysis_bandwidth_hz=args.analysis_bandwidth_hz,
        search_radius_us=args.search_radius_us,
        lowpass_hz=args.lowpass_hz,
        anchors_path=args.anchors.resolve() if args.anchors else None,
    )
    write_csv(output_dir / "rtt_guided_iq_extractions.csv", rows, OUTPUT_FIELDS)
    write_json(output_dir / "rtt_guided_iq_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
