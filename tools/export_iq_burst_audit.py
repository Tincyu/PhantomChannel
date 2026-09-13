#!/usr/bin/env python3
"""Export one-row-per-IQ-window-burst audit CSV for manual inspection.

The audit joins RTT LL ground truth with the full C++ parser output, keeps the
same time-alignment rules as the scorer, and exports every covert burst mapped
into the IQ window.  It is intended to answer whether a non-exact burst was
not captured at all, was seen only as a standard PDU, or was decoded with a
wrong Phantom post-CRC payload.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

# Allow direct execution from the project root without requiring PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import match_rtt_sdr_results as scorer


DEFAULT_FIELDS = [
    "burst_index",
    "seq",
    "in_iq_window",
    "classification",
    "tx_rtt_timestamp_us",
    "mapped_iq_timestamp_us",
    "expected_wideband_sample_index",
    "channel",
    "tx_frequency_hz",
    "access_address_hex",
    "parser_access_address_hex",
    "crc_init_hex",
    "normal_pdu_len",
    "air_extra_len",
    "covert_len_bytes",
    "covert_marker_hex",
    "tx_covert_hex",
    "tx_covert_data_hex",
    "match_method",
    "covert_exact_match",
    "covert_data_exact_match",
    "covert_integrity_ok",
    "rx_payload_marker_ok",
    "bit_errors",
    "bit_error_rate",
    "matched_sample_index",
    "matched_timestamp_us",
    "matched_sample_error_samples",
    "matched_channel",
    "matched_access_address_hex",
    "matched_dewhitened_pdu_hex",
    "matched_captured_crc_hex",
    "matched_crc_capture_status",
    "matched_rx_covert_hex",
    "matched_rx_covert_data_hex",
    "matched_parser_ble_pdu_type",
    "matched_parser_whitened_pdu_hex",
    "matched_parser_dewhitened_pdu_hex",
    "matched_parser_captured_crc_hex",
    "matched_parser_post_crc_hex",
    "matched_parser_crc_and_post_crc_hex",
    "matched_parser_payload_len",
    "matched_parser_raw_offset_info",
    "matched_parser_confidence_score",
    "matched_parser_rssi",
    "matched_parser_cfo_hz",
    "matched_parser_crc_capture_status",
    "nearest_same_aa_channel_count",
    "nearest_candidate_count_in_tolerance",
    "nearest_candidate_sample_index",
    "nearest_candidate_timestamp_us",
    "nearest_candidate_sample_error_samples",
    "nearest_candidate_dewhitened_pdu_hex",
    "nearest_candidate_captured_crc_hex",
    "nearest_candidate_post_crc_hex",
    "nearest_candidate_crc_and_post_crc_hex",
    "nearest_candidate_payload_len",
    "nearest_candidate_confidence_score",
    "nearest_candidate_rssi",
    "nearest_candidate_cfo_hz",
    "nearest_candidate_raw_offset_info",
    "notes",
]


def as01(value: Any) -> str:
    return "1" if str(value).strip().lower() in {"1", "true", "yes"} else "0"


def number_text(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(number)) if number.is_integer() else str(number)


def sample_index(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def parser_aa(row: dict[str, Any]) -> str:
    return scorer.normalize_aa(row.get("access_address", ""))


def parser_channel(row: dict[str, Any]) -> str:
    return scorer.normalize_seq(row.get("channel", ""))


def parser_raw_fields(row: dict[str, Any] | None, prefix: str) -> dict[str, str]:
    if not row:
        return {f"{prefix}{name}": "" for name in (
            "ble_pdu_type", "whitened_pdu_hex", "dewhitened_pdu_hex",
            "captured_crc_hex", "post_crc_hex", "crc_and_post_crc_hex",
            "payload_len", "raw_offset_info", "confidence_score", "rssi",
            "cfo_hz", "crc_capture_status",
        )}
    return {
        f"{prefix}ble_pdu_type": str(row.get("ble_pdu_type", "")),
        f"{prefix}whitened_pdu_hex": str(row.get("whitened_pdu_hex", "")),
        f"{prefix}dewhitened_pdu_hex": str(row.get("dewhitened_pdu_hex", "")),
        f"{prefix}captured_crc_hex": str(row.get("captured_crc_hex", "")),
        f"{prefix}post_crc_hex": str(row.get("post_crc_hex", "")),
        f"{prefix}crc_and_post_crc_hex": str(row.get("crc_and_post_crc_hex", "")),
        f"{prefix}payload_len": str(row.get("payload_len", "")),
        f"{prefix}raw_offset_info": str(row.get("raw_offset_info", "")),
        f"{prefix}confidence_score": str(row.get("confidence_score", "")),
        f"{prefix}rssi": str(row.get("rssi", "")),
        f"{prefix}cfo_hz": str(row.get("cfo_hz", "")),
        f"{prefix}crc_capture_status": str(row.get("crc_capture_status", "")),
    }


def time_alignment(
    tx_rows: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], list[float | None], float, float, float]:
    alignment = scorer.infer_rtt_to_sdr_time_alignment(tx_rows, matches)
    timestamps_us, _ = scorer.interpolate_tx_timestamps_us(tx_rows)
    sample_rate = float(metadata["actual_sample_rate_sps"])
    origins: list[float] = []
    for row in matches:
        sdr_timestamp = scorer.float_or_none(row.get("sdr_timestamp_us"))
        wideband_index = scorer.float_or_none(row.get("sdr_wideband_sample_index"))
        if sdr_timestamp is not None and wideband_index is not None:
            origins.append(sdr_timestamp - wideband_index * 1_000_000.0 / sample_rate)
    clock_origin_us = float(statistics.median(origins)) if origins else 0.0
    duration_us = float(metadata["samples"]) / sample_rate * 1_000_000.0
    return alignment, timestamps_us, clock_origin_us, duration_us, sample_rate


def export_audit(
    run_root: Path,
    parser_csv: Path,
    output_csv: Path,
    candidate_tolerance_us: float,
) -> dict[str, Any]:
    metadata = json.loads((run_root / "iq" / "metadata.json").read_text(encoding="utf-8"))
    metadata.update({
        "run_id": run_root.name,
        "distance_m": "0.5",
        "analysis_bandwidth_hz": "80000000",
    })
    ground_truth = scorer.read_csv(run_root / "ground_truth" / "rtt_ground_truth.csv")
    ll_tx = scorer.read_csv(run_root / "ground_truth" / "rtt_ll_tx.csv")
    parser_rows = scorer.read_csv(parser_csv)
    tx_rows = scorer.joined_rtt_tx_rows(ground_truth, ll_tx)
    matches, _, _, score_summary = scorer.score_phantom_run(
        ground_truth, ll_tx, parser_rows, metadata
    )
    alignment, timestamps_us, clock_origin_us, duration_us, sample_rate = time_alignment(
        tx_rows, matches, metadata
    )
    if not alignment.get("available"):
        raise RuntimeError("RTT-to-IQ time alignment is unavailable; need at least four exact anchors")

    match_by_seq = {str(row.get("seq", "")): row for row in matches}
    raw_by_wideband: dict[int, dict[str, Any]] = {}
    raw_by_aa_channel: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in parser_rows:
        wideband = sample_index(row.get("wideband_sample_index"))
        if wideband is not None:
            raw_by_wideband[wideband] = row
        key = (parser_aa(row), parser_channel(row))
        raw_by_aa_channel.setdefault(key, []).append(row)
    for rows in raw_by_aa_channel.values():
        rows.sort(key=lambda row: sample_index(row.get("wideband_sample_index")) or 0)

    tolerance_samples = candidate_tolerance_us * sample_rate / 1_000_000.0
    output_rows: list[dict[str, str]] = []
    missing_sequences: list[str] = []
    burst_index = 0
    for tx, rtt_timestamp_us in zip(tx_rows, timestamps_us):
        if not tx.get("payload") or rtt_timestamp_us is None:
            continue
        mapped_iq_us = float(alignment["slope"]) * rtt_timestamp_us + float(alignment["offset_us"])
        in_window = clock_origin_us <= mapped_iq_us <= clock_origin_us + duration_us
        if not in_window:
            continue

        seq = str(tx.get("seq", ""))
        match = match_by_seq.get(seq, {})
        exact = as01(match.get("covert_data_exact_match", match.get("covert_exact_match", "0")))
        if exact == "1":
            classification = "exact_recovery"
        elif match.get("match_method") == "aa_channel_covert_candidate" and match.get("rx_covert_hex"):
            classification = "parser_candidate_payload_mismatch"
            missing_sequences.append(seq)
        elif match.get("match_method") == "aa_channel_standard_pdu_only":
            classification = "parser_standard_only_no_phantom_tail"
            missing_sequences.append(seq)
        elif match:
            classification = "parser_candidate_without_exact_payload"
            missing_sequences.append(seq)
        else:
            classification = "no_matching_parser_burst"
            missing_sequences.append(seq)

        expected_sample = int(round((mapped_iq_us - clock_origin_us) * sample_rate / 1_000_000.0))
        tx_aa = str(tx.get("parser_access_address") or tx.get("access_address") or "").lower()
        tx_channel = str(tx.get("channel", ""))
        candidates = [
            row for row in raw_by_aa_channel.get((tx_aa.lower(), tx_channel), [])
            if sample_index(row.get("wideband_sample_index")) is not None
        ]
        candidates_with_error = sorted(
            candidates,
            key=lambda row: abs((sample_index(row.get("wideband_sample_index")) or 0) - expected_sample),
        )
        nearby = [
            row for row in candidates_with_error
            if abs((sample_index(row.get("wideband_sample_index")) or 0) - expected_sample) <= tolerance_samples
        ]
        nearest = candidates_with_error[0] if candidates_with_error else None

        matched_sample = sample_index(match.get("sdr_wideband_sample_index"))
        matched_raw = raw_by_wideband.get(matched_sample) if matched_sample is not None else None
        nearest_sample = sample_index(nearest.get("wideband_sample_index")) if nearest else None
        matched_error = matched_sample - expected_sample if matched_sample is not None else None
        nearest_error = nearest_sample - expected_sample if nearest_sample is not None else None
        tx_frequency, _ = scorer.channel_in_analysis_band(tx_channel, metadata)

        row = {
            "burst_index": str(burst_index),
            "seq": seq,
            "in_iq_window": "1",
            "classification": classification,
            "tx_rtt_timestamp_us": number_text(tx.get("timestamp_us")),
            "mapped_iq_timestamp_us": f"{mapped_iq_us - clock_origin_us:.3f}",
            "expected_wideband_sample_index": str(expected_sample),
            "channel": tx_channel,
            "tx_frequency_hz": str(tx_frequency or ""),
            "access_address_hex": tx.get("access_address", ""),
            "parser_access_address_hex": tx.get("parser_access_address", ""),
            "crc_init_hex": tx.get("crc_init", ""),
            "normal_pdu_len": str(tx.get("normal_pdu_len", "")),
            "air_extra_len": str(tx.get("air_extra_len", "")),
            "covert_len_bytes": str(tx.get("len", "")),
            "covert_marker_hex": tx.get("marker", ""),
            "tx_covert_hex": tx.get("payload", ""),
            "tx_covert_data_hex": tx.get("data_payload", ""),
            "match_method": str(match.get("match_method", "")),
            "covert_exact_match": str(match.get("covert_exact_match", "")),
            "covert_data_exact_match": str(match.get("covert_data_exact_match", "")),
            "covert_integrity_ok": str(match.get("covert_integrity_ok", "")),
            "rx_payload_marker_ok": str(match.get("rx_payload_marker_ok", "")),
            "bit_errors": str(match.get("bit_errors", "")),
            "bit_error_rate": str(match.get("bit_error_rate", "")),
            "matched_sample_index": str(matched_sample or ""),
            "matched_timestamp_us": str(match.get("sdr_timestamp_us", "")),
            "matched_sample_error_samples": str(matched_error or ""),
            "matched_channel": str(match.get("sdr_channel", "")),
            "matched_access_address_hex": str(match.get("sdr_access_address", "")),
            "matched_dewhitened_pdu_hex": str(match.get("sdr_dewhitened_pdu_hex", "")),
            "matched_captured_crc_hex": str(match.get("sdr_captured_crc_hex", "")),
            "matched_crc_capture_status": str(match.get("sdr_crc_capture_status", "")),
            "matched_rx_covert_hex": str(match.get("rx_covert_hex", "")),
            "matched_rx_covert_data_hex": str(match.get("rx_covert_data_hex", "")),
            "nearest_same_aa_channel_count": str(len(candidates)),
            "nearest_candidate_count_in_tolerance": str(len(nearby)),
            "nearest_candidate_sample_index": str(nearest_sample or ""),
            "nearest_candidate_timestamp_us": str(nearest.get("timestamp_us", "")) if nearest else "",
            "nearest_candidate_sample_error_samples": str(nearest_error or ""),
            "nearest_candidate_dewhitened_pdu_hex": str(nearest.get("dewhitened_pdu_hex", "")) if nearest else "",
            "nearest_candidate_captured_crc_hex": str(nearest.get("captured_crc_hex", "")) if nearest else "",
            "nearest_candidate_post_crc_hex": str(nearest.get("post_crc_hex", "")) if nearest else "",
            "nearest_candidate_crc_and_post_crc_hex": str(nearest.get("crc_and_post_crc_hex", "")) if nearest else "",
            "nearest_candidate_payload_len": str(nearest.get("payload_len", "")) if nearest else "",
            "nearest_candidate_confidence_score": str(nearest.get("confidence_score", "")) if nearest else "",
            "nearest_candidate_rssi": str(nearest.get("rssi", "")) if nearest else "",
            "nearest_candidate_cfo_hz": str(nearest.get("cfo_hz", "")) if nearest else "",
            "nearest_candidate_raw_offset_info": str(nearest.get("raw_offset_info", "")) if nearest else "",
            "notes": str(match.get("notes", "")),
        }
        row.update(parser_raw_fields(matched_raw, "matched_parser_"))
        output_rows.append(row)
        burst_index += 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEFAULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "schema_version": 1,
        "run_root": str(run_root),
        "parser_csv": str(parser_csv),
        "output_csv": str(output_csv),
        "candidate_tolerance_us": candidate_tolerance_us,
        "iq_window_burst_count": len(output_rows),
        "exact_burst_count": sum(row["classification"] == "exact_recovery" for row in output_rows),
        "missing_burst_count": len(missing_sequences),
        "missing_sequences": missing_sequences,
        "classification_counts": {
            key: sum(row["classification"] == key for row in output_rows)
            for key in sorted({row["classification"] for row in output_rows})
        },
        "time_alignment": alignment,
        "iq_duration_us": duration_us,
        "score_summary": {
            "tx_payload_packets_in_iq_capture_window": score_summary.get("tx_payload_packets_in_iq_capture_window"),
            "covert_exact_packets_in_iq_capture_window": score_summary.get("covert_exact_packets_in_iq_capture_window"),
        },
    }
    summary_path = output_csv.with_name("iq_burst_audit_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--parser-csv", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--candidate-tolerance-us", type=float, default=3000.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_root = args.run_root.expanduser().resolve()
    parser_csv = args.parser_csv or (run_root / "diagnostics" / "one_stage_cpp" / "ble_packets.csv")
    output_csv = args.output_csv or (
        run_root / "diagnostics" / "one_stage_cpp_score" / "iq_burst_audit.csv"
    )
    if not parser_csv.is_file():
        raise SystemExit(f"parser CSV does not exist: {parser_csv}")
    summary = export_audit(run_root, parser_csv, output_csv, args.candidate_tolerance_us)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
