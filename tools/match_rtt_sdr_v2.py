#!/usr/bin/env python3
"""Time-gated, one-to-one RTT/S​​DR matching for PhantomChannel investigation.

This is an investigation-only matcher.  It does not replace
``match_rtt_sdr_results.py`` and it never writes into BLE_encrypt_check.
The matcher uses the frozen run's existing exact rows only to estimate the
RTT-sequence to IQ-sample affine map.  The output labels this as a diagnostic
alignment source; it is not a blind-recovery result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import match_rtt_sdr_results as legacy  # noqa: E402


DEFAULT_TIME_GATE_US = 0.0
DEFAULT_DEDUPE_SAMPLES = 2_000


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str] | None = None) -> None:
    materialized = list(rows)
    if fields is None:
        fields = []
        for row in materialized:
            for key in row:
                if key not in fields:
                    fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in materialized)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _float(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _clean_hex(value: Any) -> str:
    try:
        return legacy.normalize_hex(value)
    except (TypeError, ValueError):
        return ""


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(math.ceil(fraction * len(ordered))) - 1))
    return float(ordered[index])


def _fit_affine(points: list[tuple[float, float]]) -> dict[str, Any]:
    """Fit sample ~= slope * seq + intercept using a median pairwise slope."""

    by_x: dict[float, list[float]] = defaultdict(list)
    for x, y in points:
        by_x[x].append(y)
    reduced = sorted((x, statistics.median(ys)) for x, ys in by_x.items())
    slopes: list[float] = []
    for index, (left_x, left_y) in enumerate(reduced):
        for right_x, right_y in reduced[index + 1 :]:
            if right_x > left_x:
                slopes.append((right_y - left_y) / (right_x - left_x))
    if not slopes:
        raise ValueError("at least two distinct exact seq anchors are required")
    slope = statistics.median(slopes)
    intercept = statistics.median(y - slope * x for x, y in reduced)
    residuals = [y - (slope * x + intercept) for x, y in reduced]
    absolute = [abs(value) for value in residuals]
    p95 = _percentile(absolute, 0.95)
    median_abs = statistics.median(absolute)
    return {
        "source": "legacy_exact_payload_anchors",
        "anchor_count": len(points),
        "unique_seq_anchor_count": len(reduced),
        "slope_samples_per_seq": slope,
        "intercept_samples": intercept,
        "residual_median_samples": median_abs,
        "residual_p95_samples": p95,
        "residual_max_samples": max(absolute) if absolute else 0.0,
    }


def _attempt_rows(
    ground_truth_rows: list[dict[str, str]],
    ll_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    joined = legacy.joined_rtt_tx_rows(ground_truth_rows, ll_rows)
    seen_by_seq: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(joined):
        seq = str(row.get("seq", ""))
        retry_ordinal = seen_by_seq[seq]
        seen_by_seq[seq] += 1
        rows.append(
            {
                **row,
                "attempt_index": index,
                "tx_attempt_id": f"{index}:{seq}:{retry_ordinal}",
                "unique_notification_id": seq,
                "retry_ordinal": retry_ordinal,
                "timestamp_source": "observed" if _float(row.get("timestamp_us")) is not None else "interpolated",
            }
        )
    return rows


def _old_exact_anchors(old_match_rows: list[dict[str, str]]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for row in old_match_rows:
        if str(row.get("covert_exact_match", "0")) != "1":
            continue
        seq = _float(row.get("seq"))
        sample = _float(row.get("sdr_wideband_sample_index"))
        if seq is not None and sample is not None:
            points.append((seq, sample))
    return points


def _candidate_views(sdr_rows: list[dict[str, str]], target_aa: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    target_aa = _clean_hex(target_aa)
    for index, raw in enumerate(sdr_rows):
        view = legacy.sdr_phantom_view(raw, index)
        if view["access_address"] != target_aa:
            continue
        sample = _float(raw.get("wideband_sample_index"))
        if sample is None:
            continue
        post_crc_hex = _clean_hex(raw.get("post_crc_hex"))
        frame_hex = view.get("frame_hex", "")
        segment_start = ""
        score = ""
        raw_offset = str(raw.get("raw_offset_info", ""))
        for item in raw_offset.split(";"):
            if item.startswith("segment_start_sample="):
                segment_start = item.split("=", 1)[1]
            elif item.startswith("score="):
                score = item.split("=", 1)[1]
        candidates.append(
            {
                "observation_index": index,
                "candidate_id": f"parser:{index}",
                "source": "parser",
                "raw": raw,
                "access_address": view.get("access_address", ""),
                "channel": str(view.get("channel", "")),
                "sample": sample,
                "timestamp_us": _float(raw.get("timestamp_us")),
                "packet_type": raw.get("packet_type", ""),
                "segment_start_sample": segment_start,
                "confidence_score": raw.get("confidence_score", ""),
                "parser_score": score,
                "post_crc_len": len(post_crc_hex) // 2,
                "post_crc_hex": post_crc_hex,
                "frame_hex": frame_hex,
                "frame_seq": str(view.get("seq", "")) if view.get("payload") else "",
                "payload_hex": str(view.get("payload", "")),
                "frame_integrity_ok": str(view.get("integrity_ok", "")),
                "frame_len_bytes": _int(view.get("len")) or "",
                "crc_capture_status": raw.get("crc_capture_status", ""),
                "timestamp_status": raw.get("timestamp_status", ""),
            }
        )
    return candidates


def _assign_burst_ids(candidates: list[dict[str, Any]], dedupe_samples: int) -> None:
    by_channel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_channel[candidate["channel"]].append(candidate)
    burst_counter = 0
    for channel, channel_rows in by_channel.items():
        channel_rows.sort(key=lambda row: (row["sample"], row["observation_index"]))
        previous_sample: float | None = None
        current_id = ""
        for row in channel_rows:
            if previous_sample is None or row["sample"] - previous_sample > dedupe_samples:
                current_id = f"burst:{burst_counter}"
                burst_counter += 1
            row["burst_id"] = current_id
            previous_sample = row["sample"]


def _candidate_quality(tx: dict[str, Any], candidate: dict[str, Any]) -> tuple[int, str]:
    frame_seq = candidate.get("frame_seq", "")
    tx_seq = str(tx.get("seq", ""))
    payload = candidate.get("payload_hex", "")
    tx_payload = _clean_hex(tx.get("payload", ""))
    integrity = candidate.get("frame_integrity_ok", "") == "1"
    if frame_seq:
        if frame_seq != tx_seq:
            return 0, "frame_seq_mismatch"
        if integrity and payload == tx_payload:
            return 50, "exact_payload"
        if integrity:
            return 40, "valid_frame_nonexact"
        return 30, "frame_seq_invalid_integrity"
    if candidate.get("post_crc_len", 0) > 0:
        return 20, "tail_without_valid_frame"
    return 10, "standard_pdu_only"


def _edge(tx: dict[str, Any], candidate: dict[str, Any], gate_samples: float) -> tuple[int, str] | None:
    if str(tx.get("channel", "")) != str(candidate.get("channel", "")):
        return None
    tx_aa = _clean_hex(tx.get("parser_access_address"))
    if candidate.get("access_address") != tx_aa:
        return None
    predicted = tx.get("predicted_sample")
    sample = candidate.get("sample")
    if predicted is None or sample is None or abs(sample - predicted) > gate_samples:
        return None
    frame_seq = candidate.get("frame_seq", "")
    if frame_seq and frame_seq != str(tx.get("seq", "")):
        return None
    return _candidate_quality(tx, candidate)


def _global_monotonic_assignment(
    tx_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    gate_samples: float,
) -> dict[int, tuple[dict[str, Any], int, str]]:
    """Maximize quality under monotonic one-to-one assignment.

    The score hierarchy dominates timing: a valid Phantom frame is always
    preferred over a standard-only observation, while timing breaks ties.
    """

    tx = sorted(tx_rows, key=lambda row: (row["predicted_sample"], row["attempt_index"]))
    obs = sorted(candidates, key=lambda row: (row["sample"], row["observation_index"]))
    n, m = len(tx), len(obs)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    edges: dict[tuple[int, int], tuple[int, str]] = {}
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            best = max(dp[i + 1][j], dp[i][j + 1])
            edge = _edge(tx[i], obs[j], gate_samples)
            if edge is not None:
                quality, reason = edge
                timing_bonus = max(0.0, gate_samples - abs(obs[j]["sample"] - tx[i]["predicted_sample"]))
                score = int(quality * 1_000_000_000 + timing_bonus)
                edges[(i, j)] = (score, reason)
                best = max(best, dp[i + 1][j + 1] + score)
            dp[i][j] = best

    assignment: dict[int, tuple[dict[str, Any], int, str]] = {}
    i = j = 0
    while i < n and j < m:
        edge = edges.get((i, j))
        if edge is not None and dp[i][j] == dp[i + 1][j + 1] + edge[0]:
            assignment[tx[i]["attempt_index"]] = (obs[j], edge[0] // 1_000_000_000, edge[1])
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return assignment


def _failure_stage(
    tx: dict[str, Any],
    assignment: tuple[dict[str, Any], int, str] | None,
    candidates: list[dict[str, Any]],
    gate_samples: float,
) -> tuple[str, str]:
    if not tx["in_iq_window"]:
        return "F0_WINDOW_OR_GT_AMBIGUOUS", "predicted_sample_outside_iq_window"
    if assignment is not None:
        candidate, _quality, reason = assignment
        if reason == "exact_payload":
            return "F7_EXACT_RECOVERED", reason
        if reason == "standard_pdu_only":
            return "F6_VALID_NOT_ASSIGNED", reason
        return "F5_FRAME_INVALID", reason

    nearby = []
    for candidate in candidates:
        if str(candidate.get("channel", "")) != str(tx.get("channel", "")):
            continue
        if abs(candidate["sample"] - tx["predicted_sample"]) <= gate_samples:
            nearby.append(candidate)
    if any(candidate.get("frame_seq") for candidate in nearby):
        return "F5_FRAME_INVALID", "nearby_phantom_frame_not_accepted"
    if nearby and any(candidate.get("post_crc_len", 0) > 0 for candidate in nearby):
        return "F3_SEGMENT_TRUNCATED", "nearby_tail_bytes_without_accepted_frame"
    if nearby:
        return "F3_SEGMENT_TRUNCATED", "nearby_standard_parser_row_only"
    return "F1_NO_PHYSICAL_BURST", "no_target_parser_row_in_time_gate"


def analyze_run(
    run_root: Path,
    *,
    time_gate_us: float = DEFAULT_TIME_GATE_US,
    dedupe_samples: int = DEFAULT_DEDUPE_SAMPLES,
    sdr_csv: Path | None = None,
    old_matches_csv: Path | None = None,
) -> dict[str, Any]:
    ground_truth = read_csv(run_root / "ground_truth" / "rtt_ground_truth.csv")
    ll_rows = read_csv(run_root / "ground_truth" / "rtt_ll_tx.csv")
    sdr_rows = read_csv(sdr_csv or (run_root / "sdr" / "ble_packets.csv"))
    metadata = json.loads((run_root / "iq" / "metadata.json").read_text(encoding="utf-8"))
    old_matches = read_csv(old_matches_csv or (run_root / "results" / "rtt_sdr_matches.csv"))

    attempts = _attempt_rows(ground_truth, ll_rows)
    anchors = _old_exact_anchors(old_matches)
    alignment = _fit_affine(anchors)
    sample_rate = float(metadata.get("actual_sample_rate_sps") or metadata.get("sample_rate_sps"))
    total_samples = int(metadata.get("samples", 0))
    if time_gate_us > 0:
        gate_samples = time_gate_us * sample_rate / 1_000_000.0
        gate_source = "cli"
    else:
        # Preserve every observed exact anchor, including the tail of the
        # fitted residual distribution.  The extra margin is finite and is
        # reported in the manifest; it is not an unlimited AA/channel gate.
        gate_samples = max(
            sample_rate * 250.0 / 1_000_000.0,
            alignment["residual_p95_samples"] * 1.25 + sample_rate * 200.0 / 1_000_000.0,
            alignment["residual_max_samples"] * 1.05 + sample_rate * 200.0 / 1_000_000.0,
        )
        gate_source = "anchor_residual_auto"
    alignment["time_gate_samples"] = gate_samples
    alignment["time_gate_us"] = gate_samples * 1_000_000.0 / sample_rate
    alignment["time_gate_source"] = gate_source

    for attempt in attempts:
        seq = _float(attempt.get("seq"))
        predicted = alignment["slope_samples_per_seq"] * seq + alignment["intercept_samples"] if seq is not None else None
        attempt["predicted_sample"] = predicted
        attempt["predicted_timestamp_us"] = predicted * 1_000_000.0 / sample_rate if predicted is not None else ""
        attempt["in_iq_window"] = bool(predicted is not None and 0 <= predicted < total_samples)

    target_aa = str(attempts[0].get("parser_access_address", "")) if attempts else ""
    candidates = _candidate_views(sdr_rows, target_aa)
    _assign_burst_ids(candidates, dedupe_samples)
    tx_in_window = [attempt for attempt in attempts if attempt["in_iq_window"]]
    all_channels = {str(row.get("channel", "")) for row in tx_in_window}
    candidate_window = [
        candidate
        for candidate in candidates
        if candidate["channel"] in all_channels
        and any(
            abs(candidate["sample"] - row["predicted_sample"]) <= gate_samples
            and candidate["channel"] == str(row.get("channel", ""))
            for row in tx_in_window
        )
    ]
    assignments = _global_monotonic_assignment(tx_in_window, candidate_window, gate_samples)

    for attempt in attempts:
        match = assignments.get(attempt["attempt_index"])
        if match is None:
            attempt["observation_index"] = ""
            attempt["burst_id"] = ""
            attempt["match_quality"] = ""
            attempt["match_reason"] = ""
            attempt["matched_sample"] = ""
            attempt["time_residual_samples"] = ""
        else:
            candidate, quality, reason = match
            attempt["observation_index"] = candidate["observation_index"]
            attempt["burst_id"] = candidate.get("burst_id", "")
            attempt["match_quality"] = quality
            attempt["match_reason"] = reason
            attempt["matched_sample"] = candidate["sample"]
            attempt["time_residual_samples"] = candidate["sample"] - attempt["predicted_sample"]
        stage, failure_reason = _failure_stage(attempt, match, candidate_window, gate_samples)
        attempt["failure_stage"] = stage
        attempt["failure_reason"] = failure_reason
        attempt["exact_match"] = int(stage == "F7_EXACT_RECOVERED")
        attempt["matched_frame_integrity_ok"] = (
            match[0].get("frame_integrity_ok", "") if match is not None else ""
        )
        attempt["matched_payload_hex"] = match[0].get("payload_hex", "") if match is not None else ""
        attempt["matched_post_crc_len"] = match[0].get("post_crc_len", "") if match is not None else ""

    summary = {
        "schema_version": 1,
        "run_root": str(run_root),
        "tx_attempts": len(attempts),
        "tx_attempts_in_iq_window": sum(int(row["in_iq_window"]) for row in attempts),
        "candidate_rows_target_aa": len(candidates),
        "candidate_rows_in_time_gate": len(candidate_window),
        "matched_attempts": sum(1 for row in attempts if row["observation_index"] != ""),
        "exact_attempts": sum(int(row["exact_match"]) for row in attempts),
        "unique_notifications_in_iq_window": len({row["unique_notification_id"] for row in attempts if row["in_iq_window"]}),
        "unique_exact_notifications_in_iq_window": len({row["unique_notification_id"] for row in attempts if row["in_iq_window"] and row["exact_match"]}),
        "failure_stage_counts_all": dict(sorted(Counter(row["failure_stage"] for row in attempts).items())),
        "failure_stage_counts_in_iq_window": dict(sorted(Counter(row["failure_stage"] for row in attempts if row["in_iq_window"]).items())),
        "alignment": alignment,
        "candidate_dedupe_samples": dedupe_samples,
        "target_parser_access_address": target_aa,
        "sample_rate_sps": sample_rate,
        "iq_samples": total_samples,
        "diagnostic_only": True,
        "blind_recovery_claim_allowed": False,
    }
    return {
        "attempts": attempts,
        "candidates": candidates,
        "candidate_window": candidate_window,
        "summary": summary,
    }


ATTEMPT_FIELDS = [
    "attempt_index", "tx_attempt_id", "unique_notification_id", "retry_ordinal", "seq",
    "channel", "parser_access_address", "access_address", "rtt_timestamp_us", "timestamp_source",
    "predicted_sample", "predicted_timestamp_us", "in_iq_window", "observation_index", "burst_id",
    "matched_sample", "time_residual_samples", "match_quality", "match_reason", "matched_frame_integrity_ok",
    "matched_post_crc_len", "matched_payload_hex", "exact_match", "failure_stage", "failure_reason",
]

CANDIDATE_FIELDS = [
    "candidate_id", "observation_index", "burst_id", "source", "access_address", "channel", "sample",
    "timestamp_us", "packet_type", "segment_start_sample", "confidence_score", "parser_score", "post_crc_len",
    "frame_hex", "frame_seq", "payload_hex", "frame_integrity_ok", "frame_len_bytes", "crc_capture_status",
    "timestamp_status",
]


def write_analysis(output_dir: Path, analysis: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "tx_timeline.csv", analysis["attempts"], ATTEMPT_FIELDS)
    write_csv(output_dir / "candidate_ledger.csv", analysis["candidates"], CANDIDATE_FIELDS)
    write_csv(
        output_dir / "packet_funnel.csv",
        [row for row in analysis["attempts"] if row["in_iq_window"]],
        ATTEMPT_FIELDS,
    )
    write_json(output_dir / "failure_summary.json", analysis["summary"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-gate-us", type=float, default=DEFAULT_TIME_GATE_US)
    parser.add_argument("--dedupe-samples", type=int, default=DEFAULT_DEDUPE_SAMPLES)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {args.output_dir}")
    analysis = analyze_run(
        args.run_root.resolve(),
        time_gate_us=args.time_gate_us,
        dedupe_samples=args.dedupe_samples,
    )
    write_analysis(args.output_dir.resolve(), analysis)
    print(json.dumps(analysis["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
