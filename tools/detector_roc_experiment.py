#!/usr/bin/env python3
"""Compute trace-level detector ROC curves for the Dataset-B HRS traces.

The raw captures are fixed-rate SC16 B210 files.  Each trace is reduced to
three passive-observer feature families:

* post-boundary tail energy;
* burst-duration residual after the decoded standard BLE boundary;
* inter-frame timing irregularity.

The Phantom parser candidate rate is emitted as a diagnostic curve, not as a
replacement for the PHY-aware detectors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def number(value: Any, default: float | None = None) -> float | None:
    try:
        text = str(value).strip()
        if not text:
            return default
        result = float(text)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_db(numerator: float, denominator: float) -> float:
    return float(10.0 * np.log10(max(numerator, 1e-12) / max(denominator, 1e-12)))


def standard_boundary_samples(payload_len: int, sample_rate: float) -> int:
    # 1M PHY: preamble + access address + header + payload + CRC.
    standard_bytes = 1 + 4 + 2 + max(0, payload_len) + 3
    return int(round(standard_bytes * 8.0 * sample_rate / 1_000_000.0))


def event_boundary_features(
    iq: np.memmap,
    rows: list[dict[str, str]],
    sample_rate: float,
    tail_window_us: float,
) -> list[dict[str, float]]:
    power = np.asarray(iq[:, 0], dtype=np.float32) ** 2 + np.asarray(iq[:, 1], dtype=np.float32) ** 2
    tail_samples = int(round(tail_window_us * sample_rate / 1_000_000.0))
    pre_samples = int(round(1_200.0 * sample_rate / 1_000_000.0))
    features: list[dict[str, float]] = []
    for row in rows:
        start = int(number(row.get("sample_index"), -1) or -1)
        payload_len = int(number(row.get("payload_len"), 0) or 0)
        if start < 0:
            continue
        boundary = start + standard_boundary_samples(payload_len, sample_rate)
        if boundary + tail_samples >= len(power) or start <= 0:
            continue
        pre_start = max(0, start - pre_samples)
        pre_end = max(pre_start + 1, start - int(round(200 * sample_rate / 1_000_000.0)))
        noise = float(np.median(power[pre_start:pre_end]))
        tail = power[boundary : boundary + tail_samples]
        if tail.size < max(64, tail_samples // 4):
            continue
        tail_mean = float(np.mean(tail))
        tail_p99 = float(np.percentile(tail, 99.0))
        # The tail window is 2.2 ms by default and the 231 B covert tail is
        # about 1.9 ms at 1M PHY.  Use the final 100 us as a noise reference;
        # taking an index beyond the requested window would silently reduce
        # the reference to one sample and saturate duration at the window end.
        late_samples = max(16, int(round(100 * sample_rate / 1_000_000.0)))
        late_start = max(0, tail.size - late_samples)
        late_noise = float(np.median(tail[late_start:]))
        threshold = max(late_noise, noise, 1e-12) * 8.0
        block = max(8, int(round(16 * sample_rate / 4_000_000.0)))
        usable = (tail.size // block) * block
        block_power = tail[:usable].reshape(-1, block).mean(axis=1) if usable else np.empty(0)
        active = np.flatnonzero(block_power > threshold)
        active_duration_us = float((active[-1] + 1) * block / sample_rate * 1_000_000.0) if active.size else 0.0
        features.append(
            {
                "sample_index": float(start),
                "payload_len": float(payload_len),
                "boundary_sample_index": float(boundary),
                "tail_energy_db": safe_db(tail_mean, noise),
                "tail_peak_db": safe_db(tail_p99, noise),
                "duration_residual_us": active_duration_us,
            }
        )
    return features


def timing_features(rows: list[dict[str, str]]) -> dict[str, float]:
    by_aa: dict[str, list[float]] = {}
    for row in rows:
        aa = str(row.get("access_address", ""))
        timestamp = number(row.get("timestamp_us"))
        if timestamp is not None:
            by_aa.setdefault(aa, []).append(timestamp)
    gaps: list[float] = []
    for timestamps in by_aa.values():
        ordered = sorted(set(timestamps))
        gaps.extend(right - left for left, right in zip(ordered, ordered[1:]) if 100.0 <= right - left <= 100_000.0)
    if not gaps:
        return {"timing_gap_count": 0, "timing_gap_median_us": 0.0, "timing_gap_cv": 0.0, "timing_20ms_residual_us": 0.0}
    values = np.asarray(gaps, dtype=np.float64)
    median = float(np.median(values))
    cv = float(np.std(values) / median) if median > 0 else 0.0
    period = 20_000.0
    residual = np.minimum(np.mod(values, period), period - np.mod(values, period))
    return {
        "timing_gap_count": int(len(gaps)),
        "timing_gap_median_us": median,
        "timing_gap_cv": cv,
        "timing_20ms_residual_us": float(np.median(residual)),
    }


def trace_features(run_root: Path, label: int, sample_rate: float, tail_window_us: float) -> dict[str, Any]:
    metadata = json.loads((run_root / "iq/metadata.json").read_text(encoding="utf-8"))
    parser_rows = read_csv(run_root / "diagnostics/one_stage_cpp/ble_packets.csv")
    candidates = read_csv(run_root / "results/parser_candidate_packets.csv")
    iq_path = run_root / "iq/capture.sc16"
    iq = np.memmap(iq_path, dtype="<i2", mode="r").reshape(-1, 2)
    event = event_boundary_features(iq, parser_rows, sample_rate, tail_window_us)
    timing = timing_features(parser_rows)
    tail = np.asarray([item["tail_energy_db"] for item in event], dtype=np.float64)
    peak = np.asarray([item["tail_peak_db"] for item in event], dtype=np.float64)
    duration = np.asarray([item["duration_residual_us"] for item in event], dtype=np.float64)
    return {
        "run_id": run_root.name,
        "label": label,
        "sample_rate_sps": sample_rate,
        "parser_rows": len(parser_rows),
        "phantom_candidates": len(candidates),
        "phantom_candidate_rate_hz": len(candidates) / float(metadata.get("duration_s", 10.0) or 10.0),
        "boundary_event_count": len(event),
        "tail_energy_db_mean": float(np.mean(tail)) if tail.size else 0.0,
        "tail_energy_db_p95": float(np.percentile(tail, 95)) if tail.size else 0.0,
        "tail_energy_db_max": float(np.max(tail)) if tail.size else 0.0,
        "tail_peak_db_p95": float(np.percentile(peak, 95)) if peak.size else 0.0,
        "duration_residual_us_mean": float(np.mean(duration)) if duration.size else 0.0,
        "duration_residual_us_p95": float(np.percentile(duration, 95)) if duration.size else 0.0,
        "duration_residual_us_max": float(np.max(duration)) if duration.size else 0.0,
        **timing,
    }


def roc_curve(rows: list[dict[str, Any]], score_key: str) -> tuple[list[dict[str, float]], float, dict[str, float | None]]:
    positives = [float(row[score_key]) for row in rows if int(row["label"]) == 1]
    negatives = [float(row[score_key]) for row in rows if int(row["label"]) == 0]
    if not positives or not negatives:
        return [], float("nan"), {"tpr_at_fpr_0.01": None, "tpr_at_fpr_0.05": None, "tpr_at_fpr_0.10": None}
    thresholds = sorted(set(positives + negatives), reverse=True)
    points = [{"fpr": 0.0, "tpr": 0.0, "threshold": float("inf")}]
    for threshold in thresholds:
        tp = sum(score >= threshold for score in positives)
        fp = sum(score >= threshold for score in negatives)
        points.append({"fpr": fp / len(negatives), "tpr": tp / len(positives), "threshold": threshold})
    ordered = sorted(points, key=lambda item: (item["fpr"], item["tpr"]))
    auc = 0.0
    for left, right in zip(ordered, ordered[1:]):
        auc += (right["fpr"] - left["fpr"]) * (right["tpr"] + left["tpr"]) / 2.0
    pair_total = len(positives) * len(negatives)
    pair_score = sum(pos > neg for pos in positives for neg in negatives)
    pair_score += 0.5 * sum(pos == neg for pos in positives for neg in negatives)
    auc_rank = pair_score / pair_total if pair_total else float("nan")
    # The rank AUC is stable for tied finite samples; use it for the summary.
    operating: dict[str, float | None] = {}
    for target in (0.01, 0.05, 0.10):
        eligible = [point["tpr"] for point in points if point["fpr"] <= target]
        operating[f"tpr_at_fpr_{target:.2f}"] = max(eligible) if eligible else 0.0
    return points, auc_rank if math.isfinite(auc_rank) else auc, operating


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-run-root", action="append", type=Path, required=True)
    parser.add_argument("--positive-run-root", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tail-window-us", type=float, default=2200.0)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for label, roots in ((0, args.negative_run_root), (1, args.positive_run_root)):
        for root in roots:
            root = root.expanduser().resolve()
            metadata = json.loads((root / "iq/metadata.json").read_text(encoding="utf-8"))
            rate = float(metadata.get("actual_sample_rate_sps") or 4_000_000.0)
            rows.append(trace_features(root, label, rate, args.tail_window_us))
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "detector_trace_features.csv", rows)
    detector_keys = {
        "boundary_tail_energy": "tail_energy_db_p95",
        "boundary_duration_residual_mean": "duration_residual_us_mean",
        "boundary_duration_residual": "duration_residual_us_p95",
        "event_timing_cv": "timing_gap_cv",
        "event_timing_20ms_residual": "timing_20ms_residual_us",
        "commodity_phantom_candidate_rate": "phantom_candidate_rate_hz",
    }
    curves: dict[str, Any] = {}
    summary: dict[str, Any] = {
        "negative_traces": sum(int(row["label"]) == 0 for row in rows),
        "positive_traces": sum(int(row["label"]) == 1 for row in rows),
        "tail_window_us": args.tail_window_us,
        "sample_rate_requirement": "4 MS/s",
        "notes": [
            "Trace-level ROC uses five benign and only covert traces with at least one complete PC frame.",
            "Single-channel B210 captures can miss hopped channels; zero-PC covert traces are excluded from the positive ROC set.",
            "PIP unseen-AA/context-aware CIS analysis is separate and is not represented by the HRS candidate-rate diagnostic.",
        ],
    }
    for name, key in detector_keys.items():
        points, auc, operating = roc_curve(rows, key)
        curves[name] = {"score_field": key, "auc": auc, "operating_points": operating, "points": points}
        summary[f"{name}_auc"] = auc
        summary[f"{name}_operating_points"] = operating
    (output_dir / "detector_roc.json").write_text(
        json.dumps({"summary": summary, "curves": curves}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    roc_rows: list[dict[str, Any]] = []
    for name, curve in curves.items():
        for point in curve["points"]:
            roc_rows.append({"detector": name, **point})
    write_csv(output_dir / "detector_roc_points.csv", roc_rows)
    try:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(6.4, 5.2))
        for name, curve in curves.items():
            points = curve["points"]
            if points:
                x = [item["fpr"] for item in points]
                y = [item["tpr"] for item in points]
                axis.plot(x, y, marker="o", linewidth=1.2, label=f"{name} (AUC={curve['auc']:.3f})")
        axis.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="False-positive rate", ylabel="True-positive rate")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7, loc="lower right")
        figure.tight_layout()
        figure.savefig(output_dir / "detector_roc.png", dpi=180)
        plt.close(figure)
    except Exception as exc:
        summary["plot_error"] = str(exc)
        (output_dir / "detector_roc.json").write_text(
            json.dumps({"summary": summary, "curves": curves}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
