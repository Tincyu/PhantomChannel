#!/usr/bin/env python3
"""Analyze packet-level boundary features with capture-clustered bootstrap CIs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def number(value: Any, default: float | None = None) -> float | None:
    try:
        value = float(str(value).strip())
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def auc_and_tpr(scores: np.ndarray, labels: np.ndarray, target_fpr: float = 0.05) -> tuple[float | None, float | None]:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if not positives.size or not negatives.size:
        return None, None
    negative_sorted = np.sort(negatives)
    left = np.searchsorted(negative_sorted, positives, side="left")
    right = np.searchsorted(negative_sorted, positives, side="right")
    auc = float(np.sum(left + 0.5 * (right - left)) / (positives.size * negatives.size))
    thresholds = np.unique(positives)
    negative_counts = negatives.size - np.searchsorted(negative_sorted, thresholds, side="left")
    positive_counts = positives.size - np.searchsorted(np.sort(positives), thresholds, side="left")
    allowed = negative_counts / negatives.size <= target_fpr
    best = float(np.max(positive_counts[allowed] / positives.size)) if np.any(allowed) else 0.0
    return float(auc), float(best)


def fixed_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> tuple[float | None, float | None]:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if not positives.size or not negatives.size:
        return None, None
    return float(np.mean(negatives >= threshold)), float(np.mean(positives >= threshold))


def aggregate(records: list[dict[str, Any]], threshold: float, seed: int, bootstrap: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_gain: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_gain.setdefault(str(row["gain_db"]), []).append(row)
    result_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for gain, gain_rows in sorted(by_gain.items(), key=lambda item: float(item[0])):
        labels = np.asarray([int(row["label"]) for row in gain_rows], dtype=np.int8)
        scores = np.asarray([float(row["tail_energy_db"]) for row in gain_rows], dtype=np.float64)
        captures = sorted({str(row["run_id"]) for row in gain_rows})
        capture_rows = {capture: [row for row in gain_rows if str(row["run_id"]) == capture] for capture in captures}
        auc, tpr5 = auc_and_tpr(scores, labels)
        fpr_theta, tpr_theta = fixed_threshold(scores, labels, threshold)
        capture_summaries: list[dict[str, Any]] = []
        for capture in captures:
            local = capture_rows[capture]
            local_scores = np.asarray([float(row["tail_energy_db"]) for row in local])
            local_labels = np.asarray([int(row["label"]) for row in local])
            local_fpr, local_tpr = fixed_threshold(local_scores, local_labels, threshold)
            capture_summaries.append({"run_id": capture, "label": int(local_labels[0]), "auc": auc_and_tpr(local_scores, local_labels)[0], "fpr_theta": local_fpr, "tpr_theta": local_tpr})
        boot_metrics: list[tuple[float, float, float, float]] = []
        # Resample complete captures within each label. Packets within a capture
        # remain correlated and are never treated as independent repetitions.
        negative_captures = [c for c in captures if int(capture_rows[c][0]["label"]) == 0]
        positive_captures = [c for c in captures if int(capture_rows[c][0]["label"]) == 1]
        if negative_captures and positive_captures:
            for _ in range(bootstrap):
                sampled = []
                for pool in (negative_captures, positive_captures):
                    sampled.extend(rng.choice(pool, size=len(pool), replace=True).tolist())
                sampled_rows = [row for capture in sampled for row in capture_rows[capture]]
                boot_scores = np.asarray([float(row["tail_energy_db"]) for row in sampled_rows])
                boot_labels = np.asarray([int(row["label"]) for row in sampled_rows])
                boot_auc, boot_tpr = auc_and_tpr(boot_scores, boot_labels)
                boot_fpr, boot_theta_tpr = fixed_threshold(boot_scores, boot_labels, threshold)
                if None not in (boot_auc, boot_tpr, boot_fpr, boot_theta_tpr):
                    boot_metrics.append((boot_auc, boot_tpr, boot_fpr, boot_theta_tpr))
        boot_array = np.asarray(boot_metrics, dtype=np.float64) if boot_metrics else np.empty((0, 4))
        ci: dict[str, Any] = {}
        for index, name in enumerate(("auc", "tpr_at_fpr_05", "fpr_theta_5", "tpr_theta_5")):
            if boot_array.size:
                ci[name] = [float(np.percentile(boot_array[:, index], 2.5)), float(np.percentile(boot_array[:, index], 97.5))]
            else:
                ci[name] = None
        result_rows.append({
            "gain_db": float(gain),
            "capture_count": len(captures),
            "benign_capture_count": len(negative_captures),
            "covert_capture_count": len(positive_captures),
            "eligible_packet_count": len(gain_rows),
            "benign_packet_count": int(np.sum(labels == 0)),
            "covert_packet_count": int(np.sum(labels == 1)),
            "auc": auc,
            "tpr_at_fpr_05": tpr5,
            "theta_5_db": threshold,
            "fpr_theta_5": fpr_theta,
            "tpr_theta_5": tpr_theta,
            "cluster_bootstrap_replicates": len(boot_metrics),
            "auc_ci95": json.dumps(ci["auc"]),
            "tpr_at_fpr_05_ci95": json.dumps(ci["tpr_at_fpr_05"]),
            "fpr_theta_5_ci95": json.dumps(ci["fpr_theta_5"]),
            "tpr_theta_5_ci95": json.dumps(ci["tpr_theta_5"]),
            "ambiguous_note": "ambiguous packets excluded from conditional ROC and counted in capture summary",
        })
        bootstrap_rows.extend({"gain_db": float(gain), **summary} for summary in capture_summaries)
    return result_rows, bootstrap_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--capture-summary", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.calibration.read_text(encoding="utf-8"))
    threshold = float(config["theta_5_db"])
    features = read_csv(args.features)
    # Conditional ROC estimand: complete IQ, non-ambiguous target packets only.
    eligible = [row for row in features if row.get("iq_window_complete") == "1" and row.get("collision_ambiguous") == "0"]
    result_rows, capture_rows = aggregate(eligible, threshold, args.seed, args.bootstrap)
    write_csv(output / "boundary_roc_by_gain.csv", result_rows)
    write_csv(output / "boundary_capture_operating_points.csv", capture_rows)
    summary_rows = read_csv(args.capture_summary)
    write_csv(output / "boundary_capture_summary.csv", summary_rows)
    report = {
        "schema_version": 1,
        "detector": "advertising_packet_boundary_tail_energy",
        "estimand": "conditional ROC on parser-visible CRC-valid, eligible, uncontaminated target packets",
        "threshold_source": str(args.calibration),
        "theta_5_db": threshold,
        "formal_gain_cells_db": [float(row["gain_db"]) for row in result_rows],
        "bootstrap": {"unit": "complete capture/session cluster", "replicates_requested": args.bootstrap, "seed": args.seed},
        "results": result_rows,
        "notes": [
            "C_parser uses the nominal 20 ms advertising schedule because no transmitter event counter is present in the ledger.",
            "Coverage, SNR and ambiguous fractions are retained per capture in boundary_capture_summary.csv.",
            "ROC-derived TPR@5% FPR and calibration-frozen theta_5 operating points are separate quantities.",
        ],
    }
    (output / "boundary_roc_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"gains": len(result_rows), "eligible_packets": len(eligible), "theta_5_db": threshold, "output_dir": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
