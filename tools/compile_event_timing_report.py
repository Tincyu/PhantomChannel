#!/usr/bin/env python3
"""Compile the current §6 timing results without promoting diagnostics to ROC."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-root", type=Path, required=True)
    parser.add_argument("--pip-audit-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    advertising = read_csv(args.timing_root / "advertising/timing_roc_summary.csv")
    pair = read_csv(args.timing_root / "connection_diagnostic/connected_pair_mode_diagnostic.csv")
    pip = read_csv(args.pip_audit_root / "pip_timing_session_summary.csv")
    heldout_connected_roc_path = args.timing_root / "connection_central_heldout_cal_r10/timing_roc_summary.csv"
    connected_roc_path = (
        heldout_connected_roc_path
        if heldout_connected_roc_path.is_file()
        else args.timing_root / "connection_relaxed_central_matched_cal/timing_roc_summary.csv"
    )
    connected_roc = read_csv(connected_roc_path) if connected_roc_path.is_file() else []
    rows: list[dict[str, Any]] = []
    for row in advertising:
        rows.append({
            "traffic": "advertising",
            "condition": row.get("condition", ""),
            "pair_mode": "n/a",
            "feature": row.get("feature", ""),
            "analysis": "formal_roc",
            "auc": row.get("auc", ""),
            "auc_ci95_low": row.get("auc_ci95_clustered_low", ""),
            "auc_ci95_high": row.get("auc_ci95_clustered_high", ""),
            "tpr_at_5pct_fpr": row.get("tpr_at_5pct_fpr", ""),
            "valid_capture_count": row.get("positive_capture_count", ""),
            "status": "computed" if row.get("auc", "") else "null_no_valid_positive_windows",
            "note": "advertising formal held-out result",
        })
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in pair:
        grouped[(row["condition"], row["pair_mode"])].append(row)
    for (condition, mode), group in sorted(grouped.items()):
        rows.append({
            "traffic": "connection",
            "condition": condition,
            "pair_mode": mode,
            "feature": "event_interval_and_pair_coverage_diagnostic",
            "analysis": "observed_event_diagnostic",
            "auc": "",
            "auc_ci95_low": "",
            "auc_ci95_high": "",
            "tpr_at_5pct_fpr": "",
            "valid_capture_count": 0,
            "status": "diagnostic_observed_event_summary",
            "note": f"{len(group)} observed rows; event-counter gaps allowed",
        })
    for row in connected_roc:
        rows.append({
            "traffic": "connection",
            "condition": row.get("condition", ""),
            "pair_mode": "central-request-response",
            "feature": row.get("feature", ""),
            "analysis": "heldout_observed_event_formal_roc" if connected_roc_path == heldout_connected_roc_path else "relaxed_observed_event_formal_roc",
            "auc": row.get("auc", ""),
            "auc_ci95_low": row.get("auc_ci95_clustered_low", ""),
            "auc_ci95_high": row.get("auc_ci95_clustered_high", ""),
            "tpr_at_5pct_fpr": row.get("tpr_at_5pct_fpr", ""),
            "valid_capture_count": row.get("positive_capture_count", ""),
            "status": "computed_heldout_calibration" if connected_roc_path == heldout_connected_roc_path and row.get("auc", "") else ("computed_calibration_split_not_heldout" if row.get("auc", "") else "null_no_valid_windows"),
            "note": "Independent matched benign calibration; formal test remains held out; 50 observed events/window; event-counter gaps allowed" if connected_roc_path == heldout_connected_roc_path else "Exploratory matched-normal calibration split; not final held-out ROC; 50 observed events/window; event-counter gaps allowed",
        })
    accepted = sum(row.get("accepted", "").lower() == "true" for row in pip)
    rows.append({
        "traffic": "connection",
        "condition": "PIP",
        "pair_mode": "PIP-active",
        "feature": "controller/passive eligibility",
        "analysis": "controller_and_passive_eligibility",
        "auc": "",
        "auc_ci95_low": "",
        "auc_ci95_high": "",
        "tpr_at_5pct_fpr": "",
        "valid_capture_count": accepted,
        "status": "not_accepted_no_pip_roc",
        "note": f"{len(pip)} preflight sessions; sparse events allow gaps, but no session passed gates",
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "timing_calculation_summary.csv"
    json_path = args.output_dir / "timing_calculation_summary.json"
    write_csv(csv_path, rows)
    summary = {
        "schema_version": 1,
        "purpose": "detector_roc_experiment_redesign.md §6 timing calculation handoff",
        "rows": rows,
        "outputs": {"csv": str(csv_path), "json": str(json_path)},
        "interpretation": {
            "advertising": "formal ROC where valid windows exist",
            "standard_connected": "central-request-response calibration-split exploratory ROC with observed-event windows; independent connected calibration is still needed for final held-out ROC; peripheral target-pair coverage remains diagnostic",
            "pip": "supplemental eligibility/controller timing only; no AUC",
        },
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "csv": str(csv_path), "json": str(json_path)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
