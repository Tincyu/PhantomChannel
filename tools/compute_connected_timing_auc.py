#!/usr/bin/env python3
"""Compute exploratory connected-state TIFS AUCs using the PIP method.

This is deliberately separate from the frozen advertising timing result.  It
uses the same air-interface measurement as the PIP diagnostic: one eligible
direction-changing pair per role and connection event, with
``nordic_ble.delta_time`` as TIFS.  It reports pair-pooled and
session-median AUC, without thresholds, TPR, CI, or score-direction flipping.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

from analyze_event_timing_detector import connection_events_from_rows, tshark_rows
from compute_pip_timing_auc import auc, eligible_pairs, write_csv


FIELDS = [
    "frame.number", "frame.time_epoch", "btle.access_address",
    "nordic_ble.event_counter", "nordic_ble.channel", "nordic_ble.direction",
    "nordic_ble.packet_counter", "nordic_ble.delta_time", "nordic_ble.crcok",
    "btle.data_header.llid", "btle.data_header.length",
]


def load_run(path: Path, label: str, condition: str, pair_role: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = tshark_rows(path, FIELDS)
    events, duration_s, selected_packet_count, selected_aa = connection_events_from_rows(rows, "")
    pairs: list[dict[str, Any]] = []
    for event in events:
        for pair in eligible_pairs(event):
            if pair["role"] != pair_role:
                continue
            pairs.append({
                "run_id": path.parent.parent.name,
                "run_dir": str(path.parent.parent),
                "label": label,
                "condition": condition,
                "pair_role": pair_role,
                "selected_access_address": selected_aa,
                **pair,
            })
    session = {
        "run_id": path.parent.parent.name,
        "run_dir": str(path.parent.parent),
        "label": label,
        "condition": condition,
        "pair_role": pair_role,
        "pcap": str(path),
        "selected_access_address": selected_aa,
        "pcap_duration_s": duration_s,
        "selected_packet_count": selected_packet_count,
        "tifs_pair_count": len(pairs),
        "tifs_median_us": statistics.median([p["tifs_us"] for p in pairs]) if pairs else None,
        "tifs_values_us": sorted({p["tifs_us"] for p in pairs}),
    }
    return session, pairs


def compute_rows(
    sessions: list[dict[str, Any]], pairs: list[dict[str, Any]], positive_condition: str,
) -> list[dict[str, Any]]:
    negative_sessions = [s for s in sessions if s["label"] == "matched-benign"]
    positive_sessions = [s for s in sessions if s["label"] == positive_condition]
    negative_ids = {s["run_id"] for s in negative_sessions}
    positive_ids = {s["run_id"] for s in positive_sessions}
    negative_pairs = [p for p in pairs if p["run_id"] in negative_ids]
    positive_pairs = [p for p in pairs if p["run_id"] in positive_ids]
    benign_center = statistics.median([p["tifs_us"] for p in negative_pairs]) if negative_pairs else None
    rows: list[dict[str, Any]] = []
    for sample_unit in ("pair-pooled", "session-median"):
        if sample_unit == "pair-pooled":
            neg_rows = negative_pairs
            pos_rows = positive_pairs
        else:
            neg_rows = []
            pos_rows = []
            for session in negative_sessions:
                values = [p["tifs_us"] for p in negative_pairs if p["run_id"] == session["run_id"]]
                if values:
                    neg_rows.append({"run_id": session["run_id"], "tifs_us": statistics.median(values)})
            for session in positive_sessions:
                values = [p["tifs_us"] for p in positive_pairs if p["run_id"] == session["run_id"]]
                if values:
                    pos_rows.append({"run_id": session["run_id"], "tifs_us": statistics.median(values)})
        for feature in ("tifs_us", "abs_deviation_from_benign_median_us"):
            def score(row: dict[str, Any]) -> float:
                value = float(row["tifs_us"])
                return value if feature == "tifs_us" else abs(value - float(benign_center))

            negative = [score(row) for row in neg_rows if benign_center is not None]
            positive = [score(row) for row in pos_rows if benign_center is not None]
            negative = [v for v in negative if math.isfinite(v)]
            positive = [v for v in positive if math.isfinite(v)]
            rows.append({
                "traffic": "connection",
                "condition": positive_condition,
                "pair_role": positive_sessions[0]["pair_role"] if positive_sessions else "",
                "sample_unit": sample_unit,
                "feature": feature,
                "score_direction": (
                    "higher_tifs_is_more_anomalous" if feature == "tifs_us"
                    else "farther_from_benign_median_is_more_anomalous"
                ),
                "direction_transform": "none",
                "benign_reference_median_tifs_us": benign_center,
                "negative_session_count": len({r["run_id"] for r in neg_rows}),
                "positive_session_count": len({r["run_id"] for r in pos_rows}),
                "negative_sample_count": len(negative),
                "positive_sample_count": len(positive),
                "auc": auc(negative, positive),
                "interpretation": "exploratory connected TIFS AUC computed with the PIP air-interface method",
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=("central-side-8B", "peripheral-side-240B"))
    parser.add_argument("--pair-role", required=True, choices=("central-request-response", "peripheral-notification-response"))
    parser.add_argument("--positive-run", type=Path, action="append", required=True)
    parser.add_argument("--negative-run", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sessions: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for path in args.negative_run:
        session, run_pairs = load_run(path.resolve(), "matched-benign", "matched-benign", args.pair_role)
        sessions.append(session)
        pairs.extend(run_pairs)
    for path in args.positive_run:
        session, run_pairs = load_run(path.resolve(), args.condition, args.condition, args.pair_role)
        sessions.append(session)
        pairs.extend(run_pairs)
    summary = compute_rows(sessions, pairs, args.condition)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    session_path = args.output_dir / f"{args.condition}_tifs_session_summary.csv"
    pair_path = args.output_dir / f"{args.condition}_tifs_pairs.csv"
    auc_path = args.output_dir / f"{args.condition}_tifs_auc_summary.csv"
    manifest_path = args.output_dir / f"{args.condition}_tifs_auc_manifest.json"
    write_csv(session_path, sessions)
    write_csv(pair_path, pairs)
    write_csv(auc_path, summary)
    manifest = {
        "schema_version": 1,
        "traffic": "connection",
        "condition": args.condition,
        "pair_role": args.pair_role,
        "tifs_definition": "nordic_ble.delta_time; one eligible pair per role and event",
        "score_direction": "no automatic direction flip",
        "outputs": {
            "session_summary": str(session_path),
            "pairs": str(pair_path),
            "auc_summary": str(auc_path),
        },
        "notes": [
            "This is exploratory and does not alter the active advertising-only timing plan.",
            "Pair-pooled AUC is descriptive; session-median AUC is repetition-aware.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"sessions": sessions, "auc_summary": summary}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
