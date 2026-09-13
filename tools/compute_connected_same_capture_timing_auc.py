#!/usr/bin/env python3
"""Compute connected-state TIFS AUC with labels from the same captures.

For each eligible direction-changing pair, a pair is ``normal`` when both
packets have zero payload length.  Every other eligible pair is a covert-data
candidate.  This avoids using a separate normal capture as the negative class
when the formal positive capture already contains empty (non-covert) events.

This is an exploratory connected-state diagnostic and intentionally does not
flip score direction or calculate thresholds/CI gates.
"""

from __future__ import annotations

import argparse
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


def load_run(path: Path, condition: str, pair_role: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = tshark_rows(path, FIELDS)
    events, duration_s, selected_packet_count, selected_aa = connection_events_from_rows(rows, "")
    run_id = path.parent.parent.name
    pairs: list[dict[str, Any]] = []
    for event in events:
        for pair in eligible_pairs(event):
            if pair["role"] != pair_role:
                continue
            label = "normal" if pair["both_empty_packets"] else "covert-candidate"
            pairs.append({
                "run_id": run_id,
                "run_dir": str(path.parent.parent),
                "condition": condition,
                "pair_role": pair_role,
                "label": label,
                "selected_access_address": selected_aa,
                **pair,
            })
    normal = [pair["tifs_us"] for pair in pairs if pair["label"] == "normal"]
    covert = [pair["tifs_us"] for pair in pairs if pair["label"] != "normal"]
    session = {
        "run_id": run_id,
        "run_dir": str(path.parent.parent),
        "condition": condition,
        "pair_role": pair_role,
        "pcap": str(path),
        "selected_access_address": selected_aa,
        "pcap_duration_s": duration_s,
        "selected_packet_count": selected_packet_count,
        "tifs_pair_count": len(pairs),
        "normal_pair_count": len(normal),
        "covert_candidate_pair_count": len(covert),
        "normal_tifs_median_us": statistics.median(normal) if normal else None,
        "covert_candidate_tifs_median_us": statistics.median(covert) if covert else None,
        "tifs_values_us": sorted({pair["tifs_us"] for pair in pairs}),
    }
    return session, pairs


def finite_values(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def compute_summary(sessions: list[dict[str, Any]], pairs: list[dict[str, Any]], condition: str) -> list[dict[str, Any]]:
    normal_pairs = [pair for pair in pairs if pair["label"] == "normal"]
    covert_pairs = [pair for pair in pairs if pair["label"] != "normal"]
    normal_center = statistics.median([pair["tifs_us"] for pair in normal_pairs]) if normal_pairs else None
    output: list[dict[str, Any]] = []

    for sample_unit in ("pair-pooled", "session-median"):
        if sample_unit == "pair-pooled":
            negative_rows = normal_pairs
            positive_rows = covert_pairs
            eligible_sessions = sessions
            excluded_sessions: list[str] = []
        else:
            negative_rows = []
            positive_rows = []
            eligible_sessions = []
            excluded_sessions = []
            for session in sessions:
                run_id = session["run_id"]
                normal = [pair["tifs_us"] for pair in normal_pairs if pair["run_id"] == run_id]
                covert = [pair["tifs_us"] for pair in covert_pairs if pair["run_id"] == run_id]
                if normal and covert:
                    eligible_sessions.append(session)
                    negative_rows.append({"run_id": run_id, "tifs_us": statistics.median(normal)})
                    positive_rows.append({"run_id": run_id, "tifs_us": statistics.median(covert)})
                else:
                    excluded_sessions.append(run_id)

        for feature in ("tifs_us", "abs_deviation_from_normal_median_us"):
            def score(row: dict[str, Any]) -> float:
                value = float(row["tifs_us"])
                if feature == "tifs_us":
                    return value
                return abs(value - float(normal_center)) if normal_center is not None else float("nan")

            negative = finite_values([score(row) for row in negative_rows])
            positive = finite_values([score(row) for row in positive_rows])
            output.append({
                "traffic": "connection",
                "condition": condition,
                "pair_role": sessions[0]["pair_role"] if sessions else "",
                "label_definition": "normal iff both endpoint packet lengths are zero; otherwise covert-candidate",
                "sample_unit": sample_unit,
                "feature": feature,
                "score_direction": (
                    "higher_tifs_is_more_anomalous" if feature == "tifs_us"
                    else "farther_from_normal_median_is_more_anomalous"
                ),
                "direction_transform": "none",
                "normal_reference_median_tifs_us": normal_center,
                "capture_count": len(sessions),
                "eligible_session_count": len(eligible_sessions) if sample_unit == "session-median" else len(sessions),
                "excluded_session_count": len(excluded_sessions) if sample_unit == "session-median" else 0,
                "excluded_session_ids": ";".join(excluded_sessions),
                "normal_sample_count": len(negative),
                "covert_candidate_sample_count": len(positive),
                "auc": auc(negative, positive),
                "interpretation": "exploratory connected TIFS AUC; labels are assigned within the same formal captures",
            })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--pair-role", required=True, choices=(
        "central-request-response", "peripheral-notification-response",
    ))
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    sessions: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for path in args.run:
        session, run_pairs = load_run(path.resolve(), args.condition, args.pair_role)
        sessions.append(session)
        pairs.extend(run_pairs)

    summary = compute_summary(sessions, pairs, args.condition)
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
        "label_definition": "both endpoint packets length=0 => normal; otherwise covert-candidate",
        "score_direction": "no automatic direction flip",
        "outputs": {
            "session_summary": str(session_path),
            "pairs": str(pair_path),
            "auc_summary": str(auc_path),
        },
        "notes": [
            "This same-capture classification supersedes the earlier separate-normal-capture diagnostic.",
            "This is exploratory and does not alter the active advertising-only timing plan.",
            "Peripheral captures without both labels are retained in the session manifest but excluded from session-median AUC.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"sessions": sessions, "auc_summary": summary}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
