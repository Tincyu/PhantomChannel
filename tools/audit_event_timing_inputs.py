#!/usr/bin/env python3
"""Audit §6 event-level timing pcap inputs without fitting a detector.

The report keeps invalid captures visible.  It uses the same reconstruction
and fixed-size non-overlapping window functions as the timing analyzer. The
revised connected policy allows missing event counters; event intervals are
normalized by the observed modular counter step.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from analyze_event_timing_detector import load_capture


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def specs(root: Path) -> list[tuple[str, str, str, Path]]:
    result: list[tuple[str, str, str, Path]] = []
    advertising = root / "event_timing/formal_20260809/advertising"
    for condition, prefix in (
        ("matched-nRF52840-benign", "timing_formal_adv_normal_rep"),
        ("append-last-239B", "timing_formal_adv_append_last_rep"),
        ("append-every-239B", "timing_formal_adv_append_every_rep"),
    ):
        for rep in range(1, 6):
            result.append(("standard-formal", "advertising", condition,
                           advertising / f"{prefix}{rep}/monitor/monitor.pcapng"))

    central = root / "event_timing/formal_20260809/connection_central"
    for condition, prefix in (
        ("normal", "timing_formal_conn_normal_rep"),
        ("central-side-8B", "timing_formal_conn_central_8b_rep"),
    ):
        for rep in range(1, 6):
            result.append(("standard-formal", "connection", condition,
                           central / f"{prefix}{rep}/monitor/monitor.pcapng"))

    peripheral = root / "event_timing/formal_20260809/connection_peripheral"
    for rep in range(1, 6):
        result.append(("standard-formal", "connection", "peripheral-side-240B",
                       peripheral / f"timing_formal_conn_peripheral_240b_rep{rep}/monitor/monitor.pcapng"))

    pip = root / "figure/detector_roc_20260809/event_timing/pip_supplemental/formal"
    for rep in range(1, 6):
        result.append(("pip-supplemental", "connection", "PIP",
                       pip / f"timing_formal_pip_rep{rep}/monitor/monitor.pcapng"))
    return result


def audit_one(group: str, traffic: str, condition: str, path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "group": group,
        "traffic": traffic,
        "condition": condition,
        "capture_id": path.parent.parent.name,
        "path": str(path),
        "exists": path.is_file(),
    }
    if not path.is_file():
        row.update({"valid_for_formal": False, "reason": "missing_pcap"})
        return row
    try:
        target = "d1:22:33:44:55:66" if traffic == "advertising" else ""
        capture = load_capture(
            path, traffic, condition, target, "", "all", 0.8,
            path.parent.parent.name, False,
            require_crc_valid=(group == "pip-supplemental"),
        )
        row.update({
            "duration_s": capture.duration_s,
            "pcap_duration_s": capture.pcap_duration_s,
            "raw_packet_count": capture.raw_packet_count,
            "reconstructed_event_count": capture.event_count,
            "available_window_count": len(capture.windows),
            "selected_access_address": capture.selected_access_address,
            "valid_for_formal": bool(capture.duration_s >= 60.0 and capture.windows),
        })
        if group == "pip-supplemental":
            # PIP supplemental timing was redesigned to use sparse,
            # UART-led PIP-active events.  It must not be rejected merely
            # because the standard connected 50-observed-event window
            # cannot be formed.  A separate attempt->pcap mapping and the
            # §6.7 coverage/stability gates are required instead.
            row["valid_for_formal"] = False
            if not capture.event_count:
                row["reason"] = "pip_no_reconstructed_acl_events"
            else:
                row["reason"] = "pip_requires_uart_attempt_mapping_and_coverage_gate"
        elif not capture.event_count:
            row["reason"] = "no_reconstructed_events"
        elif not capture.windows:
            required = 30 if traffic == "advertising" else 50
            qualifier = "observed_event" if traffic == "connection" else "event"
            row["reason"] = f"no_valid_{required}_{qualifier}_window"
        elif capture.duration_s < 60.0:
            row["reason"] = "duration_below_60s"
        else:
            row["reason"] = "valid_window_available"
    except Exception as exc:  # keep the audit complete for one bad pcap
        row.update({"valid_for_formal": False, "reason": f"parse_error:{exc}"})
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", type=Path,
                        default=Path("/path/to/PhantomChannel/experiments"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [audit_one(*spec) for spec in specs(args.experiments_root)]
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"captures": 0, "valid": 0, "invalid": 0})
    for row in rows:
        key = f"{row['group']}::{row['traffic']}::{row['condition']}"
        counts[key]["captures"] += 1
        if row.get("valid_for_formal"):
            counts[key]["valid"] += 1
        else:
            counts[key]["invalid"] += 1
    summary = {
        "schema_version": 1,
        "window_policy": {
            "advertising_events_per_window": 30,
            "connection_events_per_window": 50,
            "overlap": "none",
            "event_counter_continuity_required": False,
            "connection_event_definition": "50 observed reconstructed events",
            "event_interval_normalization": "timestamp delta divided by modular event-counter step",
            "standard_timing_crc_policy": "retain target rows with parser CRC errors",
            "pip_crc_policy": "CRC-valid ACL rows only",
            "min_pair_coverage": 0.8,
            "pip_supplemental": {
                "event_counter_continuity_required": False,
                "window": "session-level eligible PIP-active events",
                "requires_uart_attempt_mapping": True,
            },
        },
        "captures_audited": len(rows),
        "valid_capture_count": sum(bool(row.get("valid_for_formal")) for row in rows),
        "invalid_capture_count": sum(not bool(row.get("valid_for_formal")) for row in rows),
        "by_condition": dict(counts),
        "outputs": {
            "capture_audit": str(args.output_dir / "timing_input_audit.csv"),
            "summary": str(args.output_dir / "timing_input_audit.json"),
        },
        "notes": [
            "This is an input/window audit, not a ROC calculation.",
            "Invalid captures remain in the report and are not filtered by outcome.",
            "PIP is supplemental and is not mixed into the six-condition standard ROC.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "timing_input_audit.csv", rows)
    (args.output_dir / "timing_input_audit.json").write_text(
        json.dumps({"summary": summary, "captures": rows}, indent=2,
                   ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
