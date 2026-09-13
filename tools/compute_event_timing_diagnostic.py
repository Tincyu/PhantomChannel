#!/usr/bin/env python3
"""Compute descriptive timing diagnostics under the revised §6 gap policy."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from analyze_event_timing_detector import (
    connection_events_from_rows,
    load_capture,
    tshark_rows,
)
from audit_event_timing_inputs import specs


FIELDS = [
    "group", "condition", "capture_id", "path", "duration_s", "selected_access_address",
    "raw_packet_count", "observed_event_count", "observed_window_count",
    "event_counter_step_median", "event_counter_step_mode", "event_counter_step1_fraction",
    "event_interval_median_us", "event_interval_p95_us", "event_counter_gap_count",
    "crc_valid_acl_packet_count", "crc_error_acl_packet_count",
    "central_pair_coverage", "peripheral_pair_coverage",
    "central_gap_median_us", "peripheral_gap_median_us",
    "pip_tx_done", "pip_event_done", "pip_notifications", "pip_clean_session",
    "formal_timing_status",
]


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    left = int(position)
    right = min(left + 1, len(values) - 1)
    weight = position - left
    return values[left] * (1.0 - weight) + values[right] * weight


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def diagnostic_one(group: str, traffic: str, condition: str, path: Path) -> dict[str, Any]:
    row = {field: None for field in FIELDS}
    row.update({"group": group, "condition": condition, "capture_id": path.parent.parent.name,
                "path": str(path)})
    if traffic != "connection" or not path.is_file():
        row["formal_timing_status"] = "not_a_connected_diagnostic"
        return row
    fields = [
        "frame.number", "frame.time_epoch", "btle.access_address",
        "nordic_ble.direction", "nordic_ble.channel", "nordic_ble.event_counter",
        "nordic_ble.packet_counter", "nordic_ble.delta_time", "nordic_ble.crcok",
        "btle.data_header.llid", "btle.data_header.length",
    ]
    rows = tshark_rows(path, fields)
    require_crc_valid = group == "pip-supplemental"
    events, _pcap_duration, selected_count, selected_aa = connection_events_from_rows(
        rows, "", require_crc_valid=require_crc_valid,
    )
    capture = load_capture(
        path, "connection", condition, "", "", "all", 0.8,
        path.parent.parent.name, False, require_crc_valid=require_crc_valid,
    )
    if group == "pip-supplemental":
        summary_path = path.parent.parent / "pip_trace_summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            uart = summary.get("uart", {})
            decision = summary.get("decision", {})
            row.update({
                "pip_tx_done": uart.get("pip_ll_tx_done"),
                "pip_event_done": uart.get("pip_ll_event_done"),
                "pip_notifications": uart.get("central_notifications"),
                "pip_clean_session": decision.get("clean_session"),
            })
    counters = [event.event_counter for event in events]
    steps = [((right - left) & 0xFFFF) for left, right in zip(counters, counters[1:])]
    intervals = []
    for left, right in zip(events, events[1:]):
        step = (right.event_counter - left.event_counter) & 0xFFFF
        raw_interval = right.start_us - left.start_us
        if step > 0 and raw_interval > 0:
            intervals.append(raw_interval / float(step))
    role_gaps = {"central-request-response": [], "peripheral-notification-response": []}
    role_event_counts = {key: 0 for key in role_gaps}
    for event in events:
        for gap, role in event.inter_frame_gaps("all"):
            role_gaps[role].append(gap)
            role_event_counts[role] += 1
    row.update({
        "duration_s": capture.duration_s,
        "selected_access_address": selected_aa,
        "raw_packet_count": selected_count,
        "observed_event_count": len(events),
        "observed_window_count": len(capture.windows),
        "event_counter_step_median": median([float(x) for x in steps]),
        "event_counter_step_mode": (max(set(steps), key=steps.count) if steps else None),
        "event_counter_step1_fraction": (steps.count(1) / len(steps) if steps else None),
        "event_interval_median_us": median(intervals),
        "event_interval_p95_us": quantile(intervals, 0.95),
        "event_counter_gap_count": sum(step > 1 for step in steps),
        "crc_valid_acl_packet_count": sum(
            packet.crc_valid is True for event in events for packet in event.packets
        ),
        "crc_error_acl_packet_count": sum(
            packet.crc_valid is False for event in events for packet in event.packets
        ),
        "central_pair_coverage": (role_event_counts["central-request-response"] / len(events) if events else None),
        "peripheral_pair_coverage": (role_event_counts["peripheral-notification-response"] / len(events) if events else None),
        "central_gap_median_us": median(role_gaps["central-request-response"]),
        "peripheral_gap_median_us": median(role_gaps["peripheral-notification-response"]),
        "formal_timing_status": (
            "pip_supplemental_requires_attempt_mapping"
            if group == "pip-supplemental"
            else "valid_observed_event_window" if capture.windows
            else "invalid_no_50_observed_event_window"
        ),
    })
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", type=Path,
                        default=Path("/path/to/PhantomChannel/experiments"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [diagnostic_one(*spec) for spec in specs(args.experiments_root) if spec[1] == "connection"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "timing_observed_diagnostic.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": 1,
        "purpose": "descriptive observed-event timing only; not formal ROC",
        "window_policy": {
            "standard_connection": "formal 50 observed reconstructed events, non-overlapping; event-counter gaps allowed",
            "pip_supplemental": "session-level eligible PIP-active events; event-counter gaps allowed",
        },
        "capture_count": len(rows),
        "valid_formal_window_captures": sum(
            row["formal_timing_status"] == "valid_observed_event_window" for row in rows
        ),
        "outputs": {"csv": str(args.output_dir / "timing_observed_diagnostic.csv")},
        "captures": rows,
    }
    (args.output_dir / "timing_observed_diagnostic.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: summary[key] for key in ("purpose", "capture_count", "valid_formal_window_captures", "outputs")}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
