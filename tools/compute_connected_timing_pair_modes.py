#!/usr/bin/env python3
"""Emit descriptive connected timing diagnostics for both frozen pair roles.

The output deliberately remains diagnostic rather than silently turning
observed parser packets into a formal connected ROC.  Missing event counters
are allowed; event intervals are normalized by the modular counter step.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Any

from analyze_event_timing_detector import (
    capture_manifest_duration,
    connection_events_from_rows,
    tshark_rows,
)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def summarize(values: list[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    center = statistics.median(values)
    mad = statistics.median(abs(value - center) for value in values)
    return center, mad, percentile(values, 0.95)


def specs(root: Path) -> list[tuple[str, str, Path]]:
    result: list[tuple[str, str, Path]] = []
    central = root / "event_timing/formal_20260809/connection_central"
    for condition, prefix in (
        ("normal", "timing_formal_conn_normal_rep"),
        ("central-side-8B", "timing_formal_conn_central_8b_rep"),
    ):
        for rep in range(1, 6):
            result.append((condition, "central-group", central / f"{prefix}{rep}/monitor/monitor.pcapng"))
    peripheral = root / "event_timing/formal_20260809/connection_peripheral"
    for rep in range(1, 6):
        result.append(("peripheral-side-240B", "peripheral-group",
                       peripheral / f"timing_formal_conn_peripheral_240b_rep{rep}/monitor/monitor.pcapng"))
    return result


def one(condition: str, group: str, path: Path, mode: str) -> dict[str, Any]:
    fields = [
        "frame.number", "frame.time_epoch", "btle.access_address",
        "btle.link_layer_data.access_address", "btle.link_layer_data.interval",
        "btle.link_layer_data.latency",
        "nordic_ble.direction", "nordic_ble.channel", "nordic_ble.event_counter",
        "nordic_ble.packet_counter", "nordic_ble.delta_time", "nordic_ble.crcok",
        "btle.data_header.llid", "btle.data_header.length",
    ]
    rows = tshark_rows(path, fields)
    connect_row = next(
        (row for row in rows if row.get("btle.link_layer_data.access_address", "").strip()),
        {},
    )
    latency_text = connect_row.get("btle.link_layer_data.latency", "").strip()
    latency = int(latency_text, 0) if latency_text else None
    events, duration_s, packet_count, selected_aa = connection_events_from_rows(rows, "")
    counters = [event.event_counter for event in events]
    steps = [((right - left) & 0xFFFF) for left, right in zip(counters, counters[1:])]
    starts = [event.start_us for event in events]
    intervals = []
    for left, right in zip(events, events[1:]):
        step = (right.event_counter - left.event_counter) & 0xFFFF
        raw_interval = right.start_us - left.start_us
        if step > 0 and raw_interval > 0:
            intervals.append(raw_interval / float(step))
    gaps: list[float] = []
    paired_events = 0
    for event in events:
        pairs = event.inter_frame_gaps(mode)
        if pairs:
            paired_events += 1
            gaps.extend(value for value, _ in pairs)
    gap_median, gap_mad, gap_p95 = summarize(gaps)
    event_median, event_mad, event_p95 = summarize(intervals)
    manifest_duration = capture_manifest_duration(path)
    return {
        "condition": condition,
        "group": group,
        "capture_id": path.parent.parent.name,
        "path": str(path),
        "pair_mode": mode,
        "selected_access_address": selected_aa,
        "connect_interval_units": connect_row.get("btle.link_layer_data.interval", ""),
        "connect_latency": latency,
        "expected_event_counter_stride": 1 if latency == 0 else None,
        "duration_s": manifest_duration if manifest_duration is not None else duration_s,
        "observed_event_count": len(events),
        "crc_valid_acl_packet_count": packet_count,
        "event_counter_step_median": statistics.median(steps) if steps else None,
        "event_counter_step_mode": max(set(steps), key=steps.count) if steps else None,
        "event_counter_step1_fraction": steps.count(1) / len(steps) if steps else None,
        "event_counter_gap_count": sum(step > 1 for step in steps),
        "crc_valid_acl_packet_count": sum(
            packet.crc_valid is True for event in events for packet in event.packets
        ),
        "crc_error_acl_packet_count": sum(
            packet.crc_valid is False for event in events for packet in event.packets
        ),
        "event_interval_median_us": event_median,
        "event_interval_mad_us": event_mad,
        "event_interval_p95_us": event_p95,
        "paired_event_count": paired_events,
        "pair_coverage_observed": paired_events / len(events) if events else None,
        "pair_gap_median_us": gap_median,
        "pair_gap_mad_us": gap_mad,
        "pair_gap_p95_us": gap_p95,
        "formal_status": "observed_event_pair_diagnostic_no_roc",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", type=Path,
                        default=Path("/path/to/PhantomChannel/experiments"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [one(condition, group, path, mode)
            for condition, group, path in specs(args.experiments_root)
            if path.is_file()
            for mode in ("central", "peripheral")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "connected_pair_mode_diagnostic.csv"
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} diagnostic rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
