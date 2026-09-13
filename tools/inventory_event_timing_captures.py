#!/usr/bin/env python3
"""Inventory existing §6 timing pcaps without treating windows as repetitions.

The inventory is intentionally conservative.  It records the files visible in
the supplied ``adv_osb`` and ``data_obs`` trees, reconstructs event/window
counts with the same implementation used by the detector, and records obvious
session-continuity evidence.  It does not infer five independent repetitions
from different payload lengths or from windows cut out of one pcap.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import analyze_event_timing_detector as detector  # noqa: E402


def length_from_name(path: Path) -> str:
    match = re.search(r"_(\d+)B(?:\.|$)", path.name)
    return f"{match.group(1)}B" if match else ""


def advertising_label(relative: Path) -> tuple[str, str]:
    parts = relative.parts
    length = length_from_name(relative)
    if "adv_normal" in parts:
        return "matched-nRF52840-benign", "matched nRF52840 benign control; one long pcap"
    if "adv_last" in parts:
        return f"append-last-{length or 'unknown'}", "payload-length sweep; not five matched repetitions"
    if "adv_every" in parts:
        return f"append-every-{length or 'unknown'}", "payload-length sweep; not five matched repetitions"
    return "unknown", "unmapped advertising source"


def connection_label(relative: Path) -> tuple[str, str]:
    parts = relative.parts
    lower_name = relative.name.lower()
    role = "central" if "central" in parts else "peripheral" if "Peripheral" in parts else "unknown"
    if lower_name == "normal.pcapng":
        return f"{role}-normal", "one normal pcap for this role"
    if lower_name == "embedd_8b.pcapng":
        return "central-side-8B", "single 8 B pcap"
    if lower_name == "embed_240b.pcapng":
        return "peripheral-side-240B", "single historical matched 240 B pcap"
    return f"{role}-diagnostic", "diagnostic/pilot source; not in the six-condition matrix"


def adv_record(path: Path, root: Path, target: str) -> dict[str, Any]:
    fields = [
        "frame.number", "frame.time_epoch", "btle.advertising_address",
        "nordic_ble.channel", "nordic_ble.packet_counter",
        "nordic_ble.delta_time", "nordic_ble.crcok",
    ]
    rows = detector.tshark_rows(path, fields)
    events, duration, raw_count = detector.adv_events_from_rows(rows, target)
    windows = detector.adv_windows(events, "", path.stem, 30)
    starts = [event.start_us for event in events]
    gaps_s = [(right - left) / 1_000_000.0 for left, right in zip(starts, starts[1:])]
    condition, note = advertising_label(path.relative_to(root))
    return {
        "path": str(path),
        "traffic": "advertising",
        "condition": condition,
        "duration_s": duration,
        "file_size_bytes": path.stat().st_size,
        "raw_target_packet_count": raw_count,
        "reconstructed_event_count": len(events),
        "available_window_count": len(windows),
        "window_events": 30,
        "event_start_gap_gt_5s_count": sum(gap > 5.0 for gap in gaps_s),
        "max_event_start_gap_s": max(gaps_s) if gaps_s else None,
        "session_evidence": "one continuous pcap; no >5 s event-start break"
        if gaps_s and max(gaps_s) <= 5.0 else "needs external run ledger",
        "classification_note": note,
    }


def connection_record(path: Path, root: Path) -> dict[str, Any]:
    fields = [
        "frame.number", "frame.time_epoch", "btle.access_address",
        "nordic_ble.direction", "nordic_ble.channel", "nordic_ble.event_counter",
        "nordic_ble.packet_counter", "nordic_ble.delta_time", "nordic_ble.crcok",
        "btle.data_header.llid", "btle.data_header.length",
    ]
    rows = detector.tshark_rows(path, fields)
    events, duration, raw_count, selected_aa = detector.connection_events_from_rows(rows, "")
    windows = detector.connection_windows(events, "", path.stem, 50, "all", 0.8)
    runs: list[list[detector.ConnEvent]] = []
    current: list[detector.ConnEvent] = []
    for event in events:
        if current and not detector.pairwise_consecutive(current[-1].event_counter, event.event_counter):
            runs.append(current)
            current = []
        current.append(event)
    if current:
        runs.append(current)
    condition, note = connection_label(path.relative_to(root))
    return {
        "path": str(path),
        "traffic": "connection",
        "condition": condition,
        "duration_s": duration,
        "file_size_bytes": path.stat().st_size,
        "raw_selected_packet_count": raw_count,
        "reconstructed_event_count": len(events),
        "event_counter_run_count": len(runs),
        "event_counter_run_lengths": ",".join(str(len(run)) for run in runs),
        "available_window_count": len(windows),
        "window_events": 50,
        "selected_access_address": selected_aa,
        "session_evidence": "multiple event-counter runs; external session ledger required"
        if len(runs) > 1 else "one event-counter run in file",
        "classification_note": note,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adv-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--target-address", default="d1:22:33:44:55:66")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    for path in sorted(args.adv_root.rglob("*.pcapng")):
        records.append(adv_record(path, args.adv_root, args.target_address))
    for path in sorted(args.data_root.rglob("*.pcapng")):
        records.append(connection_record(path, args.data_root))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for record in records:
        for key in record:
            if key not in fields:
                fields.append(key)
    import csv
    with (args.output_dir / "timing_capture_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    condition_counts: dict[str, int] = {}
    for record in records:
        condition_counts[record["condition"]] = condition_counts.get(record["condition"], 0) + 1
    audit = {
        "required_formal_capture_count": 30,
        "visible_pcap_file_count": len(records),
        "advertising_file_count": sum(record["traffic"] == "advertising" for record in records),
        "connection_file_count": sum(record["traffic"] == "connection" for record in records),
        "condition_file_counts": condition_counts,
        "formal_evidence_status": "not_demonstrated_from_file_tree",
        "reason": [
            "The visible tree contains 19 pcapng files rather than 30 capture files.",
            "Advertising files are a 16/32/64/128/240 B length sweep, not five matched repetitions per formal condition.",
            "Several connected files are shorter than the required 60 s, and one long file contains multiple event-counter runs.",
            "A pcap window is retained as a correlated observation and is never counted as an independent session.",
        ],
        "records": records,
    }
    (args.output_dir / "timing_capture_inventory.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: audit[key] for key in audit if key != "records"}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
