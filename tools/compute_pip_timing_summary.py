#!/usr/bin/env python3
"""Summarize sparse PIP timing sessions under the current §6.7 rules.

This is intentionally a session/eligibility report, not a PIP ROC.  PIP
events do not need consecutive event counters, but a session still needs
UART-led attempt mapping and the frozen coverage/stability gates before it
can be accepted.  Missing mapping is reported explicitly rather than being
treated as zero timing or silently filtered.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any

from analyze_event_timing_detector import ADV_AA, tshark_rows


def q95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.95
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return ordered[left]
    weight = position - left
    return ordered[left] * (1.0 - weight) + ordered[right] * weight


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "mad": None, "p95": None}
    center = statistics.median(values)
    return {
        "median": center,
        "mad": statistics.median(abs(value - center) for value in values),
        "p95": q95(values),
    }


def last_boot_segment(path: Path, marker: str) -> str:
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    index = text.rfind(marker)
    return text[index:] if index >= 0 else text


def json_records(text: str, tag: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for match in re.finditer(rf"{re.escape(tag)}\s+(\{{.*\}})", text):
        try:
            value = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def pcap_inventory(path: Path) -> dict[str, Any]:
    fields = [
        "frame.time_relative", "btle.access_address",
        "btle.link_layer_data.access_address", "nordic_ble.crcok",
        "nordic_ble.event_counter", "nordic_ble.channel",
    ]
    rows = tshark_rows(path, fields)
    connect_aas = {
        row.get("btle.link_layer_data.access_address", "").strip().lower()
        for row in rows
        if row.get("btle.link_layer_data.access_address", "").strip()
    }
    connect_aas.discard(ADV_AA)
    non_adv = [
        row for row in rows
        if row.get("btle.access_address", "").strip().lower() not in {"", ADV_AA}
    ]
    acl = [
        row for row in non_adv
        if row.get("nordic_ble.event_counter", "").strip()
        and row.get("nordic_ble.crcok", "").strip().lower() in {"true", "1", "yes", "ok"}
        and (not connect_aas or row.get("btle.access_address", "").strip().lower() in connect_aas)
    ]
    times = [float(row["frame.time_relative"]) for row in rows if row.get("frame.time_relative", "").strip()]
    acl_times = [float(row["frame.time_relative"]) for row in acl]
    event_keys = {
        (row.get("btle.access_address", "").strip().lower(), row.get("nordic_ble.event_counter", "").strip())
        for row in acl
    }
    end_s = max(times, default=0.0)
    return {
        "pcap_bytes": path.stat().st_size if path.is_file() else 0,
        "pcap_frame_count": len(rows),
        "pcap_span_s": end_s - min(times, default=end_s),
        "connect_aa": sorted(connect_aas),
        "crc_valid_acl_packet_count": len(acl),
        "reconstructed_acl_event_count": len(event_keys),
        "last_5s_acl_packet_count": sum(time >= end_s - 5.0 for time in acl_times),
        "last_acl_time_s": max(acl_times, default=None),
        "pcap_end_time_s": end_s,
    }


def summarize_session(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = run_dir / "session_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    pcap = run_dir / "monitor" / "monitor.pcapng"
    inventory = pcap_inventory(pcap) if pcap.is_file() else {
        "pcap_bytes": 0, "pcap_frame_count": 0, "pcap_span_s": 0.0,
        "connect_aa": [], "crc_valid_acl_packet_count": 0,
        "reconstructed_acl_event_count": 0, "last_5s_acl_packet_count": 0,
        "last_acl_time_s": None, "pcap_end_time_s": 0.0,
    }
    peripheral = last_boot_segment(run_dir / "ground_truth/52840_uart.log", "*** Booting")
    central = last_boot_segment(run_dir / "ground_truth/52833_uart.log", "*** Booting")
    tx_records = json_records(peripheral, "PIP_LL_TX_DONE")
    event_records = json_records(peripheral, "PIP_LL_EVENT_DONE")
    done_values = [
        field for record in tx_records
        for field in ("tifs_us", "rx_end_us", "tx_ready_us", "tx_end_us")
        if isinstance(record.get(field), (int, float))
    ]
    timing: dict[str, Any] = {}
    for field in ("tifs_us", "rx_end_us", "tx_ready_us", "tx_end_us"):
        timing[field] = stats([
            float(record[field]) for record in tx_records
            if isinstance(record.get(field), (int, float))
        ])
    disconnect_count = len(re.findall(r"HRS_DISCONNECTED|Disconnected", peripheral + central, re.IGNORECASE))
    assertion_count = len(re.findall(r"assert|HardFault|LL_ASSERT", peripheral + central, re.IGNORECASE))
    duration_s = float(manifest.get("duration_s", 0.0) or 0.0)
    reasons: list[str] = []
    if duration_s < 30.0:
        reasons.append("duration_below_30s")
    if inventory["last_5s_acl_packet_count"] <= 0:
        reasons.append("no_acl_in_last_5s")
    if inventory["crc_valid_acl_packet_count"] < 500:
        reasons.append("acl_packet_count_below_500")
    if len(tx_records) < 20:
        reasons.append("tx_done_below_20")
    if len(event_records) != len(tx_records):
        reasons.append("tx_event_done_mismatch")
    if disconnect_count:
        reasons.append("disconnect")
    if assertion_count:
        reasons.append("assertion")
    # A reliable attempt->event mapping requires an explicit clock/channel
    # alignment stage.  This report does not infer it from channel collisions.
    reasons.append("pip_attempt_mapping_not_computed")
    row: dict[str, Any] = {
        "session": run_dir.name,
        "run_dir": str(run_dir),
        "duration_s": duration_s,
        **inventory,
        "pip_tx_records": len(json_records(peripheral, "HRS_PIP_TX")),
        "pip_tx_done": len(tx_records),
        "pip_event_done": len(event_records),
        "disconnect_count": disconnect_count,
        "assertion_count": assertion_count,
        "mapping_coverage": None,
        "eligible_pip_event_count": None,
        "accepted": False,
        "failure_reasons": ";".join(reasons),
    }
    controller = {
        "session": run_dir.name,
        "run_dir": str(run_dir),
        "completed_tx": len(tx_records),
        "event_done": len(event_records),
        "disconnect_count": disconnect_count,
        "assertion_count": assertion_count,
    }
    for field, values in timing.items():
        controller[f"{field}_median"] = values["median"]
        controller[f"{field}_mad"] = values["mad"]
        controller[f"{field}_p95"] = values["p95"]
    return row, controller


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sessions: list[dict[str, Any]] = []
    controllers: list[dict[str, Any]] = []
    for run_dir in args.run_dir:
        session, controller = summarize_session(run_dir.expanduser().resolve())
        sessions.append(session)
        controllers.append(controller)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    session_csv = args.output_dir / "pip_timing_session_summary.csv"
    controller_csv = args.output_dir / "pip_controller_timing_summary.csv"
    manifest_path = args.output_dir / "pip_timing_measurement_manifest.json"
    write_csv(session_csv, sessions)
    write_csv(controller_csv, controllers)
    measurement = {
        "schema_version": 1,
        "purpose": "PIP supplemental sparse-event timing and eligibility audit; not ROC",
        "policy": {
            "event_counter_continuity_required": False,
            "event_unit": "eligible UART-led PIP-active event",
            "preflight_min_crc_valid_acl_packets": 500,
            "preflight_min_tx_done": 20,
            "mapping_coverage_required": 0.8,
            "mapping_status": "not_computed_by_this_audit",
        },
        "sessions": sessions,
        "outputs": {
            "session_summary": str(session_csv),
            "controller_summary": str(controller_csv),
        },
    }
    manifest_path.write_text(json.dumps(measurement, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "session_count": len(sessions),
        "accepted_count": sum(bool(row["accepted"]) for row in sessions),
        "outputs": {"session_csv": str(session_csv), "controller_csv": str(controller_csv), "manifest": str(manifest_path)},
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
