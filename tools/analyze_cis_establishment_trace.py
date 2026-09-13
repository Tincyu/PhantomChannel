#!/usr/bin/env python3
"""Extract auditable CIS establishment/timeline evidence from a short run.

The script treats the pcap as authoritative for AA_acl, LL_CIS_REQ/RSP/IND,
AA_cis and CIS parameters.  UART is only ground truth for firmware state and
TX/RX.  An optional X310 parser CSV is searched in both logical and little-
endian/on-air AA representations; a hint alone never counts as an AA hit.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_aa(value: str) -> str:
    text = value.strip().lower().replace("0x", "").replace(" ", "")
    if not re.fullmatch(r"[0-9a-f]{1,8}", text):
        return ""
    return f"0x{int(text, 16):08x}"


def aa_forms(logical: str) -> set[str]:
    normalized = normalize_aa(logical)
    if not normalized:
        return set()
    raw = bytes.fromhex(normalized[2:])
    return {normalized, "0x" + raw[::-1].hex()}


def run_tshark(pcap: Path) -> list[dict[str, str]]:
    fields = [
        "frame.number",
        "frame.time_relative",
        "btle.advertising_address",
        "btle.access_address",
        "btle.control_opcode",
        "btle.data_header.length",
        "btle.control.access_address",
        "btle.control.cis_id",
        "btle.control.cis_offset_min",
        "btle.control.cis_offset_max",
        "btle.control.cis_offset",
        "btle.control.cis_sync_delay",
        "nordic_ble.crcok",
        "_ws.col.Info",
    ]
    command = ["tshark", "-r", str(pcap), "-T", "fields", "-E", "separator=\t", "-E", "quote=d"]
    for field in fields:
        command.extend(["-e", field])
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"tshark failed ({result.returncode}): {result.stderr[-1000:]}")
    rows: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        values = next(csv.reader([line], delimiter="\t", quotechar='"'), [])
        values += [""] * (len(fields) - len(values))
        rows.append(dict(zip(fields, values)))
    return rows


def classify(info: str, opcode: str = "") -> str:
    upper = info.upper()
    if "CONNECT_IND" in upper:
        return "CONNECT_IND"
    for name in ("LL_CIS_REQ", "LL_CIS_RSP", "LL_CIS_IND"):
        if name in upper:
            return name
    # Some tshark/Wireshark combinations expose only the numeric control
    # opcode field, not the Info-column label.
    opcode_name = {"0x1f": "LL_CIS_REQ", "0x20": "LL_CIS_RSP", "0x21": "LL_CIS_IND"}
    if opcode.lower() in opcode_name:
        return opcode_name[opcode.lower()]
    if "ISOCHRONOUS" in upper or "CIS DATA" in upper:
        return "CIS_ACTIVITY"
    return "OTHER"


def parse_pcap(pcap: Path, output_csv: Path) -> dict[str, Any]:
    raw_rows = run_tshark(pcap)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "frame_number", "time_s", "event", "advertising_address", "access_address",
        "control_access_address", "cis_id", "cis_offset_min", "cis_offset_max",
        "cis_offset", "cis_sync_delay", "info",
    ]
    event_rows: list[dict[str, str]] = []
    for row in raw_rows:
        event = classify(row.get("_ws.col.Info", ""), row.get("btle.control_opcode", ""))
        event_rows.append({
            "frame_number": row.get("frame.number", ""),
            "time_s": row.get("frame.time_relative", ""),
            "event": event,
            "advertising_address": row.get("btle.advertising_address", ""),
            "access_address": row.get("btle.access_address", ""),
            "control_access_address": row.get("btle.control.access_address", ""),
            "cis_id": row.get("btle.control.cis_id", ""),
            "cis_offset_min": row.get("btle.control.cis_offset_min", ""),
            "cis_offset_max": row.get("btle.control.cis_offset_max", ""),
            "cis_offset": row.get("btle.control.cis_offset", ""),
            "cis_sync_delay": row.get("btle.control.cis_sync_delay", ""),
            "info": row.get("_ws.col.Info", ""),
        })
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(event_rows)

    connect = [row for row in event_rows if row["event"] == "CONNECT_IND"]
    req = [row for row in event_rows if row["event"] == "LL_CIS_REQ"]
    rsp = [row for row in event_rows if row["event"] == "LL_CIS_RSP"]
    ind = [row for row in event_rows if row["event"] == "LL_CIS_IND"]
    aa_counts: dict[str, int] = {}
    for row in event_rows:
        aa = normalize_aa(row["access_address"])
        if aa:
            aa_counts[aa] = aa_counts.get(aa, 0) + 1
    aa_acl = next((aa for aa in aa_counts if aa not in {"0x8b00a000", "0x8e89bed6"}), "")
    aa_cis = ""
    if ind:
        aa_cis = normalize_aa(ind[0]["control_access_address"])
        if not aa_cis:
            # Wireshark versions differ in where they expose the CIS AA.
            aa_cis = normalize_aa(ind[0]["access_address"])
    ind_time = float(ind[0]["time_s"]) if ind and ind[0]["time_s"] else None
    cis_activity = []
    cis_forms = aa_forms(aa_cis)
    if ind_time is not None and cis_forms:
        for row in event_rows:
            if not row["time_s"]:
                continue
            if float(row["time_s"]) <= ind_time:
                continue
            if normalize_aa(row["access_address"]) in cis_forms:
                cis_activity.append(row)
    return {
        "pcap": str(pcap),
        "frames": len(raw_rows),
        "events_csv": str(output_csv),
        "connect_ind_count": len(connect),
        "ll_cis_req_count": len(req),
        "ll_cis_rsp_count": len(rsp),
        "ll_cis_ind_count": len(ind),
        "aa_acl_from_pcap": aa_acl,
        "aa_cis_from_pcap": aa_cis,
        "aa_cis_on_air_bytes": " ".join(bytes.fromhex(aa_cis[2:])[::-1].hex()[i:i + 2] for i in range(0, 8, 2)) if aa_cis else "",
        "aa_counts": aa_counts,
        "aa_cis_activity_count_in_pcap": len(cis_activity),
        "establishment_complete": bool(connect and req and rsp and ind and aa_cis),
    }


def uart_summary(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    return {
        "path": str(path),
        "exists": path.is_file(),
        "cis_connected_count": len(re.findall(r"CIS_CONNECTED|ISO Channel .*connected", text, re.IGNORECASE)),
        "cis_tx_count": len(re.findall(r"CIS_TX(?:\s|\{)", text)),
        "cis_rx_count": len(re.findall(r"CIS_RX(?:\s|\{)", text)),
        "cis_stats_lines": len(re.findall(r"CIS_STATS", text)),
        "disconnect_count": len(re.findall(r"Disconnected|CIS_DISCONNECTED", text, re.IGNORECASE)),
        "assertion_count": len(re.findall(r"assert|HardFault", text, re.IGNORECASE)),
    }


def iq_summary(
    csv_path: Path,
    aa_cis: str,
    ind_time_s: float | None,
    iq_time_offset_s: float | None,
) -> dict[str, Any]:
    if not csv_path.is_file() or not aa_cis:
        return {"path": str(csv_path), "exists": csv_path.is_file(), "aa_cis_hits": 0, "usable": False}
    forms = aa_forms(aa_cis)
    hits: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            aa = normalize_aa(row.get("access_address", ""))
            if aa not in forms:
                continue
            time_value = row.get("timestamp_s") or row.get("relative_time_s") or ""
            # pcap and X310 parser timestamps have independent zero points.
            # Only apply a temporal gate when the caller supplies a measured
            # pcap-minus-IQ offset.  A raw comparison of the two relative
            # clocks silently discards genuine AA hits.
            if iq_time_offset_s is not None and ind_time_s is not None and time_value:
                try:
                    if float(time_value) + iq_time_offset_s <= ind_time_s:
                        continue
                except ValueError:
                    pass
            hits.append(row)
    return {
        "path": str(csv_path),
        "exists": True,
        "aa_cis_forms_searched": sorted(forms),
        "aa_cis_hits": len(hits),
        "iq_time_offset_s": iq_time_offset_s,
        "time_alignment": "explicit_offset" if iq_time_offset_s is not None else "unlocked",
        "first_hit": hits[0] if hits else None,
        "last_hit": hits[-1] if hits else None,
        "usable": bool(hits),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--aa-cis-hint", default="", help="Search hint only; never treated as pcap-derived evidence.")
    parser.add_argument("--iq-csv", type=Path, default=None, help="Optional BLE parser CSV from X310 IQ.")
    parser.add_argument("--pcap", type=Path, default=None)
    parser.add_argument("--central-uart", type=Path, default=None)
    parser.add_argument("--peripheral-uart", type=Path, default=None)
    parser.add_argument(
        "--iq-time-offset-s",
        type=float,
        default=None,
        help="Measured pcap_time - IQ_time offset; omit because the two capture clocks are otherwise independent.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    pcap = (args.pcap or run_dir / "monitor" / "monitor.pcapng").resolve()
    central_uart = (args.central_uart or run_dir / "ground_truth" / "52833_uart.log").resolve()
    peripheral_uart = (args.peripheral_uart or run_dir / "ground_truth" / "52840_uart.log").resolve()
    events_csv = run_dir / "parsed_events.csv"
    if not pcap.is_file():
        raise SystemExit(f"pcap not found: {pcap}")
    pcap_summary = parse_pcap(pcap, events_csv)
    aa_cis = pcap_summary["aa_cis_from_pcap"]
    ind_rows = []
    with events_csv.open(newline="", encoding="utf-8") as handle:
        ind_rows = [row for row in csv.DictReader(handle) if row["event"] == "LL_CIS_IND"]
    ind_time = float(ind_rows[0]["time_s"]) if ind_rows and ind_rows[0]["time_s"] else None
    iq_result = iq_summary(args.iq_csv.resolve(), aa_cis, ind_time, args.iq_time_offset_s) if args.iq_csv else None
    central_result = uart_summary(central_uart)
    peripheral_result = uart_summary(peripheral_uart)
    result = {
        "schema_version": 1,
        "pcap": pcap_summary,
        "uart": {
            "central": central_result,
            "peripheral": peripheral_result,
        },
        "iq": iq_result,
        "aa_cis_hint": normalize_aa(args.aa_cis_hint) if args.aa_cis_hint else "",
        "decision": {
            "cis_establishment_context": bool(pcap_summary["establishment_complete"]),
            "aa_cis_activity_in_pcap": pcap_summary["aa_cis_activity_count_in_pcap"] > 0,
            "aa_cis_activity_in_x310_csv": bool(iq_result and iq_result.get("usable")),
            "uart_cis_ground_truth": bool(
                peripheral_result["cis_connected_count"]
                and (central_result["cis_tx_count"] or peripheral_result["cis_rx_count"])
            ),
        },
        "notes": [
            "AA_cis and LL_CIS parameters are accepted only from pcap dissection.",
            "The AA hint is recorded for search provenance and is never sufficient for a pass.",
            "X310 byte-order search accepts both logical AA and reversed on-air octets.",
            "pcap and X310 relative timestamps are not compared unless --iq-time-offset-s is supplied.",
        ],
    }
    # Avoid recomputing or leaking walrus-only locals into the serialized result.
    result["decision"]["aa_cis_activity_in_x310_csv"] = bool(result["iq"] and result["iq"].get("usable"))
    result["decision"]["uart_cis_ground_truth"] = bool(
        result["uart"]["peripheral"]["cis_connected_count"]
        and (result["uart"]["central"]["cis_tx_count"] or result["uart"]["peripheral"]["cis_rx_count"])
    )
    write_json(run_dir / "cis_trace_summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    required = (
        result["decision"]["cis_establishment_context"]
        and result["decision"]["aa_cis_activity_in_x310_csv"]
        and result["decision"]["uart_cis_ground_truth"]
    )
    return 0 if required else 2


if __name__ == "__main__":
    sys.exit(main())
