#!/usr/bin/env python3
"""Summarize one isolated PIP dongle/UART trace.

The pcap is authoritative for the ordinary ACL context and connection event.
The PIP firmware ledger is authoritative for the fake-AA value and for the
fact that the PIP transmitter actually produced packets; UART is not used to
invent an RF packet.  This is a stability/trace-readiness check, not a PIP
ROC or an IQ detection result.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def last_boot_segment(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    marker = "*** Booting"
    return text[text.rfind(marker):] if marker in text else text


def tshark_rows(pcap: Path) -> list[dict[str, str]]:
    fields = [
        "frame.number",
        "frame.time_relative",
        "btle.advertising_address",
        "btle.access_address",
        "_ws.col.Info",
    ]
    command = ["tshark", "-r", str(pcap), "-T", "fields", "-E", "separator=\t", "-E", "quote=d"]
    for field in fields:
        command.extend(("-e", field))
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"tshark failed ({result.returncode}): {result.stderr[-1000:]}")
    rows = []
    for line in result.stdout.splitlines():
        values = next(csv.reader([line], delimiter="\t", quotechar='"'), [])
        values += [""] * (len(fields) - len(values))
        rows.append(dict(zip(fields, values)))
    return rows


def normalize_aa(value: str) -> str:
    value = value.strip().lower().replace("0x", "")
    if not re.fullmatch(r"[0-9a-f]{1,8}", value):
        return ""
    return f"0x{int(value, 16):08x}"


def pcap_summary(pcap: Path, target: str) -> dict[str, Any]:
    rows = tshark_rows(pcap)
    target = target.lower()
    target_rows = [row for row in rows if row["btle.advertising_address"].lower() == target]
    info = [row["_ws.col.Info"].upper() for row in rows]
    aa_counts = Counter(normalize_aa(row["btle.access_address"]) for row in rows)
    aa_counts.pop("", None)
    # The advertising AA is the common 0x8e89bed6.  Any other decoded AA is
    # retained as a possible connected-link/fake activity candidate.
    acl_candidates = {aa: count for aa, count in aa_counts.items() if aa != "0x8e89bed6"}
    return {
        "path": str(pcap),
        "frames": len(rows),
        "target_advertising_frames": len(target_rows),
        "connect_ind_count": sum("CONNECT_IND" in item for item in info),
        "acl_access_address_candidates": dict(sorted(acl_candidates.items(), key=lambda item: (-item[1], item[0]))),
        "access_address_counts": dict(aa_counts),
        "decoded_extra_access_addresses": sorted(acl_candidates),
    }


def uart_summary(peripheral: Path, central: Path) -> dict[str, Any]:
    p = last_boot_segment(peripheral)
    c = last_boot_segment(central)
    fake_aas = sorted(set(re.findall(r'"fake_access_address":"?(0x[0-9a-fA-F]+)', p)))
    tx_done = re.findall(r"PIP_LL_TX_DONE\s+\{([^}]*)\}", p)
    event_done = re.findall(r"PIP_LL_EVENT_DONE\s+\{([^}]*)\}", p)
    return {
        "peripheral_path": str(peripheral),
        "central_path": str(central),
        "peripheral_exists": peripheral.is_file(),
        "central_exists": central.is_file(),
        "peripheral_pip_boot": p.count("HRS_PIP_BOOT"),
        "central_pip_boot": c.count("HRS_CENTRAL_BOOT"),
        "acl_connected": bool(re.search(r"HRS_PIP_CONNECTED|HRS_CONNECTED", p)),
        "hrs_subscribed": "HRS_SUBSCRIBE" in c and '"status":0' in c,
        "phy_2m": '"tx_phy":2' in c and '"rx_phy":2' in c,
        "central_notifications": c.count("HRS_NOTIFY {"),
        "pip_tx_records": p.count("HRS_PIP_TX {"),
        "pip_ll_tx_done": len(tx_done),
        "pip_ll_event_done": len(event_done),
        "fake_access_addresses_from_ledger": fake_aas,
        "disconnect_count": len(re.findall(r"HRS_DISCONNECTED|Disconnected", c + p, re.IGNORECASE)),
        "assertion_or_fault_count": len(re.findall(r"assert|HardFault|LL_ASSERT", c + p, re.IGNORECASE)),
        "enomem_count": len(re.findall(r"-ENOMEM|ENOMEM", c + p, re.IGNORECASE)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--target-address", required=True)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    pcap = run_dir / "monitor" / "monitor.pcapng"
    peripheral = run_dir / "ground_truth" / "52840_uart.log"
    central = run_dir / "ground_truth" / "52833_uart.log"
    manifest_path = run_dir / "session_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    pcap_result = pcap_summary(pcap, args.target_address) if pcap.is_file() else {"path": str(pcap), "frames": 0}
    uart_result = uart_summary(peripheral, central)
    follow = manifest.get("follow_sync", {})
    result = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "pcap": pcap_result,
        "uart": uart_result,
        "follow_sync": follow,
        "decision": {
            "dynamic_follow_synchronized": bool(follow.get("device_added") and follow.get("follow_sent") and follow.get("follow_ack")),
            "target_seen": pcap_result.get("target_advertising_frames", 0) > 0,
            "acl_connection_observed": pcap_result.get("connect_ind_count", 0) > 0,
            "pip_uart_ground_truth": bool(
                uart_result["acl_connected"]
                and uart_result["hrs_subscribed"]
                and uart_result["pip_tx_records"]
                and uart_result["pip_ll_tx_done"]
            ),
            "fake_aa_ledger_present": bool(uart_result["fake_access_addresses_from_ledger"]),
            "clean_session": uart_result["assertion_or_fault_count"] == 0 and uart_result["enomem_count"] == 0 and uart_result["disconnect_count"] == 0,
        },
        "notes": [
            "PIP fake AA is accepted as a firmware-ledger value, not as a decoded pcap value.",
            "This result does not claim that the nRF Sniffer decodes the custom PIP waveform.",
            "The pcap and UART are from one isolated run; stale UART prefixes are excluded after the last boot marker.",
        ],
    }
    write_json(run_dir / "pip_trace_summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    required = all(result["decision"][key] for key in ("dynamic_follow_synchronized", "pip_uart_ground_truth", "fake_aa_ledger_present", "clean_session"))
    return 0 if required else 2


if __name__ == "__main__":
    raise SystemExit(main())
