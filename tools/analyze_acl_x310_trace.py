#!/usr/bin/env python3
"""Summarize one isolated ACL pcap/IQ/UART trace."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections import Counter
from pathlib import Path


ADV_AA = "0x8e89bed6"


def normalize(value: str) -> str:
    value = value.strip().lower().replace("0x", "")
    return f"0x{int(value, 16):08x}" if re.fullmatch(r"[0-9a-f]{1,8}", value) else ""


def aa_forms(value: str) -> set[str]:
    value = normalize(value)
    if not value:
        return set()
    raw = bytes.fromhex(value[2:])
    return {value, "0x" + raw[::-1].hex()}


def pcap_access_addresses(path: Path) -> tuple[int, Counter[str]]:
    result = subprocess.run(
        ["tshark", "-r", str(path), "-T", "fields", "-e", "btle.access_address"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise SystemExit(f"tshark failed: {result.stderr[-500:]}")
    values = [normalize(line) for line in result.stdout.splitlines()]
    counts = Counter(value for value in values if value and value != ADV_AA)
    return len(values), counts


def iq_hits(path: Path, aa: str) -> tuple[int, dict[str, str] | None, dict[str, str] | None]:
    forms = aa_forms(aa)
    hits: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            if normalize(row.get("access_address", "")) in forms:
                hits.append(row)
    return len(hits), hits[0] if hits else None, hits[-1] if hits else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--iq-csv", type=Path, default=None)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    pcap = run_dir / "monitor/monitor.pcapng"
    iq_csv = args.iq_csv or run_dir / "sdr/ble_packets.csv"
    central_uart = run_dir / "ground_truth/52833_uart.log"
    peripheral_uart = run_dir / "ground_truth/52840_uart.log"
    frames, counts = pcap_access_addresses(pcap)
    aa_acl, aa_count = counts.most_common(1)[0] if counts else ("", 0)
    hits, first, last = iq_hits(iq_csv, aa_acl) if aa_acl and iq_csv.is_file() else (0, None, None)
    central = central_uart.read_text(encoding="utf-8", errors="replace") if central_uart.is_file() else ""
    peripheral = peripheral_uart.read_text(encoding="utf-8", errors="replace") if peripheral_uart.is_file() else ""
    connected_pattern = r"(?:PIP_CONNECTED|HRS_CONNECTED|Connected:)"
    result = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "pcap": {
            "path": str(pcap),
            "frames_with_field_rows": frames,
            "access_address_counts": dict(counts),
            "aa_acl": aa_acl,
            "aa_acl_pcap_count": aa_count,
        },
        "iq": {
            "path": str(iq_csv),
            "exists": iq_csv.is_file(),
            "aa_acl_forms": sorted(aa_forms(aa_acl)),
            "aa_acl_hits": hits,
            "first_hit": first,
            "last_hit": last,
        },
        "uart": {
            "central_path": str(central_uart),
            "peripheral_path": str(peripheral_uart),
            "central_connected": bool(re.search(connected_pattern, central)),
            "peripheral_connected": bool(re.search(connected_pattern, peripheral)),
            "central_acl_rx_debug_lines": len(re.findall(r"PIP_LL_RX_DEBUG", central)),
            "peripheral_acl_rx_debug_lines": len(re.findall(r"PIP_LL_RX_DEBUG", peripheral)),
            "assertion_or_fault_count": len(re.findall(r"assert|HardFault", central + peripheral, re.IGNORECASE)),
        },
        "decision": {
            "pcap_acl_activity": bool(aa_acl and aa_count),
            "x310_acl_activity": bool(hits),
            "uart_acl_ground_truth": bool(
                re.search(connected_pattern, central)
                and re.search(connected_pattern, peripheral)
            ),
        },
        "notes": [
            "AA_acl is selected from the dominant non-advertising AA in this isolated pcap.",
            "X310 search accepts logical and reversed on-air byte representations.",
            "This is an ACL baseline only; it does not claim PIP fake-AA activity.",
        ],
    }
    output = run_dir / "acl_trace_summary.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all(result["decision"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
