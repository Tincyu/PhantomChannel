#!/usr/bin/env python3
"""Create per-session ledgers for the matched PIP boundary experiment.

The ledger is deliberately derived from each session's UART record and parser
output.  It does not hard-code connection access addresses or PIP mapping
values.  Formal sessions are selected explicitly by their ``rep1``--``rep5``
names; a separate invocation can prepare the benign calibration session.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.pip_x310_audit import (  # noqa: E402
    aa_canonical,
    aa_from_parser_value,
    inverse_map_fake_access_address,
)


FORMAL_PREFIXES = ("pip_auc_benign_rep", "pip_auc_direct_tail_rep", "pip_auc_pip_rep")


def uart_events(path: Path) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split(" ", 1)
        if len(parts) != 2:
            continue
        try:
            value = json.loads(parts[1])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append((parts[0], value))
    return events


def parser_real_aa(parser_csv: Path) -> str:
    with parser_csv.open(newline="", encoding="utf-8", errors="replace") as handle:
        rows = csv.DictReader(handle)
        counts: Counter[str] = Counter(
            str(row.get("access_address", "")).strip().upper()
            for row in rows
            if str(row.get("packet_type", "")).upper() == "BLE_CONN"
            and str(row.get("access_address", "")).strip()
        )
    if not counts:
        raise RuntimeError(f"no BLE_CONN access address in {parser_csv}")
    value, _count = counts.most_common(1)[0]
    raw = aa_from_parser_value(value)
    if len(raw) != 4:
        raise RuntimeError(f"invalid parser access address {value!r} in {parser_csv}")
    return aa_canonical(raw)


def build_ledger(session_dir: Path) -> dict[str, Any]:
    session_id = session_dir.name
    ground_truth = session_dir / "ground_truth"
    peripheral = ground_truth / "peripheral_uart.log"
    if not peripheral.is_file():
        raise RuntimeError(f"missing peripheral UART log: {peripheral}")
    events = uart_events(peripheral)
    previous_ledger_path = session_dir / "sdr/pip_auc_ledger.json"
    previous_ledger: dict[str, Any] = {}
    if previous_ledger_path.is_file():
        try:
            loaded = json.loads(previous_ledger_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                previous_ledger = loaded
        except json.JSONDecodeError:
            previous_ledger = {}

    if session_id.startswith("pip_auc_benign_") or session_id == "writegate_spb1m_pip_auc_pilot_benign":
        condition = "benign"
        tx_events = [payload for tag, payload in events if tag == "HRS_TX"]
        # Use the top-level merged CSV.  The stage2 CSV is a local-window
        # artifact whose sample indices have not been remapped to the full IQ
        # file; using it for feature extraction shifts every post-boundary
        # window by stage2_source_start_sample.
        parser_csv = session_dir / "sdr/two_stage_known_aa/ble_packets.csv"
    elif session_id.startswith("pip_auc_direct_tail_"):
        condition = "direct_tail"
        tx_events = [payload for tag, payload in events if tag == "HRS_TX"]
        parser_csv = session_dir / "sdr/two_stage_known_aa/ble_packets.csv"
    elif session_id.startswith("pip_auc_pip_"):
        condition = "pip"
        tx_events = [payload for tag, payload in events if tag == "PIP_LL_TX_DONE"]
        parser_candidates = (
            session_dir / "sdr/two_stage_known_fake_aa_patched_v2/ble_packets.csv",
            session_dir / "sdr/two_stage_known_fake_aa_patched/ble_packets.csv",
            session_dir / "sdr/two_stage_known_fake_aa/ble_packets.csv",
        )
        parser_csv = next((path for path in parser_candidates if path.is_file()), parser_candidates[-1])
    else:
        raise RuntimeError(f"unrecognized session name: {session_id}")

    if not tx_events:
        raise RuntimeError(f"no expected transmit events in {peripheral}")
    if not parser_csv.is_file():
        raise RuntimeError(f"missing parser CSV: {parser_csv}")

    ledger: dict[str, Any] = {
        "schema_version": 1,
        "session_id": session_id,
        "condition": condition,
        "phy": "2m",
        "tx_done_count": len(tx_events),
        "tx_count": len(tx_events),
        "measured_snr_db": "",
        "ledger_source": str(peripheral),
        "parser_csv": str(parser_csv),
    }

    if condition == "pip":
        fake_values = {
            str(payload.get("fake_access_address", "")).strip()
            for payload in tx_events
            if payload.get("fake_access_address")
        }
        crc_values = {
            str(payload.get("crc_init", "")).strip()
            for payload in tx_events
            if payload.get("crc_init")
        }
        if len(fake_values) != 1:
            raise RuntimeError(f"expected one fake AA in {peripheral}, got {sorted(fake_values)}")
        if len(crc_values) != 1:
            raise RuntimeError(f"expected one CRCInit in {peripheral}, got {sorted(crc_values)}")
        fake_raw = aa_from_parser_value(next(iter(fake_values)))
        if len(fake_raw) != 4:
            raise RuntimeError(f"invalid fake AA in {peripheral}: {fake_values}")
        ledger.update(
            {
                "fake_aa": aa_canonical(fake_raw),
                "fake_aa_raw": fake_raw.hex(),
                "real_aa": aa_canonical(inverse_map_fake_access_address(fake_raw)),
                "crc_init": next(iter(crc_values)),
                "fake_aa_source": str(peripheral),
            }
        )
    else:
        # Preserve the AA selected from the prior ledger when available.  The
        # merged CSV contains all connection traffic; its most-common AA is
        # not necessarily the application connection AA.
        ledger["real_aa"] = previous_ledger.get("real_aa") or parser_real_aa(parser_csv)
        ledger["real_aa_source"] = str(parser_csv)
        # The formal benign/direct application packet is the 9-byte HRS
        # notification (the same real_len recorded by the PIP controller
        # ledger).  Empty LL ACKs share the connection AA but are not target
        # application transmissions and must not enter the boundary ROC.
        ledger["target_payload_len"] = 9
        ledger["target_payload_len_source"] = "HRS notification contract"

    output = session_dir / "sdr/pip_auc_ledger.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return ledger


def write_index(ledgers: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for ledger in ledgers for key in ledger})
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ledgers)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions-root", type=Path, required=True)
    parser.add_argument("--session", action="append", type=Path, default=[])
    parser.add_argument("--output-index", type=Path, required=True)
    args = parser.parse_args()

    if args.session:
        session_dirs = args.session
    else:
        session_dirs = sorted(
            path
            for path in args.sessions_root.iterdir()
            if path.is_dir() and any(path.name.startswith(prefix) for prefix in FORMAL_PREFIXES)
            and path.name.rsplit("_", 1)[-1] in {"rep1", "rep2", "rep3", "rep4", "rep5"}
        )
    ledgers = [build_ledger(path) for path in session_dirs]
    write_index(ledgers, args.output_index)
    print(json.dumps({"sessions": len(ledgers), "output_index": str(args.output_index)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
