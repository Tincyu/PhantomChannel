#!/usr/bin/env python3
"""Compute an exploratory PIP timing ROC/AUC from observed pcap TIFS.

The primary timing measurement is Nordic Sniffer ``nordic_ble.delta_time``
between one eligible direction-changing pair per role and connection event.
This is the over-the-air inter-frame timing, not the controller UART
``PIP_LL_TX_DONE.tifs_us`` transition parameter (which can differ by PHY).

PIP sessions are reported in two scopes:

* ``all_observed_pip_pcap``: every PIP-mode session with recoverable pcap
  pairs, including sessions without UART PIP completion evidence;
* ``clean_uart_pip``: only sessions with a matched TX_DONE/EVENT_DONE ledger
  and no disconnect/assertion in the final boot segment.

The clean result is the useful PIP-positive diagnostic, but it remains
exploratory because the pcap does not map every UART PIP attempt to a decoded
packet.  Both pair-pooled and session-median AUC are emitted so that the
large matched benign pcap cannot masquerade as independent repetitions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable

from analyze_event_timing_detector import ADV_AA, ConnEvent, tshark_rows, connection_events_from_rows


FIELDS = [
    "frame.number", "frame.time_epoch", "btle.access_address",
    "nordic_ble.event_counter", "nordic_ble.channel", "nordic_ble.direction",
    "nordic_ble.packet_counter", "nordic_ble.delta_time", "nordic_ble.crcok",
    "btle.data_header.llid", "btle.data_header.length",
]
FEATURES = ("tifs_us", "abs_deviation_from_benign_median_us")
PAIR_FILTERS = ("all_pairs", "both_empty_packets", "at_least_one_nonempty_packet")


def finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def last_boot_segment(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    marker = "*** Booting"
    index = text.rfind(marker)
    return text[index:] if index >= 0 else text


def uart_counts(run_dir: Path) -> dict[str, int | bool]:
    text = last_boot_segment(run_dir / "ground_truth/52840_uart.log")
    tx_done = len(re.findall(r"PIP_LL_TX_DONE\s+\{", text))
    event_done = len(re.findall(r"PIP_LL_EVENT_DONE\s+\{", text))
    disconnect = len(re.findall(r"HRS_DISCONNECTED|Disconnected", text, re.IGNORECASE))
    assertion = len(re.findall(r"assert|HardFault|LL_ASSERT", text, re.IGNORECASE))
    return {
        "pip_tx_done": tx_done,
        "pip_event_done": event_done,
        "disconnect": disconnect,
        "assertion": assertion,
        "clean_uart_pip": tx_done >= 20 and event_done == tx_done and not disconnect and not assertion,
    }


def eligible_pairs(event: ConnEvent) -> Iterable[dict[str, Any]]:
    ordered = sorted(event.packets, key=lambda packet: (packet.timestamp_us, packet.frame))
    seen_roles: set[str] = set()
    for left, right in zip(ordered, ordered[1:]):
        if left.packet_counter is None or right.packet_counter is None:
            continue
        if ((left.packet_counter + 1) & 0xFFFF) != right.packet_counter:
            continue
        if left.direction == right.direction or not left.direction or not right.direction:
            continue
        if left.direction == "C2P" and right.direction == "P2C":
            role = "central-request-response"
        elif left.direction == "P2C" and right.direction == "C2P":
            role = "peripheral-notification-response"
        else:
            continue
        if role in seen_roles:
            continue
        tifs = right.gap_us
        source = "nordic_ble.delta_time"
        if tifs is None:
            tifs = right.timestamp_us - left.timestamp_us
            source = "timestamp_delta_fallback"
        if tifs is None or tifs <= 0:
            continue
        seen_roles.add(role)
        yield {
            "event_counter": event.event_counter,
            "role": role,
            "frame_prev": left.frame,
            "frame_next": right.frame,
            "packet_counter_prev": left.packet_counter,
            "packet_counter_next": right.packet_counter,
            "tifs_us": float(tifs),
            "measurement_source": source,
            "timestamp_delta_us": right.timestamp_us - left.timestamp_us,
            "left_length": left.length,
            "right_length": right.length,
            "left_llid": left.llid,
            "right_llid": right.llid,
            "both_empty_packets": left.length == 0 and right.length == 0,
        }


def load_run(path: Path, label: str, mode: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = tshark_rows(path, FIELDS)
    events, pcap_duration_s, selected_packet_count, selected_aa = connection_events_from_rows(rows, "")
    pairs: list[dict[str, Any]] = []
    for event in events:
        for pair in eligible_pairs(event):
            pairs.append({
                "run_id": path.parent.parent.name,
                "run_dir": str(path.parent.parent),
                "label": label,
                "mode": mode,
                "selected_access_address": selected_aa,
                **pair,
            })
    uart = uart_counts(path.parent.parent) if mode == "pip" else {
        "pip_tx_done": 0, "pip_event_done": 0, "disconnect": 0,
        "assertion": 0, "clean_uart_pip": False,
    }
    session = {
        "run_id": path.parent.parent.name,
        "run_dir": str(path.parent.parent),
        "mode": mode,
        "label": label,
        "pcap": str(path),
        "selected_access_address": selected_aa,
        "pcap_duration_s": pcap_duration_s,
        "selected_packet_count": selected_packet_count,
        "reconstructed_event_count": len(events),
        "tifs_pair_count": len(pairs),
        "tifs_median_us": statistics.median([pair["tifs_us"] for pair in pairs]) if pairs else None,
        "tifs_values_us": sorted({pair["tifs_us"] for pair in pairs}),
        **uart,
    }
    return session, pairs


def auc(negative: list[float], positive: list[float]) -> float | None:
    if not negative or not positive:
        return None
    total = len(negative) * len(positive)
    rank = sum(pos > neg for pos in positive for neg in negative)
    rank += 0.5 * sum(pos == neg for pos in positive for neg in negative)
    return rank / total


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


def compute_auc_rows(
    sessions: list[dict[str, Any]], pairs: list[dict[str, Any]], pip_scope: str,
) -> list[dict[str, Any]]:
    benign = [session for session in sessions if session["mode"] == "benign"]
    pip = [session for session in sessions if session["mode"] == "pip" and (
        pip_scope == "all_observed_pip_pcap" or session["clean_uart_pip"]
    )]
    benign_ids = {session["run_id"] for session in benign}
    pip_ids = {session["run_id"] for session in pip}
    output: list[dict[str, Any]] = []
    benign_center = statistics.median([
        pair["tifs_us"] for pair in pairs if pair["run_id"] in benign_ids
    ]) if benign_ids else None

    for pair_filter in PAIR_FILTERS:
        benign_pairs = [
            pair for pair in pairs if pair["run_id"] in benign_ids
            and (pair_filter == "all_pairs" or (
                pair_filter == "both_empty_packets" and pair["both_empty_packets"]
            ) or (
                pair_filter == "at_least_one_nonempty_packet" and not pair["both_empty_packets"]
            ))
        ]
        pip_pairs = [
            pair for pair in pairs if pair["run_id"] in pip_ids
            and (pair_filter == "all_pairs" or (
                pair_filter == "both_empty_packets" and pair["both_empty_packets"]
            ) or (
                pair_filter == "at_least_one_nonempty_packet" and not pair["both_empty_packets"]
            ))
        ]
        for sample_unit in ("pair-pooled", "session-median"):
            if sample_unit == "pair-pooled":
                negative_rows = benign_pairs
                positive_rows = pip_pairs
            else:
                negative_rows = []
                positive_rows = []
                for session in benign:
                    values = [pair["tifs_us"] for pair in benign_pairs if pair["run_id"] == session["run_id"]]
                    if values:
                        negative_rows.append({"tifs_us": statistics.median(values), "run_id": session["run_id"]})
                for session in pip:
                    values = [pair["tifs_us"] for pair in pip_pairs if pair["run_id"] == session["run_id"]]
                    if values:
                        positive_rows.append({"tifs_us": statistics.median(values), "run_id": session["run_id"]})
            for feature in FEATURES:
                def score(row: dict[str, Any]) -> float:
                    if feature == "tifs_us":
                        return float(row["tifs_us"])
                    if benign_center is None:
                        return float("nan")
                    return abs(float(row["tifs_us"]) - benign_center)

                negative = [score(row) for row in negative_rows]
                positive = [score(row) for row in positive_rows]
                negative = [value for value in negative if math.isfinite(value)]
                positive = [value for value in positive if math.isfinite(value)]
                output.append({
                    "positive_condition": "PIP",
                    "pip_scope": pip_scope,
                    "pair_filter": pair_filter,
                    "sample_unit": sample_unit,
                    "feature": feature,
                    "score_direction": (
                        "higher_tifs_is_more_anomalous" if feature == "tifs_us"
                        else "farther_from_benign_median_is_more_anomalous"
                    ),
                    "direction_transform": "none",
                    "benign_reference_median_tifs_us": benign_center,
                    "negative_session_count": len({row["run_id"] for row in negative_rows}),
                    "positive_session_count": len({row["run_id"] for row in positive_rows}),
                    "negative_sample_count": len(negative),
                    "positive_sample_count": len(positive),
                    "auc": auc(negative, positive),
                    "interpretation": "exploratory PIP timing AUC; pcap does not map every UART PIP attempt to a decoded packet",
                })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pip-run", type=Path, action="append", required=True)
    parser.add_argument("--benign-run", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    sessions: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for path in args.benign_run:
        session, run_pairs = load_run(path.expanduser().resolve(), "matched-nRF52840-benign", "benign")
        sessions.append(session)
        pairs.extend(run_pairs)
    for path in args.pip_run:
        session, run_pairs = load_run(path.expanduser().resolve(), "PIP", "pip")
        sessions.append(session)
        pairs.extend(run_pairs)

    summary_rows: list[dict[str, Any]] = []
    for scope in ("all_observed_pip_pcap", "clean_uart_pip"):
        summary_rows.extend(compute_auc_rows(sessions, pairs, scope))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    session_path = args.output_dir / "pip_timing_session_summary.csv"
    pair_path = args.output_dir / "pip_timing_tifs_pairs.csv"
    auc_path = args.output_dir / "pip_timing_auc_summary.csv"
    manifest_path = args.output_dir / "pip_timing_auc_manifest.json"
    write_csv(session_path, sessions)
    write_csv(pair_path, pairs)
    write_csv(auc_path, summary_rows)
    manifest = {
        "schema_version": 1,
        "purpose": "exploratory PIP timing ROC/AUC using over-the-air TIFS",
        "tifs_definition": "nordic_ble.delta_time between one direction-changing packet pair per role and event; timestamp delta only as fallback",
        "uart_tifs_policy": "PIP_LL_TX_DONE.tifs_us is retained as controller metadata but is not mixed with over-the-air TIFS",
        "score_direction": "no automatic direction flip",
        "matched_benign_runs": [str(path.resolve()) for path in args.benign_run],
        "pip_runs": [str(path.resolve()) for path in args.pip_run],
        "outputs": {
            "session_summary": str(session_path),
            "tifs_pairs": str(pair_path),
            "auc_summary": str(auc_path),
        },
        "notes": [
            "The clean_uart_pip scope uses only sessions with >=20 matched TX_DONE/EVENT_DONE records and no final-segment disconnect/assertion.",
            "Pair-pooled AUC is descriptive; session-median AUC is the repetition-aware result.",
            "This is not a formal held-out PIP ROC because pcap-to-UART PIP attempt mapping is incomplete.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"sessions": sessions, "auc_summary": summary_rows}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
