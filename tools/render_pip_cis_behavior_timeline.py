#!/usr/bin/env python3
"""Render the final three-row ACL/CIS/PIP behavioral view from real captures.

The parser intentionally delegates BLE dissection to tshark.  Application UART
logs may annotate a session, but they never create LL_CIS_REQ/RSP/IND or AA_cis
events.  The main output contains only three real traces: normal ACL, legitimate
CIS with partial passive context, and PIP.  An optional CIS analysis JSON records
X310 AA_cis activity as evidence metadata; its independent clock is not placed
on the pcap timeline unless an explicit alignment is supplied elsewhere.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


FIELDS = (
    "frame.number",
    "frame.time_epoch",
    "btle.access_address",
    "btle.control_opcode",
    "btle.control.access_address",
)

CONTROL_LABELS = {
    # Bluetooth LE Link Layer CIS control opcodes used by the current
    # Wireshark/nRF Connect SDK stack.  Keep the values explicit because the
    # older draft mapping (0x1a/0x1b/0x1c) would collide with other controls.
    0x1F: "LL_CIS_REQ",
    0x20: "LL_CIS_RSP",
    0x21: "LL_CIS_IND",
}


def _int_field(value: str):
    value = (value or "").strip()
    if not value:
        return None
    try:
        return int(value, 0)
    except ValueError:
        try:
            return int(value, 16)
        except ValueError:
            return None


def tshark_rows(path: Path):
    command = [
        "tshark",
        "-r",
        str(path),
        "-T",
        "fields",
        "-E",
        "separator=\t",
        "-E",
        "quote=d",
        "-E",
        "occurrence=f",
    ]
    for field in FIELDS:
        command.extend(("-e", field))
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"tshark failed for {path}: {result.stderr.strip()}")

    rows = []
    for values in csv.reader(result.stdout.splitlines(), delimiter="\t", quotechar='"'):
        if len(values) < len(FIELDS):
            values += [""] * (len(FIELDS) - len(values))
        frame = _int_field(values[0])
        try:
            timestamp = float(values[1])
        except ValueError:
            continue
        rows.append(
            {
                "frame": frame,
                "timestamp": timestamp,
                "aa": _int_field(values[2]),
                "opcode": _int_field(values[3]),
                "control_aa": _int_field(values[4]),
            }
        )
    return rows


def load_config(path: Path | None):
    if path is None:
        return {}
    with path.open() as stream:
        return json.load(stream)


def configured_int(config, key):
    value = config.get(key)
    if value is None:
        return None
    return _int_field(str(value))


def configured_set(config, key):
    return {
        value
        for item in config.get(key, [])
        if (value := _int_field(str(item))) is not None
    }


def most_common_address(rows):
    counts = Counter(row["aa"] for row in rows if row["aa"] is not None)
    return counts.most_common(1)[0][0] if counts else None


def event(trace, row, label, source, view="real"):
    return {
        "trace": trace,
        "label": label,
        "event": label,
        "frame": row["frame"],
        "timestamp_epoch": row["timestamp"],
        "access_address": "" if row["aa"] is None else f"0x{row['aa']:08x}",
        "source": source,
        "view": view,
    }


def parse_acl(rows, config):
    acl_aa = configured_int(config, "acl_access_address") or most_common_address(rows)
    selected = [row for row in rows if acl_aa is not None and row["aa"] == acl_aa]
    return [event("ACL", row, "AA_acl", "tshark:btle.access_address") for row in selected], acl_aa


def parse_cis(rows, config):
    acl_aa = configured_int(config, "acl_access_address")
    cis_aa = configured_int(config, "cis_access_address")
    events = []
    for row in rows:
        label = CONTROL_LABELS.get(row["opcode"])
        if label:
            events.append(event("CIS-partial-context", row, label, "tshark:btle.control_opcode"))
            if label == "LL_CIS_IND" and row["control_aa"] is not None:
                cis_aa = row["control_aa"]
        elif cis_aa is not None and row["aa"] == cis_aa:
            events.append(event("CIS-partial-context", row, "AA_cis", "tshark:btle.access_address"))
        elif acl_aa is not None and row["aa"] == acl_aa:
            events.append(event("CIS-partial-context", row, "AA_acl", "tshark:btle.access_address"))
    return events, acl_aa, cis_aa


def parse_pip(rows, config, acl_aa, cis_aa):
    fake_aas = configured_set(config, "fake_access_addresses")
    if not fake_aas:
        fake_aas = {
            row["aa"]
            for row in rows
            if row["aa"] is not None and row["aa"] not in {acl_aa, cis_aa}
        }
    source = "tshark:btle.access_address"
    if not configured_set(config, "fake_access_addresses"):
        source += ":heuristic-extra-aa"
    events = []
    for row in rows:
        if row["aa"] in {acl_aa, cis_aa}:
            label = "AA_acl" if row["aa"] == acl_aa else "AA_cis"
            events.append(event("PIP", row, label, source))
        elif row["aa"] in fake_aas:
            events.append(event("PIP", row, "AA_f/H_f", source))
    return events


def add_relative_time(events):
    if not events:
        return
    origin = min(item["timestamp_epoch"] for item in events)
    for item in events:
        item["t_rel_s"] = item["timestamp_epoch"] - origin


def write_events(path: Path, events):
    fields = [
        "trace",
        "view",
        "event",
        "frame",
        "timestamp_epoch",
        "t_rel_s",
        "access_address",
        "source",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: item.get(key, "") for key in fields} for item in events)


def write_table(path: Path, events_by_trace):
    rows = [
        ("ACL", "Normal ACL", "No", "—", "No", "No"),
        (
            "CIS-partial-context",
            "Legitimate CIS with partial passive context",
            "Yes",
            "Yes",
            "Yes",
            "No",
        ),
        ("PIP", "PIP", "Yes", "No", "Yes", "Yes"),
    ]
    fields = [
        "trace",
        "display_label",
        "extra_aa_header",
        "observed_setup_context",
        "naive_unseen_aa_alarm",
        "unexplained_aa_alarm",
        "event_count",
        "first_event",
        "last_event",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for trace, display_label, extra, context, naive, unexplained in rows:
            events = events_by_trace.get(trace, [])
            writer.writerow(
                {
                    "trace": trace,
                    "display_label": display_label,
                    "extra_aa_header": extra,
                    "observed_setup_context": context,
                    "naive_unseen_aa_alarm": naive,
                    "unexplained_aa_alarm": unexplained,
                    "event_count": len(events),
                    "first_event": events[0]["event"] if events else "",
                    "last_event": events[-1]["event"] if events else "",
                }
            )


def render_plot(path: Path, events_by_trace, cis_evidence=None):
    order = ["ACL", "CIS-partial-context", "PIP"]
    display_labels = {
        "ACL": "Normal ACL",
        "CIS-partial-context": "Legitimate CIS with partial passive context",
        "PIP": "PIP",
    }
    fig, axes = plt.subplots(len(order), 1, figsize=(13, 7), sharex=False)
    if len(order) == 1:
        axes = [axes]
    colors = {
        "AA_acl": "#2563eb",
        "LL_CIS_REQ": "#dc2626",
        "LL_CIS_RSP": "#ea580c",
        "LL_CIS_IND": "#ca8a04",
        "AA_cis": "#16a34a",
        "AA_f/H_f": "#9333ea",
    }
    for axis, trace in zip(axes, order):
        events = events_by_trace.get(trace, [])
        times = [item["t_rel_s"] for item in events]
        axis.axhline(0, color="#9ca3af", linewidth=0.7)
        if times:
            # Keep every real event as a point, but annotate only the small
            # set of protocol markers that make the three-row figure readable.
            # The complete event stream remains in the CSV.
            key_labels = {"LL_CIS_REQ", "LL_CIS_RSP", "LL_CIS_IND", "AA_cis", "AA_f/H_f"}
            key_seen: dict[str, int] = {}
            last_index = len(events) - 1
            for index, item in enumerate(events):
                x = item["t_rel_s"]
                axis.scatter([x], [0], s=25, color=colors.get(item["event"], "#111827"))
                label = item["event"]
                should_annotate = index in {0, last_index} or label in key_labels
                if label in key_labels:
                    key_seen[label] = key_seen.get(label, 0) + 1
                    # For repeated fake-AA packets, show only the first and
                    # last occurrence; control events are few and all remain.
                    if label == "AA_f/H_f" and key_seen[label] > 1 and index != last_index:
                        should_annotate = False
                if should_annotate:
                    axis.annotate(
                        label,
                        (x, 0),
                        xytext=(0, 8),
                        textcoords="offset points",
                        rotation=45,
                        ha="left",
                        fontsize=7,
                    )
            if trace == "CIS-partial-context" and cis_evidence:
                hits = cis_evidence.get("x310_aa_cis_hits", 0)
                aa = cis_evidence.get("aa_cis_from_pcap", "")
                alignment = cis_evidence.get("x310_time_alignment", "unlocked")
                axis.text(
                    0.01,
                    0.16,
                    f"X310 AA_cis activity: {hits} hits ({aa}, clock {alignment}); RSP not observed",
                    transform=axis.transAxes,
                    fontsize=8,
                    color="#166534",
                    ha="left",
                    va="bottom",
                )
            axis.set_xlim(min(times) - 0.02, max(times) + 0.02 if max(times) > min(times) else 1)
        else:
            axis.text(0.5, 0.5, "no parsed events", transform=axis.transAxes, ha="center")
            axis.set_xlim(0, 1)
        axis.set_yticks([])
        axis.set_ylabel(display_labels[trace], rotation=0, ha="right", va="center", labelpad=55)
        axis.grid(axis="x", alpha=0.2)
    axes[-1].set_xlabel("relative time from first retained event (s)")
    fig.suptitle("Normal ACL / Legitimate CIS partial context / PIP")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acl-pcap", type=Path, required=True)
    parser.add_argument("--cis-pcap", type=Path, required=True)
    parser.add_argument("--pip-pcap", type=Path, required=True)
    parser.add_argument(
        "--cis-summary",
        type=Path,
        help="Optional analyze_cis_establishment_trace.py JSON; records X310 AA_cis evidence without clock merging.",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)

    acl_events, acl_aa = parse_acl(tshark_rows(args.acl_pcap), config)
    cis_events, cis_acl_aa, cis_aa = parse_cis(tshark_rows(args.cis_pcap), config)
    acl_aa = configured_int(config, "acl_access_address") or acl_aa or cis_acl_aa
    pip_events = parse_pip(tshark_rows(args.pip_pcap), config, acl_aa, cis_aa)
    events = acl_events + cis_events + pip_events
    events_by_trace = {}
    for item in events:
        events_by_trace.setdefault(item["trace"], []).append(item)
    for trace_events in events_by_trace.values():
        trace_events.sort(key=lambda item: item["timestamp_epoch"])
        # Each capture is an independent session.  Keep the timeline relative
        # within each row; never subtract the epoch of another session.
        add_relative_time(trace_events)

    cis_evidence = {}
    if args.cis_summary:
        with args.cis_summary.open(encoding="utf-8") as stream:
            cis_summary = json.load(stream)
        iq = cis_summary.get("iq") or {}
        cis_evidence = {
            "analysis_json": str(args.cis_summary),
            "aa_cis_from_pcap": (cis_summary.get("pcap") or {}).get("aa_cis_from_pcap", ""),
            "aa_cis_on_air_bytes": (cis_summary.get("pcap") or {}).get("aa_cis_on_air_bytes", ""),
            "x310_aa_cis_hits": iq.get("aa_cis_hits", 0),
            "x310_aa_cis_forms_searched": iq.get("aa_cis_forms_searched", []),
            "x310_time_alignment": iq.get("time_alignment", "unlocked"),
            "ll_cis_rsp_observed": (cis_summary.get("pcap") or {}).get("ll_cis_rsp_count", 0) > 0,
        }
    write_events(args.out_dir / "pip_cis_behavior_events.csv", events)
    write_table(args.out_dir / "pip_cis_behavior_table.csv", events_by_trace)
    render_plot(args.out_dir / "pip_cis_behavior_timeline.png", events_by_trace, cis_evidence)
    summary = {
        "inputs": {"acl": str(args.acl_pcap), "cis": str(args.cis_pcap), "pip": str(args.pip_pcap)},
        "config": str(args.config) if args.config else None,
        "access_addresses": {"acl": acl_aa, "cis": cis_aa},
        "event_counts": {key: len(value) for key, value in events_by_trace.items()},
        "main_traces": [
            "Normal ACL",
            "Legitimate CIS with partial passive context",
            "PIP",
        ],
        "cis_evidence": cis_evidence,
        "ll_cis_rsp_policy": "not observed is reported transparently; no synthetic marker or cross-session merge",
    }
    (args.out_dir / "pip_cis_behavior_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
