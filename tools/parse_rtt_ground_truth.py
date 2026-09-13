#!/usr/bin/env python3
"""Parse Zephyr RTT ground truth for BLE events and PhantomChannel TX records."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


EVENT_RE = re.compile(
    r"^EVT,CONN_EVENT,time_ms=(?P<time_ms>\d+),handle=(?P<handle>\d+),"
    r"event_counter=(?P<event_counter>\d+),channel=(?P<channel>\d+),"
    r"crc_ok=(?P<crc_ok>\d+),crc_error=(?P<crc_error>\d+),"
    r"nak=(?P<nak>\d+),rx_timeout=(?P<rx_timeout>\d+)$"
)
ANCHOR_RE = re.compile(
    r"^EVT,CONN_ANCHOR,time_ms=(?P<time_ms>\d+),handle=(?P<handle>\d+),"
    r"event_counter=(?P<event_counter>\d+),anchor_point_us=(?P<anchor_point_us>\d+)$"
)
PHANTOM_JSON_RE = re.compile(r"^PHANTOM_TX\s+(?P<payload>\{.*\})$")
PHANTOM_KV_RE = re.compile(r"^PHANTOM_TX,(?P<payload>.+)$")
PHANTOM_LL_META_JSON_RE = re.compile(r"^PHANTOM_LL_META\s+(?P<payload>\{.*\})$")
PHANTOM_LL_JSON_RE = re.compile(r"^PHANTOM_LL_TX\s+(?P<payload>\{.*\})$")
PHANTOM_LL_STATS_JSON_RE = re.compile(r"^PHANTOM_LL_STATS\s+(?P<payload>\{.*\})$")
RTT_DROP_RE = re.compile(r"^---\s+(?P<count>\d+) messages dropped ---$")
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


GROUND_TRUTH_FIELDS = [
    "run_id",
    "session_id",
    "seq",
    "conn_event",
    "channel",
    "phy",
    "normal_pdu_len",
    "covert_len_bytes",
    "covert_marker_hex",
    "covert_hex",
    "covert_data_len_bytes",
    "covert_data_hex",
    "rtt_timestamp_us",
]
EVENT_FIELDS = [
    "source_line",
    "time_ms",
    "handle",
    "event_counter_raw",
    "event_counter_unwrapped",
    "counter_contiguous",
    "channel",
    "crc_ok",
    "crc_error",
    "nak",
    "rx_timeout",
]
ANCHOR_FIELDS = [
    "source_line",
    "time_ms",
    "handle",
    "event_counter_raw",
    "anchor_point_us",
]
LL_TX_FIELDS = [
    "run_id",
    "seq",
    "channel",
    "access_address",
    "parser_access_address",
    "crc_init",
    "normal_pdu_len",
    "air_extra_len",
    "covert_len_bytes",
    "covert_marker_hex",
    "covert_pattern",
    "rtt_timestamp_us",
    "source_line",
]


@dataclass
class ParseResult:
    ground_truth_rows: list[dict[str, Any]]
    event_rows: list[dict[str, Any]]
    anchor_rows: list[dict[str, Any]]
    ll_tx_rows: list[dict[str, Any]]
    status: dict[str, Any]


def parse_key_value_payload(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in text.split(","):
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"expected key=value item, got {item!r}")
        key, value = item.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def normalize_hex(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.removeprefix("0x").removeprefix("0X").replace(" ", "")
    if len(text) % 2:
        text = "0" + text
    int(text, 16)
    return text.lower()


def first_present(record: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in record and record[name] not in (None, ""):
            return record[name]
    return ""


def fixed_01_to_ff_data_hex(length: int) -> str:
    return bytes(((index % 255) + 1 for index in range(length))).hex()


def reconstruct_covert_hex(record: dict[str, Any], marker_hex: str, covert_len: Any) -> str:
    pattern = str(first_present(record, "covert_pattern", "payload_pattern"))
    if pattern != "marker_fixed_01_to_ff" or not marker_hex:
        return ""
    try:
        length = int(covert_len)
    except (TypeError, ValueError):
        return ""
    if length < 1:
        return ""
    return marker_hex + fixed_01_to_ff_data_hex(length - 1)


def normalize_phantom_record(record: dict[str, Any], run_id: str, source_line: int) -> dict[str, Any]:
    covert_len = first_present(record, "covert_len_bytes", "covert_len", "payload_len", "covert_payload_len")

    marker_hex = normalize_hex(first_present(record, "covert_marker_hex", "payload_marker_hex"))
    if marker_hex and len(marker_hex) > 2:
        marker_hex = marker_hex[-2:]

    covert_hex = normalize_hex(first_present(record, "covert_hex", "payload_hex", "covert_payload_hex"))
    if not covert_hex:
        covert_hex = reconstruct_covert_hex(record, marker_hex, covert_len)
    if covert_len in ("", None) and covert_hex:
        covert_len = len(covert_hex) // 2

    covert_data_hex = normalize_hex(first_present(record, "covert_data_hex", "payload_data_hex"))
    if not covert_data_hex and marker_hex and covert_hex.startswith(marker_hex):
        covert_data_hex = covert_hex[2:]
    covert_data_len = first_present(record, "covert_data_len_bytes", "covert_data_len", "payload_data_len")
    if covert_data_len in ("", None) and covert_data_hex:
        covert_data_len = len(covert_data_hex) // 2

    timestamp = first_present(record, "rtt_timestamp_us", "timestamp_us", "time_us")
    if timestamp in ("", None):
        time_ms = first_present(record, "time_ms", "rtt_time_ms")
        timestamp = int(time_ms) * 1000 if time_ms not in ("", None) else ""

    row = {
        "run_id": first_present(record, "run_id") or run_id,
        "session_id": first_present(record, "session_id", "conn_handle", "handle"),
        "seq": first_present(record, "seq", "sequence", "covert_seq"),
        "conn_event": first_present(record, "conn_event", "event_counter", "event_counter_unwrapped"),
        "channel": first_present(record, "channel", "data_channel"),
        "phy": first_present(record, "phy", "ble_phy"),
        "normal_pdu_len": first_present(record, "normal_pdu_len", "normal_len", "pdu_len"),
        "covert_len_bytes": covert_len,
        "covert_marker_hex": marker_hex,
        "covert_hex": covert_hex,
        "covert_data_len_bytes": covert_data_len,
        "covert_data_hex": covert_data_hex,
        "rtt_timestamp_us": timestamp,
        "_source_line": source_line,
    }
    return row


def normalize_ll_tx_record(record: dict[str, Any], run_id: str, source_line: int) -> dict[str, Any]:
    covert_len = first_present(record, "covert_len_bytes", "covert_len", "payload_len")
    timestamp = first_present(record, "rtt_timestamp_us", "timestamp_us", "time_us")
    marker_hex = normalize_hex(first_present(record, "covert_marker_hex", "payload_marker_hex"))
    if marker_hex and len(marker_hex) > 2:
        marker_hex = marker_hex[-2:]

    return {
        "run_id": first_present(record, "run_id") or run_id,
        "seq": first_present(record, "seq", "sequence", "covert_seq"),
        "channel": first_present(record, "channel", "data_channel"),
        "access_address": normalize_hex(first_present(record, "access_address")),
        "parser_access_address": normalize_hex(first_present(record, "parser_access_address", "aa_parser")),
        "crc_init": normalize_hex(first_present(record, "crc_init")),
        "normal_pdu_len": first_present(record, "normal_pdu_len", "normal_len", "pdu_len"),
        "air_extra_len": first_present(record, "air_extra_len", "post_crc_len"),
        "covert_len_bytes": covert_len,
        "covert_marker_hex": marker_hex,
        "covert_pattern": first_present(record, "covert_pattern", "payload_pattern"),
        "rtt_timestamp_us": timestamp,
        "source_line": source_line,
    }


def parse_rtt_log(path: Path, run_id: str = "") -> ParseResult:
    ground_truth_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    anchor_rows: list[dict[str, Any]] = []
    ll_tx_rows: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []
    drop_records: list[dict[str, Any]] = []
    ll_meta: dict[str, Any] = {}
    ll_meta_records: list[dict[str, Any]] = []
    ll_stats_records: list[dict[str, Any]] = []

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = ANSI_ESCAPE_RE.sub("", raw_line).strip()
        if not line:
            continue

        match = RTT_DROP_RE.fullmatch(line)
        if match:
            drop_records.append(
                {
                    "source_line": line_number,
                    "kind": "rtt_transport",
                    "count": int(match.group("count")),
                    "text": line,
                }
            )
            continue

        match = EVENT_RE.fullmatch(line)
        if match:
            values = {key: int(value) for key, value in match.groupdict().items()}
            raw_counter = values["event_counter"] & 0xFFFF
            if event_rows:
                previous = event_rows[-1]
                expected = (int(previous["event_counter_raw"]) + 1) & 0xFFFF
                contiguous = values["handle"] == int(previous["handle"]) and raw_counter == expected
                unwrapped = int(previous["event_counter_unwrapped"]) + (
                    (raw_counter - int(previous["event_counter_raw"])) & 0xFFFF
                )
            else:
                contiguous = True
                unwrapped = raw_counter
            event_rows.append(
                {
                    "source_line": line_number,
                    "time_ms": values["time_ms"],
                    "handle": values["handle"],
                    "event_counter_raw": raw_counter,
                    "event_counter_unwrapped": unwrapped,
                    "counter_contiguous": int(contiguous),
                    "channel": values["channel"],
                    "crc_ok": values["crc_ok"],
                    "crc_error": values["crc_error"],
                    "nak": values["nak"],
                    "rx_timeout": values["rx_timeout"],
                }
            )
            continue

        match = ANCHOR_RE.fullmatch(line)
        if match:
            values = {key: int(value) for key, value in match.groupdict().items()}
            anchor_rows.append(
                {
                    "source_line": line_number,
                    "time_ms": values["time_ms"],
                    "handle": values["handle"],
                    "event_counter_raw": values["event_counter"] & 0xFFFF,
                    "anchor_point_us": values["anchor_point_us"],
                }
            )
            continue

        match = PHANTOM_JSON_RE.fullmatch(line)
        if match:
            try:
                record = json.loads(match.group("payload"))
                ground_truth_rows.append(normalize_phantom_record(record, run_id, line_number))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                malformed.append({"source_line": line_number, "reason": str(exc), "text": line})
            continue

        match = PHANTOM_KV_RE.fullmatch(line)
        if match:
            try:
                record = parse_key_value_payload(match.group("payload"))
                ground_truth_rows.append(normalize_phantom_record(record, run_id, line_number))
            except ValueError as exc:
                malformed.append({"source_line": line_number, "reason": str(exc), "text": line})
            continue

        match = PHANTOM_LL_JSON_RE.fullmatch(line)
        if match:
            try:
                record = json.loads(match.group("payload"))
                ll_tx_rows.append(normalize_ll_tx_record({**ll_meta, **record}, run_id, line_number))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                malformed.append({"source_line": line_number, "reason": str(exc), "text": line})
            continue

        match = PHANTOM_LL_META_JSON_RE.fullmatch(line)
        if match:
            try:
                ll_meta = json.loads(match.group("payload"))
                ll_meta_records.append({"source_line": line_number, **ll_meta})
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                malformed.append({"source_line": line_number, "reason": str(exc), "text": line})
            continue

        match = PHANTOM_LL_STATS_JSON_RE.fullmatch(line)
        if match:
            try:
                ll_stats_records.append({"source_line": line_number, **json.loads(match.group("payload"))})
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                malformed.append({"source_line": line_number, "reason": str(exc), "text": line})
            continue

        if line.startswith("EVT,CONN_EVENT_LOG_DROP") or line.startswith("EVT,CONN_ANCHOR_LOG_DROP"):
            drop_records.append({"source_line": line_number, "text": line})
        elif line.startswith("PHANTOM_TX"):
            malformed.append({"source_line": line_number, "reason": "malformed PHANTOM_TX line", "text": line})
        elif line.startswith("PHANTOM_LL_TX"):
            malformed.append({"source_line": line_number, "reason": "malformed PHANTOM_LL_TX line", "text": line})

    ground_truth_by_seq = {
        str(row.get("seq")): row
        for row in ground_truth_rows
        if row.get("seq") not in ("", None)
    }
    for ll_row in ll_tx_rows:
        seq = str(ll_row.get("seq"))
        if not seq or seq in ground_truth_by_seq:
            continue
        if not ll_row.get("covert_marker_hex") or ll_row.get("covert_pattern") != "marker_fixed_01_to_ff":
            continue
        synthesized = normalize_phantom_record(
            {
                "run_id": ll_row.get("run_id", ""),
                "session_id": 0,
                "seq": ll_row.get("seq", ""),
                "channel": ll_row.get("channel", ""),
                "phy": "1M",
                "normal_pdu_len": ll_row.get("normal_pdu_len", ""),
                "covert_len": ll_row.get("covert_len_bytes", ""),
                "covert_marker_hex": ll_row.get("covert_marker_hex", ""),
                "covert_data_len": (
                    int(ll_row["covert_len_bytes"]) - 1
                    if str(ll_row.get("covert_len_bytes", "")).isdigit()
                    else ""
                ),
                "covert_pattern": ll_row.get("covert_pattern", ""),
                "timestamp_us": ll_row.get("rtt_timestamp_us", ""),
            },
            run_id,
            int(ll_row.get("source_line", 0) or 0),
        )
        ground_truth_rows.append(synthesized)
        ground_truth_by_seq[seq] = synthesized

    required_missing = [
        {
            "source_line": row["_source_line"],
            "missing": [
                field
                for field in ("seq", "covert_hex")
                if row.get(field) in ("", None)
            ],
        }
        for row in ground_truth_rows
        if row.get("seq") in ("", None) or row.get("covert_hex") in ("", None)
    ]

    status = {
        "schema_version": 1,
        "input_log": str(path),
        "run_id": run_id,
        "valid": not malformed and not drop_records and not required_missing,
        "phantom_tx_count": len(ground_truth_rows),
        "phantom_ll_tx_count": len(ll_tx_rows),
        "phantom_ll_meta_count": len(ll_meta_records),
        "phantom_ll_stats_count": len(ll_stats_records),
        "controller_ll_stats": ll_stats_records[-1] if ll_stats_records else {},
        "event_count": len(event_rows),
        "anchor_count": len(anchor_rows),
        "drop_record_count": len(drop_records),
        "malformed_count": len(malformed),
        "ground_truth_records_missing_required_fields": required_missing,
        "has_required_phantom_match_fields": bool(ground_truth_rows) and not required_missing,
        "counter_contiguous": all(int(row["counter_contiguous"]) for row in event_rows) if event_rows else None,
        "drop_records": drop_records[:20],
        "malformed_records": malformed[:20],
    }
    return ParseResult(ground_truth_rows, event_rows, anchor_rows, ll_tx_rows, status)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Raw RTT log path.")
    parser.add_argument("--run-id", default="", help="Run id to fill when PHANTOM_TX omits run_id.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for parsed outputs.")
    parser.add_argument(
        "--require-phantom",
        action="store_true",
        help="Exit nonzero if no valid PHANTOM_TX records with seq and covert_hex are present.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f"RTT log does not exist: {args.input}")

    result = parse_rtt_log(args.input.resolve(), args.run_id)
    output_dir = args.output_dir.resolve()

    public_ground_truth_rows = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in result.ground_truth_rows
    ]
    write_csv(output_dir / "rtt_ground_truth.csv", public_ground_truth_rows, GROUND_TRUTH_FIELDS)
    write_json(output_dir / "rtt_ground_truth.json", public_ground_truth_rows)
    write_csv(output_dir / "rtt_events.csv", result.event_rows, EVENT_FIELDS)
    write_csv(output_dir / "rtt_anchors.csv", result.anchor_rows, ANCHOR_FIELDS)
    write_csv(output_dir / "rtt_ll_tx.csv", result.ll_tx_rows, LL_TX_FIELDS)
    write_json(output_dir / "rtt_ll_tx.json", result.ll_tx_rows)
    write_json(output_dir / "rtt_parse_status.json", result.status)

    print(json.dumps(result.status, indent=2, sort_keys=True))
    if args.require_phantom and not result.status["has_required_phantom_match_fields"]:
        return 2
    return 0 if result.status["valid"] or not args.require_phantom else 2


if __name__ == "__main__":
    sys.exit(main())
