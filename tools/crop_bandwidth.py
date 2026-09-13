#!/usr/bin/env python3
"""Offline passband crop for full-band BLE parser CSVs.

Reads a full-band ``ble_packets.csv`` (plus optional IQ metadata) and writes a
cropped CSV containing only packets whose BLE channel center frequency falls
inside a target passband, or an explicit channel list.  This is the offline
half of the revision C_bw experiment: the full-band capture is cropped at
different bandwidths and re-scored without re-running the SDR parser.

The per-channel packet distribution observed in the full-band capture serves
as the transmitter's effective channel-usage map (no RTT ground truth
available), and C_bw = N_inband / N_tx is reported as estimated.

Outputs:
  <output-dir>/crop_summary.json   per-channel counts, band split, C_bw
  <output-dir>/channel_map.json    channel -> observed center frequency
  <output-csv>                     cropped rows (same schema as input)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable


CENTER_FREQ_DESC_RE = re.compile(r"^\s*([0-9.]+)\s*MHz\s*$", re.IGNORECASE)


def parse_center_freq_hz(desc: str) -> float | None:
    text = str(desc or "").strip()
    match = CENTER_FREQ_DESC_RE.match(text)
    if match:
        return float(match.group(1)) * 1_000_000.0
    try:
        return float(text)
    except ValueError:
        return None


def ble_channel_freq_mhz(channel: int) -> float | None:
    """Fallback BLE channel -> center frequency used by the current parser.

    The parser CSV normally carries ``center_freq_desc`` per row; this table
    only fills rows where that field is empty.  It mirrors the mapping observed
    in the frozen X310 80 MHz parser output: data channels 0..10 at
    2404 + 2*ch MHz, 11..36 shifted +2 MHz (the parser skips 2426 for data),
    and advertising channels 37/38/39 at 2402/2426/2480 MHz.
    """

    if channel == 37:
        return 2402.0
    if channel == 38:
        return 2426.0
    if channel == 39:
        return 2480.0
    if 0 <= channel <= 36:
        base = 2404.0 + 2.0 * channel
        return base + (2.0 if channel >= 11 else 0.0)
    return None


def row_freq_hz(row: dict[str, str]) -> float | None:
    desc_freq = parse_center_freq_hz(row.get("center_freq_desc", ""))
    if desc_freq is not None:
        return desc_freq
    try:
        return ble_channel_freq_mhz(int(row.get("channel", "")))
    except (TypeError, ValueError):
        return None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def passband_filter(center_freq_hz: float, bandwidth_hz: float) -> Callable[[dict[str, str]], bool | None]:
    """Return a row classifier for a symmetric passband around center_freq_hz.

    Returns True (in band), False (out of band), or None (frequency unknown).
    """

    half = abs(bandwidth_hz) / 2.0
    low = center_freq_hz - half
    high = center_freq_hz + half

    def classify(row: dict[str, str]) -> bool | None:
        freq = row_freq_hz(row)
        if freq is None:
            return None
        return low <= freq <= high

    return classify


def channel_list_filter(channels: set[int]) -> Callable[[dict[str, str]], bool | None]:
    def classify(row: dict[str, str]) -> bool | None:
        try:
            channel = int(row.get("channel", ""))
        except (TypeError, ValueError):
            return None
        return channel in channels

    return classify


def crop_rows(
    rows: Iterable[dict[str, str]],
    classify: Callable[[dict[str, str]], bool | None],
) -> tuple[list[dict[str, str]], Counter, Counter, Counter]:
    """Split rows into in-band, out-of-band, and unknown-frequency buckets."""

    in_band: list[dict[str, str]] = []
    out_counts: Counter = Counter()
    unknown_counts: Counter = Counter()
    in_counts: Counter = Counter()
    for row in rows:
        channel = str(row.get("channel", "")).strip()
        decision = classify(row)
        if decision is None:
            unknown_counts[channel] += 1
        elif decision:
            in_band.append(row)
            in_counts[channel] += 1
        else:
            out_counts[channel] += 1
    return in_band, in_counts, out_counts, unknown_counts


def channel_freq_map(rows: Iterable[dict[str, str]]) -> dict[str, float]:
    observed: dict[str, float] = {}
    for row in rows:
        channel = str(row.get("channel", "")).strip()
        if not channel or channel in observed:
            continue
        freq = row_freq_hz(row)
        if freq is not None:
            observed[channel] = freq
    return observed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parser-csv", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None, help="IQ metadata.json")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--center-freq-hz", type=float, default=None)
    parser.add_argument(
        "--bandwidth-hz",
        type=float,
        default=None,
        help="target passband; required unless --channel-list is given",
    )
    parser.add_argument(
        "--channel-list",
        default="",
        help="explicit in-band BLE channels, e.g. 0,1,2,3 (alternative to --bandwidth-hz)",
    )
    parser.add_argument(
        "--packet-types",
        default="",
        help="comma-separated packet_type values to keep, e.g. BLE_ADV "
        "(applied before the passband/channel crop)",
    )
    parser.add_argument("--run-id", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    parser_csv = args.parser_csv.expanduser().resolve()
    rows = read_csv(parser_csv)
    fields = list(rows[0].keys()) if rows else []
    if not fields:
        raise SystemExit(f"parser CSV has no rows or no header: {parser_csv}")
    packet_types: list[str] = []
    if args.packet_types.strip():
        packet_types = [item.strip() for item in args.packet_types.split(",") if item.strip()]
        allowed = set(packet_types)
        rows = [row for row in rows if str(row.get("packet_type", "")).strip() in allowed]

    metadata: dict[str, Any] = {}
    if args.metadata is not None:
        metadata = json.loads(args.metadata.expanduser().resolve().read_text(encoding="utf-8"))
    center_hz = args.center_freq_hz
    if center_hz is None:
        center_hz = float(metadata.get("actual_center_frequency_hz") or 2440e6)
    channel_numbers = [int(item) for item in args.channel_list.split(",") if item.strip() != ""]

    if channel_numbers:
        mode = "channel_list"
        classify = channel_list_filter(set(channel_numbers))
    elif args.bandwidth_hz is not None and args.bandwidth_hz > 0:
        mode = "passband"
        classify = passband_filter(center_hz, args.bandwidth_hz)
    else:
        raise SystemExit("provide --bandwidth-hz or a non-empty --channel-list")

    in_band, in_counts, out_counts, unknown_counts = crop_rows(rows, classify)
    in_band_rows = len(in_band)
    out_of_band_rows = sum(out_counts.values())
    unknown_rows = sum(unknown_counts.values())
    classified_rows = in_band_rows + out_of_band_rows
    channel_map = channel_freq_map(rows)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_id": args.run_id,
        "input_parser_rows": len(rows),
        "output_parser_rows": in_band_rows,
        "packet_type_filter": packet_types or None,
        "mode": mode,
        "center_frequency_hz": center_hz,
        "bandwidth_hz": args.bandwidth_hz,
        "band_edges_hz": [center_hz - args.bandwidth_hz / 2.0, center_hz + args.bandwidth_hz / 2.0]
        if args.bandwidth_hz is not None
        else None,
        "channel_list": channel_numbers or None,
        "channel_map": channel_map,
        "per_channel_counts": dict(Counter(row.get("channel", "") for row in rows)),
        "per_channel_in_band_counts": dict(in_counts),
        "per_channel_out_of_band_counts": dict(out_counts),
        "per_channel_unknown_counts": dict(unknown_counts),
        "in_band_rows": in_band_rows,
        "out_of_band_rows": out_of_band_rows,
        "rows_without_frequency": unknown_rows,
        "c_bw_packets_estimated": in_band_rows / classified_rows if classified_rows else 0.0,
        "notes": [
            "Per-channel packet distribution of the full-band capture is used as the "
            "transmitter's effective channel-usage map (estimated, no RTT ground truth).",
            "Rows without a channel/frequency are retained in the output CSV but excluded "
            "from C_bw; they are counted in rows_without_frequency.",
            "Passband edge is inclusive: abs(freq - center) <= bandwidth/2.",
        ],
    }
    output_dir = (args.output_dir or args.output_csv.parent).expanduser().resolve()
    write_csv(args.output_csv.expanduser().resolve(), in_band, fields)
    write_json(output_dir / "crop_summary.json", summary)
    write_json(output_dir / "channel_map.json", channel_map)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
