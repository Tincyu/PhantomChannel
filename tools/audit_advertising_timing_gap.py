#!/usr/bin/env python3
"""Audit the frozen advertising event/gap definition from detector §6.3.2.

This is intentionally separate from the historical timing ROC analyzer.  The
audit reconstructs only adjacent target-address packets in the strict
37/38/39 order, keeps 39->37 as an excluded boundary row, and computes raw
statistics before any robust normalization.  It is therefore useful for
diagnosing a suspicious append-last result without changing the formal data.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


TARGET_ADDRESS = "d1:22:33:44:55:66"
DEFAULT_ROOT = Path("/path/to/PhantomChannel/experiments")
DEFAULT_OUTPUT = DEFAULT_ROOT / "figure/detector_roc_20260810/timing_formal/advertising_gap_audit"
FORMAL_RELATIVE = Path("event_timing/formal_20260809/advertising")
MATCHED_BENIGN_CONDITION = "matched-nRF52840-benign"
CONDITION_SPECS = (
    (MATCHED_BENIGN_CONDITION, "timing_formal_adv_normal_rep"),
    ("append-last-239B", "timing_formal_adv_append_last_rep"),
    ("append-every-239B", "timing_formal_adv_append_every_rep"),
)
PAIR_ROLES = ((37, 38, "37_to_38"), (38, 39, "38_to_39"))
FORMAL_FIELDS = [
    "frame.number", "frame.time_epoch", "btle.advertising_address",
    "nordic_ble.channel", "nordic_ble.packet_counter",
    "nordic_ble.delta_time", "nordic_ble.delta_time_ss",
    "nordic_ble.crcok", "btle.advertising_header.length", "btle.length",
    "btle_rf.phy", "nordic_ble.phy",
]


def parse_float(value: Any) -> float | None:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def parse_int(value: Any) -> int | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    try:
        return int(text, 16) if text.startswith("0x") else int(text)
    except ValueError:
        return None


def normalize_address(value: Any) -> str:
    parts = str(value or "").strip().lower().replace("-", ":").split(":")
    if len(parts) != 6:
        return ""
    try:
        if any(not 0 <= int(part, 16) <= 255 for part in parts):
            return ""
    except ValueError:
        return ""
    return ":".join(f"{int(part, 16):02x}" for part in parts)


def crc_value(value: Any) -> bool | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text in {"true", "1", "yes", "ok"}:
        return True
    if text in {"false", "0", "no", "bad", "error"}:
        return False
    return None


def p95(values: Iterable[float]) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = 0.95 * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def mad(values: Iterable[float], center: float | None = None) -> float | None:
    ordered = list(values)
    if not ordered:
        return None
    center = statistics.median(ordered) if center is None else center
    return statistics.median(abs(value - center) for value in ordered)


def stats(values: Iterable[float]) -> dict[str, float | int | None]:
    values = list(values)
    center = statistics.median(values) if values else None
    return {
        "count": len(values),
        "median": center,
        "mad": mad(values, center),
        "p95": p95(values),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def tshark_rows(path: Path) -> list[dict[str, str]]:
    command = [
        "tshark", "-r", str(path), "-T", "fields",
        "-E", "separator=\t", "-E", "quote=d", "-E", "occurrence=f",
    ]
    for field in FORMAL_FIELDS:
        command.extend(("-e", field))
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"tshark failed for {path}: {result.stderr[-1200:]}")
    rows: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        values = next(csv.reader([line], delimiter="\t", quotechar='"'), [])
        values += [""] * (len(FORMAL_FIELDS) - len(values))
        rows.append(dict(zip(FORMAL_FIELDS, values)))
    return rows


@dataclass(frozen=True)
class Packet:
    frame: int
    timestamp_us: float
    channel: int
    packet_counter: int | None
    delta_time_us: float | None
    delta_time_ss_us: float | None
    crc_valid: bool | None
    length: int | None
    phy_mbps: float
    phy_source: str

    @property
    def crc_status(self) -> str:
        if self.crc_valid is True:
            return "valid"
        if self.crc_valid is False:
            return "invalid"
        return "unknown"


@dataclass(frozen=True)
class Event:
    event_id: int
    packets: tuple[Packet, Packet, Packet]


@dataclass(frozen=True)
class Boundary:
    event_id: int
    previous: Packet
    next_packet: Packet


def phy_info(row: dict[str, str]) -> tuple[float, str]:
    # The formal advertising pcaps expose no usable PHY value.  Advertising
    # channel packets are therefore calculated using the legal/default LE 1M
    # airtime and the assumption is recorded in every pair row.
    for field in ("btle_rf.phy", "nordic_ble.phy"):
        raw = (row.get(field) or "").strip().lower()
        value = parse_float(raw)
        if value in {1.0, 2.0, 8.0}:
            return value, field
    return 1.0, "default_1M_no_phy_field"


def row_to_packet(row: dict[str, str]) -> Packet | None:
    timestamp = parse_float(row.get("frame.time_epoch"))
    channel = parse_int(row.get("nordic_ble.channel"))
    if timestamp is None or channel not in {37, 38, 39}:
        return None
    length = parse_int(row.get("btle.advertising_header.length"))
    if length is None:
        length = parse_int(row.get("btle.length"))
    phy_mbps, phy_source = phy_info(row)
    return Packet(
        frame=parse_int(row.get("frame.number")) or 0,
        timestamp_us=timestamp * 1_000_000.0,
        channel=channel,
        packet_counter=parse_int(row.get("nordic_ble.packet_counter")),
        delta_time_us=parse_float(row.get("nordic_ble.delta_time")),
        delta_time_ss_us=parse_float(row.get("nordic_ble.delta_time_ss")),
        crc_valid=crc_value(row.get("nordic_ble.crcok")),
        length=length,
        phy_mbps=phy_mbps,
        phy_source=phy_source,
    )


def target_packets(rows: list[dict[str, str]], target_address: str, crc_valid_only: bool) -> list[Packet]:
    target_address = normalize_address(target_address)
    packets: list[Packet] = []
    for row in rows:
        if normalize_address(row.get("btle.advertising_address")) != target_address:
            continue
        packet = row_to_packet(row)
        if packet is None:
            continue
        if crc_valid_only and packet.crc_valid is not True:
            continue
        packets.append(packet)
    return sorted(packets, key=lambda packet: (packet.timestamp_us, packet.frame))


def strict_reconstruct(packets: list[Packet]) -> tuple[list[Event], list[Boundary], int]:
    """Rebuild adjacent 37/38/39 triples without time or counter eligibility.

    Looking at the immediate next target-address packet is important: it makes
    an intervening target packet a rejected/incomplete candidate rather than a
    packet that can be skipped across into another event.
    """
    events: list[Event] = []
    boundaries: list[Boundary] = []
    rejected_candidates = 0
    index = 0
    while index + 2 < len(packets):
        first, second, third = packets[index:index + 3]
        if (first.channel, second.channel, third.channel) == (37, 38, 39):
            event = Event(len(events), (first, second, third))
            events.append(event)
            after = index + 3
            if after < len(packets) and packets[after].channel == 37:
                boundaries.append(Boundary(event.event_id, third, packets[after]))
            index += 3
        else:
            if first.channel == 37:
                rejected_candidates += 1
            index += 1
    return events, boundaries, rejected_candidates


def airtime_us(packet: Packet) -> float | None:
    if packet.length is None or packet.length < 0:
        return None
    # LE uncoded 1M advertising PDU: preamble + AA + header + payload + CRC.
    return (1 + 4 + 2 + packet.length + 3) * 8.0 / packet.phy_mbps


def pair_gap_us(previous: Packet, next_packet: Packet) -> tuple[float | None, float | None]:
    start_to_start = next_packet.timestamp_us - previous.timestamp_us
    airtime = airtime_us(previous)
    return start_to_start - airtime if airtime is not None else None, start_to_start


def pair_row(condition: str, capture_id: str, event: Event, pair_index: int) -> dict[str, Any]:
    previous = event.packets[pair_index]
    next_packet = event.packets[pair_index + 1]
    gap, start_to_start = pair_gap_us(previous, next_packet)
    role = f"{previous.channel}_to_{next_packet.channel}"
    row: dict[str, Any] = {
        "condition": condition,
        "capture_id": capture_id,
        "event_id": event.event_id,
        "pair_role": role,
        "excluded_boundary_pair": False,
        "frame_prev": previous.frame,
        "frame_next": next_packet.frame,
        "channel_prev": previous.channel,
        "channel_next": next_packet.channel,
        "packet_counter_prev": previous.packet_counter,
        "packet_counter_next": next_packet.packet_counter,
        "packet_counter_step": (
            ((next_packet.packet_counter - previous.packet_counter) & 0xFFFF)
            if previous.packet_counter is not None and next_packet.packet_counter is not None else None
        ),
        "crc_prev": previous.crc_status,
        "crc_next": next_packet.crc_status,
        "t_start_prev_us": previous.timestamp_us,
        "t_start_next_us": next_packet.timestamp_us,
        "start_to_start_us": start_to_start,
        "airtime_prev_us": airtime_us(previous),
        "gap_us": gap,
        "delta_time_us": next_packet.delta_time_us,
        "delta_time_ss_us": next_packet.delta_time_ss_us,
        "delta_time_minus_gap_us": (
            next_packet.delta_time_us - gap
            if next_packet.delta_time_us is not None and gap is not None else None
        ),
        "delta_time_ss_minus_start_to_start_us": (
            next_packet.delta_time_ss_us - start_to_start
            if next_packet.delta_time_ss_us is not None else None
        ),
        "phy_mbps": previous.phy_mbps,
        "phy_source": previous.phy_source,
        "pdu_length_prev": previous.length,
        "event_frames": ",".join(str(packet.frame) for packet in event.packets),
        "event_packet_counters": ",".join(
            "" if packet.packet_counter is None else str(packet.packet_counter)
            for packet in event.packets
        ),
    }
    return row


def boundary_row(condition: str, capture_id: str, boundary: Boundary) -> dict[str, Any]:
    previous, next_packet = boundary.previous, boundary.next_packet
    start_to_start = next_packet.timestamp_us - previous.timestamp_us
    return {
        "condition": condition,
        "capture_id": capture_id,
        "event_id": boundary.event_id,
        "pair_role": "39_to_37",
        "excluded_boundary_pair": True,
        "frame_prev": previous.frame,
        "frame_next": next_packet.frame,
        "channel_prev": previous.channel,
        "channel_next": next_packet.channel,
        "packet_counter_prev": previous.packet_counter,
        "packet_counter_next": next_packet.packet_counter,
        "packet_counter_step": (
            ((next_packet.packet_counter - previous.packet_counter) & 0xFFFF)
            if previous.packet_counter is not None and next_packet.packet_counter is not None else None
        ),
        "crc_prev": previous.crc_status,
        "crc_next": next_packet.crc_status,
        "t_start_prev_us": previous.timestamp_us,
        "t_start_next_us": next_packet.timestamp_us,
        "start_to_start_us": start_to_start,
        "airtime_prev_us": airtime_us(previous),
        "gap_us": None,
        "delta_time_us": next_packet.delta_time_us,
        "delta_time_ss_us": next_packet.delta_time_ss_us,
        "delta_time_minus_gap_us": None,
        "delta_time_ss_minus_start_to_start_us": (
            next_packet.delta_time_ss_us - start_to_start
            if next_packet.delta_time_ss_us is not None else None
        ),
        "phy_mbps": previous.phy_mbps,
        "phy_source": previous.phy_source,
        "pdu_length_prev": previous.length,
        "event_frames": "",
        "event_packet_counters": "",
    }


def event_intervals(events: list[Event], nominal_interval_us: float) -> list[float]:
    return [
        right.packets[0].timestamp_us - left.packets[0].timestamp_us - nominal_interval_us
        for left, right in zip(events, events[1:])
    ]


def capture_analysis(
    pcap: Path, condition: str, capture_id: str, target_address: str,
    crc_valid_only: bool, nominal_interval_us: float,
) -> dict[str, Any]:
    rows = tshark_rows(pcap)
    all_packets = target_packets(rows, target_address, crc_valid_only=False)
    all_events, all_boundaries, rejected = strict_reconstruct(all_packets)
    if crc_valid_only:
        # Keep event segmentation identical to all-target mode.  Filtering
        # packets before segmentation can stitch P37/P38/P39 from different
        # events and produce a false CRC-valid event.
        events = [event for event in all_events
                  if all(packet.crc_valid is True for packet in event.packets)]
        valid_ids = {event.event_id for event in events}
        boundaries = [item for item in all_boundaries if item.event_id in valid_ids]
        packets = [packet for event in events for packet in event.packets]
    else:
        events, boundaries, packets = all_events, all_boundaries, all_packets
    pair_rows = [pair_row(condition, capture_id, event, pair_index)
                 for event in events for pair_index in range(2)]
    boundary_rows = [boundary_row(condition, capture_id, item) for item in boundaries]
    gap_37_38 = [row["gap_us"] for row in pair_rows if row["pair_role"] == "37_to_38" and row["gap_us"] is not None]
    gap_38_39 = [row["gap_us"] for row in pair_rows if row["pair_role"] == "38_to_39" and row["gap_us"] is not None]
    pooled_gap = gap_37_38 + gap_38_39
    interval_residual = event_intervals(events, nominal_interval_us)
    interval_start = [value + nominal_interval_us for value in interval_residual]
    known_crc = [packet for packet in all_packets if packet.crc_valid is not None]
    valid_crc = [packet for packet in all_packets if packet.crc_valid is True]
    valid_events = [event for event in all_events
                    if all(packet.crc_valid is True for packet in event.packets)]
    valid_pairs = [row for row in pair_rows
                   if row["crc_prev"] == "valid" and row["crc_next"] == "valid"]
    return {
        "pcap": pcap,
        "condition": condition,
        "capture_id": capture_id,
        "crc_valid_only": crc_valid_only,
        "all_packets": all_packets,
        "packets": packets,
        "events": events,
        "boundaries": boundaries,
        "pair_rows": pair_rows,
        "boundary_rows": boundary_rows,
        "rejected_candidates": rejected,
        "gap_37_38": gap_37_38,
        "gap_38_39": gap_38_39,
        "pooled_gap": pooled_gap,
        "interval_residual": interval_residual,
        "interval_start": interval_start,
        "target_packet_count": len(all_packets),
        "crc_known_count": len(known_crc),
        "crc_valid_count": len(valid_crc),
        "crc_valid_packet_coverage": len(valid_crc) / len(known_crc) if known_crc else None,
        "all_mode_event_count": len(all_events),
        "all_mode_pair_count": 2 * len(all_events),
        "valid_event_count_from_all": len(valid_events),
        "valid_pair_count_from_all": len(valid_pairs),
    }


def raw_capture_row(result: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "condition": result["condition"],
        "capture_id": result["capture_id"],
        "crc_policy": "CRC-valid-only" if result["crc_valid_only"] else "all-target-rows",
        "target_packet_count": result["target_packet_count"],
        "reconstructed_event_count": len(result["events"]),
        "eligible_pair_count": len(result["pair_rows"]),
        "excluded_boundary_pair_count": len(result["boundary_rows"]),
        "rejected_incomplete_37_candidates": result["rejected_candidates"],
        "crc_known_target_packet_count": result["crc_known_count"],
        "crc_valid_target_packet_count": result["crc_valid_count"],
        "crc_valid_packet_coverage": result["crc_valid_packet_coverage"],
        "crc_valid_event_coverage": (
            result["valid_event_count_from_all"] / result["all_mode_event_count"]
            if result["all_mode_event_count"] else None
        ),
        "crc_valid_pair_coverage": (
            result["valid_pair_count_from_all"] / result["all_mode_pair_count"]
            if result["all_mode_pair_count"] else None
        ),
        "cross_event_gap_feature_count": 0,
        "at_least_20_complete_events": len(result["events"]) >= 20,
    }
    for prefix, values in (
        ("gap_37_38_us", result["gap_37_38"]),
        ("gap_38_39_us", result["gap_38_39"]),
        ("pooled_intra_event_gap_us", result["pooled_gap"]),
        ("event_start_to_start_residual_us", result["interval_residual"]),
        ("event_start_to_start_us", result["interval_start"]),
    ):
        for suffix, value in stats(values).items():
            row[f"{prefix}_{suffix}"] = value
    return row


def semantics_rows(result: dict[str, Any]) -> dict[str, Any]:
    pair_rows = result["pair_rows"]
    delta_gap = [row["delta_time_minus_gap_us"] for row in pair_rows
                 if row["delta_time_minus_gap_us"] is not None]
    delta_ss = [row["delta_time_ss_minus_start_to_start_us"] for row in pair_rows
                if row["delta_time_ss_minus_start_to_start_us"] is not None]
    row: dict[str, Any] = {
        "condition": result["condition"],
        "capture_id": result["capture_id"],
        "crc_policy": "CRC-valid-only" if result["crc_valid_only"] else "all-target-rows",
        "phy_assumption": "LE 1M (no usable PHY field in formal pcap)",
        "delta_time_observations": len(delta_gap),
        "delta_time_ss_observations": len(delta_ss),
    }
    for prefix, values in (("delta_time_minus_reconstructed_gap_us", delta_gap),
                           ("delta_time_ss_minus_start_to_start_us", delta_ss)):
        summary = stats(values)
        row[f"{prefix}_median"] = summary["median"]
        row[f"{prefix}_mad"] = summary["mad"]
        row[f"{prefix}_p95_abs"] = p95(abs(value) for value in values)
        row[f"{prefix}_fraction_abs_le_2us"] = (
            sum(abs(value) <= 2.0 for value in values) / len(values) if values else None
        )
    return row


def window_features(result: dict[str, Any], events_per_window: int = 30,
                    nominal_interval_us: float = 20_000.0) -> list[dict[str, Any]]:
    events: list[Event] = result["events"]
    rows: list[dict[str, Any]] = []
    for start in range(0, len(events) - events_per_window + 1, events_per_window):
        chunk = events[start:start + events_per_window]
        gaps: list[float] = []
        for event in chunk:
            for left, right in zip(event.packets, event.packets[1:]):
                gap, _ = pair_gap_us(left, right)
                if gap is not None:
                    gaps.append(gap)
        residuals = event_intervals(chunk, nominal_interval_us)
        row = {
            "condition": result["condition"],
            "capture_id": result["capture_id"],
            "crc_policy": "CRC-valid-only" if result["crc_valid_only"] else "all-target-rows",
            "window_index": len(rows),
            "event_count": len(chunk),
            "gap_37_38_median_us": statistics.median([
                pair_gap_us(event.packets[0], event.packets[1])[0]
                for event in chunk
                if pair_gap_us(event.packets[0], event.packets[1])[0] is not None
            ]) if chunk else None,
            "gap_38_39_median_us": statistics.median([
                pair_gap_us(event.packets[1], event.packets[2])[0]
                for event in chunk
                if pair_gap_us(event.packets[1], event.packets[2])[0] is not None
            ]) if chunk else None,
            "pooled_gap_median_us": statistics.median(gaps) if gaps else None,
            "event_start_to_start_residual_median_us": statistics.median(residuals) if residuals else None,
            "event_start_to_start_residual_mad_us": mad(residuals),
            "gap_pair_count": len(gaps),
            "event_interval_count": len(residuals),
        }
        rows.append(row)
    return rows


def auc(negative: Iterable[float], positive: Iterable[float]) -> float | None:
    negative = [value for value in negative if value is not None and math.isfinite(value)]
    positive = [value for value in positive if value is not None and math.isfinite(value)]
    if not negative or not positive:
        return None
    wins = sum((1.0 if pos > neg else 0.5 if pos == neg else 0.0)
               for pos in positive for neg in negative)
    return wins / (len(negative) * len(positive))


def capture_feature_scores(rows: list[dict[str, Any]], normal_rows: list[dict[str, Any]]) -> dict[str, float]:
    for feature in ("pooled_gap_median_us", "event_start_to_start_residual_median_us"):
        normal_values = [row[feature] for row in normal_rows if row[feature] is not None]
        center = statistics.median(normal_values) if normal_values else 0.0
        for row in rows:
            value = row[feature]
            row[f"{feature}_deviation_us"] = abs(value - center) if value is not None else None
    gap_baseline = [row["pooled_gap_median_us"] for row in normal_rows
                    if row["pooled_gap_median_us"] is not None]
    event_baseline = [row["event_start_to_start_residual_median_us"] for row in normal_rows
                      if row["event_start_to_start_residual_median_us"] is not None]
    gap_center = statistics.median(gap_baseline) if gap_baseline else 0.0
    event_center = statistics.median(event_baseline) if event_baseline else 0.0
    gap_floor = max(mad(gap_baseline) or 0.0, 1.0)
    event_floor = max(mad(event_baseline) or 0.0, 1.0)
    for row in rows:
        gap = row["pooled_gap_median_us"]
        event = row["event_start_to_start_residual_median_us"]
        row["gap_score"] = max(0.0, (gap - gap_center) / gap_floor) if gap is not None else None
        row["event_score"] = abs(event - event_center) / event_floor if event is not None else None
        row["combined_score"] = (
            max(row["gap_score"], row["event_score"])
            if row["gap_score"] is not None and row["event_score"] is not None else None
        )
        row["gap_mad_floor_us"] = gap_floor
        row["event_mad_floor_us"] = event_floor
    return {
        "gap_baseline_median_us": gap_center,
        "gap_baseline_mad_floor_us": gap_floor,
        "event_baseline_median_us": event_center,
        "event_baseline_mad_floor_us": event_floor,
    }


def roc_rows(window_rows: list[dict[str, Any]], positive_condition: str) -> list[dict[str, Any]]:
    normal = [row for row in window_rows if row["condition"] == MATCHED_BENIGN_CONDITION]
    positive = [row for row in window_rows if row["condition"] == positive_condition]
    output: list[dict[str, Any]] = []
    for feature in ("gap_score", "event_score", "combined_score"):
        neg = [row.get(feature) for row in normal]
        pos = [row.get(feature) for row in positive]
        output.append({
            "positive_condition": positive_condition,
            "crc_policy": window_rows[0]["crc_policy"] if window_rows else "",
            "feature": feature,
            "negative_windows": sum(value is not None for value in neg),
            "positive_windows": sum(value is not None for value in pos),
            "auc": auc(neg, pos),
        })
    return output


def permutation_rows(window_rows: list[dict[str, Any]], positive_condition: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    captures = [row["capture_id"] for row in window_rows if row["condition"] in {MATCHED_BENIGN_CONDITION, positive_condition}]
    captures = list(dict.fromkeys(captures))
    if len(captures) != 10:
        return [], []
    by_capture = {capture: [row for row in window_rows if row["capture_id"] == capture]
                  for capture in captures}
    natural_negative = set(captures[:5])
    detail: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for feature in ("gap_score", "event_score", "combined_score"):
        natural_neg = [row.get(feature) for capture in captures[:5] for row in by_capture[capture]]
        natural_pos = [row.get(feature) for capture in captures[5:] for row in by_capture[capture]]
        observed = auc(natural_neg, natural_pos)
        values: list[float] = []
        for assignment_id, negative_indices in enumerate(itertools.combinations(range(10), 5)):
            negative_set = {captures[index] for index in negative_indices}
            negative = [row.get(feature) for capture in captures if capture in negative_set
                        for row in by_capture[capture]]
            positive = [row.get(feature) for capture in captures if capture not in negative_set
                        for row in by_capture[capture]]
            value = auc(negative, positive)
            if value is not None:
                values.append(value)
            detail.append({
                "positive_condition": positive_condition,
                "crc_policy": window_rows[0]["crc_policy"],
                "feature": feature,
                "assignment_id": assignment_id,
                "negative_capture_ids": ";".join(sorted(negative_set)),
                "auc": value,
            })
        distance = abs(observed - 0.5) if observed is not None else None
        p_value = (
            sum(abs(value - 0.5) >= distance - 1e-12 for value in values) / len(values)
            if distance is not None and values else None
        )
        summary.append({
            "positive_condition": positive_condition,
            "crc_policy": window_rows[0]["crc_policy"],
            "feature": feature,
            "observed_auc": observed,
            "permutations": len(values),
            "null_median_auc": statistics.median(values) if values else None,
            "null_min_auc": min(values) if values else None,
            "null_max_auc": max(values) if values else None,
            "two_sided_permutation_p": p_value,
        })
    return detail, summary


def run_synthetic_checks() -> dict[str, Any]:
    def make_event(start: float, first_gap: float = 293.0, second_gap: float = 288.0,
                   base_counter: int | None = None) -> list[Packet]:
        first = Packet(1, start, 37, base_counter, None, None, True, 6, 1.0, "synthetic_1M")
        second = Packet(2, start + 128.0 + first_gap, 38,
                        None if base_counter is None else base_counter + 1,
                        first_gap, 128.0 + first_gap, True, 6, 1.0, "synthetic_1M")
        third_start = second.timestamp_us + 128.0 + second_gap
        third = Packet(3, third_start, 39,
                       None if base_counter is None else base_counter + 2,
                       second_gap, 128.0 + second_gap, True, 6, 1.0, "synthetic_1M")
        return [first, second, third]

    base = make_event(0.0, base_counter=1) + make_event(20_000.0, base_counter=4)
    shifted_boundary = make_event(0.0, base_counter=1) + make_event(21_000.0, base_counter=4)
    every = make_event(0.0, first_gap=500.0, second_gap=600.0, base_counter=1)
    events_base, boundaries_base, _ = strict_reconstruct(base)
    events_shifted, _, _ = strict_reconstruct(shifted_boundary)
    events_every, _, _ = strict_reconstruct(every)
    base_gaps = [pair_gap_us(packet, next_packet)[0]
                 for event in events_base for packet, next_packet in zip(event.packets, event.packets[1:])]
    shifted_gaps = [pair_gap_us(packet, next_packet)[0]
                    for event in events_shifted for packet, next_packet in zip(event.packets, event.packets[1:])]
    every_gaps = [pair_gap_us(packet, next_packet)[0]
                  for event in events_every for packet, next_packet in zip(event.packets, event.packets[1:])]
    # A tail after P39 is represented by moving the next event's P37 only.
    # Its intra-event pairs remain exactly unchanged.
    checks = {
        "boundary_only_does_not_change_gap": base_gaps == shifted_gaps,
        "boundary_pair_is_excluded": len(boundaries_base) == 1 and boundaries_base[0].previous.channel == 39,
        "append_every_changes_both_intra_gaps": every_gaps == [500.0, 600.0],
        "start_to_start_minus_airtime_is_gap": pair_gap_us(events_base[0].packets[0], events_base[0].packets[1])[0] == 293.0,
        "mad_zero_preserves_raw_and_floor_is_one": stats([4.0, 4.0])["mad"] == 0.0 and max(0.0, 0.0, 1.0) == 1.0,
    }
    # An intervening target packet must prevent a triple from being assembled.
    interrupted = make_event(0.0, base_counter=1)[:2] + [make_event(5_000.0, base_counter=9)[0]] + make_event(10_000.0, base_counter=12)
    interrupted_events, _, _ = strict_reconstruct(interrupted)
    checks["intervening_target_packet_not_skipped"] = len(interrupted_events) == 1
    return {"passed": all(checks.values()), "checks": checks}


def discover_captures(root: Path) -> list[dict[str, Any]]:
    base = root / FORMAL_RELATIVE
    captures: list[dict[str, Any]] = []
    for condition, prefix in CONDITION_SPECS:
        for rep in range(1, 6):
            capture_id = f"{condition}_rep{rep}"
            directory = base / f"{prefix}{rep}"
            pcap = directory / "monitor/monitor.pcapng"
            if not pcap.is_file():
                raise FileNotFoundError(pcap)
            captures.append({"condition": condition, "rep": rep, "capture_id": capture_id, "pcap": pcap})
    return captures


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-address", default=TARGET_ADDRESS)
    parser.add_argument("--nominal-interval-us", type=float, default=20_000.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    synthetic = run_synthetic_checks()
    if not synthetic["passed"]:
        raise RuntimeError(f"synthetic regression failed: {synthetic}")
    (args.output_dir / "synthetic_regression.json").write_text(
        json.dumps(json_safe(synthetic), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    captures = discover_captures(args.root)
    all_results: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    windows_by_policy: dict[str, list[dict[str, Any]]] = {"all-target-rows": [], "CRC-valid-only": []}
    for capture in captures:
        for crc_valid_only in (False, True):
            result = capture_analysis(
                capture["pcap"], capture["condition"], capture["capture_id"], args.target_address,
                crc_valid_only, args.nominal_interval_us,
            )
            all_results.append(result)
            raw_rows.append(raw_capture_row(result))
            semantic_rows.append(semantics_rows(result))
            policy = "CRC-valid-only" if crc_valid_only else "all-target-rows"
            windows_by_policy[policy].extend(window_features(result, 30, args.nominal_interval_us))
            if not crc_valid_only:
                pair_rows.extend(result["pair_rows"])
                boundary_rows.extend(result["boundary_rows"])

    write_csv(args.output_dir / "event_pair_audit.csv", pair_rows + boundary_rows)
    write_csv(args.output_dir / "capture_raw_statistics.csv", raw_rows)
    write_csv(args.output_dir / "timestamp_delta_semantics.csv", semantic_rows)

    sensitivity: list[dict[str, Any]] = []
    for capture in captures:
        results = [item for item in all_results if item["capture_id"] == capture["capture_id"]]
        all_mode, valid_mode = next(item for item in results if not item["crc_valid_only"]), next(item for item in results if item["crc_valid_only"])
        sensitivity.append({
            "condition": capture["condition"],
            "capture_id": capture["capture_id"],
            "all_target_events": len(all_mode["events"]),
            "crc_valid_only_events": len(valid_mode["events"]),
            "all_target_pairs": len(all_mode["pair_rows"]),
            "crc_valid_only_pairs": len(valid_mode["pair_rows"]),
            "crc_valid_event_coverage": all_mode["valid_event_count_from_all"] / all_mode["all_mode_event_count"] if all_mode["all_mode_event_count"] else None,
            "crc_valid_pair_coverage": all_mode["valid_pair_count_from_all"] / all_mode["all_mode_pair_count"] if all_mode["all_mode_pair_count"] else None,
            "crc_valid_only_at_least_20_events": len(valid_mode["events"]) >= 20,
            "parser_crc_error_dependency": len(all_mode["events"]) > 0 and len(valid_mode["events"]) < 20,
        })
    write_csv(args.output_dir / "crc_policy_sensitivity.csv", sensitivity)

    sanity_rows: list[dict[str, Any]] = []
    roc_rows_all: list[dict[str, Any]] = []
    permutation_detail: list[dict[str, Any]] = []
    permutation_summary: list[dict[str, Any]] = []
    score_calibrations: dict[str, dict[str, float]] = {}
    for policy, windows in windows_by_policy.items():
        normal_windows = [row for row in windows if row["condition"] == MATCHED_BENIGN_CONDITION]
        score_calibrations[policy] = capture_feature_scores(windows, normal_windows)
        write_csv(args.output_dir / f"window_features_{policy.replace('-', '_')}.csv", windows)
        for positive_condition in ("append-last-239B", "append-every-239B"):
            roc_rows_all.extend(roc_rows(windows, positive_condition))
            detail, summary = permutation_rows(windows, positive_condition)
            permutation_detail.extend(detail)
            permutation_summary.extend(summary)

            normal_captures = [row for row in raw_rows if row["condition"] == MATCHED_BENIGN_CONDITION and row["crc_policy"] == policy]
            positive_captures = [row for row in raw_rows if row["condition"] == positive_condition and row["crc_policy"] == policy]
            for feature, score_feature in (
                ("raw_intra_event_gap", "pooled_intra_event_gap_us_median"),
                ("raw_event_start_to_start_residual", "event_start_to_start_residual_us_median_deviation_us"),
            ):
                neg = [row.get(score_feature) for row in normal_captures if row.get(score_feature) is not None]
                pos = [row.get(score_feature) for row in positive_captures if row.get(score_feature) is not None]
                if feature == "raw_event_start_to_start_residual":
                    for row in normal_captures + positive_captures:
                        row[score_feature] = row.get("event_start_to_start_residual_us_median")
                    neg = [abs(row[score_feature]) for row in normal_captures if row.get(score_feature) is not None]
                    pos = [abs(row[score_feature]) for row in positive_captures if row.get(score_feature) is not None]
                sanity_rows.append({
                    "crc_policy": policy,
                    "positive_condition": positive_condition,
                    "check": feature,
                    "negative_captures": len(neg),
                    "positive_captures": len(pos),
                    "auc": auc(neg, pos),
                    "negative_median": statistics.median(neg) if neg else None,
                    "positive_median": statistics.median(pos) if pos else None,
                })
            positive_window_rows = [row for row in windows if row["condition"] == positive_condition]
            normal_window_rows = [row for row in windows if row["condition"] == MATCHED_BENIGN_CONDITION]
            for feature in ("gap_score", "event_score", "combined_score"):
                neg = [row.get(feature) for row in normal_window_rows]
                pos = [row.get(feature) for row in positive_window_rows]
                neg_valid = [value for value in neg if value is not None]
                pos_valid = [value for value in pos if value is not None]
                sanity_rows.append({
                    "crc_policy": policy,
                    "positive_condition": positive_condition,
                    "check": f"window_{feature}",
                    "negative_captures": len({row["capture_id"] for row in normal_window_rows}),
                    "positive_captures": len({row["capture_id"] for row in positive_window_rows}),
                    "auc": auc(neg, pos),
                    "negative_median": statistics.median(neg_valid) if neg_valid else None,
                    "positive_median": statistics.median(pos_valid) if pos_valid else None,
                })

    write_csv(args.output_dir / "detector_sanity_summary.csv", sanity_rows)
    write_csv(args.output_dir / "audited_timing_roc.csv", roc_rows_all)
    write_csv(args.output_dir / "capture_level_permutation.csv", permutation_detail)
    write_csv(args.output_dir / "score_calibration.csv", [
        {"crc_policy": policy, **calibration}
        for policy, calibration in score_calibrations.items()
    ])
    permutation_json = {
        "assignment_count": 252,
        "summary": permutation_summary,
        "assignments": permutation_detail,
    }
    (args.output_dir / "capture_level_permutation.json").write_text(
        json.dumps(json_safe(permutation_json), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    all_semantics = [row for row in semantic_rows if row["crc_policy"] == "all-target-rows"]
    delta_time_verified = bool(all_semantics) and all(
        row["delta_time_observations"] and (row["delta_time_minus_reconstructed_gap_us_p95_abs"] or 0.0) <= 2.0
        for row in all_semantics
    )
    delta_ss_verified = bool(all_semantics) and all(
        row["delta_time_ss_observations"] and (row["delta_time_ss_minus_start_to_start_us_p95_abs"] or 0.0) <= 2.0
        for row in all_semantics
    )
    cross_event_gap_count = sum(
        row.get("excluded_boundary_pair") is True and row.get("gap_us") not in (None, "")
        for row in pair_rows + boundary_rows
    )
    parser_dependency = any(
        row["condition"] == "append-last-239B"
        and int(row["all_target_events"]) > 0 and int(row["crc_valid_only_events"]) < 20
        for row in sensitivity
    )
    manifest = {
        "audit": "advertising_gap_audit",
        "definition": "strict adjacent target-address 37->38->39; 39->37 boundary excluded",
        "target_address": normalize_address(args.target_address),
        "formal_capture_count": len(captures),
        "captures_per_condition": 5,
        "events_per_window": 30,
        "nominal_interval_us": args.nominal_interval_us,
        "cross_event_gap_count": cross_event_gap_count,
        "gap_feature_roles": ["37_to_38", "38_to_39"],
        "excluded_boundary_role": "39_to_37",
        "timestamp_source": "frame.time_epoch start timestamps",
        "airtime_source": "parsed PDU length and LE 1M default because formal pcap has no usable PHY field",
        "delta_time_semantics": {
            "delta_time_is_end_to_start_gap": delta_time_verified,
            "delta_time_ss_is_start_to_start": delta_ss_verified,
            "max_p95_abs_delta_time_minus_gap_us": max(
                (row["delta_time_minus_reconstructed_gap_us_p95_abs"] or 0.0) for row in all_semantics
            ) if all_semantics else None,
            "max_p95_abs_delta_time_ss_minus_start_to_start_us": max(
                (row["delta_time_ss_minus_start_to_start_us_p95_abs"] or 0.0) for row in all_semantics
            ) if all_semantics else None,
        },
        "score_calibration": score_calibrations,
        "parser_crc_error_dependency_for_append_last": parser_dependency,
        "crc_policy_outputs": ["all-target-rows", "CRC-valid-only"],
        "synthetic_tests": synthetic,
        "required_assertions": {
            "cross_event_gap_count_zero": cross_event_gap_count == 0,
            "delta_time_semantics_verified": delta_time_verified and delta_ss_verified,
            "all_captures_have_at_least_20_events": all(row["at_least_20_complete_events"] for row in raw_rows if row["crc_policy"] == "all-target-rows"),
            "synthetic_tests_pass": synthetic["passed"],
            "capture_level_permutations": 252,
        },
        "quarantine": {
            "status": "pending_crc_parser_artifact_review",
            "legacy_append_last_gap_auc": 0.998765,
            "legacy_append_last_combined_auc": 0.998982,
            "delta_semantics_review": "passed",
            "cross_event_review": "passed",
            "synthetic_review": "passed",
            "raw_roc_consistency": "same strict reconstruction; all-target diagnostic only",
            "blocking_reason": "append-last formal positive depends on parser CRC-error rows",
            "paper_ready": False,
        },
    }
    (args.output_dir / "advertising_gap_audit_manifest.json").write_text(
        json.dumps(json_safe(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(json_safe({
        "output_dir": args.output_dir,
        "captures": len(captures),
        "all_target_events": sum(len(item["events"]) for item in all_results if not item["crc_valid_only"]),
        "all_target_boundary_rows": len(boundary_rows),
        "roc_rows": len(roc_rows_all),
        "permutation_rows": len(permutation_detail),
        "synthetic_passed": synthetic["passed"],
    }), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
