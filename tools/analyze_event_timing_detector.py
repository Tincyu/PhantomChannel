#!/usr/bin/env python3
"""Analyze the §6 event-level timing detector from nRF Sniffer pcapng files.

The analyzer deliberately does not reuse the old trace-level ``timing_cv``
feature.  Advertising captures are rebuilt as complete 37/38/39 events and
connection captures are rebuilt from ``nordic_ble.event_counter``.  It then
forms non-overlapping 30-event or 50-observed-event windows, freezes a robust benign
baseline and theta_5 from calibration captures, and reports window ROC plus
capture-clustered bootstrap intervals.

Example (advertising):

  python3 tools/analyze_event_timing_detector.py \
    --traffic advertising --target-address d1:22:33:44:55:66 \
    --calibration-benign /path/benign_cal.pcapng \
    --capture normal=/path/normal.pcapng \
    --capture append-last-239B=/path/last.pcapng \
    --capture append-every-239B=/path/every.pcapng \
    --output-dir /path/to/PhantomChannel/experiments/figure/event_timing

For connected captures, use ``--traffic connection`` and optionally
``--target-aa``.  The default pair mode uses direction-changing adjacent
packets; the packet type is retained in the output.  ``--pair-mode`` can be
set to ``central`` or ``peripheral`` to require the corresponding
request/response semantics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ADV_AA = "0x8e89bed6"
TWO_PI = 2.0 * math.pi


def finite(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


def parse_float(value: str) -> float | None:
    try:
        return finite(float(value.strip())) if value.strip() else None
    except (AttributeError, TypeError, ValueError):
        return None


def parse_int(value: str) -> int | None:
    text = (value or "").strip().lower()
    if not text:
        return None
    try:
        return int(text, 16) if text.startswith("0x") else int(text)
    except ValueError:
        return None


def normalize_hex(value: str) -> str:
    text = (value or "").strip().lower().replace("0x", "")
    if not re.fullmatch(r"[0-9a-f]{1,8}", text):
        return ""
    return f"0x{int(text, 16):08x}"


def normalize_address(value: str) -> str:
    text = (value or "").strip().lower().replace("-", ":")
    parts = text.split(":")
    if len(parts) != 6 or any(not re.fullmatch(r"[0-9a-f]{1,2}", part) for part in parts):
        return ""
    return ":".join(f"{int(part, 16):02x}" for part in parts)


def is_true(value: str) -> bool:
    return (value or "").strip().lower() in {"true", "1", "yes", "ok"}


def median(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.median(values) if values else None


def mad(values: Iterable[float], center: float | None = None) -> float | None:
    values = list(values)
    if not values:
        return None
    center = statistics.median(values) if center is None else center
    return statistics.median(abs(value - center) for value in values)


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


def capture_manifest_duration(path: Path) -> float | None:
    """Read the requested wall-clock capture duration when a run manifest exists.

    Nordic sniffer PCAP timestamps can use the dongle clock and may span a
    different numerical interval from the coordinator's wall-clock run.  The
    manifest is authoritative for false-alarms/minute and formal duration;
    legacy pilot pcaps without a manifest continue to use their PCAP span.
    """
    candidates = [path.parent.parent / "session_manifest.json", path.parent / "session_manifest.json"]
    for manifest_path in candidates:
        if not manifest_path.is_file():
            continue
        try:
            value = json.loads(manifest_path.read_text(encoding="utf-8")).get("duration_s")
            value = float(value)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if math.isfinite(value) and value > 0.0:
            return value
    return None


def tshark_rows(pcap: Path, fields: list[str]) -> list[dict[str, str]]:
    command = [
        "tshark", "-r", str(pcap), "-T", "fields",
        "-E", "separator=\t", "-E", "quote=d", "-E", "occurrence=f",
    ]
    for field_name in fields:
        command.extend(("-e", field_name))
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"tshark failed for {pcap}: {result.stderr[-1200:]}")
    rows: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        values = next(csv.reader([line], delimiter="\t", quotechar='"'), [])
        values += [""] * (len(fields) - len(values))
        rows.append(dict(zip(fields, values)))
    return rows


@dataclass
class AdvPacket:
    frame: int
    timestamp_us: float
    address: str
    channel: int
    packet_counter: int | None
    gap_us: float | None


@dataclass
class AdvEvent:
    index: int
    start_us: float
    packets: list[AdvPacket]

    @property
    def gaps_us(self) -> list[float]:
        values: list[float] = []
        for previous, current in zip(self.packets, self.packets[1:]):
            if current.gap_us is not None:
                values.append(current.gap_us)
            else:
                values.append(current.timestamp_us - previous.timestamp_us)
        return values


@dataclass
class ConnPacket:
    frame: int
    timestamp_us: float
    access_address: str
    direction: str
    channel: int | None
    event_counter: int | None
    packet_counter: int | None
    gap_us: float | None
    llid: int | None
    length: int | None
    crc_valid: bool | None = None


@dataclass
class ConnEvent:
    event_counter: int
    packets: list[ConnPacket]

    @property
    def start_us(self) -> float:
        return min(packet.timestamp_us for packet in self.packets)

    def inter_frame_gaps(self, mode: str) -> list[tuple[float, str]]:
        result: list[tuple[float, str]] = []
        seen_roles: set[str] = set()
        ordered = sorted(self.packets, key=lambda packet: (packet.timestamp_us, packet.frame))
        for left, right in zip(ordered, ordered[1:]):
            if left.packet_counter is not None and right.packet_counter is not None:
                if ((left.packet_counter + 1) & 0xFFFF) != right.packet_counter:
                    continue
            if left.direction == right.direction or not left.direction or not right.direction:
                continue
            pair = "direction-change"
            if left.direction == "C2P" and right.direction == "P2C":
                pair = "central-request-response"
            elif left.direction == "P2C" and right.direction == "C2P":
                pair = "peripheral-notification-response"
            if mode == "central" and pair != "central-request-response":
                continue
            if mode == "peripheral" and pair != "peripheral-notification-response":
                continue
            # The frozen §6 definition gives each event at most one
            # measurement for each pair role.  This prevents a long event
            # with several direction changes from receiving extra weight.
            if pair in seen_roles:
                continue
            value = right.gap_us
            if value is None:
                value = right.timestamp_us - left.timestamp_us
            if value > 0:
                result.append((value, pair))
                seen_roles.add(pair)
        return result


@dataclass
class Window:
    capture_id: str
    condition: str
    traffic: str
    window_index: int
    event_count: int
    start_us: float
    end_us: float
    gap_median_us: float | None
    gap_mad_us: float | None
    event_interval_median_us: float
    event_interval_mad_us: float
    event_counter_step_median: float | None = None
    event_counter_step1_fraction: float | None = None
    observed_event_gap_count: int = 0
    pair_coverage: float | None = None
    gap_pair_counts: dict[str, int] = field(default_factory=dict)
    label: int | None = None
    gap_score: float | None = None
    event_score: float | None = None
    combined_score: float | None = None

    def as_row(self, calibration: bool = False) -> dict[str, Any]:
        return {
            "capture_id": self.capture_id,
            "condition": self.condition,
            "traffic": self.traffic,
            "window_index": self.window_index,
            "event_count": self.event_count,
            "start_us": self.start_us,
            "end_us": self.end_us,
            "gap_median_us": self.gap_median_us,
            "gap_mad_us": self.gap_mad_us,
            "gap_feature_status": "available" if self.gap_median_us is not None else "pair_missing_censored",
            "event_interval_median_us": self.event_interval_median_us,
            "event_interval_mad_us": self.event_interval_mad_us,
            "event_counter_step_median": self.event_counter_step_median,
            "event_counter_step1_fraction": self.event_counter_step1_fraction,
            "observed_event_gap_count": self.observed_event_gap_count,
            "pair_coverage": self.pair_coverage,
            "pair_counts": json.dumps(self.gap_pair_counts, sort_keys=True),
            "calibration": int(calibration),
            "label": self.label,
            "gap_score": self.gap_score,
            "event_score": self.event_score,
            "combined_score": self.combined_score,
        }


@dataclass
class Capture:
    capture_id: str
    condition: str
    path: Path
    traffic: str
    duration_s: float
    pcap_duration_s: float
    duration_source: str
    windows: list[Window]
    event_count: int
    selected_access_address: str = ""
    raw_packet_count: int = 0
    calibration: bool = False

    def summary(self) -> dict[str, Any]:
        return {
            "capture_id": self.capture_id,
            "condition": self.condition,
            "traffic": self.traffic,
            "path": str(self.path),
            "duration_s": self.duration_s,
            "pcap_duration_s": self.pcap_duration_s,
            "duration_source": self.duration_source,
            "raw_packet_count": self.raw_packet_count,
            "reconstructed_event_count": self.event_count,
            "available_window_count": len(self.windows),
            "selected_access_address": self.selected_access_address,
            "calibration": int(self.calibration),
        }


def pairwise_consecutive(left: int | None, right: int | None) -> bool:
    return left is not None and right is not None and ((left + 1) & 0xFFFF) == right


def adv_events_from_rows(
    rows: list[dict[str, str]], target: str, require_crc_valid: bool = False,
) -> tuple[list[AdvEvent], float, int]:
    packets: list[AdvPacket] = []
    all_times: list[float] = []
    target = normalize_address(target)
    for row in rows:
        timestamp = parse_float(row.get("frame.time_epoch", ""))
        if timestamp is None:
            continue
        timestamp_us = timestamp * 1_000_000.0
        all_times.append(timestamp_us)
        address = normalize_address(row.get("btle.advertising_address", ""))
        channel = parse_int(row.get("nordic_ble.channel", ""))
        if not address or (target and address != target) or channel not in {37, 38, 39}:
            continue
        crc = row.get("nordic_ble.crcok", "")
        if require_crc_valid and crc and not is_true(crc):
            continue
        packets.append(AdvPacket(
            frame=parse_int(row.get("frame.number", "")) or 0,
            timestamp_us=timestamp_us,
            address=address,
            channel=channel,
            packet_counter=parse_int(row.get("nordic_ble.packet_counter", "")),
            gap_us=parse_float(row.get("nordic_ble.delta_time", "")),
        ))
    packets.sort(key=lambda packet: (packet.timestamp_us, packet.frame))
    events: list[AdvEvent] = []
    index = 0
    while index < len(packets):
        first = packets[index]
        if first.channel != 37:
            index += 1
            continue
        candidate = [first]
        cursor = index + 1
        expected_channel = 38
        while cursor < len(packets) and expected_channel <= 39:
            next_packet = packets[cursor]
            if next_packet.channel == 37:
                break
            if next_packet.channel == expected_channel:
                previous = candidate[-1]
                if not pairwise_consecutive(previous.packet_counter, next_packet.packet_counter):
                    break
                candidate.append(next_packet)
                expected_channel += 1
            cursor += 1
        if len(candidate) == 3:
            events.append(AdvEvent(len(events), first.timestamp_us, candidate))
            index = cursor
        else:
            index += 1
    duration_s = (max(all_times) - min(all_times)) / 1_000_000.0 if len(all_times) > 1 else 0.0
    return events, duration_s, len(packets)


def adv_windows(events: list[AdvEvent], condition: str, capture_id: str, window_events: int) -> list[Window]:
    windows: list[Window] = []
    for start in range(0, len(events) - window_events + 1, window_events):
        chunk = events[start : start + window_events]
        gaps = [gap for event in chunk for gap in event.gaps_us if gap > 0]
        intervals = [right.start_us - left.start_us for left, right in zip(chunk, chunk[1:])]
        if len(gaps) < 2 * window_events or len(intervals) != window_events - 1:
            continue
        gap_center = statistics.median(gaps)
        interval_center = statistics.median(intervals)
        windows.append(Window(
            capture_id=capture_id,
            condition=condition,
            traffic="advertising",
            window_index=len(windows),
            event_count=window_events,
            start_us=chunk[0].start_us,
            end_us=chunk[-1].start_us,
            gap_median_us=gap_center,
            gap_mad_us=mad(gaps, gap_center) or 0.0,
            event_interval_median_us=interval_center,
            event_interval_mad_us=mad(intervals, interval_center) or 0.0,
        ))
    return windows


def connection_events_from_rows(
    rows: list[dict[str, str]], target_aa: str, require_crc_valid: bool = False,
) -> tuple[list[ConnEvent], float, int, str]:
    parsed: list[ConnPacket] = []
    all_times: list[float] = []
    for row in rows:
        timestamp = parse_float(row.get("frame.time_epoch", ""))
        if timestamp is None:
            continue
        timestamp_us = timestamp * 1_000_000.0
        all_times.append(timestamp_us)
        aa = normalize_hex(row.get("btle.access_address", ""))
        if not aa or aa == ADV_AA:
            continue
        crc = row.get("nordic_ble.crcok", "")
        crc_valid = is_true(crc) if crc else None
        if require_crc_valid and crc_valid is False:
            continue
        direction_value = row.get("nordic_ble.direction", "").strip().lower()
        # Nordic Sniffer exposes this Boolean as the Master -> Slave bit:
        # True is central -> peripheral and False is peripheral -> central.
        # Keep the role names explicit for request/response pairing instead
        # of treating the field as an unlabeled Boolean.
        direction = "C2P" if direction_value in {"true", "1"} else "P2C" if direction_value in {"false", "0"} else ""
        parsed.append(ConnPacket(
            frame=parse_int(row.get("frame.number", "")) or 0,
            timestamp_us=timestamp_us,
            access_address=aa,
            direction=direction,
            channel=parse_int(row.get("nordic_ble.channel", "")),
            event_counter=parse_int(row.get("nordic_ble.event_counter", "")),
            packet_counter=parse_int(row.get("nordic_ble.packet_counter", "")),
            gap_us=parse_float(row.get("nordic_ble.delta_time", "")),
            llid=parse_int(row.get("btle.data_header.llid", "")),
            length=parse_int(row.get("btle.data_header.length", "")),
            crc_valid=crc_valid,
        ))
    counts: dict[str, int] = {}
    for packet in parsed:
        if packet.event_counter is not None:
            counts[packet.access_address] = counts.get(packet.access_address, 0) + 1
    if target_aa:
        selected = normalize_hex(target_aa)
    else:
        selected = max(counts, key=counts.get) if counts else ""
    selected_packets = [packet for packet in parsed if packet.access_address == selected and packet.event_counter is not None]
    grouped: dict[int, list[ConnPacket]] = {}
    for packet in selected_packets:
        grouped.setdefault(packet.event_counter or 0, []).append(packet)
    events = [ConnEvent(counter, sorted(items, key=lambda item: (item.timestamp_us, item.frame))) for counter, items in grouped.items()]
    events.sort(key=lambda event: event.start_us)
    duration_s = (max(all_times) - min(all_times)) / 1_000_000.0 if len(all_times) > 1 else 0.0
    return events, duration_s, len(selected_packets), selected


def connection_windows(
    events: list[ConnEvent], condition: str, capture_id: str, window_events: int, pair_mode: str,
    min_pair_coverage: float, allow_event_gaps: bool = True,
) -> list[Window]:
    # The revised §6 policy allows the sniffer to miss RF events while still
    # using the event counter carried by the recovered packets.  A standard
    # connected window therefore contains 50 observed reconstructed events;
    # it does not require 50 consecutive counter values.  Keep the strict
    # mode for reproducing the old audit and for controlled comparisons.
    if allow_event_gaps:
        runs = [events]
    else:
        runs: list[list[ConnEvent]] = []
        current: list[ConnEvent] = []
        for event in events:
            if current and not pairwise_consecutive(current[-1].event_counter, event.event_counter):
                runs.append(current)
                current = []
            current.append(event)
        if current:
            runs.append(current)

    windows: list[Window] = []
    for run in runs:
        for start in range(0, len(run) - window_events + 1, window_events):
            chunk = run[start : start + window_events]
            gap_values: list[float] = []
            pair_counts: dict[str, int] = {}
            paired_events = 0
            for event in chunk:
                pairs = event.inter_frame_gaps(pair_mode)
                if pairs:
                    paired_events += 1
                    for value, name in pairs:
                        gap_values.append(value)
                        pair_counts[name] = pair_counts.get(name, 0) + 1
            coverage = paired_events / float(window_events)
            intervals: list[float] = []
            steps: list[int] = []
            for left, right in zip(chunk, chunk[1:]):
                step = (right.event_counter - left.event_counter) & 0xFFFF
                raw_interval = right.start_us - left.start_us
                if step <= 0 or raw_interval <= 0:
                    continue
                # If the monitor missed k-1 events, normalize the observed
                # timestamp span by the modular counter step.  This keeps
                # the feature an event interval rather than a monitor-yield
                # interval.
                intervals.append(raw_interval / float(step))
                steps.append(step)
            if len(intervals) != window_events - 1:
                continue
            # Pair-missing events are censored.  Keep the event interval
            # feature for the complete observed-event window, while leaving
            # the pair-gap feature missing unless the frozen coverage gate is
            # met.  No gap is filled with zero, infinity, or a cross-event
            # measurement.
            has_usable_gap = bool(gap_values) and coverage >= min_pair_coverage
            gap_center = statistics.median(gap_values) if has_usable_gap else None
            gap_mad = mad(gap_values, gap_center) if has_usable_gap else None
            interval_center = statistics.median(intervals)
            windows.append(Window(
                capture_id=capture_id,
                condition=condition,
                traffic="connection",
                window_index=len(windows),
                event_count=window_events,
                start_us=chunk[0].start_us,
                end_us=chunk[-1].start_us,
                gap_median_us=gap_center,
                gap_mad_us=gap_mad,
                event_interval_median_us=interval_center,
                event_interval_mad_us=mad(intervals, interval_center) or 0.0,
                event_counter_step_median=statistics.median(steps) if steps else None,
                event_counter_step1_fraction=(steps.count(1) / len(steps) if steps else None),
                observed_event_gap_count=sum(step > 1 for step in steps),
                pair_coverage=coverage,
                gap_pair_counts=pair_counts,
            ))
    return windows


def load_capture(
    path: Path, traffic: str, condition: str, target_address: str, target_aa: str,
    pair_mode: str, min_pair_coverage: float, capture_id: str, calibration: bool,
    allow_event_gaps: bool = True, require_crc_valid: bool = False,
) -> Capture:
    if traffic == "advertising":
        fields = [
            "frame.number", "frame.time_epoch", "btle.advertising_address",
            "nordic_ble.channel", "nordic_ble.packet_counter",
            "nordic_ble.delta_time", "nordic_ble.crcok",
        ]
        rows = tshark_rows(path, fields)
        events, duration, raw_count = adv_events_from_rows(
            rows, target_address, require_crc_valid=require_crc_valid,
        )
        windows = adv_windows(events, condition, capture_id, 30)
        selected_aa = normalize_address(target_address)
    else:
        fields = [
            "frame.number", "frame.time_epoch", "btle.access_address",
            "nordic_ble.direction", "nordic_ble.channel", "nordic_ble.event_counter",
            "nordic_ble.packet_counter", "nordic_ble.delta_time",
            "nordic_ble.crcok", "btle.data_header.llid", "btle.data_header.length",
        ]
        rows = tshark_rows(path, fields)
        events, duration, raw_count, selected_aa = connection_events_from_rows(
            rows, target_aa, require_crc_valid=require_crc_valid,
        )
        windows = connection_windows(events, condition, capture_id, 50, pair_mode,
                                     min_pair_coverage, allow_event_gaps=allow_event_gaps)
    manifest_duration = capture_manifest_duration(path)
    effective_duration = manifest_duration if manifest_duration is not None else duration
    return Capture(
        capture_id=capture_id,
        condition=condition,
        path=path,
        traffic=traffic,
        duration_s=effective_duration,
        pcap_duration_s=duration,
        duration_source="session_manifest" if manifest_duration is not None else "pcap_timestamps",
        windows=windows,
        event_count=len(events),
        selected_access_address=selected_aa,
        raw_packet_count=raw_count,
        calibration=calibration,
    )


def parse_capture_spec(value: str, default_condition: str) -> tuple[str, Path]:
    if "=" in value:
        condition, path = value.split("=", 1)
        return condition.strip() or default_condition, Path(path).expanduser().resolve()
    path = Path(value).expanduser().resolve()
    return default_condition, path


def baseline(windows: list[Window]) -> dict[str, float | None]:
    gaps = [window.gap_median_us for window in windows if window.gap_median_us is not None]
    intervals = [window.event_interval_median_us for window in windows]
    gap_median = statistics.median(gaps) if gaps else None
    interval_median = statistics.median(intervals)
    return {
        "gap_median_us": gap_median,
        "gap_mad_us": max(mad(gaps, gap_median) or 0.0, 1.0) if gaps else None,
        "event_interval_median_us": interval_median,
        "event_interval_mad_us": max(mad(intervals, interval_median) or 0.0, 1.0),
        "calibration_window_count": float(len(windows)),
        "calibration_capture_count": float(len({window.capture_id for window in windows})),
    }


def score_windows(windows: list[Window], reference: dict[str, float | None]) -> None:
    for window in windows:
        if window.gap_median_us is not None and reference["gap_median_us"] is not None:
            gap_shift = max(0.0, window.gap_median_us - reference["gap_median_us"])
            window.gap_score = gap_shift / float(reference["gap_mad_us"])
        else:
            window.gap_score = None
        event_residual = abs(window.event_interval_median_us - reference["event_interval_median_us"])
        window.event_score = event_residual / reference["event_interval_mad_us"]
        available_scores = [score for score in (window.gap_score, window.event_score) if score is not None]
        window.combined_score = max(available_scores) if available_scores else None


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return ordered[left]
    weight = position - left
    return ordered[left] * (1.0 - weight) + ordered[right] * weight


def auc_and_roc(negative: list[float], positive: list[float]) -> tuple[float | None, list[dict[str, float]], float | None]:
    if not negative or not positive:
        return None, [], None
    pair_total = len(negative) * len(positive)
    rank = sum(pos > neg for pos in positive for neg in negative)
    rank += 0.5 * sum(pos == neg for pos in positive for neg in negative)
    auc = rank / pair_total
    thresholds = [float("inf")] + sorted(set(negative + positive), reverse=True)
    points: list[dict[str, float]] = []
    for threshold in thresholds:
        # theta_5 is an upper quantile of benign scores.  A score equal to
        # that quantile is still within the benign tie mass, so deployment
        # alarms use the strict exceedance rule score > theta.
        fp = sum(value > threshold for value in negative) / len(negative)
        tp = sum(value > threshold for value in positive) / len(positive)
        points.append({"threshold": threshold, "fpr": fp, "tpr": tp})
    eligible = [point["tpr"] for point in points if point["fpr"] <= 0.05]
    return auc, points, max(eligible) if eligible else 0.0


def clustered_auc_ci(
    negative_groups: list[list[float]], positive_groups: list[list[float]], rounds: int, seed: int,
) -> tuple[float | None, float | None]:
    if len(negative_groups) < 2 or len(positive_groups) < 2:
        return None, None
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(rounds):
        negative = [value for index in range(len(negative_groups)) for value in negative_groups[rng.randrange(len(negative_groups))]]
        positive = [value for index in range(len(positive_groups)) for value in positive_groups[rng.randrange(len(positive_groups))]]
        auc, _, _ = auc_and_roc(negative, positive)
        if auc is not None:
            values.append(auc)
    if not values:
        return None, None
    return quantile(values, 0.025), quantile(values, 0.975)


def condition_metrics(
    condition: str, negative_captures: list[Capture], positive_captures: list[Capture],
    theta: dict[str, float | None], bootstrap_rounds: int, seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    points_rows: list[dict[str, Any]] = []
    for feature in ("gap_score", "event_score", "combined_score"):
        neg_groups = [
            [value for window in capture.windows if (value := getattr(window, feature)) is not None]
            for capture in negative_captures
        ]
        pos_groups = [
            [value for window in capture.windows if (value := getattr(window, feature)) is not None]
            for capture in positive_captures
        ]
        negative = [value for group in neg_groups for value in group]
        positive = [value for group in pos_groups for value in group]
        auc, points, tpr5 = auc_and_roc(negative, positive)
        ci_lo, ci_hi = clustered_auc_ci(neg_groups, pos_groups, bootstrap_rounds, seed + len(summaries))
        threshold = theta[feature]
        false_alarms = sum(value > threshold for value in negative) if threshold is not None else 0
        true_alarms = sum(value > threshold for value in positive) if threshold is not None else 0
        total_capture_minutes = sum(capture.duration_s for capture in negative_captures) / 60.0
        summaries.append({
            "traffic": positive_captures[0].traffic if positive_captures else "",
            "condition": condition,
            "feature": feature,
            "negative_capture_count": len(negative_captures),
            "positive_capture_count": len(positive_captures),
            "negative_window_count": len(negative),
            "positive_window_count": len(positive),
            "auc": auc,
            "auc_ci95_clustered_low": ci_lo,
            "auc_ci95_clustered_high": ci_hi,
            "tpr_at_5pct_fpr": tpr5,
            "theta_5": threshold,
            "test_fpr_at_theta_5": false_alarms / len(negative) if negative else None,
            "test_tpr_at_theta_5": true_alarms / len(positive) if positive else None,
            "false_alarms_per_min_at_theta_5": false_alarms / total_capture_minutes if total_capture_minutes > 0 else None,
        })
        for point in points:
            points_rows.append({"condition": condition, "feature": feature, **point})
    return summaries, points_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traffic", choices=("advertising", "connection"), required=True)
    parser.add_argument("--calibration-benign", action="append", required=True, help="pcap or condition=pcap; repeatable")
    parser.add_argument("--capture", action="append", required=True, help="condition=pcap; repeatable")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-address", default="d1:22:33:44:55:66")
    parser.add_argument("--target-aa", default="", help="connected AA; otherwise select dominant non-advertising AA")
    parser.add_argument("--pair-mode", choices=("all", "central", "peripheral"), default="all")
    parser.add_argument("--min-pair-coverage", type=float, default=0.8)
    parser.add_argument("--require-consecutive-events", action="store_true",
                        help="legacy audit mode; revised §6 default allows missing event counters")
    parser.add_argument("--require-crc-valid", action="store_true",
                        help="strict audit mode; standard timing default retains parser CRC-error rows")
    parser.add_argument("--windows-per-capture", type=int, default=0, help="freeze this count; 0 uses the shortest valid capture")
    parser.add_argument("--negative-condition", default="normal")
    parser.add_argument("--theta-quantile", type=float, default=0.95)
    parser.add_argument("--bootstrap-rounds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args(argv)
    if not 0.0 < args.min_pair_coverage <= 1.0:
        raise SystemExit("--min-pair-coverage must be in (0, 1]")
    if not 0.0 < args.theta_quantile < 1.0:
        raise SystemExit("--theta-quantile must be in (0, 1)")
    if args.windows_per_capture < 0:
        raise SystemExit("--windows-per-capture must be non-negative")
    target_address = normalize_address(args.target_address)
    if args.traffic == "connection" and target_address:
        target_address = ""

    try:
        calibration: list[Capture] = []
        for index, spec in enumerate(args.calibration_benign):
            condition, path = parse_capture_spec(spec, "calibration-benign")
            calibration.append(load_capture(
                path, args.traffic, condition, target_address, args.target_aa,
                args.pair_mode, args.min_pair_coverage, f"calibration_{index + 1}", True,
                allow_event_gaps=not args.require_consecutive_events,
                require_crc_valid=args.require_crc_valid,
            ))
        test: list[Capture] = []
        for index, spec in enumerate(args.capture):
            condition, path = parse_capture_spec(spec, "normal")
            test.append(load_capture(
                path, args.traffic, condition, target_address, args.target_aa,
                args.pair_mode, args.min_pair_coverage, f"test_{index + 1}_{condition}", False,
                allow_event_gaps=not args.require_consecutive_events,
                require_crc_valid=args.require_crc_valid,
            ))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    all_captures = calibration + test
    available = [len(capture.windows) for capture in all_captures if capture.windows]
    if not available:
        raise SystemExit("no valid fixed-size event windows reconstructed")
    frozen_windows = args.windows_per_capture or min(available)
    if frozen_windows <= 0:
        raise SystemExit("windows-per-capture resolved to zero")
    for capture in all_captures:
        capture.windows = capture.windows[:frozen_windows]
    benign_windows = [window for capture in calibration for window in capture.windows]
    if not benign_windows:
        raise SystemExit("calibration benign captures have no usable windows")
    reference = baseline(benign_windows)
    score_windows(benign_windows, reference)
    for capture in test:
        score_windows(capture.windows, reference)
        for window in capture.windows:
            window.label = 0 if capture.condition == args.negative_condition else 1

    theta: dict[str, float | None] = {}
    for feature in ("gap_score", "event_score", "combined_score"):
        values = [
            value for window in benign_windows
            if (value := getattr(window, feature)) is not None
        ]
        theta[feature] = quantile(values, args.theta_quantile) if values else None
    negative_captures = [capture for capture in test if capture.condition == args.negative_condition]
    positive_conditions = sorted({capture.condition for capture in test if capture.condition != args.negative_condition})
    summary_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    for offset, condition in enumerate(positive_conditions):
        summaries, points = condition_metrics(
            condition,
            negative_captures,
            [capture for capture in test if capture.condition == condition],
            theta,
            args.bootstrap_rounds,
            args.seed + offset * 100,
        )
        summary_rows.extend(summaries)
        point_rows.extend(points)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    capture_rows = [capture.summary() for capture in all_captures]
    window_rows = [window.as_row(calibration=capture.calibration) for capture in all_captures for window in capture.windows]
    write_csv(args.output_dir / "timing_capture_summary.csv", capture_rows)
    write_csv(args.output_dir / "timing_window_features.csv", window_rows)
    write_csv(args.output_dir / "timing_roc_summary.csv", summary_rows)
    write_csv(args.output_dir / "timing_roc_points.csv", point_rows)
    result = {
        "schema_version": 1,
        "traffic": args.traffic,
        "window_policy": {
            "events_per_window": 30 if args.traffic == "advertising" else 50,
            "connection_event_definition": "50 observed reconstructed events; event-counter gaps allowed",
            "event_interval_normalization": "timestamp delta divided by modular event-counter step",
            "event_counter_continuity_required": args.require_consecutive_events,
            "crc_policy": (
                "crc_valid_only" if args.require_crc_valid
                else "retain_target_rows_with_parser_crc_errors_for_standard_timing"
            ),
            "overlap": "none",
            "windows_per_capture": frozen_windows,
            "source": "explicit" if args.windows_per_capture else "shortest_valid_capture",
            "capture_weight": "equal",
            "within_capture_windows": "correlated",
        },
        "reconstruction": {
            "target_address": target_address,
            "target_access_address": args.target_aa,
            "pair_mode": args.pair_mode if args.traffic == "connection" else None,
            "min_pair_coverage": args.min_pair_coverage if args.traffic == "connection" else None,
        },
        "calibration": {
            "capture_count": len(calibration),
            "window_count": len(benign_windows),
            "baseline": reference,
            "theta_quantile": args.theta_quantile,
            "theta_5": theta,
        },
        "formal_test_capture_count": len(test),
        "negative_condition": args.negative_condition,
        "positive_conditions": positive_conditions,
        "outputs": {
            "capture_summary": str(args.output_dir / "timing_capture_summary.csv"),
            "window_features": str(args.output_dir / "timing_window_features.csv"),
            "roc_summary": str(args.output_dir / "timing_roc_summary.csv"),
            "roc_points": str(args.output_dir / "timing_roc_points.csv"),
        },
        "roc_summary": summary_rows,
        "notes": [
            "Advertising events require consecutive valid 37/38/39 packets for one address and packet-counter continuity within the event.",
            "Connection events are grouped by nordic_ble.event_counter; windows use 50 observed events and normalize intervals by modular counter step.",
            "Standard timing retains target rows with parser CRC errors unless --require-crc-valid is selected; PIP §6.7 remains CRC-valid-only.",
            "The calibration baseline and theta_5 are never fitted from formal positive captures.",
            "AUC CI resamples complete captures, not individual correlated windows.",
        ],
    }
    (args.output_dir / "timing_detector_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"traffic": args.traffic, "windows_per_capture": frozen_windows, "theta_5": theta, "roc_summary": summary_rows}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
