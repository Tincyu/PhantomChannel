#!/usr/bin/env python3
"""Match RTT PhantomChannel ground truth with SDR decoded post-CRC packets."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from statistics import median
from pathlib import Path
from typing import Any


MATCHED_FIELDS = [
    "run_id",
    "distance_m",
    "analysis_bandwidth_hz",
    "session_id",
    "seq",
    "tx_covert_hex",
    "rx_covert_hex",
    "tx_len_bytes",
    "rx_len_bytes",
    "exact_match",
    "integrity_ok",
    "bit_errors",
    "bit_error_rate",
    "packet_timestamp",
    "channel",
    "phy",
    "match_method",
]

PHANTOM_MATCH_FIELDS = [
    "run_id",
    "distance_m",
    "analysis_bandwidth_hz",
    "seq",
    "covert_marker_hex",
    "tx_covert_hex",
    "rx_covert_hex",
    "tx_covert_data_hex",
    "rx_covert_data_hex",
    "tx_len_bytes",
    "rx_len_bytes",
    "tx_data_len_bytes",
    "rx_data_len_bytes",
    "covert_exact_match",
    "covert_data_exact_match",
    "rx_payload_marker_ok",
    "covert_integrity_ok",
    "bit_errors",
    "bit_error_rate",
    "standard_pdu_observed",
    "access_address",
    "parser_access_address",
    "sdr_access_address",
    "channel",
    "tx_frequency_hz",
    "tx_in_analysis_band",
    "sdr_channel",
    "crc_init",
    "normal_pdu_len",
    "air_extra_len",
    "tx_rtt_timestamp_us",
    "sdr_timestamp_us",
    "sdr_wideband_sample_index",
    "sdr_hw_timestamp_us",
    "sdr_timestamp_status",
    "sdr_captured_crc_hex",
    "sdr_crc_capture_status",
    "sdr_dewhitened_pdu_hex",
    "sdr_confidence_score",
    "sdr_rssi",
    "sdr_cfo_hz",
    "match_method",
    "notes",
]

UNMATCHED_RTT_FIELDS = [
    "run_id",
    "distance_m",
    "analysis_bandwidth_hz",
    "seq",
    "covert_marker_hex",
    "tx_covert_hex",
    "tx_covert_data_hex",
    "tx_len_bytes",
    "tx_data_len_bytes",
    "access_address",
    "parser_access_address",
    "channel",
    "tx_frequency_hz",
    "tx_in_analysis_band",
    "crc_init",
    "normal_pdu_len",
    "air_extra_len",
    "reason",
]

UNMATCHED_SDR_FIELDS = [
    "packet_type",
    "sample_index",
    "timestamp_us",
    "wideband_sample_index",
    "hw_timestamp_us",
    "timestamp_status",
    "access_address",
    "channel",
    "dewhitened_pdu_hex",
    "captured_crc_hex",
    "crc_capture_status",
    "rx_covert_hex",
    "rx_frame_hex",
    "rx_seq",
    "covert_integrity_ok",
    "confidence_score",
    "rssi",
    "cfo_hz",
    "reason",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        field_set: list[str] = []
        for row in rows:
            for key in row:
                if key not in field_set:
                    field_set.append(key)
        fields = field_set
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["empty"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def first_present(row: dict[str, Any], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    return ""


def normalize_hex(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.removeprefix("0x").removeprefix("0X").replace(" ", "").replace(":", "")
    int(text, 16)
    if len(text) % 2:
        text = "0" + text
    return text.lower()


def normalize_aa(value: Any) -> str:
    text = normalize_hex(value)
    if len(text) > 8:
        text = text[-8:]
    return text


def normalize_seq(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def normalize_bool(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "ok", "pass", "passed"}:
        return "1"
    if text in {"0", "false", "no", "fail", "failed"}:
        return "0"
    return ""


def payload_bytes(hex_text: str) -> bytes:
    if not hex_text:
        return b""
    return bytes.fromhex(hex_text)


def int_or_empty(value: Any) -> int | str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return int(float(text))
    except ValueError:
        return ""


def float_or_none(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def payload_data_hex(payload_hex: str, marker_hex: str = "") -> str:
    payload = normalize_hex(payload_hex)
    marker = normalize_hex(marker_hex)
    if marker:
        return payload[len(marker):] if payload.startswith(marker) else ""
    return payload


def payload_marker_ok(payload_hex: str, marker_hex: str = "") -> str:
    marker = normalize_hex(marker_hex)
    if not marker:
        return ""
    try:
        payload = normalize_hex(payload_hex)
    except ValueError:
        return "0"
    return "1" if payload.startswith(marker) else "0"


def ble_data_channel_frequency_hz(channel: Any) -> int | None:
    channel_i = int_or_empty(channel)
    if not isinstance(channel_i, int):
        return None
    if 0 <= channel_i <= 10:
        return (2404 + (2 * channel_i)) * 1_000_000
    if 11 <= channel_i <= 36:
        return (2428 + (2 * (channel_i - 11))) * 1_000_000
    return None


def analysis_band_hz(metadata: dict[str, Any]) -> tuple[float, float] | None:
    center_hz = (
        float_or_none(metadata.get("actual_center_frequency_hz"))
        or float_or_none(metadata.get("center_frequency_hz"))
    )
    bandwidth_hz = float_or_none(metadata.get("analysis_bandwidth_hz"))
    if center_hz is None or bandwidth_hz is None or bandwidth_hz <= 0:
        return None
    half_bw = bandwidth_hz / 2.0
    return center_hz - half_bw, center_hz + half_bw


def channel_in_analysis_band(channel: Any, metadata: dict[str, Any]) -> tuple[int | None, bool | None]:
    frequency_hz = ble_data_channel_frequency_hz(channel)
    band = analysis_band_hz(metadata)
    if frequency_hz is None:
        return None, None
    if band is None:
        return frequency_hz, None
    low_hz, high_hz = band
    return frequency_hz, low_hz <= frequency_hz < high_hz


def phantom_frame_from_hex(hex_text: str) -> dict[str, Any] | None:
    try:
        data = bytes.fromhex(normalize_hex(hex_text))
    except ValueError:
        return None
    first_candidate: dict[str, Any] | None = None
    for start in range(0, max(0, len(data) - 5)):
        if data[start:start + 2] != b"PC":
            continue
        if start + 6 > len(data):
            continue
        seq = data[start + 2] | (data[start + 3] << 8)
        covert_len = data[start + 4]
        end = start + 5 + covert_len + 1
        if end > len(data):
            continue
        frame = data[start:end]
        check = 0
        for byte in frame[:-1]:
            check ^= byte
        payload = frame[5:-1]
        candidate = {
            "frame_hex": frame.hex(),
            "seq": str(seq),
            "payload": payload.hex(),
            "len": len(payload),
            "integrity_ok": "1" if check == frame[-1] else "0",
            "frame_offset_bytes": start,
        }
        if candidate["integrity_ok"] == "1":
            return candidate
        if first_candidate is None:
            first_candidate = candidate
    if first_candidate is not None:
        return first_candidate
    return None


def sdr_phantom_view(row: dict[str, str], index: int) -> dict[str, Any]:
    direct_payload = ""
    for field in ("rx_covert_hex", "covert_payload_hex", "phantom_payload_hex"):
        try:
            direct_payload = normalize_hex(row.get(field, ""))
        except ValueError:
            direct_payload = ""
        if direct_payload:
            break

    frame = None
    for field in (
        "rx_frame_hex",
        "covert_frame_hex",
        "phantom_frame_hex",
        "frame_hex",
        "post_crc_hex",
        "captured_post_crc_hex",
        "crc_and_post_crc_hex",
    ):
        frame = phantom_frame_from_hex(row.get(field, ""))
        if frame is not None:
            break

    payload = direct_payload or (frame["payload"] if frame else "")
    seq = normalize_seq(first_present(row, "seq", "covert_seq", "phantom_seq") or (frame["seq"] if frame else ""))
    return {
        "index": index,
        "raw": row,
        "access_address": normalize_aa(first_present(row, "access_address")),
        "channel": normalize_seq(first_present(row, "channel")),
        "seq": seq,
        "payload": payload,
        "len": first_present(row, "rx_len_bytes", "covert_len_bytes", "covert_len") or (len(payload) // 2 if payload else ""),
        "integrity_ok": normalize_bool(first_present(row, "integrity_ok", "covert_integrity_ok")) or (
            frame["integrity_ok"] if frame else ""
        ),
        "frame_hex": frame["frame_hex"] if frame else "",
        "frame_offset_bytes": frame["frame_offset_bytes"] if frame else "",
    }


def joined_rtt_tx_rows(
    rtt_ground_truth_rows: list[dict[str, str]],
    rtt_ll_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    by_seq: dict[str, dict[str, str]] = {}
    for row in rtt_ground_truth_rows:
        seq = normalize_seq(first_present(row, "seq"))
        if seq and seq not in by_seq:
            by_seq[seq] = row

    tx_rows: list[dict[str, Any]] = []
    for index, ll_row in enumerate(rtt_ll_rows):
        seq = normalize_seq(first_present(ll_row, "seq"))
        app_row = by_seq.get(seq, {})
        covert_hex = ""
        try:
            covert_hex = normalize_hex(first_present(app_row, "covert_hex", "tx_covert_hex"))
        except ValueError:
            covert_hex = ""
        marker_hex = ""
        try:
            marker_hex = normalize_hex(
                first_present(app_row, "covert_marker_hex", "payload_marker_hex")
                or first_present(ll_row, "covert_marker_hex", "payload_marker_hex")
            )
            if marker_hex and len(marker_hex) > 2:
                marker_hex = marker_hex[-2:]
        except ValueError:
            marker_hex = ""
        if not covert_hex:
            pattern = first_present(app_row, "covert_pattern", "payload_pattern") or first_present(
                ll_row, "covert_pattern", "payload_pattern"
            )
            length = first_present(app_row, "covert_len_bytes", "covert_len") or first_present(
                ll_row, "covert_len_bytes", "covert_len"
            )
            if pattern == "marker_fixed_01_to_ff" and marker_hex:
                try:
                    length_i = int(length)
                    if length_i >= 1:
                        covert_hex = marker_hex + bytes(
                            ((index % 255) + 1 for index in range(length_i - 1))
                        ).hex()
                except (TypeError, ValueError):
                    covert_hex = ""
        try:
            data_hex = normalize_hex(first_present(app_row, "covert_data_hex", "tx_covert_data_hex"))
        except ValueError:
            data_hex = ""
        if not data_hex and covert_hex:
            data_hex = payload_data_hex(covert_hex, marker_hex)
        parser_aa = normalize_aa(first_present(ll_row, "parser_access_address", "access_address"))
        air_aa = normalize_aa(first_present(ll_row, "access_address"))
        tx_rows.append(
            {
                "index": index,
                "raw": {**app_row, **ll_row},
                "run_id": first_present(ll_row, "run_id") or first_present(app_row, "run_id"),
                "seq": seq,
                "marker": marker_hex,
                "payload": covert_hex,
                "data_payload": data_hex,
                "len": first_present(app_row, "covert_len_bytes", "covert_len")
                or first_present(ll_row, "covert_len_bytes", "covert_len")
                or (len(covert_hex) // 2 if covert_hex else ""),
                "data_len": first_present(app_row, "covert_data_len_bytes", "covert_data_len")
                or (len(data_hex) // 2 if data_hex else ""),
                "access_address": air_aa,
                "parser_access_address": parser_aa,
                "crc_init": normalize_hex(first_present(ll_row, "crc_init")),
                "channel": normalize_seq(first_present(ll_row, "channel")),
                "normal_pdu_len": first_present(ll_row, "normal_pdu_len"),
                "air_extra_len": first_present(ll_row, "air_extra_len"),
                "timestamp_us": first_present(ll_row, "rtt_timestamp_us", "timestamp_us", "time_us")
                or first_present(app_row, "rtt_timestamp_us", "timestamp_us", "time_us"),
            }
        )
    return tx_rows


def sdr_row_matches_tx_packet(tx: dict[str, Any], rx: dict[str, Any]) -> bool:
    if not tx["parser_access_address"] or not rx["access_address"]:
        return False
    if rx["access_address"] != tx["parser_access_address"] and rx["access_address"] != tx["access_address"]:
        return False
    if tx["channel"] and rx["channel"] and tx["channel"] != rx["channel"]:
        return False
    return True


def make_phantom_match_row(
    tx: dict[str, Any],
    rx: dict[str, Any] | None,
    standard_pdu_observed: bool,
    method: str,
    metadata: dict[str, Any],
    notes: str = "",
) -> dict[str, Any]:
    rx_payload = rx["payload"] if rx else ""
    marker_hex = tx.get("marker", "")
    tx_data_payload = tx.get("data_payload", "") or payload_data_hex(tx["payload"], marker_hex)
    rx_data_payload = payload_data_hex(rx_payload, marker_hex) if rx_payload else ""
    errors, ber = bit_errors(tx["payload"], rx_payload)
    exact = bool(tx["payload"] and rx_payload and tx["payload"] == rx_payload)
    data_exact = bool(tx_data_payload and rx_data_payload and tx_data_payload == rx_data_payload)
    raw = rx["raw"] if rx else {}
    tx_frequency_hz, tx_in_band = channel_in_analysis_band(tx["channel"], metadata)
    return {
        "run_id": tx["run_id"] or metadata.get("run_id", ""),
        "distance_m": metadata.get("distance_m", ""),
        "analysis_bandwidth_hz": metadata.get("analysis_bandwidth_hz", ""),
        "seq": tx["seq"],
        "covert_marker_hex": marker_hex,
        "tx_covert_hex": tx["payload"],
        "rx_covert_hex": rx_payload,
        "tx_covert_data_hex": tx_data_payload,
        "rx_covert_data_hex": rx_data_payload,
        "tx_len_bytes": tx["len"],
        "rx_len_bytes": rx["len"] if rx else "",
        "tx_data_len_bytes": tx.get("data_len", "") or (len(tx_data_payload) // 2 if tx_data_payload else ""),
        "rx_data_len_bytes": len(rx_data_payload) // 2 if rx_data_payload else "",
        "covert_exact_match": int(exact),
        "covert_data_exact_match": int(data_exact),
        "rx_payload_marker_ok": payload_marker_ok(rx_payload, marker_hex) if rx else "",
        "covert_integrity_ok": rx["integrity_ok"] if rx else "",
        "bit_errors": errors,
        "bit_error_rate": ber,
        "standard_pdu_observed": int(standard_pdu_observed),
        "access_address": tx["access_address"],
        "parser_access_address": tx["parser_access_address"],
        "sdr_access_address": rx["access_address"] if rx else "",
        "channel": tx["channel"],
        "tx_frequency_hz": tx_frequency_hz or "",
        "tx_in_analysis_band": "" if tx_in_band is None else int(tx_in_band),
        "sdr_channel": rx["channel"] if rx else "",
        "crc_init": tx["crc_init"],
        "normal_pdu_len": tx["normal_pdu_len"],
        "air_extra_len": tx["air_extra_len"],
        "tx_rtt_timestamp_us": tx.get("timestamp_us", ""),
        "sdr_timestamp_us": first_present(raw, "timestamp_us"),
        "sdr_wideband_sample_index": first_present(raw, "wideband_sample_index"),
        "sdr_hw_timestamp_us": first_present(raw, "hw_timestamp_us"),
        "sdr_timestamp_status": first_present(raw, "timestamp_status"),
        "sdr_captured_crc_hex": first_present(raw, "captured_crc_hex"),
        "sdr_crc_capture_status": first_present(raw, "crc_capture_status"),
        "sdr_dewhitened_pdu_hex": first_present(raw, "dewhitened_pdu_hex"),
        "sdr_confidence_score": first_present(raw, "confidence_score"),
        "sdr_rssi": first_present(raw, "rssi"),
        "sdr_cfo_hz": first_present(raw, "cfo_hz"),
        "match_method": method,
        "notes": notes,
    }


def capture_duration_seconds(metadata: dict[str, Any]) -> float:
    samples = metadata.get("samples", "")
    sample_rate = metadata.get("actual_sample_rate_sps", metadata.get("sample_rate_sps", ""))
    try:
        samples_f = float(samples)
        rate_f = float(sample_rate)
    except (TypeError, ValueError):
        return 0.0
    return samples_f / rate_f if samples_f > 0 and rate_f > 0 else 0.0


def active_tx_window_seconds(tx_rows: list[dict[str, Any]]) -> tuple[float, float]:
    timestamps = sorted(
        timestamp
        for timestamp in (float_or_none(row.get("timestamp_us")) for row in tx_rows if row["payload"])
        if timestamp is not None
    )
    if len(timestamps) < 2:
        return 0.0, 0.0

    intervals = [
        right - left
        for left, right in zip(timestamps, timestamps[1:])
        if right > left
    ]
    if not intervals:
        return 0.0, 0.0
    interval_us = float(median(intervals))
    active_duration_us = (timestamps[-1] - timestamps[0]) + interval_us
    return active_duration_us / 1_000_000.0, interval_us / 1_000_000.0


def sequence_value(row: dict[str, Any]) -> float:
    """Return a sortable sequence value, falling back to row order."""

    value = float_or_none(row.get("seq"))
    return value if value is not None else float(row.get("index", 0))


def interpolate_tx_timestamps_us(tx_rows: list[dict[str, Any]]) -> tuple[list[float | None], int]:
    """Interpolate sparse RTT timestamps over the LL TX sequence.

    The application-side PHANTOM_TX log is intentionally sparse in some runs,
    while PHANTOM_LL_TX contains every transmitted packet.  This function
    creates an estimated RTT timestamp for every LL row without changing the
    original ground truth fields.
    """

    points = sorted(
        (sequence_value(row), timestamp)
        for row in tx_rows
        if (timestamp := float_or_none(row.get("timestamp_us"))) is not None
    )
    if len(points) < 2:
        return [None] * len(tx_rows), len(points)

    def interpolate(value: float) -> float:
        if value <= points[0][0]:
            left, right = points[0], points[1]
        elif value >= points[-1][0]:
            left, right = points[-2], points[-1]
        else:
            left, right = points[0], points[1]
            for candidate_left, candidate_right in zip(points, points[1:]):
                if candidate_left[0] <= value <= candidate_right[0]:
                    left, right = candidate_left, candidate_right
                    break
        x0, y0 = left
        x1, y1 = right
        if x1 == x0:
            return y0
        return y0 + (value - x0) * (y1 - y0) / (x1 - x0)

    return [
        float_or_none(row.get("timestamp_us"))
        if float_or_none(row.get("timestamp_us")) is not None
        else interpolate(sequence_value(row))
        for row in tx_rows
    ], len(points)


def infer_rtt_to_sdr_time_alignment(
    tx_rows: list[dict[str, Any]],
    matches: list[dict[str, Any]],
) -> dict[str, Any]:
    """Infer the relative clock mapping from RTT timestamps to SDR timestamps.

    RTT timestamps and SDR timestamps use different clocks.  Exact Phantom
    payload matches provide conservative anchors because a standard-only row
    must not be allowed to define the clock mapping.  The result is used only
    for additional time-window metrics; the legacy AA/channel metrics remain
    unchanged.
    """

    tx_by_seq = {str(row.get("seq", "")): row for row in tx_rows}
    anchors: list[tuple[float, float]] = []
    for match in matches:
        if str(match.get("covert_exact_match", "0")) != "1":
            continue
        tx = tx_by_seq.get(str(match.get("seq", "")))
        tx_timestamp = float_or_none(tx.get("timestamp_us")) if tx else None
        sdr_timestamp = float_or_none(match.get("sdr_timestamp_us"))
        if tx_timestamp is not None and sdr_timestamp is not None:
            anchors.append((tx_timestamp, sdr_timestamp))

    if len(anchors) < 4:
        return {
            "available": False,
            "source": "",
            "anchor_count": len(anchors),
            "slope": "",
            "offset_us": "",
            "residual_median_us": "",
        }

    slopes = [
        (right_y - left_y) / (right_x - left_x)
        for index, (left_x, left_y) in enumerate(anchors)
        for right_x, right_y in anchors[index + 1 :]
        if right_x != left_x
    ]
    if not slopes:
        return {
            "available": False,
            "source": "",
            "anchor_count": len(anchors),
            "slope": "",
            "offset_us": "",
            "residual_median_us": "",
        }

    slope = float(median(slopes))
    offset_us = float(median(y - slope * x for x, y in anchors))
    residuals = [abs(y - (slope * x + offset_us)) for x, y in anchors]
    return {
        "available": True,
        "source": "exact_phantom_matches",
        "anchor_count": len(anchors),
        "slope": slope,
        "offset_us": offset_us,
        "residual_median_us": float(median(residuals)),
    }


def time_aligned_capture_metrics(
    tx_rows: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    metadata: dict[str, Any],
    duration_s: float,
) -> dict[str, Any]:
    """Calculate recovery only for TX packets mapped into the IQ time window."""

    alignment = infer_rtt_to_sdr_time_alignment(tx_rows, matches)
    timestamps_us, timestamp_anchor_count = interpolate_tx_timestamps_us(tx_rows)
    result: dict[str, Any] = {
        "time_aligned_metrics_available": False,
        "time_alignment_source": alignment.get("source", ""),
        "time_alignment_anchor_count": alignment.get("anchor_count", 0),
        "time_alignment_slope": alignment.get("slope", ""),
        "time_alignment_offset_us": alignment.get("offset_us", ""),
        "time_alignment_residual_median_us": alignment.get("residual_median_us", ""),
        "tx_timestamp_anchor_count": timestamp_anchor_count,
        "sdr_capture_clock_origin_us": "",
        "time_aligned_iq_start_timestamp_us": "",
        "time_aligned_iq_end_timestamp_us": "",
        "tx_payload_packets_in_iq_capture_window": 0,
        "tx_payload_packets_outside_iq_capture_window": 0,
        "tx_payload_bytes_in_iq_capture_window": 0,
        "tx_payload_data_bytes_in_iq_capture_window": 0,
        "covert_candidate_packets_in_iq_capture_window": 0,
        "covert_exact_packets_in_iq_capture_window": 0,
        "recovered_exact_payload_bytes_in_iq_capture_window": 0,
        "recovered_exact_payload_data_bytes_in_iq_capture_window": 0,
        "covert_packet_recovery_rate_in_iq_capture_window": "",
        "covert_byte_recovery_rate_in_iq_capture_window": "",
        "covert_data_packet_recovery_rate_in_iq_capture_window": "",
        "covert_data_byte_recovery_rate_in_iq_capture_window": "",
        "time_aligned_capture_recovered_bps": "",
        "time_aligned_capture_recovered_data_bps": "",
    }
    if not alignment.get("available") or duration_s <= 0:
        return result

    slope = float(alignment["slope"])
    offset_us = float(alignment["offset_us"])
    sample_rate = float_or_none(
        metadata.get("actual_sample_rate_sps", metadata.get("sample_rate_sps", ""))
    )
    clock_origins = []
    if sample_rate and sample_rate > 0:
        for match in matches:
            sdr_timestamp_us = float_or_none(match.get("sdr_timestamp_us"))
            wideband_sample_index = float_or_none(match.get("sdr_wideband_sample_index"))
            if sdr_timestamp_us is not None and wideband_sample_index is not None:
                clock_origins.append(sdr_timestamp_us - wideband_sample_index * 1_000_000.0 / sample_rate)
    sdr_clock_origin_us = float(median(clock_origins)) if clock_origins else 0.0
    capture_start_us = sdr_clock_origin_us
    capture_end_us = capture_start_us + duration_s * 1_000_000.0
    in_window_by_seq: dict[str, bool] = {}
    tx_payload_bytes_in_window = 0
    tx_payload_data_bytes_in_window = 0
    tx_payload_packets_in_window = 0
    for row, timestamp_us in zip(tx_rows, timestamps_us):
        if not row["payload"] or timestamp_us is None:
            continue
        mapped_us = slope * timestamp_us + offset_us
        in_window = capture_start_us <= mapped_us <= capture_end_us
        in_window_by_seq[str(row.get("seq", ""))] = in_window
        if in_window:
            tx_payload_packets_in_window += 1
            tx_payload_bytes_in_window += int_or_empty(row.get("len")) or 0
            tx_payload_data_bytes_in_window += int_or_empty(row.get("data_len")) or 0

    window_matches = [
        row
        for row in matches
        if row["tx_covert_hex"] and in_window_by_seq.get(str(row.get("seq", "")), False)
    ]
    exact_packets = sum(int(row["covert_exact_match"]) for row in window_matches)
    exact_bytes = sum(
        int_or_empty(row["tx_len_bytes"]) or 0
        for row in window_matches
        if int(row["covert_exact_match"]) == 1
    )
    exact_data_packets = sum(int(row["covert_data_exact_match"]) for row in window_matches)
    exact_data_bytes = sum(
        int_or_empty(row["tx_data_len_bytes"]) or 0
        for row in window_matches
        if int(row["covert_data_exact_match"]) == 1
    )
    result.update(
        {
            "time_aligned_metrics_available": True,
            "sdr_capture_clock_origin_us": sdr_clock_origin_us,
            "time_aligned_iq_start_timestamp_us": capture_start_us,
            "time_aligned_iq_end_timestamp_us": capture_end_us,
            "tx_payload_packets_in_iq_capture_window": tx_payload_packets_in_window,
            "tx_payload_packets_outside_iq_capture_window": (
                sum(1 for row in tx_rows if row["payload"]) - tx_payload_packets_in_window
            ),
            "tx_payload_bytes_in_iq_capture_window": tx_payload_bytes_in_window,
            "tx_payload_data_bytes_in_iq_capture_window": tx_payload_data_bytes_in_window,
            "covert_candidate_packets_in_iq_capture_window": sum(
                1 for row in window_matches if row["rx_covert_hex"]
            ),
            "covert_exact_packets_in_iq_capture_window": exact_packets,
            "recovered_exact_payload_bytes_in_iq_capture_window": exact_bytes,
            "recovered_exact_payload_data_bytes_in_iq_capture_window": exact_data_bytes,
            "covert_packet_recovery_rate_in_iq_capture_window": (
                exact_packets / tx_payload_packets_in_window
                if tx_payload_packets_in_window
                else 0.0
            ),
            "covert_byte_recovery_rate_in_iq_capture_window": (
                exact_bytes / tx_payload_bytes_in_window
                if tx_payload_bytes_in_window
                else 0.0
            ),
            "covert_data_packet_recovery_rate_in_iq_capture_window": (
                exact_data_packets / tx_payload_packets_in_window
                if tx_payload_packets_in_window
                else 0.0
            ),
            "covert_data_byte_recovery_rate_in_iq_capture_window": (
                exact_data_bytes / tx_payload_data_bytes_in_window
                if tx_payload_data_bytes_in_window
                else 0.0
            ),
            "time_aligned_capture_recovered_bps": (
                exact_bytes * 8.0 / duration_s if duration_s > 0 else 0.0
            ),
            "time_aligned_capture_recovered_data_bps": (
                exact_data_bytes * 8.0 / duration_s if duration_s > 0 else 0.0
            ),
        }
    )
    return result


def score_phantom_run(
    rtt_ground_truth_rows: list[dict[str, str]],
    rtt_ll_rows: list[dict[str, str]],
    sdr_ble_rows: list[dict[str, str]],
    metadata: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    metadata = metadata or {}
    tx_rows = joined_rtt_tx_rows(rtt_ground_truth_rows, rtt_ll_rows)
    rx_rows = [sdr_phantom_view(row, index) for index, row in enumerate(sdr_ble_rows)]
    unmatched_tx = set(range(len(tx_rows)))
    unmatched_rx = set(range(len(rx_rows)))
    matches_by_tx: dict[int, dict[str, Any]] = {}
    unmatched_rtt: list[dict[str, Any]] = []
    band = analysis_band_hz(metadata)

    def seq_compatible(tx: dict[str, Any], rx: dict[str, Any]) -> bool:
        return not rx["seq"] or not tx["seq"] or rx["seq"] == tx["seq"]

    def marker_compatible(tx: dict[str, Any], rx: dict[str, Any]) -> bool:
        marker = tx.get("marker", "")
        return not marker or not rx["payload"] or payload_marker_ok(rx["payload"], marker) == "1"

    def matching_unclaimed_rx(tx: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            rx for rx in rx_rows
            if rx["index"] in unmatched_rx and sdr_row_matches_tx_packet(tx, rx)
        ]

    def claim(tx: dict[str, Any], rx: dict[str, Any], method: str, notes: str = "") -> None:
        unmatched_tx.remove(tx["index"])
        unmatched_rx.remove(rx["index"])
        matches_by_tx[tx["index"]] = make_phantom_match_row(tx, rx, True, method, metadata, notes=notes)

    # Claim high-value post-CRC recoveries globally before assigning weaker
    # standard-PDU observations, so a later exact frame cannot be consumed by an
    # earlier same-AA/channel standard-only match.
    for tx in tx_rows:
        if tx["index"] not in unmatched_tx or not tx["payload"]:
            continue
        exact_candidates = [
            rx for rx in matching_unclaimed_rx(tx)
            if rx["payload"] and rx["payload"] == tx["payload"] and seq_compatible(tx, rx)
        ]
        if not exact_candidates:
            continue
        rx = sorted(exact_candidates, key=lambda item: (0 if item["integrity_ok"] == "1" else 1, item["index"]))[0]
        claim(tx, rx, "aa_channel_seq_payload")

    for tx in tx_rows:
        if tx["index"] not in unmatched_tx:
            continue
        covert_candidates = [
            rx for rx in matching_unclaimed_rx(tx)
            if tx["payload"] and rx["payload"] and seq_compatible(tx, rx) and marker_compatible(tx, rx)
        ]
        if not covert_candidates:
            continue
        rx = sorted(
            covert_candidates,
            key=lambda item: (
                0 if item["integrity_ok"] == "1" else 1,
                bit_errors(tx["payload"], item["payload"])[0] if isinstance(bit_errors(tx["payload"], item["payload"])[0], int) else 999999,
                item["index"],
            ),
        )[0]
        claim(tx, rx, "aa_channel_covert_candidate")

    for tx in tx_rows:
        if tx["index"] not in unmatched_tx:
            continue
        standard_candidates = [
            rx for rx in matching_unclaimed_rx(tx)
            if not rx["payload"]
        ]
        if not standard_candidates:
            continue
        rx = sorted(standard_candidates, key=lambda item: item["index"])[0]
        claim(
            tx,
            rx,
            "aa_channel_standard_pdu_only",
            notes="SDR parser row has matching AA/channel but no exported Phantom frame bytes",
        )

    matches = [matches_by_tx[index] for index in sorted(matches_by_tx)]

    for tx in tx_rows:
        if tx["index"] not in unmatched_tx:
            continue
        tx_frequency_hz, tx_in_band = channel_in_analysis_band(tx["channel"], metadata)
        unmatched_rtt.append(
            {
                "run_id": tx["run_id"] or metadata.get("run_id", ""),
                "distance_m": metadata.get("distance_m", ""),
                "analysis_bandwidth_hz": metadata.get("analysis_bandwidth_hz", ""),
                "seq": tx["seq"],
                "covert_marker_hex": tx.get("marker", ""),
                "tx_covert_hex": tx["payload"],
                "tx_covert_data_hex": tx.get("data_payload", ""),
                "tx_len_bytes": tx["len"],
                "tx_data_len_bytes": tx.get("data_len", ""),
                "access_address": tx["access_address"],
                "parser_access_address": tx["parser_access_address"],
                "channel": tx["channel"],
                "tx_frequency_hz": tx_frequency_hz or "",
                "tx_in_analysis_band": "" if tx_in_band is None else int(tx_in_band),
                "crc_init": tx["crc_init"],
                "normal_pdu_len": tx["normal_pdu_len"],
                "air_extra_len": tx["air_extra_len"],
                "reason": "no_sdr_row_with_matching_aa_channel",
            }
        )

    unmatched_sdr: list[dict[str, Any]] = []
    for index in sorted(unmatched_rx):
        rx = rx_rows[index]
        raw = rx["raw"]
        unmatched_sdr.append(
            {
                "packet_type": first_present(raw, "packet_type"),
                "sample_index": first_present(raw, "sample_index"),
                "timestamp_us": first_present(raw, "timestamp_us"),
                "wideband_sample_index": first_present(raw, "wideband_sample_index"),
                "hw_timestamp_us": first_present(raw, "hw_timestamp_us"),
                "timestamp_status": first_present(raw, "timestamp_status"),
                "access_address": rx["access_address"],
                "channel": rx["channel"],
                "dewhitened_pdu_hex": first_present(raw, "dewhitened_pdu_hex"),
                "captured_crc_hex": first_present(raw, "captured_crc_hex"),
                "crc_capture_status": first_present(raw, "crc_capture_status"),
                "rx_covert_hex": rx["payload"],
                "rx_frame_hex": rx["frame_hex"],
                "rx_seq": rx["seq"],
                "covert_integrity_ok": rx["integrity_ok"],
                "confidence_score": first_present(raw, "confidence_score"),
                "rssi": first_present(raw, "rssi"),
                "cfo_hz": first_present(raw, "cfo_hz"),
                "reason": "unused_sdr_packet",
            }
        )

    tx_payload_packet_count = sum(1 for row in tx_rows if row["payload"])
    tx_payload_bytes = sum(int_or_empty(row["len"]) or 0 for row in tx_rows if row["payload"])
    tx_payload_data_bytes = sum(int_or_empty(row.get("data_len")) or 0 for row in tx_rows if row["payload"])
    tx_payload_rows = [row for row in tx_rows if row["payload"]]
    if band is None:
        tx_payload_rows_in_band = tx_payload_rows
        tx_payload_rows_unknown_band = []
    else:
        tx_payload_rows_in_band = [
            row for row in tx_payload_rows
            if channel_in_analysis_band(row["channel"], metadata)[1] is True
        ]
        tx_payload_rows_unknown_band = [
            row for row in tx_payload_rows
            if channel_in_analysis_band(row["channel"], metadata)[1] is None
        ]
    tx_payload_in_band_packet_count = len(tx_payload_rows_in_band)
    tx_payload_unknown_band_packet_count = len(tx_payload_rows_unknown_band)
    tx_payload_in_band_bytes = sum(int_or_empty(row["len"]) or 0 for row in tx_payload_rows_in_band)
    tx_payload_data_in_band_bytes = sum(int_or_empty(row.get("data_len")) or 0 for row in tx_payload_rows_in_band)
    recovered_exact_bytes = sum(
        int_or_empty(row["tx_len_bytes"]) or 0
        for row in matches
        if row["tx_covert_hex"] and int(row["covert_exact_match"]) == 1
    )
    recovered_exact_in_band_bytes = sum(
        int_or_empty(row["tx_len_bytes"]) or 0
        for row in matches
        if row["tx_covert_hex"]
        and int(row["covert_exact_match"]) == 1
        and (band is None or str(row["tx_in_analysis_band"]) == "1")
    )
    recovered_exact_data_bytes = sum(
        int_or_empty(row["tx_data_len_bytes"]) or 0
        for row in matches
        if row["tx_covert_hex"] and int(row["covert_data_exact_match"]) == 1
    )
    recovered_exact_data_in_band_bytes = sum(
        int_or_empty(row["tx_data_len_bytes"]) or 0
        for row in matches
        if row["tx_covert_hex"]
        and int(row["covert_data_exact_match"]) == 1
        and (band is None or str(row["tx_in_analysis_band"]) == "1")
    )
    total_bit_errors = sum(
        int(row["bit_errors"])
        for row in matches
        if row["tx_covert_hex"] and str(row["bit_errors"]).isdigit()
    )
    duration_s = capture_duration_seconds(metadata)
    active_duration_s, active_interval_s = active_tx_window_seconds(tx_rows)
    time_aligned_metrics = time_aligned_capture_metrics(tx_rows, matches, metadata, duration_s)
    rate_duration_s = active_duration_s if active_duration_s > 0 else duration_s
    exact_packets = sum(
        int(row["covert_exact_match"])
        for row in matches
        if row["tx_covert_hex"]
    )
    exact_packets_in_band = sum(
        int(row["covert_exact_match"])
        for row in matches
        if row["tx_covert_hex"] and (band is None or str(row["tx_in_analysis_band"]) == "1")
    )
    exact_data_packets = sum(
        int(row["covert_data_exact_match"])
        for row in matches
        if row["tx_covert_hex"]
    )
    exact_data_packets_in_band = sum(
        int(row["covert_data_exact_match"])
        for row in matches
        if row["tx_covert_hex"] and (band is None or str(row["tx_in_analysis_band"]) == "1")
    )
    standard_packets = sum(int(row["standard_pdu_observed"]) for row in matches)
    standard_packets_in_band = sum(
        int(row["standard_pdu_observed"])
        for row in matches
        if band is None or str(row["tx_in_analysis_band"]) == "1"
    )
    analysis_low_hz = band[0] if band is not None else ""
    analysis_high_hz = band[1] if band is not None else ""
    summary = {
        "schema_version": 1,
        "run_id": metadata.get("run_id", ""),
        "distance_m": metadata.get("distance_m", ""),
        "analysis_bandwidth_hz": metadata.get("analysis_bandwidth_hz", ""),
        "analysis_band_low_hz": analysis_low_hz,
        "analysis_band_high_hz": analysis_high_hz,
        "analysis_band_high_edge_policy": "exclusive" if band is not None else "",
        "tx_ll_packets": len(tx_rows),
        "tx_packets_with_payload_ground_truth": tx_payload_packet_count,
        "tx_packets_without_payload_ground_truth": len(tx_rows) - tx_payload_packet_count,
        "tx_payload_packets_in_analysis_band": tx_payload_in_band_packet_count,
        "tx_payload_packets_out_of_analysis_band": (
            tx_payload_packet_count - tx_payload_in_band_packet_count - tx_payload_unknown_band_packet_count
        ),
        "tx_payload_packets_unknown_analysis_band": tx_payload_unknown_band_packet_count,
        "sdr_ble_packets": len(rx_rows),
        "standard_pdu_observed_packets": standard_packets,
        "standard_pdu_observed_packets_in_analysis_band": standard_packets_in_band,
        "covert_candidate_packets": sum(1 for row in matches if row["tx_covert_hex"] and row["rx_covert_hex"]),
        "covert_exact_packets": exact_packets,
        "covert_exact_packets_in_analysis_band": exact_packets_in_band,
        "unmatched_rtt_packets": len(unmatched_rtt),
        "unmatched_covert_rtt_packets": sum(1 for row in unmatched_rtt if row["tx_covert_hex"]),
        "unmatched_noncovert_rtt_packets": sum(1 for row in unmatched_rtt if not row["tx_covert_hex"]),
        "unmatched_sdr_packets": len(unmatched_sdr),
        "tx_payload_bytes": tx_payload_bytes,
        "tx_payload_bytes_in_analysis_band": tx_payload_in_band_bytes,
        "tx_payload_data_bytes": tx_payload_data_bytes,
        "tx_payload_data_bytes_in_analysis_band": tx_payload_data_in_band_bytes,
        "recovered_exact_payload_bytes": recovered_exact_bytes,
        "recovered_exact_payload_bytes_in_analysis_band": recovered_exact_in_band_bytes,
        "recovered_exact_payload_data_bytes": recovered_exact_data_bytes,
        "recovered_exact_payload_data_bytes_in_analysis_band": recovered_exact_data_in_band_bytes,
        "channel_map_observable_fraction": (
            tx_payload_in_band_packet_count / tx_payload_packet_count if tx_payload_packet_count else 0.0
        ),
        "covert_packet_recovery_rate": exact_packets / tx_payload_packet_count if tx_payload_packet_count else 0.0,
        "covert_byte_recovery_rate": recovered_exact_bytes / tx_payload_bytes if tx_payload_bytes else 0.0,
        "covert_data_packet_recovery_rate": exact_data_packets / tx_payload_packet_count if tx_payload_packet_count else 0.0,
        "covert_data_byte_recovery_rate": (
            recovered_exact_data_bytes / tx_payload_data_bytes if tx_payload_data_bytes else 0.0
        ),
        "covert_in_band_packet_recovery_rate": (
            exact_packets_in_band / tx_payload_in_band_packet_count if tx_payload_in_band_packet_count else 0.0
        ),
        "covert_in_band_byte_recovery_rate": (
            recovered_exact_in_band_bytes / tx_payload_in_band_bytes if tx_payload_in_band_bytes else 0.0
        ),
        "covert_data_in_band_packet_recovery_rate": (
            exact_data_packets_in_band / tx_payload_in_band_packet_count if tx_payload_in_band_packet_count else 0.0
        ),
        "covert_data_in_band_byte_recovery_rate": (
            recovered_exact_data_in_band_bytes / tx_payload_data_in_band_bytes if tx_payload_data_in_band_bytes else 0.0
        ),
        "standard_pdu_observation_rate": standard_packets / len(tx_rows) if tx_rows else 0.0,
        "bit_errors_in_matched_candidates": total_bit_errors,
        "capture_duration_s": duration_s,
        "active_tx_duration_s": active_duration_s,
        "active_tx_interval_s": active_interval_s,
        "active_tx_theoretical_bps": (tx_payload_bytes * 8.0 / active_duration_s) if active_duration_s > 0 else 0.0,
        "active_tx_in_band_theoretical_bps": (
            tx_payload_in_band_bytes * 8.0 / active_duration_s
        ) if active_duration_s > 0 else 0.0,
        "active_tx_data_theoretical_bps": (
            tx_payload_data_bytes * 8.0 / active_duration_s
        ) if active_duration_s > 0 else 0.0,
        "active_tx_data_in_band_theoretical_bps": (
            tx_payload_data_in_band_bytes * 8.0 / active_duration_s
        ) if active_duration_s > 0 else 0.0,
        "capture_window_recovered_bps": (recovered_exact_bytes * 8.0 / duration_s) if duration_s > 0 else 0.0,
        "realistic_e2e_recovered_bps": (recovered_exact_bytes * 8.0 / rate_duration_s) if rate_duration_s > 0 else 0.0,
        "realistic_e2e_recovered_bps_in_analysis_band": (
            recovered_exact_in_band_bytes * 8.0 / rate_duration_s
        ) if rate_duration_s > 0 else 0.0,
        "realistic_e2e_recovered_data_bps": (
            recovered_exact_data_bytes * 8.0 / rate_duration_s
        ) if rate_duration_s > 0 else 0.0,
        "realistic_e2e_recovered_data_bps_in_analysis_band": (
            recovered_exact_data_in_band_bytes * 8.0 / rate_duration_s
        ) if rate_duration_s > 0 else 0.0,
        **time_aligned_metrics,
        "match_methods": {
            method: sum(1 for row in matches if row["match_method"] == method)
            for method in sorted({row["match_method"] for row in matches})
        },
        "valid": bool(tx_rows),
        "notes": [
            "captured_crc_hex is treated as the standard 3-byte BLE CRC; post_crc_hex or an equivalent field is used for Phantom frame recovery.",
            "Recovery counts require a valid Phantom frame or explicit rx_covert_hex in SDR rows.",
            "realistic_e2e_recovered_bps uses the active covert TX window inferred from RTT timestamps when available; capture_window_recovered_bps keeps the full-IQ-window rate.",
            "In-analysis-band rates use RTT TX channel ground truth and BLE data-channel frequency mapping; the high band edge is treated as exclusive.",
            "Time-aligned IQ-window metrics use exact Phantom payload matches as clock anchors, interpolate sparse RTT timestamps over LL TX sequence, and retain the legacy AA/channel metrics for comparison.",
        ],
    }
    return matches, unmatched_rtt, unmatched_sdr, summary


def bit_errors(left_hex: str, right_hex: str) -> tuple[int | str, str]:
    if not left_hex or not right_hex:
        return "", ""
    left = payload_bytes(left_hex)
    right = payload_bytes(right_hex)
    compared = min(len(left), len(right))
    errors = sum((left[idx] ^ right[idx]).bit_count() for idx in range(compared))
    errors += abs(len(left) - len(right)) * 8
    total_bits = max(len(left), len(right)) * 8
    return errors, f"{errors / total_bits:.9f}" if total_bits else ""


def tx_view(row: dict[str, str], index: int) -> dict[str, Any]:
    covert_hex = normalize_hex(first_present(row, "covert_hex", "tx_covert_hex", "payload_hex"))
    return {
        "index": index,
        "raw": row,
        "run_id": first_present(row, "run_id"),
        "session_id": first_present(row, "session_id"),
        "seq": normalize_seq(first_present(row, "seq", "sequence", "covert_seq")),
        "payload": covert_hex,
        "len": first_present(row, "covert_len_bytes", "tx_len_bytes") or (len(covert_hex) // 2 if covert_hex else ""),
        "conn_event": first_present(row, "conn_event", "event_counter", "event_counter_unwrapped"),
        "channel": first_present(row, "channel"),
        "phy": first_present(row, "phy"),
        "timestamp": first_present(row, "rtt_timestamp_us", "timestamp_us", "packet_timestamp"),
    }


def rx_view(row: dict[str, str], index: int) -> dict[str, Any]:
    covert_hex = normalize_hex(first_present(row, "rx_covert_hex", "covert_payload_hex", "covert_hex", "payload_hex"))
    return {
        "index": index,
        "raw": row,
        "run_id": first_present(row, "run_id"),
        "session_id": first_present(row, "session_id"),
        "seq": normalize_seq(first_present(row, "seq", "covert_seq", "sequence")),
        "payload": covert_hex,
        "len": first_present(row, "rx_len_bytes", "covert_len_bytes", "covert_len") or (len(covert_hex) // 2 if covert_hex else ""),
        "integrity_ok": normalize_bool(first_present(row, "integrity_ok", "covert_integrity_ok")),
        "timestamp": first_present(row, "packet_timestamp", "timestamp_us", "timestamp_s"),
        "channel": first_present(row, "channel"),
        "phy": first_present(row, "phy"),
    }


def choose_best_rx(tx: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda rx: (
            0 if tx["payload"] and rx["payload"] == tx["payload"] else 1,
            0 if rx["integrity_ok"] == "1" else 1,
            rx["index"],
        ),
    )[0]


def make_match_row(tx: dict[str, Any], rx: dict[str, Any], method: str, metadata: dict[str, str]) -> dict[str, Any]:
    errors, ber = bit_errors(tx["payload"], rx["payload"])
    exact = bool(tx["payload"] and rx["payload"] and tx["payload"] == rx["payload"])
    return {
        "run_id": tx["run_id"] or rx["run_id"] or metadata.get("run_id", ""),
        "distance_m": metadata.get("distance_m", ""),
        "analysis_bandwidth_hz": metadata.get("analysis_bandwidth_hz", ""),
        "session_id": tx["session_id"] or rx["session_id"],
        "seq": tx["seq"] or rx["seq"],
        "tx_covert_hex": tx["payload"],
        "rx_covert_hex": rx["payload"],
        "tx_len_bytes": tx["len"],
        "rx_len_bytes": rx["len"],
        "exact_match": int(exact),
        "integrity_ok": rx["integrity_ok"],
        "bit_errors": errors,
        "bit_error_rate": ber,
        "packet_timestamp": rx["timestamp"],
        "channel": rx["channel"] or tx["channel"],
        "phy": rx["phy"] or tx["phy"],
        "match_method": method,
    }


def match_packets(
    ground_truth_rows: list[dict[str, str]],
    decoded_rows: list[dict[str, str]],
    metadata: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    metadata = metadata or {}
    tx_rows = [tx_view(row, index) for index, row in enumerate(ground_truth_rows)]
    rx_rows = [rx_view(row, index) for index, row in enumerate(decoded_rows)]
    unmatched_tx = set(range(len(tx_rows)))
    unmatched_rx = set(range(len(rx_rows)))
    matches: list[dict[str, Any]] = []

    def add_match(tx_index: int, rx_index: int, method: str) -> None:
        matches.append(make_match_row(tx_rows[tx_index], rx_rows[rx_index], method, metadata))
        unmatched_tx.remove(tx_index)
        unmatched_rx.remove(rx_index)

    for tx_index, tx in enumerate(tx_rows):
        if tx_index not in unmatched_tx or not tx["session_id"] or not tx["seq"]:
            continue
        candidates = [
            rx_rows[idx]
            for idx in unmatched_rx
            if rx_rows[idx]["session_id"] == tx["session_id"] and rx_rows[idx]["seq"] == tx["seq"]
        ]
        rx = choose_best_rx(tx, candidates)
        if rx is not None:
            add_match(tx_index, rx["index"], "session_id_seq")

    for tx_index, tx in enumerate(tx_rows):
        if tx_index not in unmatched_tx or not tx["seq"] or not tx["payload"]:
            continue
        candidates = [
            rx_rows[idx]
            for idx in unmatched_rx
            if rx_rows[idx]["seq"] == tx["seq"] and rx_rows[idx]["payload"] == tx["payload"]
        ]
        rx = choose_best_rx(tx, candidates)
        if rx is not None:
            add_match(tx_index, rx["index"], "seq_payload")

    last_rx_index = -1
    for tx_index, tx in enumerate(tx_rows):
        if tx_index not in unmatched_tx or not tx["payload"]:
            continue
        candidates = [
            rx_rows[idx]
            for idx in sorted(unmatched_rx)
            if idx > last_rx_index and rx_rows[idx]["payload"] == tx["payload"]
        ]
        rx = choose_best_rx(tx, candidates)
        if rx is not None:
            add_match(tx_index, rx["index"], "payload_order")
            last_rx_index = rx["index"]

    unmatched_rtt = [tx_rows[idx]["raw"] for idx in sorted(unmatched_tx)]
    unmatched_sdr = [rx_rows[idx]["raw"] for idx in sorted(unmatched_rx)]
    summary = {
        "schema_version": 1,
        "tx_packets": len(tx_rows),
        "decoded_packets": len(rx_rows),
        "matched_packets": len(matches),
        "exact_packets": sum(int(row["exact_match"]) for row in matches),
        "unmatched_rtt_packets": len(unmatched_rtt),
        "unmatched_sdr_packets": len(unmatched_sdr),
        "match_methods": {
            method: sum(1 for row in matches if row["match_method"] == method)
            for method in sorted({row["match_method"] for row in matches})
        },
    }
    return matches, unmatched_rtt, unmatched_sdr, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=None, help="Experiment run directory.")
    parser.add_argument("--rtt-ground-truth", type=Path, default=None)
    parser.add_argument("--rtt-ll", type=Path, default=None)
    parser.add_argument("--sdr-ble", type=Path, default=None)
    parser.add_argument("--capture-metadata", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for RTT-SDR scorer outputs.")
    parser.add_argument("--ground-truth", type=Path, default=None, help="Legacy generic TX CSV input.")
    parser.add_argument("--decoded", type=Path, default=None, help="Legacy generic RX CSV input.")
    parser.add_argument("--output", type=Path, default=None, help="Legacy matched_packets.csv path.")
    parser.add_argument("--unmatched-rtt", type=Path, default=None)
    parser.add_argument("--unmatched-sdr", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--distance-m", default="")
    parser.add_argument("--analysis-bandwidth-hz", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    metadata = {
        "run_id": args.run_id,
        "distance_m": args.distance_m,
        "analysis_bandwidth_hz": args.analysis_bandwidth_hz,
    }

    if args.run_root is not None or args.rtt_ll is not None or args.sdr_ble is not None:
        run_root = args.run_root.resolve() if args.run_root is not None else None
        rtt_ground_truth = args.rtt_ground_truth or (run_root / "ground_truth" / "rtt_ground_truth.csv" if run_root else None)
        rtt_ll = args.rtt_ll or (run_root / "ground_truth" / "rtt_ll_tx.csv" if run_root else None)
        sdr_ble = args.sdr_ble or (run_root / "sdr" / "ble_packets.csv" if run_root else None)
        capture_metadata = args.capture_metadata or (run_root / "iq" / "metadata.json" if run_root else None)
        output_dir = args.output_dir or (run_root / "results" if run_root else None)
        if rtt_ground_truth is None or not rtt_ground_truth.is_file():
            parser.error(f"RTT ground truth CSV does not exist: {rtt_ground_truth}")
        if rtt_ll is None or not rtt_ll.is_file():
            parser.error(f"RTT LL TX CSV does not exist: {rtt_ll}")
        if sdr_ble is None or not sdr_ble.is_file():
            parser.error(f"SDR BLE CSV does not exist: {sdr_ble}")
        if output_dir is None:
            parser.error("--output-dir is required when --run-root is not used")
        if capture_metadata is not None and capture_metadata.is_file():
            metadata.update(json.loads(capture_metadata.read_text(encoding="utf-8")))
        if args.run_id:
            metadata["run_id"] = args.run_id

        matches, unmatched_rtt, unmatched_sdr, summary = score_phantom_run(
            read_csv(rtt_ground_truth),
            read_csv(rtt_ll),
            read_csv(sdr_ble),
            metadata,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(output_dir / "rtt_sdr_matches.csv", matches, PHANTOM_MATCH_FIELDS)
        write_csv(output_dir / "unmatched_rtt.csv", unmatched_rtt, UNMATCHED_RTT_FIELDS)
        write_csv(output_dir / "unmatched_sdr.csv", unmatched_sdr, UNMATCHED_SDR_FIELDS)
        write_json(output_dir / "recovery_metrics.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["valid"] else 2

    if args.ground_truth is None or not args.ground_truth.is_file():
        parser.error(f"ground truth CSV does not exist: {args.ground_truth}")
    if args.decoded is None or not args.decoded.is_file():
        parser.error(f"decoded SDR CSV does not exist: {args.decoded}")
    if args.output is None:
        parser.error("--output is required for legacy generic matching")

    matches, unmatched_rtt, unmatched_sdr, summary = match_packets(
        read_csv(args.ground_truth),
        read_csv(args.decoded),
        metadata,
    )
    write_csv(args.output, matches, MATCHED_FIELDS)
    write_csv(args.unmatched_rtt or args.output.with_name("unmatched_rtt.csv"), unmatched_rtt)
    write_csv(args.unmatched_sdr or args.output.with_name("unmatched_sdr.csv"), unmatched_sdr)
    write_json(args.summary or args.output.with_name("match_summary.json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
