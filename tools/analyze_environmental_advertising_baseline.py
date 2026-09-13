#!/usr/bin/env python3
"""Build the heterogeneous environmental benign timing baseline.

The source pcaps are existing, IQ-free nRF Sniffer captures.  Each selected
advertiser is filtered by AdvA before event reconstruction, so packets from
other devices in the same monitor session cannot become false events.  CRC is
reported as a coverage diagnostic only; it never filters the timing rows.

This is deliberately a separate report from the matched nRF52840 ROC.  It
does not pool environmental negatives with the single matched positive class.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import analyze_event_timing_detector as detector  # noqa: E402


EXPERIMENT_ROOT = Path("/path/to/PhantomChannel/experiments")
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "figure/detector_roc_20260810/timing_formal/environmental_benign"
FROZEN_SUMMARY = EXPERIMENT_ROOT / "figure/detector_roc_20260810/timing_formal/advertising/timing_detector_summary.json"
MATCHED_WINDOW_FEATURES = FROZEN_SUMMARY.parent / "timing_window_features.csv"

FIELDS = [
    "frame.number", "frame.time_epoch", "btle.advertising_address",
    "nordic_ble.channel", "nordic_ble.packet_counter", "nordic_ble.delta_time",
    "nordic_ble.crcok", "btcommon.eir_ad.entry.company_id",
    # Do not request decoded device_name here.  Some legacy pcaps contain
    # arbitrary bytes that Wireshark exposes as an unescaped line break; that
    # can shift tabular fields and corrupt the timestamp column.  The timing
    # pass only needs the stable company-id evidence below.
]

# These are existing, separately captured files.  Some monitor sessions carry
# more than one environmental advertiser; that is allowed, but the manifest
# records the shared source session explicitly rather than pretending the
# source was dedicated to one device.
SELECTION: tuple[dict[str, Any], ...] = (
    {
        "device_id": "env_dev_01",
        "adv_address": "02:fb:ab:00:d0:af",
        "family": "Microsoft-family",
        "company_id": "0x0006",
        "captures": (
            "adv_osb/adv_every/adv_every_128B.pcapng",
            "adv_osb/adv_every/adv_every_240B.pcapng",
        ),
    },
    {
        "device_id": "env_dev_02",
        "adv_address": "25:eb:93:cf:fe:44",
        "family": "Microsoft-family",
        "company_id": "0x0006",
        "captures": (
            "adv_osb/adv_last/adv_last_128B.pcapng",
            "adv_osb/adv_last/adv_last_240B.pcapng",
        ),
    },
    {
        "device_id": "env_dev_03",
        "adv_address": "54:cc:17:da:ff:7e",
        "family": "Apple-family",
        "company_id": "0x004c",
        "captures": (
            "adv_osb/adv_every/adv_every_128B.pcapng",
            "adv_osb/adv_every/adv_every_240B.pcapng",
        ),
    },
    {
        "device_id": "env_dev_04",
        "adv_address": "d4:6c:27:29:c3:14",
        "family": "Xiaomi-family",
        "company_id": "0x038f",
        "captures": (
            "adv_osb/adv_last/adv_last_16B.pcapng",
            "adv_osb/adv_last/adv_last_32B.pcapng",
        ),
    },
    {
        "device_id": "env_dev_05",
        "adv_address": "dc:49:52:e7:18:42",
        "family": "Xiaomi-family",
        "company_id": "0x038f",
        "captures": (
            "adv_osb/adv_every/adv_every_16B.pcapng",
            "adv_osb/adv_every/adv_every_32B.pcapng",
        ),
    },
)


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


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def p95(values: Iterable[float]) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * 0.95
    left = math.floor(position)
    right = math.ceil(position)
    if left == right:
        return ordered[left]
    weight = position - left
    return ordered[left] * (1.0 - weight) + ordered[right] * weight


def robust_stats(values: Iterable[float]) -> dict[str, float | int | None]:
    ordered = list(values)
    center = statistics.median(ordered) if ordered else None
    return {
        "count": len(ordered),
        "median_us": center,
        "mad_us": statistics.median(abs(value - center) for value in ordered) if center is not None else None,
        "p95_us": p95(ordered),
    }


def score_value(row: dict[str, Any], feature: str) -> float | None:
    value = finite_float(row.get(feature))
    return value


def auc_from_rows(
    negative_rows: list[dict[str, Any]],
    positive_rows: list[dict[str, Any]],
    feature: str,
    weighting: str,
) -> float | None:
    """Compute a diagnostic AUC without changing the frozen score direction.

    ``window-pooled`` gives every usable window equal weight.  The
    ``device-balanced`` weighting gives every environmental physical device
    equal total weight, then gives its captures and windows equal weight.  The
    positive matched-nRF52840 side is capture-balanced in the latter case.
    """
    negative = [(row, score_value(row, feature)) for row in negative_rows]
    positive = [(row, score_value(row, feature)) for row in positive_rows]
    negative = [(row, value) for row, value in negative if value is not None]
    positive = [(row, value) for row, value in positive if value is not None]
    if not negative or not positive:
        return None

    if weighting == "window-pooled":
        negative_weights = {id(row): 1.0 / len(negative) for row, _ in negative}
        positive_weights = {id(row): 1.0 / len(positive) for row, _ in positive}
    elif weighting == "device-balanced":
        devices = sorted({str(row.get("device_id", "")) for row, _ in negative})
        if not devices or any(not row.get("device_id") for row, _ in negative):
            return None
        device_count = len(devices)
        captures_by_device: dict[str, set[str]] = {}
        windows_by_capture: Counter[str] = Counter()
        for row, _ in negative:
            device = str(row["device_id"])
            capture = str(row.get("capture_id", ""))
            captures_by_device.setdefault(device, set()).add(capture)
            windows_by_capture[capture] += 1
        negative_weights = {}
        for row, _ in negative:
            device = str(row["device_id"])
            capture = str(row.get("capture_id", ""))
            capture_count = len(captures_by_device[device])
            negative_weights[id(row)] = 1.0 / device_count / capture_count / windows_by_capture[capture]

        positive_captures = sorted({str(row.get("capture_id", "")) for row, _ in positive})
        positive_windows = Counter(str(row.get("capture_id", "")) for row, _ in positive)
        positive_weights = {
            id(row): 1.0 / len(positive_captures) / positive_windows[str(row.get("capture_id", ""))]
            for row, _ in positive
        }
    else:
        raise ValueError(f"unknown AUC weighting: {weighting}")

    pair_weight = 0.0
    weighted_rank = 0.0
    for negative_row, negative_value in negative:
        negative_weight = negative_weights[id(negative_row)]
        for positive_row, positive_value in positive:
            weight = negative_weight * positive_weights[id(positive_row)]
            pair_weight += weight
            if positive_value > negative_value:
                weighted_rank += weight
            elif positive_value == negative_value:
                weighted_rank += 0.5 * weight
    return weighted_rank / pair_weight if pair_weight else None


def read_matched_positive_windows(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load the already-scored matched formal positive windows.

    This is intentionally read-only reuse of the frozen formal timing output;
    no environmental rows participate in calibration or threshold selection.
    """
    positives: dict[str, list[dict[str, Any]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            condition = row.get("condition", "")
            if condition not in {"append-last-239B", "append-every-239B"}:
                continue
            positives.setdefault(condition, []).append(row)
    return positives


def diagnostic_auc_row(
    *,
    analysis_scope: str,
    weighting: str,
    family: str,
    positive_condition: str,
    feature: str,
    negative_rows: list[dict[str, Any]],
    positive_rows: list[dict[str, Any]],
    provenance_status: str,
) -> dict[str, Any]:
    negative_rows = [row for row in negative_rows if score_value(row, feature) is not None]
    positive_rows = [row for row in positive_rows if score_value(row, feature) is not None]
    return {
        "analysis_scope": analysis_scope,
        "weighting": weighting,
        "family": family,
        "positive_condition": positive_condition,
        "feature": feature,
        "negative_device_count": len({row.get("device_id", "") for row in negative_rows}),
        "negative_capture_count": len({row.get("capture_id", "") for row in negative_rows}),
        "negative_window_count": len(negative_rows),
        "positive_capture_count": len({row.get("capture_id", "") for row in positive_rows}),
        "positive_window_count": len(positive_rows),
        "auc": auc_from_rows(negative_rows, positive_rows, feature, weighting),
        "score_direction": "higher_is_more_anomalous",
        "direction_transform": "none",
        "diagnostic_label": "device-confounded diagnostic AUC",
        "provenance_status": provenance_status,
    }


def compute_diagnostic_auc(
    environmental_rows: list[dict[str, Any]],
    matched_window_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute pooled, device-balanced, family and leave-one-out AUCs."""
    negative_rows = [row for row in environmental_rows if row.get("scope") == "capture"]
    positives = read_matched_positive_windows(matched_window_path)
    features = ("gap_score", "event_score", "combined_score")
    main_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    provenance = "exploratory_inventory_reuse; shared source PCAPs; diagnostic only"

    for condition, positive_rows in positives.items():
        for feature in features:
            for weighting in ("window-pooled", "device-balanced"):
                main_rows.append(diagnostic_auc_row(
                    analysis_scope="pooled" if weighting == "window-pooled" else "device-balanced",
                    weighting=weighting,
                    family="all",
                    positive_condition=condition,
                    feature=feature,
                    negative_rows=negative_rows,
                    positive_rows=positive_rows,
                    provenance_status=provenance,
                ))

            for family in sorted({str(row.get("family", "")) for row in negative_rows}):
                family_rows = [row for row in negative_rows if row.get("family") == family]
                for weighting in ("window-pooled", "device-balanced"):
                    main_rows.append(diagnostic_auc_row(
                        analysis_scope="per-family",
                        weighting=weighting,
                        family=family,
                        positive_condition=condition,
                        feature=feature,
                        negative_rows=family_rows,
                        positive_rows=positive_rows,
                        provenance_status=provenance,
                    ))

            devices = sorted({str(row.get("device_id", "")) for row in negative_rows})
            for excluded in devices:
                remaining = [row for row in negative_rows if row.get("device_id") != excluded]
                for weighting in ("window-pooled", "device-balanced"):
                    row = diagnostic_auc_row(
                        analysis_scope="leave-one-device-out",
                        weighting=weighting,
                        family="all",
                        positive_condition=condition,
                        feature=feature,
                        negative_rows=remaining,
                        positive_rows=positive_rows,
                        provenance_status=provenance,
                    )
                    row["excluded_unit"] = "device"
                    row["excluded_value"] = excluded
                    sensitivity_rows.append(row)

            sources = sorted({str(row.get("source_pcap", "")) for row in negative_rows})
            for excluded in sources:
                remaining = [row for row in negative_rows if row.get("source_pcap") != excluded]
                for weighting in ("window-pooled", "device-balanced"):
                    row = diagnostic_auc_row(
                        analysis_scope="leave-one-source-PCAP-out",
                        weighting=weighting,
                        family="all",
                        positive_condition=condition,
                        feature=feature,
                        negative_rows=remaining,
                        positive_rows=positive_rows,
                        provenance_status=provenance,
                    )
                    row["excluded_unit"] = "source_pcap"
                    row["excluded_value"] = excluded
                    sensitivity_rows.append(row)
    return main_rows, sensitivity_rows


def crc_stats(rows: list[dict[str, str]]) -> tuple[int, int, float | None]:
    known = [row.get("nordic_ble.crcok", "").strip().lower() for row in rows]
    known = [value for value in known if value]
    valid = sum(value in {"true", "1", "yes", "ok"} for value in known)
    return len(known), valid, valid / len(known) if known else None


def company_evidence(rows: list[dict[str, str]]) -> tuple[str, str]:
    values = []
    for row in rows:
        raw = (row.get("btcommon.eir_ad.entry.company_id") or "").strip().lower()
        if raw:
            try:
                values.append(f"0x{int(raw, 16):04x}")
            except ValueError:
                pass
    counts = Counter(values)
    if not counts:
        return "", ""
    company_id, count = counts.most_common(1)[0]
    return company_id, f"{company_id} observed in {count}/{len(values)} decoded manufacturer fields"


def selected_rows(rows: list[dict[str, str]], address: str) -> list[dict[str, str]]:
    target = detector.normalize_address(address)
    return [
        row for row in rows
        if detector.normalize_address(row.get("btle.advertising_address", "")) == target
    ]


def pair_rows(events: list[detector.AdvEvent]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for event_index, event in enumerate(events):
        intervals: list[float | None] = [None, None]
        if event_index:
            intervals = [
                event.start_us - events[event_index - 1].start_us,
                event.start_us - events[event_index - 1].start_us,
            ]
        for pair_index, role in enumerate(("37_to_38", "38_to_39")):
            previous = event.packets[pair_index]
            current = event.packets[pair_index + 1]
            output.append({
                "event_id": event_index,
                "pair_role": role,
                "channel_prev": previous.channel,
                "channel_next": current.channel,
                "frame_prev": previous.frame,
                "frame_next": current.frame,
                "packet_counter_prev": previous.packet_counter,
                "packet_counter_next": current.packet_counter,
                "gap_us": event.gaps_us[pair_index],
                "start_to_start_pair_us": current.timestamp_us - previous.timestamp_us,
                "event_start_to_start_us": intervals[pair_index],
            })
    return output


def load_frozen_detector() -> tuple[dict[str, float], dict[str, float], str]:
    payload = json.loads(FROZEN_SUMMARY.read_text(encoding="utf-8"))
    calibration = payload["calibration"]
    baseline = {key: float(value) for key, value in calibration["baseline"].items() if key.endswith("_us")}
    theta = {key: float(value) for key, value in calibration["theta_5"].items() if value is not None}
    return baseline, theta, str(FROZEN_SUMMARY)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--root", type=Path, default=EXPERIMENT_ROOT)
    args = parser.parse_args()

    baseline, theta, threshold_source = load_frozen_detector()
    manifest_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    detector_rows: list[dict[str, Any]] = []
    all_device_summaries: list[dict[str, Any]] = []
    seen_sources: dict[str, list[str]] = {}

    for device in SELECTION:
        address = device["adv_address"]
        device_capture_rows: list[dict[str, Any]] = []
        for capture_index, relative in enumerate(device["captures"], start=1):
            path = args.root / relative
            if not path.is_file():
                raise SystemExit(f"missing selected pcap: {path}")
            rows = detector.tshark_rows(path, FIELDS)
            target_rows = selected_rows(rows, address)
            events, pcap_duration_s, raw_target_count = detector.adv_events_from_rows(rows, address)
            windows = detector.adv_windows(events, "heterogeneous-environmental-benign", f"{device['device_id']}_r{capture_index}", 30)
            event_pairs = pair_rows(events)
            active_times = [finite_float(row.get("frame.time_epoch")) for row in target_rows]
            active_times = [value for value in active_times if value is not None]
            active_span_s = max(active_times) - min(active_times) if len(active_times) > 1 else 0.0
            crc_known, crc_valid, crc_fraction = crc_stats(target_rows)
            company_id, company_note = company_evidence(target_rows)
            channel_counts = {
                str(channel): sum(detector.parse_int(row.get("nordic_ble.channel", "")) == channel for row in target_rows)
                for channel in (37, 38, 39)
            }
            complete_37 = sum(event.packets[0].channel == 37 for event in events)
            incomplete_37 = max(0, channel_counts["37"] - complete_37)
            capture_id = f"{device['device_id']}_r{capture_index}"
            source = str(path)
            seen_sources.setdefault(source, []).append(device["device_id"])
            device_capture_rows.append({
                "capture_id": capture_id,
                "device_id": device["device_id"],
                "source_pcap": source,
                "source_session_shared_with": "",
                "adv_address": address,
                "family": device["family"],
                "company_id_manifest": device["company_id"],
                "company_id_observed": company_id,
                "company_evidence": company_note,
                "phy_stratum": "legacy_1M_assumed_from_37_38_39_adv_capture",
                "event_type_stratum": "legacy_primary_three_channel_advertising",
                "pcap_duration_s": pcap_duration_s,
                "active_adv_span_s": active_span_s,
                "raw_target_packet_count": raw_target_count,
                "channel_37_count": channel_counts["37"],
                "channel_38_count": channel_counts["38"],
                "channel_39_count": channel_counts["39"],
                "complete_event_count": len(events),
                "complete_37_38_pair_count": len(events),
                "complete_38_39_pair_count": len(events),
                "incomplete_37_candidate_count": incomplete_37,
                "incomplete_event_fraction_among_37_candidates": incomplete_37 / channel_counts["37"] if channel_counts["37"] else None,
                "crc_known_packet_count_diagnostic": crc_known,
                "crc_valid_packet_count_diagnostic": crc_valid,
                "crc_valid_coverage_diagnostic": crc_fraction,
                "available_30_event_window_count": len(windows),
                "usable_60s_capture": pcap_duration_s >= 60.0,
                "timing_crc_policy": "all target rows retained; CRC not a timing filter",
                "provenance_status": "reused_existing_pcap; no dedicated environmental session manifest",
            })
            pair_values: dict[str, list[float]] = {"37_to_38": [], "38_to_39": [], "pooled": []}
            event_intervals: list[float] = []
            for index, event in enumerate(events):
                gaps = event.gaps_us
                for pair_index, role in enumerate(("37_to_38", "38_to_39")):
                    gap = gaps[pair_index]
                    pair_values[role].append(gap)
                    pair_values["pooled"].append(gap)
                    previous = event.packets[pair_index]
                    current = event.packets[pair_index + 1]
                    raw_rows.append({
                        "capture_id": capture_id,
                        "device_id": device["device_id"],
                        "family": device["family"],
                        "adv_address": address,
                        "source_pcap": source,
                        "event_id": index,
                        "pair_role": role,
                        "channel_prev": previous.channel,
                        "channel_next": current.channel,
                        "gap_us": gap,
                        "start_to_start_pair_us": current.timestamp_us - previous.timestamp_us,
                        "event_start_to_start_us": None if index == 0 else event.start_us - events[index - 1].start_us,
                        "event_interval_residual_us": None,
                    })
                if index:
                    event_intervals.append(event.start_us - events[index - 1].start_us)
            nominal_interval = statistics.median(event_intervals) if event_intervals else None
            for row in raw_rows:
                if row["capture_id"] == capture_id and row["event_start_to_start_us"] is not None and nominal_interval is not None:
                    row["event_interval_residual_us"] = row["event_start_to_start_us"] - nominal_interval
            stats_row = {
                "capture_id": capture_id,
                "device_id": device["device_id"],
                "family": device["family"],
                "source_pcap": source,
                "observed_nominal_event_interval_us": nominal_interval,
            }
            for role, values in pair_values.items():
                stats_row.update({f"{role}_{key}": value for key, value in robust_stats(values).items()})
            stats_row.update({f"event_interval_{key}": value for key, value in robust_stats(event_intervals).items()})
            inventory_rows.append({**device_capture_rows[-1], **stats_row})

            for window in windows:
                detector.score_windows([window], baseline)
                detector_rows.append({
                    "scope": "capture",
                    "capture_id": capture_id,
                    "device_id": device["device_id"],
                    "family": device["family"],
                    "source_pcap": source,
                    "window_index": window.window_index,
                    "event_count": window.event_count,
                    "gap_median_us": window.gap_median_us,
                    "event_interval_median_us": window.event_interval_median_us,
                    "gap_score": window.gap_score,
                    "event_score": window.event_score,
                    "combined_score": window.combined_score,
                    "theta_gap_score": theta.get("gap_score"),
                    "theta_event_score": theta.get("event_score"),
                    "theta_combined_score": theta.get("combined_score"),
                    "gap_alarm": bool(window.gap_score is not None and window.gap_score > theta.get("gap_score", math.inf)),
                    "event_alarm": bool(window.event_score is not None and window.event_score > theta.get("event_score", math.inf)),
                    "combined_alarm": bool(window.combined_score is not None and window.combined_score > theta.get("combined_score", math.inf)),
                    "duration_s": pcap_duration_s,
                    "threshold_source": threshold_source,
                })

        manifest_rows.extend(device_capture_rows)

    for row in manifest_rows:
        shared = sorted(set(seen_sources.get(row["source_pcap"], [])) - {row["device_id"]})
        row["source_session_shared_with"] = ";".join(shared)

    # Aggregate alarms at physical-device cluster level, never at individual
    # window level as if windows were independent repetitions.
    for device in SELECTION:
        device_rows = [row for row in detector_rows if row["device_id"] == device["device_id"]]
        capture_rows = [row for row in manifest_rows if row["device_id"] == device["device_id"]]
        duration_s = sum(float(row["pcap_duration_s"]) for row in capture_rows)
        for scope, selected in (("device", device_rows),):
            detector_rows.append({
                "scope": scope,
                "capture_id": "",
                "device_id": device["device_id"],
                "family": device["family"],
                "source_pcap": ";".join(row["source_pcap"] for row in capture_rows),
                "window_index": "",
                "event_count": sum(int(row["event_count"]) for row in selected),
                "gap_median_us": statistics.median([row["gap_median_us"] for row in selected]) if selected else None,
                "event_interval_median_us": statistics.median([row["event_interval_median_us"] for row in selected]) if selected else None,
                "gap_score": "",
                "event_score": "",
                "combined_score": "",
                "theta_gap_score": theta.get("gap_score"),
                "theta_event_score": theta.get("event_score"),
                "theta_combined_score": theta.get("combined_score"),
                "gap_alarm": sum(bool(row["gap_alarm"]) for row in selected),
                "event_alarm": sum(bool(row["event_alarm"]) for row in selected),
                "combined_alarm": sum(bool(row["combined_alarm"]) for row in selected),
                "window_count": len(selected),
                "external_fpr_gap": sum(bool(row["gap_alarm"]) for row in selected) / len(selected) if selected else None,
                "external_fpr_event": sum(bool(row["event_alarm"]) for row in selected) / len(selected) if selected else None,
                "external_fpr_combined": sum(bool(row["combined_alarm"]) for row in selected) / len(selected) if selected else None,
                "false_alarms_per_min_gap": sum(bool(row["gap_alarm"]) for row in selected) / (duration_s / 60.0) if duration_s else None,
                "false_alarms_per_min_event": sum(bool(row["event_alarm"]) for row in selected) / (duration_s / 60.0) if duration_s else None,
                "false_alarms_per_min_combined": sum(bool(row["combined_alarm"]) for row in selected) / (duration_s / 60.0) if duration_s else None,
                "duration_s": duration_s,
                "threshold_source": threshold_source,
            })

    family_count = len({device["family"] for device in SELECTION})
    device_count = len(SELECTION)
    capture_count = len(manifest_rows)
    unique_source_count = len({row["source_pcap"] for row in manifest_rows})
    device_external_rows = [row for row in detector_rows if row.get("scope") == "device"]
    external_by_device = {
        row["device_id"]: {
            "family": row["family"],
            "window_count": row.get("window_count"),
            "external_fpr_gap": row.get("external_fpr_gap"),
            "external_fpr_event": row.get("external_fpr_event"),
            "external_fpr_combined": row.get("external_fpr_combined"),
            "false_alarms_per_min_combined": row.get("false_alarms_per_min_combined"),
        }
        for row in device_external_rows
    }
    combined_fprs = [
        float(row["external_fpr_combined"])
        for row in device_external_rows
        if row.get("external_fpr_combined") not in (None, "")
    ]
    diagnostic_auc_rows, diagnostic_auc_sensitivity_rows = compute_diagnostic_auc(
        detector_rows,
        MATCHED_WINDOW_FEATURES,
    )
    summary = {
        "baseline": "heterogeneous-environmental-benign",
        "device_count": device_count,
        "family_count": family_count,
        "family_labels": sorted({device["family"] for device in SELECTION}),
        "capture_record_count": capture_count,
        "unique_source_pcap_count": unique_source_count,
        "formal_provenance_gate_pass": unique_source_count >= 10 and all(
            row["provenance_status"] == "dedicated_environmental_session_manifest"
            for row in manifest_rows
        ),
        "required_device_count": 5,
        "required_family_count": 3,
        "required_captures_per_device": 2,
        "required_capture_duration_s": 60,
        "requirements_surface_pass": device_count >= 5 and family_count >= 3 and all(
            sum(row["device_id"] == device["device_id"] and row["usable_60s_capture"] for row in manifest_rows) >= 2
            for device in SELECTION
        ),
        "session_independence_note": "Eight unique existing source pcaps are reused across five device clusters; per-device records are two distinct monitor sessions, but the source sessions are not device-dedicated.",
        "provenance_status": "exploratory_inventory_reuse; device model/controller silicon and original session manifest are unavailable",
        "external_fpr_by_device": external_by_device,
        "external_fpr_combined_range": {
            "min": min(combined_fprs) if combined_fprs else None,
            "max": max(combined_fprs) if combined_fprs else None,
        },
        "paper_interpretation": "external FPR is high under the frozen matched threshold; timing detector must be described as a device-calibrated cue, not a cross-device universal detector",
        "timing_crc_policy": "all target rows retained; append-last/every CRC is not used to accept/reject timing rows",
        "frozen_detector": {
            "summary": threshold_source,
            "baseline": baseline,
            "theta_5": theta,
            "alarm_rule": "strict score > frozen theta_5",
        },
        "external_fpr_definition": "alarms / non-overlapping 30-event windows per physical device cluster",
        "false_alarms_per_min_definition": "alarms / summed source-pcap wall-clock minutes per physical device cluster",
        "diagnostic_auc": {
            "positive_conditions": sorted({row["positive_condition"] for row in diagnostic_auc_rows}),
            "features": ["gap_score", "event_score", "combined_score"],
            "score_direction": "higher_is_more_anomalous",
            "direction_transform": "none",
            "matched_positive_source": str(MATCHED_WINDOW_FEATURES),
            "interpretation": "device-confounded diagnostic AUC; not the matched mechanistic AUC and not a cross-device universal detector claim",
        },
        "outputs": {
            "environmental_device_manifest": str(args.output_dir / "environmental_device_manifest.csv"),
            "environmental_capture_inventory": str(args.output_dir / "environmental_capture_inventory.csv"),
            "environmental_raw_timing": str(args.output_dir / "environmental_raw_timing.csv"),
            "environmental_external_fpr": str(args.output_dir / "environmental_external_fpr.csv"),
            "environmental_diagnostic_auc": str(args.output_dir / "environmental_diagnostic_auc.csv"),
            "environmental_diagnostic_auc_sensitivity": str(args.output_dir / "environmental_diagnostic_auc_sensitivity.csv"),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "environmental_device_manifest.csv", manifest_rows)
    write_csv(args.output_dir / "environmental_capture_inventory.csv", inventory_rows)
    write_csv(args.output_dir / "environmental_raw_timing.csv", raw_rows)
    write_csv(args.output_dir / "environmental_external_fpr.csv", detector_rows)
    write_csv(args.output_dir / "environmental_diagnostic_auc.csv", diagnostic_auc_rows)
    write_csv(args.output_dir / "environmental_diagnostic_auc_sensitivity.csv", diagnostic_auc_sensitivity_rows)
    (args.output_dir / "environmental_benign_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
