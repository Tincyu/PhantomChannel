#!/usr/bin/env python3
"""Extract the frozen advertising packet-boundary detector features.

This tool deliberately operates on standard BLE observations only.  It selects
one parser candidate per target advertising packet, verifies the advertising
CRC using the fixed advertising CRC init, and computes the fixed short-window
tail-energy score.  Phantom post-CRC fields are never read by the selector.

The primary score is::

    10 log10(mean(power[b + guard:b + guard + W]) / median(idle_noise))

where ``b`` is the protocol-implied end of the legal BLE packet.  The default
parameters are the values frozen in detector_roc_experiment_redesign.md:
4 us guard, 64 us window, and an independent pre-packet idle window.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ADV_AA = "0xD6BE898E"
ADV_CRC_INIT = bytes.fromhex("555555")
TARGET_ADDRESS = "D12233445566"
TARGET_PDU = "ADV_SCAN_IND"
TARGET_CHANNEL = "39"
DEFAULT_GAP_SAMPLES = 200
DEFAULT_GUARD_US = 4.0
DEFAULT_WINDOW_US = 64.0
DEFAULT_NOISE_START_US = 1_000.0
DEFAULT_NOISE_END_US = 250.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


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


def number(value: Any, default: float | None = None) -> float | None:
    try:
        text = str(value).strip()
        if not text:
            return default
        result = float(text)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def clean_hex(value: Any) -> str:
    return re.sub(r"[^0-9a-f]", "", str(value or "").lower())


def reverse8(value: int) -> int:
    return int(f"{value:08b}"[::-1], 2)


def ble_crc(data: bytes, crc_init: bytes = ADV_CRC_INIT) -> bytes:
    """Return the three BLE CRC bytes in on-air byte order."""

    crc = int.from_bytes(crc_init, "little")
    for value in data:
        crc ^= reverse8(value) << 16
        for _ in range(8):
            crc = ((crc << 1) ^ 0x00065B) & 0xFFFFFF if crc & 0x800000 else (crc << 1) & 0xFFFFFF
    return bytes(
        (
            reverse8((crc >> 16) & 0xFF),
            reverse8((crc >> 8) & 0xFF),
            reverse8(crc & 0xFF),
        )
    )


def standard_crc_valid(row: dict[str, str]) -> bool:
    """Verify the legal advertising PDU and CRC, independent of covert bytes."""

    pdu_hex = clean_hex(row.get("dewhitened_pdu_hex"))
    crc_hex = clean_hex(row.get("captured_crc_hex"))
    if len(pdu_hex) < 4 or len(crc_hex) != 6:
        return False
    try:
        return ble_crc(bytes.fromhex(pdu_hex), ADV_CRC_INIT) == bytes.fromhex(crc_hex)
    except ValueError:
        return False


def target_candidate(row: dict[str, str]) -> bool:
    """Apply only standard, label-independent target-packet conditions."""

    return (
        row.get("packet_type", "") == "BLE_ADV"
        and row.get("access_address", "").strip().lower() == ADV_AA.lower()
        and row.get("channel", "").strip() == TARGET_CHANNEL
        and row.get("ble_pdu_type", "").strip() == TARGET_PDU
        and clean_hex(row.get("advertiser_address")) == TARGET_ADDRESS.lower()
        and int(number(row.get("payload_len"), -1) or -1) == 6
        and str(row.get("crc_capture_status", "")).strip().lower() == "ok"
        and standard_crc_valid(row)
    )


def confidence(row: dict[str, str]) -> float:
    return float(number(row.get("confidence_score"), -1.0) or -1.0)


def deduplicate_rows(rows: Iterable[dict[str, str]], gap_samples: int) -> list[dict[str, str]]:
    ordered = sorted(
        (row for row in rows if number(row.get("sample_index")) is not None),
        key=lambda row: int(number(row.get("sample_index"), -1) or -1),
    )
    selected: list[dict[str, str]] = []
    for row in ordered:
        sample = int(number(row.get("sample_index"), -1) or -1)
        if not selected:
            selected.append(row)
            continue
        previous_sample = int(number(selected[-1].get("sample_index"), -1) or -1)
        if sample - previous_sample <= gap_samples:
            if confidence(row) > confidence(selected[-1]):
                selected[-1] = row
        else:
            selected.append(row)
    return selected


def power_slice(iq: np.memmap, start: int, end: int) -> np.ndarray:
    if end <= start:
        return np.empty(0, dtype=np.float32)
    values = np.asarray(iq[start:end], dtype=np.float32)
    return values[:, 0] * values[:, 0] + values[:, 1] * values[:, 1]


def db_ratio(signal: float, noise: float) -> float:
    return float(10.0 * np.log10(max(signal, 1e-12) / max(noise, 1e-12)))


def all_parser_starts(rows: Iterable[dict[str, str]]) -> list[int]:
    starts: list[int] = []
    for row in rows:
        value = number(row.get("sample_index"))
        if value is not None:
            starts.append(int(value))
    return sorted(starts)


def has_other_parser_activity(starts: list[int], start: int, left: int, right: int, gap: int) -> bool:
    """Return true when another parser-visible burst overlaps the short tail."""

    for candidate in starts:
        if abs(candidate - start) <= gap:
            continue
        if left <= candidate <= right:
            return True
        if candidate > right:
            break
    return False


def extract_run(
    run_root: Path,
    label: int,
    *,
    gain_db: float | None = None,
    guard_us: float = DEFAULT_GUARD_US,
    window_us: float = DEFAULT_WINDOW_US,
    noise_start_us: float = DEFAULT_NOISE_START_US,
    noise_end_us: float = DEFAULT_NOISE_END_US,
    dedup_gap_samples: int = DEFAULT_GAP_SAMPLES,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata = json.loads((run_root / "iq/metadata.json").read_text(encoding="utf-8"))
    sample_rate = float(metadata.get("actual_sample_rate_sps") or metadata.get("sample_rate_sps") or 4_000_000.0)
    duration_s = float(metadata.get("received_duration_seconds") or metadata.get("duration_s") or 0.0)
    interval_ms = 20.0
    try:
        status = json.loads((run_root / "run_status.json").read_text(encoding="utf-8"))
        interval_ms = float(status.get("adv", {}).get("interval_ms") or interval_ms)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    parser_rows = read_csv(run_root / "diagnostics/one_stage_cpp/ble_packets.csv")
    target_rows = deduplicate_rows(
        (row for row in parser_rows if target_candidate(row)),
        dedup_gap_samples,
    )
    starts = all_parser_starts(parser_rows)
    iq = np.memmap(run_root / "iq/capture.sc16", dtype="<i2", mode="r").reshape(-1, 2)
    samples_per_bit = sample_rate / 1_000_000.0
    guard = int(round(guard_us * sample_rate / 1_000_000.0))
    window = int(round(window_us * sample_rate / 1_000_000.0))
    noise_start = int(round(noise_start_us * sample_rate / 1_000_000.0))
    noise_end = int(round(noise_end_us * sample_rate / 1_000_000.0))
    legal_bytes = 1 + 4 + 2 + 6 + 3
    rows: list[dict[str, Any]] = []
    invalid_window_count = 0
    collision_count = 0
    for index, packet in enumerate(target_rows):
        start = int(number(packet.get("sample_index"), -1) or -1)
        boundary = start + int(round(legal_bytes * 8 * samples_per_bit))
        tail_left = boundary + guard
        tail_right = tail_left + window
        noise_left = start - noise_start
        noise_right = start - noise_end
        complete = noise_left >= 0 and tail_right <= len(iq) and noise_right > noise_left
        if not complete:
            invalid_window_count += 1
            continue
        collision = has_other_parser_activity(starts, start, boundary - guard, tail_right, dedup_gap_samples)
        if collision:
            collision_count += 1
        legal = power_slice(iq, start, start + int(round(5 * 8 * samples_per_bit)))
        idle = power_slice(iq, noise_left, noise_right)
        tail = power_slice(iq, tail_left, tail_right)
        if not legal.size or not idle.size or not tail.size:
            invalid_window_count += 1
            continue
        noise_power = float(np.median(idle))
        rows.append(
            {
                "run_id": run_root.name,
                "label": int(label),
                "gain_db": gain_db if gain_db is not None else metadata.get("actual_gain_db", ""),
                "packet_index": index,
                "sample_index": start,
                "boundary_sample_index": boundary,
                "payload_len": 6,
                "target_pdu": TARGET_PDU,
                "standard_crc_valid": 1,
                "iq_window_complete": 1,
                "collision_ambiguous": int(collision),
                "legal_ble_snr_db": db_ratio(float(np.mean(legal)), noise_power),
                "tail_energy_db": db_ratio(float(np.mean(tail)), noise_power),
                "tail_peak_db": db_ratio(float(np.percentile(tail, 99.0)), noise_power),
                "idle_noise_power": noise_power,
            }
        )
    nominal_tx = int(round(duration_s * 1000.0 / interval_ms)) if interval_ms > 0 else 0
    summary = {
        "run_id": run_root.name,
        "label": int(label),
        "gain_db": gain_db if gain_db is not None else metadata.get("actual_gain_db", ""),
        "sample_rate_sps": sample_rate,
        "duration_s": duration_s,
        "interval_ms": interval_ms,
        "parser_rows_all": len(parser_rows),
        "target_crc_valid_dedup": len(target_rows),
        "eligible_packet_count": len(rows),
        "ambiguous_packet_count": sum(int(row["collision_ambiguous"]) for row in rows),
        "invalid_window_count": invalid_window_count,
        "ambiguous_fraction_of_target": (collision_count / len(target_rows)) if target_rows else None,
        "parser_coverage_nominal_schedule": (len(target_rows) / nominal_tx) if nominal_tx else None,
        "nominal_target_tx_packets": nominal_tx,
        "coverage_denominator_source": "nominal_advertising_schedule_20ms; no transmitter counter in capture ledger",
        "legal_ble_snr_median_db": float(np.median([row["legal_ble_snr_db"] for row in rows])) if rows else None,
        "legal_ble_snr_p05_db": float(np.percentile([row["legal_ble_snr_db"] for row in rows], 5)) if rows else None,
        "legal_ble_snr_p95_db": float(np.percentile([row["legal_ble_snr_db"] for row in rows], 95)) if rows else None,
        "tail_energy_median_db": float(np.median([row["tail_energy_db"] for row in rows])) if rows else None,
        "tail_energy_p95_db": float(np.percentile([row["tail_energy_db"] for row in rows], 95)) if rows else None,
        "tail_energy_max_db": float(np.max([row["tail_energy_db"] for row in rows])) if rows else None,
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-run-root", action="append", type=Path, default=[])
    parser.add_argument("--positive-run-root", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--guard-us", type=float, default=DEFAULT_GUARD_US)
    parser.add_argument("--window-us", type=float, default=DEFAULT_WINDOW_US)
    parser.add_argument("--noise-start-us", type=float, default=DEFAULT_NOISE_START_US)
    parser.add_argument("--noise-end-us", type=float, default=DEFAULT_NOISE_END_US)
    parser.add_argument("--dedup-gap-samples", type=int, default=DEFAULT_GAP_SAMPLES)
    args = parser.parse_args()
    if not args.negative_run_root and not args.positive_run_root:
        parser.error("at least one --negative-run-root or --positive-run-root is required")
    packet_rows: list[dict[str, Any]] = []
    capture_rows: list[dict[str, Any]] = []
    for label, roots in ((0, args.negative_run_root), (1, args.positive_run_root)):
        for root in roots:
            rows, summary = extract_run(
                root.expanduser().resolve(),
                label,
                guard_us=args.guard_us,
                window_us=args.window_us,
                noise_start_us=args.noise_start_us,
                noise_end_us=args.noise_end_us,
                dedup_gap_samples=args.dedup_gap_samples,
            )
            packet_rows.extend(rows)
            capture_rows.append(summary)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "boundary_packet_features.csv", packet_rows)
    write_csv(output_dir / "boundary_capture_summary.csv", capture_rows)
    config = {
        "schema_version": 1,
        "detector": "advertising_packet_boundary_tail_energy",
        "target_access_address": ADV_AA,
        "target_advertiser_address": TARGET_ADDRESS,
        "target_channel": int(TARGET_CHANNEL),
        "target_pdu": TARGET_PDU,
        "target_payload_len_bytes": 6,
        "standard_crc": {"algorithm": "BLE_CRC24", "init_hex": ADV_CRC_INIT.hex(), "verified_before_selection": True},
        "score": "10log10(mean(power[b+guard:b+guard+W])/median(independent_idle_noise))",
        "guard_us": args.guard_us,
        "window_us": args.window_us,
        "noise_start_us": args.noise_start_us,
        "noise_end_us": args.noise_end_us,
        "dedup_gap_samples": args.dedup_gap_samples,
        "primary_eligibility": "target CRC-valid packet, complete IQ window, no parser-visible overlapping activity in short score window",
        "label_independent_selector": True,
        "covert_post_crc_fields_used": False,
        "duration_residual": "not used by this primary score; auxiliary implementation remains separate",
    }
    (output_dir / "boundary_detector_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"capture_count": len(capture_rows), "packet_count": len(packet_rows), "output_dir": str(output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
