#!/usr/bin/env python3
"""Measure target advertising-burst power from SC16 IQ and scored timestamps.

This is a relative, same-receiver comparison.  It reports dBFS and signal to
local-noise ratios; it does not claim calibrated antenna gain or field strength.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def db(value: float) -> float:
    return 10.0 * math.log10(max(value, 1e-15))


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(np.asarray(values), q)) if values else None


def read_events(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        rows = list(csv.DictReader(handle))
    unique: dict[int, dict[str, str]] = {}
    for row in rows:
        text = str(row.get("timestamp_us", "")).strip()
        if not text:
            continue
        key = int(round(float(text)))
        unique.setdefault(key, row)
    return [unique[key] for key in sorted(unique)]


def analyze_run(run_root: Path, events_csv: Path, signal_start_us: float, signal_end_us: float) -> dict[str, Any]:
    metadata_path = run_root / "iq" / "metadata.json"
    iq_path = run_root / "iq" / "capture.sc16"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest_path = run_root / "adv_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    rate = float(metadata["actual_sample_rate_sps"])
    raw = np.memmap(iq_path, dtype="<i2", mode="r")
    complex_count = raw.size // 2
    events = read_events(events_csv)
    burst_peak_dbfs: list[float] = []
    burst_median_dbfs: list[float] = []
    noise_dbfs: list[float] = []
    snr_db: list[float] = []
    clipped = 0
    inspected = 0

    for event in events:
        access_us = float(event["timestamp_us"])
        sig_lo = int(round((access_us + signal_start_us) * rate / 1e6))
        sig_hi = int(round((access_us + signal_end_us) * rate / 1e6))
        noise_lo = int(round((access_us - 1000.0) * rate / 1e6))
        noise_hi = int(round((access_us - 250.0) * rate / 1e6))
        if noise_lo < 0 or sig_hi > complex_count or sig_hi <= sig_lo or noise_hi <= noise_lo:
            continue
        sig_iq = raw[sig_lo * 2:sig_hi * 2].reshape(-1, 2).astype(np.float32) / 32768.0
        noise_iq = raw[noise_lo * 2:noise_hi * 2].reshape(-1, 2).astype(np.float32) / 32768.0
        sig_power = np.square(sig_iq[:, 0]) + np.square(sig_iq[:, 1])
        noise_power = np.square(noise_iq[:, 0]) + np.square(noise_iq[:, 1])
        peak_power = float(np.percentile(sig_power, 99.9))
        median_power = float(np.median(sig_power))
        local_noise = float(np.median(noise_power))
        burst_peak_dbfs.append(db(peak_power))
        burst_median_dbfs.append(db(median_power))
        noise_dbfs.append(db(local_noise))
        snr_db.append(db(median_power / max(local_noise, 1e-15)))
        clipped += int(np.count_nonzero(np.abs(sig_iq) >= (32760.0 / 32768.0)))
        inspected += int(sig_iq.size)

    def median(values: list[float]) -> float | None:
        return float(np.median(np.asarray(values))) if values else None

    return {
        "run_root": str(run_root),
        "events_csv": str(events_csv),
        "event_count_input": len(events),
        "event_count_measured": len(burst_peak_dbfs),
        "actual_gain_db": metadata.get("actual_gain_db"),
        "rx_antenna_label": manifest.get("rx_antenna_label", ""),
        "rx_antenna_model": manifest.get("rx_antenna_model", ""),
        "distance_m": manifest.get("distance_m"),
        "burst_peak_dbfs_median": median(burst_peak_dbfs),
        "burst_peak_dbfs_p05": percentile(burst_peak_dbfs, 5),
        "burst_peak_dbfs_p95": percentile(burst_peak_dbfs, 95),
        "burst_power_dbfs_median": median(burst_median_dbfs),
        "noise_power_dbfs_median": median(noise_dbfs),
        "burst_snr_db_median": median(snr_db),
        "clipped_component_fraction": clipped / inspected if inspected else None,
        "signal_window_relative_to_access_us": [signal_start_us, signal_end_us],
        "peak_definition": "per-event 99.9th-percentile complex-sample power; median across events",
    }


def parse_input(text: str) -> tuple[str, Path, Path]:
    parts = text.split("=", 1)
    if len(parts) != 2 or not parts[0].strip():
        raise ValueError("--input must be LABEL=RUN_ROOT[:EVENTS_CSV]")
    label, payload = parts[0].strip(), parts[1]
    run_text, separator, csv_text = payload.partition(":")
    run_root = Path(run_text).expanduser().resolve()
    events = Path(csv_text).expanduser().resolve() if separator else run_root / "results" / "parser_candidate_packets.csv"
    return label, run_root, events


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="LABEL=RUN_ROOT[:EVENTS_CSV]; repeatable")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--signal-start-us", type=float, default=-10.0)
    parser.add_argument("--signal-end-us", type=float, default=2100.0)
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for item in args.input:
        label, run_root, events = parse_input(item)
        result = analyze_run(run_root, events, args.signal_start_us, args.signal_end_us)
        result["label"] = label
        rows.append(result)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["label"]), []).append(row)
    group_rows: list[dict[str, Any]] = []
    for label, members in grouped.items():
        peaks = [float(row["burst_peak_dbfs_median"]) for row in members if row["burst_peak_dbfs_median"] is not None]
        snrs = [float(row["burst_snr_db_median"]) for row in members if row["burst_snr_db_median"] is not None]
        group_rows.append({
            "label": label,
            "run_count": len(members),
            "measured_event_count": sum(int(row["event_count_measured"]) for row in members),
            "burst_peak_dbfs_run_median": float(np.median(peaks)) if peaks else None,
            "burst_snr_db_run_median": float(np.median(snrs)) if snrs else None,
            "max_clipped_component_fraction": max(
                (float(row["clipped_component_fraction"]) for row in members if row["clipped_component_fraction"] is not None),
                default=None,
            ),
        })
    reference = next((row for row in group_rows if row["label"] == "default"), group_rows[0])
    ref_peak = reference["burst_peak_dbfs_run_median"]
    ref_snr = reference["burst_snr_db_run_median"]
    for row in group_rows:
        row["peak_delta_vs_default_db"] = (
            row["burst_peak_dbfs_run_median"] - ref_peak
            if row["burst_peak_dbfs_run_median"] is not None and ref_peak is not None else None
        )
        row["snr_delta_vs_default_db"] = (
            row["burst_snr_db_run_median"] - ref_snr
            if row["burst_snr_db_run_median"] is not None and ref_snr is not None else None
        )
    for row in rows:
        row["peak_delta_vs_default_group_db"] = row["burst_peak_dbfs_median"] - ref_peak if row["burst_peak_dbfs_median"] is not None and ref_peak is not None else None
        row["snr_delta_vs_default_group_db"] = row["burst_snr_db_median"] - ref_snr if row["burst_snr_db_median"] is not None and ref_snr is not None else None
    with (output / "antenna_peak_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "antenna_peak_summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with (output / "antenna_peak_by_model.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_rows[0]))
        writer.writeheader()
        writer.writerows(group_rows)
    (output / "antenna_peak_by_model.json").write_text(json.dumps(group_rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"runs": rows, "by_model": group_rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
