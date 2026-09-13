#!/usr/bin/env python3
"""Offline bandwidth-crop analysis for revision tasks 15/16/17.

Given one full-band X310 run (parser CSV + IQ metadata), this script:

  1. scores the full band (denominator for C_bw, G_e2e, PSR_exact);
  2. crops the parser rows to each target passband with crop_bandwidth.py;
  3. re-scores every cropped band;
  4. writes a per-band metric table plus the empirical channel-usage map.

The reported C_bw is estimated from the full-band observation (no RTT ground
truth): C_bw_candidates = unique-seq candidates inside the band / unique-seq
candidates observed across the full band.  A packet-level C_bw over all parser
rows is also reported.

Outputs under <output-dir>:
  bandwidth_crop_metrics.csv/json
  empirical_active_map.csv/json
  crops/bw<N>mhz/ble_packets.csv, parser_candidate_rate.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from tools import crop_bandwidth  # noqa: E402
from tools import score_iq_parser_candidates as scorer  # noqa: E402


DEFAULT_BANDWIDTHS_HZ = [2e6, 5e6, 10e6, 20e6, 40e6, 80e6]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def find_run_inputs(run_root: Path) -> tuple[Path, Path]:
    parser_csv = None
    for candidate in (
        run_root / "diagnostics/one_stage_cpp/ble_packets.csv",
        run_root / "sdr/ble_packets.csv",
    ):
        if candidate.is_file():
            parser_csv = candidate
            break
    metadata = run_root / "iq/metadata.json"
    if parser_csv is None or not metadata.is_file():
        raise SystemExit(
            f"run-root lacks parser CSV or iq/metadata.json: {run_root}\n"
            "pass --parser-csv/--metadata explicitly instead"
        )
    return parser_csv, metadata


def score_rows(
    metadata: dict[str, Any],
    rows: list[dict[str, str]],
    *,
    pattern: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return scorer.score_candidates(metadata, rows, pattern_enabled=pattern)


def metric_row(
    *,
    bandwidth_hz: float | None,
    full_summary: dict[str, Any],
    band_summary: dict[str, Any],
    crop_summary: dict[str, Any],
    in_band_rows: int,
) -> dict[str, Any]:
    full_unique = int(full_summary.get("seq_unique_count", 0) or 0)
    band_unique = int(band_summary.get("seq_unique_count", 0) or 0)
    full_candidates = int(full_summary.get("rate_candidates", 0) or 0)
    return {
        "bandwidth_mhz": bandwidth_hz / 1e6 if bandwidth_hz is not None else None,
        "in_band_parser_rows": in_band_rows,
        "full_band_parser_rows": int(crop_summary.get("input_parser_rows", 0) or 0),
        "c_bw_packets_estimated": crop_summary.get("c_bw_packets_estimated"),
        "in_band_unique_seq_candidates": band_unique,
        "full_band_unique_seq_candidates": full_unique,
        "c_bw_candidates_estimated": band_unique / full_unique if full_unique else 0.0,
        "g_e2e_bps_in_band": band_summary.get("iq_window_parser_candidate_data_bps"),
        "g_e2e_kbps_in_band": band_summary.get("iq_window_parser_candidate_data_kbps"),
        "payload_data_bps_in_band": band_summary.get("iq_window_parser_candidate_payload_data_bps"),
        "rate_candidates_in_band": band_unique,
        "rate_candidates_full_band": full_candidates,
        "psr_exact_in_band": band_summary.get("psr_exact"),
        "pattern_exact_packets_in_band": band_summary.get("pattern_exact_packets"),
        "pattern_byte_recovery_in_band": band_summary.get("pattern_byte_recovery"),
        "pattern_ber_in_band": band_summary.get("pattern_ber"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--parser-csv", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--center-freq-hz", type=float, default=None)
    parser.add_argument(
        "--bandwidths-hz",
        default=",".join(str(int(bw)) for bw in DEFAULT_BANDWIDTHS_HZ),
        help="comma-separated passbands in Hz",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--pattern",
        action="store_true",
        help="enable scorer pattern comparison (PSR_exact/BER/byte recovery)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.parser_csv is not None and args.metadata is not None:
        parser_csv = args.parser_csv.expanduser().resolve()
        metadata_path = args.metadata.expanduser().resolve()
        run_root = args.run_root.expanduser().resolve() if args.run_root else None
    elif args.run_root is not None:
        run_root = args.run_root.expanduser().resolve()
        parser_csv, metadata_path = find_run_inputs(run_root)
    else:
        raise SystemExit("provide --run-root or both --parser-csv and --metadata")

    rows = crop_bandwidth.read_csv(parser_csv)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    center_hz = args.center_freq_hz
    if center_hz is None:
        center_hz = float(metadata.get("actual_center_frequency_hz") or 2440e6)
    bandwidths = [float(item) for item in args.bandwidths_hz.split(",") if item.strip()]
    output_dir = (args.output_dir or (run_root / "results/bandwidth_crops")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    full_candidates, full_summary = score_rows(metadata, rows, pattern=args.pattern)
    write_json(output_dir / "full_band_parser_candidate_rate.json", full_summary)

    metrics: list[dict[str, Any]] = []
    for bandwidth_hz in bandwidths:
        label = f"bw{int(bandwidth_hz / 1e6)}mhz"
        band_dir = output_dir / "crops" / label
        classify = crop_bandwidth.passband_filter(center_hz, bandwidth_hz)
        in_band, in_counts, out_counts, unknown_counts = crop_bandwidth.crop_rows(rows, classify)
        fields = list(rows[0].keys()) if rows else []
        crop_bandwidth.write_csv(band_dir / "ble_packets.csv", in_band, fields)
        crop_summary = {
            "schema_version": 1,
            "run_id": str(parser_csv),
            "input_parser_rows": len(rows),
            "output_parser_rows": len(in_band),
            "mode": "passband",
            "center_frequency_hz": center_hz,
            "bandwidth_hz": bandwidth_hz,
            "band_edges_hz": [center_hz - bandwidth_hz / 2.0, center_hz + bandwidth_hz / 2.0],
            "channel_map": crop_bandwidth.channel_freq_map(rows),
            "per_channel_in_band_counts": dict(in_counts),
            "per_channel_out_of_band_counts": dict(out_counts),
            "per_channel_unknown_counts": dict(unknown_counts),
            "in_band_rows": len(in_band),
            "out_of_band_rows": sum(out_counts.values()),
            "rows_without_frequency": sum(unknown_counts.values()),
            "c_bw_packets_estimated": len(in_band)
            / (len(in_band) + sum(out_counts.values()))
            if (len(in_band) + sum(out_counts.values()))
            else 0.0,
            "notes": ["Offline crop produced by analyze_bandwidth_crops.py."],
        }
        write_json(band_dir / "crop_summary.json", crop_summary)
        band_candidates, band_summary = score_rows(metadata, in_band, pattern=args.pattern)
        write_json(band_dir / "parser_candidate_rate.json", band_summary)
        metrics.append(
            metric_row(
                bandwidth_hz=bandwidth_hz,
                full_summary=full_summary,
                band_summary=band_summary,
                crop_summary=crop_summary,
                in_band_rows=len(in_band),
            )
        )

    metrics.append(
        metric_row(
            bandwidth_hz=None,
            full_summary=full_summary,
            band_summary=full_summary,
            crop_summary={
                "input_parser_rows": len(rows),
                "c_bw_packets_estimated": 1.0,
            },
            in_band_rows=len(rows),
        )
    )
    write_json(output_dir / "bandwidth_crop_metrics.json", metrics)
    csv_fields = list(metrics[0].keys())
    with (output_dir / "bandwidth_crop_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics)

    chan_freq_map = crop_bandwidth.channel_freq_map(rows)
    per_channel = {}
    for channel, count in Counter(row.get("channel", "") for row in rows).items():
        freq_hz = chan_freq_map.get(channel)
        per_channel[channel] = {
            "center_freq_mhz": freq_hz / 1e6 if freq_hz is not None else None,
            "rows": count,
            "unique_seq_candidates": full_summary.get("unique_seq_channel_distribution", {}).get(
                channel, 0
            ),
        }
    active_map = {
        "schema_version": 1,
        "run_id": str(parser_csv),
        "channel_map": chan_freq_map,
        "per_channel": per_channel,
        "notes": [
            "Empirical active map from the full-band capture; used for C_bw (estimated).",
        ],
    }
    write_json(output_dir / "empirical_active_map.json", active_map)
    with (output_dir / "empirical_active_map.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["channel", "center_freq_mhz", "rows", "unique_seq_candidates"])
        for channel in sorted(per_channel, key=lambda item: int(item)):
            info = per_channel[channel]
            writer.writerow([channel, info["center_freq_mhz"], info["rows"], info["unique_seq_candidates"]])

    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
