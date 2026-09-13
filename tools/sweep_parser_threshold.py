#!/usr/bin/env python3
"""Sweep the BLE parser detection threshold over frozen X310 runs.

The phone-HRS runner does not pass ``--ble-threshold`` today, so the parser
has been running with its default 0.01.  This script re-parses the same frozen
IQ files with different ``--ble-threshold`` values and re-scores each result,
so the best detection threshold can be picked without RTT ground truth.

Usage:
  python3 tools/sweep_parser_threshold.py \
    --runs 20260807_phonehrs_x31080m_los_d0.5m_rep3 \
    --thresholds 0.003,0.005,0.0075,0.01,0.015,0.02,0.03 \
    --output-dir artifacts/threshold_sweep_20260807 \
    --pattern

``--max-chunks N`` limits the parse to N chunk_samples for a quick pilot.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from tools import run_x310_bandwidth_experiment as x310_runner  # noqa: E402
from tools import run_x310_phone_hrs_experiment as hrs_runner  # noqa: E402
from tools import score_iq_parser_candidates as scorer  # noqa: E402


DEFAULT_IQ_ROOTS = [
    Path("/path/to/PhantomChannel/testdata"),
    PROJECT_ROOT / "testdata",
]


def find_run_files(run_id: str, roots: list[Path]) -> tuple[Path, Path]:
    for root in roots:
        run_root = root / run_id
        iq = run_root / "iq/capture.sc16"
        metadata = run_root / "iq/metadata.json"
        if iq.is_file() and metadata.is_file():
            return iq, metadata
    raise FileNotFoundError(f"IQ/metadata not found for {run_id} under {roots}")


def build_parser_command(
    run_id: str,
    iq_path: Path,
    metadata_path: Path,
    output_dir: Path,
    threshold: float,
    max_chunks: int,
) -> tuple[list[str], argparse.Namespace, dict[str, str]]:
    cli = hrs_runner.build_parser().parse_args([
        "--phone-ready",
        "--gain-db", "37.5",
        "--capture-id", run_id,
    ])
    config = hrs_runner.load_config(hrs_runner.DEFAULT_CONFIG)
    args = hrs_runner.build_capture_args(cli, config, run_id, "pssd")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    actual_rate = float(metadata["actual_sample_rate_sps"])
    command = x310_runner.build_parser_command(
        args,
        metadata_path,
        iq_path,
        output_dir,
        actual_rate,
    )
    command.extend(["--ble-threshold", str(threshold)])
    if max_chunks > 0:
        command.extend(["--max-chunks", str(max_chunks)])
    return command, args, x310_runner.parser_environment(args)


def run_parser(
    command: list[str],
    env: dict[str, str],
    output_dir: Path,
    iq_path: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "parse.log"
    start_ns = time.time_ns()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + shlex.join(command) + "\n")
        log.flush()
        proc = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    end_ns = time.time_ns()
    csv_path = output_dir / "ble_packets.csv"
    result = {
        "returncode": proc.returncode,
        "duration_seconds": (end_ns - start_ns) / 1_000_000_000.0,
        "log_path": str(log_path),
        "ble_packets_csv": str(csv_path),
        "valid": proc.returncode == 0 and csv_path.is_file(),
    }
    return result


def score_run(
    metadata_path: Path,
    parser_csv: Path,
    output_dir: Path,
    *,
    pattern: bool,
) -> dict[str, Any]:
    status = scorer.main([
        "--metadata", str(metadata_path),
        "--parser-csv", str(parser_csv),
        "--output-dir", str(output_dir),
        *(["--pattern"] if pattern else []),
    ])
    rate_path = output_dir / "parser_candidate_rate.json"
    rate = json.loads(rate_path.read_text(encoding="utf-8")) if rate_path.is_file() else {}
    rate["scorer_returncode"] = status
    return rate


def metric_row(run_id: str, threshold: float, rate: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "ble_threshold": threshold,
        "parser_candidates_deduplicated": rate.get("parser_candidates_deduplicated"),
        "seq_unique_count": rate.get("seq_unique_count"),
        "pattern_exact_packets": rate.get("pattern_exact_packets"),
        "psr_exact": rate.get("psr_exact"),
        "pattern_byte_recovery": rate.get("pattern_byte_recovery"),
        "pattern_ber": rate.get("pattern_ber"),
        "integrity_valid_candidates": rate.get("integrity_valid_candidates"),
        "integrity_invalid_candidates": rate.get("integrity_invalid_candidates"),
        "seq_span_theoretical_packets": rate.get("seq_span_theoretical_packets"),
        "recovered_within_span_fraction": rate.get("recovered_within_span_fraction"),
        "iq_window_parser_candidate_data_bps": rate.get("iq_window_parser_candidate_data_bps"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, help="comma-separated run ids")
    parser.add_argument("--thresholds", required=True, help="comma-separated --ble-threshold values")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iq-roots", default=",".join(str(p) for p in DEFAULT_IQ_ROOTS))
    parser.add_argument("--max-chunks", type=int, default=0, help="pilot: limit parser chunks")
    parser.add_argument("--pattern", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs = [item.strip() for item in args.runs.split(",") if item.strip()]
    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    roots = [Path(item).expanduser().resolve() for item in args.iq_roots.split(",") if item.strip()]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for run_id in runs:
        iq_path, metadata_path = find_run_files(run_id, roots)
        for threshold in thresholds:
            tag = f"{run_id}__thr{threshold:g}"
            tag_dir = output_dir / tag
            command, _, env = build_parser_command(
                run_id, iq_path, metadata_path, tag_dir / "parser", threshold, args.max_chunks
            )
            parser_result = run_parser(
                command,
                env,
                tag_dir / "parser",
                iq_path,
                metadata_path,
            )
            if not parser_result["valid"]:
                print(f"✗ {tag}: parser failed (rc={parser_result['returncode']})")
                rows.append({"run_id": run_id, "ble_threshold": threshold, "parse_failed": True})
                continue
            rate = score_run(
                metadata_path,
                Path(parser_result["ble_packets_csv"]),
                tag_dir / "score",
                pattern=args.pattern,
            )
            row = metric_row(run_id, threshold, rate)
            row["parser_duration_seconds"] = parser_result["duration_seconds"]
            rows.append(row)
            print(
                f"✓ {tag}: cand={row['seq_unique_count']} exact={row['pattern_exact_packets']} "
                f"psr_exact={row['psr_exact']} ({parser_result['duration_seconds']:.0f}s)"
            )
            with (output_dir / "sweep_summary.csv").open("w", newline="", encoding="utf-8") as handle:
                fields = list(rows[0].keys())
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)

    with (output_dir / "sweep_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
