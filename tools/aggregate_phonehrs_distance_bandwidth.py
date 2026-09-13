#!/usr/bin/env python3
"""Aggregate multi-bandwidth crops across phone-HRS distance runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


RUN_RE = re.compile(
    r"^20260807_phonehrs_x31080m_(?P<path>los|nlos)_d(?P<distance>[0-9.]+)m_rep(?P<rep>[0-9]+)$"
)


def number(value: Any) -> float | None:
    if value in (None, "", "null", "None"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def integer(value: Any) -> int | None:
    result = number(value)
    return int(result) if result is not None else None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_info(run_root: Path) -> dict[str, Any]:
    match = RUN_RE.match(run_root.name)
    if not match:
        raise ValueError(f"run name does not match expected format: {run_root.name}")
    manifest_path = run_root / "phone_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    distance = float(match.group("distance"))
    manifest_distance = number(manifest.get("distance_m"))
    return {
        "run_id": run_root.name,
        "distance_m": distance,
        "los_nlos": match.group("path").upper(),
        "repetition": int(match.group("rep")),
        "manifest_distance_m": manifest_distance,
        "distance_provenance": (
            "run_id_label_operator_confirmed_manifest_stale"
            if manifest_distance != distance
            else "run_id_and_manifest"
        ),
    }


def enrich_row(run_root: Path, row: dict[str, str]) -> dict[str, Any]:
    info = run_info(run_root)
    bandwidth = number(row.get("bandwidth_mhz"))
    bandwidth_label = f"bw{int(bandwidth)}mhz" if bandwidth is not None else ""
    rate_path = (
        run_root / "results" / "bandwidth_crops" / "crops" / bandwidth_label / "parser_candidate_rate.json"
        if bandwidth is not None
        else None
    )
    rate = json.loads(rate_path.read_text(encoding="utf-8")) if rate_path and rate_path.is_file() else {}
    unique = integer(row.get("in_band_unique_seq_candidates"))
    # Use the requested inclusive first-to-last sequence span.  The scorer's
    # dominant consecutive-run span can be shorter when channel hopping leaves
    # several separated in-band runs, which would make PSR exceed 100%.
    span = integer(rate.get("seq_span_raw_packets"))
    return {
        **info,
        "bandwidth_mhz": bandwidth,
        "C_bw": number(row.get("c_bw_candidates_estimated")),
        "PSR_exact": number(row.get("psr_exact_in_band")),
        "PSR_seqspan": unique / span if unique is not None and span else None,
        "BER": number(row.get("pattern_ber_in_band")),
        "BER_non_collision": number(rate.get("pattern_ber_excluding_outliers")),
        "G_e2e_kbps": number(row.get("g_e2e_kbps_in_band")),
        "payload_data_kbps": (
            number(row.get("payload_data_bps_in_band")) / 1000.0
            if number(row.get("payload_data_bps_in_band")) is not None
            else None
        ),
        "in_band_parser_rows": integer(row.get("in_band_parser_rows")),
        "in_band_unique_seq": unique,
        "full_band_unique_seq": integer(row.get("full_band_unique_seq_candidates")),
        "seq_span_first_last_packets": span,
    }


def summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "C_bw",
        "PSR_exact",
        "PSR_seqspan",
        "BER",
        "BER_non_collision",
        "G_e2e_kbps",
        "payload_data_kbps",
    )
    groups: dict[tuple[float, str, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["distance_m"], row["los_nlos"], row["bandwidth_mhz"])
        groups.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for (distance, path, bandwidth), selected in sorted(groups.items()):
        result: dict[str, Any] = {
            "distance_m": distance,
            "los_nlos": path,
            "rep_count": len(selected),
            "bandwidth_mhz": bandwidth,
        }
        for metric in metrics:
            values = [row[metric] for row in selected if row[metric] is not None]
            result[f"{metric}_mean"] = mean(values) if values else None
            result[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0 if values else None
        output.append(result)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    per_run: list[dict[str, Any]] = []
    for run_root in args.run_root:
        run_root = run_root.expanduser().resolve()
        metrics_path = run_root / "results" / "bandwidth_crops" / "bandwidth_crop_metrics.csv"
        if not metrics_path.is_file():
            raise SystemExit(f"missing crop metrics: {metrics_path}")
        per_run.extend(
            enrich_row(run_root, row)
            for row in read_csv(metrics_path)
            if row.get("bandwidth_mhz")
        )

    per_run_fields = [
        "run_id", "distance_m", "los_nlos", "repetition", "manifest_distance_m",
        "distance_provenance", "bandwidth_mhz", "C_bw", "PSR_exact", "PSR_seqspan",
        "BER", "BER_non_collision", "G_e2e_kbps", "payload_data_kbps",
        "in_band_parser_rows", "in_band_unique_seq", "full_band_unique_seq",
        "seq_span_first_last_packets",
    ]
    summary = summary_rows(per_run)
    summary_fields = [
        "distance_m", "los_nlos", "rep_count", "bandwidth_mhz",
        "C_bw_mean", "C_bw_std", "PSR_exact_mean", "PSR_exact_std",
        "PSR_seqspan_mean", "PSR_seqspan_std", "BER_mean", "BER_std",
        "BER_non_collision_mean", "BER_non_collision_std", "G_e2e_kbps_mean",
        "G_e2e_kbps_std", "payload_data_kbps_mean", "payload_data_kbps_std",
    ]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "phonehrs_distance_bandwidth_per_run.csv", per_run, per_run_fields)
    write_csv(output_dir / "phonehrs_distance_bandwidth_summary.csv", summary, summary_fields)
    metadata = {
        "schema_version": 1,
        "experiment": "phonehrs_x31080m_distance_bandwidth_crop",
        "distance_limit_m": 20,
        "bandwidths_mhz": sorted({row["bandwidth_mhz"] for row in per_run}),
        "runs": [str(path.expanduser().resolve()) for path in args.run_root],
        "metric_definitions": {
            "C_bw": "in-band unique-seq candidates / full-band unique-seq candidates",
            "PSR_exact": "pattern-exact packets / in-band unique-seq candidates",
            "PSR_seqspan": "in-band unique-seq candidates / inclusive first-to-last seq span",
            "BER": "pattern bit errors / compared bits",
            "G_e2e_kbps": "unique-seq PC-frame data rate in the IQ window",
        },
        "notes": [
            "Parser rows were produced by the existing one-stage X310 HRS parser.",
            "Only existing 80 MHz distance IQ/parser outputs were cropped; raw IQ was not modified.",
            "Distance uses the run-id label; stale --distance-m CLI values are retained as provenance.",
        ],
    }
    (output_dir / "phonehrs_distance_bandwidth_summary.json").write_text(
        json.dumps({"metadata": metadata, "summary": summary}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"per_run_rows": len(per_run), "summary_rows": len(summary), "output_dir": str(output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
