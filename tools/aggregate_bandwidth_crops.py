#!/usr/bin/env python3
"""Aggregate per-run bandwidth-crop metrics for a fixed experiment point.

The bandwidth crop tool writes one metric table per run.  This helper keeps
the per-run values and emits a mean/std summary across the supplied runs.
``PSR`` is the existing handoff metric ``psr_exact_in_band`` (pattern-exact
packet ratio).  ``PSR_seqspan`` is also retained as the sequence-span
estimate, using the scorer's dominant consecutive sequence span.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


RUN_RE = re.compile(r"(?P<run>20260807_phonehrs_x31080m_(?P<path>los|nlos)_d(?P<distance>[0-9.]+)m_rep(?P<rep>[0-9]+))$")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
    manifest = json.loads((run_root / "phone_manifest.json").read_text(encoding="utf-8"))
    return {
        "run_id": match.group("run"),
        "distance_label_m": float(match.group("distance")),
        "los_nlos": match.group("path").upper(),
        "repetition": int(match.group("rep")),
        "manifest_distance_m": manifest.get("distance_m"),
        "distance_provenance": "run_id_label_confirmed_by_operator;manifest_stale"
        if number(manifest.get("distance_m")) != float(match.group("distance"))
        else "run_id_and_manifest",
    }


def enrich_row(run_root: Path, row: dict[str, str]) -> dict[str, Any]:
    info = run_info(run_root)
    bandwidth = number(row.get("bandwidth_mhz"))
    seq_unique = integer(row.get("in_band_unique_seq_candidates"))
    span_path = (
        run_root
        / "results"
        / "bandwidth_crops"
        / "crops"
        / f"bw{int(bandwidth)}mhz"
        / "parser_candidate_rate.json"
    ) if bandwidth is not None else None
    span_data: dict[str, Any] = {}
    if span_path is not None and span_path.is_file():
        span_data = json.loads(span_path.read_text(encoding="utf-8"))
    span = integer(span_data.get("seq_span_theoretical_packets"))
    psr_seqspan = seq_unique / span if seq_unique and span else None
    return {
        **info,
        "bandwidth_mhz": bandwidth,
        "C_bw": number(row.get("c_bw_candidates_estimated")),
        "PSR": number(row.get("psr_exact_in_band")),
        "PSR_seqspan": psr_seqspan,
        "BER": number(row.get("pattern_ber_in_band")),
        "BER_non_collision": number(span_data.get("pattern_ber_excluding_outliers")),
        "G_e2e_kbps": number(row.get("g_e2e_kbps_in_band")),
        "payload_data_kbps": number(row.get("payload_data_bps_in_band")) / 1000.0
        if number(row.get("payload_data_bps_in_band")) is not None
        else None,
        "in_band_unique_seq": seq_unique,
        "full_band_unique_seq": integer(row.get("full_band_unique_seq_candidates")),
        "seq_span_theoretical_packets": span,
        "source_bandwidth_metrics": str(
            run_root / "results/bandwidth_crops/bandwidth_crop_metrics.csv"
        ),
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = ["C_bw", "PSR", "PSR_seqspan", "BER", "BER_non_collision", "G_e2e_kbps", "payload_data_kbps"]
    output: list[dict[str, Any]] = []
    for bandwidth in sorted({row["bandwidth_mhz"] for row in rows if row["bandwidth_mhz"] is not None}):
        selected = [row for row in rows if row["bandwidth_mhz"] == bandwidth]
        result: dict[str, Any] = {
            "distance_label_m": selected[0]["distance_label_m"],
            "los_nlos": selected[0]["los_nlos"],
            "rep_count": len(selected),
            "bandwidth_mhz": bandwidth,
        }
        for field in fields:
            values = [row[field] for row in selected if row[field] is not None]
            result[f"{field}_mean"] = mean(values) if values else None
            result[f"{field}_std"] = pstdev(values) if len(values) > 1 else 0.0 if values else None
        output.append(result)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    per_run: list[dict[str, Any]] = []
    for run_root in args.run_root:
        metrics_path = run_root / "results/bandwidth_crops/bandwidth_crop_metrics.csv"
        if not metrics_path.is_file():
            raise SystemExit(f"missing crop metrics: {metrics_path}")
        per_run.extend(enrich_row(run_root, row) for row in read_csv(metrics_path) if row.get("bandwidth_mhz"))

    per_run_fields = [
        "run_id", "distance_label_m", "los_nlos", "repetition", "manifest_distance_m",
        "distance_provenance", "bandwidth_mhz", "C_bw", "PSR", "PSR_seqspan", "BER",
        "BER_non_collision", "G_e2e_kbps", "payload_data_kbps", "in_band_unique_seq",
        "full_band_unique_seq", "seq_span_theoretical_packets", "source_bandwidth_metrics",
    ]
    summary = aggregate(per_run)
    summary_fields = [
        "distance_label_m", "los_nlos", "rep_count", "bandwidth_mhz",
        "C_bw_mean", "C_bw_std", "PSR_mean", "PSR_std", "PSR_seqspan_mean", "PSR_seqspan_std",
        "BER_mean", "BER_std", "BER_non_collision_mean", "BER_non_collision_std",
        "G_e2e_kbps_mean", "G_e2e_kbps_std", "payload_data_kbps_mean", "payload_data_kbps_std",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "phonehrs_los_d3m_bandwidth_per_run.csv", per_run, per_run_fields)
    write_csv(args.output_dir / "phonehrs_los_d3m_bandwidth_summary.csv", summary, summary_fields)
    metadata = {
        "schema_version": 1,
        "experiment": "phonehrs_x31080m_los_d3m",
        "distance_label_m": 3.0,
        "distance_confirmation": "operator_confirmed_physical_distance_from_run_id_label",
        "manifest_note": "phone_manifest.json records stale --distance-m 0.5; actual setup was 3 m per operator clarification",
        "los_nlos": "LOS",
        "runs": [str(path) for path in args.run_root],
        "metric_definitions": {
            "C_bw": "in-band unique-seq candidates / full-band unique-seq candidates",
            "PSR": "pattern-exact packets / in-band unique-seq candidates (handoff psr_exact)",
            "PSR_seqspan": "in-band unique-seq candidates / scorer dominant consecutive sequence span",
            "BER": "pattern bit errors / compared bits",
            "G_e2e_kbps": "unique-seq PC-frame data rate in the IQ window",
        },
    }
    (args.output_dir / "phonehrs_los_d3m_bandwidth_summary.json").write_text(
        json.dumps({"metadata": metadata, "summary": summary}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"per_run_rows": len(per_run), "summary_rows": len(summary), "output_dir": str(args.output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
