#!/usr/bin/env python3
"""Recompute the available paper-table metrics from release inputs.

The inputs are derived parser metrics and anonymized timing vectors taken from
the existing experiment archives.  This script intentionally does not read
paper/main.tex and does not recreate the old paper-table snapshot files.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Iterable


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot aggregate an empty vector")
    return statistics.mean(values), statistics.pstdev(values) if len(values) > 1 else 0.0


def read_table3(root: Path) -> list[dict[str, str]]:
    path = root / "data/recomputed_tables/table3/bandwidth_per_run.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 18:
        raise ValueError(f"expected 18 Table 3 run-bandwidth rows, got {len(rows)}")
    return rows


def recompute_table3(root: Path) -> dict[str, object]:
    source = read_table3(root)
    output: list[dict[str, object]] = []
    for bandwidth in sorted({int(row["bandwidth_mhz"]) for row in source}):
        selected = sorted(
            (row for row in source if int(row["bandwidth_mhz"]) == bandwidth),
            key=lambda row: int(row["repetition"]),
        )
        c_mean, c_std = mean_std([float(row["c_bw"]) for row in selected])
        g_mean, g_std = mean_std([float(row["g_e2e_kbps"]) for row in selected])
        output.append(
            {
                "Bandwidth_MHz": bandwidth,
                "rep_count": len(selected),
                "Data_channels_n": "/".join(row["active_channels"] for row in selected),
                "C_obs_percent_mean": c_mean * 100.0,
                "C_obs_percent_std": c_std * 100.0,
                "G_e2e_kbps_mean": g_mean,
                "G_e2e_kbps_std": g_std,
            }
        )
    path = root / "outputs/recomputed_tables/table3_bandwidth.csv"
    write_csv(
        path,
        [
            "Bandwidth_MHz",
            "rep_count",
            "Data_channels_n",
            "C_obs_percent_mean",
            "C_obs_percent_std",
            "G_e2e_kbps_mean",
            "G_e2e_kbps_std",
        ],
        output,
    )
    return {"source_rows": len(source), "output_rows": len(output), "path": str(path.relative_to(root))}


def read_samples(path: Path) -> dict[tuple[str, str], list[float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    samples: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        key = (row["dataset"], row["metric"])
        value = float(row["value"])
        if not math.isfinite(value):
            raise ValueError(f"non-finite timing sample in {path}: {row}")
        samples.setdefault(key, []).append(value)
    return samples


def recompute_table5(root: Path) -> dict[str, object]:
    central = read_samples(root / "data/recomputed_tables/table5/central_samples_longform.csv")
    peripheral = read_samples(root / "data/recomputed_tables/table5/peripheral_samples_longform.csv")
    specs = [
        ("Central-side embedding", "normal", "normal_1", "gap_us", central),
        ("Central-side embedding", "embed_8B", "embedd_8B", "gap_us", central),
        ("Peripheral-side embedding", "normal", "normal", "empty_to_att_gap_us", peripheral),
        ("Peripheral-side embedding", "embed_8B", "embed_16B", "empty_to_att_gap_us", peripheral),
        ("Peripheral-side embedding", "embed_240B", "embed_240B", "empty_to_att_gap_us", peripheral),
    ]
    rows: list[dict[str, object]] = []
    for side, condition, dataset, gap_metric, source in specs:
        gaps = source[(dataset, gap_metric)]
        intervals = source[(dataset, "event_interval_ms")]
        rows.append(
            {
                "Side": side,
                "Condition": condition,
                "Input_dataset": dataset,
                "Gap_n": len(gaps),
                "Observable_gap_mean_us": statistics.mean(gaps),
                "Observable_gap_median_us": statistics.median(gaps),
                "Event_interval_n": len(intervals),
                "Event_interval_mean_ms": statistics.mean(intervals),
                "Event_interval_median_ms": statistics.median(intervals),
            }
        )
    path = root / "outputs/recomputed_tables/table5_conn_timing.csv"
    write_csv(
        path,
        [
            "Side",
            "Condition",
            "Input_dataset",
            "Gap_n",
            "Observable_gap_mean_us",
            "Observable_gap_median_us",
            "Event_interval_n",
            "Event_interval_mean_ms",
            "Event_interval_median_ms",
        ],
        rows,
    )
    return {"central_sample_rows": sum(map(len, central.values())), "peripheral_sample_rows": sum(map(len, peripheral.values())), "output_rows": len(rows), "path": str(path.relative_to(root))}


def recompute_table4(root: Path) -> dict[str, object]:
    source_path = root / "data/recomputed_tables/table4/ablation_results.csv"
    with source_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 4:
        raise ValueError(f"expected 4 Table 4 rows, got {len(rows)}")
    output = [
        {
            "Method": row["method"],
            "Source_group": row["source_group"],
            "Total_count": int(row["total_count"]),
            "Success_count": int(row["success_count"]),
            "PSR_percent": float(row["psr_percent"]),
        }
        for row in rows
    ]
    path = root / "outputs/recomputed_tables/table4_ablation.csv"
    write_csv(path, ["Method", "Source_group", "Total_count", "Success_count", "PSR_percent"], output)
    return {"source_rows": len(rows), "output_rows": len(output), "path": str(path.relative_to(root))}


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    result = {
        "table3": recompute_table3(root),
        "table5": recompute_table5(root),
        "table4": recompute_table4(root),
        "paper_source_read": False,
    }
    provenance = root / "outputs/recomputed_tables/recomputation_provenance.json"
    provenance.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
