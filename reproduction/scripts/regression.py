"""Regression checks against the anonymized references shipped with the release."""

from __future__ import annotations

import csv
import math
from pathlib import Path

TIMING_FLOAT_TOLERANCE_MS = 1e-3
FREQUENCY_FLOAT_TOLERANCE = 1e-6
RANGE_FLOAT_TOLERANCE = 1e-12


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def run(root: Path) -> dict[str, int]:
    expected = read(root / "checks/reference_timing_points.csv")
    actual = read(root / "outputs/summary_embedded_points.csv")
    actual_by_label = {row["label"]: row for row in actual}
    for row in expected:
        got = actual_by_label.get(row["label"])
        if got is None:
            raise AssertionError(f"missing timing point {row['label']}")
        for key in ("gap_median_ms", "adv_median_ms"):
            if not close(float(got[key]), float(row[key]), TIMING_FLOAT_TOLERANCE_MS):
                raise AssertionError(f"timing mismatch {row['label']} {key}: {got[key]} != {row[key]}")

    expected_pop = read(root / "checks/reference_timing_population.csv")
    actual_pop = [r for r in read(root / "outputs/summary_per_address_normal.csv") if str(r.get("include", "")).lower() in {"true", "1"}]
    expected_pop.sort(key=lambda r: (float(r["gap_median_ms"]), float(r["adv_median_ms"])))
    actual_pop.sort(key=lambda r: (float(r["gap_median_ms"]), float(r["adv_median_ms"])))
    if len(expected_pop) != len(actual_pop):
        raise AssertionError(f"benign population count mismatch: {len(actual_pop)} != {len(expected_pop)}")
    for got, want in zip(actual_pop, expected_pop):
        if not close(float(got["gap_median_ms"]), float(want["gap_median_ms"]), TIMING_FLOAT_TOLERANCE_MS) or not close(float(got["adv_median_ms"]), float(want["adv_median_ms"]), TIMING_FLOAT_TOLERANCE_MS):
            raise AssertionError("benign population timing mismatch")

    timing_counts = read(root / "outputs/timing_counts.csv")[0]
    expected_counts = {
        "normal_packet_records": 292666,
        "normal_groups": 19814,
        "included_benign_groups": 152,
        "embedded_points": 11,
    }
    for key, want in expected_counts.items():
        if int(timing_counts[key]) != want:
            raise AssertionError(f"timing count mismatch {key}: {timing_counts[key]} != {want}")

    expected_samples = read(root / "checks/reference_timing_sample_checks.csv")
    actual_samples = {row["table_file"]: row for row in read(root / "outputs/timing_sample_checks.csv")}
    for row in expected_samples:
        got = actual_samples.get(row["table_file"])
        if got is None:
            raise AssertionError(f"missing timing sample check {row['table_file']}")
        for key in ("packet_records", "groups", "event_count", "gap_samples", "event_samples", "gap_event_sha256"):
            if got[key] != row[key]:
                raise AssertionError(f"timing sample mismatch {row['table_file']} {key}: {got[key]} != {row[key]}")

    freq = {row["metric"]: float(row["value"]) for row in read(root / "outputs/frequency_similarity.csv")}
    for key, want in {"freq_adv": 0.997693121433, "freq_data": 0.988912940025, "PIP_NORMAL_cross_mean": 0.960272093614}.items():
        if not close(freq[key], want, FREQUENCY_FLOAT_TOLERANCE):
            raise AssertionError(f"frequency mismatch {key}: {freq[key]} != {want}")

    expected_range = read(root / "checks/reference_range_metrics.csv")
    actual_range = {row["run_id"]: row for row in read(root / "outputs/range_packet_metrics.csv")}
    for row in expected_range:
        got = actual_range.get(row["run_id"])
        if got is None:
            raise AssertionError(f"missing range metric {row['run_id']}")
        for key in ("candidate_rows", "unique_candidates", "exact_packets", "outlier_packets"):
            if int(got[key]) != int(row[key]):
                raise AssertionError(f"range mismatch {row['run_id']} {key}: {got[key]} != {row[key]}")
        for key in ("psr_exact", "pattern_ber", "pattern_ber_excluding_outliers", "seqspan_psr"):
            if not close(float(got[key]), float(row[key]), RANGE_FLOAT_TOLERANCE):
                raise AssertionError(f"range mismatch {row['run_id']} {key}: {got[key]} != {row[key]}")

    return {"timing_points": len(expected), "benign_population": len(expected_pop), "timing_sample_tables": len(expected_samples), "range_runs": len(expected_range), "frequency_metrics": len(freq), "timing_float_tolerance_ms": TIMING_FLOAT_TOLERANCE_MS, "frequency_float_tolerance": FREQUENCY_FLOAT_TOLERANCE, "range_float_tolerance": RANGE_FLOAT_TOLERANCE}
