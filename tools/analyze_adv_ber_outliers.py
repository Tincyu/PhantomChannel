#!/usr/bin/env python3
"""Per-packet BER distribution for advertising (ch39) runs.

The global BER is dominated by a small tail of collision-corrupted packets
(per-packet BER > 5-10%).  This tool reads each run's
``results/parser_candidate_packets.csv`` and ``run_summary.json``, deduplicates
by frame sequence (keeping the first candidate, matching the scorer), and
prints:

* per-packet BER histogram buckets;
* how many packets exceed 1/5/10/20/50% per-packet BER;
* the share of global bit errors carried by the >10% tail;
* the global BER after excluding packets above each threshold.

Usage:
    python3 tools/analyze_adv_ber_outliers.py <run_dir> [<run_dir> ...]
    python3 tools/analyze_adv_ber_outliers.py --root /path/to/PhantomChannel/testdata
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


BUCKETS = [
    (0.0, 0.0),
    (0.0001, 0.01),
    (0.01, 0.05),
    (0.05, 0.10),
    (0.10, 0.25),
    (0.25, 0.50),
    (0.50, 1.0),
]
THRESHOLDS = (0.01, 0.05, 0.10, 0.20, 0.50)


def interval_of(run_name: str, run_dir: Path) -> int | None:
    # The run name carries the actual on-air interval for the sweep runs; the
    # adv_manifest.json of those runs records the script default (20 ms), so
    # the explicit suffix wins.  Manifest is the fallback for other runs.
    if "_50ms" in run_name:
        return 50
    if "_100ms" in run_name:
        return 100
    if "_200ms" in run_name:
        return 200
    if "_1000ms" in run_name:
        return 1000
    if "delay0" in run_name:
        return 20
    manifest = run_dir / "adv_manifest.json"
    if manifest.is_file():
        try:
            value = json.loads(manifest.read_text(encoding="utf-8")).get("interval_ms")
            if value is not None:
                return int(value)
        except (OSError, ValueError, TypeError):
            pass
    return None


def run_candidates(csv_path: Path) -> list[dict[str, Any]]:
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8", errors="replace")))
    by_seq: dict[str, dict[str, Any]] = {}
    for row in rows:
        seq = str(row.get("frame_seq", "")).strip()
        if seq and seq not in by_seq:
            by_seq[seq] = row
    return list(by_seq.values())


def analyze_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    summary_path = run_dir / "results/run_summary.json"
    csv_path = run_dir / "results/parser_candidate_packets.csv"
    if not summary_path.is_file() or not csv_path.is_file():
        return {"run": str(run_dir), "error": "missing results/run_summary.json or parser_candidate_packets.csv"}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    candidates = run_candidates(csv_path)

    per: list[tuple[float, int, int, str, str]] = []
    for row in candidates:
        errors = int(row.get("pattern_bit_errors") or 0)
        bits = int(row.get("pattern_bits_compared") or 0)
        ber = errors / bits if bits else 0.0
        per.append((ber, errors, bits, str(row.get("frame_seq", "")), str(row.get("sample_index", ""))))

    total_errors = sum(item[1] for item in per)
    total_bits = sum(item[2] for item in per)
    histogram = [0] * len(BUCKETS)
    for ber, _, _, _, _ in per:
        for index, (low, high) in enumerate(BUCKETS):
            if low <= ber <= high:
                histogram[index] += 1
                break

    counts = {threshold: sum(1 for ber, _, _, _, _ in per if ber > threshold) for threshold in THRESHOLDS}
    error_shares = {
        threshold: sum(errors for ber, errors, _, _, _ in per if ber > threshold)
        for threshold in THRESHOLDS
    }
    excluded_ber = {}
    for threshold in THRESHOLDS:
        errors = sum(e for ber, e, _, _, _ in per if not ber > threshold)
        bits = sum(b for ber, _, b, _, _ in per if not ber > threshold)
        excluded_ber[threshold] = errors / bits if bits else None

    top = sorted(per, key=lambda item: item[0], reverse=True)[:5]
    return {
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "interval_ms": interval_of(run_dir.name, run_dir),
        "events": summary.get("marker_valid_candidates"),
        "unique": summary.get("seq_unique_count"),
        "global_ber": summary.get("pattern_ber"),
        "n_packets": len(per),
        "n_exact": sum(1 for ber, _, _, _, _ in per if ber == 0.0),
        "histogram": {f"{low * 100:.2f}-{high * 100:.2f}%": count for (low, high), count in zip(BUCKETS, histogram)},
        "counts_gt": counts,
        "error_share_gt": {f"{t:.0%}": share / total_errors if total_errors else None for t, share in error_shares.items()},
        "ber_excluded": excluded_ber,
        "top5": [{"ber": ber, "errors": errors, "bits": bits, "seq": seq, "sample_index": sample}
                 for ber, errors, bits, seq, sample in top],
        "total_errors": total_errors,
        "total_bits": total_bits,
    }


def print_table(results: list[dict[str, Any]]) -> None:
    header = (
        f"{'run':<44}{'int':>5}{'ev':>4}{'uniq':>5}{'gBER':>8}{'>1%':>5}{'>5%':>5}"
        f"{'>10%':>5}{'>20%':>5}{'top10err':>9}{'excl10%':>9}"
    )
    print(header)
    for result in results:
        if "error" in result:
            print(f"{result['run']:<44}  ERROR: {result['error']}")
            continue
        ber = result["global_ber"]
        share = result["error_share_gt"].get("10%")
        excluded = result["ber_excluded"].get(0.10)
        interval = result["interval_ms"] if result["interval_ms"] is not None else "?"
        print(
            f"{result['run'][-42:]:<44}{interval!s:>5}{result['events']!s:>4}{result['unique']!s:>5}"
            f"{(ber or 0) * 100:>7.2f}%"
            f"{result['counts_gt'][0.01]:>5}{result['counts_gt'][0.05]:>5}{result['counts_gt'][0.10]:>5}"
            f"{result['counts_gt'][0.20]:>5}"
            f"{(share or 0) * 100:>8.1f}%{(excluded or 0) * 100:>8.3f}%"
        )


def print_detail(result: dict[str, Any]) -> None:
    if "error" in result:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    print(f"\n=== {result['run']} (interval {result['interval_ms']} ms) ===")
    print(f"events={result['events']} unique={result['unique']} N={result['n_packets']} "
          f"global BER={result['global_ber'] * 100:.4f}%")
    print("per-packet BER histogram:", "  ".join(f"{k}={v}" for k, v in result["histogram"].items()))
    print("packets > threshold:", "  ".join(f">{t * 100:.0f}%={v}" for t, v in result["counts_gt"].items()))
    print("error share >10%:", f"{result['error_share_gt']['10%']:.1%}")
    print("BER excluding >10%:", f"{result['ber_excluded'][0.10]:.4%}")
    print("top5: BER  seq  sample_index")
    for item in result["top5"]:
        print(f"  {item['ber']:6.2%}  {item['seq']:>6}  {item['sample_index']:>10}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path, help="run directories (results/... must exist)")
    parser.add_argument("--root", type=Path, default=Path("/path/to/PhantomChannel/testdata"),
                        help="scan this root for 20260807_adv_ch39* runs when no explicit runs are given")
    parser.add_argument("--filter", default="20260807_adv_ch39_2480_len239_rep5_test*", help="glob filter under --root")
    parser.add_argument("--detail", type=Path, help="print the per-packet detail block for one run")
    args = parser.parse_args(argv)

    runs = list(args.runs)
    if not runs:
        runs = sorted(args.root.expanduser().resolve().glob(args.filter))
        extra = [Path("/path/to/PhantomChannel/testdata/20260807_adv_ch39_2480_b210_delay0_len239_rep1")]
        runs = runs + [run for run in extra if run.is_dir()]
    if not runs:
        parser.error("no runs matched")

    results = [analyze_run(run) for run in runs]
    results.sort(key=lambda result: ((result.get("interval_ms") or 99999), result.get("run", "")))
    print_table(results)
    if args.detail is not None:
        detail_path = args.detail.expanduser().resolve()
        matched = next((result for result in results if Path(result.get("run_dir", "")).resolve() == detail_path), None)
        if matched is None:
            matched = analyze_run(detail_path)
        print_detail(matched)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
