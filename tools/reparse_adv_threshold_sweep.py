#!/usr/bin/env python3
"""Replay one advertising IQ capture with isolated BLE burst thresholds.

The capture and external parser sources remain read-only.  Every threshold is
parsed and scored in a separate directory so that a single threshold can be
frozen on prescan data before the formal antenna comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLE_ROOT = Path("/path/to/PhantomChannel/receiver")
SCORER = PROJECT_ROOT / "tools" / "score_iq_parser_candidates.py"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def replace_option(command: list[str], option: str, value: str) -> list[str]:
    result = list(command)
    indices = [index for index, token in enumerate(result) if token == option]
    if indices:
        first = indices[0]
        result[first + 1] = value
        for index in reversed(indices[1:]):
            del result[index:index + 2]
    else:
        result.extend([option, value])
    return result


def parser_environment() -> dict[str, str]:
    env = os.environ.copy()
    paths = [str(BLE_ROOT / "experiment"), str(BLE_ROOT / "ble_fun_test"), str(BLE_ROOT / "build-native")]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(paths)
    return env


def threshold_name(value: float) -> str:
    return f"thr_{value:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def load_base_command(run_root: Path) -> list[str]:
    command_path = run_root / "diagnostics" / "one_stage_cpp" / "command.txt"
    if not command_path.is_file():
        raise FileNotFoundError(command_path)
    return shlex.split(command_path.read_text(encoding="utf-8").strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--thresholds", default="0.02,0.015,0.01,0.0075,0.005")
    parser.add_argument("--pattern-outlier-ber-threshold", type=float, default=0.10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    run_root = args.run_root.expanduser().resolve()
    output_root = (args.output_dir or (run_root / "results" / "antenna_threshold_sweep")).expanduser().resolve()
    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    if not thresholds or any(value <= 0 for value in thresholds):
        raise SystemExit("--thresholds must contain positive comma-separated values")
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    base = load_base_command(run_root)
    metadata = run_root / "iq" / "metadata.json"
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        variant_root = output_root / threshold_name(threshold)
        parser_root = variant_root / "parser"
        score_root = variant_root / "score"
        parser_root.mkdir(parents=True)
        command = replace_option(base, "--output-dir", str(parser_root))
        command = replace_option(command, "--ble-threshold", str(threshold))
        (variant_root / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
        returncode: int | None = None
        if not args.dry_run:
            with (variant_root / "parse.log").open("w", encoding="utf-8") as log:
                log.write("$ " + shlex.join(command) + "\n")
                log.flush()
                result = subprocess.run(command, cwd=BLE_ROOT, env=parser_environment(), stdout=log, stderr=subprocess.STDOUT, text=True)
            returncode = result.returncode
            if returncode == 0:
                score_command = [
                    sys.executable, str(SCORER), "--metadata", str(metadata),
                    "--parser-csv", str(parser_root / "ble_packets.csv"),
                    "--output-dir", str(score_root), "--pattern",
                    "--pattern-outlier-ber-threshold", str(args.pattern_outlier_ber_threshold),
                ]
                with (variant_root / "score.log").open("w", encoding="utf-8") as log:
                    score = subprocess.run(score_command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
                returncode = score.returncode

        summary_path = score_root / "parser_candidate_rate.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        rows.append({
            "threshold": threshold,
            "status": "planned" if args.dry_run else ("completed" if returncode == 0 else "failed"),
            "seq_unique_count": summary.get("seq_unique_count"),
            "seq_span_theoretical_packets": summary.get("seq_span_theoretical_packets"),
            "recovered_within_span_fraction": summary.get("recovered_within_span_fraction"),
            "g_e2e_kbps": summary.get("iq_window_parser_candidate_data_kbps"),
            "ber": summary.get("pattern_ber_excluding_outliers"),
            "candidate_csv": str(score_root / "parser_candidate_packets.csv"),
        })

    with (output_root / "threshold_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output_root / "threshold_sweep.json", {
        "schema_version": 1,
        "run_root": str(run_root),
        "thresholds": thresholds,
        "results": rows,
        "selection_rule": "Freeze one threshold on all prescan captures; reuse it unchanged for every formal antenna condition.",
    })
    print(json.dumps(rows, indent=2))
    return 0 if all(row["status"] in ("completed", "planned") for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
