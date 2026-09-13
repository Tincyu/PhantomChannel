#!/usr/bin/env python3
"""Run isolated parser parameter variants for a frozen PhantomChannel IQ run.

This tool invokes the existing BLE_encrypt_check parser as an external,
read-only process.  Every variant gets a separate output directory and Python
cache location under PhantomChannel.  It never edits the external project and
fails if the guarded source/native hashes change.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BLE_ROOT = Path("/path/to/BLE_encrypt_check")
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import match_rtt_sdr_results as legacy  # noqa: E402
import match_rtt_sdr_v2 as matching  # noqa: E402
from build_recovery_ledger import ble_guard_snapshot  # noqa: E402


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_command(run_root: Path) -> list[str]:
    path = run_root / "sdr" / "command.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    command = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise ValueError(f"parser command must be a JSON string list: {path}")
    return list(command)


def replace_option(command: list[str], option: str, value: str) -> list[str]:
    result = list(command)
    try:
        index = result.index(option)
    except ValueError as exc:
        raise ValueError(f"parser command is missing {option}") from exc
    if index + 1 >= len(result):
        raise ValueError(f"parser command has no value after {option}")
    result[index + 1] = value
    return result


def parse_variant(text: str) -> dict[str, Any]:
    """Parse name:threshold:score[:lpf] into a variant specification."""

    pieces = text.split(":")
    if len(pieces) not in (3, 4):
        raise ValueError("variant must be name:ble_threshold:ble_score_threshold[:ble_lpf_cutoff]")
    name = pieces[0].strip()
    if not name or "/" in name or ".." in name:
        raise ValueError(f"invalid variant name: {name!r}")
    item: dict[str, Any] = {
        "name": name,
        "ble_threshold": float(pieces[1]),
        "ble_score_threshold": float(pieces[2]),
    }
    if len(pieces) == 4:
        item["ble_lpf_cutoff"] = float(pieces[3])
    return item


def run_variant(
    *,
    run_root: Path,
    sweep_root: Path,
    base_command: list[str],
    variant: dict[str, Any],
    timeout_s: float,
    dry_run: bool,
) -> dict[str, Any]:
    variant_root = sweep_root / str(variant["name"])
    parser_output = variant_root / "sdr"
    parser_output.mkdir(parents=True, exist_ok=True)
    command = replace_option(base_command, "--output-dir", str(parser_output))
    command = replace_option(command, "--ble-threshold", str(variant["ble_threshold"]))
    command = replace_option(command, "--ble-score-threshold", str(variant["ble_score_threshold"]))
    if "ble_lpf_cutoff" in variant:
        command = replace_option(command, "--ble-lpf-cutoff", str(variant["ble_lpf_cutoff"]))
    command_json = variant_root / "command.json"
    write_json(command_json, command)
    write_json(variant_root / "variant.json", variant)

    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPYCACHEPREFIX"] = str(variant_root / "pycache")
    env["XDG_CACHE_HOME"] = str(variant_root / "xdg_cache")
    pythonpath_parts = [
        str(BLE_ROOT / "experiment"),
        str(BLE_ROOT / "ble_fun_test"),
        str(BLE_ROOT / "build-native"),
    ]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath_parts)
    log_path = variant_root / "parser.log"
    started = time.monotonic()
    result: subprocess.CompletedProcess[str] | None = None
    if not dry_run:
        with log_path.open("w", encoding="utf-8") as log:
            log.write("$ " + shlex.join(command) + "\n")
            log.flush()
            result = subprocess.run(
                command,
                cwd=str(BLE_ROOT),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_s,
                check=False,
            )
    elapsed_s = time.monotonic() - started
    item: dict[str, Any] = {
        **variant,
        "command": command,
        "command_shell": shlex.join(command),
        "output_dir": str(parser_output),
        "log": str(log_path),
        "dry_run": dry_run,
        "elapsed_s": elapsed_s,
        "returncode": None if result is None else result.returncode,
    }
    if result is None or result.returncode != 0:
        item["status"] = "planned" if dry_run else "failed"
        return item

    packets = parser_output / "ble_packets.csv"
    if not packets.is_file():
        item["status"] = "failed_no_ble_packets_csv"
        return item
    rtt_ground_truth = legacy.read_csv(run_root / "ground_truth" / "rtt_ground_truth.csv")
    ll_rows = legacy.read_csv(run_root / "ground_truth" / "rtt_ll_tx.csv")
    sdr_rows = legacy.read_csv(packets)
    metadata = json.loads((run_root / "iq" / "metadata.json").read_text(encoding="utf-8"))
    legacy_matches, _unmatched_rtt, _unmatched_sdr, legacy_summary = legacy.score_phantom_run(
        rtt_ground_truth,
        ll_rows,
        sdr_rows,
        metadata,
    )
    v2 = matching.analyze_run(
        run_root,
        sdr_csv=packets,
        old_matches_csv=run_root / "results" / "rtt_sdr_matches.csv",
    )
    matching.write_analysis(variant_root / "matching_v2", v2)
    item["status"] = "completed"
    item["parser_row_count"] = len(sdr_rows)
    item["legacy_exact_attempts"] = int(
        legacy_summary.get("covert_exact_packets_in_iq_capture_window", legacy_summary.get("covert_exact_packets", 0))
    )
    item["legacy_match_summary"] = legacy_summary
    item["matching_v2_summary"] = v2["summary"]
    return item


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--variant",
        action="append",
        default=None,
        help="name:ble_threshold:ble_score_threshold[:ble_lpf_cutoff]; repeatable",
    )
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_root = args.run_root.expanduser().resolve()
    sweep_root = args.output_dir.expanduser().resolve()
    if sweep_root.exists() and any(sweep_root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {sweep_root}")
    sweep_root.mkdir(parents=True, exist_ok=True)
    base_command = load_command(run_root)
    variant_texts = args.variant or ["baseline:0.01:3.0", "loose_threshold:0.0075:3.0"]
    variants = [parse_variant(item) for item in variant_texts]
    before = ble_guard_snapshot()
    results: list[dict[str, Any]] = []
    for variant in variants:
        results.append(
            run_variant(
                run_root=run_root,
                sweep_root=sweep_root,
                base_command=base_command,
                variant=variant,
                timeout_s=args.timeout_s,
                dry_run=args.dry_run,
            )
        )
        current = ble_guard_snapshot()
        if current != before:
            raise RuntimeError("BLE_encrypt_check guard changed during parser sweep")
    after = ble_guard_snapshot()
    summary = {
        "schema_version": 1,
        "run_root": str(run_root),
        "output_dir": str(sweep_root),
        "variants": results,
        "ble_encrypt_check_guard": {
            "before": before,
            "after": after,
            "unchanged": before == after,
            "external_project_touched_by_this_tool": False,
        },
        "notes": [
            "Parser outputs and Python/cache locations are isolated under this sweep directory.",
            "matching_v2 outputs are diagnostic and remain separate from frozen recovery_metrics.json.",
            "No parser source, configuration, build artifact, or default command in BLE_encrypt_check is modified.",
        ],
    }
    write_json(sweep_root / "sweep_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if all(item["status"] in ("completed", "planned") for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
