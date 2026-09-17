#!/usr/bin/env python3
"""Build a read-only recovery-investigation ledger for one PhantomChannel run.

The tool creates a new analysis directory and never overwrites frozen run
artifacts.  It records the PhantomChannel receiver dependency state before and after
the analysis, then runs the investigation-only matching v2 and a bounded local
IQ duration probe around the parser/diagnostic sample positions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from portable_paths import BLE_ROOT  # noqa: E402

import match_rtt_sdr_v2 as matching  # noqa: E402
from phantom_postprocess_scorer import (  # noqa: E402
    ble_1m_airtime_us,
    estimate_burst_duration_us,
)
import match_rtt_sdr_results as legacy  # noqa: E402


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def git_snapshot(root: Path, tracked_files: list[Path]) -> dict[str, Any]:
    def run_git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip()

    status = run_git("status", "--porcelain=v1", "--untracked-files=all")
    hashes: dict[str, str] = {}
    for path in tracked_files:
        if path.is_file():
            hashes[str(path)] = sha256_file(path)
    return {
        "root": str(root),
        "head": run_git("rev-parse", "HEAD"),
        "status_porcelain": status.splitlines() if status else [],
        "diff_stat": run_git("diff", "--stat"),
        "tracked_file_sha256": hashes,
    }


def ble_guard_snapshot() -> dict[str, Any]:
    files = [
        BLE_ROOT / "experiment" / "bt_40m_pfb_realtime.py",
        BLE_ROOT / "experiment" / "bt_40m_pfb_pipeline.py",
        BLE_ROOT / "experiment" / "bt_pipeline" / "realtime_sources.py",
        BLE_ROOT / "experiment" / "bt_pipeline" / "wideband_channelizer.py",
        BLE_ROOT / "experiment" / "bt_pipeline" / "pfb_channelizer.py",
        BLE_ROOT / "experiment" / "bt_pipeline" / "native_backend.py",
        BLE_ROOT / "experiment" / "bt_pipeline" / "parsers.py",
        BLE_ROOT / "build-native" / "bt_native.cpython-312-x86_64-linux-gnu.so",
    ]
    return git_snapshot(BLE_ROOT, files)


def read_csv(path: Path) -> list[dict[str, str]]:
    return matching.read_csv(path)


def _float(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _nearest_candidate(
    tx: dict[str, Any],
    candidates: list[dict[str, Any]],
    gate_samples: float,
) -> dict[str, Any] | None:
    compatible = [
        candidate
        for candidate in candidates
        if str(candidate.get("channel", "")) == str(tx.get("channel", ""))
        and abs(candidate["sample"] - tx["predicted_sample"]) <= gate_samples
    ]
    if not compatible:
        return None
    return min(compatible, key=lambda candidate: abs(candidate["sample"] - tx["predicted_sample"]))


def add_local_iq_probes(
    attempts: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    iq_path: Path,
    sample_rate_sps: float,
    center_frequency_hz: float,
    gate_samples: float,
) -> dict[str, Any]:
    """Probe energy duration around one sample per in-window TX.

    This is deliberately called a probe, not an independent burst detector:
    if v2 has a candidate, that candidate's sample is used; otherwise the
    predicted affine sample is used.  The output is diagnostic evidence only.
    """

    config = {
        "expected_phantom_us": 5_000.0,
        "pre_margin_us": 20.0,
        "post_margin_us": 200.0,
        "lowpass_hz": 900_000.0,
        "smooth_us": 1.5,
        "threshold_sigma": 8.0,
        "min_threshold_ratio": 2.0,
    }
    measured = 0
    long_bursts = 0
    unavailable = 0
    candidates_by_index = {
        int(candidate["observation_index"]): candidate
        for candidate in candidates
        if candidate.get("observation_index") is not None
    }
    for tx in attempts:
        tx["local_iq_probe_status"] = "outside_window"
        tx["local_probe_sample"] = ""
        tx["physical_burst_duration_us"] = ""
        tx["physical_residual_tail_length_bytes"] = ""
        tx["long_burst_evidence"] = ""
        if not tx.get("in_iq_window"):
            continue
        candidate = None
        matched_index = _int(tx.get("observation_index"))
        if matched_index is not None:
            candidate = candidates_by_index.get(matched_index)
        if candidate is None:
            candidate = _nearest_candidate(tx, candidates, gate_samples)
        probe_sample = candidate["sample"] if candidate is not None else tx["predicted_sample"]
        if probe_sample is None:
            tx["local_iq_probe_status"] = "no_sample"
            unavailable += 1
            continue
        channel = _int(tx.get("channel"))
        if channel is None:
            tx["local_iq_probe_status"] = "invalid_channel"
            unavailable += 1
            continue
        # Pass the textual channel through the existing helper: its
        # int_or_empty compatibility path treats integer 0 as an empty value.
        packet_frequency_hz = legacy.ble_data_channel_frequency_hz(str(channel))
        if packet_frequency_hz is None:
            tx["local_iq_probe_status"] = "invalid_channel_frequency"
            unavailable += 1
            continue
        try:
            result = estimate_burst_duration_us(
                iq_path,
                sample_rate_hz=sample_rate_sps,
                center_frequency_hz=center_frequency_hz,
                packet_start_sample=int(round(probe_sample)),
                packet_frequency_hz=packet_frequency_hz,
                expected_phantom_us=config["expected_phantom_us"],
                pre_margin_us=config["pre_margin_us"],
                post_margin_us=config["post_margin_us"],
                lowpass_hz=config["lowpass_hz"],
                smooth_us=config["smooth_us"],
                threshold_sigma=config["threshold_sigma"],
                min_threshold_ratio=config["min_threshold_ratio"],
            )
        except Exception as exc:  # pragma: no cover - hardware/file-specific guard
            tx["local_iq_probe_status"] = f"error:{type(exc).__name__}"
            unavailable += 1
            continue
        measured_us = _float(result.get("measured_duration_us"))
        tx["local_probe_sample"] = int(round(probe_sample))
        tx["local_iq_probe_status"] = "measured" if measured_us is not None else str(result.get("notes", "unavailable"))
        tx["local_probe_notes"] = result.get("notes", "")
        tx["local_probe_confidence_db"] = result.get("duration_confidence", "")
        if measured_us is None:
            unavailable += 1
            continue
        measured += 1
        tx["physical_burst_duration_us"] = f"{measured_us:.3f}"
        standard_us = ble_1m_airtime_us(_int(tx.get("normal_pdu_len")) or 0)
        residual_us = measured_us - standard_us
        residual_bytes = max(0, int(round(residual_us * 1_000_000.0 / (8.0 * 1_000_000.0))))
        tx["physical_residual_tail_length_bytes"] = residual_bytes
        # 237 B post-CRC is 1,896 us at 1M.  The 2,000-us threshold is a
        # slightly conservative diagnostic threshold, not a frame acceptance rule.
        if measured_us >= 2_000.0:
            tx["long_burst_evidence"] = "long_physical_burst"
            long_bursts += 1
        else:
            tx["long_burst_evidence"] = "standard_or_short_burst"
    return {
        "status": "bounded_iq_duration_probe",
        "iq_path": str(iq_path),
        "probes_requested": sum(int(tx.get("in_iq_window")) for tx in attempts),
        "probes_measured": measured,
        "probes_unavailable": unavailable,
        "long_burst_count_threshold_2000us": long_bursts,
        "threshold_us": 2_000.0,
        "parameters": config,
        "independent_burst_detector": False,
    }


def snapshot_input_files(run_root: Path) -> dict[str, Any]:
    paths = {
        "iq": run_root / "iq" / "capture.sc16",
        "iq_metadata": run_root / "iq" / "metadata.json",
        "rtt_log": run_root / "ground_truth" / "peripheral_rtt.log",
        "rtt_ground_truth": run_root / "ground_truth" / "rtt_ground_truth.csv",
        "rtt_ll_tx": run_root / "ground_truth" / "rtt_ll_tx.csv",
        "sdr_ble_packets": run_root / "sdr" / "ble_packets.csv",
        "old_recovery_metrics": run_root / "results" / "recovery_metrics.json",
        "old_matches": run_root / "results" / "rtt_sdr_matches.csv",
        "parser_command": run_root / "sdr" / "command.txt",
    }
    result: dict[str, Any] = {}
    for name, path in paths.items():
        item: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            stat = path.stat()
            item.update({"size_bytes": stat.st_size, "sha256": sha256_file(path)})
        result[name] = item
    return result


def build_manifest(
    run_root: Path,
    output_dir: Path,
    analysis_id: str,
    before_guard: dict[str, Any],
    after_guard: dict[str, Any],
    input_files: dict[str, Any],
    analysis: dict[str, Any],
    probe: dict[str, Any],
) -> dict[str, Any]:
    metadata = json.loads((run_root / "iq" / "metadata.json").read_text(encoding="utf-8"))
    command_path = run_root / "sdr" / "command.json"
    parser_command = json.loads(command_path.read_text(encoding="utf-8")) if command_path.is_file() else []
    return {
        "schema_version": 1,
        "analysis_id": analysis_id,
        "created_epoch_ns": time.time_ns(),
        "run_root": str(run_root),
        "output_dir": str(output_dir),
        "frozen_reference": "20260803_covert230_interval20ms_compactlog_lab_los_d0.5m_bw80mhz_rep2",
        "input_files": input_files,
        "capture_metadata_summary": {
            "actual_sample_rate_sps": metadata.get("actual_sample_rate_sps"),
            "actual_rx_bandwidth_hz": metadata.get("actual_rx_bandwidth_hz"),
            "samples": metadata.get("samples"),
            "overflows": metadata.get("overflows"),
            "gaps": metadata.get("gaps"),
            "udp_drops": metadata.get("udp_drops"),
            "valid_no_overflow_or_gap": metadata.get("valid_no_overflow_or_gap"),
        },
        "frozen_parser_command": parser_command,
        "analysis_parameters": {
            "matcher": "match_rtt_sdr_v2",
            "diagnostic_alignment_source": analysis["summary"]["alignment"]["source"],
            "time_gate_us": analysis["summary"]["alignment"]["time_gate_us"],
            "candidate_dedupe_samples": analysis["summary"]["candidate_dedupe_samples"],
            "blind_recovery_claim_allowed": False,
        },
        "local_iq_probe": probe,
        "receiver_guard": {
            "before": before_guard,
            "after": after_guard,
            "unchanged": before_guard == after_guard,
            "external_project_touched_by_this_tool": False,
        },
        "environment": {
            "PYTHONDONTWRITEBYTECODE": "1",
            "parser_external_project_is_read_only": True,
            "command_note": shlex.join(parser_command) if parser_command else "",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--analysis-id", default="matching_v2_ledger")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--time-gate-us", type=float, default=0.0)
    parser.add_argument("--dedupe-samples", type=int, default=2_000)
    parser.add_argument("--skip-local-iq-probe", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_root = args.run_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_root / "results" / "recovery_investigation" / args.analysis_id
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {output_dir}")
    required = [
        run_root / "iq" / "capture.sc16",
        run_root / "iq" / "metadata.json",
        run_root / "ground_truth" / "rtt_ground_truth.csv",
        run_root / "ground_truth" / "rtt_ll_tx.csv",
        run_root / "sdr" / "ble_packets.csv",
        run_root / "results" / "rtt_sdr_matches.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing required run artifacts:\n" + "\n".join(missing))

    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    before_guard = ble_guard_snapshot()
    input_files = snapshot_input_files(run_root)
    analysis = matching.analyze_run(
        run_root,
        time_gate_us=args.time_gate_us,
        dedupe_samples=args.dedupe_samples,
    )
    metadata = json.loads((run_root / "iq" / "metadata.json").read_text(encoding="utf-8"))
    if args.skip_local_iq_probe:
        probe = {
            "status": "skipped",
            "independent_burst_detector": False,
            "reason": "--skip-local-iq-probe",
        }
    else:
        probe = add_local_iq_probes(
            analysis["attempts"],
            analysis["candidate_window"],
            iq_path=run_root / "iq" / "capture.sc16",
            sample_rate_sps=float(metadata.get("actual_sample_rate_sps") or metadata.get("sample_rate_sps")),
            center_frequency_hz=float(metadata.get("actual_center_frequency_hz") or metadata.get("center_frequency_hz")),
            gate_samples=float(analysis["summary"]["alignment"]["time_gate_samples"]),
        )
    # Add probe columns after matching; write_analysis's fixed fields are
    # intentionally extended here for the investigation-only funnel.
    matching.ATTEMPT_FIELDS.extend(
        [
            "local_iq_probe_status", "local_probe_sample", "physical_burst_duration_us",
            "physical_residual_tail_length_bytes", "long_burst_evidence", "local_probe_notes",
            "local_probe_confidence_db",
        ]
    )
    before_after_summary = dict(analysis["summary"])
    before_after_summary["local_iq_probe"] = probe
    analysis["summary"] = before_after_summary
    matching.write_analysis(output_dir, analysis)
    after_guard = ble_guard_snapshot()
    manifest = build_manifest(
        run_root,
        output_dir,
        args.analysis_id,
        before_guard,
        after_guard,
        input_files,
        analysis,
        probe,
    )
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "receiver_guard.json", manifest["receiver_guard"])
    write_json(output_dir / "validation_report.json", {
        "analysis_id": args.analysis_id,
        "diagnostic_only": True,
        "funnel_closes_to_all_ll_attempts": len(analysis["attempts"]) == sum(analysis["summary"]["failure_stage_counts_all"].values()),
        "funnel_closes_to_iq_window": analysis["summary"]["tx_attempts_in_iq_window"] == sum(analysis["summary"]["failure_stage_counts_in_iq_window"].values()),
        "receiver_unchanged": before_guard == after_guard,
        "blind_recovery_claim_allowed": False,
        "notes": [
            "The local IQ duration probe uses parser/diagnostic sample positions and is not an independent burst detector.",
            "No frozen artifact was overwritten.",
        ],
    })
    print(json.dumps({**analysis["summary"], "output_dir": str(output_dir)}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
