#!/usr/bin/env python3
"""Run an isolated X310 80 MHz capture and optional BLE parser replay.

This runner is intentionally separate from run_range_bandwidth_experiment.py.
It uses UHD's rx_samples_to_file example so that the X310 analog bandwidth is
set explicitly, records the actual rate selected by UHD, preserves the UHD
log, and can replay the resulting SC16 file through the bundled
PhantomChannel receiver-compatible parser.

The default test captures 10 seconds at 2440 MHz with a requested 80 MS/s
sample rate and 80 MHz bandwidth on RX2/channel 0. The X310 currently selects
100 MS/s for the requested 80 MS/s rate, so the actual value from the UHD log
is written to metadata.json and passed to the parser.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
from portable_paths import BLE_ROOT, PARSER_PYTHON, UHD_LIBRARY  # noqa: E402

DEFAULT_CAPTURE_BIN = UHD_LIBRARY / "uhd" / "examples" / "rx_samples_to_file"
DEFAULT_UHD_LIBRARY = UHD_LIBRARY
DEFAULT_BLE_ROOT = BLE_ROOT
# The X310 orchestrator and local CUDA physical-length backend use the
# PhantomChannel environment.  The parser defaults to the bundled read-only
# source tree and can be overridden with PHANTOM_RECEIVER_ROOT. The legacy
# PHANTOM_BLE_ROOT name remains accepted for compatibility.
DEFAULT_PARSER_PYTHON = PARSER_PYTHON
DEFAULT_PARSER_ENTRYPOINT = DEFAULT_BLE_ROOT / "experiment/bt_40m_pfb_realtime.py"
DEFAULT_TWO_STAGE_SCRIPT = PROJECT_ROOT / "tools/two_stage_known_aa_parse.py"
DEFAULT_NVME_ROOT = PROJECT_ROOT / "testdata"
DEFAULT_PSSD_ROOT = Path("/path/to/PhantomChannel/testdata")

ACTUAL_RATE_RE = re.compile(r"Actual RX Rate:\s*([0-9.+-eE]+)\s*Msps", re.IGNORECASE)
ACTUAL_FREQ_RE = re.compile(r"Actual RX Freq:\s*([0-9.+-eE]+)\s*MHz", re.IGNORECASE)
ACTUAL_GAIN_RE = re.compile(r"Actual RX Gain:\s*([0-9.+-eE]+)\s*dB", re.IGNORECASE)
ACTUAL_BW_RE = re.compile(r"Actual RX Bandwidth:\s*([0-9.+-eE]+)\s*MHz", re.IGNORECASE)
RECEIVED_SAMPLES_RE = re.compile(r"Received\s+(\d+)\s+samples\s+in\s+([0-9.+-eE]+)\s+seconds", re.IGNORECASE)
WRITE_SPEED_RE = re.compile(r"write test returned write speed of\s+([0-9.+-eE]+)\s*MB/s", re.IGNORECASE)


@dataclass(frozen=True)
class Target:
    name: str
    root: Path


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def utcish_run_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def slug(value: Any) -> str:
    text = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value).strip())
    return text.strip("_") or "na"


def parse_log_float(pattern: re.Pattern[str], text: str, multiplier: float = 1.0) -> float | None:
    match = pattern.search(text)
    return float(match.group(1)) * multiplier if match else None


def build_capture_command(args: argparse.Namespace, iq_path: Path) -> list[str]:
    return [
        str(args.capture_bin),
        "--args",
        args.usrp_args,
        "--file",
        str(iq_path),
        "--type",
        "short",
        "--wirefmt",
        "sc16",
        "--rate",
        str(args.sample_rate_sps),
        "--bw",
        str(args.bandwidth_hz),
        "--freq",
        str(args.center_frequency_hz),
        "--gain",
        str(args.gain_db),
        "--ant",
        args.antenna,
        "--channels",
        str(args.channel),
        "--duration",
        str(args.duration_s),
        "--spb",
        str(args.samples_per_buffer),
        "--setup",
        str(args.setup_s),
        "--progress",
        "--stats",
        "--continue",
    ]


def capture_environment(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    library = str(args.uhd_library)
    env["LD_LIBRARY_PATH"] = f"{library}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    return env


def ensure_capture_space(args: argparse.Namespace, target_root: Path) -> None:
    expected_bytes = int(args.expected_sample_rate_sps * args.duration_s * 4)
    reserve_bytes = int(args.reserve_bytes)
    usage_root = target_root if target_root.exists() else target_root.parent
    free_bytes = shutil.disk_usage(usage_root).free
    required = expected_bytes + reserve_bytes
    if free_bytes < required:
        raise RuntimeError(
            f"insufficient free space on {usage_root}: need at least {required} bytes, "
            f"have {free_bytes}"
        )


def parse_capture_log(log_text: str, args: argparse.Namespace) -> dict[str, Any]:
    actual_rate = parse_log_float(ACTUAL_RATE_RE, log_text, 1e6)
    actual_freq = parse_log_float(ACTUAL_FREQ_RE, log_text, 1e6)
    actual_gain = parse_log_float(ACTUAL_GAIN_RE, log_text)
    actual_bandwidth = parse_log_float(ACTUAL_BW_RE, log_text, 1e6)
    received = RECEIVED_SAMPLES_RE.search(log_text)
    received_samples = int(received.group(1)) if received else None
    received_duration = float(received.group(2)) if received else None
    write_speed = parse_log_float(WRITE_SPEED_RE, log_text)

    textual_overflow_lines = [
        line.strip()
        for line in log_text.splitlines()
        if "overflow" in line.lower()
    ]
    disk_warning_lines = [
        line.strip()
        for line in textual_overflow_lines
        if "likely to occur" in line.lower() or "write test" in line.lower()
    ]

    return {
        "actual_sample_rate_sps": actual_rate,
        "actual_center_frequency_hz": actual_freq,
        "actual_gain_db": actual_gain,
        "actual_rx_bandwidth_hz": actual_bandwidth,
        "samples": received_samples,
        "received_duration_s": received_duration,
        "disk_write_test_mb_s": write_speed,
        "overflow_text_matches": textual_overflow_lines,
        "disk_write_warning_lines": disk_warning_lines,
        "actual_rate_detected": actual_rate is not None,
        "actual_bandwidth_matches_requested": (
            actual_bandwidth is not None
            and abs(actual_bandwidth - args.bandwidth_hz) <= 1_000.0
        ),
        "received_sample_count_detected": received_samples is not None,
    }


def build_capture_metadata(
    args: argparse.Namespace,
    target: Target,
    run_id: str,
    run_root: Path,
    iq_path: Path,
    log_path: Path,
    capture_command: list[str],
    capture_returncode: int,
    log_info: dict[str, Any],
) -> dict[str, Any]:
    actual_rate = log_info.get("actual_sample_rate_sps")
    samples = log_info.get("samples")
    file_size = iq_path.stat().st_size if iq_path.is_file() else 0
    expected_file_size = int(samples * 4) if samples is not None else None
    expected_samples = int(round((actual_rate or args.expected_sample_rate_sps) * args.duration_s))
    sample_fraction = (samples / expected_samples) if samples is not None and expected_samples else None

    return {
        "schema_version": 1,
        "capture_id": run_id,
        "target": target.name,
        "source": "usrp_x310",
        "device_model": "X310",
        "usrp_args": args.usrp_args,
        "sample_format": "sc16_le_interleaved_iq",
        "iq_format": "int16",
        "bytes_per_complex_sample": 4,
        "timestamp_mode": "sample_index",
        "rx_tags_csv": "",
        "requested_sample_rate_sps": args.sample_rate_sps,
        "actual_sample_rate_sps": actual_rate,
        "expected_sample_rate_sps_for_space_check": args.expected_sample_rate_sps,
        "requested_center_frequency_hz": args.center_frequency_hz,
        "actual_center_frequency_hz": log_info.get("actual_center_frequency_hz"),
        "requested_bandwidth_hz": args.bandwidth_hz,
        "processing_bandwidth_hz": args.bandwidth_hz,
        "actual_rx_bandwidth_hz": log_info.get("actual_rx_bandwidth_hz"),
        "requested_gain_db": args.gain_db,
        "actual_gain_db": log_info.get("actual_gain_db"),
        "antenna": args.antenna,
        "channel": args.channel,
        "requested_duration_seconds": args.duration_s,
        "received_duration_seconds": log_info.get("received_duration_s"),
        "samples": samples,
        "expected_samples_at_actual_rate": expected_samples,
        "sample_count_fraction_of_expected": sample_fraction,
        "file_size_bytes": file_size,
        "expected_file_size_bytes_from_received_samples": expected_file_size,
        "file_size_matches_received_samples": expected_file_size == file_size if expected_file_size is not None else False,
        "actual_bandwidth_matches_requested": log_info.get("actual_bandwidth_matches_requested", False),
        "disk_write_test_mb_s": log_info.get("disk_write_test_mb_s"),
        "overflow_text_matches": log_info.get("overflow_text_matches", []),
        "disk_write_warning_lines": log_info.get("disk_write_warning_lines", []),
        "capture_returncode": capture_returncode,
        "capture_command": capture_command,
        "capture_log_path": str(log_path),
        "iq_path": str(iq_path),
        "run_root": str(run_root),
        "notes": [
            "rx_samples_to_file does not expose UHD rx_time tags; parser replay uses sample_index timestamps.",
            "The actual X310 rate is taken from the UHD log and must be used by the channelizer.",
            "Review capture.log for overflow indicators; rx_samples_to_file does not emit the framed capture overflow counter.",
        ],
    }


def parser_environment(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    source_root = args.ble_root
    pythonpath = [
        str(source_root / "experiment"),
        str(source_root / "ble_fun_test"),
        str(source_root / "build-native"),
    ]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath)
    return env


def build_parser_command(
    args: argparse.Namespace,
    metadata_path: Path,
    iq_path: Path,
    output_dir: Path,
    actual_sample_rate_sps: float,
) -> list[str]:
    command = [
        str(args.parser_python),
        str(args.parser_entrypoint),
        "--source",
        "file",
        "--input-bin",
        str(iq_path),
        "--metadata",
        str(metadata_path),
        "--timestamp-mode",
        "sample_index",
        "--sample-rate",
        str(actual_sample_rate_sps),
        "--center-freq",
        str(args.center_frequency_hz),
        "--bandwidth",
        str(args.bandwidth_hz),
        "--subband-sample-rate",
        str(args.subband_sample_rate_sps),
        "--iq-format",
        "int16",
        "--output-dir",
        str(output_dir),
        "--chunk-samples",
        str(args.chunk_samples),
        "--overlap-samples",
        str(args.overlap_samples),
        "--ble-parser-backend",
        "cpp",
        "--bredr-parser-backend",
        "hybrid",
        "--cpp-parser-threads",
        str(args.cpp_parser_threads),
        "--cuda-device",
        str(args.cuda_device),
        "--skip-bredr",
        "--use-cuda",
        "--cuda-threshold-detect",
        "--cuda-fuse-target-dsp",
        "--cuda-batch-targets",
        "--quiet",
        "--timing",
    ]
    if args.parser_cpus:
        command = ["taskset", "-c", args.parser_cpus] + command
    if args.max_chunks > 0:
        command.extend(["--max-chunks", str(args.max_chunks)])
    return command


def build_two_stage_parser_command(
    args: argparse.Namespace,
    metadata_path: Path,
    iq_path: Path,
    output_dir: Path,
    actual_sample_rate_sps: float,
) -> list[str]:
    command = [
        str(args.parser_python),
        str(DEFAULT_TWO_STAGE_SCRIPT),
        "--iq-path",
        str(iq_path),
        "--metadata-path",
        str(metadata_path),
        "--output-dir",
        str(output_dir),
        "--parser-python",
        str(args.parser_python),
        "--parser-entrypoint",
        str(args.parser_entrypoint),
        "--ble-root",
        str(args.ble_root),
        "--sample-rate-sps",
        str(actual_sample_rate_sps),
        "--subband-sample-rate-sps",
        str(args.subband_sample_rate_sps),
        "--center-frequency-hz",
        str(args.center_frequency_hz),
        "--bandwidth-hz",
        str(args.bandwidth_hz),
        "--bootstrap-duration-s",
        str(args.bootstrap_duration_s),
        "--known-aa-min-count",
        str(args.known_aa_min_count),
        "--known-aa-hamming-tolerance",
        str(args.known_aa_hamming_tolerance),
        "--stage2-known-aa-bit-tolerance",
        str(args.stage2_known_aa_bit_tolerance),
        "--stage1-ble-score-threshold",
        str(args.stage1_ble_score_threshold),
        "--stage2-ble-score-threshold",
        str(args.stage2_ble_score_threshold),
        "--overlap-samples",
        str(args.overlap_samples),
        "--chunk-samples",
        str(args.chunk_samples),
        "--cpp-parser-threads",
        str(args.cpp_parser_threads),
        "--cuda-device",
        str(args.cuda_device),
        "--frontend-backend",
        str(args.frontend_backend),
        "--parser-cpus",
        str(args.parser_cpus),
        "--stage1-ble-parser-backend",
        "cpp",
        "--stage2-ble-parser-backend",
        "python",
        "--physical-length-backend",
        str(args.physical_length_backend),
        "--length-cuda-batch-size",
        str(args.length_cuda_batch_size),
    ]
    if args.measure_physical_length:
        command.append("--measure-physical-length")
    return command


def run_parser(
    args: argparse.Namespace,
    run_root: Path,
    metadata_path: Path,
    iq_path: Path,
    actual_sample_rate_sps: float,
) -> dict[str, Any]:
    if args.two_stage_known_aa:
        output_dir = run_root / "sdr" / "two_stage_known_aa"
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        command = build_two_stage_parser_command(
            args,
            metadata_path,
            iq_path,
            output_dir,
            actual_sample_rate_sps,
        )
        command_path = output_dir.parent / "two_stage_known_aa_command.txt"
        log_path = output_dir.parent / "two_stage_known_aa_runner.log"
        command_path.write_text(shlex.join(command) + "\n", encoding="utf-8")
        start_ns = time.time_ns()
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            log.write("$ " + shlex.join(command) + "\n")
            log.flush()
            proc = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=parser_environment(args),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        end_ns = time.time_ns()
        summary_path = output_dir / "validation_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        result = {
            "mode": "two_stage_known_access_address",
            "returncode": proc.returncode,
            "start_epoch_ns": start_ns,
            "end_epoch_ns": end_ns,
            "command": command,
            "command_path": str(command_path),
            "log_path": str(log_path),
            "output_dir": str(output_dir),
            "validation_summary": str(summary_path),
            "valid": proc.returncode == 0 and bool(summary.get("valid")),
        }
        write_json(output_dir.parent / "two_stage_known_aa_parse_status.json", result)
        return result

    output_dir = run_root / "sdr"
    if output_dir.exists():
        raise FileExistsError(f"parser output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    command = build_parser_command(args, metadata_path, iq_path, output_dir, actual_sample_rate_sps)
    command_path = output_dir / "command.txt"
    log_path = output_dir / "parse.log"
    command_path.write_text(shlex.join(command) + "\n", encoding="utf-8")
    start_ns = time.time_ns()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + shlex.join(command) + "\n")
        log.flush()
        proc = subprocess.run(
            command,
            cwd=args.ble_root,
            env=parser_environment(args),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    end_ns = time.time_ns()
    outputs = {
        "ble_packets": output_dir / "ble_packets.csv",
        "btclassic_packets": output_dir / "btclassic_packets.csv",
        "packet_events": output_dir / "packet_events.csv",
        "target_selection": output_dir / "target_selection.csv",
    }
    result = {
        "returncode": proc.returncode,
        "start_epoch_ns": start_ns,
        "end_epoch_ns": end_ns,
        "command": command,
        "command_path": str(command_path),
        "log_path": str(log_path),
        "output_dir": str(output_dir),
        "outputs": {name: str(path) for name, path in outputs.items()},
        "output_exists": {name: path.is_file() for name, path in outputs.items()},
        "valid": proc.returncode == 0 and all(path.is_file() for path in outputs.values()),
    }
    write_json(output_dir / "parse_status.json", result)
    return result


def target_list(args: argparse.Namespace) -> list[Target]:
    if args.target == "nvme":
        return [Target("nvme", args.nvme_root)]
    if args.target == "pssd":
        return [Target("pssd", args.pssd_root)]
    return [Target("nvme", args.nvme_root), Target("pssd", args.pssd_root)]


def run_target(args: argparse.Namespace, target: Target, run_id: str) -> dict[str, Any]:
    target_root = target.root.expanduser().resolve()
    run_root = target_root / run_id
    if run_root.exists():
        raise FileExistsError(f"run directory already exists: {run_root}")
    if not args.capture_bin.is_file():
        raise FileNotFoundError(f"UHD capture program not found: {args.capture_bin}")
    if not args.capture_bin.stat().st_mode & 0o111:
        raise PermissionError(f"UHD capture program is not executable: {args.capture_bin}")
    if args.with_sdr_parse:
        if not args.parser_python.is_file():
            raise FileNotFoundError(f"parser Python not found: {args.parser_python}")
        if not args.parser_entrypoint.is_file():
            raise FileNotFoundError(f"parser entrypoint not found: {args.parser_entrypoint}")

    run_root.mkdir(parents=True, exist_ok=False)
    iq_dir = run_root / "iq"
    iq_dir.mkdir()
    iq_path = iq_dir / "capture.sc16"
    log_path = iq_dir / "capture.log"
    metadata_path = iq_dir / "metadata.json"
    status_path = run_root / "run_status.json"
    ensure_capture_space(args, target_root)

    command = build_capture_command(args, iq_path)
    status: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "target": target.name,
        "status": "running",
        "stage": "x310_capture",
        "created_epoch_ns": time.time_ns(),
        "parameters": {
            "usrp_args": args.usrp_args,
            "requested_sample_rate_sps": args.sample_rate_sps,
            "requested_bandwidth_hz": args.bandwidth_hz,
            "center_frequency_hz": args.center_frequency_hz,
            "gain_db": args.gain_db,
            "antenna": args.antenna,
            "channel": args.channel,
            "duration_s": args.duration_s,
            "subband_sample_rate_sps": args.subband_sample_rate_sps,
            "with_sdr_parse": args.with_sdr_parse,
        },
        "paths": {
            "run_root": str(run_root),
            "iq": str(iq_dir),
            "capture": str(iq_path),
            "capture_log": str(log_path),
            "metadata": str(metadata_path),
        },
        "capture_command": command,
    }
    write_json(status_path, status)

    env = capture_environment(args)
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
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    log_info = parse_capture_log(log_text, args)
    metadata = build_capture_metadata(
        args,
        target,
        run_id,
        run_root,
        iq_path,
        log_path,
        command,
        proc.returncode,
        log_info,
    )
    metadata.update({"start_epoch_ns": start_ns, "end_epoch_ns": end_ns})
    write_json(metadata_path, metadata)
    status["capture"] = metadata
    status["stage"] = "sdr_parse" if args.with_sdr_parse else "complete"
    write_json(status_path, status)

    capture_ok = (
        proc.returncode == 0
        and iq_path.is_file()
        and iq_path.stat().st_size > 0
        and bool(log_info.get("actual_sample_rate_sps"))
        and bool(log_info.get("actual_bandwidth_matches_requested"))
        and bool(metadata.get("file_size_matches_received_samples"))
        and (
            metadata.get("sample_count_fraction_of_expected") is not None
            and metadata["sample_count_fraction_of_expected"] >= args.min_sample_fraction
        )
    )
    if args.with_sdr_parse:
        if not capture_ok:
            status["sdr_parse"] = {
                "valid": False,
                "skipped": True,
                "reason": f"X310 capture validation failed; see {log_path}",
            }
        else:
            parser_result = run_parser(
                args,
                run_root,
                metadata_path,
                iq_path,
                float(log_info["actual_sample_rate_sps"]),
            )
            status["sdr_parse"] = parser_result
            capture_ok = capture_ok and bool(parser_result["valid"])

    status.update(
        {
            "status": "passed" if capture_ok else "failed",
            "stage": "complete" if capture_ok else "validation",
            "completed_epoch_ns": time.time_ns(),
        }
    )
    write_json(status_path, status)
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("nvme", "pssd", "both"), default="both")
    parser.add_argument("--capture-id", default="", help="Run directory name; defaults to a timestamped X310 run ID.")
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--sample-rate-sps", type=float, default=80e6, help="Requested X310 rate; UHD currently selects 100 MS/s.")
    parser.add_argument("--expected-sample-rate-sps", type=float, default=100e6, help="Conservative rate for disk-space checks.")
    parser.add_argument("--min-sample-fraction", type=float, default=0.99, help="Minimum received/expected sample ratio for a passed capture.")
    parser.add_argument("--bandwidth-hz", type=float, default=80e6)
    parser.add_argument("--center-frequency-hz", type=float, default=2440e6)
    parser.add_argument("--gain-db", type=float, default=50.0)
    parser.add_argument("--antenna", default="RX2")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--usrp-args", default="type=x300,addr=192.168.40.2,master_clock_rate=200e6")
    parser.add_argument("--capture-bin", type=Path, default=DEFAULT_CAPTURE_BIN)
    parser.add_argument("--uhd-library", type=Path, default=DEFAULT_UHD_LIBRARY)
    parser.add_argument("--samples-per-buffer", type=int, default=8000)
    parser.add_argument("--setup-s", type=float, default=1.0)
    parser.add_argument("--reserve-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--nvme-root", type=Path, default=DEFAULT_NVME_ROOT)
    parser.add_argument("--pssd-root", type=Path, default=DEFAULT_PSSD_ROOT)
    parser.add_argument("--with-sdr-parse", action="store_true", help="Replay each completed capture through PhantomChannel receiver.")
    parser.add_argument(
        "--two-stage-known-aa",
        action="store_true",
        help="Use PhantomChannel's opt-in 2-second bootstrap + known-AA stage-2 parser.",
    )
    parser.add_argument("--bootstrap-duration-s", type=float, default=2.0)
    parser.add_argument("--known-aa-min-count", type=int, default=10)
    parser.add_argument("--known-aa-hamming-tolerance", type=int, default=2)
    parser.add_argument(
        "--stage2-known-aa-bit-tolerance",
        type=int,
        default=1,
        help="Add this many synthetic AA Hamming bits around canonical AAs; observed aliases stay included (default: 1).",
    )
    parser.add_argument("--stage1-ble-score-threshold", type=float, default=3.0)
    parser.add_argument(
        "--stage2-ble-score-threshold",
        type=float,
        default=6.0,
        help="Stage-2 physical/decode length mismatch tolerance in bytes (default: 6).",
    )
    parser.add_argument(
        "--measure-physical-length",
        action="store_true",
        help="With --two-stage-known-aa, measure IQ burst duration minus standard BLE length.",
    )
    parser.add_argument(
        "--physical-length-backend",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Backend for optional physical-length measurement; auto prefers the CUDA batch path.",
    )
    parser.add_argument("--length-cuda-batch-size", type=int, default=16)
    parser.add_argument("--ble-root", type=Path, default=DEFAULT_BLE_ROOT)
    parser.add_argument("--parser-python", type=Path, default=DEFAULT_PARSER_PYTHON)
    parser.add_argument("--parser-entrypoint", type=Path, default=DEFAULT_PARSER_ENTRYPOINT)
    parser.add_argument("--subband-sample-rate-sps", type=float, default=4e6)
    parser.add_argument("--chunk-samples", type=int, default=16_000_000)
    parser.add_argument("--overlap-samples", type=int, default=200_000)
    parser.add_argument("--cpp-parser-threads", type=int, default=4)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument(
        "--frontend-backend",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Two-stage DSP frontend; auto uses CUDA when available, otherwise CPU fallback.",
    )
    parser.add_argument("--parser-cpus", default="", help="Optional taskset CPU list for the parser.")
    parser.add_argument("--max-chunks", type=int, default=0, help="Limit parser chunks; 0 parses the full IQ file.")
    parser.add_argument("--dry-run", action="store_true", help="Print capture plans without touching hardware or disks.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.capture_bin = args.capture_bin.expanduser().resolve()
    args.uhd_library = args.uhd_library.expanduser().resolve()
    args.ble_root = args.ble_root.expanduser().resolve()
    # Keep the venv launcher path intact. Resolving .venv-cuda/bin/python can
    # turn it into /usr/bin/python3 and silently drop the CUDA environment.
    args.parser_python = args.parser_python.expanduser()
    args.parser_entrypoint = args.parser_entrypoint.expanduser().resolve()
    args.nvme_root = args.nvme_root.expanduser().resolve()
    args.pssd_root = args.pssd_root.expanduser().resolve()

    if args.duration_s <= 0:
        parser.error("--duration-s must be positive")
    if args.sample_rate_sps <= 0 or args.expected_sample_rate_sps <= 0:
        parser.error("sample rates must be positive")
    if not 0 < args.min_sample_fraction <= 1:
        parser.error("--min-sample-fraction must be in the interval (0, 1]")
    if args.bandwidth_hz <= 0 or args.subband_sample_rate_sps <= 0:
        parser.error("bandwidth and subband sample rate must be positive")
    if abs((args.sample_rate_sps / args.subband_sample_rate_sps) - round(args.sample_rate_sps / args.subband_sample_rate_sps)) > 1e-6:
        parser.error("--sample-rate-sps must be an integer multiple of --subband-sample-rate-sps")
    if args.bandwidth_hz > args.sample_rate_sps:
        parser.error("--bandwidth-hz cannot exceed --sample-rate-sps")
    if args.two_stage_known_aa and not args.with_sdr_parse:
        parser.error("--two-stage-known-aa requires --with-sdr-parse")
    if args.bootstrap_duration_s <= 0:
        parser.error("--bootstrap-duration-s must be positive")
    if args.known_aa_min_count < 1:
        parser.error("--known-aa-min-count must be positive")
    if not 0 <= args.known_aa_hamming_tolerance <= 32:
        parser.error("--known-aa-hamming-tolerance must be in [0, 32]")
    if not 0 <= args.stage2_known_aa_bit_tolerance <= 2:
        parser.error("--stage2-known-aa-bit-tolerance must be in [0, 2]")
    if args.stage1_ble_score_threshold <= 0 or args.stage2_ble_score_threshold <= 0:
        parser.error("BLE score thresholds must be positive")
    if args.length_cuda_batch_size < 1:
        parser.error("--length-cuda-batch-size must be positive")
    if args.target == "both" and args.nvme_root == args.pssd_root:
        parser.error("NVMe and PSSD roots must be different when --target=both")

    run_id = args.capture_id or f"{utcish_run_stamp()}_x310_bw{slug(args.bandwidth_hz / 1e6)}mhz"
    targets = target_list(args)
    plans = []
    for target in targets:
        run_root = target.root.expanduser().resolve() / run_id
        plans.append(
            {
                "target": target.name,
                "run_root": str(run_root),
                "capture_command": build_capture_command(args, run_root / "iq" / "capture.sc16"),
            }
        )

    if args.dry_run:
        print(json.dumps({"run_id": run_id, "plans": plans}, indent=2, sort_keys=True))
        return 0

    summaries = []
    overall_ok = True
    for target in targets:
        try:
            summary = run_target(args, target, run_id)
        except Exception as exc:
            overall_ok = False
            summary = {
                "run_id": run_id,
                "target": target.name,
                "status": "failed",
                "reason": str(exc),
            }
        summaries.append(summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        overall_ok = overall_ok and summary.get("status") == "passed"

    return 0 if overall_ok else 2


if __name__ == "__main__":
    sys.exit(main())
