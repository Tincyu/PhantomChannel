#!/usr/bin/env python3
"""Parse an archived CIS X310 IQ capture on the PSSD.

The capture coordinator deliberately keeps IQ collection and parser replay
separate.  This tool creates parser metadata from the archived UHD log, runs
the same BLE PFB parser used by the existing X310 workflow, and then invokes
the CIS evidence analyzer.  It never writes back to the NVMe staging tree.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
from portable_paths import BLE_ROOT, PARSER_PYTHON  # noqa: E402

DEFAULT_BLE_ROOT = BLE_ROOT
DEFAULT_PARSER_PYTHON = PARSER_PYTHON
DEFAULT_PARSER_ENTRYPOINT = DEFAULT_BLE_ROOT / "experiment/bt_40m_pfb_realtime.py"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def log_float(pattern: str, text: str, default: float) -> float:
    match = re.search(pattern, text, re.IGNORECASE)
    return float(match.group(1)) * (1e6 if "Msps" in pattern else 1.0) if match else default


def parser_environment(ble_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    roots = [ble_root / "experiment", ble_root / "ble_fun_test", ble_root / "build-native"]
    env["PYTHONPATH"] = ":".join(str(item) for item in roots) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def build_command(
    *,
    parser_python: Path,
    parser_entrypoint: Path,
    iq_path: Path,
    metadata_path: Path,
    output_dir: Path,
    actual_rate: float,
    center_hz: float,
    bandwidth_hz: float,
    subband_rate: float,
    chunk_samples: int,
    overlap_samples: int,
    cpp_threads: int,
    cuda_device: int,
    parser_cpus: str,
    max_chunks: int,
    known_ble_aa: str,
) -> list[str]:
    command = [
        str(parser_python), str(parser_entrypoint),
        "--source", "file",
        "--input-bin", str(iq_path),
        "--metadata", str(metadata_path),
        "--timestamp-mode", "sample_index",
        "--sample-rate", str(actual_rate),
        "--center-freq", str(center_hz),
        "--bandwidth", str(bandwidth_hz),
        "--subband-sample-rate", str(subband_rate),
        "--iq-format", "int16",
        "--output-dir", str(output_dir),
        "--chunk-samples", str(chunk_samples),
        "--overlap-samples", str(overlap_samples),
        "--ble-parser-backend", "cpp",
        "--bredr-parser-backend", "hybrid",
        "--cpp-parser-threads", str(cpp_threads),
        "--cuda-device", str(cuda_device),
        "--skip-bredr",
        "--use-cuda",
        "--cuda-threshold-detect",
        "--cuda-fuse-target-dsp",
        "--cuda-batch-targets",
        "--quiet",
        "--timing",
    ]
    if parser_cpus:
        command = ["taskset", "-c", parser_cpus] + command
    if max_chunks > 0:
        command.extend(["--max-chunks", str(max_chunks)])
    if known_ble_aa:
        command.extend(["--known-ble-aa", known_ble_aa])
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ble-root", type=Path, default=DEFAULT_BLE_ROOT)
    parser.add_argument("--parser-python", type=Path, default=DEFAULT_PARSER_PYTHON)
    parser.add_argument("--parser-entrypoint", type=Path, default=DEFAULT_PARSER_ENTRYPOINT)
    parser.add_argument("--center-frequency-hz", type=float, default=2440e6)
    parser.add_argument("--bandwidth-hz", type=float, default=80e6)
    parser.add_argument("--subband-sample-rate-sps", type=float, default=4e6)
    parser.add_argument("--chunk-samples", type=int, default=16_000_000)
    parser.add_argument("--overlap-samples", type=int, default=200_000)
    parser.add_argument("--cpp-parser-threads", type=int, default=4)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--parser-cpus", default="")
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument(
        "--known-ble-aa",
        default="",
        help="Optional comma-separated AA list for targeted parser replay; never substitutes for pcap-derived AA.",
    )
    parser.add_argument("--aa-cis-hint", default="0x8dc48a55")
    args = parser.parse_args(argv)

    run_dir = args.run_dir.expanduser().resolve()
    iq_path = run_dir / "iq" / "capture.sc16"
    capture_log_path = run_dir / "iq" / "capture.log"
    metadata_path = run_dir / "iq" / "metadata.json"
    output_dir = run_dir / "sdr"
    if not iq_path.is_file() or iq_path.stat().st_size == 0:
        raise SystemExit(f"IQ file missing or empty: {iq_path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"parser output already exists: {output_dir}")
    log_text = capture_log_path.read_text(encoding="utf-8", errors="replace")
    rate_match = re.search(r"Actual RX Rate:\s*([0-9.+-eE]+)\s*Msps", log_text, re.IGNORECASE)
    bw_match = re.search(r"Actual RX Bandwidth:\s*([0-9.+-eE]+)\s*MHz", log_text, re.IGNORECASE)
    actual_rate = float(rate_match.group(1)) * 1e6 if rate_match else 100e6
    actual_bw = float(bw_match.group(1)) * 1e6 if bw_match else args.bandwidth_hz
    samples = iq_path.stat().st_size // 4
    metadata = {
        "schema_version": 1,
        "source": "usrp_x310_rx_samples_to_file",
        "iq_path": str(iq_path),
        "capture_log_path": str(capture_log_path),
        "capture_id": run_dir.name,
        "sample_format": "sc16_le_interleaved_iq",
        "iq_format": "int16",
        "bytes_per_complex_sample": 4,
        "samples": samples,
        "file_size_bytes": iq_path.stat().st_size,
        "actual_sample_rate_sps": actual_rate,
        "actual_rx_bandwidth_hz": actual_bw,
        "requested_sample_rate_sps": 80e6,
        "processing_bandwidth_hz": args.bandwidth_hz,
        "center_frequency_hz": args.center_frequency_hz,
        "gain_db_requested": 50.0,
        "note": "Created on PSSD from archived rx_samples_to_file output; no NVMe source is used.",
    }
    write_json(metadata_path, metadata)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(
        # Do not resolve this symlink: the CUDA venv's interpreter path may
        # resolve to /usr/bin/python3.12 while its site-packages (CuPy) are
        # only selected through the venv entrypoint.
        parser_python=args.parser_python.expanduser(),
        parser_entrypoint=args.parser_entrypoint.expanduser().resolve(),
        iq_path=iq_path,
        metadata_path=metadata_path,
        output_dir=output_dir,
        actual_rate=actual_rate,
        center_hz=args.center_frequency_hz,
        bandwidth_hz=args.bandwidth_hz,
        subband_rate=args.subband_sample_rate_sps,
        chunk_samples=args.chunk_samples,
        overlap_samples=args.overlap_samples,
        cpp_threads=args.cpp_parser_threads,
        cuda_device=args.cuda_device,
        parser_cpus=args.parser_cpus,
        max_chunks=args.max_chunks,
        known_ble_aa=args.known_ble_aa,
    )
    (output_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    log_path = output_dir / "parse.log"
    start_ns = time.time_ns()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + shlex.join(command) + "\n")
        log.flush()
        proc = subprocess.run(command, cwd=args.ble_root, env=parser_environment(args.ble_root), stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
    end_ns = time.time_ns()
    outputs = {name: output_dir / name for name in ("ble_packets.csv", "packet_events.csv", "target_selection.csv")}
    status = {
        "schema_version": 1,
        "returncode": proc.returncode,
        "start_epoch_ns": start_ns,
        "end_epoch_ns": end_ns,
        "command": command,
        "metadata": str(metadata_path),
        "outputs": {name: str(path) for name, path in outputs.items()},
        "output_exists": {name: path.is_file() for name, path in outputs.items()},
        "valid": proc.returncode == 0 and outputs["ble_packets.csv"].is_file(),
        "aa_cis_hint": args.aa_cis_hint,
        "known_ble_aa": args.known_ble_aa,
    }
    write_json(output_dir / "parse_status.json", status)
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0 if status["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
