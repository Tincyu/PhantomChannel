#!/usr/bin/env python3
"""Capture one independent event-timing repetition with the nRF52840 dongle.

This is a dongle-only coordinator for §6.  It intentionally does not start
X310 or write IQ.  Each invocation creates one pcapng session; the caller
must restart advertising or reconnect the link between repetitions.  Data is
staged under the NVMe root and copied to the archive root before the NVMe
session directory is removed.  ``--no-copy-delete`` is available for a
diagnostic dry run.

Examples:

  python3 tools/run_event_timing_capture.py \
    --traffic advertising --condition append-last-239B \
    --target-address d1:22:33:44:55:66 --duration-s 60 \
    --run-id timing_adv_last_rep1

  python3 tools/run_event_timing_capture.py \
    --traffic connection --condition central-side-8B \
    --target-address c0:de:52:84:00:33 --duration-s 60 \
    --run-id timing_conn_central_rep1
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NVME_ROOT = PROJECT_ROOT / "testdata"
DEFAULT_ARCHIVE_ROOT = Path("/path/to/PhantomChannel/experiments/event_timing")
DEFAULT_SNIFFER = Path("/path/to/.nrfutil/bin/nrfutil-ble-sniffer")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def normalize_address(value: str) -> str:
    parts = value.strip().lower().replace("-", ":").split(":")
    if len(parts) != 6 or any(len(part) not in (1, 2) for part in parts):
        raise ValueError(f"invalid BLE address: {value}")
    return ":".join(f"{int(part, 16):02x}" for part in parts)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def stop_process(process: subprocess.Popen[str] | None) -> int | None:
    if process is None:
        return None
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=8.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3.0)
    return process.returncode


def uart_collector(port: str, output: Path, stop_event: threading.Event, state: dict[str, Any], label: str) -> None:
    try:
        import serial
    except ImportError:
        state[f"{label}_error"] = "pyserial is not installed"
        return
    try:
        with serial.Serial(port=port, baudrate=115200, timeout=0.2) as handle:
            handle.reset_input_buffer()
            with output.open("wb") as stream:
                stream.write(f"# UART_START label={label} port={port} epoch_ns={time.time_ns()}\n".encode())
                while not stop_event.is_set():
                    data = handle.read(4096)
                    if data:
                        stream.write(data)
                        stream.flush()
                stream.write(f"\n# UART_STOP label={label} epoch_ns={time.time_ns()}\n".encode())
        state[f"{label}_ok"] = True
    except Exception as exc:  # preserve the failure in the manifest
        state[f"{label}_error"] = repr(exc)
        output.write_text(f"# UART_ERROR label={label} port={port} error={exc!r}\n", encoding="utf-8")


def build_sniffer_command(args: argparse.Namespace, pcap: Path) -> list[str]:
    return [
        str(args.sniffer_bin),
        "sniff",
        "--port", args.sniffer_port,
        "--output-pcap-file", str(pcap),
        "--follow", args.target_address,
        "--scan-follow-rsp",
        "--scan-follow-aux",
        "--scan-follow-aux-chain",
        "--scan-follow-aux-rsp",
        "--timeout", str(args.sniffer_timeout_ms),
        "--log-level", "info",
        "--log-output", "stdout",
    ]


def copy_then_delete(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"archive destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    shutil.rmtree(source)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traffic", choices=("advertising", "connection"), required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--target-address", required=True)
    parser.add_argument("--sniffer-bin", type=Path, default=DEFAULT_SNIFFER)
    parser.add_argument("--sniffer-port", default="/dev/ttyACM4")
    parser.add_argument("--sniffer-timeout-ms", type=int, default=500)
    parser.add_argument("--uart", action="append", metavar="LABEL=PORT", help="optional UART collector; repeatable")
    parser.add_argument("--nvme-root", type=Path, default=DEFAULT_NVME_ROOT)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--setup-s", type=float, default=1.0)
    parser.add_argument("--no-copy-delete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.duration_s <= 0:
        raise SystemExit("--duration-s must be positive")
    try:
        args.target_address = normalize_address(args.target_address)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.sniffer_bin = args.sniffer_bin.expanduser().resolve()
    args.nvme_root = args.nvme_root.expanduser().resolve()
    args.archive_root = args.archive_root.expanduser().resolve()
    run_id = args.run_id or f"{utc_stamp()}_timing_{args.traffic}_{args.condition}"
    nvme_run = args.nvme_root / run_id
    archive_run = args.archive_root / run_id
    if nvme_run.exists() or archive_run.exists():
        raise SystemExit(f"run already exists: {nvme_run} or {archive_run}")
    uart_specs = []
    for value in args.uart or []:
        if "=" not in value:
            raise SystemExit(f"--uart expects LABEL=PORT, got {value!r}")
        label, port = value.split("=", 1)
        uart_specs.append((label.strip(), port.strip()))

    pcap = nvme_run / "monitor/monitor.pcapng"
    sniffer_log = nvme_run / "monitor/sniffer.log"
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "traffic": args.traffic,
        "condition": args.condition,
        "duration_s": args.duration_s,
        "target_address": args.target_address,
        "sniffer_port": args.sniffer_port,
        "stage_root": str(nvme_run),
        "archive_root": str(archive_run),
        "sniffer_command": build_sniffer_command(args, pcap),
        "uart": [{"label": label, "port": port} for label, port in uart_specs],
        "copy_delete": not args.no_copy_delete,
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0

    (nvme_run / "monitor").mkdir(parents=True, exist_ok=False)
    (nvme_run / "ground_truth").mkdir()
    write_json(nvme_run / "session_manifest.json", {**manifest, "status": "prepared"})
    stop_event = threading.Event()
    uart_state: dict[str, Any] = {}
    threads: list[threading.Thread] = []
    for label, port in uart_specs:
        thread = threading.Thread(
            target=uart_collector,
            args=(port, nvme_run / "ground_truth" / f"{label}.log", stop_event, uart_state, label),
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    with sniffer_log.open("w", encoding="utf-8") as log:
        command = build_sniffer_command(args, pcap)
        process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
        write_json(nvme_run / "session_manifest.json", {**manifest, "status": "running", "started_epoch_ns": time.time_ns()})
        time.sleep(max(0.0, args.setup_s))
        deadline = time.monotonic() + args.duration_s
        while time.monotonic() < deadline and process.poll() is None:
            time.sleep(0.2)
        sniffer_returncode = stop_process(process)
    stop_event.set()
    for thread in threads:
        thread.join(timeout=2.0)
    manifest = {
        **manifest,
        "status": "captured",
        "sniffer_returncode": sniffer_returncode,
        "completed_epoch_ns": time.time_ns(),
        "pcap_bytes": pcap.stat().st_size if pcap.is_file() else 0,
        "uart_state": uart_state,
    }
    write_json(nvme_run / "session_manifest.json", manifest)
    if not pcap.is_file() or pcap.stat().st_size == 0:
        print(json.dumps({**manifest, "status": "failed_no_pcap"}, indent=2, ensure_ascii=False))
        return 2
    if args.no_copy_delete:
        print(json.dumps({**manifest, "archive_path": None, "nvme_deleted": False}, indent=2, ensure_ascii=False))
        return 0
    try:
        copy_then_delete(nvme_run, archive_run)
    except Exception as exc:
        print(json.dumps({**manifest, "status": "archive_failed", "archive_error": repr(exc)}, indent=2, ensure_ascii=False))
        return 2
    archived_manifest = {**manifest, "status": "archived", "archive_path": str(archive_run), "nvme_deleted": True}
    write_json(archive_run / "session_manifest.json", archived_manifest)
    print(json.dumps(archived_manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
