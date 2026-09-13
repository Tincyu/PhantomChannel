#!/usr/bin/env python3
"""Capture a connected timing session with sniffer follow running before reset.

The normal capture coordinator starts the sniffer and assumes the boards are
already in a clean advertising/connection state.  This variant starts the
sniffer first, then flashes the matched sink and central images through the
PIP-plan ``nrfutil device program`` path, so the dongle can observe the
connection establishment packet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from pip_program_common import flash_image, require_device
from program_event_timing_firmware import load_entry, SERIAL_BY_BOARD
from run_event_timing_capture import build_sniffer_command, stop_process, uart_collector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NVME_ROOT = PROJECT_ROOT / "testdata"
DEFAULT_ARCHIVE_ROOT = Path("/path/to/PhantomChannel/experiments/event_timing")
DEFAULT_SNIFFER = Path("/path/to/.nrfutil/bin/nrfutil-ble-sniffer")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flash_condition(condition: str) -> dict[str, str]:
    entry = load_entry(condition)
    board = entry["board"]
    serial, board_version = SERIAL_BY_BOARD[board]
    image = Path(entry["image"])
    require_device(serial, board_version)
    flash_image(image, serial, board_version)
    return {
        "condition": condition,
        "image": str(image),
        "image_sha256": sha256(image),
        "serial_number": serial,
        "board_version": board_version,
        "method": "pip_program_common.flash_image -> nrfutil device program",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--target-address", required=True)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--sniffer-port", default="/dev/ttyACM4")
    parser.add_argument("--central-uart", default="/dev/ttyACM0")
    parser.add_argument("--peripheral-uart", default="/dev/ttyACM2")
    parser.add_argument("--nvme-root", type=Path, default=DEFAULT_NVME_ROOT)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--sniffer-bin", type=Path, default=DEFAULT_SNIFFER)
    parser.add_argument("--setup-s", type=float, default=2.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.duration_s <= 0:
        raise SystemExit("--duration-s must be positive")
    target = ":".join(f"{int(part, 16):02x}" for part in args.target_address.replace("-", ":").split(":"))
    nvme_root = args.nvme_root.expanduser().resolve()
    archive_root = args.archive_root.expanduser().resolve()
    nvme_run = nvme_root / args.run_id
    archive_run = archive_root / args.run_id
    if nvme_run.exists() or archive_run.exists():
        raise SystemExit(f"run already exists: {nvme_run} or {archive_run}")

    pcap = nvme_run / "monitor/monitor.pcapng"
    sniffer_log = nvme_run / "monitor/sniffer.log"
    central_log = nvme_run / "ground_truth/central.log"
    peripheral_log = nvme_run / "ground_truth/peripheral.log"
    nvme_run.joinpath("monitor").mkdir(parents=True, exist_ok=False)
    nvme_run.joinpath("ground_truth").mkdir()
    manifest = {
        "schema_version": 1,
        "run_id": args.run_id,
        "traffic": "connection",
        "condition": "normal",
        "duration_s": args.duration_s,
        "target_address": target,
        "sniffer_port": args.sniffer_port,
        "stage_root": str(nvme_run),
        "archive_root": str(archive_run),
        "capture_order": "sniffer_started_before_firmware_reset",
        "sniffer_command": build_sniffer_command(
            argparse.Namespace(
                sniffer_bin=args.sniffer_bin.resolve(),
                sniffer_port=args.sniffer_port,
                target_address=target,
                sniffer_timeout_ms=500,
            ),
            pcap,
        ),
        "firmware": [],
        "uart": [
            {"label": "central", "port": args.central_uart},
            {"label": "peripheral", "port": args.peripheral_uart},
        ],
    }
    (nvme_run / "session_manifest.json").write_text(
        json.dumps({**manifest, "status": "prepared"}, indent=2) + "\n", encoding="utf-8"
    )

    stop_event = threading.Event()
    uart_state: dict[str, object] = {}
    threads = [
        threading.Thread(target=uart_collector, args=(args.central_uart, central_log, stop_event, uart_state, "central"), daemon=True),
        threading.Thread(target=uart_collector, args=(args.peripheral_uart, peripheral_log, stop_event, uart_state, "peripheral"), daemon=True),
    ]
    for thread in threads:
        thread.start()

    with sniffer_log.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            manifest["sniffer_command"], cwd=PROJECT_ROOT,
            stdout=log, stderr=subprocess.STDOUT, text=True,
        )
        time.sleep(max(0.0, args.setup_s))
        manifest["firmware"].append(flash_condition("conn_sink_52833"))
        manifest["firmware"].append(flash_condition("conn_central_normal"))
        manifest["status"] = "firmware_reset_and_capture"
        manifest["firmware_reset_epoch_ns"] = time.time_ns()
        (nvme_run / "session_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        time.sleep(max(0.0, args.setup_s))
        deadline = time.monotonic() + args.duration_s
        while time.monotonic() < deadline and process.poll() is None:
            time.sleep(0.2)
        sniffer_returncode = stop_process(process)

    stop_event.set()
    for thread in threads:
        thread.join(timeout=3.0)
    manifest.update({
        "status": "captured",
        "sniffer_returncode": sniffer_returncode,
        "completed_epoch_ns": time.time_ns(),
        "pcap_bytes": pcap.stat().st_size if pcap.is_file() else 0,
        "uart_state": uart_state,
    })
    (nvme_run / "session_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if not pcap.is_file() or pcap.stat().st_size == 0:
        print(json.dumps({**manifest, "status": "failed_no_pcap"}, indent=2))
        return 2

    archive_run.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(nvme_run, archive_run)
    shutil.rmtree(nvme_run)
    archived_manifest = {
        **manifest,
        "status": "archived",
        "archive_path": str(archive_run),
        "nvme_deleted": True,
    }
    (archive_run / "session_manifest.json").write_text(
        json.dumps(archived_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(archived_manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
