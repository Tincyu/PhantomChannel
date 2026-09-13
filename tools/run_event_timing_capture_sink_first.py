#!/usr/bin/env python3
"""Capture a connected timing calibration with the sink advertising first.

The nrfutil sniffer's command-line ``--follow`` path enters FOLLOWING very
quickly.  Starting it before the target advertises can therefore miss device
registration.  This coordinator resets the sink first, waits for its legacy
advertisements, starts the follow process, and only then resets the central.
The capture is archived from the NVMe staging directory and the exact staging
session is removed after a successful copy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from pip_program_common import flash_image, require_device
from program_event_timing_firmware import SERIAL_BY_BOARD, load_entry
from run_event_timing_capture import build_sniffer_command, stop_process, uart_collector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NVME_ROOT = PROJECT_ROOT / "testdata"
DEFAULT_ARCHIVE_ROOT = Path("/path/to/PhantomChannel/experiments/event_timing")
DEFAULT_SNIFFER = Path("/path/to/.nrfutil/bin/nrfutil-ble-sniffer")
DEFAULT_CENTRAL_JLINK_SERIAL = "NRF52840_SERIAL"
DEFAULT_CENTRAL_JLINK_DEVICE = "nRF52840_xxAA"


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


def halt_board(serial_number: str, device: str) -> dict[str, object]:
    """Keep a stale initiator from auto-connecting before follow is ready."""
    command = [
        "JLinkExe", "-SelectEmuBySN", serial_number, "-device", device,
        "-if", "SWD", "-speed", "4000", "-autoconnect", "1",
    ]
    result = subprocess.run(
        command, input="Reset\nHalt\nExit\n", text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    return {"command": command, "returncode": result.returncode, "ok": result.returncode == 0}


def normalize_address(value: str) -> str:
    parts = value.replace("-", ":").split(":")
    if len(parts) != 6:
        raise ValueError(f"invalid BLE address: {value}")
    return "".join(f"{int(part, 16):02x}" for part in parts)


def target_seen(log_path: Path, target_hex: str) -> bool:
    if not log_path.is_file():
        return False
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    for line in text.splitlines():
        if "device added" not in line.lower():
            continue
        # nrfutil's JSON log omits leading zeroes in individual address
        # octets (for example c0:de:52:84:0:3).  Compare parsed octets,
        # rather than a raw compact substring, so c0:de:52:84:00:03 is
        # recognized as the same address.
        for address in re.findall(r'"address"\s*:\s*"([0-9a-fA-F:.-]+)"', line):
            try:
                if normalize_address(address) == target_hex:
                    return True
            except ValueError:
                continue
    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--target-address", required=True)
    parser.add_argument("--advertiser-condition", default="conn_sink_52833")
    parser.add_argument("--initiator-condition", default="conn_central_normal")
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--registration-timeout-s", type=float, default=8.0)
    parser.add_argument("--advertising-settle-s", type=float, default=1.0)
    parser.add_argument("--sniffer-setup-s", type=float, default=1.0)
    parser.add_argument("--sniffer-port", default="/dev/ttyACM4")
    parser.add_argument("--central-uart", default="/dev/ttyACM0")
    parser.add_argument("--peripheral-uart", default="/dev/ttyACM2")
    parser.add_argument("--central-jlink-serial", default=DEFAULT_CENTRAL_JLINK_SERIAL)
    parser.add_argument("--central-jlink-device", default=DEFAULT_CENTRAL_JLINK_DEVICE)
    parser.add_argument("--no-halt-central", action="store_true",
                        help="do not hold a stale central before target registration")
    parser.add_argument("--nvme-root", type=Path, default=DEFAULT_NVME_ROOT)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--sniffer-bin", type=Path, default=DEFAULT_SNIFFER)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.duration_s <= 0:
        raise SystemExit("--duration-s must be positive")
    target = ":".join(f"{int(part, 16):02x}" for part in args.target_address.replace("-", ":").split(":"))
    target_hex = normalize_address(target)
    nvme_root = args.nvme_root.expanduser().resolve()
    archive_root = args.archive_root.expanduser().resolve()
    nvme_run = nvme_root / args.run_id
    archive_run = archive_root / args.run_id
    if nvme_run.exists() or archive_run.exists():
        raise SystemExit(f"run already exists: {nvme_run} or {archive_run}")

    monitor = nvme_run / "monitor"
    ground_truth = nvme_run / "ground_truth"
    monitor.mkdir(parents=True, exist_ok=False)
    ground_truth.mkdir()
    pcap = monitor / "monitor.pcapng"
    sniffer_log = monitor / "sniffer.log"
    central_log = ground_truth / "central.log"
    peripheral_log = ground_truth / "peripheral.log"
    sniffer_command = build_sniffer_command(
        argparse.Namespace(
            sniffer_bin=args.sniffer_bin.expanduser().resolve(),
            sniffer_port=args.sniffer_port,
            target_address=target,
            sniffer_timeout_ms=500,
        ),
        pcap,
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "run_id": args.run_id,
        "traffic": "connection",
        "condition": "normal",
        "duration_s": args.duration_s,
        "target_address": target,
        "sniffer_port": args.sniffer_port,
        "stage_root": str(nvme_run),
        "archive_root": str(archive_run),
        "capture_order": "sink_reset_and_advertise_before_sniffer_follow_then_central_reset",
        "advertiser_condition": args.advertiser_condition,
        "initiator_condition": args.initiator_condition,
        "sniffer_command": sniffer_command,
        "firmware": [],
        "uart": [
            {"label": "central", "port": args.central_uart},
            {"label": "peripheral", "port": args.peripheral_uart},
        ],
    }
    (nvme_run / "session_manifest.json").write_text(
        json.dumps({**manifest, "status": "prepared"}, indent=2) + "\n", encoding="utf-8"
    )

    if not args.no_halt_central:
        manifest["central_halt"] = halt_board(args.central_jlink_serial, args.central_jlink_device)
        if not manifest["central_halt"]["ok"]:
            raise RuntimeError("failed to halt central before sink advertising")
        (nvme_run / "session_manifest.json").write_text(
            json.dumps({**manifest, "status": "central_halted"}, indent=2) + "\n", encoding="utf-8"
        )

    stop_event = threading.Event()
    uart_state: dict[str, object] = {}
    threads = [
        threading.Thread(target=uart_collector, args=(args.central_uart, central_log, stop_event, uart_state, "central"), daemon=True),
        threading.Thread(target=uart_collector, args=(args.peripheral_uart, peripheral_log, stop_event, uart_state, "peripheral"), daemon=True),
    ]
    for thread in threads:
        thread.start()

    sniffer_process: subprocess.Popen[str] | None = None
    sniffer_returncode: int | None = None
    target_registered = False
    try:
        manifest["firmware"] = [flash_condition(args.advertiser_condition)]
        manifest["sink_reset_epoch_ns"] = time.time_ns()
        (nvme_run / "session_manifest.json").write_text(
            json.dumps({**manifest, "status": "sink_advertising"}, indent=2) + "\n", encoding="utf-8"
        )
        time.sleep(max(0.0, args.advertising_settle_s))

        with sniffer_log.open("w", encoding="utf-8") as log:
            sniffer_process = subprocess.Popen(
                sniffer_command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True,
            )
            registration_deadline = time.monotonic() + args.registration_timeout_s
            while time.monotonic() < registration_deadline and sniffer_process.poll() is None:
                if target_seen(sniffer_log, target_hex):
                    target_registered = True
                    break
                time.sleep(0.1)
            if not target_registered:
                raise RuntimeError("sniffer did not register target address before central reset")
            time.sleep(max(0.0, args.sniffer_setup_s))
            manifest["firmware"].append(flash_condition(args.initiator_condition))
            manifest["central_reset_epoch_ns"] = time.time_ns()
            manifest["target_registered_before_central"] = True
            manifest["status"] = "capturing"
            (nvme_run / "session_manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            deadline = time.monotonic() + args.duration_s
            while time.monotonic() < deadline and sniffer_process.poll() is None:
                time.sleep(0.2)
            sniffer_returncode = stop_process(sniffer_process)
            sniffer_process = None
    except Exception:
        if sniffer_process is not None:
            sniffer_returncode = stop_process(sniffer_process)
            sniffer_process = None
        stop_event.set()
        for thread in threads:
            thread.join(timeout=3.0)
        manifest.update({
            "status": "failed",
            "target_registered_before_central": target_registered,
            "sniffer_returncode": sniffer_returncode,
            "completed_epoch_ns": time.time_ns(),
            "pcap_bytes": pcap.stat().st_size if pcap.is_file() else 0,
            "uart_state": uart_state,
        })
        (nvme_run / "session_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        raise

    stop_event.set()
    for thread in threads:
        thread.join(timeout=3.0)
    manifest.update({
        "status": "captured",
        "target_registered_before_central": target_registered,
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
