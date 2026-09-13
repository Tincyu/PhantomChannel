#!/usr/bin/env python3
"""Capture a short isolated BLE behavioral trace with X310 and nRF Sniffer.

The script is intentionally a capture coordinator, not an ISO decoder.  It
starts UART collectors, the nRF Sniffer ACL/CIS-control capture, and the
frozen X310 80 MHz receiver before optionally resetting the two boards.  IQ is
staged on the local NVMe and copied to the archive root after the run; the
NVMe run directory is then removed, including an overflowed/partial IQ file.

The nRF Sniffer is used only for establishment context.  The X310 is the
source used later to search for CIS Access Address activity.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import serial
except ImportError as exc:  # pragma: no cover - checked at runtime
    raise SystemExit("pyserial is required for UART collection") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NVME_ROOT = PROJECT_ROOT / "testdata"
DEFAULT_ARCHIVE_ROOT = Path("/path/to/PhantomChannel/experiments/pip_cis_behavior")
DEFAULT_SNIFFER = Path("/path/to/.nrfutil/bin/nrfutil-ble-sniffer")
DEFAULT_CAPTURE_BIN = Path("/path/to/uhd-4.6.0.0/lib/uhd/examples/rx_samples_to_file")
DEFAULT_UHD_LIBRARY = Path("/path/to/uhd-4.6.0.0/lib")
DEFAULT_UHD_FIND = Path("/path/to/uhd-4.6.0.0/bin/uhd_find_devices")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def log_line(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip("\n") + "\n")


def normalize_address(value: str) -> str:
    parts = value.strip().lower().replace("-", ":").split(":")
    if len(parts) != 6 or any(not re.fullmatch(r"[0-9a-f]{1,2}", part) for part in parts):
        raise ValueError(f"invalid BLE address: {value}")
    return ":".join(f"{int(part, 16):02x}" for part in parts)


def address_appears_in_text(text: str, target: str) -> bool:
    """Match sniffer log addresses even when it omits leading zeroes."""
    target = normalize_address(target)
    for match in re.finditer(r"(?<![0-9a-f])([0-9a-f]{1,2}(?::[0-9a-f]{1,2}){5})(?![0-9a-f])", text.lower()):
        try:
            if normalize_address(match.group(1)) == target:
                return True
        except ValueError:
            continue
    return False


def run_checked(command: list[str], *, input_text: str = "", timeout_s: float = 30.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        check=False,
    )


def collect_uart(
    port: str,
    output_path: Path,
    label: str,
    baudrate: int,
    stop_event: threading.Event,
    state: dict[str, Any],
) -> None:
    start = time.time_ns()
    state[f"{label}_start_epoch_ns"] = start
    try:
        with serial.Serial(port=port, baudrate=baudrate, timeout=0.2) as ser:
            ser.reset_input_buffer()
            with output_path.open("wb") as output:
                output.write(f"# UART_COLLECTOR_START label={label} port={port} epoch_ns={start}\n".encode())
                while not stop_event.is_set():
                    data = ser.read(4096)
                    if data:
                        output.write(data)
                        output.flush()
                # Drain only the bytes already present; do not wait for a new boot.
                deadline = time.monotonic() + 0.5
                while time.monotonic() < deadline:
                    data = ser.read(4096)
                    if not data:
                        break
                    output.write(data)
                output.write(f"\n# UART_COLLECTOR_STOP label={label} epoch_ns={time.time_ns()}\n".encode())
    except Exception as exc:  # preserve the failure in the manifest and continue other collectors
        state[f"{label}_error"] = repr(exc)
        output_path.write_text(
            f"# UART_COLLECTOR_ERROR label={label} port={port} error={exc!r}\n",
            encoding="utf-8",
        )
    finally:
        state[f"{label}_end_epoch_ns"] = time.time_ns()


def reset_board(serial_number: str, device: str, output_path: Path) -> dict[str, Any]:
    command = [
        "JLinkExe",
        "-SelectEmuBySN",
        serial_number,
        "-device",
        device,
        "-if",
        "SWD",
        "-speed",
        "4000",
        "-autoconnect",
        "1",
    ]
    result = run_checked(command, input_text="Reset\nGo\nExit\n", timeout_s=20.0)
    output_path.write_text(result.stdout, encoding="utf-8", errors="replace")
    return {
        "command": command,
        "returncode": result.returncode,
        "log": str(output_path),
        "ok": result.returncode == 0,
    }


def halt_board(serial_number: str, device: str, output_path: Path) -> dict[str, Any]:
    """Reset and hold a board before the passive monitor learns its target."""
    command = [
        "JLinkExe",
        "-SelectEmuBySN",
        serial_number,
        "-device",
        device,
        "-if",
        "SWD",
        "-speed",
        "4000",
        "-autoconnect",
        "1",
    ]
    result = run_checked(command, input_text="Reset\nHalt\nExit\n", timeout_s=20.0)
    output_path.write_text(result.stdout, encoding="utf-8", errors="replace")
    return {
        "command": command,
        "returncode": result.returncode,
        "log": str(output_path),
        "ok": result.returncode == 0,
    }


def go_board(serial_number: str, device: str, output_path: Path) -> dict[str, Any]:
    """Release a board held by :func:`halt_board`."""
    command = [
        "JLinkExe",
        "-SelectEmuBySN",
        serial_number,
        "-device",
        device,
        "-if",
        "SWD",
        "-speed",
        "4000",
        "-autoconnect",
        "1",
    ]
    result = run_checked(command, input_text="Go\nExit\n", timeout_s=20.0)
    output_path.write_text(result.stdout, encoding="utf-8", errors="replace")
    return {
        "command": command,
        "returncode": result.returncode,
        "log": str(output_path),
        "ok": result.returncode == 0,
    }


def build_sniffer_command(args: argparse.Namespace, pcap_path: Path) -> list[str]:
    return [
        str(args.sniffer_bin),
        "sniff",
        "--port",
        args.sniffer_port,
        "--output-pcap-file",
        str(pcap_path),
        "--follow",
        args.target_address,
        "--scan-follow-rsp",
        "--scan-follow-aux",
        "--scan-follow-aux-chain",
        "--scan-follow-aux-rsp",
        "--timeout",
        str(args.sniffer_timeout_ms),
        "--log-level",
        "info",
        "--log-output",
        "stdout",
    ]


def build_extcap_command(args: argparse.Namespace, fifo: Path, control_in: Path, control_out: Path) -> list[str]:
    return [
        str(args.extcap_shim),
        "--capture",
        "--extcap-interface",
        args.sniffer_port,
        "--fifo",
        str(fifo),
        "--extcap-control-in",
        str(control_in),
        "--extcap-control-out",
        str(control_out),
        "--extcap-version",
        "4.0",
    ]


def extcap_control_packet(control_number: int, command: int, payload: bytes = b"") -> bytes:
    body = bytes([control_number, command]) + payload
    if len(body) > 0xFFFF:
        raise ValueError("extcap control payload is too long")
    return b"T" + len(body).to_bytes(3, "big") + body


class DynamicExtcapSniffer:
    """Drive nRF Sniffer's toolbar controls without a Wireshark GUI.

    The extcap control protocol is: ``T`` + three-byte network-order length +
    control number + command + payload.  Key control 1 value 7 means
    ``Follow LE address``; value control 2 carries ``address random``.
    """

    def __init__(self, args: argparse.Namespace, pcap_path: Path, monitor_dir: Path):
        self.args = args
        self.pcap_path = pcap_path
        self.monitor_dir = monitor_dir
        self.tmpdir = Path(tempfile.mkdtemp(prefix="nrf_extcap_", dir="/tmp"))
        self.fifo = self.tmpdir / "pcap.fifo"
        self.control_in = self.tmpdir / "control.in"
        self.control_out = self.tmpdir / "control.out"
        for path in (self.fifo, self.control_in, self.control_out):
            os.mkfifo(path)
        self.control_out_log = monitor_dir / "control-out.bin"
        self.control_in_log = monitor_dir / "control-in.bin"
        self.control_messages_log = monitor_dir / "control-messages.log"
        self.stdout_log = monitor_dir / "extcap.stdout"
        self.stderr_log = monitor_dir / "extcap.stderr"
        self.pcap_handle = self.pcap_path.open("wb")
        self.stdout_handle = self.stdout_log.open("w", encoding="utf-8")
        self.stderr_handle = self.stderr_log.open("w", encoding="utf-8")
        self.control_out_handle = self.control_out_log.open("wb")
        self.control_in_handle = self.control_in_log.open("wb")
        self.control_message_handle = self.control_messages_log.open("w", encoding="utf-8")
        self.control_in_fd = os.open(self.control_in, os.O_RDWR | os.O_NONBLOCK)
        self.control_out_fd = os.open(self.control_out, os.O_RDWR | os.O_NONBLOCK)
        self.cat = subprocess.Popen(["cat", str(self.fifo)], stdout=self.pcap_handle, stderr=subprocess.STDOUT)
        self.proc: subprocess.Popen[str] | None = None
        self.stop_event = threading.Event()
        self.reader_thread: threading.Thread | None = None
        self.buffer = bytearray()
        self.follow_sent = False
        self.follow_ack = False
        self.device_added = False
        self.sync: dict[str, Any] = {"device_added": False, "follow_sent": False, "follow_ack": False}

    def start(self) -> None:
        command = build_extcap_command(self.args, self.fifo, self.control_in, self.control_out)
        self.proc = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=self.stdout_handle,
            stderr=self.stderr_handle,
            text=True,
        )
        self.reader_thread = threading.Thread(target=self._read_control, daemon=True)
        self.reader_thread.start()

    def _send_control(self, control_number: int, payload: bytes) -> None:
        packet = extcap_control_packet(control_number, 1, payload)
        os.write(self.control_in_fd, packet)
        self.control_in_handle.write(packet)
        self.control_in_handle.flush()

    def _try_follow(self) -> None:
        if self.follow_sent:
            return
        self._send_control(1, b"7")
        self._send_control(2, f"{self.args.target_address} random".encode())
        self.follow_sent = True
        self.sync["follow_sent"] = True

    def _handle_message(self, control_number: int, command: int, payload: bytes) -> None:
        if control_number != 6:
            return
        text = payload.decode("utf-8", errors="replace")
        self.control_message_handle.write(text)
        if not text.endswith("\n"):
            self.control_message_handle.write("\n")
        self.control_message_handle.flush()
        text_lower = text.lower()
        target_compact = self.args.target_address.lower()
        target_seen = (
            target_compact in text_lower
            or target_compact.replace(":", "") in text_lower.replace(":", "")
            or address_appears_in_text(text_lower, target_compact)
        )
        device_added_message = (
            ("Device added:" in text or "Device with address" in text)
            and target_seen
            and ("random" in text_lower or '"type":"Random"' in text)
        )
        if device_added_message:
            self.device_added = True
            self.sync["device_added"] = True
            self._try_follow()
        if "Sending follow request" in text and target_seen:
            self.follow_ack = True
            self.sync["follow_ack"] = True

    def _read_control(self) -> None:
        while not self.stop_event.is_set():
            try:
                data = os.read(self.control_out_fd, 65536)
            except BlockingIOError:
                data = b""
            if data:
                self.control_out_handle.write(data)
                self.control_out_handle.flush()
                self.buffer.extend(data)
                while len(self.buffer) >= 4:
                    if self.buffer[0] != ord("T"):
                        del self.buffer[0]
                        continue
                    length = int.from_bytes(self.buffer[1:4], "big")
                    if length < 2 or len(self.buffer) < 4 + length:
                        break
                    control_number = self.buffer[4]
                    command = self.buffer[5]
                    payload = bytes(self.buffer[6:4 + length])
                    del self.buffer[:4 + length]
                    self._handle_message(control_number, command, payload)
            elif self.proc is not None and self.proc.poll() is not None:
                break
            time.sleep(0.01)

    def stop(self) -> int | None:
        self.stop_event.set()
        rc = stop_process(self.proc)
        if self.reader_thread is not None:
            self.reader_thread.join(timeout=2.0)
        if self.cat.poll() is None:
            self.cat.terminate()
            try:
                self.cat.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.cat.kill()
                self.cat.wait(timeout=3.0)
        self.sync.update({"device_added": self.device_added, "follow_sent": self.follow_sent, "follow_ack": self.follow_ack})
        (self.monitor_dir / "follow_sync.json").write_text(json.dumps(self.sync, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for fd in (self.control_in_fd, self.control_out_fd):
            try:
                os.close(fd)
            except OSError:
                pass
        for handle in (self.control_out_handle, self.control_in_handle, self.control_message_handle, self.stdout_handle, self.stderr_handle, self.pcap_handle):
            handle.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        return rc


def build_x310_command(args: argparse.Namespace, iq_path: Path) -> list[str]:
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


def check_x310(args: argparse.Namespace, log_path: Path) -> dict[str, Any]:
    command = [str(args.uhd_find), "--args", args.usrp_args]
    result = run_checked(command, timeout_s=30.0)
    log_path.write_text(result.stdout, encoding="utf-8", errors="replace")
    return {
        "command": command,
        "returncode": result.returncode,
        "found": result.returncode == 0 and "No UHD Devices Found" not in result.stdout,
        "log": str(log_path),
    }


def stop_process(process: subprocess.Popen[str] | None, timeout_s: float = 5.0) -> int | None:
    if process is None:
        return None
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=timeout_s)
    return process.returncode


def copy_then_delete(nvme_run: Path, archive_run: Path) -> None:
    if archive_run.exists():
        raise FileExistsError(f"archive run already exists: {archive_run}")
    archive_run.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(nvme_run, archive_run)
    # Explicitly requested experiment policy: no checksum pass here; remove
    # the NVMe staging directory, including partial/overflow IQ.
    shutil.rmtree(nvme_run)


def archived_capture_is_usable(archive_run: Path, *, sniffer_only: bool, x310_returncode: int | None) -> bool:
    """Return whether the archived capture contains the requested data.

    The NVMe staging directory is intentionally removed after the copy, so
    post-copy validation must use the archive path.  UHD may still be
    terminated by the coordinator after it has written a complete IQ file;
    in that case a non-zero signal return is acceptable only when the archive
    contains a non-empty IQ file.
    """
    pcap = archive_run / "monitor" / "monitor.pcapng"
    if not pcap.is_file() or pcap.stat().st_size == 0:
        return False
    if sniffer_only:
        return True
    iq = archive_run / "iq" / "capture.sc16"
    if not iq.is_file() or iq.stat().st_size == 0:
        return False
    return x310_returncode == 0 or x310_returncode == -15


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="", help="Unique run name; defaults to a UTC timestamp.")
    parser.add_argument("--trace-kind", choices=("cis", "pip", "acl"), default="cis", help="Behavioral trace label for the manifest.")
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--target-address", default="c3:79:59:f9:a5:c0")
    parser.add_argument("--sniffer-port", default="/dev/ttyACM4")
    parser.add_argument("--sniffer-bin", type=Path, default=DEFAULT_SNIFFER)
    parser.add_argument("--extcap-shim", type=Path, default=Path("/path/to/.local/lib/wireshark/extcap/nrfutil-ble-sniffer-shim"))
    parser.add_argument("--sniffer-timeout-ms", type=int, default=500)
    parser.add_argument("--central-uart", default="/dev/ttyACM2")
    parser.add_argument("--peripheral-uart", default="/dev/ttyACM0")
    parser.add_argument("--uart-baudrate", type=int, default=115200)
    parser.add_argument("--capture-bin", type=Path, default=DEFAULT_CAPTURE_BIN)
    parser.add_argument("--uhd-find", type=Path, default=DEFAULT_UHD_FIND)
    parser.add_argument("--uhd-library", type=Path, default=DEFAULT_UHD_LIBRARY)
    parser.add_argument("--usrp-args", default="type=x300,addr=192.168.40.2,master_clock_rate=200e6")
    parser.add_argument("--center-frequency-hz", type=float, default=2440e6)
    parser.add_argument("--sample-rate-sps", type=float, default=80e6)
    parser.add_argument("--bandwidth-hz", type=float, default=80e6)
    parser.add_argument("--gain-db", type=float, default=50.0)
    parser.add_argument("--antenna", default="RX2")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--samples-per-buffer", type=int, default=8000)
    parser.add_argument("--setup-s", type=float, default=1.0)
    parser.add_argument("--nvme-root", type=Path, default=DEFAULT_NVME_ROOT)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--central-jlink", default="1050631205")
    parser.add_argument("--peripheral-jlink", default="1050216757")
    parser.add_argument("--central-device", default="nRF52833_xxAA")
    parser.add_argument("--peripheral-device", default="nRF52840_xxAA")
    parser.add_argument("--reset-boards", action="store_true", help="Reset peripheral then central after monitors start.")
    parser.add_argument(
        "--hold-central-until-follow",
        action="store_true",
        help=(
            "Reset and halt the central before starting the dynamic sniffer; "
            "release it only after the sniffer has registered and followed the target. "
            "Use the correct central J-Link/device arguments for the traffic role."
        ),
    )
    parser.add_argument(
        "--follow-settle-s",
        type=float,
        default=1.0,
        help="settling time after the dynamic follow command before releasing a held central",
    )
    parser.add_argument("--sniffer-only", action="store_true", help="Collect only the short dongle/UART establishment trace; do not require or start X310.")
    parser.add_argument("--static-follow", action="store_true", help="Use direct nrfutil --follow instead of dynamic extcap follow control.")
    parser.add_argument("--no-copy-delete", action="store_true", help="Keep NVMe staging; diagnostic only.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.duration_s <= 0:
        raise SystemExit("--duration-s must be positive")
    if args.follow_settle_s < 0:
        raise SystemExit("--follow-settle-s must be non-negative")
    if args.hold_central_until_follow and args.static_follow:
        raise SystemExit("--hold-central-until-follow requires dynamic extcap follow")
    if args.bandwidth_hz > args.sample_rate_sps:
        raise SystemExit("--bandwidth-hz cannot exceed --sample-rate-sps")
    args.target_address = normalize_address(args.target_address)
    args.sniffer_bin = args.sniffer_bin.expanduser().resolve()
    args.extcap_shim = args.extcap_shim.expanduser().resolve()
    args.capture_bin = args.capture_bin.expanduser().resolve()
    args.uhd_find = args.uhd_find.expanduser().resolve()
    args.uhd_library = args.uhd_library.expanduser().resolve()
    args.nvme_root = args.nvme_root.expanduser().resolve()
    args.archive_root = args.archive_root.expanduser().resolve()
    run_id = args.run_id or f"{utc_stamp()}_{args.trace_kind}_x310_80mhz"
    nvme_run = args.nvme_root / run_id
    archive_run = args.archive_root / run_id
    if nvme_run.exists() or archive_run.exists():
        raise SystemExit(f"run already exists: {nvme_run} or {archive_run}")

    if args.dry_run:
        print(json.dumps({
            "run_id": run_id,
            "mode": args.trace_kind,
            "rf": {
                "center_frequency_hz": args.center_frequency_hz,
                "requested_sample_rate_sps": args.sample_rate_sps,
                "requested_bandwidth_hz": args.bandwidth_hz,
                "gain_db": args.gain_db,
                "antenna": args.antenna,
                "channel": args.channel,
                "usrp_args": args.usrp_args,
            },
            "sniffer": build_sniffer_command(args, nvme_run / "monitor/monitor.pcapng"),
            "sniffer_mode": "static_direct" if args.static_follow else "dynamic_extcap",
            "extcap": build_extcap_command(args, Path("/tmp/monitor.pcapng.fifo"), Path("/tmp/control.in"), Path("/tmp/control.out")),
            "x310": build_x310_command(args, nvme_run / "iq/capture.sc16"),
            "nvme_root": str(args.nvme_root),
            "archive_root": str(args.archive_root),
            "reset_boards": args.reset_boards,
            "sniffer_only": args.sniffer_only,
        }, indent=2, sort_keys=True))
        return 0

    iq_dir = nvme_run / "iq"
    monitor_dir = nvme_run / "monitor"
    ground_truth_dir = nvme_run / "ground_truth"
    for directory in (iq_dir, monitor_dir, ground_truth_dir):
        directory.mkdir(parents=True, exist_ok=False)
    iq_path = iq_dir / "capture.sc16"
    capture_log = iq_dir / "capture.log"
    pcap_path = monitor_dir / "monitor.pcapng"
    sniffer_log = monitor_dir / "sniffer.log"
    uhd_find_log = monitor_dir / "uhd_find_devices.log"
    central_uart = ground_truth_dir / "52833_uart.log"
    peripheral_uart = ground_truth_dir / "52840_uart.log"
    reset_dir = ground_truth_dir / "resets"
    reset_dir.mkdir()

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "prepared",
        "mode": args.trace_kind,
        "created_epoch_ns": time.time_ns(),
        "target_address": args.target_address,
        "rf": {
            "center_frequency_hz": args.center_frequency_hz,
            "requested_sample_rate_sps": args.sample_rate_sps,
            "requested_bandwidth_hz": args.bandwidth_hz,
            "gain_db": args.gain_db,
            "antenna": args.antenna,
            "channel": args.channel,
            "usrp_args": args.usrp_args,
        },
        "duration_s": args.duration_s,
        "paths_before_archive": {
            "iq": str(iq_path),
            "capture_log": str(capture_log),
            "monitor_pcapng": str(pcap_path),
            "sniffer_log": str(sniffer_log),
            "central_uart": str(central_uart),
            "peripheral_uart": str(peripheral_uart),
        },
        "commands": {},
    }
    if args.trace_kind == "cis":
        manifest["expected_aa_cis_hint"] = "0x8dc48a55"
        manifest["expected_aa_cis_on_air_bytes_hint"] = "55 8a c4 8d"
    if args.hold_central_until_follow:
        manifest["pre_monitor_central_halt"] = halt_board(
            args.central_jlink,
            args.central_device,
            reset_dir / "central_halt_jlink.log",
        )
        # Give the peer a short interval to observe the central disconnect and
        # return to advertising before the sniffer begins learning the target.
        time.sleep(0.8)
    write_json(nvme_run / "session_manifest.json", manifest)

    if args.dry_run:
        x310_check = {"skipped": True, "reason": "dry-run"}
    elif args.sniffer_only:
        x310_check = {"skipped": True, "reason": "sniffer-only"}
    else:
        x310_check = check_x310(args, uhd_find_log)
    manifest["x310_device_check"] = x310_check
    if not x310_check.get("found", False) and not args.dry_run and not args.sniffer_only:
        manifest["status"] = "blocked_no_x310"
        manifest["completed_epoch_ns"] = time.time_ns()
        write_json(nvme_run / "session_manifest.json", manifest)
        if not args.no_copy_delete:
            copy_then_delete(nvme_run, archive_run)
            write_json(archive_run / "session_manifest.json", {**manifest, "archive_path": str(archive_run), "nvme_deleted": True})
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 3

    stop_event = threading.Event()
    uart_state: dict[str, Any] = {}
    uart_threads = [
        threading.Thread(target=collect_uart, args=(args.central_uart, central_uart, "central", args.uart_baudrate, stop_event, uart_state), daemon=True),
        threading.Thread(target=collect_uart, args=(args.peripheral_uart, peripheral_uart, "peripheral", args.uart_baudrate, stop_event, uart_state), daemon=True),
    ]
    for thread in uart_threads:
        thread.start()

    sniffer_command = build_sniffer_command(args, pcap_path)
    x310_command = build_x310_command(args, iq_path)
    manifest["commands"] = {
        "sniffer": sniffer_command,
        "sniffer_mode": "static_direct" if args.static_follow else "dynamic_extcap",
        "extcap": build_extcap_command(args, Path("<runtime-pcap-fifo>"), Path("<runtime-control-in>"), Path("<runtime-control-out>")),
        "x310": x310_command,
    }
    manifest["status"] = "running"
    manifest["monitors_started_epoch_ns"] = time.time_ns()
    write_json(nvme_run / "session_manifest.json", manifest)

    x310_log_handle = capture_log.open("w", encoding="utf-8")
    dynamic_sniffer: DynamicExtcapSniffer | None = None
    sniffer_log_handle = None
    if args.static_follow:
        sniffer_log_handle = sniffer_log.open("w", encoding="utf-8")
        sniffer = subprocess.Popen(
            sniffer_command,
            cwd=PROJECT_ROOT,
            stdout=sniffer_log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    else:
        dynamic_sniffer = DynamicExtcapSniffer(args, pcap_path, monitor_dir)
        dynamic_sniffer.start()
        sniffer = dynamic_sniffer.proc
    time.sleep(0.5)
    x310 = None
    if not args.sniffer_only:
        x310_env = os.environ.copy()
        x310_env["LD_LIBRARY_PATH"] = f"{args.uhd_library}:{x310_env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
        x310 = subprocess.Popen(
            x310_command,
            cwd=PROJECT_ROOT,
            env=x310_env,
            stdout=x310_log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    time.sleep(max(0.5, args.setup_s + 0.2))

    if args.hold_central_until_follow:
        if dynamic_sniffer is None:
            raise RuntimeError("central hold requires dynamic sniffer")
        follow_deadline = time.monotonic() + max(10.0, args.setup_s + 8.0)
        while time.monotonic() < follow_deadline and not dynamic_sniffer.follow_ack:
            time.sleep(0.05)
        manifest["follow_gate"] = {
            "device_added": dynamic_sniffer.device_added,
            "follow_sent": dynamic_sniffer.follow_sent,
            "follow_ack": dynamic_sniffer.follow_ack,
        }
        time.sleep(args.follow_settle_s)
        manifest["central_release"] = go_board(
            args.central_jlink,
            args.central_device,
            reset_dir / "central_go_jlink.log",
        )
    elif args.reset_boards:
        manifest["resets"] = {"peripheral": reset_board(args.peripheral_jlink, args.peripheral_device, reset_dir / "peripheral_jlink.log")}
        time.sleep(0.5)
        manifest["resets"]["central"] = reset_board(args.central_jlink, args.central_device, reset_dir / "central_jlink.log")
    else:
        manifest["resets"] = {"skipped": True, "reason": "boards must be reset/reconnected externally"}
    write_json(nvme_run / "session_manifest.json", manifest)

    deadline = time.monotonic() + args.duration_s + max(5.0, args.setup_s + 2.0)
    while time.monotonic() < deadline and (x310 is None or x310.poll() is None):
        time.sleep(0.2)
    x310_rc = stop_process(x310)
    sniffer_rc = dynamic_sniffer.stop() if dynamic_sniffer is not None else stop_process(sniffer)
    x310_log_handle.close()
    if sniffer_log_handle is not None:
        sniffer_log_handle.close()
    stop_event.set()
    for thread in uart_threads:
        thread.join(timeout=2.0)

    manifest.update(
        {
            "status": "captured",
            "capture_returncode": x310_rc,
            "sniffer_returncode": sniffer_rc,
            "follow_sync": dynamic_sniffer.sync if dynamic_sniffer is not None else {"mode": "static_direct"},
            "uart": uart_state,
            "completed_epoch_ns": time.time_ns(),
            "files": {
                "iq_bytes": iq_path.stat().st_size if iq_path.exists() else 0,
                "pcap_bytes": pcap_path.stat().st_size if pcap_path.exists() else 0,
            },
        }
    )
    write_json(nvme_run / "session_manifest.json", manifest)

    if args.no_copy_delete:
        manifest["archive_path"] = None
        manifest["nvme_deleted"] = False
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0 if (args.sniffer_only or x310_rc == 0) and pcap_path.exists() else 2

    try:
        copy_then_delete(nvme_run, archive_run)
    except Exception as exc:
        manifest["status"] = "archive_failed"
        manifest["archive_error"] = repr(exc)
        write_json(nvme_run / "session_manifest.json", manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 2
    manifest["archive_path"] = str(archive_run)
    manifest["nvme_deleted"] = True
    manifest["capture_data_usable"] = archived_capture_is_usable(
        archive_run,
        sniffer_only=args.sniffer_only,
        x310_returncode=x310_rc,
    )
    write_json(archive_run / "session_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["capture_data_usable"] else 2


if __name__ == "__main__":
    sys.exit(main())
