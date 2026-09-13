#!/usr/bin/env python3
"""Capture the matched 2M PIP AUC matrix with NVMe-first retention.

Each session is independently reset, captured to the local NVMe-backed
workspace, copied directly to the PSSD, and only then permanently removed from
the NVMe.  A failed X310 capture or failed copy leaves the NVMe source in
place for inspection.  The script intentionally does not hash or verify the
PSSD copy, matching the experiment handoff; it relies on successful ``cp``
completion before deletion.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NVME_ROOT_DEFAULT = PROJECT_ROOT / "testdata"
PSSD_ROOT_DEFAULT = Path(
    "/path/to/PhantomChannel/experiments/figure/pip_boundary_auc_20260808/sessions"
)
MANIFEST = PROJECT_ROOT / "artifacts/firmware/pip_auc_20260808/manifest.json"
CENTRAL_SERIAL = "NRF52833_SERIAL"
PERIPHERAL_SERIAL = "NRF52840_SERIAL"
CENTRAL_PORT = Path("/dev/ttyACM2")
PERIPHERAL_PORT = Path("/dev/ttyACM0")
CAPTURE_RUNNER = PROJECT_ROOT / "tools/run_x310_bandwidth_experiment.py"


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=PROJECT_ROOT, text=True, check=check)


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def firmware_by_condition(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        item["condition"]: item
        for item in manifest["variants"]
        if item["condition"] in {"benign", "direct_tail", "pip"}
    }


def flash_firmware(path: Path, serial: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    run(
        [
            "nrfutil",
            "device",
            "program",
            "--serial-number",
            serial,
            "--firmware",
            str(path),
        ]
    )


def reset_devices() -> None:
    run(
        [
            "nrfutil",
            "device",
            "reset",
            "--serial-number",
            f"{CENTRAL_SERIAL},{PERIPHERAL_SERIAL}",
        ]
    )


def prepare_port(path: Path) -> None:
    run(["stty", "-F", str(path), "115200", "raw", "-echo", "-ixon"], check=True)


def start_uart(path: Path, output: Path) -> subprocess.Popen[Any]:
    prepare_port(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    handle = output.open("wb")
    process = subprocess.Popen(
        ["stdbuf", "-o0", "cat", str(path)],
        stdin=subprocess.DEVNULL,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    process._pip_auc_log_handle = handle  # type: ignore[attr-defined]
    return process


def stop_uart(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=3)
    handle = getattr(process, "_pip_auc_log_handle", None)
    if handle is not None:
        handle.close()


def check_space(nvme_root: Path, duration_s: float, reserve_gb: float) -> dict[str, int]:
    usage = shutil.disk_usage(nvme_root)
    expected = int(100_000_000 * duration_s * 4)
    reserve = int(reserve_gb * 1024**3)
    required = expected + reserve
    if usage.free < required:
        raise RuntimeError(
            f"NVMe space gate failed: free={usage.free} bytes, required={required} "
            f"(capture={expected}, reserve={reserve})"
        )
    return {"free_before": usage.free, "expected_iq_bytes": expected, "reserve_bytes": reserve}


def capture_with_runner(
    session_id: str,
    duration_s: float,
    nvme_root: Path,
    samples_per_buffer: int,
) -> Path:
    command = [
        sys.executable,
        str(CAPTURE_RUNNER),
        "--target",
        "nvme",
        "--capture-id",
        session_id,
        "--duration-s",
        str(duration_s),
        "--sample-rate-sps",
        "80000000",
        "--expected-sample-rate-sps",
        "100000000",
        "--bandwidth-hz",
        "80000000",
        "--center-frequency-hz",
        "2440000000",
        "--gain-db",
        "50",
        "--antenna",
        "RX2",
        "--usrp-args",
        "type=x300,addr=192.168.40.2,master_clock_rate=200e6",
        "--nvme-root",
        str(nvme_root),
        "--reserve-bytes",
        str(512 * 1024 * 1024),
        "--samples-per-buffer",
        str(samples_per_buffer),
    ]
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, check=False)
    run_root = nvme_root / session_id
    status_path = run_root / "run_status.json"
    if result.returncode != 0 or not status_path.is_file():
        raise RuntimeError(f"X310 runner failed for {session_id}; NVMe source retained at {run_root}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "passed":
        raise RuntimeError(f"X310 capture validation failed for {session_id}; source retained at {run_root}")
    capture_log = run_root / "iq/capture.log"
    overflow_lines = [
        line.strip()
        for line in capture_log.read_text(encoding="utf-8", errors="replace").splitlines()
        if "got an overflow indication" in line.lower()
    ]
    if overflow_lines:
        raise RuntimeError(
            f"X310 reported an actual overflow for {session_id}; source retained at {run_root}: "
            + " | ".join(overflow_lines)
        )
    return run_root


def copy_and_remove(run_root: Path, pssd_root: Path, session_id: str) -> Path:
    destination = pssd_root / session_id
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    run(["cp", "-a", str(run_root), str(destination.parent)])
    # The user explicitly requested permanent deletion after copy completion.
    shutil.rmtree(run_root)
    (destination / "copy_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "session_id": session_id,
                "source_nvme": str(run_root),
                "destination_pssd": str(destination),
                "copy_command": "cp -a",
                "copy_verified": False,
                "nvme_source_retained": False,
                "nvme_source_permanently_deleted_after_cp": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def pilot_gate(destination: Path, condition: str) -> dict[str, Any]:
    peripheral_log = destination / "ground_truth/peripheral_uart.log"
    central_log = destination / "ground_truth/central_uart.log"
    peripheral = peripheral_log.read_text(encoding="utf-8", errors="replace")
    central = central_log.read_text(encoding="utf-8", errors="replace")
    phy_ok = 'HRS_PHY_UPDATED {"tx_phy":2,"rx_phy":2}' in central
    subscribed = 'HRS_SUBSCRIBE {"status":0' in central
    notifications = central.count("HRS_NOTIFY {")
    if condition == "pip":
        tx_done = peripheral.count("PIP_LL_TX_DONE {")
        condition_ok = peripheral.count("HRS_PIP_TX {") > 0 and tx_done > 0
    else:
        tx_done = peripheral.count("HRS_TX {")
        condition_ok = tx_done > 0 and '"status":0' in peripheral
    disconnects = central.count("HRS_DISCONNECTED {")
    result = {
        "condition": condition,
        "phy_2m": phy_ok,
        "subscribed": subscribed,
        "central_notifications": notifications,
        "peripheral_tx_records": tx_done,
        "condition_signal": condition_ok,
        "disconnect_records": disconnects,
        "passed": phy_ok and subscribed and notifications > 0 and condition_ok and disconnects == 0,
    }
    (destination / "pilot_gate.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def capture_one(
    *,
    condition: str,
    session_id: str,
    duration_s: float,
    nvme_root: Path,
    pssd_root: Path,
    firmware: dict[str, Any],
    settle_s: float,
    run_gate: bool,
    samples_per_buffer: int,
) -> dict[str, Any]:
    space = check_space(nvme_root, duration_s, 1.0)
    flash_firmware(Path(firmware["firmware"]), PERIPHERAL_SERIAL)
    nvme_session = nvme_root / session_id
    if nvme_session.exists():
        raise FileExistsError(nvme_session)
    # The X310 runner creates the session directory itself, so UART capture is
    # kept in a sibling staging directory until the IQ runner has created the
    # NVMe session root.
    uart_dir = nvme_root / f".{session_id}.uart"
    if uart_dir.exists():
        raise FileExistsError(uart_dir)
    uart_dir.mkdir(parents=True)
    peripheral_uart = start_uart(PERIPHERAL_PORT, uart_dir / "peripheral_uart.log")
    central_uart = start_uart(CENTRAL_PORT, uart_dir / "central_uart.log")
    capture_error: Exception | None = None
    destination: Path | None = None
    try:
        reset_devices()
        time.sleep(settle_s)
        run_root = capture_with_runner(session_id, duration_s, nvme_root, samples_per_buffer)
    except Exception as exc:
        capture_error = exc
        run_root = nvme_session
    finally:
        stop_uart(peripheral_uart)
        stop_uart(central_uart)
        if nvme_session.exists():
            ground_truth = nvme_session / "ground_truth"
            ground_truth.mkdir(parents=True, exist_ok=True)
            for log_path in uart_dir.iterdir():
                shutil.move(str(log_path), str(ground_truth / log_path.name))
        shutil.rmtree(uart_dir, ignore_errors=True)
    if capture_error is not None:
        raise capture_error
    destination = copy_and_remove(run_root, pssd_root, session_id)
    result: dict[str, Any] = {
        "condition": condition,
        "session_id": session_id,
        "destination": str(destination),
        "nvme_source_deleted": True,
        "space_gate": space,
    }
    if run_gate:
        result["pilot_gate"] = pilot_gate(destination, condition)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--settle-s", type=float, default=8.0)
    parser.add_argument("--nvme-root", type=Path, default=NVME_ROOT_DEFAULT)
    parser.add_argument("--pssd-root", type=Path, default=PSSD_ROOT_DEFAULT)
    parser.add_argument("--condition", choices=("benign", "direct_tail", "pip"))
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--start-repetition",
        type=int,
        default=1,
        help="Formal mode repetition to start from (inclusive), for resuming after an invalid session.",
    )
    parser.add_argument("--samples-per-buffer", type=int, default=8000)
    parser.add_argument(
        "--session-prefix",
        default="",
        help="Prefix session IDs, useful for isolated write-gate pilots.",
    )
    args = parser.parse_args()
    if args.repetitions < 1 or args.repetitions > 5:
        parser.error("--repetitions must be in 1..5")
    if args.start_repetition < 1 or args.start_repetition > args.repetitions:
        parser.error("--start-repetition must be in 1..repetitions")
    if args.duration_s is None:
        args.duration_s = 8.0 if args.mode == "pilot" else 15.0
    if args.duration_s <= 0:
        parser.error("--duration-s must be positive")
    if args.samples_per_buffer < 1:
        parser.error("--samples-per-buffer must be positive")
    args.nvme_root = args.nvme_root.expanduser().resolve()
    args.pssd_root = args.pssd_root.expanduser().resolve()
    args.nvme_root.mkdir(parents=True, exist_ok=True)
    args.pssd_root.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    variants = firmware_by_condition(manifest)
    if args.condition:
        conditions = [args.condition]
    else:
        conditions = ["benign", "direct_tail", "pip"]
    flash_firmware(Path(next(item for item in manifest["variants"] if item["condition"] == "central")["firmware"]), CENTRAL_SERIAL)
    results = []
    if args.mode == "pilot":
        for condition in conditions:
            session_id = f"{args.session_prefix}pip_auc_pilot_{condition}"
            results.append(
                capture_one(
                    condition=condition,
                    session_id=session_id,
                    duration_s=args.duration_s,
                    nvme_root=args.nvme_root,
                    pssd_root=args.pssd_root,
                    firmware=variants[condition],
                    settle_s=args.settle_s,
                    run_gate=True,
                    samples_per_buffer=args.samples_per_buffer,
                )
            )
    else:
        for condition in conditions:
            for repetition in range(args.start_repetition, args.repetitions + 1):
                session_id = f"{args.session_prefix}pip_auc_{condition}_rep{repetition}"
                results.append(
                    capture_one(
                        condition=condition,
                        session_id=session_id,
                        duration_s=args.duration_s,
                        nvme_root=args.nvme_root,
                        pssd_root=args.pssd_root,
                        firmware=variants[condition],
                        settle_s=args.settle_s,
                        run_gate=False,
                        samples_per_buffer=args.samples_per_buffer,
                    )
                )
    print(json.dumps({"mode": args.mode, "results": results}, indent=2, ensure_ascii=False))
    if args.mode == "pilot" and not all(item["pilot_gate"]["passed"] for item in results):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
