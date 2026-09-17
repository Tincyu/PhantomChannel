#!/usr/bin/env python3
"""Capture X310 IQ only after the nRF52840 link and notifications are ready.

This runner is deliberately local to PhantomChannel.  It reuses the existing
X310 capture/parser helpers and the existing 52840 RTT/BlueZ conventions, but
does not modify or execute code from PhantomChannel receiver except as the parser
entrypoint during the optional replay stage.

The hardware order is:

    RTT logger -> BLE scan/connect -> GATT notify on -> X310 IQ -> disconnect

The IQ window is therefore aligned with a live 52840 notification window, and
the RTT log and UHD log are retained beside the IQ file for later scoring.
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
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "x310_range_bandwidth_experiment.yaml"
DEFAULT_PSSD_ROOT = Path("/path/to/PhantomChannel/testdata")

sys.path.insert(0, str(PROJECT_ROOT))
from tools import run_range_bandwidth_experiment as range_runner  # noqa: E402
from tools import run_x310_bandwidth_experiment as x310_runner  # noqa: E402


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_x310_args(cli: argparse.Namespace, run_id: str) -> argparse.Namespace:
    values = [
        "--target", "pssd",
        "--capture-id", run_id,
        "--duration-s", str(cli.duration_s),
        "--sample-rate-sps", "80000000",
        "--expected-sample-rate-sps", "100000000",
        "--bandwidth-hz", "80000000",
        "--center-frequency-hz", "2440000000",
        "--gain-db", "50",
        "--antenna", "RX2",
        "--channel", "0",
        "--usrp-args", "type=x300,addr=192.168.40.2,master_clock_rate=200e6",
        "--pssd-root", str(cli.pssd_root),
        "--with-sdr-parse",
        "--two-stage-known-aa",
        "--frontend-backend", "cuda",
        "--stage2-known-aa-bit-tolerance", "1",
        "--stage1-ble-score-threshold", "3",
        "--stage2-ble-score-threshold", "6",
    ]
    args = x310_runner.build_parser().parse_args(values)
    args.capture_bin = args.capture_bin.expanduser().resolve()
    args.uhd_library = args.uhd_library.expanduser().resolve()
    args.ble_root = args.ble_root.expanduser().resolve()
    args.parser_python = args.parser_python.expanduser()
    args.parser_entrypoint = args.parser_entrypoint.expanduser().resolve()
    args.pssd_root = args.pssd_root.expanduser().resolve()
    return args


def capture_ok(metadata: dict[str, Any], iq_path: Path, returncode: int, min_fraction: float) -> bool:
    fraction = metadata.get("sample_count_fraction_of_expected")
    return bool(
        returncode == 0
        and iq_path.is_file()
        and iq_path.stat().st_size > 0
        and metadata.get("actual_sample_rate_sps")
        and metadata.get("actual_bandwidth_matches_requested")
        and metadata.get("file_size_matches_received_samples")
        and fraction is not None
        and fraction >= min_fraction
    )


def start_x310_capture(
    xargs: argparse.Namespace,
    run_root: Path,
    status: dict[str, Any],
) -> tuple[subprocess.Popen[bytes], Path, Path, list[str], int]:
    iq_dir = run_root / "iq"
    iq_dir.mkdir(parents=True, exist_ok=True)
    iq_path = iq_dir / "capture.sc16"
    log_path = iq_dir / "capture.log"
    command = x310_runner.build_capture_command(xargs, iq_path)
    env = x310_runner.capture_environment(xargs)
    with log_path.open("wb") as log:
        log.write(("$ " + shlex.join(command) + "\n").encode("utf-8"))
        log.flush()
        proc = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    start_ns = time.time_ns()
    status["capture_start_epoch_ns"] = start_ns
    status["paths"].update({
        "iq": str(iq_path),
        "capture_log": str(log_path),
    })
    write_json(run_root / "run_status.json", status)
    return proc, iq_path, log_path, command, start_ns


def stop_process(proc: subprocess.Popen[bytes], timeout_s: float = 20.0) -> int:
    if proc.poll() is None:
        try:
            return proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                return proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                return proc.wait(timeout=10)
    return int(proc.returncode or 0)


def run_capture_and_connection(
    config: dict[str, Any],
    xargs: argparse.Namespace,
    run_id: str,
    run_root: Path,
    status: dict[str, Any],
) -> dict[str, Any]:
    firmware = config["firmware"]
    peripheral = config["peripheral"]
    central = config.get("central", {})
    rtt_cfg = config.get("rtt", {})
    ground_truth = run_root / "ground_truth"
    ground_truth.mkdir(parents=True, exist_ok=True)
    rtt_log = ground_truth / "peripheral_rtt.log"
    rtt_stdout = ground_truth / "jlink_rtt_logger.stdout.log"
    bt_log = ground_truth / "bluetoothctl.log"
    elf = Path(firmware["build_dir"]) / "phantomchannel_peripheral" / "zephyr" / "zephyr.elf"
    if not elf.is_file():
        raise FileNotFoundError(f"52840 RTT ELF not found: {elf}")
    rtt_addr = range_runner.rtt_address(elf, Path(firmware["ncs_toolchain"]))
    serial = str(firmware["serial_number"])
    address = str(peripheral["address"])
    uuid_candidates = [str(peripheral["data_characteristic_uuid"])]
    fallbacks = peripheral.get("data_characteristic_uuid_fallbacks", [])
    if isinstance(fallbacks, str):
        fallbacks = [fallbacks]
    uuid_candidates.extend(str(item) for item in fallbacks if str(item) not in uuid_candidates)
    notify_s = float(xargs.duration_s)
    scan_s = float(peripheral.get("scan_s", 7))
    connect_wait_s = float(peripheral.get("connect_wait_s", 8))
    notify_timeout_s = float(peripheral.get("notify_start_timeout_s", 20))
    post_disconnect_s = float(rtt_cfg.get("post_disconnect_s", 4))

    rtt_command = [
        "JLinkRTTLogger",
        "-Device", str(rtt_cfg.get("jlink_device", "NRF52840_XXAA")),
        "-If", str(rtt_cfg.get("interface", "SWD")),
        "-Speed", str(rtt_cfg.get("speed", 4000)),
        "-USB", serial,
        "-RTTAddress", rtt_addr,
        "-RTTChannel", str(rtt_cfg.get("channel", 0)),
        str(rtt_log),
    ]
    rtt_proc: subprocess.Popen[bytes] | None = None
    bt_proc: subprocess.Popen[bytes] | None = None
    capture_proc: subprocess.Popen[bytes] | None = None
    iq_path: Path | None = None
    capture_log: Path | None = None
    capture_command: list[str] = []
    capture_start_ns: int | None = None
    capture_end_ns: int | None = None
    bt_returncode = -1
    rtt_returncode = -1
    notify_started_ns: int | None = None
    iq_returncode = -1
    connected = False
    notify_enabled = False

    with rtt_stdout.open("w", encoding="utf-8", errors="replace") as rtt_out:
        rtt_proc = subprocess.Popen(
            rtt_command,
            stdin=subprocess.PIPE,
            stdout=rtt_out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        time.sleep(2)
        try:
            with bt_log.open("wb") as bt_out:
                bt_proc = subprocess.Popen(
                    ["stdbuf", "-oL", "-eL", "bluetoothctl"],
                    stdin=subprocess.PIPE,
                    stdout=bt_out,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                range_runner.send_bluetoothctl_command(bt_proc, "scan on", scan_s)
                range_runner.send_bluetoothctl_command(bt_proc, "scan off", 2)
                range_runner.send_bluetoothctl_command(bt_proc, f"connect {address}", connect_wait_s)
                bt_text = bt_log.read_text(encoding="utf-8", errors="replace")
                connected = "Connection successful" in bt_text
                if not connected:
                    raise RuntimeError(f"52840 connection was not confirmed; see {bt_log}")
                range_runner.send_bluetoothctl_command(bt_proc, f"info {address}", 1)
                range_runner.send_bluetoothctl_command(bt_proc, "menu gatt", 1)
                range_runner.send_bluetoothctl_command(bt_proc, f"list-attributes {address}", 2)
                for index, characteristic_uuid in enumerate(uuid_candidates):
                    range_runner.send_bluetoothctl_command(bt_proc, f"select-attribute {characteristic_uuid}", 1)
                    range_runner.send_bluetoothctl_command(bt_proc, "notify on", 0.1)
                    probe_timeout = min(notify_timeout_s, 3.0) if index + 1 < len(uuid_candidates) else notify_timeout_s
                    try:
                        range_runner.wait_for_log_text(bt_log, bt_proc, "Notify started", probe_timeout)
                        notify_enabled = True
                        break
                    except TimeoutError:
                        if index + 1 == len(uuid_candidates):
                            raise
                if not notify_enabled:
                    raise RuntimeError(f"52840 notifications were not enabled; see {bt_log}")
                notify_started_ns = time.time_ns()
                status["connection"] = {
                    "connected": connected,
                    "notify_enabled": notify_enabled,
                    "notify_started_epoch_ns": notify_started_ns,
                    "address": address,
                    "characteristic_candidates": uuid_candidates,
                }
                status["stage"] = "x310_capture"
                write_json(run_root / "run_status.json", status)

                capture_proc, iq_path, capture_log, capture_command, capture_start_ns = start_x310_capture(
                    xargs, run_root, status
                )
                # rx_samples_to_file owns the ten-second window.  Keeping the
                # BLE link open until it exits keeps RTT truth and IQ aligned.
                iq_returncode = stop_process(capture_proc, timeout_s=max(40.0, xargs.duration_s + 30.0))
                capture_end_ns = time.time_ns()
                status["capture_end_epoch_ns"] = capture_end_ns
                status["capture_returncode"] = iq_returncode
                write_json(run_root / "run_status.json", status)

                range_runner.send_bluetoothctl_command(bt_proc, "notify off", 1)
                range_runner.send_bluetoothctl_command(bt_proc, "back", 1)
                range_runner.send_bluetoothctl_command(bt_proc, f"disconnect {address}", 1)
                range_runner.send_bluetoothctl_command(bt_proc, "quit", 0)
                if bt_proc.stdin:
                    bt_proc.stdin.close()
                bt_returncode = bt_proc.wait(timeout=15)
            time.sleep(post_disconnect_s)
        finally:
            if capture_proc is not None and capture_proc.poll() is None:
                iq_returncode = stop_process(capture_proc)
                capture_end_ns = time.time_ns()
            if bt_proc is not None and bt_proc.poll() is None:
                if bt_proc.stdin:
                    bt_proc.stdin.close()
                bt_proc.terminate()
                try:
                    bt_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    bt_proc.kill()
                    bt_proc.wait(timeout=5)
            if rtt_proc.stdin:
                rtt_proc.stdin.write(b"\n")
                rtt_proc.stdin.flush()
            rtt_returncode = rtt_proc.wait(timeout=15)

    if iq_path is None or capture_log is None:
        raise RuntimeError("X310 capture did not start")
    log_text = capture_log.read_text(encoding="utf-8", errors="replace") if capture_log.is_file() else ""
    log_info = x310_runner.parse_capture_log(log_text, xargs)
    target = x310_runner.Target("pssd", xargs.pssd_root)
    metadata = x310_runner.build_capture_metadata(
        xargs,
        target,
        run_id,
        run_root,
        iq_path,
        capture_log,
        capture_command,
        iq_returncode,
        log_info,
    )
    metadata.update({
        "start_epoch_ns": capture_start_ns,
        "end_epoch_ns": capture_end_ns,
        "52840_notify_started_epoch_ns": notify_started_ns,
        "52840_connected_before_iq": connected and notify_enabled and bool(
            notify_started_ns and capture_start_ns and capture_start_ns >= notify_started_ns
        ),
        "rtt_log_path": str(rtt_log),
        "bluetoothctl_log_path": str(bt_log),
        "jlink_stdout_log_path": str(rtt_stdout),
        "rtt_returncode": rtt_returncode,
        "bluetoothctl_returncode": bt_returncode,
    })
    metadata_path = run_root / "iq" / "metadata.json"
    write_json(metadata_path, metadata)
    status["capture"] = metadata
    status["paths"].update({
        "metadata": str(metadata_path),
        "rtt": str(rtt_log),
        "bluetoothctl": str(bt_log),
    })
    write_json(run_root / "run_status.json", status)

    rtt_parse_rc, rtt_parse_status = range_runner.parse_rtt(
        run_id, rtt_log, ground_truth
    )
    smoke = range_runner.summarize_smoke(
        rtt_log,
        bt_log,
        rtt_parse_status,
        notify_s=notify_s,
        notify_interval_ms=float(peripheral.get("notify_interval_ms", 20)),
        min_notify_fraction=float(peripheral.get("min_notify_fraction", 0.5)),
    )
    smoke["iq_started_after_notify"] = bool(
        notify_started_ns and capture_start_ns and capture_start_ns >= notify_started_ns
    )
    status["ground_truth"] = {"rtt_parse_returncode": rtt_parse_rc, "rtt_parse_status": rtt_parse_status}
    status["smoke"] = smoke
    status["stage"] = "sdr_parse"
    write_json(run_root / "run_status.json", status)
    return {
        "metadata": metadata,
        "metadata_path": metadata_path,
        "rtt_log": rtt_log,
        "bt_log": bt_log,
        "smoke": smoke,
        "rtt_parse_status": rtt_parse_status,
        "capture_valid": capture_ok(metadata, iq_path, iq_returncode, xargs.min_sample_fraction),
    }


def run_score(run_root: Path, run_id: str) -> dict[str, Any]:
    parser_output = run_root / "sdr" / "two_stage_known_aa" / "ble_packets.csv"
    results = run_root / "results"
    results.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "match_rtt_sdr_results.py"),
        "--run-root", str(run_root),
        "--run-id", run_id,
        "--distance-m", "0.5",
        "--analysis-bandwidth-hz", "80000000",
        "--sdr-ble", str(parser_output),
        "--output-dir", str(results),
    ]
    log_path = results / "rtt_sdr_scorer.log"
    start_ns = time.time_ns()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + shlex.join(command) + "\n")
        log.flush()
        proc = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    end_ns = time.time_ns()
    metrics_path = results / "recovery_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    return {
        "returncode": proc.returncode,
        "valid": proc.returncode == 0 and bool(metrics.get("valid")),
        "metrics_path": str(metrics_path),
        "metrics": metrics,
        "log_path": str(log_path),
        "start_epoch_ns": start_ns,
        "end_epoch_ns": end_ns,
        "duration_seconds": (end_ns - start_ns) / 1_000_000_000.0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--capture-id", default="")
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--pssd-root", type=Path, default=DEFAULT_PSSD_ROOT)
    parser.add_argument("--no-sdr-parse", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    cli = build_parser().parse_args(argv)
    if cli.duration_s <= 0:
        raise SystemExit("--duration-s must be positive")
    config = range_runner.load_config(cli.config.expanduser().resolve())
    run_id = cli.capture_id or f"{time.strftime('%Y%m%d_%H%M%S')}_x310_80m_52840_connected"
    xargs = load_x310_args(cli, run_id)
    run_root = cli.pssd_root.expanduser().resolve() / run_id
    if run_root.exists():
        raise SystemExit(f"run directory already exists: {run_root}")
    run_root.mkdir(parents=True)
    status: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "target": "pssd",
        "status": "running",
        "stage": "52840_connect",
        "created_epoch_ns": time.time_ns(),
        "parameters": {
            "duration_s": cli.duration_s,
            "sample_rate_sps": 80000000,
            "expected_actual_sample_rate_sps": 100000000,
            "bandwidth_hz": 80000000,
            "center_frequency_hz": 2440000000,
            "gain_db": 50,
            "antenna": "RX2",
            "channel": 0,
            "usrp_args": "type=x300,addr=192.168.40.2,master_clock_rate=200e6",
            "parser_backend": "CUDA frontend + stage1 C++ + stage2 Python",
            "ble_project_untouched": True,
        },
        "paths": {"run_root": str(run_root)},
    }
    write_json(run_root / "run_status.json", status)
    try:
        connection = run_capture_and_connection(config, xargs, run_id, run_root, status)
        status["capture_valid"] = connection["capture_valid"]
        if not connection["capture_valid"]:
            status.update({"status": "failed", "stage": "capture_validation", "completed_epoch_ns": time.time_ns()})
            write_json(run_root / "run_status.json", status)
            print(json.dumps(status, indent=2, sort_keys=True))
            return 2
        if not cli.no_sdr_parse:
            status["stage"] = "sdr_parse"
            write_json(run_root / "run_status.json", status)
            parser_result = x310_runner.run_parser(
                xargs,
                run_root,
                connection["metadata_path"],
                Path(connection["metadata"]["iq_path"]),
                float(connection["metadata"]["actual_sample_rate_sps"]),
            )
            status["sdr_parse"] = parser_result
            write_json(run_root / "run_status.json", status)
            if not parser_result.get("valid"):
                raise RuntimeError("two-stage parser failed; see sdr/two_stage_known_aa_runner.log")
            status["stage"] = "rtt_sdr_score"
            write_json(run_root / "run_status.json", status)
            status["rtt_sdr_score"] = run_score(run_root, run_id)
        status.update({
            "status": "passed",
            "stage": "complete",
            "completed_epoch_ns": time.time_ns(),
        })
    except Exception as exc:
        status.update({
            "status": "failed",
            "reason": str(exc),
            "completed_epoch_ns": time.time_ns(),
        })
        write_json(run_root / "run_status.json", status)
        print(json.dumps(status, indent=2, sort_keys=True))
        return 1
    write_json(run_root / "run_status.json", status)
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
