#!/usr/bin/env python3
"""Run a PhantomChannel range/bandwidth experiment scaffold.

The first supported hardware path is a ground-truth smoke run:
build/flash the nRF52840DK firmware if requested, optionally capture B210 IQ,
capture RTT, connect from BlueZ, enable CCC notifications, parse RTT, and write
run_status.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only on minimal hosts
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
from portable_paths import BLE_ROOT, CAPTURE_BIN, PARSER_PYTHON, project_path  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "range_bandwidth_experiment.yaml"
PHANTOM_LINE_RE = re.compile(r"^PHANTOM_(?P<kind>[A-Z_]+)")
PHANTOM_TX_JSON_RE = re.compile(r"^PHANTOM_TX\s+(?P<payload>\{.*\})$")
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
FAULT_RE = re.compile(r"ASSERTION FAIL|FATAL|<err> os")


@dataclass
class CommandResult:
    command: list[str]
    returncode: int
    log_path: str


@dataclass
class B210Capture:
    process: subprocess.Popen[bytes]
    command: list[str]
    env: dict[str, str]
    iq_dir: Path
    iq_path: Path
    tags_path: Path
    log_path: Path
    metadata_path: Path
    requested_duration_s: float
    start_epoch_ns: int
    process_duration_s: float = 0.0
    notification_window_s: float | None = None
    ready_epoch_ns: int | None = None
    notify_started_epoch_ns: int | None = None
    stop_reason: str | None = None
    end_epoch_ns: int | None = None
    returncode: int | None = None


def load_config(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read the experiment config")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return data


def utcish_run_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def slug(text: Any) -> str:
    value = str(text).strip().lower()
    value = re.sub(r"[^a-z0-9_.-]+", "_", value)
    return value.strip("_") or "na"


def build_run_id(args: argparse.Namespace) -> str:
    if args.capture_id:
        return args.capture_id
    return (
        f"{utcish_run_stamp()}_"
        f"{slug(args.environment)}_"
        f"d{slug(args.distance_m)}m_"
        f"bw{slug(args.analysis_bandwidth_mhz)}mhz_"
        f"rep{slug(args.repetition)}"
    )


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_number(text: str) -> int | float | str:
    value = text.strip()
    if not value:
        return ""
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return int(number)
    return number


def log_value(log_text: str, label: str) -> str:
    pattern = re.compile(rf"^{re.escape(label)}:\s*(?P<value>.+)$", re.MULTILINE)
    matches = list(pattern.finditer(log_text))
    return matches[-1].group("value").strip() if matches else ""


def done_values(log_text: str) -> dict[str, int]:
    pattern = re.compile(
        r"Done\. samples=(?P<samples>\d+), blocks=(?P<blocks>\d+), "
        r"overflows=(?P<overflows>\d+), gaps=(?P<gaps>\d+), udp_drops=(?P<udp_drops>\d+)"
    )
    matches = list(pattern.finditer(log_text))
    if not matches:
        return {"samples": -1, "blocks": -1, "overflows": -1, "gaps": -1, "udp_drops": -1}
    return {key: int(value) for key, value in matches[-1].groupdict().items()}


def ncs_env(toolchain: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["NCS_TOOLCHAIN"] = str(toolchain)
    env["PATH"] = (
        f"/usr/bin:/bin:{toolchain}/usr/local/bin:{toolchain}/usr/bin:"
        f"{toolchain}/bin:{toolchain}/opt/bin"
    )
    env["PYTHONHOME"] = f"{toolchain}/usr/local"
    env["PYTHONPATH"] = (
        f"{toolchain}/usr/local/lib/python3.12:"
        f"{toolchain}/usr/local/lib/python3.12/site-packages"
    )
    env["ZEPHYR_TOOLCHAIN_VARIANT"] = "zephyr"
    env["ZEPHYR_SDK_INSTALL_DIR"] = f"{toolchain}/opt/zephyr-sdk"
    env.pop("GIT_EXEC_PATH", None)
    env.pop("GIT_TEMPLATE_DIR", None)
    return env


def run_logged(command: list[str], log_path: Path, cwd: Path, env: dict[str, str] | None = None) -> CommandResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        proc = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    return CommandResult(command=command, returncode=proc.returncode, log_path=str(log_path))


def channel_map_bytes(map_hex: str) -> list[str]:
    text = str(map_hex or "").strip().lower().removeprefix("0x")
    text = text.replace(" ", "").replace(":", "").replace(",", "")
    if len(text) != 10:
        raise ValueError(f"BLE data channel map must be exactly 5 bytes: {map_hex}")
    int(text, 16)
    return [text[index:index + 2] for index in range(0, len(text), 2)]


def set_host_channel_classification(
    central_cfg: dict[str, Any],
    logs_dir: Path,
    map_hex: str,
    name: str,
) -> CommandResult:
    hci_device = str(central_cfg.get("hci_device", "hci0"))
    command = [
        str(central_cfg.get("hcitool", "hcitool")),
        "-i",
        hci_device,
        "cmd",
        "0x08",
        "0x0014",
        *channel_map_bytes(map_hex),
    ]
    return run_logged(command, logs_dir / f"{name}.log", PROJECT_ROOT)


def firmware_paths(firmware_cfg: dict[str, Any]) -> tuple[Path, Path, Path]:
    build_dir = project_path(firmware_cfg["build_dir"])
    merged_hex = build_dir / "merged.hex"
    elf = build_dir / "phantomchannel_peripheral" / "zephyr" / "zephyr.elf"
    return build_dir, merged_hex, elf


def build_firmware(config: dict[str, Any], logs_dir: Path) -> CommandResult:
    firmware = config["firmware"]
    workspace = project_path(firmware["zephyr_workspace"])
    sample = project_path(firmware["sample_path"])
    build_dir = project_path(firmware["build_dir"])
    toolchain = project_path(firmware["ncs_toolchain"])
    command = [
        "west",
        "build",
        "-p",
        "always",
        "-b",
        str(firmware.get("board", "nrf52840dk/nrf52840")),
        str(sample),
        "-d",
        str(build_dir),
    ]
    return run_logged(command, logs_dir / "firmware_build.log", workspace, ncs_env(toolchain))


def flash_firmware(config: dict[str, Any], merged_hex: Path, logs_dir: Path) -> CommandResult:
    firmware = config["firmware"]
    command = [
        "nrfutil",
        "device",
        "program",
        "--firmware",
        str(merged_hex),
        "--serial-number",
        str(firmware["serial_number"]),
        "--options",
        "chip_erase_mode=ERASE_RANGES_TOUCHED_BY_FIRMWARE,verify=VERIFY_READ,reset=RESET_SYSTEM",
    ]
    return run_logged(command, logs_dir / "firmware_flash.log", PROJECT_ROOT)


def reset_firmware(config: dict[str, Any], logs_dir: Path) -> CommandResult:
    firmware = config["firmware"]
    command = [
        "nrfutil",
        "device",
        "reset",
        "--serial-number",
        str(firmware["serial_number"]),
        "--reset-kind",
        "RESET_SYSTEM",
    ]
    return run_logged(command, logs_dir / "firmware_reset.log", PROJECT_ROOT)


def effective_iq_duration_s(config: dict[str, Any]) -> float:
    capture = config.get("capture", {})
    if capture.get("stop_after_notification", False):
        peripheral = config.get("peripheral", {})
        scan_s = float(peripheral.get("scan_s", 7))
        connect_wait_s = float(peripheral.get("connect_wait_s", 8))
        notify_s = float(
            capture.get(
                "iq_duration_s",
                peripheral.get("notify_s", config.get("experiment", {}).get("duration_s", 10)),
            )
        )
        notify_start_timeout_s = float(peripheral.get("notify_start_timeout_s", 20))
        stop_guard_s = float(capture.get("stop_guard_s", 5))
        # This is only a disk-space/exception safety estimate. The actual
        # capture is stopped explicitly after the notification window.
        pre_notification_s = 0.0 if capture.get("start_after_notification", False) else (
            scan_s + 2 + connect_wait_s + 1 + 1 + 2 + 1
        )
        return pre_notification_s + max(notify_s, notify_start_timeout_s) + stop_guard_s
    if capture.get("iq_duration_s") not in (None, ""):
        return float(capture["iq_duration_s"])
    peripheral = config.get("peripheral", {})
    return (
        float(peripheral.get("scan_s", 7))
        + 2
        + float(peripheral.get("connect_wait_s", 8))
        + 1
        + 1
        + 2
        + float(peripheral.get("notify_s", 10))
        + 4
    )


def ensure_capture_space(config: dict[str, Any], run_root: Path, duration_s: float) -> None:
    capture = config.get("capture", {})
    sample_rate = int(float(capture.get("sample_rate_sps", 40000000)))
    expected_bytes = int(sample_rate * duration_s * 4)
    free_bytes = shutil.disk_usage(run_root.parent).free
    reserve_bytes = int(capture.get("reserve_bytes", 512 * 1024 * 1024))
    if free_bytes < expected_bytes + reserve_bytes:
        raise RuntimeError(
            f"insufficient free space for B210 IQ: need {expected_bytes + reserve_bytes} bytes, "
            f"have {free_bytes}"
        )


def start_b210_capture(config: dict[str, Any], run_id: str, run_root: Path) -> B210Capture:
    capture = config.get("capture", {})
    duration_s = effective_iq_duration_s(config)
    stop_after_notification = bool(capture.get("stop_after_notification", False))
    process_duration_s = 0.0 if stop_after_notification else duration_s
    notification_window_s = None
    if stop_after_notification:
        peripheral = config.get("peripheral", {})
        notification_window_s = float(
            capture.get(
                "iq_duration_s",
                peripheral.get("notify_s", config.get("experiment", {}).get("duration_s", 10)),
            )
        )
    ensure_capture_space(config, run_root, duration_s)

    capture_bin = project_path(capture.get("capture_bin", CAPTURE_BIN))
    if not capture_bin.is_file():
        raise FileNotFoundError(f"B210 capture binary not found: {capture_bin}")
    if not os.access(capture_bin, os.X_OK):
        raise PermissionError(f"B210 capture binary is not executable: {capture_bin}")

    iq_dir = run_root / "iq"
    iq_dir.mkdir(parents=True, exist_ok=True)
    iq_path = iq_dir / "capture.sc16"
    tags_path = iq_dir / "rx_time_tags_40m.csv"
    log_path = iq_dir / "capture.log"
    metadata_path = iq_dir / "metadata.json"
    if iq_path.exists() or tags_path.exists() or metadata_path.exists():
        raise FileExistsError(f"IQ outputs already exist under {iq_dir}")

    command = [
        str(capture_bin),
        "--args",
        str(capture.get("usrp_args", "type=b200,master_clock_rate=40e6")),
        "--rate",
        str(capture.get("sample_rate_sps", 40000000)),
        "--freq",
        str(capture.get("center_frequency_hz", 2420000000)),
        "--gain",
        str(capture.get("gain_db", 50)),
        "--ant",
        str(capture.get("antenna", "RX2")),
        "--duration",
        str(process_duration_s),
        "--cpu-format",
        "sc16",
        "--otw-format",
        "sc16",
        "--udp-mode",
        "none",
        "--iq-file",
        str(iq_path),
        "--meta-file",
        str(tags_path),
        "--save-local-iq",
        "--save-metadata",
        "--send-local-time",
        "--queue-blocks",
        str(capture.get("queue_blocks", 1024)),
        "--stats-interval",
        str(capture.get("stats_interval_s", 1)),
    ]
    if capture.get("capture_cpus"):
        command = ["taskset", "-c", str(capture["capture_cpus"])] + command
    if capture.get("use_realtime", False):
        command = ["chrt", "-f", str(capture.get("realtime_priority", 20))] + command

    env = os.environ.copy()
    uhd_library_path = str(capture.get("uhd_library_path", ""))
    if uhd_library_path:
        env["LD_LIBRARY_PATH"] = f"{uhd_library_path}:{env.get('LD_LIBRARY_PATH', '')}"

    with log_path.open("wb") as log:
        log.write(("$ " + " ".join(command) + "\n").encode("utf-8"))
        log.flush()
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)

    return B210Capture(
        process=process,
        command=command,
        env={"LD_LIBRARY_PATH": env.get("LD_LIBRARY_PATH", "")},
        iq_dir=iq_dir,
        iq_path=iq_path,
        tags_path=tags_path,
        log_path=log_path,
        metadata_path=metadata_path,
        requested_duration_s=duration_s,
        process_duration_s=process_duration_s,
        start_epoch_ns=time.time_ns(),
        notification_window_s=notification_window_s,
    )


def wait_for_nonempty_file(path: Path, process: subprocess.Popen[bytes], timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.is_file() and path.stat().st_size > 0:
            return
        if process.poll() is not None:
            raise RuntimeError(f"process exited before {path} became non-empty")
        time.sleep(0.1)
    raise TimeoutError(f"timed out waiting for {path} to become non-empty")


def wait_for_log_text(
    path: Path,
    process: subprocess.Popen[bytes],
    marker: str,
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.is_file() and marker in path.read_text(encoding="utf-8", errors="replace"):
            return
        if process.poll() is not None:
            raise RuntimeError(f"process exited before bluetoothctl logged {marker!r}")
        time.sleep(0.05)
    raise TimeoutError(f"timed out waiting for bluetoothctl log marker: {marker}")


def wait_b210_capture(capture: B210Capture) -> None:
    try:
        capture.returncode = capture.process.wait(timeout=max(30.0, capture.requested_duration_s + 20.0))
    except subprocess.TimeoutExpired:
        capture.process.terminate()
        try:
            capture.returncode = capture.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            capture.process.kill()
            capture.returncode = capture.process.wait(timeout=10)
    capture.end_epoch_ns = time.time_ns()


def stop_b210_capture(capture: B210Capture, reason: str) -> None:
    """Stop the framed UHD capture and let it flush its queued IQ blocks."""
    capture.stop_reason = reason
    if capture.process.poll() is None:
        capture.process.terminate()
        try:
            capture.returncode = capture.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            capture.process.kill()
            capture.returncode = capture.process.wait(timeout=10)
    elif capture.returncode is None:
        capture.returncode = capture.process.wait()
    capture.end_epoch_ns = time.time_ns()


def finalize_b210_capture(
    capture: B210Capture,
    run_id: str,
    rtt_log: Path,
    compute_sha256: bool = True,
) -> dict[str, Any]:
    if capture.returncode is None:
        wait_b210_capture(capture)

    log_text = capture.log_path.read_text(encoding="utf-8", errors="replace") if capture.log_path.is_file() else ""
    counts = done_values(log_text)
    iq_size = capture.iq_path.stat().st_size if capture.iq_path.is_file() else 0
    tags_size = capture.tags_path.stat().st_size if capture.tags_path.is_file() else 0
    expected_file_bytes = counts["samples"] * 4 if counts["samples"] >= 0 else -1
    valid = (
        capture.returncode == 0
        and capture.iq_path.is_file()
        and capture.tags_path.is_file()
        and counts["overflows"] == 0
        and counts["gaps"] == 0
        and (expected_file_bytes < 0 or expected_file_bytes == iq_size)
    )
    metadata = {
        "schema_version": 1,
        "capture_id": run_id,
        "source": "usrp_b210",
        "sample_format": "sc16_le_interleaved_iq",
        "bytes_per_complex_sample": 4,
        "command": capture.command,
        "env": capture.env,
        "requested_duration_seconds": capture.requested_duration_s,
        "capture_command_duration_seconds": capture.process_duration_s,
        "notification_window_seconds": capture.notification_window_s,
        "start_epoch_ns": capture.start_epoch_ns,
        "ready_epoch_ns": capture.ready_epoch_ns,
        "notify_started_epoch_ns": capture.notify_started_epoch_ns,
        "stop_reason": capture.stop_reason,
        "end_epoch_ns": capture.end_epoch_ns,
        "returncode": capture.returncode,
        "actual_center_frequency_hz": parse_number(log_value(log_text, "Actual RX freq")),
        "actual_sample_rate_sps": parse_number(log_value(log_text, "Actual RX rate")),
        "actual_gain_db": parse_number(log_value(log_text, "Actual RX gain")),
        "actual_rx_bandwidth_hz": parse_number(log_value(log_text, "Actual RX bandwidth")),
        "actual_antenna": log_value(log_text, "Actual RX antenna"),
        "b210_serial": log_value(log_text, "Actual B210 serial"),
        "uhd_version": log_value(log_text, "UHD version"),
        "samples": counts["samples"],
        "blocks": counts["blocks"],
        "overflows": counts["overflows"],
        "gaps": counts["gaps"],
        "udp_drops": counts["udp_drops"],
        "file_size_bytes": iq_size,
        "expected_file_size_bytes": expected_file_bytes,
        "rx_time_tags_size_bytes": tags_size,
        "iq_path": str(capture.iq_path),
        "rx_time_tags_path": str(capture.tags_path),
        "capture_log_path": str(capture.log_path),
        "rtt_log_path": str(rtt_log),
        "valid_no_overflow_or_gap": valid,
        "sha256_enabled": compute_sha256,
    }
    if compute_sha256:
        metadata["iq_sha256"] = sha256_file(capture.iq_path) if capture.iq_path.is_file() else ""
        metadata["rx_time_tags_sha256"] = sha256_file(capture.tags_path) if capture.tags_path.is_file() else ""
        metadata["rtt_log_sha256"] = sha256_file(rtt_log) if rtt_log.is_file() else ""
    write_json(capture.metadata_path, metadata)
    return metadata


def csv_data_row_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def parser_python(parser_cfg: dict[str, Any], source_root: Path) -> Path:
    configured = parser_cfg.get("python", "")
    if configured:
        return project_path(configured)
    return PARSER_PYTHON


def bool_cli_flag(command: list[str], enabled: bool, true_flag: str, false_flag: str = "") -> None:
    if enabled:
        command.append(true_flag)
    elif false_flag:
        command.append(false_flag)


def resolve_parser_bandwidth_hz(
    parser_cfg: dict[str, Any],
    sample_rate_sps: Any,
    analysis_bandwidth_hz: float | None = None,
) -> float:
    configured = float(parser_cfg.get("processing_bandwidth_hz", 40000000))
    policy = str(parser_cfg.get("processing_bandwidth_policy", "fixed")).strip().lower()
    if policy == "analysis" and analysis_bandwidth_hz is not None:
        bandwidth = float(analysis_bandwidth_hz)
    elif policy in ("", "fixed"):
        bandwidth = configured
    else:
        raise ValueError(f"unsupported parser.processing_bandwidth_policy: {policy}")

    sample_rate = float(sample_rate_sps)
    if bandwidth <= 0:
        raise ValueError(f"parser bandwidth must be positive: {bandwidth}")
    if sample_rate > 0:
        bandwidth = min(bandwidth, sample_rate)
    return bandwidth


def build_sdr_parser_command(
    config: dict[str, Any],
    run_root: Path,
    max_chunks: int = 0,
    analysis_bandwidth_hz: float | None = None,
) -> tuple[list[str], Path, dict[str, str]]:
    parser_cfg = config.get("parser", {})
    capture_cfg = config.get("capture", {})
    source_root = project_path(parser_cfg.get("source_project_root", BLE_ROOT))
    python = parser_python(parser_cfg, source_root)
    entrypoint = source_root / str(parser_cfg.get("entrypoint", "experiment/bt_40m_pfb_realtime.py"))
    iq_dir = run_root / "iq"
    sdr_dir = run_root / "sdr"
    metadata_path = iq_dir / "metadata.json"
    iq_path = iq_dir / "capture.sc16"
    tags_path = iq_dir / "rx_time_tags_40m.csv"

    if not python.is_file():
        raise FileNotFoundError(f"parser python not found: {python}")
    if not entrypoint.is_file():
        raise FileNotFoundError(f"SDR parser entrypoint not found: {entrypoint}")
    if not iq_path.is_file():
        raise FileNotFoundError(f"IQ capture not found: {iq_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"IQ metadata not found: {metadata_path}")
    if not tags_path.is_file():
        raise FileNotFoundError(f"rx_time tags not found: {tags_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sample_rate = metadata.get("actual_sample_rate_sps") or capture_cfg.get("sample_rate_sps", 40000000)
    center_freq = metadata.get("actual_center_frequency_hz") or capture_cfg.get("center_frequency_hz", 2420000000)
    parser_bandwidth_hz = resolve_parser_bandwidth_hz(parser_cfg, sample_rate, analysis_bandwidth_hz)

    command = [
        str(python),
        str(entrypoint),
        "--source",
        "file",
        "--input-bin",
        str(iq_path),
        "--metadata",
        str(metadata_path),
        "--rx-tags-csv",
        str(tags_path),
        "--output-dir",
        str(sdr_dir),
        "--iq-format",
        str(parser_cfg.get("iq_format", "int16")),
        "--sample-rate",
        str(sample_rate),
        "--center-freq",
        str(center_freq),
        "--bandwidth",
        str(parser_bandwidth_hz),
        "--subband-sample-rate",
        str(parser_cfg.get("subband_sample_rate_sps", 4000000)),
        "--chunk-samples",
        str(parser_cfg.get("chunk_samples", 16000000)),
        "--overlap-samples",
        str(parser_cfg.get("overlap_samples", 200000)),
        "--ble-candidate-detector",
        str(parser_cfg.get("ble_candidate_detector", "fixed")),
        "--ble-threshold",
        str(parser_cfg.get("ble_threshold", 0.01)),
        "--ble-score-threshold",
        str(parser_cfg.get("ble_score_threshold", 3.0)),
        "--ble-lpf-cutoff",
        str(parser_cfg.get("ble_lpf_cutoff_hz", 1000000)),
        "--cleanup-numtaps",
        str(parser_cfg.get("cleanup_numtaps", 31)),
        "--cuda-device",
        str(parser_cfg.get("cuda_device", 0)),
        "--cuda-pfb-backend",
        str(parser_cfg.get("cuda_pfb_backend", "kernel_multi_float_phase_t")),
        "--cuda-cleanup-backend",
        str(parser_cfg.get("cuda_cleanup_backend", "kernel_multi_float")),
        "--cuda-target-dsp-materialization",
        str(parser_cfg.get("cuda_target_dsp_materialization", "full")),
        "--cuda-segment-copy-merge-gap-samples",
        str(parser_cfg.get("cuda_segment_copy_merge_gap_samples", 4096)),
        "--cuda-segment-copy-max-merged-samples",
        str(parser_cfg.get("cuda_segment_copy_max_merged_samples", 262144)),
        "--ble-parser-backend",
        str(parser_cfg.get("ble_parser_backend", "cpp")),
        "--bredr-parser-backend",
        str(parser_cfg.get("bredr_parser_backend", "hybrid")),
        "--cpp-parser-threads",
        str(parser_cfg.get("cpp_parser_threads", 4)),
        "--native-segment-input",
        str(parser_cfg.get("native_segment_input", "legacy")),
        "--target-selection",
        str(parser_cfg.get("target_selection", "full")),
    ]
    bool_cli_flag(command, bool(parser_cfg.get("use_cuda", True)), "--use-cuda", "--no-use-cuda")
    bool_cli_flag(command, bool(parser_cfg.get("cuda_fuse_target_dsp", True)), "--cuda-fuse-target-dsp", "--no-cuda-fuse-target-dsp")
    bool_cli_flag(command, bool(parser_cfg.get("cuda_batch_targets", True)), "--cuda-batch-targets", "--no-cuda-batch-targets")
    bool_cli_flag(command, bool(parser_cfg.get("cuda_threshold_detect", True)), "--cuda-threshold-detect", "--no-cuda-threshold-detect")
    bool_cli_flag(
        command,
        bool(parser_cfg.get("cuda_known_candidate_filter", True)),
        "--cuda-known-candidate-filter",
        "--no-cuda-known-candidate-filter",
    )
    bool_cli_flag(
        command,
        bool(parser_cfg.get("learned_parser_fast_path", False)),
        "--learned-parser-fast-path",
        "--no-learned-parser-fast-path",
    )
    if parser_cfg.get("skip_bredr", False):
        command.append("--skip-bredr")
    if parser_cfg.get("skip_ble", False):
        command.append("--skip-ble")
    if parser_cfg.get("quiet", True):
        command.append("--quiet")
    else:
        command.append("--verbose")
    if parser_cfg.get("timing", True):
        command.append("--timing")
    else:
        command.append("--no-timing")
    if max_chunks > 0:
        command.extend(["--max-chunks", str(max_chunks)])

    if parser_cfg.get("parser_cpus"):
        command = ["taskset", "-c", str(parser_cfg["parser_cpus"])] + command

    env = os.environ.copy()
    pythonpath_parts = [
        str(source_root / "experiment"),
        str(source_root / "ble_fun_test"),
        str(source_root / "build-native"),
    ]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath_parts)
    return command, source_root, {"PYTHONPATH": env["PYTHONPATH"]}


def run_sdr_parser(
    config: dict[str, Any],
    run_root: Path,
    max_chunks: int = 0,
    analysis_bandwidth_hz: float | None = None,
) -> dict[str, Any]:
    sdr_dir = run_root / "sdr"
    if sdr_dir.exists():
        raise FileExistsError(f"SDR parser output already exists: {sdr_dir}")
    sdr_dir.mkdir(parents=True)
    command, cwd, env_delta = build_sdr_parser_command(
        config,
        run_root,
        max_chunks=max_chunks,
        analysis_bandwidth_hz=analysis_bandwidth_hz,
    )
    command_txt = sdr_dir / "command.txt"
    command_json = sdr_dir / "command.json"
    parse_log = sdr_dir / "parse.log"
    command_txt.write_text(shlex.join(command) + "\n", encoding="utf-8")
    write_json(command_json, command)

    env = os.environ.copy()
    env.update(env_delta)
    start_ns = time.time_ns()
    with parse_log.open("w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + shlex.join(command) + "\n")
        log.flush()
        proc = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    end_ns = time.time_ns()

    outputs = {
        "ble_packets": sdr_dir / "ble_packets.csv",
        "btclassic_packets": sdr_dir / "btclassic_packets.csv",
        "packet_events": sdr_dir / "packet_events.csv",
    }
    status = {
        "schema_version": 1,
        "returncode": proc.returncode,
        "start_epoch_ns": start_ns,
        "end_epoch_ns": end_ns,
        "command": command,
        "command_txt": str(command_txt),
        "command_json": str(command_json),
        "parse_log": str(parse_log),
        "output_dir": str(sdr_dir),
        "outputs": {name: str(path) for name, path in outputs.items()},
        "row_counts": {name: csv_data_row_count(path) for name, path in outputs.items()},
        "valid": proc.returncode == 0 and all(path.is_file() for path in outputs.values()),
    }
    write_json(sdr_dir / "parse_status.json", status)
    return status


def run_rtt_sdr_scorer(
    run_id: str,
    run_root: Path,
    distance_m: str,
    analysis_bandwidth_hz: str,
) -> dict[str, Any]:
    results_dir = run_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "match_rtt_sdr_results.py"),
        "--run-root",
        str(run_root),
        "--run-id",
        run_id,
        "--distance-m",
        distance_m,
        "--analysis-bandwidth-hz",
        analysis_bandwidth_hz,
        "--output-dir",
        str(results_dir),
    ]
    log_path = results_dir / "rtt_sdr_scorer.log"
    start_ns = time.time_ns()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + shlex.join(command) + "\n")
        log.flush()
        proc = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    end_ns = time.time_ns()
    metrics_path = results_dir / "recovery_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    return {
        "returncode": proc.returncode,
        "command": command,
        "log_path": str(log_path),
        "start_epoch_ns": start_ns,
        "end_epoch_ns": end_ns,
        "duration_seconds": (end_ns - start_ns) / 1_000_000_000.0,
        "output_dir": str(results_dir),
        "metrics_path": str(metrics_path),
        "metrics": metrics,
        "valid": proc.returncode == 0 and bool(metrics.get("valid")),
    }


def run_phantom_postprocess_scorer(config: dict[str, Any], run_root: Path) -> dict[str, Any]:
    post_cfg = config.get("phantom_postprocess", {})
    output_dir = run_root / "results" / "phantom_duration_rescore"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "phantom_postprocess_scorer.py"),
        "--run-root",
        str(run_root),
        "--output-dir",
        str(output_dir),
        "--tolerance-us",
        str(post_cfg.get("tolerance_us", 18)),
        "--pre-margin-us",
        str(post_cfg.get("pre_margin_us", 40)),
        "--post-margin-us",
        str(post_cfg.get("post_margin_us", 80)),
        "--lowpass-hz",
        str(post_cfg.get("lowpass_hz", 900000)),
        "--smooth-us",
        str(post_cfg.get("smooth_us", 1.5)),
        "--threshold-sigma",
        str(post_cfg.get("threshold_sigma", 8)),
        "--min-threshold-ratio",
        str(post_cfg.get("min_threshold_ratio", 2)),
        "--prediction-search-radius-us",
        str(post_cfg.get("prediction_search_radius_us", 30000)),
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "phantom_postprocess_scorer.log"
    start_ns = time.time_ns()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + shlex.join(command) + "\n")
        log.flush()
        proc = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    end_ns = time.time_ns()
    summary_path = output_dir / "duration_rescore_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    return {
        "returncode": proc.returncode,
        "command": command,
        "log_path": str(log_path),
        "start_epoch_ns": start_ns,
        "end_epoch_ns": end_ns,
        "duration_seconds": (end_ns - start_ns) / 1_000_000_000.0,
        "output_dir": str(output_dir),
        "summary_path": str(summary_path),
        "summary": summary,
        "valid": proc.returncode == 0 and bool(summary),
    }


def rtt_address(elf: Path, toolchain: Path) -> str:
    nm = toolchain / "opt" / "zephyr-sdk" / "arm-zephyr-eabi" / "bin" / "arm-zephyr-eabi-nm"
    proc = subprocess.run([str(nm), str(elf)], check=True, capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[2] == "_SEGGER_RTT":
            return "0x" + parts[0]
    raise RuntimeError(f"_SEGGER_RTT not found in {elf}")


def send_bluetoothctl_command(proc: subprocess.Popen[bytes], command: str, delay_s: float) -> None:
    assert proc.stdin is not None
    proc.stdin.write((command + "\n").encode("utf-8"))
    proc.stdin.flush()
    time.sleep(delay_s)


def run_rtt_notify_smoke(
    config: dict[str, Any],
    run_id: str,
    run_root: Path,
    elf: Path,
    with_b210: bool = False,
) -> dict[str, Any]:
    firmware = config["firmware"]
    peripheral = config["peripheral"]
    central = config.get("central", {})
    rtt = config.get("rtt", {})
    ground_truth_dir = run_root / "ground_truth"
    ground_truth_dir.mkdir(parents=True, exist_ok=True)
    rtt_log = ground_truth_dir / "peripheral_rtt.log"
    rtt_stdout = ground_truth_dir / "jlink_rtt_logger.stdout.log"
    bt_log = ground_truth_dir / "bluetoothctl.log"
    b210_capture: B210Capture | None = None
    b210_metadata: dict[str, Any] | None = None
    bt_proc: subprocess.Popen[bytes] | None = None
    notify_started_epoch_ns: int | None = None
    bt_returncode = -1
    central_channel_map_results: list[dict[str, Any]] = []
    central_channel_map_enabled = bool(central.get("set_host_channel_classification", False))
    capture_cfg = config.get("capture", {})
    start_capture_after_notification = bool(capture_cfg.get("start_after_notification", False))

    notify_s = float(peripheral.get("notify_s", config.get("experiment", {}).get("duration_s", 10)))
    scan_s = float(peripheral.get("scan_s", 7))
    connect_wait_s = float(peripheral.get("connect_wait_s", 8))
    post_disconnect_s = float(rtt.get("post_disconnect_s", 4))
    notify_start_timeout_s = float(peripheral.get("notify_start_timeout_s", 20))
    serial_number = str(firmware["serial_number"])
    address = str(peripheral["address"])
    characteristic_uuid = str(peripheral["data_characteristic_uuid"])
    characteristic_candidates = [characteristic_uuid]
    fallback_values = peripheral.get("data_characteristic_uuid_fallbacks", [])
    if isinstance(fallback_values, str):
        fallback_values = [fallback_values]
    for fallback_uuid in fallback_values:
        fallback_uuid = str(fallback_uuid).strip()
        if fallback_uuid and fallback_uuid not in characteristic_candidates:
            characteristic_candidates.append(fallback_uuid)
    address_value = rtt_address(elf, project_path(firmware["ncs_toolchain"]))

    rtt_command = [
        "JLinkRTTLogger",
        "-Device",
        str(rtt.get("jlink_device", "NRF52840_XXAA")),
        "-If",
        str(rtt.get("interface", "SWD")),
        "-Speed",
        str(rtt.get("speed", 4000)),
        "-USB",
        serial_number,
        "-RTTAddress",
        address_value,
        "-RTTChannel",
        str(rtt.get("channel", 0)),
        str(rtt_log),
    ]
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
            # Apply any optional host channel map before opening IQ. This
            # keeps the post-IQ-ready path available for scan immediately.
            if central_channel_map_enabled:
                result = set_host_channel_classification(
                    central,
                    ground_truth_dir,
                    str(central.get("data_channel_map_hex", "ffff030000")),
                    "central_channel_map_set",
                )
                central_channel_map_results.append(result.__dict__)
                if result.returncode and bool(central.get("require_success", True)):
                    raise RuntimeError(f"central channel map set failed, see {result.log_path}")

            if with_b210 and not start_capture_after_notification:
                b210_capture = start_b210_capture(config, run_id, run_root)
                wait_for_nonempty_file(
                    b210_capture.iq_path,
                    b210_capture.process,
                    float(capture_cfg.get("startup_timeout_s", 30)),
                )
                b210_capture.ready_epoch_ns = time.time_ns()

            with bt_log.open("wb") as bt_out:
                bt_proc = subprocess.Popen(
                    ["stdbuf", "-oL", "-eL", "bluetoothctl"],
                    stdin=subprocess.PIPE,
                    stdout=bt_out,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                # In start-after-notification mode, establish the BLE link
                # first. Otherwise begin scanning as soon as the IQ file
                # proves that UHD is receiving.
                send_bluetoothctl_command(bt_proc, "scan on", scan_s)
                send_bluetoothctl_command(bt_proc, "scan off", 2)
                send_bluetoothctl_command(bt_proc, f"connect {address}", connect_wait_s)
                send_bluetoothctl_command(bt_proc, f"info {address}", 1)
                send_bluetoothctl_command(bt_proc, "menu gatt", 1)
                send_bluetoothctl_command(bt_proc, f"list-attributes {address}", 2)
                notify_enabled_by_host = False
                for candidate_index, candidate_uuid in enumerate(characteristic_candidates):
                    send_bluetoothctl_command(bt_proc, f"select-attribute {candidate_uuid}", 1)
                    # The notification window starts only after bluetoothctl
                    # confirms that notify was enabled. This avoids losing the
                    # first part of the 10-second IQ window during connection.
                    send_bluetoothctl_command(bt_proc, "notify on", 0.1)
                    try:
                        # A stale BlueZ GATT cache can leave the current UUID
                        # unselectable. Give the primary UUID a short probe
                        # window, then try the configured cache-compatible
                        # fallback on the same characteristic handle.
                        probe_timeout_s = (
                            min(notify_start_timeout_s, 3.0)
                            if candidate_index + 1 < len(characteristic_candidates)
                            else notify_start_timeout_s
                        )
                        wait_for_log_text(bt_log, bt_proc, "Notify started", probe_timeout_s)
                        notify_enabled_by_host = True
                        break
                    except TimeoutError:
                        if candidate_index + 1 >= len(characteristic_candidates):
                            raise
                if not notify_enabled_by_host:
                    raise RuntimeError("bluetoothctl did not enable notifications")
                notify_started_epoch_ns = time.time_ns()
                if b210_capture is not None:
                    b210_capture.notify_started_epoch_ns = notify_started_epoch_ns

                if with_b210 and start_capture_after_notification:
                    b210_capture = start_b210_capture(config, run_id, run_root)
                    wait_for_nonempty_file(
                        b210_capture.iq_path,
                        b210_capture.process,
                        float(capture_cfg.get("startup_timeout_s", 30)),
                    )
                    b210_capture.ready_epoch_ns = time.time_ns()
                    b210_capture.notify_started_epoch_ns = notify_started_epoch_ns

                # The requested IQ window starts when IQ is confirmed to be
                # writing, after the link has been connected and
                # notification has been enabled.
                time.sleep(notify_s)

                # Stop IQ immediately after the notification window. In
                # either mode the process uses duration=0 and is stopped
                # explicitly here.
                if b210_capture is not None:
                    stop_b210_capture(b210_capture, "notification_window_complete")

                send_bluetoothctl_command(bt_proc, "notify off", 1)
                send_bluetoothctl_command(bt_proc, "back", 1)
                send_bluetoothctl_command(bt_proc, f"disconnect {address}", 1)
                send_bluetoothctl_command(bt_proc, "quit", 0)
                if bt_proc.stdin:
                    bt_proc.stdin.close()
                bt_returncode = bt_proc.wait(timeout=15)

            if b210_capture is not None and b210_capture.returncode is None:
                wait_b210_capture(b210_capture)

            time.sleep(post_disconnect_s)
        finally:
            if bt_proc is not None and bt_proc.poll() is None:
                if bt_proc.stdin:
                    bt_proc.stdin.close()
                bt_proc.terminate()
                try:
                    bt_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    bt_proc.kill()
                    bt_proc.wait(timeout=5)
            if central_channel_map_enabled and central.get("restore_channel_map_hex"):
                try:
                    result = set_host_channel_classification(
                        central,
                        ground_truth_dir,
                        str(central["restore_channel_map_hex"]),
                        "central_channel_map_restore",
                    )
                    central_channel_map_results.append(result.__dict__)
                except Exception as exc:  # pragma: no cover - hardware cleanup path
                    central_channel_map_results.append(
                        {
                            "command": [],
                            "returncode": -1,
                            "log_path": "",
                            "error": str(exc),
                        }
                    )
            if b210_capture is not None and b210_capture.returncode is None:
                stop_b210_capture(b210_capture, "cleanup_after_exception")
            if rtt_proc.stdin:
                rtt_proc.stdin.write(b"\n")
                rtt_proc.stdin.flush()
            rtt_returncode = rtt_proc.wait(timeout=15)

    if b210_capture is not None:
        b210_metadata = finalize_b210_capture(
            b210_capture,
            run_id,
            rtt_log,
            compute_sha256=bool(config.get("capture", {}).get("compute_sha256", True)),
        )

    result = {
        "rtt_command": rtt_command,
        "rtt_returncode": rtt_returncode,
        "bluetoothctl_returncode": bt_returncode,
        "central_channel_map": central_channel_map_results,
        "rtt_address": address_value,
        "peripheral_rtt_log": str(rtt_log),
        "bluetoothctl_log": str(bt_log),
        "jlink_stdout_log": str(rtt_stdout),
    }
    if b210_metadata is not None:
        result["b210_iq_capture"] = b210_metadata
    return result


def parse_rtt(run_id: str, rtt_log: Path, output_dir: Path) -> tuple[int, dict[str, Any]]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "parse_rtt_ground_truth.py"),
        "--input",
        str(rtt_log),
        "--run-id",
        run_id,
        "--output-dir",
        str(output_dir),
        "--require-phantom",
    ]
    proc = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)
    (output_dir / "parse_rtt_ground_truth.stdout.log").write_text(proc.stdout, encoding="utf-8")
    (output_dir / "parse_rtt_ground_truth.stderr.log").write_text(proc.stderr, encoding="utf-8")
    status_path = output_dir / "rtt_parse_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    return proc.returncode, status


def summarize_smoke(
    rtt_log: Path,
    bt_log: Path,
    parse_status: dict[str, Any],
    notify_s: float = 0,
    notify_interval_ms: float = 0,
    min_notify_fraction: float = 0.0,
) -> dict[str, Any]:
    rtt_text = rtt_log.read_text(encoding="utf-8", errors="replace") if rtt_log.is_file() else ""
    bt_text = bt_log.read_text(encoding="utf-8", errors="replace") if bt_log.is_file() else ""
    counts: dict[str, int] = {}
    tx_status_counts: dict[str, int] = {}
    for line in rtt_text.splitlines():
        stripped = ANSI_ESCAPE_RE.sub("", line).strip()
        match = PHANTOM_LINE_RE.match(stripped)
        if match:
            key = "PHANTOM_" + match.group("kind")
            counts[key] = counts.get(key, 0) + 1
        match = PHANTOM_TX_JSON_RE.match(stripped)
        if match:
            try:
                status = str(json.loads(match.group("payload")).get("status", ""))
            except json.JSONDecodeError:
                status = "malformed"
            tx_status_counts[status] = tx_status_counts.get(status, 0) + 1

    expected_packets = 0
    min_tx_packets = 1
    if notify_s > 0 and notify_interval_ms > 0:
        expected_packets = int((notify_s * 1000.0) // notify_interval_ms)
        min_tx_packets = max(1, int(expected_packets * min_notify_fraction))

    notify_started = "Notify started" in bt_text
    ccc_notify_enabled = "PHANTOM_CCC {\"notify\":1" in rtt_text
    notify_tx_observed = counts.get("PHANTOM_TX", 0) > 0

    return {
        "connected": "Connection successful" in bt_text and counts.get("PHANTOM_CONNECTED", 0) > 0,
        "services_resolved": "ServicesResolved: yes" in bt_text,
        "notify_enabled": notify_started and (ccc_notify_enabled or notify_tx_observed),
        "notify_started_by_central": notify_started,
        "ccc_notify_enabled_observed": ccc_notify_enabled,
        "notify_tx_observed": notify_tx_observed,
        "fault_detected": bool(FAULT_RE.search(rtt_text)),
        "counts": counts,
        "phantom_tx_status_counts": tx_status_counts,
        "expected_notify_packets": expected_packets,
        "min_tx_packets": min_tx_packets,
        "min_tx_packets_met": counts.get("PHANTOM_LL_TX", 0) >= min_tx_packets,
        "notify_status_ok": "-128" not in tx_status_counts and "malformed" not in tx_status_counts,
        "nonzero_notify_status_count": sum(
            count for status, count in tx_status_counts.items() if status not in {"", "0"}
        ),
        "parse_valid": bool(parse_status.get("valid")),
        "has_required_phantom_match_fields": bool(parse_status.get("has_required_phantom_match_fields")),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--distance-m", required=True)
    parser.add_argument("--analysis-bandwidth-mhz", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--repetition", required=True)
    parser.add_argument("--capture-id", default="")
    parser.add_argument("--duration-s", type=float, default=None, help="Override notify duration.")
    parser.add_argument("--scan-s", type=float, default=None, help="Override bluetoothctl scan duration.")
    parser.add_argument("--connect-wait-s", type=float, default=None, help="Override post-connect wait duration.")
    parser.add_argument("--serial-number", default="", help="Override firmware.serial_number.")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-flash", action="store_true")
    parser.add_argument("--with-b210", action="store_true", help="Capture iq/capture.sc16 with the B210 worker.")
    parser.add_argument("--no-b210", action="store_true", help="Disable B210 capture even if config enables it.")
    parser.add_argument(
        "--iq-duration-s",
        type=float,
        default=None,
        help="Override IQ duration, or the post-notification IQ window in stop-after-notification mode.",
    )
    parser.add_argument("--with-sdr-parse", action="store_true", help="Run BLE_encrypt_check SDR parser after B210 capture.")
    parser.add_argument("--no-sdr-parse", action="store_true", help="Disable SDR parser even if config enables it.")
    parser.add_argument("--sdr-max-chunks", type=int, default=0, help="Limit parser chunks for diagnostics; 0 parses full IQ.")
    parser.add_argument("--dry-run", action="store_true", help="Create metadata only; do not touch hardware.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config.resolve())
    if args.duration_s is not None:
        config.setdefault("peripheral", {})["notify_s"] = args.duration_s
    if args.scan_s is not None:
        config.setdefault("peripheral", {})["scan_s"] = args.scan_s
    if args.connect_wait_s is not None:
        config.setdefault("peripheral", {})["connect_wait_s"] = args.connect_wait_s
    if args.serial_number:
        config.setdefault("firmware", {})["serial_number"] = args.serial_number
    if args.iq_duration_s is not None:
        config.setdefault("capture", {})["iq_duration_s"] = args.iq_duration_s
    capture_enabled = (bool(config.get("capture", {}).get("enabled")) or args.with_b210) and not args.no_b210
    sdr_parse_enabled = (bool(config.get("parser", {}).get("enabled")) or args.with_sdr_parse) and not args.no_sdr_parse
    if sdr_parse_enabled and not capture_enabled:
        parser.error("--with-sdr-parse requires B210 capture; add --with-b210 or set capture.enabled=true")

    run_id = build_run_id(args)
    output_root = Path(config.get("paths", {}).get("output_root", "./experiments"))
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    run_root = output_root / run_id
    if run_root.exists():
        parser.error(f"run directory already exists: {run_root}")
    logs_dir = run_root / "logs"
    ground_truth_dir = run_root / "ground_truth"
    logs_dir.mkdir(parents=True, exist_ok=False)
    ground_truth_dir.mkdir(parents=True, exist_ok=True)

    status: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "stage": "start",
        "created_epoch_ns": time.time_ns(),
        "parameters": {
            "distance_m": args.distance_m,
            "analysis_bandwidth_mhz": args.analysis_bandwidth_mhz,
            "environment": args.environment,
            "repetition": args.repetition,
            "with_b210": capture_enabled,
            "with_sdr_parse": sdr_parse_enabled,
        },
        "paths": {
            "run_root": str(run_root),
            "ground_truth": str(ground_truth_dir),
            "iq": str(run_root / "iq"),
            "sdr": str(run_root / "sdr"),
            "results": str(run_root / "results"),
            "logs": str(logs_dir),
        },
        "commands": [],
    }
    write_json(run_root / "run_status.json", status)

    if args.dry_run:
        status.update({"status": "planned", "stage": "dry_run", "completed_epoch_ns": time.time_ns()})
        write_json(run_root / "run_status.json", status)
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0

    try:
        build_dir, merged_hex, elf = firmware_paths(config["firmware"])
        if not args.skip_build:
            status.update({"stage": "firmware_build"})
            write_json(run_root / "run_status.json", status)
            result = build_firmware(config, logs_dir)
            status["commands"].append(result.__dict__)
            if result.returncode:
                raise RuntimeError(f"firmware build failed, see {result.log_path}")

        if not merged_hex.is_file() or not elf.is_file():
            raise FileNotFoundError(f"missing firmware outputs under {build_dir}")

        status["firmware"] = {
            "merged_hex": str(merged_hex),
            "merged_hex_sha256": sha256_file(merged_hex),
            "zephyr_elf": str(elf),
            "zephyr_elf_sha256": sha256_file(elf),
        }

        if not args.skip_flash:
            status.update({"stage": "firmware_flash"})
            write_json(run_root / "run_status.json", status)
            result = flash_firmware(config, merged_hex, logs_dir)
            status["commands"].append(result.__dict__)
            if result.returncode:
                raise RuntimeError(f"firmware flash failed, see {result.log_path}")
            time.sleep(float(config.get("firmware", {}).get("reset_wait_s", 2)))
        else:
            status.update({"stage": "firmware_reset"})
            write_json(run_root / "run_status.json", status)
            result = reset_firmware(config, logs_dir)
            status["commands"].append(result.__dict__)
            if result.returncode:
                raise RuntimeError(f"firmware reset failed, see {result.log_path}")
            time.sleep(float(config.get("firmware", {}).get("reset_wait_s", 2)))

        status.update({"stage": "rtt_notify_smoke"})
        write_json(run_root / "run_status.json", status)
        smoke = run_rtt_notify_smoke(config, run_id, run_root, elf, with_b210=capture_enabled)
        status["smoke"] = smoke

        status.update({"stage": "rtt_parse"})
        write_json(run_root / "run_status.json", status)
        parse_returncode, parse_status = parse_rtt(
            run_id,
            Path(smoke["peripheral_rtt_log"]),
            ground_truth_dir,
        )
        status["rtt_parse"] = {"returncode": parse_returncode, **parse_status}
        summary = summarize_smoke(
            Path(smoke["peripheral_rtt_log"]),
            Path(smoke["bluetoothctl_log"]),
            parse_status,
            notify_s=float(config.get("peripheral", {}).get("notify_s", 0)),
            notify_interval_ms=float(config.get("peripheral", {}).get("notify_interval_ms", 0)),
            min_notify_fraction=float(config.get("peripheral", {}).get("min_notify_fraction", 0.0)),
        )
        status["smoke_summary"] = summary
        ok = (
            smoke["rtt_returncode"] == 0
            and parse_returncode == 0
            and summary["connected"]
            and summary["services_resolved"]
            and summary["notify_enabled"]
            and not summary["fault_detected"]
            and summary["counts"].get("PHANTOM_LL_TX", 0) > 0
            and summary["min_tx_packets_met"]
            and summary["notify_status_ok"]
        )
        if capture_enabled:
            iq_capture = smoke.get("b210_iq_capture", {})
            ok = ok and bool(iq_capture.get("valid_no_overflow_or_gap"))

        if sdr_parse_enabled:
            iq_capture = smoke.get("b210_iq_capture", {})
            if not iq_capture.get("valid_no_overflow_or_gap"):
                raise RuntimeError("refusing SDR parse because B210 capture was invalid")
            status.update({"stage": "sdr_parse"})
            write_json(run_root / "run_status.json", status)
            analysis_bandwidth_hz = float(args.analysis_bandwidth_mhz) * 1_000_000.0
            sdr_parse = run_sdr_parser(
                config,
                run_root,
                max_chunks=args.sdr_max_chunks,
                analysis_bandwidth_hz=analysis_bandwidth_hz,
            )
            status["sdr_parse"] = sdr_parse
            ok = ok and bool(sdr_parse.get("valid"))
            if sdr_parse.get("valid"):
                status.update({"stage": "rtt_sdr_score"})
                write_json(run_root / "run_status.json", status)
                score = run_rtt_sdr_scorer(
                    run_id,
                    run_root,
                    args.distance_m,
                    str(analysis_bandwidth_hz),
                )
                if sdr_parse.get("end_epoch_ns") and score.get("start_epoch_ns"):
                    score["parse_end_to_rate_start_seconds"] = (
                        score["start_epoch_ns"] - sdr_parse["end_epoch_ns"]
                    ) / 1_000_000_000.0
                if sdr_parse.get("start_epoch_ns") and score.get("end_epoch_ns"):
                    score["parse_start_to_rate_end_seconds"] = (
                        score["end_epoch_ns"] - sdr_parse["start_epoch_ns"]
                    ) / 1_000_000_000.0
                status["rtt_sdr_score"] = score
                ok = ok and bool(score.get("valid"))
                if bool(config.get("phantom_postprocess", {}).get("enabled", True)):
                    status.update({"stage": "phantom_postprocess_score"})
                    write_json(run_root / "run_status.json", status)
                    postprocess = run_phantom_postprocess_scorer(config, run_root)
                    status["phantom_postprocess_score"] = postprocess
                    ok = ok and bool(postprocess.get("valid"))
        status.update(
            {
                "status": "passed" if ok else "failed",
                "stage": "complete" if ok else "smoke_validation",
                "completed_epoch_ns": time.time_ns(),
            }
        )
    except Exception as exc:
        status.update(
            {
                "status": "failed",
                "stage": status.get("stage", "unknown"),
                "reason": str(exc),
                "completed_epoch_ns": time.time_ns(),
            }
        )
        write_json(run_root / "run_status.json", status)
        print(json.dumps(status, indent=2, sort_keys=True))
        return 1

    write_json(run_root / "run_status.json", status)
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0 if status["status"] == "passed" else 2


if __name__ == "__main__":
    sys.exit(main())
