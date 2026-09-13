#!/usr/bin/env python3
"""Run one phone-ready X310 HRS rate test without RTT or BlueZ.

The X310 first captures to local NVMe.  After a clean capture (no UHD
overflow indication), the run is copied to PSSD and parsed there.  The raw
NVMe IQ file is deleted after the copy; the metadata and capture logs remain
on NVMe.  Distance metadata is optional so the same entry point can be used
for the current rate-only test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "x310_phone_hrs_distance_experiment.yaml"
DEFAULT_NVME_ROOT = PROJECT_ROOT / "testdata"
DEFAULT_PSSD_ROOT = Path("/path/to/PhantomChannel/testdata")
MAX_X310_RX_GAIN_DB = 37.5

sys.path.insert(0, str(PROJECT_ROOT))
from tools import run_range_bandwidth_experiment as range_runner  # noqa: E402
from tools import run_x310_bandwidth_experiment as x310_runner  # noqa: E402


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_config(path: Path) -> dict[str, Any]:
    return range_runner.load_config(path.expanduser().resolve())


def build_capture_args(
    cli: argparse.Namespace,
    config: dict[str, Any],
    run_id: str,
    target: str = "pssd",
) -> argparse.Namespace:
    capture = config.get("capture", {})
    parser_cfg = config.get("parser", {})
    values = [
        "--target", target,
        "--capture-id", run_id,
        "--duration-s", str(cli.duration_s),
        "--sample-rate-sps", str(capture.get("requested_sample_rate_sps", 80e6)),
        "--expected-sample-rate-sps", str(capture.get("expected_sample_rate_sps", 100e6)),
        "--bandwidth-hz", str(capture.get("bandwidth_hz", 80e6)),
        "--center-frequency-hz", str(capture.get("center_frequency_hz", 2440e6)),
        "--gain-db", str(
            cli.gain_db if cli.gain_db is not None else capture.get("gain_db", 50)
        ),
        "--antenna", str(capture.get("antenna", "RX2")),
        "--channel", str(capture.get("channel", 0)),
        "--usrp-args", str(capture.get("usrp_args", "type=x300,addr=192.168.40.2,master_clock_rate=200e6")),
        "--nvme-root", str(cli.nvme_root),
        "--pssd-root", str(cli.pssd_root),
        "--subband-sample-rate-sps", str(parser_cfg.get("subband_sample_rate_sps", 4e6)),
        "--chunk-samples", str(parser_cfg.get("chunk_samples", 16_000_000)),
        "--overlap-samples", str(parser_cfg.get("overlap_samples", 200_000)),
        "--cpp-parser-threads", str(parser_cfg.get("cpp_parser_threads", 4)),
        "--cuda-device", str(parser_cfg.get("cuda_device", 0)),
        "--parser-python", str(parser_cfg.get("python", x310_runner.DEFAULT_PARSER_PYTHON)),
        "--parser-entrypoint", str(parser_cfg.get("entrypoint", x310_runner.DEFAULT_PARSER_ENTRYPOINT)),
        "--ble-root", str(parser_cfg.get("source_project_root", x310_runner.DEFAULT_BLE_ROOT)),
    ]
    args = x310_runner.build_parser().parse_args(values)
    args.capture_bin = args.capture_bin.expanduser().resolve()
    args.uhd_library = args.uhd_library.expanduser().resolve()
    args.ble_root = args.ble_root.expanduser().resolve()
    args.parser_python = args.parser_python.expanduser().resolve()
    args.parser_entrypoint = args.parser_entrypoint.expanduser().resolve()
    args.nvme_root = args.nvme_root.expanduser().resolve()
    args.pssd_root = cli.pssd_root.expanduser().resolve()
    return args


def phone_manifest(cli: argparse.Namespace, config: dict[str, Any], run_id: str) -> dict[str, Any]:
    phone = config.get("phone", {})
    firmware = config.get("firmware", {})
    return {
        "schema_version": 1,
        "run_id": run_id,
        "device_name": phone.get("device_name", "PhantomHRS"),
        "firmware_variant": phone.get("firmware_variant", "phantomchannel_hrs_peripheral"),
        "firmware_build_dir": firmware.get("build_dir", ""),
        "phone_model": cli.phone_model,
        "phone_os": cli.phone_os,
        "phone_app": cli.phone_app,
        "phone_ready_confirmed": bool(cli.phone_ready),
        "operator_confirmed_connected": bool(cli.phone_ready),
        "operator_confirmed_hrs_subscribed": bool(cli.phone_ready),
        "hrs_service_uuid": "180d",
        "hrs_measurement_uuid": "2a37",
        "rtt_available": False,
        "bluez_central_used": False,
        "jlink_used": False,
        "distance_m": cli.distance_m,
        "repetition": cli.repetition,
        "rate_test_scope": "rate_only; distance experiment not executed",
    }


def confirm_phone_ready(cli: argparse.Namespace, config: dict[str, Any]) -> None:
    if cli.phone_ready:
        return
    if not sys.stdin.isatty():
        raise RuntimeError("non-interactive run requires --phone-ready after phone connected and 0x2A37 subscribed")
    phone = config.get("phone", {})
    checklist = (
        "Phone-ready gate:\n"
        f"  Scan and connect {phone.get('device_name', 'PhantomHRS')}\n"
        "  Discover HRS 0x180D and subscribe to Heart Rate Measurement 0x2A37\n"
        "  Confirm BPM notifications continue; keep phone unlocked and stationary\n"
        "  Do not use BlueZ/J-Link for this run\n"
        "Press Enter only after all checks are complete."
    )
    print(checklist)
    input()


def run_one_stage_parser(
    args: argparse.Namespace,
    run_root: Path,
    metadata_path: Path,
    iq_path: Path,
    *,
    ble_threshold: float | None = None,
) -> dict[str, Any]:
    output_dir = run_root / "diagnostics" / "one_stage_cpp"
    output_dir.mkdir(parents=True, exist_ok=False)
    actual_rate = float(json.loads(metadata_path.read_text(encoding="utf-8"))["actual_sample_rate_sps"])
    command = x310_runner.build_parser_command(args, metadata_path, iq_path, output_dir, actual_rate)
    if ble_threshold is not None:
        command.extend(["--ble-threshold", str(ble_threshold)])
    command_path = output_dir / "command.txt"
    log_path = output_dir / "parse.log"
    command_path.write_text(shlex.join(command) + "\n", encoding="utf-8")
    start_ns = time.time_ns()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + shlex.join(command) + "\n")
        log.flush()
        proc = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=x310_runner.parser_environment(args),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    end_ns = time.time_ns()
    outputs = {
        "ble_packets": output_dir / "ble_packets.csv",
        "packet_events": output_dir / "packet_events.csv",
        "target_selection": output_dir / "target_selection.csv",
    }
    result = {
        "mode": "one_stage_cpp_cuda_frontend",
        "returncode": proc.returncode,
        "start_epoch_ns": start_ns,
        "end_epoch_ns": end_ns,
        "duration_seconds": (end_ns - start_ns) / 1_000_000_000.0,
        "command": command,
        "command_path": str(command_path),
        "log_path": str(log_path),
        "output_dir": str(output_dir),
        "outputs": {name: str(path) for name, path in outputs.items()},
        "output_exists": {name: path.is_file() for name, path in outputs.items()},
        "valid": proc.returncode == 0 and all(path.is_file() for path in outputs.values()),
        "ble_project_untouched": True,
    }
    write_json(output_dir / "parse_status.json", result)
    return result


def run_candidate_score(
    run_root: Path,
    metadata_path: Path,
    parser_csv: Path,
    *,
    pattern_enabled: bool = False,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    output_dir = run_root / "results"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "score_iq_parser_candidates.py"),
        "--metadata", str(metadata_path),
        "--parser-csv", str(parser_csv),
        "--output-dir", str(output_dir),
    ]
    if pattern_enabled:
        command.append("--pattern")
    if extra_args:
        command.extend(extra_args)
    log_path = output_dir / "parser_candidate_score.log"
    start_ns = time.time_ns()
    output_dir.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + shlex.join(command) + "\n")
        log.flush()
        proc = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    end_ns = time.time_ns()
    rate_path = output_dir / "parser_candidate_rate.json"
    rate = json.loads(rate_path.read_text(encoding="utf-8")) if rate_path.is_file() else {}
    return {
        "returncode": proc.returncode,
        "start_epoch_ns": start_ns,
        "end_epoch_ns": end_ns,
        "duration_seconds": (end_ns - start_ns) / 1_000_000_000.0,
        "command": command,
        "log_path": str(log_path),
        "rate_path": str(rate_path),
        "rate": rate,
        "valid": proc.returncode == 0 and rate_path.is_file(),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def detect_overflow_indications(capture_status: dict[str, Any]) -> list[str]:
    """Return actual UHD overflow-indication lines from the capture log.

    The rx_samples_to_file disk-write pre-test warning ("Disk write test
    indicates that an overflow is likely to occur.") is not treated as an
    overflow; only real "Got an overflow indication" lines trigger a
    re-capture.
    """

    matches = (capture_status.get("capture") or {}).get("overflow_text_matches") or []
    return [line for line in matches if re.search(r"overflow indication", line, re.IGNORECASE)]


def verify_pssd_run(run_root: Path, run_id: str) -> dict[str, Any]:
    """In-place verification for a run captured directly on PSSD."""

    iq_path = run_root / "iq/capture.sc16"
    metadata_path = run_root / "iq/metadata.json"
    required = [
        iq_path,
        metadata_path,
        run_root / "diagnostics/one_stage_cpp/ble_packets.csv",
        run_root / "results/parser_candidate_rate.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    size_matches = False
    if not missing and iq_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = int(metadata.get("samples", 0) or 0) * int(metadata.get("bytes_per_complex_sample", 4) or 4)
        size_matches = iq_path.stat().st_size == expected
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "run_root": str(run_root),
        "target": "pssd",
        "required_files_missing": missing,
        "iq_size_matches_metadata": size_matches,
        "verified": not missing and size_matches,
    }
    write_json(run_root / "pssd_verify_manifest.json", manifest)
    return manifest


def build_metrics(
    rate: dict[str, Any],
    interval_ms: int,
    fallback_duration_s: float,
    *,
    expected_events: int | None = None,
    covert_len: int = 0,
) -> dict[str, Any]:
    """PSR / BER metrics shared with the advertising one-click runner.

    PSR uses the nominal denominator ``round(window / interval)`` (with
    delay-zero firmware the on-air rate equals the nominal rate).  The
    boundary band covers the ±1 phase uncertainty of the window plus at most
    one end-of-window tail-clip packet.  BER is reported both globally and
    excluding per-packet BER outliers (collision-corrupted packets) when the
    scorer ran in pattern mode.
    """

    events = int(rate.get("marker_valid_candidates", 0) or 0)
    unique = int(rate.get("seq_unique_count", 0) or 0)
    duration = float(rate.get("iq_capture_duration_s", 0) or 0) or float(fallback_duration_s or 0)
    if expected_events is None:
        expected = int(round(duration * 1000.0 / interval_ms)) if duration > 0 and interval_ms > 0 else 0
    else:
        expected = int(expected_events)
    psr = unique / expected if expected else 0.0
    # Boundary analysis for the PSR denominator: the window catches either
    # floor(T/P) or ceil(T/P) event starts depending on the phase between the
    # broadcast/notification train and the capture window (±1 packet), and an
    # event whose post-CRC tail extends past the end of the file cannot yield a
    # complete PC frame (≤ ~1 more packet for 239 B @ 20 ms).
    boundary_uncertainty_packets = 2
    packet_duration_s = (25 + max(0, int(covert_len))) * 8e-6  # 1M PHY, on-air bytes ≈ 25 + covert
    period_s = interval_ms / 1000.0
    expected_observable = 0
    if duration > 0 and period_s > 0 and duration > packet_duration_s:
        expected_observable = int((duration - packet_duration_s) // period_s) + 1
    ber = rate.get("pattern_ber")
    psr_exact = rate.get("psr_exact")
    exact = rate.get("pattern_exact_packets")
    compared = rate.get("pattern_candidates_compared")
    byte_recovery = rate.get("pattern_byte_recovery")
    g_e2e = rate.get("iq_window_parser_candidate_data_kbps")
    payload = rate.get("iq_window_parser_candidate_payload_data_kbps")
    outlier_threshold = float(rate.get("pattern_outlier_ber_threshold", 0.10) or 0.10)
    outlier_raw = rate.get("pattern_outlier_packets")
    outlier_packets = int(outlier_raw) if outlier_raw is not None else None
    return {
        "events": events,
        "unique_seq": unique,
        "expected_events": expected,
        "psr": psr,
        "packet_duration_s": packet_duration_s,
        "expected_events_observable": expected_observable,
        "boundary_uncertainty_packets": boundary_uncertainty_packets,
        "psr_boundary_min": unique / (expected + boundary_uncertainty_packets) if expected else None,
        "psr_boundary_max": unique / max(1, expected - boundary_uncertainty_packets) if expected else None,
        "ber": ber,
        "ber_excluding_outliers": rate.get("pattern_ber_excluding_outliers"),
        "outlier_threshold": outlier_threshold,
        "outlier_packets": outlier_packets,
        "outlier_fraction": rate.get("pattern_outlier_fraction"),
        "outlier_error_share": rate.get("pattern_outlier_error_share"),
        "collision_outlier_rate": outlier_packets / expected if expected and outlier_packets is not None else None,
        "psr_exact": psr_exact,
        "exact_packets": exact,
        "compared_packets": compared,
        "byte_recovery": byte_recovery,
        "g_e2e_kbps": g_e2e,
        "payload_kbps": payload,
    }


def psr_ber_metric_lines(metrics: dict[str, Any]) -> list[str]:
    """PSR / boundary / BER / collision-outlier lines shared by both runners."""

    lines: list[str] = []
    unique = metrics.get("unique_seq")
    expected = metrics.get("expected_events")
    psr = metrics.get("psr")
    if isinstance(unique, int) and expected and isinstance(psr, (int, float)):
        lines.append(f"PSR\t{psr:.1%}（{unique}/{expected}）")
    psr_min = metrics.get("psr_boundary_min")
    psr_max = metrics.get("psr_boundary_max")
    uncertainty = metrics.get("boundary_uncertainty_packets")
    if (
        isinstance(psr_min, (int, float))
        and isinstance(psr_max, (int, float))
        and isinstance(uncertainty, int)
        and expected
    ):
        lines.append(
            f"PSR 边界范围\t{psr_min:.1%}–{psr_max:.1%}"
            f"（分母 {expected - uncertainty}–{expected + uncertainty}，"
            f"窗口首尾相位 ±1 + 尾部裁剪 ≤1）"
        )
    ber = metrics.get("ber")
    if isinstance(ber, (int, float)):
        lines.append(f"BER\t{ber * 100:.2f}%")
    ber_clean = metrics.get("ber_excluding_outliers")
    threshold = metrics.get("outlier_threshold")
    if isinstance(ber_clean, (int, float)) and isinstance(threshold, (int, float)):
        lines.append(f"BER（剔除冲突包，>{threshold:.0%}）\t{ber_clean * 100:.2f}%")
    outliers = metrics.get("outlier_packets")
    collision_rate = metrics.get("collision_outlier_rate")
    if isinstance(outliers, int) and isinstance(collision_rate, (int, float)) and expected:
        lines.append(
            f"冲突包数 / 冲突包率\t{outliers} / {collision_rate:.1%}（{outliers}/{expected}）"
        )
    return lines


def print_tail_summary(status: dict[str, Any]) -> None:
    """Print a concise metric summary at the tail of a successful run output."""

    run_id = status.get("run_id", "")
    rate = (status.get("candidate_score") or {}).get("rate") or {}
    metrics = (status.get("metrics") or {}).get("summary") or {}
    shared_lines = psr_ber_metric_lines(metrics)
    psr_lines = [line for line in shared_lines if line.startswith("PSR")]
    ber_lines = [line for line in shared_lines if line.startswith(("BER", "冲突包数"))]
    if not rate.get("seq_unique_count"):
        print(f"\n=== 指标汇总 (run_id={run_id}) ===")
        print("候选 / 唯一 seq\t0 / 0")
        input_rows = rate.get("input_parser_rows")
        detail = f"解析器 BLE 包数 = {input_rows}" if input_rows is not None else "解析器无输出"
        print(
            "⚠ 未恢复任何 Phantom 帧"
            f"（{detail}）：请检查天线连接、SDR 是否在线、"
            "手机是否已连接并正在发送 HRS 通知"
        )
        return
    candidates = int(rate.get("marker_valid_candidates", 0) or 0)
    unique = int(rate.get("seq_unique_count", 0) or 0)
    g_e2e_kbps = float(rate.get("iq_window_parser_candidate_data_kbps", 0.0) or 0.0)
    pc_lens = rate.get("rate_pc_frame_length_distribution") or {}
    if len(pc_lens) == 1:
        pc_len, count = next(iter(pc_lens.items()))
        covert = max(0, int(pc_len) - 6)
        frame_desc = f"全部 {pc_len} B（{covert}+6），{count} 包无杂帧"
    else:
        frame_desc = ", ".join(
            f"{length} B×{count}" for length, count in sorted(pc_lens.items(), key=lambda kv: int(kv[0]))
        )
    lines = [
        f"候选 / 唯一 seq\t{candidates} / {unique}",
        f"G_e2e\t{g_e2e_kbps:.2f} kbps",
    ]
    # Insert PSR / boundary lines after the recovered-count line.
    if psr_lines:
        lines[1:1] = psr_lines
    psr_exact = rate.get("psr_exact")
    if isinstance(psr_exact, (int, float)):
        exact = rate.get("pattern_exact_packets", "?")
        compared = rate.get("pattern_candidates_compared", "?")
        lines.append(f"PSR_exact\t{psr_exact:.2%}（{exact}/{compared} 逐字节一致）")
    byte_recovery = rate.get("pattern_byte_recovery")
    if isinstance(byte_recovery, (int, float)):
        lines.append(f"byte recovery\t{byte_recovery:.2%}")
    ber = rate.get("pattern_ber")
    metrics_has_ber = any(line.startswith("BER\t") for line in ber_lines)
    if isinstance(ber, (int, float)) and not metrics_has_ber:
        lines.append(f"BER\t{ber:.4g}")
    # Insert the shared BER / collision-outlier lines right after BER.
    lines.extend(ber_lines)
    lines.append(f"帧长度\t{frame_desc}")
    theoretical = rate.get("seq_span_theoretical_packets")
    if isinstance(theoretical, (int, float)) and theoretical:
        lines.append(f"seq 理论窗口包数（主体跨度）\t{int(theoretical)}")
        span_fraction = rate.get("recovered_within_span_fraction")
        if isinstance(span_fraction, (int, float)):
            lines.append(f"恢复率（候选 / 理论跨度）\t{span_fraction:.1%}")

    paths = status.get("paths", {})
    run_root = Path(paths.get("run_root_pssd") or paths.get("run_root_nvme", ""))
    crop_metrics = run_root / "results/bandwidth_crops/bandwidth_crop_metrics.json"
    if crop_metrics.is_file():
        try:
            rows = json.loads(crop_metrics.read_text(encoding="utf-8"))
            c_bw = ", ".join(
                f"{int(row['bandwidth_mhz'])} MHz:{row['c_bw_candidates_estimated']:.1%}"
                for row in rows
                if row.get("bandwidth_mhz") is not None
            )
            lines.append(f"C_bw（候选级）\t{c_bw}")
        except (OSError, ValueError, TypeError, KeyError):
            pass
    else:
        lines.append("C_bw（候选级）\t未计算（运行 tools/analyze_bandwidth_crops.py 后输出）")

    print(f"\n=== 指标汇总 (run_id={run_id}) ===")
    print("\n".join(lines))


def file_inventory(root: Path) -> tuple[int, int]:
    files = [path for path in root.rglob("*") if path.is_file()]
    return len(files), sum(path.stat().st_size for path in files)


def copy_and_verify(
    run_root: Path,
    pssd_root: Path,
    run_id: str,
    *,
    delete_nvme_iq: bool = True,
) -> dict[str, Any]:
    """Copy a validated run to PSSD, verify it, and by default delete the NVMe IQ.

    Only ``iq/capture.sc16`` is removed from the NVMe run root after a fully
    verified PSSD copy; ``iq/metadata.json`` and logs are retained so the run
    stays re-scorable and auditable.  Pass ``delete_nvme_iq=False`` to keep
    the raw IQ on NVMe.
    """

    target = pssd_root / run_id
    if target.exists():
        raise FileExistsError(f"PSSD target already exists; refusing overwrite: {target}")
    shutil.copytree(run_root, target)
    required_relative = [
        Path("iq/capture.sc16"),
        Path("iq/metadata.json"),
        Path("diagnostics/one_stage_cpp/ble_packets.csv"),
        Path("results/parser_candidate_rate.json"),
    ]
    missing = [str(rel) for rel in required_relative if not (run_root / rel).is_file() or not (target / rel).is_file()]
    iq_source = run_root / "iq/capture.sc16"
    iq_target = target / "iq/capture.sc16"
    source_count, source_bytes = file_inventory(run_root)
    target_count, target_bytes = file_inventory(target)
    source_hash = file_sha256(iq_source) if iq_source.is_file() else None
    target_hash = file_sha256(iq_target) if iq_target.is_file() else None
    verified = not missing and source_count == target_count and source_bytes == target_bytes and source_hash == target_hash
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "source_root": str(run_root),
        "target_root": str(target),
        "source_file_count": source_count,
        "target_file_count": target_count,
        "source_total_bytes": source_bytes,
        "target_total_bytes": target_bytes,
        "iq_source_sha256": source_hash,
        "iq_target_sha256": target_hash,
        "required_files_missing": missing,
        "verified": verified,
        "nvme_source_retained": True,
        "nvme_iq_deleted": False,
    }
    if verified and delete_nvme_iq and iq_source.is_file():
        iq_source.unlink()
        post_count, post_bytes = file_inventory(run_root)
        manifest["nvme_source_retained"] = False
        manifest["nvme_iq_deleted"] = True
        manifest["nvme_iq_path"] = str(iq_source)
        manifest["nvme_iq_sha256"] = source_hash
        manifest["nvme_iq_deleted_epoch_ns"] = time.time_ns()
        manifest["source_file_count_after_iq_delete"] = post_count
        manifest["source_total_bytes_after_iq_delete"] = post_bytes
    write_json(target / "copy_manifest.json", manifest)
    write_json(run_root / "copy_manifest.json", manifest)
    return manifest


def copy_to_pssd(
    run_root: Path,
    pssd_root: Path,
    run_id: str,
    *,
    delete_nvme_iq: bool = True,
) -> dict[str, Any]:
    """Copy an NVMe capture to PSSD without validating the copied contents."""

    target = pssd_root / run_id
    if target.exists():
        raise FileExistsError(f"PSSD target already exists; refusing overwrite: {target}")
    shutil.copytree(run_root, target)

    iq_source = run_root / "iq/capture.sc16"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "source_root": str(run_root),
        "target_root": str(target),
        "copied_epoch_ns": time.time_ns(),
        "nvme_iq_path": str(iq_source),
        "nvme_iq_deleted": False,
    }
    if delete_nvme_iq and iq_source.is_file():
        iq_source.unlink()
        manifest["nvme_iq_deleted"] = True
        manifest["nvme_iq_deleted_epoch_ns"] = time.time_ns()

    write_json(target / "copy_manifest.json", manifest)
    write_json(run_root / "copy_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--capture-id", default="")
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--distance-m", type=float, default=None)
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=None,
        help="notification/advertising interval in ms for the PSR denominator "
        "(default from config phone.notification_interval_ms; must match firmware)",
    )
    parser.add_argument(
        "--expected-events",
        type=int,
        default=0,
        help="override the PSR denominator (default: round(window x 1000/interval_ms))",
    )
    parser.add_argument(
        "--pattern-outlier-ber-threshold",
        type=float,
        default=0.10,
        help="per-packet BER above which a packet is a collision outlier excluded "
        "from BER（剔除冲突包）; default: 0.10",
    )
    parser.add_argument(
        "--gain-db",
        type=float,
        default=None,
        help=f"X310 RX gain in dB (device range 0.0..{MAX_X310_RX_GAIN_DB}, "
        "step 0.5; default from config)",
    )
    parser.add_argument(
        "--ble-threshold",
        type=float,
        default=None,
        help="BLE parser detection threshold (default from config parser.ble_threshold)",
    )
    parser.add_argument("--phone-ready", action="store_true")
    parser.add_argument("--phone-model", default="unspecified")
    parser.add_argument("--phone-os", default="unspecified")
    parser.add_argument("--phone-app", default="nRF Connect")
    parser.add_argument("--nvme-root", type=Path, default=DEFAULT_NVME_ROOT)
    parser.add_argument("--pssd-root", type=Path, default=DEFAULT_PSSD_ROOT)
    parser.add_argument("--min-sample-fraction", type=float, default=0.99)
    parser.add_argument(
        "--capture-target",
        choices=("pssd", "nvme"),
        default="nvme",
        help="capture destination (default: nvme); NVMe captures are copied "
        "to PSSD before parsing",
    )
    parser.add_argument(
        "--keep-nvme-iq",
        action="store_true",
        help="retain iq/capture.sc16 on NVMe after the PSSD copy "
        "(NVMe-staging mode only; default: delete the raw IQ after copying)",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    cli = build_parser().parse_args(argv)
    if cli.duration_s <= 0:
        raise SystemExit("--duration-s must be positive")
    if cli.repetition < 1:
        raise SystemExit("--repetition must be positive")
    if not 0 < cli.min_sample_fraction <= 1:
        raise SystemExit("--min-sample-fraction must be in (0, 1]")
    if cli.gain_db is not None and cli.gain_db > MAX_X310_RX_GAIN_DB:
        print(
            f"⚠ --gain-db {cli.gain_db} 超过 X310 最大 RX 增益 "
            f"{MAX_X310_RX_GAIN_DB} dB，将被 UHD 钳制",
            file=sys.stderr,
        )
    config = load_config(cli.config)
    run_id = cli.capture_id or f"{time.strftime('%Y%m%d_%H%M%S')}_x310_phonehrs_80m_rate_rep{cli.repetition}"
    nvme_root = cli.nvme_root.expanduser().resolve()
    pssd_root = cli.pssd_root.expanduser().resolve()
    capture_target = cli.capture_target
    capture_root = pssd_root if capture_target == "pssd" else nvme_root
    capture_run_root = capture_root / run_id
    pssd_run_root = pssd_root / run_id
    if capture_run_root.exists():
        raise SystemExit(f"{capture_target.upper()} run directory already exists: {capture_run_root}")
    if capture_target == "nvme" and pssd_run_root.exists():
        raise SystemExit(f"PSSD run directory already exists: {pssd_run_root}")
    confirm_phone_ready(cli, config)
    capture_args = build_capture_args(cli, config, run_id, capture_target)
    manifest = phone_manifest(cli, config, run_id)
    plan = {
        "run_id": run_id,
        "capture_target": capture_target,
        "run_root_nvme": str(nvme_root / run_id),
        "run_root_pssd": str(pssd_root / run_id),
        "phone": manifest,
        "capture": {
            "duration_s": cli.duration_s,
            "requested_sample_rate_sps": capture_args.sample_rate_sps,
            "requested_bandwidth_hz": capture_args.bandwidth_hz,
            "center_frequency_hz": capture_args.center_frequency_hz,
            "gain_db": capture_args.gain_db,
            "ble_threshold": (
                cli.ble_threshold
                if cli.ble_threshold is not None
                else float(config.get("parser", {}).get("ble_threshold", 0.01))
            ),
            "antenna": capture_args.antenna,
            "channel": capture_args.channel,
        },
        "psr_denominator": {
            "interval_ms": (
                cli.interval_ms
                if cli.interval_ms is not None
                else int(config.get("phone", {}).get("notification_interval_ms", 20))
            ),
            "expected_events_override": cli.expected_events or None,
            "pattern_outlier_ber_threshold": cli.pattern_outlier_ber_threshold,
        },
        "flow": (
            [
                "phone_ready",
                "pssd_capture",
                "capture_validation",
                "one_stage_cpp_parse",
                "no_rtt_candidate_score",
            ]
            if capture_target == "pssd"
            else [
                "phone_ready",
                "nvme_capture",
                "capture_validation",
                "pssd_copy",
                "nvme_iq_cleanup" if not cli.keep_nvme_iq else "nvme_iq_retained",
                "one_stage_cpp_parse",
                "no_rtt_candidate_score",
            ]
        ),
        "nvme_iq_after_copy": (
            "retained"
            if capture_target == "pssd" or cli.keep_nvme_iq
            else "deleted"
        ),
        "distance_experiment_executed": False,
    }
    if cli.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    status: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "stage": "x310_capture",
        "created_epoch_ns": time.time_ns(),
        "phone": manifest,
        "plan": plan,
        "paths": {"run_root_nvme": str(capture_run_root), "run_root_pssd": str(pssd_run_root)},
    }
    run_root = capture_run_root
    try:
        capture_status = x310_runner.run_target(
            capture_args,
            x310_runner.Target(capture_target, capture_root),
            run_id,
        )
        status["capture"] = capture_status
        status["stage"] = "capture_validation"
        write_json(run_root / "phone_manifest.json", manifest)
        write_json(run_root / "run_status.json", status)
        if capture_status.get("status") != "passed":
            raise RuntimeError(
                f"X310 capture validation failed; IQ retained on {capture_target.upper()}"
            )
        overflow_lines = detect_overflow_indications(capture_status)
        if overflow_lines:
            status["status"] = "failed"
            status["stage"] = "capture_validation"
            status["reason"] = "overflow indication; re-capture required"
            status["overflow_indications"] = overflow_lines
            status["completed_epoch_ns"] = time.time_ns()
            write_json(run_root / "run_status.json", status)
            print(json.dumps(status, indent=2, sort_keys=True))
            print("⚠ 采集发生溢出（overflow indication），需要重采（re-capture required）")
            return 3

        if capture_target == "nvme":
            status["stage"] = "pssd_copy"
            write_json(capture_run_root / "run_status.json", status)
            copy_manifest = copy_to_pssd(
                capture_run_root,
                pssd_root,
                run_id,
                delete_nvme_iq=not cli.keep_nvme_iq,
            )
            status["copy"] = copy_manifest
            status["nvme_iq_cleanup"] = {
                "nvme_iq_deleted": copy_manifest["nvme_iq_deleted"],
                "nvme_iq_path": copy_manifest["nvme_iq_path"],
            }
            run_root = pssd_run_root
            status["stage"] = "one_stage_cpp_parse"
            write_json(capture_run_root / "run_status.json", status)
            write_json(run_root / "run_status.json", status)

        metadata_path = run_root / "iq/metadata.json"
        iq_path = run_root / "iq/capture.sc16"
        ble_threshold = (
            cli.ble_threshold
            if cli.ble_threshold is not None
            else float(config.get("parser", {}).get("ble_threshold", 0.01))
        )
        parser_result = run_one_stage_parser(
            capture_args,
            run_root,
            metadata_path,
            iq_path,
            ble_threshold=ble_threshold,
        )
        status["stage"] = "one_stage_cpp_parse"
        status["parser"] = parser_result
        write_json(run_root / "run_status.json", status)
        if not parser_result["valid"]:
            raise RuntimeError("one-stage parser failed; see diagnostics/one_stage_cpp/parse.log")
        pattern_enabled = bool(config.get("phone", {}).get("covert_len_bytes", 0))
        score_result = run_candidate_score(
            run_root,
            metadata_path,
            run_root / "diagnostics/one_stage_cpp/ble_packets.csv",
            pattern_enabled=pattern_enabled,
            extra_args=[
                "--pattern-outlier-ber-threshold",
                str(cli.pattern_outlier_ber_threshold),
            ],
        )
        status["stage"] = "no_rtt_candidate_score"
        status["candidate_score"] = score_result
        write_json(run_root / "run_status.json", status)
        if not score_result["valid"]:
            raise RuntimeError("no-RTT candidate scoring failed; see results/parser_candidate_score.log")
        interval_ms = (
            cli.interval_ms
            if cli.interval_ms is not None
            else int(config.get("phone", {}).get("notification_interval_ms", 20) or 20)
        )
        covert_len = int(config.get("phone", {}).get("covert_len_bytes", 0) or 0)
        metrics = build_metrics(
            score_result["rate"],
            interval_ms,
            cli.duration_s,
            expected_events=cli.expected_events or None,
            covert_len=covert_len,
        )
        if not metrics.get("unique_seq"):
            status["zero_recovery"] = True
        status["metrics"] = {"summary": metrics, "rate": score_result["rate"]}
        write_json(run_root / "run_status.json", status)
        status["stage"] = "complete"
        status["status"] = "passed"
        status["completed_epoch_ns"] = time.time_ns()
        write_json(run_root / "run_status.json", status)
        if capture_target == "nvme":
            write_json(capture_run_root / "run_status.json", status)
        print(json.dumps(status, indent=2, sort_keys=True))
        print_tail_summary(status)
        return 0
    except Exception as exc:
        status["status"] = "failed"
        status["reason"] = str(exc)
        status["completed_epoch_ns"] = time.time_ns()
        write_json(run_root / "run_status.json", status)
        print(json.dumps(status, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
