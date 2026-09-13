#!/usr/bin/env python3
"""One-click advertising-channel (ch39) capture + parse + score experiment.

No phone is required.  The flow is the same as the phone HRS one-click runner:
capture 10 s of IQ to local NVMe, copy a clean capture to PSSD, replay it
through the read-only BLE_encrypt_check one-stage parser, score Phantom
PC-frame candidates, and print the metric table at the tail:

    指标	结果
    事件数 / 唯一 seq	500 / 489
    PSR	97.8%（489/500）
    BER	2.33%
    exact 帧率	82.6%（404/489）
    byte recovery	95.6%
    G_e2e	95.8 kbps（payload 93.5 kbps）

The B210 path uses BLE_encrypt_check's framed UHD capture binary (the
rx_samples_to_file example cannot set RX gain on B200-family devices).  The
X310 path is selected automatically when the configured ``usrp_args`` do not
name a B200/B210 device.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "x310_adv_ch39_2480_b210_20mhz.yaml"
DEFAULT_NVME_ROOT = PROJECT_ROOT / "testdata"
DEFAULT_PSSD_ROOT = Path("/path/to/PhantomChannel/testdata")
B210_CAPTURE_BIN = Path(
    "/path/to/BLE_encrypt_check/build-uhd460/uhd_b210_capture_framed_udp"
)
UHD_LIBRARY = Path("/path/to/uhd-4.6.0.0/lib")
SC16_BYTES_PER_SAMPLE = 4
RESERVE_BYTES = 512 * 1024 * 1024

sys.path.insert(0, str(PROJECT_ROOT))
from tools import run_range_bandwidth_experiment as range_runner  # noqa: E402
from tools import run_x310_bandwidth_experiment as x310_runner  # noqa: E402
from tools import run_x310_phone_hrs_experiment as phone_runner  # noqa: E402


B210_DONE_RE = re.compile(
    r"Done\.\s+samples=(\d+),\s+blocks=(\d+),\s+overflows=(\d+),\s+gaps=(\d+),\s+udp_drops=(\d+)"
)
B210_RATE_RE = re.compile(r"Actual RX rate:\s*([0-9.]+)")
B210_FREQ_RE = re.compile(r"Actual RX freq:\s*([0-9.]+)")
B210_GAIN_RE = re.compile(r"Actual RX gain:\s*([0-9.]+)")
B210_ANT_RE = re.compile(r"Actual RX antenna:\s*(\S+)")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_config(path: Path) -> dict[str, Any]:
    return range_runner.load_config(path.expanduser().resolve())


def adv_manifest(cli: argparse.Namespace, config: dict[str, Any], run_id: str) -> dict[str, Any]:
    phone = config.get("phone", {})
    firmware = config.get("firmware", {})
    benign = cli.covert_len == 0
    firmware_variant = (
        "phantomchannel_adv_broadcaster_benign"
        if benign
        else phone.get("firmware_variant", "phantomchannel_adv_broadcaster")
    )
    firmware_build_dir = (
        str(PROJECT_ROOT / "artifacts/firmware/phantomchannel_adv_broadcaster_benign")
        if benign
        else firmware.get("build_dir", "")
    )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "device_name": phone.get("device_name", "PhantomAdv"),
        "address": phone.get("address", "D1:22:33:44:55:66"),
        "firmware_variant": firmware_variant,
        "firmware_build_dir": firmware_build_dir,
        "covert_len_bytes": cli.covert_len,
        "interval_ms": cli.interval_ms,
        "rtt_available": False,
        "bluez_central_used": False,
        "jlink_used": False,
        "phone_not_required": True,
        "operator_confirmed_adv_running": bool(cli.no_prompt),
        "repetition": cli.repetition,
        "distance_m": cli.distance_m,
        "rx_antenna_label": cli.rx_antenna_label,
        "rx_antenna_model": cli.rx_antenna_model,
        "rx_antenna_orientation": cli.rx_antenna_orientation,
        "experiment_phase": cli.experiment_phase,
        "rate_test_scope": "rate_only; ch39 advertising experiment without RTT",
    }


def confirm_adv_ready(cli: argparse.Namespace) -> None:
    if cli.no_prompt:
        return
    if not sys.stdin.isatty():
        raise RuntimeError("non-interactive run requires --no-prompt after the DK is advertising")
    checklist = (
        "Preflight (no phone needed):\n"
        f"  DK is flashed and continuously advertising PhantomAdv with covert "
        f"{cli.covert_len} B / interval {cli.interval_ms} ms (delay-zero build)\n"
        "  DK antenna is near the RX antenna (B210 is weak at distance; keep within ~30 cm)\n"
        "  SDR is connected: B210 serial B210_SERIAL (default) or X310 via --config/--usrp-args\n"
        "Press Enter only after the DK is advertising and the SDR is ready."
    )
    print(checklist)
    input()


def ensure_capture_space(target_root: Path, expected_bytes: int) -> None:
    usage_root = target_root if target_root.exists() else target_root.parent
    free_bytes = shutil.disk_usage(usage_root).free
    required = expected_bytes + RESERVE_BYTES
    if free_bytes < required:
        raise RuntimeError(
            f"insufficient free space on {usage_root}: need at least {required} bytes, "
            f"have {free_bytes}"
        )


def run_b210_capture(
    args: argparse.Namespace,
    run_root: Path,
    run_id: str,
    target_name: str,
) -> dict[str, Any]:
    """Capture SC16 IQ with BLE_encrypt_check's framed UHD binary."""

    if not B210_CAPTURE_BIN.is_file():
        raise FileNotFoundError(f"B210 capture binary not found: {B210_CAPTURE_BIN}")
    iq_dir = run_root / "iq"
    iq_dir.mkdir(parents=True, exist_ok=True)
    iq_path = iq_dir / "capture.sc16"
    meta_path = iq_dir / "b210_meta.csv"
    log_path = iq_dir / "capture.log"
    metadata_path = iq_dir / "metadata.json"

    command = [
        str(B210_CAPTURE_BIN),
        "--args",
        args.usrp_args,
        "--rate",
        str(args.sample_rate_sps),
        "--freq",
        str(args.center_frequency_hz),
        "--gain",
        str(args.gain_db),
        "--ant",
        args.antenna,
        "--channel",
        str(args.channel),
        "--duration",
        str(args.duration_s),
        "--udp-mode",
        "none",
        "--iq-file",
        str(iq_path),
        "--meta-file",
        str(meta_path),
        "--save-local-iq",
        "--send-local-time",
        "--queue-blocks",
        "1024",
        "--stats-interval",
        "1",
    ]
    env = dict(__import__("os").environ)
    env["LD_LIBRARY_PATH"] = f"{UHD_LIBRARY}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")

    start_ns = time.time_ns()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        proc = subprocess.run(command, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    end_ns = time.time_ns()
    log_text = log_path.read_text(encoding="utf-8", errors="replace")

    def first(pattern: re.Pattern[str]) -> float | None:
        match = pattern.search(log_text)
        return float(match.group(1)) if match else None

    done = B210_DONE_RE.search(log_text)
    samples = int(done.group(1)) if done else None
    overflows = int(done.group(3)) if done else 0
    gaps = int(done.group(4)) if done else 0
    actual_rate = first(B210_RATE_RE)
    antenna_match = B210_ANT_RE.search(log_text)
    actual_antenna = antenna_match.group(1) if antenna_match else ""
    file_size = iq_path.stat().st_size if iq_path.is_file() else 0
    expected_bytes = samples * SC16_BYTES_PER_SAMPLE if samples is not None else None
    expected_samples = int(round((actual_rate or args.expected_sample_rate_sps) * args.duration_s))
    sample_fraction = samples / expected_samples if samples is not None and expected_samples else None

    metadata = {
        "schema_version": 1,
        "capture_id": run_id,
        "target": target_name,
        "source": "usrp_b210",
        "device_model": "B210",
        "usrp_args": args.usrp_args,
        "sample_format": "sc16_le_interleaved_iq",
        "iq_format": "int16",
        "bytes_per_complex_sample": SC16_BYTES_PER_SAMPLE,
        "timestamp_mode": "sample_index",
        "requested_sample_rate_sps": args.sample_rate_sps,
        "actual_sample_rate_sps": actual_rate,
        "expected_sample_rate_sps_for_space_check": args.expected_sample_rate_sps,
        "requested_center_frequency_hz": args.center_frequency_hz,
        "actual_center_frequency_hz": first(B210_FREQ_RE),
        "requested_bandwidth_hz": args.bandwidth_hz,
        "processing_bandwidth_hz": args.bandwidth_hz,
        "actual_rx_bandwidth_hz": args.bandwidth_hz,
        "requested_gain_db": args.gain_db,
        "actual_gain_db": first(B210_GAIN_RE),
        "antenna": args.antenna,
        "actual_antenna": actual_antenna,
        "channel": args.channel,
        "requested_duration_seconds": args.duration_s,
        "received_duration_seconds": samples / actual_rate if samples is not None and actual_rate else None,
        "samples": samples,
        "expected_samples_at_actual_rate": expected_samples,
        "sample_count_fraction_of_expected": sample_fraction,
        "file_size_bytes": file_size,
        "expected_file_size_bytes_from_received_samples": expected_bytes,
        "file_size_matches_received_samples": expected_bytes == file_size if expected_bytes is not None else False,
        "overflows": overflows,
        "gaps": gaps,
        "capture_returncode": proc.returncode,
        "capture_command": command,
        "capture_log_path": str(log_path),
        "b210_meta_csv": str(meta_path),
        "iq_path": str(iq_path),
        "run_root": str(run_root),
        "start_epoch_ns": start_ns,
        "end_epoch_ns": end_ns,
        "notes": [
            "Framed UHD B210 capture with per-block rx_time tags in b210_meta.csv; parser replay uses sample_index timestamps.",
            "Review capture.log 'Done' line for overflows/gaps; a non-zero count triggers a re-capture.",
        ],
    }
    write_json(metadata_path, metadata)

    capture_ok = (
        proc.returncode == 0
        and samples is not None
        and samples > 0
        and overflows == 0
        and gaps == 0
        and iq_path.is_file()
        and metadata["file_size_matches_received_samples"]
        and (sample_fraction is not None and sample_fraction >= args.min_sample_fraction)
    )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "target": target_name,
        "status": "passed" if capture_ok else "failed",
        "stage": "complete" if capture_ok else "validation",
        "start_epoch_ns": start_ns,
        "end_epoch_ns": end_ns,
        "duration_seconds": (end_ns - start_ns) / 1_000_000_000.0,
        "command": command,
        "capture": metadata,
        "overflow_text_matches": ["overflows>0"] if overflows else [],
        "gap_text_matches": ["gaps>0"] if gaps else [],
        "reason": "" if capture_ok else "B210 capture validation failed; IQ retained on target",
    }


def print_tail_summary(status: dict[str, Any]) -> None:
    metrics = (status.get("metrics") or {}).get("summary") or {}
    run_id = status.get("run_id", "")
    events = metrics.get("events")
    unique = metrics.get("unique_seq")
    psr_exact = metrics.get("psr_exact")
    exact = metrics.get("exact_packets")
    compared = metrics.get("compared_packets")
    byte_recovery = metrics.get("byte_recovery")
    g_e2e = metrics.get("g_e2e_kbps")
    payload = metrics.get("payload_kbps")

    if events is None or unique is None:
        return
    print(f"\n=== 指标汇总 (run_id={run_id}) ===")
    print("指标\t结果")
    print(f"事件数 / 唯一 seq\t{events} / {unique}")
    if not unique:
        print("⚠ 未恢复任何 Phantom 帧：检查天线连接、SDR 是否在线、DK 是否在广播")
        return
    for line in phone_runner.psr_ber_metric_lines(metrics):
        print(line)
    if isinstance(psr_exact, (int, float)) and exact is not None and compared is not None:
        print(f"exact 帧率\t{psr_exact:.1%}（{exact}/{compared}）")
    if isinstance(byte_recovery, (int, float)):
        print(f"byte recovery\t{byte_recovery:.1%}")
    if isinstance(g_e2e, (int, float)):
        payload_text = f"{payload:.1f}" if isinstance(payload, (int, float)) else "N/A"
        print(f"G_e2e\t{g_e2e:.1f} kbps（payload {payload_text} kbps）")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--capture-id", default="")
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--interval-ms", type=int, default=None, help="advertising interval in ms (must match the flashed firmware); default from config")
    parser.add_argument("--covert-len", type=int, default=None, help="covert payload length in bytes (must match the flashed firmware); default from config")
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--distance-m", type=float, default=None, help="physical TX-RX separation recorded in the manifest")
    parser.add_argument("--rx-antenna-label", default="", help="operator label, e.g. default, sas571, or hyperlog4060")
    parser.add_argument("--rx-antenna-model", default="", help="receive-antenna model recorded in the manifest")
    parser.add_argument("--rx-antenna-orientation", default="", help="boresight/polarization note recorded in the manifest")
    parser.add_argument("--experiment-phase", choices=("", "prescan", "formal"), default="", help="directional-antenna experiment phase")
    parser.add_argument("--gain-db", type=float, default=None, help="SDR RX gain in dB (B210 max 50; default from config)")
    parser.add_argument(
        "--ble-threshold",
        type=float,
        default=None,
        help="BLE parser detection threshold (default from config parser.ble_threshold)",
    )
    parser.add_argument("--usrp-args", default="", help="override the configured USRP args (e.g. X310 x300 args)")
    parser.add_argument("--expected-events", type=int, default=0, help="override the PSR denominator (default: round(window x 1000/interval_ms))")
    parser.add_argument(
        "--pattern-outlier-ber-threshold",
        type=float,
        default=0.10,
        help="per-packet BER above which a packet is a collision outlier excluded "
        "from BER（剔除冲突包）; default: 0.10",
    )
    parser.add_argument("--no-prompt", action="store_true", help="skip the preflight confirmation prompt")
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
    if cli.distance_m is not None and cli.distance_m <= 0:
        raise SystemExit("--distance-m must be positive")
    if not 0 < cli.min_sample_fraction <= 1:
        raise SystemExit("--min-sample-fraction must be in (0, 1]")

    config = load_config(cli.config)
    phone_cfg = config.get("phone", {})
    cli.interval_ms = cli.interval_ms if cli.interval_ms is not None else int(phone_cfg.get("notification_interval_ms", 20))
    cli.covert_len = cli.covert_len if cli.covert_len is not None else int(phone_cfg.get("covert_len_bytes", 239))
    if not 0 <= cli.covert_len <= 239:
        raise SystemExit("--covert-len must be in 0..239 (0 is the benign label)")
    if not 20 <= cli.interval_ms <= 10000:
        raise SystemExit("--interval-ms must be in 20..10000")
    if cli.gain_db is not None and cli.gain_db > 50:
        print("⚠ --gain-db 超过 B210 常用上限 50 dB，将被 UHD 钳制", file=sys.stderr)

    run_id = cli.capture_id or (
        f"{time.strftime('%Y%m%d_%H%M%S')}_adv_ch39_2480_len{cli.covert_len}"
        f"_rep{cli.repetition}"
    )
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

    confirm_adv_ready(cli)
    capture_args = phone_runner.build_capture_args(cli, config, run_id, capture_target)
    capture_args.min_sample_fraction = cli.min_sample_fraction
    if cli.usrp_args:
        capture_args.usrp_args = cli.usrp_args
    manifest = adv_manifest(cli, config, run_id)
    plan = {
        "run_id": run_id,
        "capture_target": capture_target,
        "run_root_nvme": str(nvme_root / run_id),
        "run_root_pssd": str(pssd_root / run_id),
        "adv": manifest,
        "capture": {
            "duration_s": cli.duration_s,
            "requested_sample_rate_sps": capture_args.sample_rate_sps,
            "requested_bandwidth_hz": capture_args.bandwidth_hz,
            "center_frequency_hz": capture_args.center_frequency_hz,
            "gain_db": capture_args.gain_db,
            "usrp_args": capture_args.usrp_args,
            "ble_threshold": (
                cli.ble_threshold
                if cli.ble_threshold is not None
                else float(config.get("parser", {}).get("ble_threshold", 0.02))
            ),
            "antenna": capture_args.antenna,
            "channel": capture_args.channel,
        },
        "expected_events": cli.expected_events or None,
        "pattern_outlier_ber_threshold": cli.pattern_outlier_ber_threshold,
        "flow": (
            [
                "preflight",
                "pssd_capture",
                "capture_validation",
                "one_stage_cpp_parse",
                "no_rtt_candidate_score",
            ]
            if capture_target == "pssd"
            else [
                "preflight",
                "nvme_capture",
                "capture_validation",
                "pssd_copy",
                "nvme_iq_cleanup" if not cli.keep_nvme_iq else "nvme_iq_retained",
                "one_stage_cpp_parse",
                "no_rtt_candidate_score",
            ]
        ),
    }
    if cli.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    status: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "stage": "capture",
        "created_epoch_ns": time.time_ns(),
        "adv": manifest,
        "plan": plan,
        "paths": {"run_root_nvme": str(capture_run_root), "run_root_pssd": str(pssd_run_root)},
    }
    run_root = capture_run_root
    try:
        ensure_capture_space(
            capture_root,
            int(capture_args.expected_sample_rate_sps * cli.duration_s * SC16_BYTES_PER_SAMPLE),
        )
        usrp_args = capture_args.usrp_args.lower()
        b210_backend = ("b200" in usrp_args) or ("b210" in usrp_args)
        if b210_backend:
            run_root.mkdir(parents=True, exist_ok=True)
            write_json(run_root / "adv_manifest.json", manifest)
            write_json(run_root / "run_status.json", status)
            capture_status = run_b210_capture(capture_args, run_root, run_id, capture_target)
        else:
            capture_status = x310_runner.run_target(
                capture_args,
                x310_runner.Target(capture_target, capture_root),
                run_id,
            )
        status["capture"] = capture_status
        status["stage"] = "capture_validation"
        run_root = capture_run_root
        write_json(run_root / "adv_manifest.json", manifest)
        write_json(run_root / "run_status.json", status)
        if capture_status.get("status") != "passed":
            raise RuntimeError(
                f"capture validation failed; IQ retained on {capture_target.upper()}"
            )
        overflow_lines = (
            capture_status.get("overflow_text_matches") or []
            if b210_backend
            else phone_runner.detect_overflow_indications(capture_status)
        )
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
            copy_manifest = phone_runner.copy_to_pssd(
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
            else float(config.get("parser", {}).get("ble_threshold", 0.02))
        )
        parser_result = phone_runner.run_one_stage_parser(
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

        score_result = phone_runner.run_candidate_score(
            run_root,
            metadata_path,
            run_root / "diagnostics/one_stage_cpp/ble_packets.csv",
            pattern_enabled=True,
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

        rate = score_result["rate"]
        metrics = phone_runner.build_metrics(
            rate,
            cli.interval_ms,
            cli.duration_s,
            expected_events=cli.expected_events or None,
            covert_len=cli.covert_len,
        )
        if not metrics.get("unique_seq"):
            status["zero_recovery"] = True
        status["metrics"] = {"summary": metrics, "rate": rate}
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
