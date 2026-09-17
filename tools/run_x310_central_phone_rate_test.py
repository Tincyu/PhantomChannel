#!/usr/bin/env python3
"""Run one 10-second X310 rate test for central -> phone GATT server.

The phone must advertise the PhantomSink GATT service before the flashed
nRF52840 central starts. IQ is captured to local NVMe, parsed with the CUDA/C++
BLE pipeline, scored without RTT ground truth, copied and verified on PSSD, and
then the validated NVMe IQ file is removed. The existing phone HRS runner and
the bundled PhantomChannel receiver are left untouched.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
from portable_paths import NCS_WORKSPACE  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "x310_central_phone_rate_experiment.yaml"
DEFAULT_NVME_ROOT = PROJECT_ROOT / "testdata"
DEFAULT_PSSD_ROOT = Path("/path/to/PhantomChannel/testdata")
CENTRAL_PRJ_CONF = NCS_WORKSPACE / "zephyr/samples/bluetooth/phantomchannel_central_gatt_write/prj.conf"

sys.path.insert(0, str(PROJECT_ROOT))
from tools import run_x310_phone_hrs_experiment as base_runner  # noqa: E402


def read_symbol(path: Path, name: str) -> str | None:
    text = path.read_text(encoding="utf-8")
    match = re.search(rf"^{re.escape(name)}=(.*)$", text, re.MULTILINE)
    return match.group(1) if match else None


def require_int_symbol(path: Path, name: str) -> int:
    value = read_symbol(path, name)
    if value is None:
        raise RuntimeError(f"{name} is missing from {path}")
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} in {path} is not an integer: {value!r}") from exc


def configured_settings(config: dict[str, Any]) -> dict[str, Any]:
    phone = config.get("phone", {})
    firmware = config.get("firmware", {})
    source_conf = Path(firmware.get("prj_conf", CENTRAL_PRJ_CONF)).expanduser().resolve()
    build_root = Path(firmware.get("build_dir", "")).expanduser().resolve()
    build_conf = build_root / "phantomchannel_central_gatt_write/zephyr/.config"

    if not source_conf.is_file():
        raise FileNotFoundError(f"central prj.conf not found: {source_conf}")
    if not build_conf.is_file():
        raise FileNotFoundError(
            f"central build .config not found: {build_conf}; run the central build/flash script first"
        )

    source_len = require_int_symbol(source_conf, "CONFIG_PHANTOMCHANNEL_COVERT_LEN")
    source_interval = require_int_symbol(source_conf, "CONFIG_PHANTOMCHANNEL_WRITE_INTERVAL_MS")
    build_len = require_int_symbol(build_conf, "CONFIG_PHANTOMCHANNEL_COVERT_LEN")
    build_interval = require_int_symbol(build_conf, "CONFIG_PHANTOMCHANNEL_WRITE_INTERVAL_MS")
    yaml_len = int(phone.get("covert_len_bytes", -1))
    yaml_interval = int(phone.get("write_interval_ms", -1))
    if source_len != yaml_len or source_interval != yaml_interval:
        raise RuntimeError(
            "central source/config mismatch: "
            f"prj.conf=({source_len} B, {source_interval} ms), "
            f"YAML=({yaml_len} B, {yaml_interval} ms)"
        )
    if source_len != build_len or source_interval != build_interval:
        raise RuntimeError(
            "central source/build mismatch: "
            f"source=({source_len} B, {source_interval} ms), "
            f"build=({build_len} B, {build_interval} ms); rebuild/reflash central first"
        )

    expected_symbols = {
        "CONFIG_PHANTOMCHANNEL_CENTRAL_TX": "y",
        "CONFIG_PHANTOMCHANNEL_EMBED_ENABLE": "y",
        "CONFIG_PHANTOMCHANNEL_DYNAMIC_TIMING": "y",
        "CONFIG_PHANTOMCHANNEL_RX_GUARD_US": str(int(phone.get("rx_guard_us", 0))),
        "CONFIG_BT_CENTRAL": "y",
    }
    for name, expected in expected_symbols.items():
        actual = read_symbol(build_conf, name)
        if actual is None and expected == "n":
            actual = "n"
        if actual != expected:
            raise RuntimeError(
                f"central build mismatch for {name}: expected {expected!r}, got {actual!r}; "
                "rebuild/reflash central first"
            )

    return {
        "covert_len_bytes": source_len,
        "write_interval_ms": source_interval,
        "source_prj_conf": str(source_conf),
        "build_config": str(build_conf),
        "build_dir": str(build_root),
    }


def central_manifest(
    cli: argparse.Namespace,
    config: dict[str, Any],
    settings: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    phone = config.get("phone", {})
    firmware = config.get("firmware", {})
    return {
        "schema_version": 1,
        "run_id": run_id,
        "device_name": phone.get("device_name", "PhantomSink"),
        "phone_role": phone.get("role", "gatt_server"),
        "service_uuid": phone.get("service_uuid", ""),
        "characteristic_uuid": phone.get("characteristic_uuid", ""),
        "firmware_variant": phone.get("firmware_variant", "phantomchannel_central_gatt_write"),
        "central_target_mode": phone.get("target_mode", "phone-nearest"),
        "firmware_build_dir": firmware.get("build_dir", ""),
        "firmware_prj_conf": firmware.get("prj_conf", ""),
        "covert_len_bytes": settings["covert_len_bytes"],
        "write_interval_ms": settings["write_interval_ms"],
        "dynamic_timing": bool(phone.get("dynamic_timing", True)),
        "rx_guard_us": int(phone.get("rx_guard_us", 0)),
        "phone_model": cli.phone_model,
        "phone_os": cli.phone_os,
        "phone_app": cli.phone_app,
        "phone_ready_confirmed": bool(cli.phone_ready),
        "operator_confirmed_phone_gatt_server": bool(cli.phone_ready),
        "rtt_available": False,
        "bluez_central_used": False,
        "jlink_used": False,
        "rate_test_scope": "central_phone_rate_only; distance experiment not executed",
    }


def confirm_phone_ready(cli: argparse.Namespace, config: dict[str, Any]) -> None:
    if cli.phone_ready:
        return
    if not sys.stdin.isatty():
        raise RuntimeError(
            "non-interactive run requires --phone-ready after PhantomSink is advertising"
        )
    phone = config.get("phone", {})
    print(
        "Central-phone ready gate:\n"
        f"  Start the phone GATT server as {phone.get('device_name', 'PhantomSink')}\n"
        f"  Advertise service {phone.get('service_uuid', '0000ff00-0000-1000-8000-00805f9b34fb')}\n"
        f"  Provide Write Without Response characteristic {phone.get('characteristic_uuid', '0000ff01-0000-1000-8000-00805f9b34fb')}\n"
        "  Keep the phone advertising and do not connect it to another central\n"
        "  Confirm that the nRF52840 central has been flashed and is ready to connect\n"
        "Press Enter only after all checks are complete."
    )
    input()


def update_retention_metadata(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "nvme_source_retained" in data:
        data["nvme_source_retained"] = False
    if isinstance(data.get("copy"), dict) and "nvme_source_retained" in data["copy"]:
        data["copy"]["nvme_source_retained"] = False
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def remove_nvme_iq(run_root: Path, pssd_run: Path) -> None:
    manifest_path = pssd_run / "copy_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("verified", False):
        raise RuntimeError("refusing to remove NVMe IQ because PSSD copy is not verified")
    iq_path = run_root / "iq/capture.sc16"
    if not iq_path.is_file():
        raise RuntimeError(f"expected NVMe IQ is missing: {iq_path}")
    iq_path.unlink()
    update_retention_metadata(run_root / "copy_manifest.json")
    update_retention_metadata(run_root / "run_status.json")
    update_retention_metadata(pssd_run / "copy_manifest.json")
    update_retention_metadata(pssd_run / "run_status.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--capture-id", default="")
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--phone-ready", action="store_true")
    parser.add_argument("--phone-model", default="unspecified")
    parser.add_argument("--phone-os", default="unspecified")
    parser.add_argument("--phone-app", default="unspecified-gatt-server")
    parser.add_argument("--nvme-root", type=Path, default=DEFAULT_NVME_ROOT)
    parser.add_argument("--pssd-root", type=Path, default=DEFAULT_PSSD_ROOT)
    parser.add_argument("--min-sample-fraction", type=float, default=0.99)
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

    config = base_runner.load_config(cli.config.expanduser().resolve())
    settings = configured_settings(config)
    run_id = cli.capture_id or (
        f"{time.strftime('%Y%m%d_%H%M%S')}_x310_central_phone_80m_rate_rep{cli.repetition}"
    )
    nvme_root = cli.nvme_root.expanduser().resolve()
    pssd_root = cli.pssd_root.expanduser().resolve()
    run_root = nvme_root / run_id
    pssd_run = pssd_root / run_id
    if run_root.exists():
        raise SystemExit(f"NVMe run directory already exists: {run_root}")
    capture_args = base_runner.build_capture_args(cli, config, run_id)
    manifest = central_manifest(cli, config, settings, run_id)
    plan = {
        "run_id": run_id,
        "run_root_nvme": str(run_root),
        "run_root_pssd": str(pssd_run),
        "phone": manifest,
        "capture": {
            "duration_s": cli.duration_s,
            "requested_sample_rate_sps": capture_args.sample_rate_sps,
            "requested_bandwidth_hz": capture_args.bandwidth_hz,
            "center_frequency_hz": capture_args.center_frequency_hz,
            "antenna": capture_args.antenna,
            "channel": capture_args.channel,
        },
        "flow": [
            "phone_gatt_server_ready",
            "nvme_capture",
            "capture_validation",
            "one_stage_cpp_parse_cuda",
            "no_rtt_candidate_score",
            "pssd_copy_verify",
            "remove_nvme_iq",
        ],
        "distance_experiment_executed": False,
    }
    if cli.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    confirm_phone_ready(cli, config)

    status: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "stage": "x310_capture",
        "created_epoch_ns": time.time_ns(),
        "phone": manifest,
        "plan": plan,
        "paths": {"run_root_nvme": str(run_root), "run_root_pssd": str(pssd_run)},
    }
    try:
        capture_status = base_runner.x310_runner.run_target(
            capture_args, base_runner.x310_runner.Target("nvme", nvme_root), run_id
        )
        status["capture"] = capture_status
        run_root.mkdir(parents=True, exist_ok=True)
        base_runner.write_json(run_root / "phone_manifest.json", manifest)
        base_runner.write_json(run_root / "run_status.json", status)
        if capture_status.get("status") != "passed":
            raise RuntimeError("X310 capture validation failed; IQ retained on NVMe")

        metadata_path = run_root / "iq/metadata.json"
        iq_path = run_root / "iq/capture.sc16"
        parser_result = base_runner.run_one_stage_parser(
            capture_args, run_root, metadata_path, iq_path
        )
        status["stage"] = "one_stage_cpp_parse_cuda"
        status["parser"] = parser_result
        base_runner.write_json(run_root / "run_status.json", status)
        if not parser_result["valid"]:
            raise RuntimeError("one-stage parser failed; see diagnostics/one_stage_cpp/parse.log")

        score_result = base_runner.run_candidate_score(
            run_root, metadata_path, run_root / "diagnostics/one_stage_cpp/ble_packets.csv"
        )
        status["stage"] = "no_rtt_candidate_score"
        status["candidate_score"] = score_result
        base_runner.write_json(run_root / "run_status.json", status)
        if not score_result["valid"]:
            raise RuntimeError("candidate scoring failed; see results/parser_candidate_score.log")

        copy_manifest = base_runner.copy_and_verify(run_root, pssd_root, run_id)
        status["copy"] = copy_manifest
        if not copy_manifest["verified"]:
            raise RuntimeError("PSSD copy verification failed; NVMe IQ retained")

        status["stage"] = "remove_nvme_iq"
        base_runner.write_json(run_root / "run_status.json", status)
        base_runner.write_json(pssd_run / "run_status.json", status)
        remove_nvme_iq(run_root, pssd_run)

        status["copy"]["nvme_source_retained"] = False
        status["stage"] = "complete"
        status["status"] = "passed"
        status["completed_epoch_ns"] = time.time_ns()
        base_runner.write_json(run_root / "run_status.json", status)
        base_runner.write_json(pssd_run / "run_status.json", status)
        rate = json.loads(
            (pssd_run / "results/parser_candidate_rate.json").read_text(encoding="utf-8")
        )
        print(json.dumps(status, indent=2, sort_keys=True))
        print(
            "iq_window_parser_candidate_data_kbps="
            f"{rate['iq_window_parser_candidate_data_kbps']:.6f} "
            "(one full PC frame per unique seq)"
        )
        print(f"rate_denominator={rate['rate_denominator']}")
        print(f"seq_unique_count={rate['seq_unique_count']}")
        print(f"parser_candidate_pc_frame_bytes={rate['parser_candidate_pc_frame_bytes']}")
        print(
            "parser_candidate_pc_frame_bytes_all_deduplicated_candidates="
            f"{rate['parser_candidate_pc_frame_bytes_all_deduplicated_candidates']} (diagnostic)"
        )
        print(
            "iq_window_parser_candidate_payload_data_kbps="
            f"{rate['iq_window_parser_candidate_payload_data_kbps']:.6f} (diagnostic)"
        )
        print(f"parser_candidates_deduplicated={rate['parser_candidates_deduplicated']}")
        print(f"integrity_valid_candidates={rate['integrity_valid_candidates']}")
        print(f"rate_json={pssd_run / 'results/parser_candidate_rate.json'}")
        return 0
    except Exception as exc:
        status["status"] = "failed"
        status["reason"] = str(exc)
        status["completed_epoch_ns"] = time.time_ns()
        if run_root.exists():
            base_runner.write_json(run_root / "run_status.json", status)
        print(json.dumps(status, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
