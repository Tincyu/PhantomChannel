#!/usr/bin/env python3
"""One-command 10-second phone HRS IQ capture, parse, and rate test.

The phone must already be connected to PhantomHRS with HRS 0x2A37
notifications enabled.  The command itself is non-interactive and assumes
that phone-ready gate has been satisfied.  IQ is copied to PSSD and the
validated NVMe source IQ is then removed.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
from portable_paths import NCS_WORKSPACE  # noqa: E402

RUNNER = PROJECT_ROOT / "tools/run_x310_phone_hrs_experiment.py"
CONFIG = PROJECT_ROOT / "configs/x310_phone_hrs_distance_experiment.yaml"
FIRMWARE_CONF = NCS_WORKSPACE / "zephyr/samples/bluetooth/phantomchannel_hrs_peripheral/prj.conf"
NVME_ROOT = PROJECT_ROOT / "testdata"
PSSD_ROOT = Path("/path/to/PhantomChannel/testdata")
BUILD_CONFIG = PROJECT_ROOT / (
    "artifacts/firmware/phantomchannel_hrs_peripheral/"
    "phantomchannel_hrs_peripheral/zephyr/.config"
)


def configured_settings() -> tuple[int, int, int, int, int, int]:
    firmware_text = FIRMWARE_CONF.read_text(encoding="utf-8")
    config_text = CONFIG.read_text(encoding="utf-8")
    firmware_match = re.search(r"^CONFIG_PHANTOMCHANNEL_COVERT_LEN=(\d+)$", firmware_text, re.MULTILINE)
    config_match = re.search(r"^  covert_len_bytes: (\d+)$", config_text, re.MULTILINE)
    interval_firmware_match = re.search(
        r"^CONFIG_PHANTOMCHANNEL_NOTIFY_INTERVAL_MS=(\d+)$", firmware_text, re.MULTILINE
    )
    interval_config_match = re.search(
        r"^  notification_interval_ms: (\d+)$", config_text, re.MULTILINE
    )
    build_text = BUILD_CONFIG.read_text(encoding="utf-8") if BUILD_CONFIG.is_file() else ""
    build_len_match = re.search(
        r"^CONFIG_PHANTOMCHANNEL_COVERT_LEN=(\d+)$", build_text, re.MULTILINE
    )
    build_interval_match = re.search(
        r"^CONFIG_PHANTOMCHANNEL_NOTIFY_INTERVAL_MS=(\d+)$", build_text, re.MULTILINE
    )
    if (
        firmware_match is None
        or config_match is None
        or interval_firmware_match is None
        or interval_config_match is None
        or build_len_match is None
        or build_interval_match is None
    ):
        raise RuntimeError("cannot determine configured covert payload length/interval")
    return (
        int(firmware_match.group(1)),
        int(config_match.group(1)),
        int(interval_firmware_match.group(1)),
        int(interval_config_match.group(1)),
        int(build_len_match.group(1)),
        int(build_interval_match.group(1)),
    )


def update_retention_metadata(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "nvme_source_retained" in data:
        data["nvme_source_retained"] = False
    if isinstance(data.get("copy"), dict) and "nvme_source_retained" in data["copy"]:
        data["copy"]["nvme_source_retained"] = False
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def remove_nvme_iq(run_root: Path, pssd_root: Path) -> None:
    pssd_manifest = pssd_root / "copy_manifest.json"
    manifest = json.loads(pssd_manifest.read_text(encoding="utf-8"))
    if not manifest.get("verified", False):
        raise RuntimeError("refusing to remove NVMe IQ because PSSD copy is not verified")
    nvme_iq = run_root / "iq/capture.sc16"
    if not nvme_iq.is_file():
        raise RuntimeError(f"expected NVMe IQ is missing: {nvme_iq}")
    nvme_iq.unlink()
    update_retention_metadata(run_root / "copy_manifest.json")
    update_retention_metadata(run_root / "run_status.json")
    update_retention_metadata(pssd_root / "copy_manifest.json")
    update_retention_metadata(pssd_root / "run_status.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--capture-id", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    (
        firmware_len,
        config_len,
        firmware_interval_ms,
        config_interval_ms,
        build_len,
        build_interval_ms,
    ) = configured_settings()
    if firmware_len != config_len or firmware_interval_ms != config_interval_ms:
        raise SystemExit(
            f"firmware/config mismatch: firmware=({firmware_len} B, {firmware_interval_ms} ms), "
            f"config=({config_len} B, {config_interval_ms} ms); "
            "run tools/program_phone_hrs_firmware.py first"
        )
    if firmware_len != build_len or firmware_interval_ms != build_interval_ms:
        raise SystemExit(
            f"source/build mismatch: source=({firmware_len} B, {firmware_interval_ms} ms), "
            f"build=({build_len} B, {build_interval_ms} ms); "
            "run tools/program_phone_hrs_firmware.py first"
        )
    run_id = args.capture_id or (
        f"{time.strftime('%Y%m%d_%H%M%S')}_x310_phonehrs_80m_rate_rep{args.repetition}"
    )
    command = [
        sys.executable,
        str(RUNNER),
        "--config",
        str(CONFIG),
        "--capture-id",
        run_id,
        "--duration-s",
        str(args.duration_s),
        "--phone-ready",
        "--repetition",
        str(args.repetition),
        "--nvme-root",
        str(NVME_ROOT),
        "--pssd-root",
        str(PSSD_ROOT),
    ]
    if args.dry_run:
        print(" ".join(command))
        print(f"configured_covert_len={firmware_len} B")
        print(f"configured_notification_interval_ms={firmware_interval_ms}")
        return 0

    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        return result.returncode

    nvme_run = NVME_ROOT / run_id
    pssd_run = PSSD_ROOT / run_id
    remove_nvme_iq(nvme_run, pssd_run)
    rate = json.loads(
        (pssd_run / "results/parser_candidate_rate.json").read_text(encoding="utf-8")
    )
    print(f"run_id={run_id}")
    print(f"covert_len={firmware_len} B; PC frame=6+{firmware_len} B (full-frame rate)")
    print(f"notification_interval_ms={firmware_interval_ms}")
    print(f"iq_window_parser_candidate_data_kbps={rate['iq_window_parser_candidate_data_kbps']:.6f}")
    print(f"rate_denominator={rate['rate_denominator']}")
    print(f"seq_unique_count={rate['seq_unique_count']}")
    print(f"parser_candidates_deduplicated={rate['parser_candidates_deduplicated']}")
    print(f"integrity_valid_candidates={rate['integrity_valid_candidates']}")
    print(f"rate_json={pssd_run / 'results/parser_candidate_rate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
