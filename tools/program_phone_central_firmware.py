#!/usr/bin/env python3
"""Build and flash the independent PhantomChannel central firmware.

This script intentionally operates only on the
``phantomchannel_central_gatt_write`` sample.  It does not modify the HRS
peripheral sample, BLE_encrypt_check, or the X310 test configuration.  The
central sample defaults to symmetric 2M PHY and can be selected for symmetric
1M comparison mode before it starts Phantom writes.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
from portable_paths import NCS_TOOLCHAIN as TOOLCHAIN_ROOT, NCS_WORKSPACE as SDK_ROOT  # noqa: E402

SAMPLE_DIR = SDK_ROOT / "zephyr/samples/bluetooth/phantomchannel_central_gatt_write"
BUILD_DIR = PROJECT_ROOT / "artifacts/firmware/phantomchannel_central_gatt_write"
PRJ_CONF = SAMPLE_DIR / "prj.conf"
VARIANT_CONFIG_DIR = PROJECT_ROOT / "configs"
WEST_PYTHON = TOOLCHAIN_ROOT / "usr/local/bin/python3"
ZEPHYR_SDK = TOOLCHAIN_ROOT / "opt/zephyr-sdk"
DEFAULT_SERIAL = "1050216757"

MIN_INTERVAL_MS = 20
MAX_INTERVAL_MS = 5000
MIN_COVERT_LEN = 0
MAX_COVERT_LEN = 234
MIN_RX_GUARD_US = 0
MAX_RX_GUARD_US = 32

VARIANTS = {
    "phone-nearest": {
        "directory": "phone_nearest",
        "overlay": VARIANT_CONFIG_DIR / "central_phone_nearest.conf",
        "fixed_target": "n",
        "description": "scan for one second and connect to the strongest connectable advertiser",
    },
    "sink-52833-fixed": {
        "directory": "sink_52833_fixed",
        "overlay": VARIANT_CONFIG_DIR / "central_sink_52833_fixed.conf",
        "fixed_target": "y",
        "description": "connect directly to static random address C0:DE:52:83:00:33",
    },
}


def replace_once(path: Path, pattern: str, replacement: str) -> None:
    text = path.read_text()
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"Could not update exactly one setting in {path}: {pattern}")
    path.write_text(updated)


def validate_settings(covert_len: int, interval_ms: int, rx_guard_us: int = 0) -> None:
    if not MIN_COVERT_LEN <= covert_len <= MAX_COVERT_LEN:
        raise ValueError(
            f"--covert-len must be {MIN_COVERT_LEN}..{MAX_COVERT_LEN} B, got {covert_len}"
        )
    if not MIN_INTERVAL_MS <= interval_ms <= MAX_INTERVAL_MS:
        raise ValueError(
            f"--interval-ms must be {MIN_INTERVAL_MS}..{MAX_INTERVAL_MS} ms, got {interval_ms}"
        )
    if not MIN_RX_GUARD_US <= rx_guard_us <= MAX_RX_GUARD_US:
        raise ValueError(
            f"--rx-guard-us must be {MIN_RX_GUARD_US}..{MAX_RX_GUARD_US} us, got {rx_guard_us}"
        )


def set_central_settings(
    covert_len: int,
    interval_ms: int,
    embed_enable: str,
    dynamic_timing: str,
    rx_guard_us: int,
    phy: str,
) -> None:
    validate_settings(covert_len, interval_ms, rx_guard_us)
    if not PRJ_CONF.is_file():
        raise FileNotFoundError(f"Central project configuration not found: {PRJ_CONF}")

    replace_once(
        PRJ_CONF,
        r"^CONFIG_PHANTOMCHANNEL_COVERT_LEN=.*$",
        f"CONFIG_PHANTOMCHANNEL_COVERT_LEN={covert_len}",
    )
    replace_once(
        PRJ_CONF,
        r"^CONFIG_PHANTOMCHANNEL_WRITE_INTERVAL_MS=.*$",
        f"CONFIG_PHANTOMCHANNEL_WRITE_INTERVAL_MS={interval_ms}",
    )
    replace_once(
        PRJ_CONF,
        r"^CONFIG_PHANTOMCHANNEL_EMBED_ENABLE=.*$",
        f"CONFIG_PHANTOMCHANNEL_EMBED_ENABLE={embed_enable}",
    )
    replace_once(
        PRJ_CONF,
        r"^CONFIG_PHANTOMCHANNEL_DYNAMIC_TIMING=.*$",
        f"CONFIG_PHANTOMCHANNEL_DYNAMIC_TIMING={dynamic_timing}",
    )
    replace_once(
        PRJ_CONF,
        r"^CONFIG_PHANTOMCHANNEL_RX_GUARD_US=.*$",
        f"CONFIG_PHANTOMCHANNEL_RX_GUARD_US={rx_guard_us}",
    )
    replace_once(
        PRJ_CONF,
        r"^CONFIG_PHANTOMCHANNEL_FORCE_2M=.*$",
        f"CONFIG_PHANTOMCHANNEL_FORCE_2M={'y' if phy == '2m' else 'n'}",
    )
    replace_once(
        PRJ_CONF,
        r"^CONFIG_PHANTOMCHANNEL_FORCE_1M=.*$",
        f"CONFIG_PHANTOMCHANNEL_FORCE_1M={'y' if phy == '1m' else 'n'}",
    )
    replace_once(
        PRJ_CONF,
        r"^CONFIG_BT_AUTO_PHY_UPDATE=.*$",
        f"CONFIG_BT_AUTO_PHY_UPDATE={'y' if phy == '2m' else 'n'}",
    )


def build_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["ZEPHYR_SDK_INSTALL_DIR"] = str(ZEPHYR_SDK)
    environment["PATH"] = os.pathsep.join(
        [str(TOOLCHAIN_ROOT / "usr/local/bin"), "/usr/bin", "/bin"]
    )
    python_packages = TOOLCHAIN_ROOT / "usr/local/lib/python3.12/site-packages"
    environment["PYTHONPATH"] = str(python_packages)
    return environment


def build_firmware(target_mode: str) -> Path:
    if not WEST_PYTHON.is_file():
        raise FileNotFoundError(f"Toolchain Python not found: {WEST_PYTHON}")
    if not SAMPLE_DIR.is_dir():
        raise FileNotFoundError(f"Central sample not found: {SAMPLE_DIR}")
    variant = VARIANTS[target_mode]
    overlay = variant["overlay"]
    if not overlay.is_file():
        raise FileNotFoundError(f"Central variant overlay not found: {overlay}")
    variant_build_dir = BUILD_DIR / variant["directory"]

    command = [
        str(WEST_PYTHON),
        "-m",
        "west",
        "build",
        "-p",
        "always",
        "-b",
        "nrf52840dk/nrf52840",
        "-d",
        str(variant_build_dir),
        str(SAMPLE_DIR),
        "--",
        f"-DZEPHYR_SDK_INSTALL_DIR={ZEPHYR_SDK}",
        f"-DOVERLAY_CONFIG={overlay}",
    ]
    print(f"Building central firmware variant {target_mode} ({variant['description']}):")
    print(" ".join(command))
    subprocess.run(command, cwd=SDK_ROOT, env=build_environment(), check=True)

    firmware = variant_build_dir / "merged.hex"
    if not firmware.is_file() or firmware.stat().st_size == 0:
        raise RuntimeError(f"Build completed without a usable firmware image: {firmware}")
    return firmware


def read_config_value(config: str, name: str) -> str | None:
    match = re.search(rf"^{re.escape(name)}=(.*)$", config, flags=re.MULTILINE)
    return match.group(1) if match else None


def verify_build(
    target_mode: str,
    covert_len: int,
    interval_ms: int,
    embed_enable: str,
    dynamic_timing: str,
    rx_guard_us: int,
    phy: str,
) -> None:
    variant = VARIANTS[target_mode]
    variant_build_dir = BUILD_DIR / variant["directory"]
    config_path = variant_build_dir / "phantomchannel_central_gatt_write/zephyr/.config"
    if not config_path.is_file():
        raise RuntimeError(f"Generated Kconfig file not found: {config_path}")
    config = config_path.read_text()

    required = {
        "CONFIG_PHANTOMCHANNEL_CENTRAL_TX": "y",
        "CONFIG_PHANTOMCHANNEL_COVERT_LEN": str(covert_len),
        "CONFIG_PHANTOMCHANNEL_WRITE_INTERVAL_MS": str(interval_ms),
        "CONFIG_BT_CENTRAL": "y",
        "CONFIG_BT_PHY_UPDATE": "y",
        "CONFIG_BT_USER_PHY_UPDATE": "y",
        "CONFIG_BT_CTLR_PHY": "y",
        "CONFIG_BT_CTLR_PHY_2M": "y",
        "CONFIG_BT_CTLR_DATA_LENGTH_MAX": "251",
        "CONFIG_PHANTOMCHANNEL_FIXED_TARGET": variant["fixed_target"],
        "CONFIG_PHANTOMCHANNEL_EMBED_ENABLE": embed_enable,
        "CONFIG_PHANTOMCHANNEL_DYNAMIC_TIMING": dynamic_timing,
        "CONFIG_PHANTOMCHANNEL_RX_GUARD_US": str(rx_guard_us),
        "CONFIG_PHANTOMCHANNEL_FORCE_2M": "y" if phy == "2m" else "n",
        "CONFIG_PHANTOMCHANNEL_FORCE_1M": "y" if phy == "1m" else "n",
        "CONFIG_BT_AUTO_PHY_UPDATE": "y" if phy == "2m" else "n",
    }
    for name, expected in required.items():
        actual = read_config_value(config, name)
        # Kconfig omits boolean symbols whose final value is n.
        if name in {
            "CONFIG_PHANTOMCHANNEL_FIXED_TARGET",
            "CONFIG_PHANTOMCHANNEL_EMBED_ENABLE",
            "CONFIG_PHANTOMCHANNEL_DYNAMIC_TIMING",
            "CONFIG_PHANTOMCHANNEL_RX_GUARD_US",
            "CONFIG_PHANTOMCHANNEL_FORCE_2M",
            "CONFIG_PHANTOMCHANNEL_FORCE_1M",
            "CONFIG_BT_AUTO_PHY_UPDATE",
        } and actual is None:
            actual = "n" if name != "CONFIG_PHANTOMCHANNEL_RX_GUARD_US" else "0"
        if actual != expected:
            raise RuntimeError(
                f"Build configuration mismatch for {name}: expected {expected!r}, got {actual!r}"
            )

    if "CONFIG_PHANTOMCHANNEL_PERIPHERAL_TX=y" in config:
        raise RuntimeError("Central image unexpectedly enables peripheral covert TX")
    if "CONFIG_BT_PERIPHERAL=y" in config:
        raise RuntimeError("Central image unexpectedly enables Bluetooth peripheral role")

    print(
        f"Verified build configuration: target_mode={target_mode}, "
        f"covert_len={covert_len} B, interval={interval_ms} ms, phy={phy}"
    )
    print(f"Verified central PHY policy: symmetric {phy} required before writes")


def flash_firmware(firmware: Path, serial_number: str) -> None:
    nrfutil = shutil.which("nrfutil")
    if nrfutil is None:
        raise FileNotFoundError("nrfutil was not found in PATH")

    command = [
        nrfutil,
        "device",
        "program",
        "--serial-number",
        serial_number,
        "--family",
        "nrf52",
        "--firmware",
        str(firmware),
        "--options",
        "verify=VERIFY_READ,reset=RESET_DEFAULT",
    ]
    print(f"Flashing nRF52840 central firmware to DK {serial_number}:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and flash the independent PhantomChannel central firmware."
    )
    parser.add_argument(
        "--target-mode",
        choices=tuple(VARIANTS),
        default="sink-52833-fixed",
        help="central target behavior (default: sink-52833-fixed)",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=100,
        help=f"central write interval in milliseconds ({MIN_INTERVAL_MS}..{MAX_INTERVAL_MS}; default: 100)",
    )
    parser.add_argument(
        "--covert-len",
        type=int,
        default=0,
        help=f"covert payload length in bytes ({MIN_COVERT_LEN}..{MAX_COVERT_LEN}; default: 0)",
    )
    parser.add_argument(
        "--serial-number",
        default=DEFAULT_SERIAL,
        help=f"nRF52840 DK serial number (default: {DEFAULT_SERIAL})",
    )
    parser.add_argument(
        "--embed-enable",
        choices=("y", "n"),
        default="y",
        help="enable the central post-CRC embedding hook (default: y)",
    )
    parser.add_argument(
        "--dynamic-timing",
        choices=("y", "n"),
        default="n",
        help="enable dynamic PhantomChannel timing correction (default: n)",
    )
    parser.add_argument(
        "--rx-guard-us",
        type=int,
        default=0,
        help=f"extra central RX guard in microseconds ({MIN_RX_GUARD_US}..{MAX_RX_GUARD_US}; default: 0)",
    )
    parser.add_argument(
        "--phy",
        choices=("1m", "2m"),
        default="2m",
        help="required symmetric connection PHY before writes (default: 2m)",
    )
    parser.add_argument(
        "--build-both",
        action="store_true",
        help="build both target variants into separate artifact directories; do not flash",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the selected settings without changing, building, or flashing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        validate_settings(args.covert_len, args.interval_ms, args.rx_guard_us)
        write_value_len = args.covert_len + 10
        print(
            "Central settings: "
            f"covert_len={args.covert_len} B, "
            f"write_value_len={write_value_len} B, "
            f"interval={args.interval_ms} ms, "
            f"embed_enable={args.embed_enable}, "
            f"dynamic_timing={args.dynamic_timing}, "
            f"rx_guard_us={args.rx_guard_us}, "
            f"phy={args.phy}, "
            f"serial={args.serial_number}"
        )
        if args.build_both:
            print("Target variants: phone-nearest, sink-52833-fixed")
        if args.dry_run:
            print("Dry run: no project files changed, build run, or device flashed.")
            return 0

        set_central_settings(
            args.covert_len,
            args.interval_ms,
            args.embed_enable,
            args.dynamic_timing,
            args.rx_guard_us,
            args.phy,
        )
        if args.build_both:
            for target_mode in VARIANTS:
                firmware = build_firmware(target_mode)
                verify_build(
                    target_mode,
                    args.covert_len,
                    args.interval_ms,
                    args.embed_enable,
                    args.dynamic_timing,
                    args.rx_guard_us,
                    args.phy,
                )
                print(f"Built {target_mode}: {firmware}")
            print("Both central variants built; no device was flashed.")
            return 0

        firmware = build_firmware(args.target_mode)
        verify_build(
            args.target_mode,
            args.covert_len,
            args.interval_ms,
            args.embed_enable,
            args.dynamic_timing,
            args.rx_guard_us,
            args.phy,
        )
        flash_firmware(firmware, args.serial_number)
        print(
            "Central firmware programmed successfully: "
            f"target_mode={args.target_mode}, covert_len={args.covert_len} B, "
            f"interval={args.interval_ms} ms, "
            f"embed_enable={args.embed_enable}, "
            f"dynamic_timing={args.dynamic_timing}, "
            f"rx_guard_us={args.rx_guard_us}"
        )
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
