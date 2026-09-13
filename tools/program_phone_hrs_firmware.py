#!/usr/bin/env python3
"""Set, build, and flash the isolated phone-facing Phantom HRS firmware.

The default covert payload length is 231 bytes and the default notification
interval is 20 ms, which matches the current rate-test setup.  No interactive
input is required.  The two settings are optional when another configuration
is deliberately being tested.

``--tx-disabled`` builds the benign HRS mirror used for the CIS baseline
(dataset B): a normal HRS notification without the Phantom marker or PC frame.
The benign build goes to its own build directory and ``prj.conf`` is restored
afterwards so the covert default remains the source-of-truth setting.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
from portable_paths import NCS_TOOLCHAIN as TOOLCHAIN_ROOT, NCS_WORKSPACE as SDK_ROOT  # noqa: E402

SAMPLE_DIR = SDK_ROOT / "zephyr/samples/bluetooth/phantomchannel_hrs_peripheral"
BUILD_DIR = PROJECT_ROOT / "artifacts/firmware/phantomchannel_hrs_peripheral"
BENIGN_BUILD_DIR = PROJECT_ROOT / "artifacts/firmware/phantomchannel_hrs_peripheral_benign"
PRJ_CONF = SAMPLE_DIR / "prj.conf"
RATE_CONFIG = PROJECT_ROOT / "configs/x310_phone_hrs_distance_experiment.yaml"
WEST_PYTHON = TOOLCHAIN_ROOT / "usr/local/bin/python3"
ZEPHYR_SDK = TOOLCHAIN_ROOT / "opt/zephyr-sdk"
DEFAULT_SERIAL = "1050216757"


def replace_once(path: Path, pattern: str, replacement: str) -> None:
    text = path.read_text(encoding="utf-8")
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"expected exactly one setting in {path}: {pattern}")
    path.write_text(updated, encoding="utf-8")


def set_test_settings(covert_len: int, interval_ms: int) -> None:
    if not 1 <= covert_len <= 237:
        raise ValueError("covert length must be between 1 and 237 bytes")
    if not 1 <= interval_ms <= 65535:
        raise ValueError("notification interval must be between 1 and 65535 ms")
    replace_once(
        PRJ_CONF,
        r"^CONFIG_PHANTOMCHANNEL_COVERT_LEN=\d+$",
        f"CONFIG_PHANTOMCHANNEL_COVERT_LEN={covert_len}",
    )
    replace_once(
        PRJ_CONF,
        r"^CONFIG_PHANTOMCHANNEL_NOTIFY_INTERVAL_MS=\d+$",
        f"CONFIG_PHANTOMCHANNEL_NOTIFY_INTERVAL_MS={interval_ms}",
    )
    replace_once(
        RATE_CONFIG,
        r"^  covert_len_bytes: \d+$",
        f"  covert_len_bytes: {covert_len}",
    )
    replace_once(
        RATE_CONFIG,
        r"^  notification_interval_ms: \d+$",
        f"  notification_interval_ms: {interval_ms}",
    )


def set_benign_settings(interval_ms: int) -> None:
    if not 1 <= interval_ms <= 65535:
        raise ValueError("notification interval must be between 1 and 65535 ms")
    replace_once(
        PRJ_CONF,
        r"^CONFIG_PHANTOMCHANNEL_PERIPHERAL_TX=\w+$",
        "CONFIG_PHANTOMCHANNEL_PERIPHERAL_TX=n",
    )
    replace_once(
        PRJ_CONF,
        r"^CONFIG_PHANTOMCHANNEL_COVERT_LEN=\d+$",
        "# CONFIG_PHANTOMCHANNEL_COVERT_LEN is not set (benign build)",
    )
    replace_once(
        PRJ_CONF,
        r"^CONFIG_PHANTOMCHANNEL_COVERT_MARKER_BYTE=\d+$",
        "# CONFIG_PHANTOMCHANNEL_COVERT_MARKER_BYTE is not set (benign build)",
    )
    replace_once(
        PRJ_CONF,
        r"^CONFIG_PHANTOMCHANNEL_NOTIFY_INTERVAL_MS=\d+$",
        f"CONFIG_PHANTOMCHANNEL_NOTIFY_INTERVAL_MS={interval_ms}",
    )


def build_environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(TOOLCHAIN_ROOT / "usr/local/lib/python3.12/site-packages")
    env["PATH"] = f"{TOOLCHAIN_ROOT / 'usr/local/bin'}:/usr/bin:/bin"
    env["ZEPHYR_SDK_INSTALL_DIR"] = str(ZEPHYR_SDK)
    return env


def build_firmware(build_dir: Path) -> None:
    command = [
        str(WEST_PYTHON),
        "-m",
        "west",
        "build",
        "-b",
        "nrf52840dk/nrf52840",
        "-d",
        str(build_dir),
        str(SAMPLE_DIR),
        "--",
        f"-DZEPHYR_SDK_INSTALL_DIR={ZEPHYR_SDK}",
    ]
    subprocess.run(command, cwd=SDK_ROOT, env=build_environment(), check=True)


def verify_build(covert_len: int, *, benign: bool = False) -> None:
    build_dir = BENIGN_BUILD_DIR if benign else BUILD_DIR
    config_path = build_dir / "phantomchannel_hrs_peripheral/zephyr/.config"
    config = config_path.read_text(encoding="utf-8")
    if benign:
        if "# CONFIG_PHANTOMCHANNEL_PERIPHERAL_TX is not set" not in config:
            raise RuntimeError("benign build did not disable CONFIG_PHANTOMCHANNEL_PERIPHERAL_TX")
        if "CONFIG_PHANTOMCHANNEL_COVERT_LEN=" in config:
            raise RuntimeError("benign build still sets CONFIG_PHANTOMCHANNEL_COVERT_LEN")
    else:
        expected = f"CONFIG_PHANTOMCHANNEL_COVERT_LEN={covert_len}"
        if expected not in config:
            raise RuntimeError(f"built configuration does not contain {expected}")
    if "# CONFIG_LOG is not set" not in config or "# CONFIG_UART_CONSOLE is not set" not in config:
        raise RuntimeError("built configuration re-enabled blocking LOG/UART output")


def flash(serial_number: str, *, benign: bool = False) -> None:
    if shutil.which("nrfutil") is None:
        raise RuntimeError("nrfutil is not in PATH")
    build_dir = BENIGN_BUILD_DIR if benign else BUILD_DIR
    command = [
        "nrfutil",
        "device",
        "program",
        "--serial-number",
        serial_number,
        "--family",
        "nrf52",
        "--firmware",
        str(build_dir / "merged.hex"),
        "--options",
        "verify=VERIFY_READ,reset=RESET_DEFAULT",
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--covert-len",
        type=int,
        default=231,
        help="covert payload length in bytes; default: 231",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=20,
        help="notification interval in milliseconds; default: 20",
    )
    parser.add_argument("--serial-number", default=DEFAULT_SERIAL)
    parser.add_argument(
        "--tx-disabled",
        action="store_true",
        help="build/flash the benign HRS mirror (no Phantom marker, no PC frame)",
    )
    parser.add_argument(
        "--skip-flash",
        action="store_true",
        help="build and verify only; do not program the DK",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.tx_disabled:
        notification_value_len = 2  # flags + heart rate, no marker/PC frame
    else:
        notification_value_len = args.covert_len + 11
    if args.dry_run:
        print(f"tx_disabled={args.tx_disabled}")
        print(f"covert_len={args.covert_len}")
        print(f"notification_interval_ms={args.interval_ms}")
        print(f"hrs_notification_value_len={notification_value_len}")
        print(f"serial_number={args.serial_number}")
        return 0

    if args.tx_disabled:
        prj_original = PRJ_CONF.read_text(encoding="utf-8")
        set_benign_settings(args.interval_ms)
        try:
            build_firmware(BENIGN_BUILD_DIR)
            verify_build(args.covert_len, benign=True)
            if not args.skip_flash:
                flash(args.serial_number, benign=True)
        finally:
            PRJ_CONF.write_text(prj_original, encoding="utf-8")
    else:
        set_test_settings(args.covert_len, args.interval_ms)
        build_firmware(BUILD_DIR)
        verify_build(args.covert_len)
        if not args.skip_flash:
            flash(args.serial_number)
    if args.tx_disabled:
        mode = "benign HRS mirror (TX disabled)"
    else:
        mode = f"covert_len={args.covert_len} B"
    print(
        f"{'built' if args.skip_flash else 'flashed'} PhantomHRS: {mode}, "
        f"notification_interval={args.interval_ms} ms, "
        f"HRS notification value={notification_value_len} B, "
        f"serial={args.serial_number}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
