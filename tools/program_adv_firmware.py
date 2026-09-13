#!/usr/bin/env python3
"""Set, build, and flash the PhantomChannel advertising broadcaster firmware.

The two tunable parameters are the covert payload length (bytes) and the
advertising interval (ms).  The frozen rate-test configuration is 239 B / 20 ms
with ``CONFIG_BT_CTLR_ADV_DELAY_ZERO=y``, so the on-air event rate equals the
nominal ``1000 / interval_ms`` value instead of the spec-randomized lower rate.

No interactive input is required; pass ``--skip-flash`` to only rebuild, or
``--dry-run`` to print the planned settings without touching the source tree.
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

SAMPLE_DIR = SDK_ROOT / "zephyr/samples/bluetooth/phantomchannel_adv_broadcaster"
BUILD_DIR = PROJECT_ROOT / "artifacts/firmware/phantomchannel_adv_broadcaster"
BENIGN_BUILD_DIR = PROJECT_ROOT / "artifacts/firmware/phantomchannel_adv_broadcaster_benign"
PRJ_CONF = SAMPLE_DIR / "prj.conf"
WEST_PYTHON = TOOLCHAIN_ROOT / "usr/local/bin/python3"
ZEPHYR_SDK = TOOLCHAIN_ROOT / "opt/zephyr-sdk"
DEFAULT_SERIAL = "1050216757"

COVERT_LEN_RANGE = (1, 239)
INTERVAL_MS_RANGE = (20, 10000)


def replace_once(path: Path, pattern: str, replacement: str) -> None:
    text = path.read_text(encoding="utf-8")
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"expected exactly one setting in {path}: {pattern}")
    path.write_text(updated, encoding="utf-8")


def set_test_settings(covert_len: int, interval_ms: int, *, benign: bool) -> None:
    if not COVERT_LEN_RANGE[0] <= covert_len <= COVERT_LEN_RANGE[1]:
        raise ValueError(f"covert length must be in {COVERT_LEN_RANGE}")
    if not INTERVAL_MS_RANGE[0] <= interval_ms <= INTERVAL_MS_RANGE[1]:
        raise ValueError(f"advertising interval must be in {INTERVAL_MS_RANGE}")
    replace_once(
        PRJ_CONF,
        r"^CONFIG_PHANTOMCHANNEL_ADV_COVERT_LEN=\d+$",
        f"CONFIG_PHANTOMCHANNEL_ADV_COVERT_LEN={covert_len}",
    )
    replace_once(
        PRJ_CONF,
        r"^CONFIG_PHANTOMCHANNEL_ADV_INTERVAL_MS=\d+$",
        f"CONFIG_PHANTOMCHANNEL_ADV_INTERVAL_MS={interval_ms}",
    )
    replace_once(
        PRJ_CONF,
        r"^CONFIG_PHANTOMCHANNEL_ADV_BENIGN=(?:y|n)$",
        f"CONFIG_PHANTOMCHANNEL_ADV_BENIGN={'y' if benign else 'n'}",
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


def verify_build(covert_len: int, interval_ms: int, *, benign: bool, build_dir: Path) -> None:
    config_path = build_dir / "phantomchannel_adv_broadcaster/zephyr/.config"
    config = config_path.read_text(encoding="utf-8")
    for expected in (
        f"CONFIG_PHANTOMCHANNEL_ADV_COVERT_LEN={covert_len}",
        f"CONFIG_PHANTOMCHANNEL_ADV_INTERVAL_MS={interval_ms}",
    ):
        if expected not in config:
            raise RuntimeError(f"built configuration does not contain {expected}")
    benign_expected = (
        "CONFIG_PHANTOMCHANNEL_ADV_BENIGN=y"
        if benign
        else "# CONFIG_PHANTOMCHANNEL_ADV_BENIGN is not set"
    )
    if benign_expected not in config:
        raise RuntimeError(f"built configuration does not contain {benign_expected}")
    if "CONFIG_BT_CTLR_ADV_DELAY_ZERO=y" not in config:
        print(
            "⚠ 注意：本次构建未启用 CONFIG_BT_CTLR_ADV_DELAY_ZERO，空口事件率"
            "低于名义 1000/interval，一键脚本的 PSR 分母需按实测事件率修正。",
            file=sys.stderr,
        )


def flash(serial_number: str, *, build_dir: Path) -> None:
    if shutil.which("nrfutil") is None:
        raise RuntimeError("nrfutil is not in PATH")
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
        default=239,
        help=f"covert payload length in bytes; range {COVERT_LEN_RANGE}; default: 239",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=20,
        help=f"advertising interval in ms; range {INTERVAL_MS_RANGE}; default: 20",
    )
    parser.add_argument("--serial-number", default=DEFAULT_SERIAL)
    parser.add_argument(
        "--benign",
        action="store_true",
        help="build/flash a true benign broadcaster with no PhantomChannel frame or tail",
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
    if not COVERT_LEN_RANGE[0] <= args.covert_len <= COVERT_LEN_RANGE[1]:
        raise SystemExit(f"--covert-len must be in {COVERT_LEN_RANGE}")
    if not INTERVAL_MS_RANGE[0] <= args.interval_ms <= INTERVAL_MS_RANGE[1]:
        raise SystemExit(f"--interval-ms must be in {INTERVAL_MS_RANGE}")
    if args.dry_run:
        print(f"covert_len={args.covert_len}")
        print(f"interval_ms={args.interval_ms}")
        print(f"serial_number={args.serial_number}")
        print(f"delay_zero=CONFIG_BT_CTLR_ADV_DELAY_ZERO=y (frozen)")
        print(f"build_dir={BUILD_DIR}")
        print(f"skip_flash={args.skip_flash}")
        print(f"benign={args.benign}")
        return 0

    build_dir = BENIGN_BUILD_DIR if args.benign else BUILD_DIR
    set_test_settings(args.covert_len, args.interval_ms, benign=args.benign)
    build_firmware(build_dir)
    verify_build(args.covert_len, args.interval_ms, benign=args.benign, build_dir=build_dir)
    if not args.skip_flash:
        flash(args.serial_number, build_dir=build_dir)
    mode = "构建并烧录" if not args.skip_flash else "仅构建（未烧录）"
    variant = "benign 0 B" if args.benign else f"covert {args.covert_len} B"
    print(
        f"{mode}完成：{variant} / interval {args.interval_ms} ms"
        f" / serial {args.serial_number}"
    )
    print("DK 复位后应持续广播，可用 nRF Connect 或 ubertooth 确认 D1:22:33:44:55:66。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
