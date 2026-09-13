#!/usr/bin/env python3
"""Build and flash the isolated nRF52840 PIP peripheral image."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from pip_program_common import (
    PROJECT_ROOT,
    SDK_ROOT,
    build_image,
    flash_image,
    require_device,
)


SAMPLE_DIR = SDK_ROOT / "zephyr/samples/bluetooth/phantomchannel_pip_peripheral"
BUILD_DIR = PROJECT_ROOT / "artifacts/firmware/phantomchannel_pip_peripheral"
FIRMWARE = BUILD_DIR / "merged.hex"
BOARD = "nrf52840dk/nrf52840"
BOARD_VERSION = "PCA10056"
DEFAULT_SERIAL = "NRF52840_SERIAL"
DEFAULT_COVERT_LEN = 2
DEFAULT_INTERVAL_MS = 1000
MIN_COVERT_LEN = 0
MAX_COVERT_LEN = 32
MIN_INTERVAL_MS = 20
MAX_INTERVAL_MS = 5000


def validate_settings(covert_len: int, interval_ms: int) -> None:
    if not MIN_COVERT_LEN <= covert_len <= MAX_COVERT_LEN:
        raise ValueError(
            f"--covert-len must be {MIN_COVERT_LEN}..{MAX_COVERT_LEN} B, got {covert_len}"
        )
    if not MIN_INTERVAL_MS <= interval_ms <= MAX_INTERVAL_MS:
        raise ValueError(
            f"--interval-ms must be {MIN_INTERVAL_MS}..{MAX_INTERVAL_MS} ms, got {interval_ms}"
        )


def build_with_settings(covert_len: int, interval_ms: int):
    validate_settings(covert_len, interval_ms)
    overlay_dir = PROJECT_ROOT / "artifacts/pip_build_configs"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="phantomchannel_pip_peripheral_",
        suffix=".conf",
        dir=overlay_dir,
        delete=False,
    ) as overlay_file:
        overlay_path = Path(overlay_file.name)
        overlay_file.write(
            "CONFIG_PHANTOMCHANNEL_COVERT_LEN={}\n"
            "CONFIG_PHANTOMCHANNEL_NOTIFY_INTERVAL_MS={}\n".format(
                covert_len, interval_ms
            )
        )
    try:
        return build_image(SAMPLE_DIR, BUILD_DIR, BOARD, overlay_config=overlay_path)
    finally:
        overlay_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial-number", default=DEFAULT_SERIAL)
    parser.add_argument(
        "--covert-len",
        type=int,
        default=None,
        help=f"covert payload length in bytes ({MIN_COVERT_LEN}..{MAX_COVERT_LEN})",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=None,
        help=f"notification interval in milliseconds ({MIN_INTERVAL_MS}..{MAX_INTERVAL_MS})",
    )
    parser.add_argument("--build", action="store_true", help="rebuild before flashing")
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="rebuild and verify the image without flashing",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="re-enumerate the exact board and verify the existing image, without flashing",
    )
    args = parser.parse_args(argv)

    covert_len = DEFAULT_COVERT_LEN if args.covert_len is None else args.covert_len
    interval_ms = DEFAULT_INTERVAL_MS if args.interval_ms is None else args.interval_ms
    validate_settings(covert_len, interval_ms)
    parameterized_build = args.covert_len is not None or args.interval_ms is not None
    if args.verify_only and (parameterized_build or args.build_only):
        raise SystemExit(
            "--verify-only cannot be combined with --covert-len/--interval-ms/--build-only"
        )
    if args.build and args.build_only:
        raise SystemExit("--build and --build-only are mutually exclusive")
    if args.build_only:
        build_with_settings(covert_len, interval_ms)
        print(
            f"PIP peripheral built: {FIRMWARE}; "
            f"covert_len={covert_len} B, interval={interval_ms} ms"
        )
        return 0
    if not args.verify_only and (args.build or parameterized_build):
        build_with_settings(covert_len, interval_ms)
    if not FIRMWARE.is_file():
        raise SystemExit(f"firmware image not found: {FIRMWARE}; pass --build")
    require_device(args.serial_number, BOARD_VERSION)
    if args.verify_only:
        print(
            f"PIP peripheral ready: {args.serial_number} {BOARD_VERSION} {FIRMWARE}; "
            f"covert_len={covert_len} B, interval={interval_ms} ms"
        )
        return 0
    flash_image(FIRMWARE, args.serial_number, BOARD_VERSION)
    print(
        f"PIP peripheral flashed: {args.serial_number} {BOARD_VERSION}; "
        f"covert_len={covert_len} B, interval={interval_ms} ms"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
