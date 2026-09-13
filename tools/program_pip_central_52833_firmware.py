#!/usr/bin/env python3
"""Build and flash the isolated nRF52833 legal-notification central image."""

from __future__ import annotations

import argparse

from pip_program_common import (
    PROJECT_ROOT,
    SDK_ROOT,
    build_image,
    flash_image,
    require_device,
)


SAMPLE_DIR = SDK_ROOT / "zephyr/samples/bluetooth/phantomchannel_pip_central_52833"
BUILD_DIR = PROJECT_ROOT / "artifacts/firmware/phantomchannel_pip_central_52833"
FIRMWARE = BUILD_DIR / "merged.hex"
BOARD = "nrf52833dk/nrf52833"
BOARD_VERSION = "PCA10100"
DEFAULT_SERIAL = "NRF52833_SERIAL"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial-number", default=DEFAULT_SERIAL)
    parser.add_argument("--build", action="store_true", help="rebuild before flashing")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="re-enumerate the exact board and verify the existing image, without flashing",
    )
    args = parser.parse_args(argv)

    if args.build:
        build_image(SAMPLE_DIR, BUILD_DIR, BOARD)
    if not FIRMWARE.is_file():
        raise SystemExit(f"firmware image not found: {FIRMWARE}; pass --build")
    require_device(args.serial_number, BOARD_VERSION)
    if args.verify_only:
        print(f"PIP central ready: {args.serial_number} {BOARD_VERSION} {FIRMWARE}")
        return 0
    flash_image(FIRMWARE, args.serial_number, BOARD_VERSION)
    print(f"PIP central flashed: {args.serial_number} {BOARD_VERSION}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
