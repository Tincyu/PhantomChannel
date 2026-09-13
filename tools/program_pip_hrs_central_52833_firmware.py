#!/usr/bin/env python3
"""Flash the stable HRS/PIP 52833 central using the PIP-plan method."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pip_program_common import flash_image, require_device


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIRMWARE = PROJECT_ROOT / "artifacts/firmware/phantomchannel_pip_hrs_central_52833/merged.hex"
BOARD_VERSION = "PCA10100"
DEFAULT_SERIAL = "NRF52833_SERIAL"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial-number", default=DEFAULT_SERIAL)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not FIRMWARE.is_file() or FIRMWARE.stat().st_size == 0:
        raise SystemExit(f"firmware image not found or empty: {FIRMWARE}")
    require_device(args.serial_number, BOARD_VERSION)
    result = {
        "firmware": str(FIRMWARE),
        "sha256": file_sha256(FIRMWARE),
        "serial_number": args.serial_number,
        "board_version": BOARD_VERSION,
        "method": "pip_program_common.flash_image -> nrfutil device program",
        "verify_only": args.verify_only,
    }
    if args.verify_only:
        result["status"] = "verified_ready"
    else:
        flash_image(FIRMWARE, args.serial_number, BOARD_VERSION)
        result["status"] = "flashed"
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
