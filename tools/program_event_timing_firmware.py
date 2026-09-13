#!/usr/bin/env python3
"""Flash one §6 event-timing image using the PIP-plan nrfutil path.

The image manifest is produced by ``prepare_event_timing_firmware.py``.  This
wrapper deliberately delegates device enumeration and programming to
``pip_program_common`` so the exact J-Link serial and board-version checks used
by the CIS/PIP firmware plan remain in force.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pip_program_common import flash_image, require_device


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "artifacts/firmware/event_timing/firmware_manifest.json"
SERIAL_BY_BOARD = {
    "nrf52840dk/nrf52840": ("NRF52840_SERIAL", "PCA10056"),
    "nrf52833dk/nrf52833": ("NRF52833_SERIAL", "PCA10100"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_entry(condition: str) -> dict:
    if not MANIFEST.is_file():
        raise SystemExit(f"manifest not found: {MANIFEST}; run prepare_event_timing_firmware.py --build")
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = {entry["name"]: entry for entry in data.get("images", [])}
    if condition not in entries:
        raise SystemExit(f"unknown condition {condition!r}; choices: {', '.join(sorted(entries))}")
    entry = entries[condition]
    image = Path(entry["image"])
    if not image.is_file() or image.stat().st_size == 0:
        raise SystemExit(f"firmware image is missing or empty: {image}")
    if not entry.get("verified"):
        raise SystemExit(f"manifest entry is not verified: {condition}")
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--serial-number", default="", help="override the plan-mapped J-Link serial")
    parser.add_argument("--verify-only", action="store_true", help="check board and image without flashing")
    args = parser.parse_args()

    entry = load_entry(args.condition)
    board = entry["board"]
    if board not in SERIAL_BY_BOARD:
        raise SystemExit(f"no PIP-plan serial mapping for board {board!r}")
    default_serial, board_version = SERIAL_BY_BOARD[board]
    serial = args.serial_number or default_serial
    image = Path(entry["image"])
    require_device(serial, board_version)
    result = {
        "condition": args.condition,
        "image": str(image),
        "image_sha256": sha256(image),
        "serial_number": serial,
        "board": board,
        "board_version": board_version,
        "method": "pip_program_common.flash_image -> nrfutil device program",
        "verify_only": args.verify_only,
    }
    if not args.verify_only:
        flash_image(image, serial, board_version)
        result["status"] = "flashed"
    else:
        result["status"] = "verified_ready"
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
