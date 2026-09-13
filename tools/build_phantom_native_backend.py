#!/usr/bin/env python3
"""Build PhantomChannel's read-only-source native BLE facade."""

from __future__ import annotations

import argparse
import sys
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
from portable_paths import BLE_ROOT, PARSER_PYTHON  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--python",
        type=Path,
        default=PARSER_PYTHON,
        help="Python interpreter providing pybind11 and the extension ABI",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/native/phantom_bt_native",
    )
    parser.add_argument(
        "--ble-root",
        type=Path,
        default=BLE_ROOT,
    )
    args = parser.parse_args()
    build_dir = args.build_dir.expanduser().resolve()
    ble_root = args.ble_root.expanduser().resolve()
    configure = [
        "cmake",
        "-S",
        str(PROJECT_ROOT / "native"),
        "-B",
        str(build_dir),
        # Preserve a venv launcher symlink; CMake must use that interpreter's
        # site-packages for pybind11 and the parser runtime.
        f"-DPython3_EXECUTABLE={args.python.expanduser()}",
        f"-DPHANTOM_BLE_ROOT={ble_root}",
    ]
    subprocess.run(configure, cwd=PROJECT_ROOT, check=True)
    subprocess.run(["cmake", "--build", str(build_dir), "--parallel"], cwd=PROJECT_ROOT, check=True)
    print(build_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
