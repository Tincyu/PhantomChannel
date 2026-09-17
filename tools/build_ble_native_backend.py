#!/usr/bin/env python3
"""Build the project-owned PhantomChannel receiver C++ parser extension."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

try:
    from portable_paths import BLE_ROOT, PARSER_PYTHON, PROJECT_ROOT
except ImportError:  # pragma: no cover - package-style import
    from tools.portable_paths import BLE_ROOT, PARSER_PYTHON, PROJECT_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=PARSER_PYTHON)
    parser.add_argument("--source-root", type=Path, default=BLE_ROOT)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "native" / "receiver",
    )
    args = parser.parse_args()
    # Preserve a venv launcher symlink.  Resolving it would silently select the
    # system interpreter and hide pybind11/CUDA packages installed in the venv.
    python = args.python.expanduser()
    source_root = args.source_root.expanduser().resolve()
    build_dir = args.build_dir.expanduser().resolve()
    native_dir = source_root / "native"
    if not (native_dir / "CMakeLists.txt").is_file():
        raise SystemExit(f"receiver native source not found: {native_dir}")
    if not python.is_file():
        raise SystemExit(f"Python interpreter not found: {python}")
    configure = [
        "cmake",
        "-S",
        str(native_dir),
        "-B",
        str(build_dir),
        f"-DPython3_EXECUTABLE={python}",
        # Keep the extension version aligned with the receiver smoke test and
        # the public native-backend contract.
        "-DBT_NATIVE_VERSION=0.1.0",
    ]
    subprocess.run(configure, cwd=PROJECT_ROOT, check=True)
    subprocess.run(["cmake", "--build", str(build_dir), "--parallel"], cwd=PROJECT_ROOT, check=True)
    print(build_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
