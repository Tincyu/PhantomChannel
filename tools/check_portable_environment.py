#!/usr/bin/env python3
"""Read-only diagnostic for a new PhantomChannel workstation.

The command never installs packages and never changes the project.  It
reports the bundled files, optional external SDK paths, and host tools needed
by the receiver/parser workflows.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

try:
    from portable_paths import (
        BLE_ROOT,
        CAPTURE_BIN,
        NCS_TOOLCHAIN,
        NCS_WORKSPACE,
        NRF5_SDK_ROOT,
        PARSER_PYTHON,
        PROJECT_ROOT,
        bundled_firmware_sample,
        bundled_parser_entrypoint,
    )
except ImportError:  # pragma: no cover - package-style import
    from tools.portable_paths import (
        BLE_ROOT,
        CAPTURE_BIN,
        NCS_TOOLCHAIN,
        NCS_WORKSPACE,
        NRF5_SDK_ROOT,
        PARSER_PYTHON,
        PROJECT_ROOT,
        bundled_firmware_sample,
        bundled_parser_entrypoint,
    )


def command_version(command: str) -> str:
    executable = shutil.which(command)
    if executable is None:
        return "missing"
    try:
        result = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"error: {exc}"
    text = (result.stdout or result.stderr).strip().splitlines()
    return text[0] if text else f"exit={result.returncode}"


def path_status(path: Path) -> dict[str, object]:
    return {"path": str(path), "exists": path.exists(), "is_file": path.is_file()}


def build_report() -> dict[str, object]:
    internal = {
        "bundled_ble_root": path_status(BLE_ROOT),
        "bundled_parser_entrypoint": path_status(bundled_parser_entrypoint()),
        "bundled_firmware_sample": path_status(bundled_firmware_sample()),
        "phantom_native_cmake": path_status(PROJECT_ROOT / "native" / "CMakeLists.txt"),
        "parser_python": path_status(PARSER_PYTHON),
    }
    external = {
        "ncs_workspace": path_status(NCS_WORKSPACE),
        "ncs_toolchain": path_status(NCS_TOOLCHAIN),
        "nrf5_sdk": path_status(NRF5_SDK_ROOT),
        "capture_binary": path_status(CAPTURE_BIN),
    }
    tools = {
        name: command_version(name)
        for name in (
            "python3",
            "cmake",
            "ninja",
            "west",
            "nrfutil",
            "nrfjprog",
            "JLinkExe",
            "gdb-multiarch",
            "arm-none-eabi-gcc",
        )
    }
    return {
        "project_root": str(PROJECT_ROOT),
        "internal": internal,
        "external": external,
        "tools": tools,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()
    report = build_report()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"project_root: {report['project_root']}")
        for section in ("internal", "external", "tools"):
            print(f"[{section}]")
            values = report[section]
            assert isinstance(values, dict)
            for name, value in values.items():
                if isinstance(value, dict):
                    state = "OK" if value.get("exists") else "MISSING"
                    print(f"{state:7} {name}: {value.get('path')}")
                else:
                    print(f"{name}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

