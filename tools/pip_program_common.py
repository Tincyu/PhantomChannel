#!/usr/bin/env python3
"""Shared build and device-enumeration helpers for the isolated PIP images."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
from portable_paths import NCS_TOOLCHAIN, NCS_WORKSPACE  # noqa: E402


SDK_ROOT = NCS_WORKSPACE
TOOLCHAIN_ROOT = NCS_TOOLCHAIN
WEST_PYTHON = TOOLCHAIN_ROOT / "usr/local/bin/python3"
ZEPHYR_SDK = TOOLCHAIN_ROOT / "opt/zephyr-sdk"


def build_environment() -> dict[str, str]:
    env = os.environ.copy()
    env["ZEPHYR_SDK_INSTALL_DIR"] = str(ZEPHYR_SDK)
    env["PATH"] = os.pathsep.join(
        [str(TOOLCHAIN_ROOT / "usr/local/bin"), "/usr/bin", "/bin"]
    )
    env["PYTHONPATH"] = str(TOOLCHAIN_ROOT / "usr/local/lib/python3.12/site-packages")
    return env


def build_image(
    sample_dir: Path,
    build_dir: Path,
    board: str,
    overlay_config: Path | None = None,
) -> Path:
    command = [
        str(WEST_PYTHON),
        "-m",
        "west",
        "build",
        "-p",
        "always",
        "-b",
        board,
        str(sample_dir),
        "-d",
        str(build_dir),
        "--",
        f"-DZEPHYR_SDK_INSTALL_DIR={ZEPHYR_SDK}",
    ]
    if overlay_config is not None:
        command.append(f"-DOVERLAY_CONFIG={overlay_config}")
    subprocess.run(command, cwd=SDK_ROOT, env=build_environment(), check=True)
    firmware = build_dir / "merged.hex"
    if not firmware.is_file() or firmware.stat().st_size == 0:
        raise RuntimeError(f"build did not produce a usable image: {firmware}")
    return firmware


def _json_objects(output: str) -> list[Any]:
    objects: list[Any] = []
    for line in output.splitlines():
        try:
            objects.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return objects


def _find_devices(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        devices = value.get("devices")
        if isinstance(devices, list) and all(isinstance(item, dict) for item in devices):
            return devices
        for child in value.values():
            found = _find_devices(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_devices(child)
            if found:
                return found
    return []


def enumerate_devices() -> list[dict[str, Any]]:
    nrfutil = shutil.which("nrfutil")
    if nrfutil is None:
        raise RuntimeError("nrfutil is not in PATH")
    result = subprocess.run(
        [nrfutil, "device", "list", "--json", "--timeout-ms", "1500"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    for obj in _json_objects(result.stdout):
        devices = _find_devices(obj)
        if devices:
            return devices
    raise RuntimeError("nrfutil returned no parseable device list")


def require_device(serial_number: str, board_version: str) -> dict[str, Any]:
    devices = enumerate_devices()
    matches = [
        device
        for device in devices
        if str(device.get("serialNumber", "")) == serial_number
    ]
    if len(matches) != 1:
        available = [str(device.get("serialNumber", "?")) for device in devices]
        raise RuntimeError(
            f"serial {serial_number} was not uniquely enumerated; available={available}"
        )
    device = matches[0]
    actual_board = device.get("devkit", {}).get("boardVersion")
    if actual_board != board_version:
        raise RuntimeError(
            f"serial {serial_number} is {actual_board!r}, expected {board_version!r}"
        )
    return device


def flash_image(firmware: Path, serial_number: str, board_version: str) -> None:
    require_device(serial_number, board_version)
    nrfutil = shutil.which("nrfutil")
    if nrfutil is None:
        raise RuntimeError("nrfutil is not in PATH")
    jlink_dll = os.environ.get(
        "PHANTOM_JLINK_DLL",
        "/opt/embedded/toolchains/jlink/9.28/libjlinkarm.so",
    )
    command = [
        nrfutil,
        "device",
        "program",
        "--serial-number",
        serial_number,
        "--firmware",
        str(firmware),
        "--jlink-dll",
        jlink_dll,
        "--options",
        "verify=VERIFY_READ,reset=RESET_DEFAULT",
    ]
    if not Path(jlink_dll).is_file():
        raise RuntimeError(f"J-Link DLL not found: {jlink_dll}")
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        stdin=subprocess.DEVNULL,
        check=True,
    )
