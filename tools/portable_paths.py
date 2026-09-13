"""Shared path defaults for the portable PhantomChannel receiver bundle.

The original experiments were run with absolute paths on one workstation.
This module keeps those paths out of new-device entry points: bundled source
defaults to the project tree, while SDKs and hardware-specific tools can be
selected with environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_path(names: tuple[str, ...], default: Path, *, resolve: bool = True) -> Path:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            path = Path(os.path.expandvars(os.path.expanduser(value)))
            return path.resolve() if resolve else path
    return default.resolve() if resolve else default


def project_path(value: str | os.PathLike[str], *, base: Path = PROJECT_ROOT) -> Path:
    """Resolve a config path relative to the project unless it is absolute."""

    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(expanded)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


BLE_ROOT = _env_path(
    ("PHANTOM_BLE_ROOT",),
    PROJECT_ROOT / "vendor" / "BLE_encrypt_check",
)
PARSER_PYTHON = _env_path(
    ("PHANTOM_PARSER_PYTHON",),
    PROJECT_ROOT / ".venv-cuda" / "bin" / "python",
    resolve=False,
)
NCS_WORKSPACE = _env_path(
    ("PHANTOM_NCS_WORKSPACE", "NCS_WORKSPACE"),
    PROJECT_ROOT / "external" / "nrf_sdk" / "v3.0.0-rc1",
)
NCS_TOOLCHAIN = _env_path(
    ("PHANTOM_NCS_TOOLCHAIN", "NCS_TOOLCHAIN"),
    PROJECT_ROOT / "external" / "nrf_sdk" / "toolchains" / "7cbc0036f4",
)
NRF5_SDK_ROOT = _env_path(
    ("PHANTOM_NRF5_SDK_ROOT", "NRF5_SDK_ROOT"),
    PROJECT_ROOT / "external" / "nRF5_SDK_17.1.0_ddde560",
)
CAPTURE_BIN = _env_path(
    ("PHANTOM_CAPTURE_BIN",),
    PROJECT_ROOT / "build-local" / "uhd_b210_capture_framed_udp",
)
UHD_LIBRARY = _env_path(
    ("PHANTOM_UHD_LIBRARY", "UHD_LIBRARY_PATH"),
    Path("/path/to/uhd-4.6.0.0/lib"),
)


def bundled_parser_entrypoint() -> Path:
    return BLE_ROOT / "experiment" / "bt_40m_pfb_realtime.py"


def bundled_firmware_sample() -> Path:
    return PROJECT_ROOT / "firmware" / "nrf52840dk" / "phantomchannel_peripheral"
