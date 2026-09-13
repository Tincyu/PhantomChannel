#!/usr/bin/env python3
"""Run the external BLE parser with a ledger-supplied PIP AA exception.

The ordinary parser correctly rejects malformed connection access addresses.
PIP intentionally uses such an outer AA, so this local entrypoint only relaxes
that one validation for the AA supplied in ``PHANTOM_PIP_ALLOW_AA``.  It does
not modify the BLE_encrypt_check source tree and otherwise executes the same
entrypoint and command-line path.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
from portable_paths import BLE_ROOT  # noqa: E402

sys.path.insert(0, str(BLE_ROOT / "experiment"))
sys.path.insert(0, str(BLE_ROOT / "ble_fun_test"))

import pkt_match  # noqa: E402


def normalize(value: object) -> str:
    text = str(value or "").strip().replace("0x", "").replace("0X", "")
    return text.upper().zfill(8)[-8:] if text else ""


allowed = {
    normalize(value)
    for value in os.environ.get("PHANTOM_PIP_ALLOW_AA", "").split(",")
    if normalize(value)
}
original_validator = pkt_match.valid_ble_connection_access_address


def allow_verified_pip_aa(value: object) -> bool:
    normalized = normalize(value)
    return normalized in allowed or original_validator(value)


pkt_match.valid_ble_connection_access_address = allow_verified_pip_aa
runpy.run_path(
    str(BLE_ROOT / "experiment/bt_40m_pfb_realtime.py"),
    run_name="__main__",
)
