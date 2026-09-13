#!/usr/bin/env python3
"""Audit an oracle/controller PIP record without touching BLE_encrypt_check.

The first version is intentionally a packet-structure audit, not an IQ
demodulator.  It accepts JSON emitted by ``pip_packet_oracle.py`` or a JSON
array of such records and checks both fake and real views of the same modeled
waveform.  A future X310 worker can feed its recovered hex/metadata through
the same checks before any rate result is accepted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.pip_packet_oracle import (
    PIPOracleError,
    _find_real_lock,
    _preamble_for_address,
    ble_crc,
    map_fake_access_address,
    whiten,
)


def _record_values(record: dict[str, Any]) -> tuple[bytes, bytes, bytes, bytes]:
    try:
        return (
            bytes.fromhex(str(record["real_access_address"])),
            bytes.fromhex(str(record["fake_access_address"])),
            bytes.fromhex(str(record["fake_dma"])),
            bytes.fromhex(str(record["air_body"])),
        )
    except (KeyError, ValueError) as exc:
        raise PIPOracleError(f"record is missing/has invalid hex fields: {exc}") from exc


def audit_record(record: dict[str, Any]) -> dict[str, Any]:
    real_aa, fake_aa, fake_dma, air_body = _record_values(record)
    channel = int(record["channel"])
    real_len = int(record["real_len"])
    covert_len = int(record["covert_len"])
    fake_len = int(record["fake_len"])
    crc_init = bytes.fromhex(str(record["crc_init"]))
    real_header = bytes.fromhex(str(record["real_header"]))
    fake_header = bytes.fromhex(str(record["fake_header"]))
    real_ext = bytes.fromhex(str(record["real_ext"]))
    real_whitened = bytes.fromhex(str(record["real_whitened"]))
    real_crc = bytes.fromhex(str(record["real_crc"]))
    fake_crc = bytes.fromhex(str(record["fake_crc"]))

    if len(real_aa) != 4 or len(fake_aa) != 4:
        raise PIPOracleError("both access addresses must contain four bytes")
    if fake_aa != map_fake_access_address(real_aa):
        raise PIPOracleError("fake AA does not match the frozen reverse+ROL1 mapping")
    if fake_header != bytes((real_header[0], fake_len)):
        raise PIPOracleError("fake Header/LENGTH is not the raw contract value")
    if len(fake_dma) != fake_len + 2:
        raise PIPOracleError("DMA size is not two raw header bytes plus H_f.LENGTH")
    if len(real_ext) != real_len + covert_len + 5:
        raise PIPOracleError("real extension size is inconsistent with Lr/C")
    if real_ext[:2] != real_header or real_ext[-3 - covert_len : -covert_len if covert_len else None] != real_crc:
        raise PIPOracleError("real CRC is not in the expected position")
    if ble_crc(real_ext[: 2 + real_len], crc_init) != real_crc:
        raise PIPOracleError("real CRC does not validate")
    if real_whitened != whiten(real_ext, channel):
        raise PIPOracleError("real whitening does not match W_r")
    if fake_crc != ble_crc(fake_dma, crc_init):
        raise PIPOracleError("fake CRC does not validate the raw DMA PDU")
    if air_body != whiten(fake_dma + fake_crc, channel):
        raise PIPOracleError("air body does not match the final hardware whitening model")
    if air_body[2:6] != real_aa:
        raise PIPOracleError("embedded real AA is not raw/on-air after the fake header")
    if air_body[6 : 6 + len(real_whitened)] != real_whitened:
        raise PIPOracleError("real-phase-whitened extension is not present exactly once")
    locks = _find_real_lock(air_body, real_aa)
    if locks != [8]:
        raise PIPOracleError(f"real preamble/AA lock offsets are {locks}, expected [8]")
    if int(record["statlen"]) != 0:
        raise PIPOracleError("STATLEN must be zero")
    if str(record["fake_preamble"]).lower() != f"{_preamble_for_address(fake_aa):02x}":
        raise PIPOracleError("fake preamble does not match fake AA")

    return {
        "status": "ok",
        "channel": channel,
        "phy": record.get("phy", "unknown"),
        "real_len": real_len,
        "covert_len": covert_len,
        "fake_len": fake_len,
        "real_lock_offsets_bits": locks,
        "same_waveform_fake_and_real_views": True,
    }


def load_records(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict) and "records" in value:
        value = value["records"]
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return value
    raise PIPOracleError("input JSON must be one packet object or an array of packet objects")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet_json", type=Path)
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()

    results = []
    failures = 0
    for index, record in enumerate(load_records(args.packet_json)):
        try:
            result = audit_record(record)
        except (KeyError, TypeError, ValueError, PIPOracleError) as exc:
            failures += 1
            result = {"status": "fail", "index": index, "error": str(exc)}
        results.append(result)

    output = {
        "records": results,
        "record_count": len(results),
        "failure_count": failures,
        "valid": failures == 0 and bool(results),
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if output["valid"] or args.allow_failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
