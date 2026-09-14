#!/usr/bin/env python3
"""Audit PIP packets from an X310 BLE-parser CSV.

This path is deliberately independent of RTT/controller records.  The parser's
observed access address is treated as the fake on-air AA.  The frozen inverse
of the PIP mapping then supplies the real AA, and the dewhitened PDU is walked
back through the second and first software-whitening phases:

    fake parser PDU = H_f(raw) | W_f(F)[2:]
    F              = H_f(raw) | AA_r | R1
    R_ext          = W_r(R1)

The tool accepts the address forms used by the existing parser (for example
``0x12345678``) and keeps both representations in its output:

* ``*_aa_canonical`` is the conventional 32-bit integer spelling;
* ``*_aa_raw`` is the four-byte on-air/controller-array order used by the PIP
  oracle and controller logs.

No advertising AA and no cross-layer timestamp/sequence match is required.
When a row cannot be decoded completely, the tool still performs a byte/bit
deep search for the mapped real AA in every available parser hex field.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.pip_packet_oracle import (  # noqa: E402
    PIPOracleError,
    _air_bits,
    ble_crc,
    map_fake_access_address,
    whiten,
)


HEX_RE = re.compile(r"^[0-9a-fA-F]+$")
HEX_FIELDS = (
    "dewhitened_pdu_hex",
    "whitened_pdu_hex",
    "post_crc_hex",
    "crc_and_post_crc_hex",
    "captured_crc_hex",
    "frame_hex",
    "payload_hex",
)


def parse_hex_bytes(value: Any) -> bytes:
    text = str(value or "").strip()
    text = text[2:] if text.lower().startswith("0x") else text
    text = "".join(text.split())
    if not text or len(text) % 2 or not HEX_RE.fullmatch(text):
        return b""
    return bytes.fromhex(text)


def parse_int(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(text, 0)
    except ValueError:
        try:
            return int(float(text))
        except ValueError:
            return None


def aa_from_parser_value(value: Any) -> bytes:
    """Convert the parser's hexadecimal AA spelling to on-air byte order.

    The BLE parser prints the four AA bytes in the same order used by the
    controller logs (for example ``0x4D1E21F6`` -> ``4d1e21f6``).  Do not use
    ``int.to_bytes(..., "little")`` here: that would reverse the mapping while
    still producing a plausible-looking 32-bit value.
    """

    text = str(value or "").strip()
    if not text:
        return b""
    try:
        number = int(text, 0)
    except ValueError:
        cleaned = text.replace("0x", "").replace("0X", "")
        if not cleaned or not HEX_RE.fullmatch(cleaned):
            return b""
        number = int(cleaned, 16)
    if not 0 <= number <= 0xFFFFFFFF:
        return b""
    return number.to_bytes(4, "big")


def aa_canonical(raw: bytes) -> str:
    return f"0x{int.from_bytes(raw, 'big'):08X}"


def aa_raw_hex(raw: bytes) -> str:
    return raw.hex()


def inverse_map_fake_access_address(fake_access_address: bytes) -> bytes:
    """Invert reverse-byte + per-byte ROL1 mapping in raw byte order."""

    if len(fake_access_address) != 4:
        raise PIPOracleError("fake access address must contain four bytes")

    def ror1(value: int) -> int:
        return ((value >> 1) | ((value & 1) << 7)) & 0xFF

    return bytes(ror1(value) for value in fake_access_address[::-1])


def whiten_at_offset(data: bytes, offset: int, channel: int) -> bytes:
    """Apply the BLE whitening XOR stream beginning at a byte offset."""

    if offset < 0 or not 0 <= channel <= 36:
        raise PIPOracleError("invalid whitening offset or data channel")
    stream = whiten(bytes(offset + len(data)), channel)[offset:]
    return bytes(left ^ right for left, right in zip(data, stream))


def exact_bit_search(haystack: bytes, needle: bytes, max_offset: int = 7) -> list[int]:
    if not haystack or not needle:
        return []
    source = _air_bits(haystack)
    target = _air_bits(needle)
    return [
        offset
        for offset in range(max_offset + 1)
        if source[offset : offset + len(target)] == target
    ]


def deep_search_real_aa(row: dict[str, Any], real_raw: bytes) -> dict[str, Any]:
    hits: list[dict[str, Any]] = []
    for field in HEX_FIELDS:
        blob = parse_hex_bytes(row.get(field))
        if not blob:
            continue
        byte_offsets = [
            index
            for index in range(0, max(0, len(blob) - len(real_raw) + 1))
            if blob[index : index + len(real_raw)] == real_raw
        ]
        bit_offsets = exact_bit_search(blob, real_raw)
        if byte_offsets or bit_offsets:
            hits.append(
                {
                    "field": field,
                    "byte_offsets": byte_offsets,
                    "bit_offsets_0_to_7": bit_offsets,
                }
            )
    return {"found": bool(hits), "hits": hits}


def decode_pip_row(
    row: dict[str, Any],
    fake_raw: bytes,
    real_raw_from_mapping: bytes,
    crc_init: bytes | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "fake_aa_raw": aa_raw_hex(fake_raw),
        "fake_aa_canonical": aa_canonical(fake_raw),
        "real_aa_from_mapping_raw": aa_raw_hex(real_raw_from_mapping),
        "real_aa_from_mapping_canonical": aa_canonical(real_raw_from_mapping),
    }
    channel = parse_int(row.get("channel"))
    pdu = parse_hex_bytes(row.get("dewhitened_pdu_hex"))
    result["channel"] = channel
    result["parser_pdu_bytes"] = len(pdu)
    if channel is None or not 0 <= channel <= 36 or len(pdu) < 6:
        result["decode_status"] = "insufficient_dewhitened_pdu"
        result["deep_search"] = deep_search_real_aa(row, real_raw_from_mapping)
        return result

    fake_header = pdu[:2]
    fake_len = fake_header[1]
    result["fake_header"] = fake_header.hex()
    result["fake_len"] = fake_len
    if len(pdu) != fake_len + 2:
        result["decode_status"] = "pdu_length_mismatch"
        result["deep_search"] = deep_search_real_aa(row, real_raw_from_mapping)
        return result

    # The parser has already removed the hardware whitening.  Undo the
    # software fake-phase whitening at its original byte offset 2.
    fake_tail = whiten_at_offset(pdu[2:], 2, channel)
    full_fake_phase = fake_header + fake_tail
    real_raw_from_payload = full_fake_phase[2:6]
    result["real_aa_from_payload_raw"] = aa_raw_hex(real_raw_from_payload)
    result["real_aa_from_payload_canonical"] = aa_canonical(real_raw_from_payload)
    result["mapping_matches_payload"] = real_raw_from_payload == real_raw_from_mapping
    if len(full_fake_phase) < 6:
        result["decode_status"] = "missing_embedded_real_aa"
        result["deep_search"] = deep_search_real_aa(row, real_raw_from_mapping)
        return result

    r1 = full_fake_phase[6:]
    real_ext = whiten(r1, channel)
    if len(real_ext) < 5:
        result["decode_status"] = "short_real_extension"
        return result

    real_header = real_ext[:2]
    real_len = real_header[1]
    result["real_header"] = real_header.hex()
    result["real_len"] = real_len
    result["real_ext_hex"] = real_ext.hex()
    if len(real_ext) < 2 + real_len + 3:
        result["decode_status"] = "short_real_length"
        return result

    real_payload = real_ext[2 : 2 + real_len]
    real_crc = real_ext[2 + real_len : 2 + real_len + 3]

    # The parser's dewhitened PDU contains the raw covert tail after the
    # fixed PIP prefix.  The controller's second software-whitening pass
    # stops at the real CRC; applying the real-phase whitening to the tail
    # again would return a transformed value rather than the transmitted
    # covert bytes.  Keep the inner real-header/payload/CRC decode above,
    # but take the covert bytes directly from the parser PDU at the fixed
    # fake-AA + fake-header + real-AA + real-header + real-CRC offset.
    expected_covert_len = fake_len - real_len - 9
    covert_offset = 2 + 4 + 2 + real_len + 3
    covert = pdu[covert_offset : covert_offset + max(expected_covert_len, 0)]
    if len(covert) != max(expected_covert_len, 0):
        # Preserve the old diagnostic behavior for truncated rows while
        # making the source of the short tail explicit in the result.
        covert = real_ext[2 + real_len + 3 :]
    result.update(
        {
            "real_payload_hex": real_payload.hex(),
            "real_crc_hex": real_crc.hex(),
            "covert_hex": covert.hex(),
            "covert_len": len(covert),
            "expected_covert_len": expected_covert_len,
            "covert_source": "parser_dewhitened_pdu_fixed_pip_offset",
            "length_contract_ok": expected_covert_len == len(covert) and expected_covert_len >= 0,
        }
    )
    if crc_init is not None and len(crc_init) == 3:
        result["real_crc_ok"] = ble_crc(real_header + real_payload, crc_init) == real_crc
    else:
        result["real_crc_ok"] = None

    result["decode_status"] = (
        "ok"
        if result["mapping_matches_payload"] and result["length_contract_ok"]
        else "decoded_but_inconsistent"
    )
    return result


def audit(rows: list[dict[str, Any]], crc_init: bytes | None = None) -> dict[str, Any]:
    outputs: list[dict[str, Any]] = []
    real_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        fake_raw = aa_from_parser_value(
            row.get("access_address")
            or row.get("access_address_hex")
            or row.get("dominant_access_address_hex")
        )
        if len(fake_raw) != 4:
            output = {"row_index": index, "decode_status": "missing_fake_aa"}
            output["deep_search"] = {"found": False, "hits": []}
        else:
            real_raw = inverse_map_fake_access_address(fake_raw)
            output = decode_pip_row(row, fake_raw, real_raw, crc_init)
            output["row_index"] = index
            real_counts[aa_raw_hex(real_raw)] += 1
        status_counts[output["decode_status"]] += 1
        outputs.append(output)

    return {
        "mode": "pip_x310_fake_to_real_without_cross_layer_match",
        "rows": len(rows),
        "status_counts": dict(status_counts),
        "inferred_real_aa_raw_counts": dict(real_counts),
        "records": outputs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path, help="X310 BLE parser ble_packets.csv")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--crc-init",
        help="optional connection CRCInit in raw/controller byte order, e.g. 555555",
    )
    args = parser.parse_args()

    with args.csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    crc_init = parse_hex_bytes(args.crc_init) if args.crc_init else None
    if crc_init is not None and len(crc_init) != 3:
        parser.error("--crc-init must contain exactly three bytes")
    report = audit(rows, crc_init)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
