#!/usr/bin/env python3
"""Independent bit-level oracle for the PhantomChannel PIP packet contract.

The oracle deliberately does not import or execute the Nordic controller
whitening/CRC helpers.  It models the BLE bit order directly and exposes the
same intermediate buffers that the controller must produce:

* ``real_ext`` is H_r | payload_r | CRC_r | covert;
* ``real_whitened`` is W_r(real_ext);
* ``fake_dma`` keeps H_f and H_f.LENGTH raw, while the remainder is W_f(F);
* ``air_body`` is the result of the final hardware whitening and contains the
  raw embedded AA_r followed by the real-phase-whitened bytes.

The address and preamble gate is intentionally strict.  The one-bit/byte
alignment is checked against an LSB-first air bitstream, rather than by
comparing a convenient C-array representation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any


MAX_LL_PAYLOAD = 251
MAX_RADIO_PAYLOAD = 255
FAKE_PREFIX_LEN = 6  # fake AA (4) + fake header (2)
REAL_PACKET_OVERHEAD = 9  # AA (4) + header (2) + CRC (3)
PIP_MARKER = bytes((0xAA, 0xAA, 0x02))


class PIPOracleError(ValueError):
    """Input or contract failure reported by the independent oracle."""


def _reverse8(value: int) -> int:
    return int(f"{value:08b}"[::-1], 2)


def _reverse24(value: int) -> int:
    return (
        (_reverse8((value >> 16) & 0xFF))
        | (_reverse8((value >> 8) & 0xFF) << 8)
        | (_reverse8(value & 0xFF) << 16)
    )


def _validate_channel(channel: int) -> None:
    if not 0 <= channel <= 36:
        raise PIPOracleError(f"data channel must be 0..36, got {channel}")


def _validate_phy(phy: str) -> None:
    if phy not in {"1m", "2m"}:
        raise PIPOracleError(f"PIP supports only 1m/2m PHY, got {phy!r}")


def _as_bytes(value: bytes | bytearray | str, *, name: str) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    try:
        return bytes.fromhex(value.replace("0x", "").replace(":", ""))
    except ValueError as exc:
        raise PIPOracleError(f"{name} is not hexadecimal: {value!r}") from exc


def whiten(data: bytes, channel: int) -> bytes:
    """BLE whitening with bytes and bits consumed LSB first."""

    _validate_channel(channel)
    lfsr = channel | 0x40
    result = bytearray()
    for value in data:
        output = 0
        for bit_index in range(8):
            lfsr_bit = lfsr & 1
            output |= (((value >> bit_index) & 1) ^ lfsr_bit) << bit_index
            lfsr >>= 1
            if lfsr_bit:
                lfsr ^= 0x44
        result.append(output)
    return bytes(result)


def ble_crc(data: bytes, crc_init: bytes) -> bytes:
    """Return the three CRC bytes in the controller's on-air byte order."""

    init = _as_bytes(crc_init, name="crc_init")
    if len(init) != 3:
        raise PIPOracleError(f"crc_init must contain 3 bytes, got {len(init)}")

    # This is the non-reflected BLE polynomial, with input bits consumed in
    # LSB-first order.  It is deliberately table-free and independent from
    # lll_conn.c's optimized lookup implementation.
    crc = int.from_bytes(init, "little")
    for value in data:
        crc ^= _reverse8(value) << 16
        for _ in range(8):
            if crc & 0x800000:
                crc = ((crc << 1) ^ 0x00065B) & 0xFFFFFF
            else:
                crc = (crc << 1) & 0xFFFFFF
    return bytes(
        (
            _reverse8((crc >> 16) & 0xFF),
            _reverse8((crc >> 8) & 0xFF),
            _reverse8(crc & 0xFF),
        )
    )


def map_fake_access_address(real_access_address: bytes) -> bytes:
    """Apply the frozen reverse-byte + per-byte ROL1 mapping."""

    address = _as_bytes(real_access_address, name="real_access_address")
    if len(address) != 4:
        raise PIPOracleError(f"access address must contain 4 bytes, got {len(address)}")
    return bytes(((value << 1) | (value >> 7)) & 0xFF for value in address[::-1])


def _preamble_for_address(address: bytes) -> int:
    """Return the non-coded preamble byte for an AA in on-air byte order."""

    if len(address) != 4:
        raise PIPOracleError("access address must contain exactly 4 bytes")
    # The first transmitted bit is bit 0 of the first AA byte.  Nordic's
    # non-coded RADIO selects the alternating preamble so that this bit
    # continues the preamble transition: 0 -> 0xAA, 1 -> 0x55.
    return 0x55 if (address[0] & 1) else 0xAA


def _preamble_bytes_for_address(address: bytes, phy: str) -> bytes:
    value = _preamble_for_address(address)
    return bytes((value,)) if phy == "1m" else bytes((value, value))


def _air_bits(data: bytes) -> list[int]:
    return [(value >> bit) & 1 for value in data for bit in range(8)]


def _bytes_from_bits(bits: list[int], offset: int, length: int) -> bytes:
    if offset + length * 8 > len(bits):
        raise PIPOracleError("bit window exceeds packet")
    return bytes(
        sum(bits[offset + index * 8 + bit] << bit for bit in range(8))
        for index in range(length)
    )


def _find_real_lock(
    body: bytes, real_access_address: bytes, phy: str
) -> list[int]:
    target = _preamble_bytes_for_address(real_access_address, phy) + real_access_address
    body_bits = _air_bits(body)
    target_bits = _air_bits(target)
    return [
        offset
        for offset in range(0, min(16, len(body_bits)))
        if body_bits[offset : offset + len(target_bits)] == target_bits
    ]


@dataclass(frozen=True)
class PIPPacket:
    channel: int
    phy: str
    crc_init: str
    flag_pos: int
    covert_len: int
    real_len: int
    fake_len: int
    real_access_address: str
    fake_access_address: str
    real_header: str
    fake_header: str
    real_crc: str
    fake_crc: str
    real_ext: str
    real_whitened: str
    fake_dma: str
    air_body: str
    air_packet: str
    fake_preamble: str
    real_preamble: str
    real_lock_offsets_bits: tuple[int, ...]
    preamble_valid: bool
    statlen: int
    maxlen: int
    buffer_len: int


def build_pip_packet(
    *,
    real_header0: int,
    real_payload: bytes,
    covert: bytes,
    real_access_address: bytes,
    crc_init: bytes,
    channel: int,
    phy: str,
    maxlen: int = MAX_RADIO_PAYLOAD,
) -> PIPPacket:
    """Build and validate one PIP packet according to the frozen contract."""

    _validate_channel(channel)
    _validate_phy(phy)
    aa_real = _as_bytes(real_access_address, name="real_access_address")
    init = _as_bytes(crc_init, name="crc_init")
    payload = bytes(real_payload)
    covert = bytes(covert)
    if len(aa_real) != 4:
        raise PIPOracleError("real access address must be exactly 4 bytes")
    if len(init) != 3:
        raise PIPOracleError("crc_init must be exactly 3 bytes")
    if not 0 <= real_header0 <= 0xFF:
        raise PIPOracleError("real_header0 must be one byte")
    if len(payload) < 4:
        raise PIPOracleError("real payload must include the 4-byte L2CAP header")
    if len(payload) > MAX_LL_PAYLOAD:
        raise PIPOracleError("real LL payload exceeds the 251-byte BLE limit")
    if len(covert) > MAX_LL_PAYLOAD:
        raise PIPOracleError("covert payload is too large")

    flag_pos = len(payload)
    l2cap_length = flag_pos - 4
    if l2cap_length > 0xFFFF:
        raise PIPOracleError("L2CAP length does not fit in 16 bits")
    payload = bytearray(payload)
    # The PIP marker is already outside real_payload.  Repair both bytes of
    # the L2CAP length, including the high byte when a large test vector uses
    # it; do not rely on a one-byte assignment.
    payload[0] = l2cap_length & 0xFF
    payload[1] = (l2cap_length >> 8) & 0xFF
    payload = bytes(payload)

    real_header = bytes((real_header0, flag_pos))
    real_crc = ble_crc(real_header + payload, init)
    real_ext = real_header + payload + real_crc + covert
    real_whitened = whiten(real_ext, channel)

    fake_len = flag_pos + len(covert) + REAL_PACKET_OVERHEAD
    if fake_len > MAX_RADIO_PAYLOAD:
        raise PIPOracleError(
            f"H_f.LENGTH={fake_len} exceeds 8-bit/MAX radio payload limit"
        )
    if fake_len > maxlen:
        raise PIPOracleError(f"H_f.LENGTH={fake_len} exceeds configured MAXLEN={maxlen}")

    fake_header = bytes((real_header0, fake_len))
    fake = fake_header + aa_real + real_whitened
    fake_twice_whitened = whiten(fake, channel)
    # H_f and H_f.LENGTH remain raw in DMA.  Only bytes after that header are
    # copied from the second software-whitening pass.
    fake_dma = fake_header + fake_twice_whitened[2:]
    fake_crc = ble_crc(fake_dma, init)
    fake_aa = map_fake_access_address(aa_real)
    fake_preamble_bytes = _preamble_bytes_for_address(fake_aa, phy)
    real_preamble_bytes = _preamble_bytes_for_address(aa_real, phy)

    # This is exactly the final hardware whitening pass.  Its CRC is also
    # generated by the hardware and therefore follows STATLEN=0 semantics.
    air_body = whiten(fake_dma + fake_crc, channel)
    air_packet = fake_preamble_bytes + fake_aa + air_body
    real_locks = tuple(_find_real_lock(air_body, aa_real, phy))
    expected_real_lock = (8,) if phy == "1m" else (0,)

    return PIPPacket(
        channel=channel,
        phy=phy,
        crc_init=init.hex(),
        flag_pos=flag_pos,
        covert_len=len(covert),
        real_len=flag_pos,
        fake_len=fake_len,
        real_access_address=aa_real.hex(),
        fake_access_address=fake_aa.hex(),
        real_header=real_header.hex(),
        fake_header=fake_header.hex(),
        real_crc=real_crc.hex(),
        fake_crc=fake_crc.hex(),
        real_ext=real_ext.hex(),
        real_whitened=real_whitened.hex(),
        fake_dma=fake_dma.hex(),
        air_body=air_body.hex(),
        air_packet=air_packet.hex(),
        fake_preamble=fake_preamble_bytes.hex(),
        real_preamble=real_preamble_bytes.hex(),
        real_lock_offsets_bits=real_locks,
        preamble_valid=real_locks == expected_real_lock,
        statlen=0,
        maxlen=maxlen,
        buffer_len=len(fake_dma),
    )


def validate_packet(
    packet: PIPPacket, covert: bytes, *, require_preamble: bool = True
) -> None:
    """Assert all Gate-A invariants for a generated packet."""

    if require_preamble and not packet.preamble_valid:
        raise PIPOracleError(
            "real preamble/AA lock is not valid at the required PHY offset: "
            f"{packet.real_lock_offsets_bits}"
        )
    if packet.statlen != 0:
        raise PIPOracleError("PIP requires STATLEN=0")
    if packet.fake_len != packet.real_len + len(covert) + REAL_PACKET_OVERHEAD:
        raise PIPOracleError("fake length contract failed")
    if packet.buffer_len != 2 + packet.fake_len:
        raise PIPOracleError("DMA buffer length does not match H_f.LENGTH")
    if packet.covert_len != len(covert):
        raise PIPOracleError("covert length changed during construction")

    fake_dma = bytes.fromhex(packet.fake_dma)
    air_body = bytes.fromhex(packet.air_body)
    real_ext = bytes.fromhex(packet.real_ext)
    real_whitened = bytes.fromhex(packet.real_whitened)
    if fake_dma[:2] != bytes.fromhex(packet.fake_header):
        raise PIPOracleError("DMA fake header/LENGTH is not raw")
    real_ext_len = 2 + packet.real_len + 3 + len(covert)
    if air_body[6 : 6 + real_ext_len] != real_whitened:
        raise PIPOracleError("air body does not contain the real-phase-whitened region once")
    if whiten(real_whitened, packet.channel) != real_ext:
        raise PIPOracleError("real-phase whitening does not invert to the raw extension")
    if covert and real_ext[-len(covert) :] != bytes(covert):
        raise PIPOracleError("covert was not recovered exactly once at the real tail")
    if whiten(air_body[:2], packet.channel) != fake_dma[:2]:
        raise PIPOracleError("fake Header/LENGTH does not cancel the hardware whitening")
    expected_fake_body = whiten(fake_dma + bytes.fromhex(packet.fake_crc), packet.channel)
    if air_body != expected_fake_body:
        raise PIPOracleError("hardware whitening/fake CRC model mismatch")


def _json_packet(packet: PIPPacket) -> dict[str, Any]:
    value = asdict(packet)
    value["real_lock_offsets_bits"] = list(packet.real_lock_offsets_bits)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-header0", type=lambda value: int(value, 0), default=0x01)
    parser.add_argument("--real-payload", default="0000040001020304")
    parser.add_argument("--covert", default="0102")
    parser.add_argument(
        "--access-address",
        required=True,
        help="Current connection data-channel AA in on-air byte order; do not use the advertising AA.",
    )
    parser.add_argument("--crc-init", default="555555")
    parser.add_argument("--channel", type=int, default=18)
    parser.add_argument("--phy", choices=("1m", "2m"), default="1m")
    parser.add_argument("--maxlen", type=int, default=MAX_RADIO_PAYLOAD)
    parser.add_argument("--allow-invalid-preamble", action="store_true")
    args = parser.parse_args()

    packet = build_pip_packet(
        real_header0=args.real_header0,
        real_payload=_as_bytes(args.real_payload, name="real_payload"),
        covert=_as_bytes(args.covert, name="covert"),
        real_access_address=_as_bytes(args.access_address, name="access_address"),
        crc_init=_as_bytes(args.crc_init, name="crc_init"),
        channel=args.channel,
        phy=args.phy,
        maxlen=args.maxlen,
    )
    if not args.allow_invalid_preamble:
        validate_packet(packet, _as_bytes(args.covert, name="covert"))
    print(json.dumps(_json_packet(packet), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
