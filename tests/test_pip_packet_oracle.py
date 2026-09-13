from __future__ import annotations

import pytest

from tools.pip_x310_audit import audit
from tools.pip_packet_oracle import (
    PIPOracleError,
    ble_crc,
    build_pip_packet,
    map_fake_access_address,
    validate_packet,
    whiten,
)


def payload(length: int) -> bytes:
    value = bytearray((index * 13) & 0xFF for index in range(length))
    value[0:2] = (length - 4).to_bytes(2, "little")
    return bytes(value)


TEST_DATA_ACCESS_ADDRESS_A = "0e542595"
TEST_DATA_ACCESS_ADDRESS_B = "86a52ecb"


def build(
    length: int = 35,
    *,
    channel: int = 18,
    aa: str = TEST_DATA_ACCESS_ADDRESS_A,
    phy: str = "1m",
    real_header0: int = 0x01,
):
    return build_pip_packet(
        real_header0=real_header0,
        real_payload=payload(length),
        covert=bytes.fromhex("0102"),
        real_access_address=bytes.fromhex(aa),
        crc_init=bytes.fromhex("555555"),
        channel=channel,
        phy=phy,
    )


def test_whitening_known_zero_vector_and_round_trip():
    assert whiten(b"\x00", 0) == bytes.fromhex("40")
    assert whiten(bytes.fromhex("0102030405"), 18).hex() == "43794dc965"
    source = bytes(range(32))
    assert whiten(whiten(source, 36), 36) == source


def test_crc_is_table_free_ble_byte_order_vector():
    assert ble_crc(bytes.fromhex("020106"), bytes.fromhex("555555")) == bytes.fromhex(
        "e23ef4"
    )


def test_access_address_mapping_is_bit_ordered_contract():
    assert map_fake_access_address(bytes.fromhex(TEST_DATA_ACCESS_ADDRESS_A)) == bytes.fromhex("2b4aa81c")


@pytest.mark.parametrize("phy", ["1m"])
@pytest.mark.parametrize("channel", [0, 18, 36])
@pytest.mark.parametrize(
    ("aa", "fake_length"),
    [
        (TEST_DATA_ACCESS_ADDRESS_A, {0: 24, 18: 209, 36: 138}),
        (TEST_DATA_ACCESS_ADDRESS_B, {0: 24, 18: 209, 36: 138}),
    ],
)
def test_valid_gate_vectors_cover_required_channels_and_phys(
    phy: str, channel: int, aa: str, fake_length: dict[int, int]
):
    # The available length is channel/AA dependent because the last fake
    # header byte must become the real alternating preamble after W_f.
    packet = build(
        length=fake_length[channel] - 2 - 9,
        channel=channel,
        aa=aa,
        phy=phy,
    )
    validate_packet(packet, bytes.fromhex("0102"))
    assert packet.preamble_valid
    assert packet.real_lock_offsets_bits == (8,)
    assert packet.fake_len == packet.real_len + 2 + 9
    assert packet.statlen == 0
    assert packet.buffer_len == packet.fake_len + 2
    assert bytes.fromhex(packet.fake_dma)[:2] == bytes.fromhex(packet.fake_header)


@pytest.mark.parametrize(
    ("aa", "real_header0", "length", "channel", "expected_preamble"),
    [
        ("add49635", 0x17, 35, 18, "5555"),
        (TEST_DATA_ACCESS_ADDRESS_A, 0x03, 168, 33, "aaaa"),
    ],
)
def test_2m_checks_both_preamble_bytes(
    aa: str,
    real_header0: int,
    length: int,
    channel: int,
    expected_preamble: str,
):
    packet = build(
        length=length,
        channel=channel,
        aa=aa,
        phy="2m",
        real_header0=real_header0,
    )
    validate_packet(packet, bytes.fromhex("0102"))
    assert packet.real_preamble == expected_preamble
    assert packet.real_lock_offsets_bits == (0,)
    assert bytes.fromhex(packet.air_body)[:2] == bytes.fromhex(expected_preamble)


def test_force_2m_200b_length_and_payload_contract():
    covert = bytes([0xA5]) + bytes((index + 1) & 0xFF for index in range(199))
    packet = build_pip_packet(
        real_header0=0x01,
        real_payload=payload(9),
        covert=covert,
        real_access_address=bytes.fromhex(TEST_DATA_ACCESS_ADDRESS_A),
        crc_init=bytes.fromhex("555555"),
        channel=18,
        phy="2m",
    )

    # FORCE deliberately does not require fake Header/LENGTH to double as a
    # valid real-AA preamble. All other nested PIP contracts remain mandatory.
    validate_packet(packet, covert, require_preamble=False)
    assert packet.real_len == 9
    assert packet.covert_len == 200
    assert packet.fake_len == 218
    assert packet.buffer_len == 220
    assert bytes.fromhex(packet.real_ext)[-200:] == covert


def test_non_coded_preamble_follows_real_access_address_first_bit():
    packet = build(
        length=61,
        channel=13,
        aa="add49635",
    )
    validate_packet(packet, bytes.fromhex("0102"))
    assert packet.real_preamble == "55"
    assert packet.fake_preamble == "aa"


def test_strict_1m_preamble_mask_scans_all_37_data_channels():
    # This is the frozen v3 length/AA combination used by the 1000 ms
    # strict image.  A strict implementation must scan all CSA#1 channels;
    # channel 4 is the only valid 1M candidate for this fixed fake length.
    valid_channels = []
    for channel in range(37):
        packet = build(
            length=33,
            channel=channel,
            aa="7b900fa6",
            phy="1m",
            real_header0=0x42,
        )
        if packet.preamble_valid:
            valid_channels.append(channel)

    assert valid_channels == [4]


def test_x310_audit_recovers_real_aa_without_rtt_context():
    packet = build(length=127, channel=36, aa=TEST_DATA_ACCESS_ADDRESS_A)
    fake_raw = bytes.fromhex(packet.fake_access_address)
    parser_aa = f"0x{fake_raw.hex()}"
    report = audit(
        [
            {
                "access_address": parser_aa,
                "channel": "36",
                "dewhitened_pdu_hex": packet.fake_dma,
            }
        ]
    )
    record = report["records"][0]
    assert record["decode_status"] == "ok"
    assert record["mapping_matches_payload"]
    assert record["real_aa_from_payload_raw"] == TEST_DATA_ACCESS_ADDRESS_A
    assert record["covert_hex"] == "0102"
    assert record["real_crc_ok"] is None


def test_high_byte_l2cap_repair_is_not_truncated_in_model():
    # A BLE LL PDU cannot normally reach a non-zero high byte, but the oracle
    # still owns the 16-bit repair contract.  Confirm the generated header is
    # derived from the complete payload length, not a stale input byte.
    packet = build(length=35)
    real_ext = bytes.fromhex(packet.real_ext)
    assert real_ext[2:4] == (35 - 4).to_bytes(2, "little")


def test_invalid_preamble_is_reported_before_enablement():
    packet = build(length=5, channel=18, aa="add49635")
    assert not packet.preamble_valid
    with pytest.raises(PIPOracleError, match="preamble/AA lock"):
        validate_packet(packet, bytes.fromhex("0102"))


def test_coded_phy_is_rejected():
    with pytest.raises(PIPOracleError, match="only 1m/2m"):
        build(35, phy="coded")


def test_oversized_fake_length_is_rejected():
    with pytest.raises(PIPOracleError, match="251-byte"):
        build(252)
