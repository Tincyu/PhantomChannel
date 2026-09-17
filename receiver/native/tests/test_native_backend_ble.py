import numpy as np

from pkt_match import crc
from bt_pipeline.native_backend import parse_ble_packet_segments_native
from bt_pipeline.parsers import parse_ble_packet_segments
from bt_pipeline.pfb_channelizer import HostSegmentBatch


def iq_from_bits(bits, samples_per_symbol, gain):
    demod = np.zeros(len(bits) * samples_per_symbol, dtype=np.float64)
    center_indices = np.arange(samples_per_symbol // 2, len(demod), samples_per_symbol)
    demod[center_indices] = np.where(np.asarray(bits) > 0, 1.0, -1.0)
    dphase = demod / gain
    phase = np.r_[0.0, np.cumsum(dphase)]
    return np.exp(1j * phase).astype(np.complex64)


def swap_bits(value):
    return (value * 0x0202020202 & 0x010884422010) % 1023


def whiten(data, channel):
    output = []
    lfsr = swap_bits(channel) | 2
    for byte in data:
        value = swap_bits(byte)
        for mask in (128, 64, 32, 16, 8, 4, 2, 1):
            if lfsr & 0x80:
                lfsr ^= 0x11
                value ^= mask
            lfsr <<= 1
        output.append(swap_bits(value))
    return bytes(output)


def phantom_frame(seq=7, payload=bytes.fromhex("aabb")):
    frame = bytearray(b"PC")
    frame.extend(int(seq).to_bytes(2, "little"))
    frame.append(len(payload))
    frame.extend(payload)
    check = 0
    for byte in frame:
        check ^= byte
    frame.append(check)
    return bytes(frame)


def valid_data_packet_bits(channel=8, access_address=bytes.fromhex("529CD3A7"), post_crc=b""):
    pdu = bytes([0x01, 37]) + bytes(range(37))
    captured_crc = bytes(crc(pdu, len(pdu), 0xA1B2C3))
    pdu_and_crc = pdu + captured_crc + post_crc
    # The parser reports AA bytes in over-the-air byte order.
    air_bytes = bytes([0xAA]) + access_address + whiten(pdu_and_crc, channel)
    return np.asarray(
        [(byte >> bit) & 1 for byte in air_bytes for bit in range(8)],
        dtype=np.uint8,
    )


def main():
    sample_rate = 4e6
    gain = 1.0
    freq_dev = sample_rate / (2 * np.pi * gain * 8)
    center_freq = 2402e6
    score_threshold = 100.0
    bits_1m = np.array(
        [0, 1, 0, 1, 0, 1, 0, 1] +
        [0, 1, 1, 0] * 8,
        dtype=np.uint8,
    )
    iq = iq_from_bits(bits_1m, samples_per_symbol=4, gain=gain)
    segments = [(0, len(iq), iq, 0)]

    python_packets = parse_ble_packet_segments(
        segments,
        sample_rate,
        center_freq,
        freq_dev,
        score_threshold,
    )
    native_packets = parse_ble_packet_segments_native(
        segments,
        sample_rate,
        center_freq,
        freq_dev,
        score_threshold,
    )
    compact_packets = parse_ble_packet_segments_native(
        HostSegmentBatch.from_segments(segments),
        sample_rate,
        center_freq,
        freq_dev,
        score_threshold,
    )
    threaded_segments = segments + [(len(iq), len(iq) * 2, iq, 1)]
    native_single_thread_packets = parse_ble_packet_segments_native(
        threaded_segments,
        sample_rate,
        center_freq,
        freq_dev,
        score_threshold,
        thread_count=1,
    )
    native_threaded_packets = parse_ble_packet_segments_native(
        threaded_segments,
        sample_rate,
        center_freq,
        freq_dev,
        score_threshold,
        thread_count=2,
    )

    assert native_packets == python_packets
    assert compact_packets == native_packets
    assert native_threaded_packets == native_single_thread_packets

    valid_iq = iq_from_bits(valid_data_packet_bits(), samples_per_symbol=4, gain=gain)
    for sampling_phase in range(4):
        prefix = np.ones(17 + sampling_phase, dtype=np.complex64)
        postpad = np.ones(32, dtype=np.complex64)
        padded_iq = np.concatenate((prefix, valid_iq, postpad))
        window_start = 1000
        core_end = window_start + len(prefix) + len(valid_iq)
        padded_segments = HostSegmentBatch.from_segments(
            [(
                window_start,
                window_start + len(padded_iq),
                padded_iq,
                0,
                window_start + len(prefix),
                core_end,
            )]
        )
        phase_packets = parse_ble_packet_segments_native(
            padded_segments,
            sample_rate,
            2420e6,
            freq_dev,
            3.0,
        )
        assert len(phase_packets) == 1
        assert phase_packets[0]["access_address"] == "0x529CD3A7"
        assert abs(phase_packets[0]["sample_index"] - (window_start + len(prefix))) <= 2
        assert phase_packets[0]["payload_len"] == 37
        assert phase_packets[0]["dewhitened_pdu_hex"] == (
            bytes([0x01, 37]) + bytes(range(37))
        ).hex().upper()
        assert phase_packets[0]["captured_crc_hex"] == bytes(
            crc(bytes([0x01, 37]) + bytes(range(37)), 39, 0xA1B2C3)
        ).hex().upper()
        assert phase_packets[0]["post_crc_hex"] == ""
        assert phase_packets[0]["crc_and_post_crc_hex"] == phase_packets[0]["captured_crc_hex"]
        assert phase_packets[0]["crc_capture_status"] == "ok"

    post_crc_frame = phantom_frame(seq=7, payload=bytes.fromhex("aabb"))
    phantom_iq = iq_from_bits(
        valid_data_packet_bits(post_crc=post_crc_frame),
        samples_per_symbol=4,
        gain=gain,
    )
    phantom_packets = parse_ble_packet_segments_native(
        [(3000, 3000 + len(phantom_iq), phantom_iq, 0)],
        sample_rate,
        2420e6,
        freq_dev,
        3.0,
    )
    assert len(phantom_packets) == 1
    assert phantom_packets[0]["captured_crc_hex"] == bytes(
        crc(bytes([0x01, 37]) + bytes(range(37)), 39, 0xA1B2C3)
    ).hex().upper()
    assert phantom_packets[0]["post_crc_hex"] == post_crc_frame.hex().upper()
    assert phantom_packets[0]["crc_and_post_crc_hex"] == (
        phantom_packets[0]["captured_crc_hex"] + post_crc_frame.hex().upper()
    )
    assert phantom_packets[0]["crc_capture_status"] == "ok"

    packet_iq = iq_from_bits(valid_data_packet_bits(), samples_per_symbol=4, gain=gain)
    two_packet_iq = np.concatenate(
        (packet_iq, np.ones(100, dtype=np.complex64), packet_iq)
    )
    two_packet_results = parse_ble_packet_segments_native(
        [(2000, 2000 + len(two_packet_iq), two_packet_iq, 0)],
        sample_rate,
        2420e6,
        freq_dev,
        3.0,
    )
    calibration_packets = [
        packet for packet in two_packet_results
        if packet["access_address"] == "0x529CD3A7" and packet["payload_len"] == 37
    ]
    assert len(calibration_packets) == 2
    assert calibration_packets[1]["sample_index"] > calibration_packets[0]["sample_index"]

    bad_adv_aa_bits = np.array(
        [0, 1, 0, 1, 0, 1, 0, 1] +
        [0, 0, 1, 0] * 8 +
        [0, 1, 0, 1] * 16,
        dtype=np.uint8,
    )
    bad_adv_iq = iq_from_bits(bad_adv_aa_bits, samples_per_symbol=4, gain=gain)
    bad_adv_segments = [(0, len(bad_adv_iq), bad_adv_iq, 0)]
    assert parse_ble_packet_segments(
        bad_adv_segments,
        sample_rate,
        center_freq,
        freq_dev,
        score_threshold,
    ) == []
    assert parse_ble_packet_segments_native(
        bad_adv_segments,
        sample_rate,
        center_freq,
        freq_dev,
        score_threshold,
    ) == []

    for invalid_access_address in (
        bytes.fromhex("AAAAAAAA"),
        bytes.fromhex("55555555"),
        bytes.fromhex("AAA8AAAA"),
    ):
        invalid_bits = valid_data_packet_bits(access_address=invalid_access_address)
        invalid_iq = iq_from_bits(invalid_bits, samples_per_symbol=4, gain=gain)
        invalid_segments = [(0, len(invalid_iq), invalid_iq, 0)]
        assert parse_ble_packet_segments(
            invalid_segments,
            sample_rate,
            2420e6,
            freq_dev,
            score_threshold,
        ) == []
        assert parse_ble_packet_segments_native(
            invalid_segments,
            sample_rate,
            2420e6,
            freq_dev,
            score_threshold,
        ) == []
    print("PASS: native BLE backend wrapper")


if __name__ == "__main__":
    main()
