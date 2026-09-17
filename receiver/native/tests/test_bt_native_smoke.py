import numpy as np

import bt_native
from BER_Func import build_bluetooth_sync_word
from bt_pipeline.parsers import PatientSniffer, decode_br_header_candidates
from pkt_match import (
    Parser_pkt,
    compute_rssi_db,
    decision,
    detect_access_address,
    gfsk_demodulate,
    valid_ble_connection_access_address,
)


def main():
    values = np.array([1.0, 2.5, 3.5], dtype=np.float32)
    assert bt_native.version() == "0.1.0"
    assert bt_native.self_test() is True
    assert bt_native.sum_float32(values) == 7.0

    iq = np.array(
        [1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j, 1 + 0j, 1 + 1j],
        dtype=np.complex64,
    )
    gain = 0.75
    np.testing.assert_allclose(
        bt_native.gfsk_demodulate_complex64(iq, gain),
        gfsk_demodulate(iq, gain),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        bt_native.compute_rssi_db_complex64(iq),
        compute_rssi_db(iq),
        rtol=1e-6,
        atol=1e-6,
    )

    freq_dev = np.array([-1.0, 0.5, 2.0, -0.25, 0.1, -0.2, 0.3, 0.4], dtype=np.float64)
    np.testing.assert_array_equal(
        bt_native.decision_bits_float64(freq_dev, 2),
        decision(freq_dev, 2).astype(np.uint8),
    )

    bits_1m = np.array(
        [0, 1, 0, 1, 0, 1, 0, 1] +
        [0, 1, 1, 0] * 8,
        dtype=np.uint8,
    )
    assert (
        bt_native.detect_ble_access_address_bits(bits_1m, "1M")
        == detect_access_address(bits_1m, "1M")
    )
    native_packet = bt_native.parse_ble_packet_bits(bits_1m, "1M", len(bits_1m) * 4, 37)
    python_packet = Parser_pkt(bits_1m, "1M", len(bits_1m) * 4, 37)
    assert native_packet == python_packet
    noise_bits = np.zeros(32, dtype=np.uint8)
    assert bt_native.parse_ble_packet_bits(noise_bits, "1M", len(noise_bits) * 4, 37) is None
    assert Parser_pkt(noise_bits, "1M", len(noise_bits) * 4, 37) is None

    access_address_cases = {
        "A2E48935": True,
        "529CD3A7": True,
        "AAAAAAAA": False,
        "55555555": False,
        "AAA8AAAA": False,
        "AAAAAAA2": False,
        "D6BE898E": False,
        "D7BE898E": False,
        "12121212": False,
        "00000001": False,
        "not-an-aa": False,
    }
    for access_address, expected in access_address_cases.items():
        assert valid_ble_connection_access_address(access_address) is expected
        assert bt_native.valid_ble_connection_access_address(access_address) is expected

    summaries = bt_native.summarize_segments_complex64(
        iq,
        np.array([0, 2], dtype=np.int64),
        np.array([2, 3], dtype=np.int64),
    )
    assert summaries == [
        {"offset": 0, "length": 2, "power_sum": 2.0},
        {"offset": 2, "length": 3, "power_sum": 3.0},
    ]

    lap = 0x9E8B33
    sync_word = build_bluetooth_sync_word(lap)
    assert bt_native.build_bluetooth_sync_word(lap) == sync_word
    access_bits = np.array([(sync_word >> idx) & 1 for idx in range(64)], dtype=np.uint8)
    bits = np.r_[np.array([1, 0, 1, 0], dtype=np.uint8), access_bits, np.zeros(16, dtype=np.uint8)]
    match = bt_native.find_br_access_code_bits(bits)
    assert match == {"offset": 0, "lap": lap, "valid": True}

    raw_bits = np.zeros(144, dtype=np.uint8)
    python_candidates = decode_br_header_candidates(raw_bits, range(256), stop_after_first=False)
    native_candidates = bt_native.decode_br_header_candidates_bits(
        raw_bits,
        np.arange(256, dtype=np.int32),
        False,
    )
    assert [
        {
            "uap": item["uap"],
            "clk": item["clk"],
            "header": item["header"],
            "lfsr": item["lfsr"],
            "payload_bit_count": item["_payload_len"] if "_payload_len" in item else len(item.get("p_raw", [])),
        }
        for item in python_candidates
    ] == [
        {
            "uap": item["uap"],
            "clk": item["clk"],
            "header": item["header"],
            "lfsr": item["lfsr"],
            "payload_bit_count": item["payload_bit_count"],
        }
        for item in native_candidates
    ]
    assert all("payload_prefix_bits" in item for item in native_candidates)

    if native_candidates:
        python_result = PatientSniffer().process_candidates(python_candidates, len(raw_bits))
        native_result = bt_native.process_br_header_candidates(
            native_candidates,
            len(raw_bits),
            None,
            0,
            10,
        )
        for key in ("status", "uap", "clk", "type", "type_val", "len", "total_bytes"):
            assert native_result[key] == python_result[key]
        locked_result = bt_native.process_br_header_candidates(
            native_candidates,
            len(raw_bits),
            native_candidates[0]["uap"],
            0,
            10,
        )
        assert locked_result["status"] == "Locked"
        assert locked_result["locked_uap"] == native_candidates[0]["uap"]

        sequence_results = bt_native.process_br_header_candidate_sequence(
            [native_candidates, native_candidates, []],
            np.asarray([len(raw_bits), len(raw_bits), len(raw_bits)], dtype=np.int64),
            None,
            0,
            10,
            False,
        )
        assert [item["status"] for item in sequence_results] == [
            "New_Lock",
            "Locked",
            "Searching",
        ]
        assert sequence_results[0]["locked_uap"] == native_candidates[0]["uap"]
        assert sequence_results[1]["locked_uap"] == native_candidates[0]["uap"]
        assert sequence_results[2]["locked_uap"] == native_candidates[0]["uap"]
        assert sequence_results[2]["miss_count"] == 1

    searching_result = bt_native.process_br_header_candidates([], len(raw_bits), 7, 9, 10)
    assert searching_result["status"] == "Searching"
    assert searching_result["locked_uap"] is None
    assert searching_result["miss_count"] == 0
    print("PASS: bt_native smoke")


if __name__ == "__main__":
    main()
