import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "match_rtt_sdr_results.py"
SPEC = importlib.util.spec_from_file_location("match_rtt_sdr_results", MODULE_PATH)
match_rtt_sdr_results = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = match_rtt_sdr_results
SPEC.loader.exec_module(match_rtt_sdr_results)


class MatchRttSdrResultsTest(unittest.TestCase):
    def test_ble_data_channel_frequency_mapping(self):
        self.assertEqual(match_rtt_sdr_results.ble_data_channel_frequency_hz("0"), 2404000000)
        self.assertEqual(match_rtt_sdr_results.ble_data_channel_frequency_hz("10"), 2424000000)
        self.assertEqual(match_rtt_sdr_results.ble_data_channel_frequency_hz("11"), 2428000000)
        self.assertEqual(match_rtt_sdr_results.ble_data_channel_frequency_hz("16"), 2438000000)
        self.assertEqual(match_rtt_sdr_results.ble_data_channel_frequency_hz("17"), 2440000000)
        self.assertIsNone(match_rtt_sdr_results.ble_data_channel_frequency_hz("37"))

    def test_matches_by_session_seq_then_payload_order(self):
        tx = [
            {"run_id": "r1", "session_id": "s1", "seq": "1", "covert_hex": "aa"},
            {"run_id": "r1", "session_id": "s1", "seq": "2", "covert_hex": "bb"},
            {"run_id": "r1", "session_id": "", "seq": "", "covert_hex": "cc"},
        ]
        rx = [
            {"session_id": "s1", "covert_seq": "1", "covert_payload_hex": "aa", "integrity_ok": "true"},
            {"session_id": "", "covert_seq": "99", "covert_payload_hex": "cc", "integrity_ok": "true"},
            {"session_id": "other", "covert_seq": "2", "covert_payload_hex": "bb", "integrity_ok": "true"},
        ]

        matches, unmatched_rtt, unmatched_sdr, summary = match_rtt_sdr_results.match_packets(tx, rx)

        self.assertEqual(summary["matched_packets"], 3)
        self.assertEqual(summary["exact_packets"], 3)
        self.assertEqual(summary["match_methods"]["session_id_seq"], 1)
        self.assertEqual(summary["match_methods"]["seq_payload"], 1)
        self.assertEqual(summary["match_methods"]["payload_order"], 1)
        self.assertEqual(unmatched_rtt, [])
        self.assertEqual(unmatched_sdr, [])
        self.assertEqual(matches[0]["bit_errors"], 0)

    def test_payload_mismatch_counts_bit_errors(self):
        tx = [{"session_id": "s1", "seq": "1", "covert_hex": "00ff"}]
        rx = [{"session_id": "s1", "covert_seq": "1", "covert_payload_hex": "0fff"}]

        matches, _unmatched_rtt, _unmatched_sdr, summary = match_rtt_sdr_results.match_packets(tx, rx)

        self.assertEqual(summary["matched_packets"], 1)
        self.assertEqual(summary["exact_packets"], 0)
        self.assertEqual(matches[0]["bit_errors"], 4)
        self.assertEqual(matches[0]["bit_error_rate"], "0.250000000")

    def test_phantom_score_reports_standard_pdu_without_covert_recovery(self):
        rtt_gt = [{"run_id": "r1", "seq": "7", "covert_hex": "aabb", "covert_len_bytes": "2"}]
        rtt_ll = [
            {
                "run_id": "r1",
                "seq": "7",
                "channel": "18",
                "access_address": "8e89bed6",
                "parser_access_address": "d6be898e",
                "crc_init": "555555",
                "normal_pdu_len": "7",
                "air_extra_len": "8",
                "covert_len_bytes": "2",
            }
        ]
        sdr_ble = [
            {
                "access_address": "0xD6BE898E",
                "channel": "18",
                "timestamp_us": "100.0",
                "dewhitened_pdu_hex": "1b050102030405",
                "captured_crc_hex": "112233",
                "crc_capture_status": "ok",
            }
        ]

        matches, unmatched_rtt, unmatched_sdr, summary = match_rtt_sdr_results.score_phantom_run(
            rtt_gt,
            rtt_ll,
            sdr_ble,
            {"run_id": "r1", "samples": 40000000, "actual_sample_rate_sps": 40000000},
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(unmatched_rtt, [])
        self.assertEqual(unmatched_sdr, [])
        self.assertEqual(matches[0]["match_method"], "aa_channel_standard_pdu_only")
        self.assertEqual(summary["standard_pdu_observed_packets"], 1)
        self.assertEqual(summary["covert_exact_packets"], 0)
        self.assertEqual(summary["covert_packet_recovery_rate"], 0.0)
        self.assertEqual(summary["realistic_e2e_recovered_bps"], 0.0)

    def test_phantom_score_extracts_future_post_crc_frame(self):
        # Frame: PC, seq=7, len=2, payload=aabb, xor check.
        frame = "5043070002aabb07"
        rtt_gt = [{"run_id": "r1", "seq": "7", "covert_hex": "aabb", "covert_len_bytes": "2"}]
        rtt_ll = [
            {
                "run_id": "r1",
                "seq": "7",
                "channel": "18",
                "access_address": "8e89bed6",
                "parser_access_address": "d6be898e",
                "crc_init": "555555",
                "normal_pdu_len": "7",
                "air_extra_len": "8",
                "covert_len_bytes": "2",
            }
        ]
        sdr_ble = [
            {
                "access_address": "d6be898e",
                "channel": "18",
                "post_crc_hex": "112233" + frame,
                "timestamp_us": "100.0",
            }
        ]

        matches, _unmatched_rtt, _unmatched_sdr, summary = match_rtt_sdr_results.score_phantom_run(
            rtt_gt,
            rtt_ll,
            sdr_ble,
            {"run_id": "r1", "samples": 40000000, "actual_sample_rate_sps": 40000000},
        )

        self.assertEqual(matches[0]["rx_covert_hex"], "aabb")
        self.assertEqual(matches[0]["covert_exact_match"], 1)
        self.assertEqual(matches[0]["covert_integrity_ok"], "1")
        self.assertEqual(summary["covert_packet_recovery_rate"], 1.0)
        self.assertEqual(summary["covert_byte_recovery_rate"], 1.0)
        self.assertEqual(summary["realistic_e2e_recovered_bps"], 16.0)

    def test_phantom_marker_payload_reports_data_goodput(self):
        # Frame: PC, seq=7, len=2, payload=a5bb, xor check.
        frame = "5043070002a5bb08"
        rtt_gt = [
            {
                "run_id": "r1",
                "seq": "7",
                "covert_marker_hex": "a5",
                "covert_hex": "a5bb",
                "covert_data_hex": "bb",
                "covert_len_bytes": "2",
                "covert_data_len_bytes": "1",
            }
        ]
        rtt_ll = [
            {
                "run_id": "r1",
                "seq": "7",
                "channel": "18",
                "access_address": "8e89bed6",
                "parser_access_address": "d6be898e",
                "crc_init": "555555",
                "normal_pdu_len": "7",
                "air_extra_len": "8",
                "covert_len_bytes": "2",
            }
        ]
        sdr_ble = [
            {
                "access_address": "d6be898e",
                "channel": "18",
                "post_crc_hex": frame,
                "timestamp_us": "100.0",
            }
        ]

        matches, _unmatched_rtt, _unmatched_sdr, summary = match_rtt_sdr_results.score_phantom_run(
            rtt_gt,
            rtt_ll,
            sdr_ble,
            {"run_id": "r1", "samples": 40000000, "actual_sample_rate_sps": 40000000},
        )

        self.assertEqual(matches[0]["covert_marker_hex"], "a5")
        self.assertEqual(matches[0]["rx_payload_marker_ok"], "1")
        self.assertEqual(matches[0]["tx_covert_data_hex"], "bb")
        self.assertEqual(matches[0]["rx_covert_data_hex"], "bb")
        self.assertEqual(matches[0]["covert_exact_match"], 1)
        self.assertEqual(matches[0]["covert_data_exact_match"], 1)
        self.assertEqual(summary["covert_data_byte_recovery_rate"], 1.0)
        self.assertEqual(summary["realistic_e2e_recovered_data_bps"], 8.0)

    def test_phantom_exact_frame_is_not_consumed_by_earlier_standard_match(self):
        # Frame: PC, seq=2, len=1, payload=bb, xor check.
        frame = "5043020001bbab"
        rtt_gt = [
            {"run_id": "r1", "seq": "1", "covert_hex": "aa", "covert_len_bytes": "1"},
            {"run_id": "r1", "seq": "2", "covert_hex": "bb", "covert_len_bytes": "1"},
        ]
        rtt_ll = [
            {
                "run_id": "r1",
                "seq": "1",
                "channel": "18",
                "access_address": "8e89bed6",
                "parser_access_address": "d6be898e",
                "crc_init": "555555",
                "normal_pdu_len": "7",
                "air_extra_len": "7",
                "covert_len_bytes": "1",
            },
            {
                "run_id": "r1",
                "seq": "2",
                "channel": "18",
                "access_address": "8e89bed6",
                "parser_access_address": "d6be898e",
                "crc_init": "555555",
                "normal_pdu_len": "7",
                "air_extra_len": "7",
                "covert_len_bytes": "1",
            },
        ]
        sdr_ble = [
            {
                "access_address": "d6be898e",
                "channel": "18",
                "post_crc_hex": frame,
                "timestamp_us": "100.0",
            }
        ]

        matches, unmatched_rtt, unmatched_sdr, summary = match_rtt_sdr_results.score_phantom_run(
            rtt_gt,
            rtt_ll,
            sdr_ble,
            {"run_id": "r1", "samples": 40000000, "actual_sample_rate_sps": 40000000},
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["seq"], "2")
        self.assertEqual(matches[0]["rx_covert_hex"], "bb")
        self.assertEqual(matches[0]["match_method"], "aa_channel_seq_payload")
        self.assertEqual(summary["covert_exact_packets"], 1)
        self.assertEqual(summary["standard_pdu_observed_packets"], 1)
        self.assertEqual(len(unmatched_rtt), 1)
        self.assertEqual(unmatched_rtt[0]["seq"], "1")
        self.assertEqual(unmatched_sdr, [])

    def test_phantom_covert_rate_uses_only_packets_with_payload_ground_truth(self):
        # Frame: PC, seq=2, len=1, payload=bb, xor check.
        frame = "5043020001bbab"
        rtt_gt = [
            {"run_id": "r1", "seq": "2", "covert_hex": "bb", "covert_len_bytes": "1"},
        ]
        rtt_ll = [
            {
                "run_id": "r1",
                "seq": "1",
                "channel": "18",
                "access_address": "8e89bed6",
                "parser_access_address": "d6be898e",
                "crc_init": "555555",
                "normal_pdu_len": "2",
                "air_extra_len": "0",
                "covert_len_bytes": "0",
            },
            {
                "run_id": "r1",
                "seq": "2",
                "channel": "18",
                "access_address": "8e89bed6",
                "parser_access_address": "d6be898e",
                "crc_init": "555555",
                "normal_pdu_len": "7",
                "air_extra_len": "7",
                "covert_len_bytes": "1",
            },
        ]
        sdr_ble = [
            {
                "access_address": "d6be898e",
                "channel": "18",
                "timestamp_us": "90.0",
                "dewhitened_pdu_hex": "0100",
                "captured_crc_hex": "112233",
                "crc_capture_status": "ok",
            },
            {
                "access_address": "d6be898e",
                "channel": "18",
                "post_crc_hex": frame,
                "timestamp_us": "100.0",
            },
        ]

        matches, unmatched_rtt, unmatched_sdr, summary = match_rtt_sdr_results.score_phantom_run(
            rtt_gt,
            rtt_ll,
            sdr_ble,
            {"run_id": "r1", "samples": 40000000, "actual_sample_rate_sps": 40000000},
        )

        self.assertEqual(len(matches), 2)
        self.assertEqual(unmatched_rtt, [])
        self.assertEqual(unmatched_sdr, [])
        self.assertEqual(summary["tx_ll_packets"], 2)
        self.assertEqual(summary["tx_packets_with_payload_ground_truth"], 1)
        self.assertEqual(summary["tx_packets_without_payload_ground_truth"], 1)
        self.assertEqual(summary["standard_pdu_observed_packets"], 2)
        self.assertEqual(summary["standard_pdu_observation_rate"], 1.0)
        self.assertEqual(summary["covert_exact_packets"], 1)
        self.assertEqual(summary["covert_packet_recovery_rate"], 1.0)
        self.assertEqual(summary["covert_byte_recovery_rate"], 1.0)

    def test_phantom_recovered_bps_uses_active_tx_window_when_timestamps_exist(self):
        frames = [
            "5043000001aab8",
            "5043010001bba8",
        ]
        rtt_gt = [
            {
                "run_id": "r1",
                "seq": "0",
                "covert_hex": "aa",
                "covert_len_bytes": "1",
                "rtt_timestamp_us": "1000000",
            },
            {
                "run_id": "r1",
                "seq": "1",
                "covert_hex": "bb",
                "covert_len_bytes": "1",
                "rtt_timestamp_us": "1100000",
            },
        ]
        rtt_ll = [
            {
                "run_id": "r1",
                "seq": "0",
                "channel": "18",
                "access_address": "8e89bed6",
                "parser_access_address": "d6be898e",
                "crc_init": "555555",
                "normal_pdu_len": "7",
                "air_extra_len": "7",
                "covert_len_bytes": "1",
            },
            {
                "run_id": "r1",
                "seq": "1",
                "channel": "18",
                "access_address": "8e89bed6",
                "parser_access_address": "d6be898e",
                "crc_init": "555555",
                "normal_pdu_len": "7",
                "air_extra_len": "7",
                "covert_len_bytes": "1",
            },
        ]
        sdr_ble = [
            {"access_address": "d6be898e", "channel": "18", "post_crc_hex": frames[0]},
            {"access_address": "d6be898e", "channel": "18", "post_crc_hex": frames[1]},
        ]

        _matches, _unmatched_rtt, _unmatched_sdr, summary = match_rtt_sdr_results.score_phantom_run(
            rtt_gt,
            rtt_ll,
            sdr_ble,
            {"run_id": "r1", "samples": 40000000, "actual_sample_rate_sps": 40000000},
        )

        self.assertAlmostEqual(summary["active_tx_duration_s"], 0.2)
        self.assertAlmostEqual(summary["active_tx_interval_s"], 0.1)
        self.assertAlmostEqual(summary["active_tx_theoretical_bps"], 80.0)
        self.assertAlmostEqual(summary["capture_window_recovered_bps"], 16.0)
        self.assertAlmostEqual(summary["realistic_e2e_recovered_bps"], 80.0)

    def test_time_aligned_iq_window_metrics_use_exact_payload_clock_anchors(self):
        rtt_gt = [
            {
                "run_id": "r1",
                "seq": str(seq),
                "covert_hex": "aa",
                "covert_len_bytes": "1",
                "rtt_timestamp_us": str(1000 + seq * 100),
            }
            for seq in range(4)
        ]
        rtt_ll = [
            {
                "run_id": "r1",
                "seq": str(seq),
                "channel": "18",
                "access_address": "8e89bed6",
                "parser_access_address": "d6be898e",
                "crc_init": "555555",
                "normal_pdu_len": "7",
                "air_extra_len": "7",
                "covert_len_bytes": "1",
            }
            for seq in range(4)
        ]
        frames = [
            "5043000001aab8",
            "5043010001aab9",
            "5043020001aaba",
            "5043030001aabb",
        ]
        sdr_ble = [
            {
                "access_address": "d6be898e",
                "channel": "18",
                "post_crc_hex": frame,
                "timestamp_us": str(100 + seq * 100),
                "wideband_sample_index": str(100 + seq * 100),
            }
            for seq, frame in enumerate(frames)
        ]

        _matches, _unmatched_rtt, _unmatched_sdr, summary = match_rtt_sdr_results.score_phantom_run(
            rtt_gt,
            rtt_ll,
            sdr_ble,
            {
                "run_id": "r1",
                "samples": 1_000_000,
                "actual_sample_rate_sps": 1_000_000,
            },
        )

        self.assertTrue(summary["time_aligned_metrics_available"])
        self.assertEqual(summary["time_alignment_anchor_count"], 4)
        self.assertEqual(summary["tx_payload_packets_in_iq_capture_window"], 4)
        self.assertEqual(summary["covert_exact_packets_in_iq_capture_window"], 4)
        self.assertEqual(summary["covert_packet_recovery_rate_in_iq_capture_window"], 1.0)

    def test_phantom_in_band_rates_use_actual_rtt_channels(self):
        frames = [
            "5043000001aab8",
            "5043010001bba8",
        ]
        rtt_gt = [
            {
                "run_id": "r1",
                "seq": "0",
                "covert_hex": "aa",
                "covert_len_bytes": "1",
                "rtt_timestamp_us": "1000000",
            },
            {
                "run_id": "r1",
                "seq": "1",
                "covert_hex": "bb",
                "covert_len_bytes": "1",
                "rtt_timestamp_us": "1100000",
            },
        ]
        rtt_ll = [
            {
                "run_id": "r1",
                "seq": "0",
                "channel": "16",
                "access_address": "8e89bed6",
                "parser_access_address": "d6be898e",
                "crc_init": "555555",
                "normal_pdu_len": "7",
                "air_extra_len": "7",
                "covert_len_bytes": "1",
            },
            {
                "run_id": "r1",
                "seq": "1",
                "channel": "17",
                "access_address": "8e89bed6",
                "parser_access_address": "d6be898e",
                "crc_init": "555555",
                "normal_pdu_len": "7",
                "air_extra_len": "7",
                "covert_len_bytes": "1",
            },
        ]
        sdr_ble = [
            {"access_address": "d6be898e", "channel": "16", "post_crc_hex": frames[0]},
            {"access_address": "d6be898e", "channel": "17", "post_crc_hex": frames[1]},
        ]

        matches, _unmatched_rtt, _unmatched_sdr, summary = match_rtt_sdr_results.score_phantom_run(
            rtt_gt,
            rtt_ll,
            sdr_ble,
            {
                "run_id": "r1",
                "samples": 40000000,
                "actual_sample_rate_sps": 40000000,
                "actual_center_frequency_hz": 2420000000,
                "analysis_bandwidth_hz": 40000000,
            },
        )

        self.assertEqual(matches[0]["tx_frequency_hz"], 2438000000)
        self.assertEqual(matches[0]["tx_in_analysis_band"], 1)
        self.assertEqual(matches[1]["tx_frequency_hz"], 2440000000)
        self.assertEqual(matches[1]["tx_in_analysis_band"], 0)
        self.assertEqual(summary["analysis_band_low_hz"], 2400000000)
        self.assertEqual(summary["analysis_band_high_hz"], 2440000000)
        self.assertEqual(summary["analysis_band_high_edge_policy"], "exclusive")
        self.assertEqual(summary["tx_payload_packets_in_analysis_band"], 1)
        self.assertEqual(summary["tx_payload_packets_out_of_analysis_band"], 1)
        self.assertEqual(summary["covert_exact_packets"], 2)
        self.assertEqual(summary["covert_exact_packets_in_analysis_band"], 1)
        self.assertEqual(summary["covert_in_band_packet_recovery_rate"], 1.0)
        self.assertAlmostEqual(summary["active_tx_theoretical_bps"], 80.0)
        self.assertAlmostEqual(summary["active_tx_in_band_theoretical_bps"], 40.0)
        self.assertAlmostEqual(summary["realistic_e2e_recovered_bps_in_analysis_band"], 40.0)


if __name__ == "__main__":
    unittest.main()
