import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "score_iq_parser_candidates.py"
SPEC = importlib.util.spec_from_file_location("score_iq_parser_candidates", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def make_frame(seq: int, payload: bytes, corrupt_check: bool = False) -> str:
    frame = bytearray(b"PC")
    frame.extend(seq.to_bytes(2, "little"))
    frame.append(len(payload))
    frame.extend(payload)
    check = 0
    for byte in frame:
        check ^= byte
    if corrupt_check:
        check ^= 0x01
    frame.append(check)
    return bytes(frame).hex()


class ScoreIqParserCandidatesTest(unittest.TestCase):
    def test_extracts_complete_pc_frame_and_keeps_bad_integrity(self):
        payload = bytes([0xA5, 0x01, 0x02, 0x03])
        row = {
            "post_crc_hex": make_frame(7, payload, corrupt_check=True) + "deadbeef",
            "crc_capture_status": "ok",
        }
        frame = module.extract_pc_frame(row)
        self.assertIsNotNone(frame)
        self.assertEqual(frame["covert_length_bytes"], 4)
        self.assertEqual(frame["pc_frame_bytes"], 10)
        self.assertEqual(frame["data_hex"], "010203")
        self.assertFalse(frame["integrity_ok"])

    def test_deduplicates_overlap_and_prefers_integrity_valid_row(self):
        payload = bytes([0xA5]) + bytes(range(1, 8))
        frame_ok = make_frame(3, payload)
        frame_bad = make_frame(3, payload, corrupt_check=True)
        metadata = {"samples": 1_000_000, "actual_sample_rate_sps": 100_000_000}
        rows = [
            {
                "wideband_sample_index": "1000",
                "channel": "10",
                "access_address": "0x12345678",
                "post_crc_hex": frame_bad,
                "crc_capture_status": "ok",
            },
            {
                "wideband_sample_index": "1125",
                "channel": "10",
                "access_address": "0x12345678",
                "post_crc_hex": frame_ok,
                "crc_capture_status": "ok",
            },
        ]
        candidates, summary = module.score_candidates(metadata, rows)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["pc_frame_bytes"], 14)
        self.assertEqual(candidates[0]["payload_data_bytes"], 8)
        self.assertEqual(candidates[0]["payload_data_bytes_excluding_marker"], 7)
        self.assertEqual(candidates[0]["integrity_ok"], 1)
        self.assertEqual(summary["raw_duplicate_rows_removed"], 1)
        self.assertAlmostEqual(summary["iq_window_parser_candidate_data_bps"], 11200.0)
        self.assertAlmostEqual(summary["iq_window_parser_candidate_payload_data_bps"], 6400.0)
        self.assertEqual(summary["parser_candidate_pc_frame_bytes"], 14)
        self.assertEqual(summary["rate_denominator"], "seq_unique_count")
        self.assertEqual(summary["parser_candidate_covert_data_bytes_excluding_marker"], 7)

    def test_default_rate_counts_one_frame_per_unique_sequence(self):
        payload = bytes([0xA5, 0x01, 0x02, 0x03])
        metadata = {"samples": 1_000_000, "actual_sample_rate_sps": 100_000_000}
        rows = [
            {
                "wideband_sample_index": "1000",
                "channel": "1",
                "access_address": "0x12345678",
                "post_crc_hex": make_frame(7, payload),
                "crc_capture_status": "ok",
            },
            {
                "wideband_sample_index": "50000",
                "channel": "2",
                "access_address": "0x12345678",
                "post_crc_hex": make_frame(7, payload),
                "crc_capture_status": "ok",
            },
        ]
        candidates, summary = module.score_candidates(metadata, rows)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(summary["parser_candidates_deduplicated"], 2)
        self.assertEqual(summary["seq_unique_count"], 1)
        self.assertEqual(summary["seq_duplicate_count"], 1)
        self.assertEqual(summary["parser_candidate_pc_frame_bytes"], 10)
        self.assertEqual(summary["parser_candidate_pc_frame_bytes_all_deduplicated_candidates"], 20)
        self.assertAlmostEqual(summary["iq_window_parser_candidate_data_bps"], 8000.0)

    def test_summary_explicitly_marks_rtt_metrics_unavailable(self):
        payload = bytes([0xA5, 0x10, 0x11])
        metadata = {"samples": 1000, "actual_sample_rate_sps": 1000}
        rows = [{
            "wideband_sample_index": "0",
            "channel": "1",
            "access_address": "0x12345678",
            "post_crc_hex": make_frame(0, payload),
            "crc_capture_status": "ok",
        }]
        _, summary = module.score_candidates(metadata, rows)
        self.assertEqual(summary["rtt_metrics"], "unavailable")
        self.assertEqual(summary["rtt_exact_packets"], "unavailable")
        self.assertEqual(summary["bit_error_rate_against_rtt"], "unavailable")

    def test_pattern_metrics_are_disabled_by_default(self):
        payload = bytes([0xA5, 0x01, 0x02])
        metadata = {"samples": 1000, "actual_sample_rate_sps": 1000}
        rows = [{
            "wideband_sample_index": "0",
            "channel": "1",
            "access_address": "0x12345678",
            "post_crc_hex": make_frame(0, payload),
            "crc_capture_status": "ok",
        }]
        candidates, summary = module.score_candidates(metadata, rows)
        self.assertEqual(summary["pattern_mode"], "disabled")
        self.assertNotIn("pattern_exact", candidates[0])

    def test_pattern_exact_match_gives_psr_exact_one(self):
        payload = bytes([0xA5]) + bytes(((i - 1) % 255) + 1 for i in range(1, 8))
        metadata = {"samples": 1000, "actual_sample_rate_sps": 1000}
        rows = [
            {
                "wideband_sample_index": "0",
                "channel": "1",
                "access_address": "0x12345678",
                "post_crc_hex": make_frame(0, payload),
                "crc_capture_status": "ok",
            },
            {
                "wideband_sample_index": "50000",
                "channel": "2",
                "access_address": "0x12345678",
                "post_crc_hex": make_frame(1, payload),
                "crc_capture_status": "ok",
            },
        ]
        candidates, summary = module.score_candidates(metadata, rows, pattern_enabled=True)
        self.assertEqual(summary["pattern_mode"], "a5_increment")
        self.assertEqual(summary["pattern_exact_packets"], 2)
        self.assertAlmostEqual(summary["psr_exact"], 1.0)
        self.assertAlmostEqual(summary["pattern_byte_recovery"], 1.0)
        self.assertAlmostEqual(summary["pattern_ber"], 0.0)
        self.assertEqual(candidates[0]["pattern_exact"], 1)

    def test_pattern_byte_error_lowers_psr_exact_and_byte_recovery(self):
        payload = bytearray([0xA5, 0x01, 0x02, 0x03, 0x04])
        payload[2] = 0x42  # one wrong byte
        metadata = {"samples": 1000, "actual_sample_rate_sps": 1000}
        rows = [{
            "wideband_sample_index": "0",
            "channel": "1",
            "access_address": "0x12345678",
            "post_crc_hex": make_frame(0, bytes(payload)),
            "crc_capture_status": "ok",
        }]
        candidates, summary = module.score_candidates(metadata, rows, pattern_enabled=True)
        self.assertEqual(summary["pattern_exact_packets"], 0)
        self.assertAlmostEqual(summary["psr_exact"], 0.0)
        self.assertAlmostEqual(summary["pattern_byte_recovery"], 4 / 5)
        expected_ber = (0x02 ^ 0x42).bit_count() / (5 * 8)  # 0x02 -> 0x42 flips 1 bit
        self.assertAlmostEqual(summary["pattern_ber"], expected_ber)
        self.assertEqual(candidates[0]["pattern_bit_errors"], (0x02 ^ 0x42).bit_count())

    def test_pattern_outlier_fields_are_unavailable_when_disabled(self):
        payload = bytes([0xA5, 0x01, 0x02])
        metadata = {"samples": 1000, "actual_sample_rate_sps": 1000}
        rows = [{
            "wideband_sample_index": "0",
            "channel": "1",
            "access_address": "0x12345678",
            "post_crc_hex": make_frame(0, payload),
            "crc_capture_status": "ok",
        }]
        _, summary = module.score_candidates(metadata, rows)
        self.assertEqual(summary["pattern_mode"], "disabled")
        self.assertEqual(summary["pattern_outlier_ber_threshold"], 0.10)
        self.assertEqual(summary["pattern_outlier_packets"], "unavailable")
        self.assertEqual(summary["pattern_ber_excluding_outliers"], "unavailable")

    def test_pattern_outlier_exclusion_isolates_collision_packet(self):
        payload_clean = bytes([0xA5]) + bytes(((i - 1) % 255) + 1 for i in range(1, 8))
        payload_collision = bytearray(payload_clean)
        # Corrupt the last four bytes so per-packet BER is well above 10%.
        for offset in (4, 5, 6, 7):
            payload_collision[offset] ^= 0xFF
        metadata = {"samples": 1000, "actual_sample_rate_sps": 1000}
        rows = [
            {
                "wideband_sample_index": "0",
                "channel": "1",
                "access_address": "0x12345678",
                "post_crc_hex": make_frame(0, bytes(payload_clean)),
                "crc_capture_status": "ok",
            },
            {
                "wideband_sample_index": "50000",
                "channel": "2",
                "access_address": "0x12345678",
                "post_crc_hex": make_frame(1, bytes(payload_collision)),
                "crc_capture_status": "ok",
            },
        ]
        candidates, summary = module.score_candidates(
            metadata,
            rows,
            pattern_enabled=True,
            pattern_outlier_ber_threshold=0.10,
        )
        self.assertEqual(summary["pattern_candidates_compared"], 2)
        self.assertEqual(summary["pattern_outlier_packets"], 1)
        self.assertAlmostEqual(summary["pattern_outlier_fraction"], 0.5)
        self.assertAlmostEqual(summary["pattern_outlier_error_share"], 1.0)
        self.assertAlmostEqual(summary["pattern_ber_excluding_outliers"], 0.0)
        self.assertGreater(summary["pattern_ber"], 0.0)

    def test_extracts_frame_from_dewhitened_pdu_when_post_crc_empty(self):
        payload = bytes([0xA5, 0x01, 0x02, 0x03])
        frame = make_frame(11, payload)
        # Simulate an in-app embedded notification: LL/L2CAP/ATT headers plus
        # a 2-byte HRS value and the aa aa 00 marker before the PC frame.
        pdu = "1afbfa0004001b0e000659aaaa00" + frame
        row = {
            "wideband_sample_index": "0",
            "channel": "5",
            "access_address": "0x12345678",
            "post_crc_hex": "",
            "dewhitened_pdu_hex": pdu,
            "crc_capture_status": "ok",
        }
        metadata = {"samples": 1000, "actual_sample_rate_sps": 1000}
        candidates, summary = module.score_candidates(
            metadata,
            [row],
            pattern_enabled=True,
            inline_frame_source=True,
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["frame_seq"], 11)
        self.assertEqual(candidates[0]["payload_data_bytes"], 4)
        self.assertEqual(summary["seq_unique_count"], 1)
        self.assertAlmostEqual(summary["psr_exact"], 1.0)

    def test_inline_frame_source_off_ignores_dewhitened_pdu(self):
        payload = bytes([0xA5, 0x01, 0x02, 0x03])
        pdu = "1afbfa0004001b0e000659aaaa00" + make_frame(11, payload)
        row = {
            "wideband_sample_index": "0",
            "channel": "5",
            "access_address": "0x12345678",
            "post_crc_hex": "",
            "dewhitened_pdu_hex": pdu,
            "crc_capture_status": "ok",
        }
        metadata = {"samples": 1000, "actual_sample_rate_sps": 1000}
        candidates, summary = module.score_candidates(metadata, [row], pattern_enabled=True)
        self.assertEqual(len(candidates), 0)
        self.assertEqual(summary["seq_unique_count"], 0)

    def test_seq_span_stats_clean_consecutive_run(self):
        stats = module.seq_span_stats(list(range(10, 20)))
        self.assertEqual(stats["seq_main_first"], 10)
        self.assertEqual(stats["seq_main_last"], 19)
        self.assertEqual(stats["seq_span_theoretical_packets"], 10)
        self.assertAlmostEqual(stats["recovered_within_span_fraction"], 1.0)
        self.assertEqual(stats["seq_span_raw_packets"], 10)

    def test_seq_span_stats_ignores_corrupt_seq_outliers(self):
        seqs = [5, 9999] + list(range(100, 121))
        stats = module.seq_span_stats(seqs)
        self.assertEqual(stats["seq_span_raw_packets"], 9995)
        self.assertEqual(stats["seq_main_first"], 100)
        self.assertEqual(stats["seq_main_last"], 120)
        self.assertEqual(stats["seq_span_theoretical_packets"], 21)
        self.assertAlmostEqual(stats["recovered_within_span_fraction"], 23 / 21)

    def test_seq_span_stats_handles_16bit_wrap(self):
        stats = module.seq_span_stats([65534, 65535, 0, 1, 2])
        self.assertEqual(stats["seq_main_first"], 65534)
        self.assertEqual(stats["seq_main_last"], 2)
        self.assertEqual(stats["seq_span_theoretical_packets"], 5)
        self.assertAlmostEqual(stats["recovered_within_span_fraction"], 1.0)

    def test_seq_span_stats_empty(self):
        stats = module.seq_span_stats([])
        self.assertIsNone(stats["seq_span_theoretical_packets"])
        self.assertIsNone(stats["recovered_within_span_fraction"])

    def test_seq_span_stats_keeps_run_across_short_loss_burst(self):
        seqs = list(range(100, 180)) + list(range(185, 200))
        stats = module.seq_span_stats(seqs)
        self.assertEqual(stats["seq_main_first"], 100)
        self.assertEqual(stats["seq_main_last"], 199)
        self.assertEqual(stats["seq_span_theoretical_packets"], 100)
        self.assertAlmostEqual(stats["recovered_within_span_fraction"], 95 / 100)


if __name__ == "__main__":
    unittest.main()
