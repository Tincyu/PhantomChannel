import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "match_rtt_sdr_v2.py"
SPEC = importlib.util.spec_from_file_location("match_rtt_sdr_v2", MODULE_PATH)
match_v2 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = match_v2
SPEC.loader.exec_module(match_v2)


class MatchRttSdrV2Test(unittest.TestCase):
    def test_duplicate_seq_gets_distinct_attempt_ids(self):
        ground_truth = [
            {"run_id": "r", "seq": "7", "covert_hex": "a5", "covert_len_bytes": "1", "covert_marker_hex": "a5"}
        ]
        ll_rows = [
            {"run_id": "r", "seq": "7", "channel": "1", "access_address": "11223344", "parser_access_address": "44332211", "covert_len_bytes": "1"},
            {"run_id": "r", "seq": "7", "channel": "1", "access_address": "11223344", "parser_access_address": "44332211", "covert_len_bytes": "1"},
        ]
        rows = match_v2._attempt_rows(ground_truth, ll_rows)
        self.assertEqual([row["tx_attempt_id"] for row in rows], ["0:7:0", "1:7:1"])
        self.assertEqual([row["retry_ordinal"] for row in rows], [0, 1])

    def test_affine_fit_reports_residuals(self):
        result = match_v2._fit_affine([(10, 1_000), (11, 3_000), (12, 5_000), (13, 7_000)])
        self.assertEqual(result["slope_samples_per_seq"], 2_000)
        self.assertEqual(result["intercept_samples"], -19_000)
        self.assertEqual(result["residual_p95_samples"], 0)

    def test_global_assignment_is_one_to_one_and_monotonic(self):
        tx_rows = [
            {"attempt_index": 0, "seq": "1", "channel": "1", "parser_access_address": "44332211", "payload": "a5", "predicted_sample": 100.0},
            {"attempt_index": 1, "seq": "2", "channel": "1", "parser_access_address": "44332211", "payload": "a5", "predicted_sample": 200.0},
        ]
        candidates = [
            {"observation_index": 10, "channel": "1", "access_address": "44332211", "sample": 100.0, "frame_seq": "", "payload_hex": "", "frame_integrity_ok": "", "post_crc_len": 0},
            {"observation_index": 11, "channel": "1", "access_address": "44332211", "sample": 200.0, "frame_seq": "2", "payload_hex": "a5", "frame_integrity_ok": "1", "post_crc_len": 8},
        ]
        assignment = match_v2._global_monotonic_assignment(tx_rows, candidates, 10.0)
        self.assertEqual(set(assignment), {0, 1})
        self.assertEqual(assignment[0][0]["observation_index"], 10)
        self.assertEqual(assignment[1][0]["observation_index"], 11)

    def test_edge_rejects_out_of_window_candidate(self):
        tx = {"seq": "1", "channel": "1", "parser_access_address": "44332211", "predicted_sample": 100.0}
        candidate = {"channel": "1", "access_address": "44332211", "sample": 111.0, "frame_seq": "", "payload_hex": "", "frame_integrity_ok": "", "post_crc_len": 0}
        self.assertIsNone(match_v2._edge(tx, candidate, 10.0))


if __name__ == "__main__":
    unittest.main()
