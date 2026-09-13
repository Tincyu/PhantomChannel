import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


postprocess = load_module("phantom_postprocess_scorer_local_test", ROOT / "tools" / "phantom_postprocess_scorer.py")
local_tail = load_module("local_tail_recovery_v2_test", ROOT / "tools" / "local_tail_recovery_v2.py")


class LocalTailRecoveryV2Test(unittest.TestCase):
    def test_empty_iq_is_reported_without_rtt_selection(self):
        result = postprocess.extract_post_crc_hypotheses_from_channel_samples(
            np.empty(0, dtype=np.complex64),
            sample_rate_hz=100_000_000,
            packet_local_start=0,
            access_address=bytes.fromhex("e888866a"),
            dewhitened_pdu_crc=bytes.fromhex("0100") + bytes.fromhex("112233"),
            channel=0,
            tail_len_bytes=8,
        )
        self.assertEqual(result["notes"], "empty_channel_samples")
        self.assertEqual(result["hypotheses"], [])

    def test_grid_parser_accepts_comma_separated_values(self):
        self.assertEqual(local_tail.parse_float_grid(["700000, 900000", "1100000"]), (700000.0, 900000.0, 1100000.0))

    def test_audit_uses_rtt_only_after_blind_winner_exists(self):
        blind = {
            "run_id": "run",
            "observation_index": "4",
            "extracted_seq": "7",
            "extracted_payload_hex": "a5aabb",
            "extracted_integrity_ok": "1",
            "extracted_frame_len_bytes": "8",
            "frame_len_matches": "1",
            "known_bit_errors": "0",
        }
        funnel = {
            "failure_stage": "F5_FRAME_INVALID",
            "tx_attempt_id": "4:7:0",
            "seq": "7",
            "channel": "1",
        }
        tx = {"seq": "7", "payload": "a5aabb", "marker": "a5", "data_payload": "aabb"}
        row = local_tail.make_audit_row(blind, funnel, {"7": tx})
        self.assertEqual(row["rtt_guided_exact"], 1)
        self.assertEqual(row["rtt_guided_data_exact"], 1)


if __name__ == "__main__":
    unittest.main()
