import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "phantom_tail_first_recovery.py"
SPEC = importlib.util.spec_from_file_location("phantom_tail_first_recovery", MODULE_PATH)
tail_first = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tail_first
SPEC.loader.exec_module(tail_first)


class PhantomTailFirstRecoveryTest(unittest.TestCase):
    def test_direct_tail_dewhitening_does_not_need_pdu_bytes(self):
        channel = 8
        prefix = bytes.fromhex("0100") + bytes.fromhex("112233")
        frame = bytes.fromhex("5043070004aabbccdd00")
        raw_tail = tail_first.postprocess.ble_whiten(prefix + frame, channel)[len(prefix):]
        raw_bits = tail_first.postprocess.bytes_to_lsb_bits(raw_tail)
        result = tail_first.dewhiten_direct_tail(raw_bits, channel, len(prefix))
        self.assertEqual(result, frame)

    def test_soft_sync_finds_zero_hamming_candidate(self):
        aa = bytes.fromhex("e888866a")
        expected = tail_first.postprocess.bytes_to_lsb_bits(bytes([0xAA]) + aa)
        samples_per_bit = 40
        demod = np.repeat(np.where(expected == 1, 1.0, -1.0), samples_per_bit).astype(np.float32)
        candidates = tail_first.sync_hypotheses(
            demod,
            nominal_start=0,
            access_address=aa,
            samples_per_bit_values=(40.0,),
            search_samples=2,
            max_hamming=0,
        )
        self.assertTrue(candidates)
        self.assertEqual(candidates[0]["aa_hamming_distance"], 0)

    def test_segment_thresholds_cover_entire_tail(self):
        values = np.linspace(-1.0, 1.0, 257, dtype=np.float32)
        thresholds = tail_first.segment_thresholds(values, segment_bits=64)
        self.assertEqual(thresholds.shape, values.shape)
        self.assertTrue(np.all(np.isfinite(thresholds)))


if __name__ == "__main__":
    unittest.main()
