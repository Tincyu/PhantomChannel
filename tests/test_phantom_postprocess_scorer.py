import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "phantom_postprocess_scorer.py"
SPEC = importlib.util.spec_from_file_location("phantom_postprocess_scorer", MODULE_PATH)
phantom_postprocess_scorer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = phantom_postprocess_scorer
SPEC.loader.exec_module(phantom_postprocess_scorer)


class PhantomPostprocessScorerTest(unittest.TestCase):
    def test_ble_1m_airtime_us_includes_standard_fields(self):
        self.assertEqual(phantom_postprocess_scorer.ble_1m_airtime_us(8), 144.0)
        self.assertEqual(phantom_postprocess_scorer.ble_1m_airtime_us(8, 14), 256.0)

    def test_classify_duration_supports_phantom_tail(self):
        result = phantom_postprocess_scorer.classify_duration(252.0, 144.0, 256.0, 18.0)
        self.assertEqual(result["duration_support"], "phantom")
        self.assertGreater(result["duration_margin_us"], 0)

    def test_classify_duration_supports_standard_packet(self):
        result = phantom_postprocess_scorer.classify_duration(145.0, 144.0, 256.0, 18.0)
        self.assertEqual(result["duration_support"], "standard")
        self.assertLess(result["duration_margin_us"], 0)

    def test_classify_duration_handles_unknown_measurement(self):
        result = phantom_postprocess_scorer.classify_duration(None, 144.0, 256.0, 18.0)
        self.assertEqual(result["duration_support"], "unknown")

    def test_extract_post_crc_from_synthetic_channel_samples(self):
        channel = 8
        access_address = bytes.fromhex("15732e94")
        pdu_crc = bytes.fromhex("0100") + bytes.fromhex("112233")
        payload = bytes.fromhex("aabbccdd")
        frame = bytearray(b"PC")
        frame.extend((7).to_bytes(2, "little"))
        frame.append(len(payload))
        frame.extend(payload)
        check = 0
        for byte in frame:
            check ^= byte
        frame.append(check)

        air = (
            bytes([0xAA])
            + access_address
            + phantom_postprocess_scorer.ble_whiten(pdu_crc + bytes(frame), channel)
        )
        bits = phantom_postprocess_scorer.bytes_to_lsb_bits(air)
        samples_per_bit = 40
        phase = 0.0
        iq = []
        for bit in bits:
            step = 0.08 if bit else -0.08
            for _ in range(samples_per_bit):
                phase += step
                iq.append(np.exp(1j * phase))
        prefix = np.ones(300, dtype=np.complex64)
        samples = np.concatenate([prefix, np.asarray(iq, dtype=np.complex64), prefix])

        result = phantom_postprocess_scorer.extract_post_crc_from_channel_samples(
            samples,
            sample_rate_hz=40_000_000,
            packet_local_start=len(prefix),
            access_address=access_address,
            dewhitened_pdu_crc=pdu_crc,
            channel=channel,
            tail_len_bytes=len(frame),
            lowpass_hz=900_000,
            search_samples=100,
        )

        self.assertEqual(result["extracted_frame_hex"], bytes(frame).hex())
        self.assertEqual(result["extracted_seq"], "7")
        self.assertEqual(result["extracted_payload_hex"], payload.hex())
        self.assertEqual(result["extracted_integrity_ok"], "1")
        self.assertLessEqual(result["extractor_known_bit_errors"], 1)


if __name__ == "__main__":
    unittest.main()
