import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "tools" / "analyze_bandwidth_crops.py"
SPEC = importlib.util.spec_from_file_location("analyze_bandwidth_crops", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def make_frame(seq: int, payload: bytes) -> str:
    frame = bytearray(b"PC")
    frame.extend(seq.to_bytes(2, "little"))
    frame.append(len(payload))
    frame.extend(payload)
    check = 0
    for byte in frame:
        check ^= byte
    frame.append(check)
    return bytes(frame).hex()


def make_row(index: int, channel: int, desc: str, seq: int) -> dict:
    payload = bytes([0xA5, 0x01, 0x02, 0x03])
    return {
        "wideband_sample_index": str(index),
        "channel": str(channel),
        "center_freq_desc": desc,
        "access_address": "0x12345678",
        "post_crc_hex": make_frame(seq, payload),
        "crc_capture_status": "ok",
        "packet_type": "BLE_CONN",
    }


class AnalyzeBandwidthCropsTest(unittest.TestCase):
    def test_end_to_end_crop_score_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            (run_root / "iq").mkdir(parents=True)
            (run_root / "diagnostics/one_stage_cpp").mkdir(parents=True)
            metadata = {
                "samples": 100_000_000,
                "actual_sample_rate_sps": 100_000_000,
                "actual_center_frequency_hz": 2440e6,
            }
            (run_root / "iq/metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            rows = []
            for seq in range(6):
                channel = 5 if seq % 2 == 0 else 17
                desc = "2414.000 MHz" if channel == 5 else "2440.000 MHz"
                rows.append(make_row(seq * 1000, channel, desc, seq))
            csv_path = run_root / "diagnostics/one_stage_cpp/ble_packets.csv"
            csv_path.write_text(
                "wideband_sample_index,channel,center_freq_desc,access_address,"
                "post_crc_hex,crc_capture_status,packet_type\n"
                + "\n".join(
                    ",".join(row.values()) for row in rows
                )
                + "\n",
                encoding="utf-8",
            )

            status = module.main([
                "--run-root", str(run_root),
                "--bandwidths-hz", "20000000",
                "--pattern",
            ])
            self.assertEqual(status, 0)
            out_dir = run_root / "results/bandwidth_crops"
            metrics = json.loads((out_dir / "bandwidth_crop_metrics.json").read_text(encoding="utf-8"))
            band = [item for item in metrics if item["bandwidth_mhz"] == 20.0][0]
            self.assertEqual(band["in_band_parser_rows"], 3)
            self.assertAlmostEqual(band["c_bw_candidates_estimated"], 0.5)
            self.assertAlmostEqual(band["c_bw_packets_estimated"], 0.5)
            self.assertAlmostEqual(band["psr_exact_in_band"], 1.0)
            self.assertAlmostEqual(band["g_e2e_bps_in_band"], 3 * 10 * 8)
            full = [item for item in metrics if item["bandwidth_mhz"] is None][0]
            self.assertEqual(full["full_band_unique_seq_candidates"], 6)
            self.assertTrue((out_dir / "empirical_active_map.csv").is_file())
            self.assertTrue((out_dir / "crops/bw20mhz/parser_candidate_rate.json").is_file())


if __name__ == "__main__":
    unittest.main()
