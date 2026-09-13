import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "crop_bandwidth.py"
SPEC = importlib.util.spec_from_file_location("crop_bandwidth", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def row(channel: str, desc: str) -> dict:
    return {
        "channel": channel,
        "center_freq_desc": desc,
        "packet_type": "BLE_CONN",
        "sample_index": "0",
    }


class CropBandwidthTest(unittest.TestCase):
    def test_parse_center_freq_desc(self):
        self.assertEqual(module.parse_center_freq_hz("2440.000 MHz"), 2440e6)
        self.assertEqual(module.parse_center_freq_hz("2402MHz"), 2402e6)
        self.assertIsNone(module.parse_center_freq_hz(""))

    def test_fallback_channel_map_matches_parser_observations(self):
        self.assertEqual(module.ble_channel_freq_mhz(0), 2404.0)
        self.assertEqual(module.ble_channel_freq_mhz(10), 2424.0)
        self.assertEqual(module.ble_channel_freq_mhz(11), 2428.0)
        self.assertEqual(module.ble_channel_freq_mhz(17), 2440.0)
        self.assertEqual(module.ble_channel_freq_mhz(36), 2478.0)
        self.assertEqual(module.ble_channel_freq_mhz(37), 2402.0)
        self.assertEqual(module.ble_channel_freq_mhz(38), 2426.0)
        self.assertEqual(module.ble_channel_freq_mhz(39), 2480.0)
        self.assertIsNone(module.ble_channel_freq_mhz(40))

    def test_passband_crop_and_c_bw(self):
        rows = [row("5", "2414.000 MHz"), row("17", "2440.000 MHz"), row("24", "2454.000 MHz")]
        classify = module.passband_filter(2440e6, 20e6)
        in_band, in_counts, out_counts, unknown = module.crop_rows(rows, classify)
        self.assertEqual(len(in_band), 1)
        self.assertEqual(in_band[0]["channel"], "17")
        self.assertEqual(dict(in_counts), {"17": 1})
        self.assertEqual(dict(out_counts), {"5": 1, "24": 1})
        self.assertEqual(dict(unknown), {})
        classified = len(in_band) + sum(out_counts.values())
        self.assertAlmostEqual(len(in_band) / classified, 1 / 3)

    def test_passband_edge_is_inclusive(self):
        classify = module.passband_filter(2440e6, 20e6)
        self.assertTrue(classify(row("11", "2430.000 MHz")))
        self.assertTrue(classify(row("22", "2450.000 MHz")))
        self.assertFalse(classify(row("10", "2424.000 MHz")))

    def test_channel_list_crop(self):
        rows = [row("0", ""), row("5", ""), row("10", "")]
        classify = module.channel_list_filter({0, 5})
        in_band, _, _, _ = module.crop_rows(rows, classify)
        self.assertEqual([item["channel"] for item in in_band], ["0", "5"])

    def test_unknown_frequency_rows_kept_but_excluded_from_c_bw(self):
        rows = [row("5", "2414.000 MHz"), row("17", "2440.000 MHz"), row("", "")]
        classify = module.passband_filter(2440e6, 20e6)
        in_band, _, out_counts, unknown = module.crop_rows(rows, classify)
        self.assertEqual(len(in_band), 1)
        self.assertEqual(dict(out_counts), {"5": 1})
        self.assertEqual(dict(unknown), {"": 1})

    def test_main_writes_cropped_csv_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "ble_packets.csv"
            source.write_text(
                "channel,center_freq_desc,packet_type,sample_index\n"
                "5,2414.000 MHz,BLE_CONN,0\n"
                "17,2440.000 MHz,BLE_CONN,1\n"
                "24,2454.000 MHz,BLE_CONN,2\n",
                encoding="utf-8",
            )
            out_csv = root / "cropped.csv"
            status = module.main([
                "--parser-csv", str(source),
                "--output-csv", str(out_csv),
                "--output-dir", str(root),
                "--center-freq-hz", "2440000000",
                "--bandwidth-hz", "20000000",
                "--run-id", "test-run",
            ])
            self.assertEqual(status, 0)
            lines = out_csv.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            summary = __import__("json").loads((root / "crop_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["in_band_rows"], 1)
            self.assertAlmostEqual(summary["c_bw_packets_estimated"], 1 / 3)

    def test_main_filters_packet_types_before_crop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "ble_packets.csv"
            source.write_text(
                "channel,center_freq_desc,packet_type,sample_index\n"
                "17,2440.000 MHz,BLE_ADV,0\n"
                "17,2440.000 MHz,BLE_CONN,1\n"
                "5,2414.000 MHz,BLE_ADV,2\n",
                encoding="utf-8",
            )
            out_csv = root / "cropped.csv"
            module.main([
                "--parser-csv", str(source),
                "--output-csv", str(out_csv),
                "--output-dir", str(root),
                "--center-freq-hz", "2440000000",
                "--bandwidth-hz", "20000000",
                "--packet-types", "BLE_ADV",
                "--run-id", "test-adv",
            ])
            lines = out_csv.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)  # header + only the ch17 BLE_ADV row
            summary = __import__("json").loads((root / "crop_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["packet_type_filter"], ["BLE_ADV"])
            self.assertEqual(summary["input_parser_rows"], 2)


if __name__ == "__main__":
    unittest.main()
