import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "parse_rtt_ground_truth.py"
SPEC = importlib.util.spec_from_file_location("parse_rtt_ground_truth", MODULE_PATH)
parse_rtt_ground_truth = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = parse_rtt_ground_truth
SPEC.loader.exec_module(parse_rtt_ground_truth)


class ParseRttGroundTruthTest(unittest.TestCase):
    def test_parse_existing_evt_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "rtt.log"
            log.write_text(
                "\n".join(
                    [
                        "EVT,BOOT,time_ms=3,bt_enable=0,name=BLE_EVAL_PERIPH",
                        "EVT,CONN_EVENT,time_ms=1612,handle=0,event_counter=0,channel=24,crc_ok=1,crc_error=0,nak=0,rx_timeout=0",
                        "EVT,CONN_ANCHOR,time_ms=1612,handle=0,event_counter=0,anchor_point_us=2043285",
                        "EVT,CONN_EVENT,time_ms=1657,handle=0,event_counter=1,channel=12,crc_ok=1,crc_error=0,nak=0,rx_timeout=0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = parse_rtt_ground_truth.parse_rtt_log(log, run_id="run_001")

        self.assertTrue(result.status["valid"])
        self.assertEqual(result.status["phantom_tx_count"], 0)
        self.assertFalse(result.status["has_required_phantom_match_fields"])
        self.assertTrue(result.status["counter_contiguous"])
        self.assertEqual(len(result.event_rows), 2)
        self.assertEqual(result.event_rows[1]["event_counter_unwrapped"], 1)
        self.assertEqual(len(result.anchor_rows), 1)

    def test_parse_phantom_json_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "rtt.log"
            log.write_text(
                'PHANTOM_TX {"session_id":1,"seq":7,"conn_event":11,"channel":18,'
                '"phy":"1M","normal_pdu_len":27,"covert_len":3,'
                '"covert_marker_hex":"A5","covert_hex":"A5BBCC",'
                '"covert_data_len":2,"covert_data_hex":"BBCC","timestamp_us":123456}\n',
                encoding="utf-8",
            )

            result = parse_rtt_ground_truth.parse_rtt_log(log, run_id="run_002")

        self.assertTrue(result.status["valid"])
        self.assertTrue(result.status["has_required_phantom_match_fields"])
        self.assertEqual(result.ground_truth_rows[0]["run_id"], "run_002")
        self.assertEqual(result.ground_truth_rows[0]["seq"], 7)
        self.assertEqual(result.ground_truth_rows[0]["covert_marker_hex"], "a5")
        self.assertEqual(result.ground_truth_rows[0]["covert_hex"], "a5bbcc")
        self.assertEqual(result.ground_truth_rows[0]["covert_len_bytes"], 3)
        self.assertEqual(result.ground_truth_rows[0]["covert_data_hex"], "bbcc")
        self.assertEqual(result.ground_truth_rows[0]["covert_data_len_bytes"], 2)

    def test_derives_covert_data_from_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "rtt.log"
            log.write_text(
                'PHANTOM_TX {"seq":8,"covert_marker_hex":"a5","covert_hex":"a50102"}\n',
                encoding="utf-8",
            )

            result = parse_rtt_ground_truth.parse_rtt_log(log, run_id="run_002")

        self.assertEqual(result.ground_truth_rows[0]["covert_data_hex"], "0102")
        self.assertEqual(result.ground_truth_rows[0]["covert_data_len_bytes"], 2)

    def test_reconstructs_fixed_compact_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "rtt.log"
            log.write_text(
                'PHANTOM_TX {"seq":9,"covert_len":5,"covert_marker_hex":"a5",'
                '"covert_data_len":4,"covert_pattern":"marker_fixed_01_to_ff"}\n',
                encoding="utf-8",
            )

            result = parse_rtt_ground_truth.parse_rtt_log(log, run_id="run_004")

        self.assertTrue(result.status["valid"])
        self.assertTrue(result.status["has_required_phantom_match_fields"])
        self.assertEqual(result.ground_truth_rows[0]["covert_hex"], "a501020304")
        self.assertEqual(result.ground_truth_rows[0]["covert_data_hex"], "01020304")

    def test_parse_phantom_ll_tx_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "rtt.log"
            log.write_text(
                'PHANTOM_LL_TX {"seq":7,"channel":18,'
                '"access_address":"0x8e89bed6","parser_access_address":"0xd6be898e",'
                '"crc_init":"0x555555","normal_pdu_len":1,'
                '"air_extra_len":22,"covert_len":16}\n',
                encoding="utf-8",
            )

            result = parse_rtt_ground_truth.parse_rtt_log(log, run_id="run_003")

        self.assertTrue(result.status["valid"])
        self.assertEqual(result.status["phantom_ll_tx_count"], 1)
        self.assertEqual(result.ll_tx_rows[0]["run_id"], "run_003")
        self.assertEqual(result.ll_tx_rows[0]["seq"], 7)
        self.assertEqual(result.ll_tx_rows[0]["channel"], 18)
        self.assertEqual(result.ll_tx_rows[0]["access_address"], "8e89bed6")
        self.assertEqual(result.ll_tx_rows[0]["parser_access_address"], "d6be898e")
        self.assertEqual(result.ll_tx_rows[0]["crc_init"], "555555")
        self.assertEqual(result.ll_tx_rows[0]["covert_len_bytes"], 16)

    def test_compact_phantom_ll_records_inherit_connection_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "rtt.log"
            log.write_text(
                'PHANTOM_LL_META {"access_address":"0x8a270ec7",'
                '"parser_access_address":"0xc70e278a","crc_init":"0xbd4b0b",'
                '"normal_pdu_len":8,"air_extra_len":237,"covert_len":231,'
                '"covert_marker_hex":"a5","covert_pattern":"marker_fixed_01_to_ff"}\n'
                'PHANTOM_LL_TX {"seq":185,"channel":9}\n'
                'PHANTOM_LL_STATS {"enqueued":192,"queue_drops":0,"emitted":192}\n',
                encoding="utf-8",
            )

            result = parse_rtt_ground_truth.parse_rtt_log(log, run_id="run_005")

        self.assertTrue(result.status["valid"])
        self.assertEqual(result.status["phantom_ll_meta_count"], 1)
        self.assertEqual(result.status["controller_ll_stats"]["queue_drops"], 0)
        self.assertEqual(result.ll_tx_rows[0]["seq"], 185)
        self.assertEqual(result.ll_tx_rows[0]["channel"], 9)
        self.assertEqual(result.ll_tx_rows[0]["access_address"], "8a270ec7")
        self.assertEqual(result.ll_tx_rows[0]["covert_len_bytes"], 231)

    def test_detects_jlink_rtt_message_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "rtt.log"
            log.write_text("--- 3 messages dropped ---\n", encoding="utf-8")

            result = parse_rtt_ground_truth.parse_rtt_log(log, run_id="run_006")

        self.assertFalse(result.status["valid"])
        self.assertEqual(result.status["drop_record_count"], 1)
        self.assertEqual(result.status["drop_records"][0]["count"], 3)

    def test_missing_required_phantom_fields_marks_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "rtt.log"
            log.write_text('PHANTOM_TX {"session_id":1,"covert_hex":"AA"}\n', encoding="utf-8")

            result = parse_rtt_ground_truth.parse_rtt_log(log)

        self.assertFalse(result.status["valid"])
        self.assertEqual(
            result.status["ground_truth_records_missing_required_fields"][0]["missing"],
            ["seq"],
        )


if __name__ == "__main__":
    unittest.main()
