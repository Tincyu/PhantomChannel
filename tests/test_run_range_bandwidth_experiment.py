import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "run_range_bandwidth_experiment.py"
SPEC = importlib.util.spec_from_file_location("run_range_bandwidth_experiment", MODULE_PATH)
run_range_bandwidth_experiment = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_range_bandwidth_experiment
SPEC.loader.exec_module(run_range_bandwidth_experiment)


class RunRangeBandwidthExperimentTest(unittest.TestCase):
    def test_channel_map_bytes_accepts_five_byte_hex(self):
        self.assertEqual(
            run_range_bandwidth_experiment.channel_map_bytes("ff:ff:03:00:00"),
            ["ff", "ff", "03", "00", "00"],
        )

    def test_channel_map_bytes_rejects_wrong_length(self):
        with self.assertRaises(ValueError):
            run_range_bandwidth_experiment.channel_map_bytes("ffff03")

    def test_summarize_smoke_detects_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rtt = root / "rtt.log"
            bt = root / "bluetoothctl.log"
            rtt.write_text(
                "\n".join(
                    [
                        'PHANTOM_CONNECTED {"timestamp_us":1}',
                        'PHANTOM_CCC {"notify":1,"timestamp_us":2}',
                        'PHANTOM_TX {"seq":1,"covert_hex":"aa"}',
                        'PHANTOM_LL_TX {"seq":1,"channel":3}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            bt.write_text(
                "\n".join(["Connection successful", "ServicesResolved: yes", "Notify started"]) + "\n",
                encoding="utf-8",
            )

            summary = run_range_bandwidth_experiment.summarize_smoke(
                rtt,
                bt,
                {"valid": True, "has_required_phantom_match_fields": True},
            )

        self.assertTrue(summary["connected"])
        self.assertTrue(summary["services_resolved"])
        self.assertTrue(summary["notify_enabled"])
        self.assertFalse(summary["fault_detected"])
        self.assertEqual(summary["counts"]["PHANTOM_TX"], 1)
        self.assertEqual(summary["counts"]["PHANTOM_LL_TX"], 1)

    def test_summarize_smoke_accepts_tx_when_ccc_enable_log_was_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rtt = root / "rtt.log"
            bt = root / "bluetoothctl.log"
            rtt.write_text(
                "\n".join(
                    [
                        'PHANTOM_CONNECTED {"timestamp_us":1}',
                        'PHANTOM_TX {"seq":1,"covert_hex":"a5aa","status":0}',
                        'PHANTOM_LL_TX {"seq":1,"channel":3}',
                        'PHANTOM_CCC {"notify":0,"timestamp_us":9}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            bt.write_text(
                "\n".join(["Connection successful", "ServicesResolved: yes", "Notify started"]) + "\n",
                encoding="utf-8",
            )

            summary = run_range_bandwidth_experiment.summarize_smoke(
                rtt,
                bt,
                {"valid": True, "has_required_phantom_match_fields": True},
            )

        self.assertTrue(summary["notify_enabled"])
        self.assertTrue(summary["notify_tx_observed"])
        self.assertFalse(summary["ccc_notify_enabled_observed"])

    def test_dry_run_writes_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.yaml"
            output = root / "experiments"
            config.write_text(
                "\n".join(
                    [
                        "experiment:",
                        "  duration_s: 1",
                        "paths:",
                        f"  output_root: {output}",
                        "firmware:",
                        "  build_dir: /tmp/fw",
                        "  serial_number: '1'",
                        "peripheral:",
                        "  address: C0:DE:52:84:00:01",
                        "  data_characteristic_uuid: 70630002-5048-414e-544f-4d4348414e4c",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            rc = run_range_bandwidth_experiment.main(
                [
                    "--config",
                    str(config),
                    "--distance-m",
                    "1",
                    "--analysis-bandwidth-mhz",
                    "8",
                    "--environment",
                    "lab_los",
                    "--repetition",
                    "1",
                    "--capture-id",
                    "dry_run_001",
                    "--dry-run",
                ]
            )

            status = json.loads((output / "dry_run_001" / "run_status.json").read_text(encoding="utf-8"))

        self.assertEqual(rc, 0)
        self.assertEqual(status["status"], "planned")
        self.assertEqual(status["stage"], "dry_run")
        self.assertEqual(status["parameters"]["analysis_bandwidth_mhz"], "8")

    def test_summarize_smoke_rejects_short_or_failed_notify(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rtt = root / "rtt.log"
            bt = root / "bluetoothctl.log"
            rtt.write_text(
                "\n".join(
                    [
                        'PHANTOM_CONNECTED {"timestamp_us":1}',
                        'PHANTOM_CCC {"notify":1,"timestamp_us":2}',
                        'PHANTOM_TX {"seq":1,"covert_hex":"aa","status":0}',
                        'PHANTOM_LL_TX {"seq":1,"channel":3}',
                        'PHANTOM_TX {"seq":2,"covert_hex":"bb","status":-128}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            bt.write_text(
                "\n".join(["Connection successful", "ServicesResolved: yes", "Notify started"]) + "\n",
                encoding="utf-8",
            )

            summary = run_range_bandwidth_experiment.summarize_smoke(
                rtt,
                bt,
                {"valid": True, "has_required_phantom_match_fields": True},
                notify_s=10,
                notify_interval_ms=100,
                min_notify_fraction=0.5,
            )

        self.assertFalse(summary["min_tx_packets_met"])
        self.assertFalse(summary["notify_status_ok"])
        self.assertEqual(summary["phantom_tx_status_counts"]["-128"], 1)

    def test_summarize_smoke_allows_transient_notify_buffer_pressure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rtt = root / "rtt.log"
            bt = root / "bluetoothctl.log"
            rtt.write_text(
                "\n".join(
                    [
                        'PHANTOM_CONNECTED {"timestamp_us":1}',
                        'PHANTOM_CCC {"notify":1,"timestamp_us":2}',
                        'PHANTOM_TX {"seq":1,"covert_hex":"aa","status":0}',
                        'PHANTOM_LL_TX {"seq":1,"channel":3}',
                        'PHANTOM_TX {"seq":2,"covert_hex":"bb","status":-12}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            bt.write_text(
                "\n".join(["Connection successful", "ServicesResolved: yes", "Notify started"]) + "\n",
                encoding="utf-8",
            )

            summary = run_range_bandwidth_experiment.summarize_smoke(
                rtt,
                bt,
                {"valid": True, "has_required_phantom_match_fields": True},
                notify_s=1,
                notify_interval_ms=100,
                min_notify_fraction=0.1,
            )

        self.assertTrue(summary["min_tx_packets_met"])
        self.assertTrue(summary["notify_status_ok"])
        self.assertEqual(summary["nonzero_notify_status_count"], 1)

    def test_finalize_b210_capture_validates_iq_and_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            iq_dir = root / "iq"
            iq_dir.mkdir()
            iq_path = iq_dir / "capture.sc16"
            tags_path = iq_dir / "rx_time_tags_40m.csv"
            log_path = iq_dir / "capture.log"
            metadata_path = iq_dir / "metadata.json"
            rtt_log = root / "ground_truth" / "peripheral_rtt.log"
            rtt_log.parent.mkdir()
            iq_path.write_bytes(b"\x01\x02\x03\x04" * 4)
            tags_path.write_text("host_epoch_ns,rx_time_full_secs,rx_time_frac_secs\n", encoding="utf-8")
            rtt_log.write_text('PHANTOM_TX {"seq":1}\n', encoding="utf-8")
            log_path.write_text(
                "\n".join(
                    [
                        "UHD version: 4.6.0.0",
                        "Actual B210 serial: B210_SERIAL",
                        "Actual RX freq: 2420000000",
                        "Actual RX rate: 40000000",
                        "Actual RX gain: 50",
                        "Actual RX bandwidth: 56000000",
                        "Actual RX antenna: RX2",
                        "Done. samples=4, blocks=1, overflows=0, gaps=0, udp_drops=0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            capture = run_range_bandwidth_experiment.B210Capture(
                process=None,
                command=["capture"],
                env={},
                iq_dir=iq_dir,
                iq_path=iq_path,
                tags_path=tags_path,
                log_path=log_path,
                metadata_path=metadata_path,
                requested_duration_s=1.0,
                start_epoch_ns=1,
                ready_epoch_ns=2,
                end_epoch_ns=3,
                returncode=0,
            )

            metadata = run_range_bandwidth_experiment.finalize_b210_capture(
                capture,
                "b210_test",
                rtt_log,
                compute_sha256=False,
            )

        self.assertTrue(metadata["valid_no_overflow_or_gap"])
        self.assertEqual(metadata["file_size_bytes"], 16)
        self.assertEqual(metadata["expected_file_size_bytes"], 16)
        self.assertEqual(metadata["overflows"], 0)
        self.assertEqual(metadata["gaps"], 0)
        self.assertFalse(metadata["sha256_enabled"])
        self.assertNotIn("iq_sha256", metadata)

    def test_build_sdr_parser_command_uses_capture_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ble_root = root / "receiver"
            entrypoint = ble_root / "experiment" / "bt_40m_pfb_realtime.py"
            python = ble_root / ".venv-cuda" / "bin" / "python"
            entrypoint.parent.mkdir(parents=True)
            python.parent.mkdir(parents=True)
            entrypoint.write_text("# parser\n", encoding="utf-8")
            python.write_text("# python\n", encoding="utf-8")
            run_root = root / "experiments" / "run_001"
            iq_dir = run_root / "iq"
            iq_dir.mkdir(parents=True)
            (iq_dir / "capture.sc16").write_bytes(b"\x00" * 16)
            (iq_dir / "rx_time_tags_40m.csv").write_text("header\n", encoding="utf-8")
            (iq_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "actual_sample_rate_sps": 40000000,
                        "actual_center_frequency_hz": 2420000000,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config = {
                "capture": {},
                "parser": {
                    "source_project_root": str(ble_root),
                    "python": str(python),
                    "entrypoint": "experiment/bt_40m_pfb_realtime.py",
                    "processing_bandwidth_policy": "analysis",
                    "parser_cpus": "0-3",
                    "skip_bredr": True,
                    "learned_parser_fast_path": False,
                },
            }

            command, cwd, env = run_range_bandwidth_experiment.build_sdr_parser_command(
                config,
                run_root,
                max_chunks=2,
                analysis_bandwidth_hz=8000000.0,
            )

        self.assertEqual(command[:3], ["taskset", "-c", "0-3"])
        self.assertEqual(cwd, ble_root)
        self.assertIn(str(run_root / "iq" / "capture.sc16"), command)
        self.assertIn(str(run_root / "iq" / "metadata.json"), command)
        self.assertIn(str(run_root / "iq" / "rx_time_tags_40m.csv"), command)
        self.assertIn(str(run_root / "sdr"), command)
        self.assertEqual(command[command.index("--bandwidth") + 1], "8000000.0")
        self.assertIn("--skip-bredr", command)
        self.assertIn("--no-learned-parser-fast-path", command)
        self.assertIn("--max-chunks", command)
        self.assertIn(str(ble_root / "build-native"), env["PYTHONPATH"])

    def test_build_sdr_parser_command_can_keep_fixed_processing_bandwidth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ble_root = root / "receiver"
            entrypoint = ble_root / "experiment" / "bt_40m_pfb_realtime.py"
            python = ble_root / ".venv-cuda" / "bin" / "python"
            entrypoint.parent.mkdir(parents=True)
            python.parent.mkdir(parents=True)
            entrypoint.write_text("# parser\n", encoding="utf-8")
            python.write_text("# python\n", encoding="utf-8")
            run_root = root / "experiments" / "run_001"
            iq_dir = run_root / "iq"
            iq_dir.mkdir(parents=True)
            (iq_dir / "capture.sc16").write_bytes(b"\x00" * 16)
            (iq_dir / "rx_time_tags_40m.csv").write_text("header\n", encoding="utf-8")
            (iq_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "actual_sample_rate_sps": 40000000,
                        "actual_center_frequency_hz": 2420000000,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config = {
                "capture": {},
                "parser": {
                    "source_project_root": str(ble_root),
                    "python": str(python),
                    "entrypoint": "experiment/bt_40m_pfb_realtime.py",
                    "processing_bandwidth_policy": "fixed",
                    "processing_bandwidth_hz": 20000000,
                    "skip_bredr": True,
                },
            }

            command, _cwd, _env = run_range_bandwidth_experiment.build_sdr_parser_command(
                config,
                run_root,
                analysis_bandwidth_hz=8000000.0,
            )

        self.assertEqual(command[command.index("--bandwidth") + 1], "20000000.0")


if __name__ == "__main__":
    unittest.main()
