import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "run_x310_phone_hrs_experiment.py"
SPEC = importlib.util.spec_from_file_location("run_x310_phone_hrs_experiment", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RunX310PhoneHrsExperimentTest(unittest.TestCase):
    def test_phone_manifest_marks_rtt_and_bluez_unavailable(self):
        cli = MODULE.build_parser().parse_args([
            "--phone-ready",
            "--phone-model", "test-phone",
            "--phone-os", "test-os",
            "--phone-app", "test-app",
        ])
        manifest = MODULE.phone_manifest(cli, {"phone": {"device_name": "PhantomHRS"}}, "run")
        self.assertTrue(manifest["phone_ready_confirmed"])
        self.assertFalse(manifest["rtt_available"])
        self.assertFalse(manifest["bluez_central_used"])
        self.assertEqual(manifest["hrs_measurement_uuid"], "2a37")

    def test_copy_refuses_overwrite_and_verifies_iq(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "nvme" / "run"
            target_root = root / "pssd"
            (source / "iq").mkdir(parents=True)
            (source / "diagnostics/one_stage_cpp").mkdir(parents=True)
            (source / "results").mkdir(parents=True)
            (source / "iq/capture.sc16").write_bytes(b"iq")
            (source / "iq/metadata.json").write_text("{}\n", encoding="utf-8")
            (source / "diagnostics/one_stage_cpp/ble_packets.csv").write_text("x\n", encoding="utf-8")
            (source / "results/parser_candidate_rate.json").write_text("{}\n", encoding="utf-8")
            manifest = MODULE.copy_and_verify(source, target_root, "run", delete_nvme_iq=False)
            self.assertTrue(manifest["verified"])
            self.assertEqual(manifest["iq_source_sha256"], manifest["iq_target_sha256"])
            self.assertTrue(manifest["nvme_source_retained"])
            self.assertTrue((source / "iq/capture.sc16").is_file())
            with self.assertRaises(FileExistsError):
                MODULE.copy_and_verify(source, target_root, "run", delete_nvme_iq=False)

    def test_copy_deletes_nvme_iq_after_verified_copy_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "nvme" / "run"
            target_root = root / "pssd"
            (source / "iq").mkdir(parents=True)
            (source / "diagnostics/one_stage_cpp").mkdir(parents=True)
            (source / "results").mkdir(parents=True)
            (source / "iq/capture.sc16").write_bytes(b"iq")
            (source / "iq/metadata.json").write_text("{}\n", encoding="utf-8")
            (source / "diagnostics/one_stage_cpp/ble_packets.csv").write_text("x\n", encoding="utf-8")
            (source / "results/parser_candidate_rate.json").write_text("{}\n", encoding="utf-8")
            manifest = MODULE.copy_and_verify(source, target_root, "run")
            self.assertTrue(manifest["verified"])
            self.assertTrue(manifest["nvme_iq_deleted"])
            self.assertFalse(manifest["nvme_source_retained"])
            self.assertFalse((source / "iq/capture.sc16").exists())
            self.assertTrue((source / "iq/metadata.json").is_file())
            self.assertTrue((target_root / "run/iq/capture.sc16").is_file())
            self.assertEqual(manifest["iq_source_sha256"], manifest["iq_target_sha256"])

    def test_keep_nvme_iq_flag_retains_source(self):
        cli = MODULE.build_parser().parse_args(["--keep-nvme-iq"])
        self.assertTrue(cli.keep_nvme_iq)

    def test_copy_to_pssd_deletes_iq_without_copy_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "nvme" / "run"
            (source / "iq").mkdir(parents=True)
            (source / "iq/capture.sc16").write_bytes(b"iq")
            (source / "iq/metadata.json").write_text("{}\n", encoding="utf-8")
            manifest = MODULE.copy_to_pssd(source, root / "pssd", "run")
            self.assertTrue(manifest["nvme_iq_deleted"])
            self.assertNotIn("verified", manifest)
            self.assertFalse((source / "iq/capture.sc16").exists())
            self.assertTrue((root / "pssd/run/iq/capture.sc16").is_file())

    def test_tail_summary_prints_metric_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            status = {
                "run_id": "run",
                "paths": {"run_root_nvme": str(run_root)},
                "candidate_score": {
                    "rate": {
                        "marker_valid_candidates": 478,
                        "seq_unique_count": 476,
                        "iq_window_parser_candidate_data_kbps": 90.9006507,
                        "psr_exact": 0.957983193,
                        "pattern_exact_packets": 456,
                        "pattern_candidates_compared": 476,
                        "pattern_byte_recovery": 0.997444432,
                        "pattern_ber": 0.000461548,
                        "rate_pc_frame_length_distribution": {"237": 476},
                        "seq_unique_count": 476,
                        "seq_span_theoretical_packets": 483,
                        "recovered_within_span_fraction": 0.9772,
                    }
                },
            }
            captured = io.StringIO()
            with redirect_stdout(captured):
                MODULE.print_tail_summary(status)
            text = captured.getvalue()
            self.assertIn("候选 / 唯一 seq\t478 / 476", text)
            self.assertIn("G_e2e\t90.90 kbps", text)
            self.assertIn("PSR_exact\t95.80%（456/476 逐字节一致）", text)
            self.assertIn("byte recovery\t99.74%", text)
            self.assertIn("帧长度\t全部 237 B（231+6），476 包无杂帧", text)
            self.assertIn("seq 理论窗口包数（主体跨度）\t483", text)
            self.assertIn("恢复率（候选 / 理论跨度）\t97.7%", text)
            self.assertIn("C_bw（候选级）\t未计算", text)

    def test_tail_summary_includes_c_bw_when_crops_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            (run_root / "results/bandwidth_crops").mkdir(parents=True)
            (run_root / "results/bandwidth_crops/bandwidth_crop_metrics.json").write_text(
                json.dumps([
                    {"bandwidth_mhz": 20.0, "c_bw_candidates_estimated": 0.1827},
                    {"bandwidth_mhz": 80.0, "c_bw_candidates_estimated": 1.0},
                    {"bandwidth_mhz": None, "c_bw_candidates_estimated": 1.0},
                ]),
                encoding="utf-8",
            )
            status = {
                "run_id": "run",
                "paths": {"run_root_nvme": str(run_root)},
                "candidate_score": {
                    "rate": {
                        "marker_valid_candidates": 1,
                        "seq_unique_count": 1,
                        "iq_window_parser_candidate_data_kbps": 1.0,
                        "rate_pc_frame_length_distribution": {"237": 1},
                    }
                },
            }
            captured = io.StringIO()
            with redirect_stdout(captured):
                MODULE.print_tail_summary(status)
            self.assertIn("20 MHz:18.3%", captured.getvalue())
            self.assertIn("80 MHz:100.0%", captured.getvalue())

    def test_tail_summary_prints_psr_ber_outlier_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            rate = {
                "marker_valid_candidates": 500,
                "seq_unique_count": 489,
                "iq_window_parser_candidate_data_kbps": 95.8,
                "iq_capture_duration_s": 9.999978,
                "pattern_ber": 0.0233,
                "pattern_ber_excluding_outliers": 0.0011,
                "pattern_outlier_ber_threshold": 0.10,
                "pattern_outlier_packets": 42,
                "pattern_outlier_fraction": 0.084,
                "pattern_outlier_error_share": 0.956,
                "psr_exact": 0.826,
                "pattern_exact_packets": 404,
                "pattern_candidates_compared": 489,
                "pattern_byte_recovery": 0.956,
                "rate_pc_frame_length_distribution": {"245": 489},
            }
            metrics = MODULE.build_metrics(rate, 20, 10.0, covert_len=239)
            status = {
                "run_id": "run",
                "paths": {"run_root_nvme": str(run_root)},
                "candidate_score": {"rate": rate},
                "metrics": {"summary": metrics, "rate": rate},
            }
            captured = io.StringIO()
            with redirect_stdout(captured):
                MODULE.print_tail_summary(status)
            text = captured.getvalue()
            self.assertIn("PSR\t97.8%（489/500）", text)
            self.assertIn("PSR 边界范围\t97.4%–98.2%", text)
            self.assertIn("BER\t2.33%", text)
            self.assertIn("BER（剔除冲突包，>10%）\t0.11%", text)
            self.assertIn("冲突包数 / 冲突包率\t42 / 8.4%（42/500）", text)

    def test_tail_summary_warns_on_zero_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = {
                "run_id": "run",
                "paths": {"run_root_nvme": str(Path(tmp) / "run")},
                "candidate_score": {
                    "rate": {
                        "marker_valid_candidates": 0,
                        "seq_unique_count": 0,
                        "input_parser_rows": 0,
                    }
                },
            }
            captured = io.StringIO()
            with redirect_stdout(captured):
                MODULE.print_tail_summary(status)
            text = captured.getvalue()
            self.assertIn("候选 / 唯一 seq\t0 / 0", text)
            self.assertIn("未恢复任何 Phantom 帧", text)

    def test_overflow_detection_ignores_disk_write_warning(self):
        capture_status = {
            "capture": {
                "overflow_text_matches": [
                    "Disk write test indicates that an overflow is likely to occur.",
                    "OGot an overflow indication. Please consider the following:",
                ]
            }
        }
        lines = MODULE.detect_overflow_indications(capture_status)
        self.assertEqual(len(lines), 1)
        self.assertIn("overflow indication", lines[0].lower())

    def test_overflow_detection_empty_when_no_real_overflow(self):
        capture_status = {
            "capture": {
                "overflow_text_matches": [
                    "Disk write test indicates that an overflow is likely to occur."
                ]
            }
        }
        self.assertEqual(MODULE.detect_overflow_indications(capture_status), [])

    def test_verify_pssd_run_checks_required_files_and_iq_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            (run_root / "iq").mkdir(parents=True)
            (run_root / "diagnostics/one_stage_cpp").mkdir(parents=True)
            (run_root / "results").mkdir(parents=True)
            (run_root / "iq/capture.sc16").write_bytes(b"\x00" * 8)
            (run_root / "iq/metadata.json").write_text(
                json.dumps({"samples": 2, "bytes_per_complex_sample": 4}),
                encoding="utf-8",
            )
            (run_root / "diagnostics/one_stage_cpp/ble_packets.csv").write_text("x\n", encoding="utf-8")
            (run_root / "results/parser_candidate_rate.json").write_text("{}\n", encoding="utf-8")
            manifest = MODULE.verify_pssd_run(run_root, "run")
            self.assertTrue(manifest["verified"])
            self.assertEqual(manifest["required_files_missing"], [])
            self.assertTrue(manifest["iq_size_matches_metadata"])
            self.assertTrue((run_root / "pssd_verify_manifest.json").is_file())

    def test_verify_pssd_run_flags_missing_parser_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            (run_root / "iq").mkdir(parents=True)
            (run_root / "iq/capture.sc16").write_bytes(b"\x00" * 8)
            (run_root / "iq/metadata.json").write_text(
                json.dumps({"samples": 2, "bytes_per_complex_sample": 4}),
                encoding="utf-8",
            )
            manifest = MODULE.verify_pssd_run(run_root, "run")
            self.assertFalse(manifest["verified"])
            self.assertTrue(any("ble_packets.csv" in item for item in manifest["required_files_missing"]))

    def test_build_capture_args_targets_pssd_by_default(self):
        cli = MODULE.build_parser().parse_args(["--phone-ready"])
        config = {"capture": {}, "parser": {}}
        args = MODULE.build_capture_args(cli, config, "run", "pssd")
        self.assertEqual(args.target, "pssd")
        self.assertEqual(args.pssd_root, cli.pssd_root.expanduser().resolve())
        self.assertEqual(args.gain_db, 50.0)

    def test_build_capture_args_honors_gain_override(self):
        cli = MODULE.build_parser().parse_args(["--phone-ready", "--gain-db", "30"])
        config = {"capture": {"gain_db": 50}, "parser": {}}
        args = MODULE.build_capture_args(cli, config, "run", "pssd")
        self.assertEqual(args.gain_db, 30.0)

    def test_dry_run_plan_uses_nvme_staging_by_default(self):
        captured = io.StringIO()
        with redirect_stdout(captured):
            status = MODULE.main(["--phone-ready", "--dry-run", "--capture-id", "run"])
        self.assertEqual(status, 0)
        plan = json.loads(captured.getvalue())
        self.assertEqual(plan["capture_target"], "nvme")
        self.assertEqual(plan["capture"]["gain_db"], 50.0)
        self.assertEqual(plan["capture"]["ble_threshold"], 0.02)
        self.assertIn("nvme_capture", plan["flow"])
        self.assertIn("pssd_copy", plan["flow"])
        self.assertIn("one_stage_cpp_parse", plan["flow"])
        self.assertNotIn("pssd_verify", plan["flow"])

    def test_dry_run_plan_honors_ble_threshold_override(self):
        captured = io.StringIO()
        with redirect_stdout(captured):
            status = MODULE.main(["--phone-ready", "--dry-run", "--capture-id", "run", "--ble-threshold", "0.015"])
        self.assertEqual(status, 0)
        plan = json.loads(captured.getvalue())
        self.assertEqual(plan["capture"]["ble_threshold"], 0.015)

    def test_dry_run_has_rate_only_scope(self):
        config = MODULE.load_config(Path(__file__).resolve().parents[1] / "configs/x310_phone_hrs_distance_experiment.yaml")
        self.assertFalse(config["experiment"]["distance_experiment_enabled"])


if __name__ == "__main__":
    unittest.main()
