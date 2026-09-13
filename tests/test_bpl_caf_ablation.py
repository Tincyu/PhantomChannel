from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from tools.analyze_bpl_caf_failures import aggregate_funnel, recompute_mode_metrics
from tools.bpl_sync import BPLConfig, generate_gfsk_template, ble_preamble_bits
from tools.run_bpl_caf_ablation import (
    MODES,
    add_posthoc_reference,
    guard_unchanged,
    narrowband_burst,
    read_iq,
    run_burst_modes,
)


def test_crop_sc16_alignment(tmp_path: Path) -> None:
    path = tmp_path / "capture.sc16"
    values = np.asarray([[index, -index] for index in range(100)], dtype="<i2")
    values.tofile(path)
    mapped = np.memmap(path, dtype="<i2", mode="r", shape=(100, 2))
    result = read_iq(mapped, 10, 3)
    assert result.tolist() == [10 - 10j, 11 - 11j, 12 - 12j]


def test_wideband_subband_sample_mapping(tmp_path: Path) -> None:
    values = np.asarray([[index, 0] for index in range(400)], dtype="<i2")
    path = tmp_path / "capture.sc16"
    values.tofile(path)
    mapped = np.memmap(path, dtype="<i2", mode="r", shape=(400, 2))
    burst = {"physical_burst_start_sample": 100, "physical_burst_end_sample": 200}
    narrow, raw_start = narrowband_burst(
        mapped,
        burst,
        sample_rate_sps=8_000_000.0,
        subband_sample_rate_sps=4_000_000.0,
        frequency_offset_hz=0.0,
        pre_margin_us=0.0,
        post_margin_us=0.0,
    )
    assert raw_start == 100
    assert 49 <= narrow.size <= 51


def test_four_modes_share_burst_ids_and_switches(tmp_path: Path) -> None:
    config = {
        "sample_rate_sps": 4_000_000.0,
        "subband_sample_rate_sps": 4_000_000.0,
        "bandwidth_hz": 2_000_000.0,
        "physical_detector": {"frequency_fft_samples": 1024},
        "caf_offset_samples": [-2, -1, 0, 1, 2],
    }
    bpl_config = BPLConfig(min_normalized_peak=0.01, min_peak_ratio=1.0, cfo_search_hz=(0.0,))
    template = generate_gfsk_template(ble_preamble_bits("LE_1M", polarity=0), bpl_config)
    narrow = np.zeros(800, dtype=np.complex64)
    narrow[100 : 100 + template.size] = template
    # A raw 4 MS/s temporary capture is enough to exercise the mode contract.
    raw = np.asarray([[int(value.real * 1000), int(value.imag * 1000)] for value in narrow], dtype="<i2")
    raw_path = tmp_path / "capture.sc16"
    raw.tofile(raw_path)
    mapped = np.memmap(raw_path, dtype="<i2", mode="r", shape=(len(raw), 2))
    burst = {
        "burst_id": "burst_000001",
        "physical_burst_start_sample": 0,
        "physical_burst_end_sample": len(raw),
    }
    bpl_row, candidates, stage_rows = run_burst_modes(
        burst,
        memmap=mapped,
        config=config,
        bpl_config=bpl_config,
    )
    assert bpl_row["burst_id"] == "burst_000001"
    assert {row["burst_id"] for row in stage_rows} == {"burst_000001"}
    assert {row["mode"] for row in stage_rows} == set(MODES)
    assert len([row for row in candidates if row["mode"] == "caf_only"]) == 5
    assert len([row for row in candidates if row["mode"] == "bpl_only"]) <= 1
    caf_rows = [row for row in candidates if row["mode"] == "caf_only"]
    bpl_rows = [row for row in candidates if row["mode"] == "bpl_only"]
    assert all("bpl_accepted" not in row for row in caf_rows)
    assert len(bpl_rows) <= 1
    assert all(not row.get("caf_accepted", False) for row in bpl_rows)


def test_bpl_only_does_not_call_caf_and_caf_only_does_not_read_bpl_metrics(tmp_path: Path) -> None:
    # The mode contract is visible in candidate schemas: BPL-only has one
    # start and no CAF acceptance; CAF-only has an offset scan but no BPL
    # fields.  This prevents a future refactor from silently sharing the
    # other module's selection metric.
    config = {
        "sample_rate_sps": 4_000_000.0,
        "subband_sample_rate_sps": 4_000_000.0,
        "bandwidth_hz": 2_000_000.0,
        "physical_detector": {"frequency_fft_samples": 256},
        "caf_offset_samples": [-1, 0, 1],
    }
    path = tmp_path / "capture.sc16"
    np.zeros((400, 2), dtype="<i2").tofile(path)
    mapped = np.memmap(path, dtype="<i2", mode="r", shape=(400, 2))
    burst = {"burst_id": "b", "physical_burst_start_sample": 0, "physical_burst_end_sample": 400}
    _, candidates, _ = run_burst_modes(
        burst,
        memmap=mapped,
        config=config,
        bpl_config=BPLConfig(cfo_search_hz=(0.0,)),
    )
    assert len([row for row in candidates if row["mode"] == "caf_only"]) == 3
    assert len([row for row in candidates if row["mode"] == "bpl_only"]) <= 1
    assert all("bpl_accepted" not in row for row in candidates if row["mode"] == "caf_only")
    assert all(not row.get("caf_accepted", False) for row in candidates if row["mode"] == "bpl_only")


def test_funnel_closes_and_reports_both_denominators() -> None:
    rows = [
        {"condition": "LOS", "mode": "bpl_caf", "window": "smoke", "burst_id": "a", "stage": "S6_CAF_ACCEPTED"},
        {"condition": "LOS", "mode": "bpl_caf", "window": "smoke", "burst_id": "b", "stage": "S3_AA_RECOVERED"},
    ]
    funnel = aggregate_funnel(rows)
    assert len(funnel) == 8
    s0 = next(row for row in funnel if row["stage"] == "S0_PHYSICAL_BURST")
    s6 = next(row for row in funnel if row["stage"] == "S6_CAF_ACCEPTED")
    assert s0["physical_bursts"] == 2
    assert s6["cumulative_count"] == 1
    assert s6["cumulative_rate_physical"] == 0.5
    assert s6["conditional_rate_previous_stage"] <= 1.0


def test_conditional_and_cumulative_caf_denominators() -> None:
    rows = [
        {
            "run_id": "r",
            "condition": "LOS",
            "window": "w",
            "mode": "bpl_caf",
            "burst_id": "a",
            "stage": "S6_CAF_ACCEPTED",
            "bpl_accepted": "True",
            "caf_accepted": "True",
            "posthoc_target_exact": "False",
        },
        {
            "run_id": "r",
            "condition": "LOS",
            "window": "w",
            "mode": "bpl_caf",
            "burst_id": "b",
            "stage": "S1_BPL_ACCEPTED",
            "bpl_accepted": "True",
            "caf_accepted": "False",
            "posthoc_target_exact": "False",
        },
    ]
    result = recompute_mode_metrics(rows)
    row = next(item for item in result if item["mode"] == "bpl_caf")
    assert row["physical_bursts"] == 2
    assert row["caf_accepted"] == 1
    assert row["caf_candidate_count"] == 2
    assert row["caf_conditional_on_candidates"] == 0.5


def test_posthoc_exact_does_not_promote_non_caf_stage(tmp_path: Path) -> None:
    parser_dir = tmp_path / "results"
    parser_dir.mkdir()
    with (parser_dir / "parser_candidate_packets.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_index", "pattern_exact", "integrity_ok"])
        writer.writeheader()
        writer.writerow({"sample_index": "100", "pattern_exact": "1", "integrity_ok": "1"})
    rows = [
        {"mode": "baseline", "stage": "S5_CRC_OR_STRUCTURE_VALID", "candidate_start_sample": "100", "narrowband_raw_start_sample": "0"},
        {"mode": "caf_only", "stage": "S6_CAF_ACCEPTED", "candidate_start_sample": "100", "narrowband_raw_start_sample": "0"},
    ]
    add_posthoc_reference(rows, tmp_path, 4_000_000.0, 4_000_000.0)
    assert rows[0]["posthoc_target_exact"] == "1"
    assert rows[0]["stage"] == "S5_CRC_OR_STRUCTURE_VALID"
    assert rows[1]["stage"] == "S7_TARGET_EXACT"


def test_ble_guard_compares_external_state_only() -> None:
    before = {"ble_encrypt_check": {"head": "a", "status": [" D file"]}}
    after = {"ble_encrypt_check": {"head": "a", "status": [" D file"]}}
    changed = {"ble_encrypt_check": {"head": "b", "status": [" D file"]}}
    assert guard_unchanged(before, after)
    assert not guard_unchanged(before, changed)
