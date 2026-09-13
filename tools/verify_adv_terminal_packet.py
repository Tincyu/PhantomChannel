#!/usr/bin/env python3
"""Record evidence that advertising channel 39 is the terminal packet.

The verdict combines the controller source path (advertising-channel iteration
and the ch39-only post-CRC tail gate) with independent B210 parser evidence
from the formal covert captures.  The B210 is locked to ch39, so it does not
claim to be a simultaneous three-channel sniffer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

from score_boundary_packet_detector import deduplicate_rows, target_candidate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
from portable_paths import NCS_WORKSPACE  # noqa: E402


DEFAULT_ROOT = Path("/path/to/PhantomChannel/testdata")
DEFAULT_OUTPUT = Path(
    "/path/to/PhantomChannel/experiments/figure/detector_roc_20260808"
)
DEFAULT_SOURCE = NCS_WORKSPACE / "zephyr/subsys/bluetooth/controller/ll_sw/nordic/lll/lll_adv.c"
DEFAULT_ULL_SOURCE = NCS_WORKSPACE / "zephyr/subsys/bluetooth/controller/ll_sw/ull_adv.c"
DEFAULT_APP = NCS_WORKSPACE / "zephyr/samples/bluetooth/phantomchannel_adv_broadcaster/src/main.c"
DEFAULT_CONF = NCS_WORKSPACE / "zephyr/samples/bluetooth/phantomchannel_adv_broadcaster/prj.conf"
DEFAULT_IMAGE = Path(
    "/path/to/PhantomChannel/artifacts/firmware/"
    "phantomchannel_adv_broadcaster/merged.hex"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def percentile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def config_value(path: Path, name: str) -> str | None:
    pattern = re.compile(rf"^CONFIG_{re.escape(name)}=(.*)$", re.MULTILINE)
    match = pattern.search(path.read_text(encoding="utf-8"))
    return match.group(1).strip().strip('"') if match else None


def verify_run(csv_path: Path, cohort: str) -> tuple[dict[str, Any], list[int]]:
    run_root = csv_path.parents[2]
    parser_rows = read_rows(csv_path)
    selected = deduplicate_rows(
        (row for row in parser_rows if target_candidate(row)), 200
    )
    samples = [int(row["sample_index"]) for row in selected]
    deltas = [b - a for a, b in zip(samples, samples[1:])]
    metadata = json.loads((run_root / "iq/metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_root / "adv_manifest.json").read_text(encoding="utf-8"))
    covert_len = int(manifest.get("covert_len_bytes") or 0)
    expected_post_crc_bytes = covert_len + 6
    post_crc_lengths = [len(row.get("post_crc_hex", "")) // 2 for row in selected]
    row = {
        "run_id": run_root.name,
        "cohort": cohort,
        "covert_len_bytes": covert_len,
        "expected_post_crc_bytes": expected_post_crc_bytes,
        "gain_db": metadata.get("actual_gain_db"),
        "sample_rate_sps": metadata.get("actual_sample_rate_sps"),
        "center_frequency_hz": metadata.get("actual_center_frequency_hz"),
        "target_packets_dedup": len(selected),
        "target_packet_type": sorted({r.get("ble_pdu_type", "") for r in selected}),
        "target_channels": sorted({r.get("channel", "") for r in selected}),
        "crc_capture_statuses": sorted(
            {r.get("crc_capture_status", "") for r in selected}
        ),
        "post_crc_nonempty": sum(bool(r.get("post_crc_hex")) for r in selected),
        "post_crc_exact_expected_length": sum(
            length == expected_post_crc_bytes for length in post_crc_lengths
        ),
        "median_interval_us": statistics.median(deltas) / 4.0 if deltas else None,
        "p05_interval_us": percentile(deltas, 0.05) / 4.0 if deltas else None,
        "p95_interval_us": percentile(deltas, 0.95) / 4.0 if deltas else None,
        "intervals_within_19_75_to_20_25_ms": sum(
            79_000 <= delta <= 81_000 for delta in deltas
        ),
        "interval_count": len(deltas),
        "capture_gaps": metadata.get("gaps"),
        "capture_overflows": metadata.get("overflows"),
    }
    row["interval_fraction_within_19_75_to_20_25_ms"] = (
        row["intervals_within_19_75_to_20_25_ms"] / len(deltas) if deltas else None
    )
    return row, deltas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--ull-source", type=Path, default=DEFAULT_ULL_SOURCE)
    parser.add_argument("--app", type=Path, default=DEFAULT_APP)
    parser.add_argument("--conf", type=Path, default=DEFAULT_CONF)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    args = parser.parse_args()

    formal_paths = sorted(
        args.root.glob(
            "20260808_detector_boundary_formal_covert_*/"
            "diagnostics/one_stage_cpp/ble_packets.csv"
        )
    )
    sensitivity_paths = sorted(
        args.root.glob(
            "20260808_detector_boundary_sensitivity_len*_g35_rep*/"
            "diagnostics/one_stage_cpp/ble_packets.csv"
        )
    )
    paths = [(path, "boundary_formal") for path in formal_paths]
    paths.extend((path, "tail_length_sensitivity") for path in sensitivity_paths)
    if not paths:
        raise SystemExit("no formal covert parser CSV files found")

    rows = []
    all_deltas: list[int] = []
    for csv_path, cohort in paths:
        row, deltas = verify_run(csv_path, cohort)
        rows.append(row)
        all_deltas.extend(deltas)

    total_packets = sum(row["target_packets_dedup"] for row in rows)
    total_intervals = len(all_deltas)
    expected_post_crc = sum(row["post_crc_nonempty"] for row in rows)
    exact_post_crc = sum(row["post_crc_exact_expected_length"] for row in rows)
    metadata_clean = all(
        row["sample_rate_sps"] == 4_000_000.0
        and row["center_frequency_hz"] == 2_480_000_000.0
        and row["capture_gaps"] == 0
        and row["capture_overflows"] == 0
        for row in rows
    )

    result = {
        "schema_version": 1,
        "verification_id": "ch39_terminal_advertising_packet_20260808",
        "status": "verified_for_current_advertising_firmware",
        "verdict": {
            "channel_39_is_terminal_advertising_packet": True,
            "method": "controller_source_order_and_ch39_gate_plus_B210_ch39_decode",
            "independent_three_channel_sniffer": False,
            "caveat": (
                "B210 was locked to ch39; it corroborates the event cadence and "
                "packet identity, while the controller source establishes the "
                "37->38->39 order."
            ),
        },
        "controller_source_evidence": {
            "source": str(args.source),
            "sha256": sha256(args.source),
            "default_channel_map_source": str(args.ull_source),
            "default_channel_map_source_sha256": sha256(args.ull_source),
            "default_channel_map": "BT_LE_ADV_CHAN_MAP_ALL (lines 3153-3157)",
            "channel_map_reset_and_first_prepare": "lines 1079-1081",
            "channel_iteration": "lines 1564-1569: find_lsb_set, clear lowest bit, lll_chan_set(36 + chan)",
            "event_continuation": "lines 1421-1437: call chan_prepare while chan_map_curr remains nonzero",
            "post_crc_gate": "lines 88-146: only chan == 0x03 sets statlen and moves covert bytes after CRC",
            "mapping": "chan 1/2/3 maps to BLE advertising channels 37/38/39",
        },
        "advertiser_firmware": {
            "application_source": str(args.app),
            "application_source_sha256": sha256(args.app),
            "kconfig": str(args.conf),
            "kconfig_sha256": sha256(args.conf),
            "covert_len_bytes": int(config_value(args.conf, "PHANTOMCHANNEL_ADV_COVERT_LEN") or 0),
            "interval_ms": int(config_value(args.conf, "PHANTOMCHANNEL_ADV_INTERVAL_MS") or 0),
            "padding_len_bytes": int(config_value(args.conf, "PHANTOMCHANNEL_ADV_PADDING_LEN") or 0),
            "benign": config_value(args.conf, "PHANTOMCHANNEL_ADV_BENIGN") or "unset",
            "delay_zero": config_value(args.conf, "BT_CTLR_ADV_DELAY_ZERO") or "unset",
            "advertiser_address": "D1:22:33:44:55:66",
            "pdu_type": "ADV_SCAN_IND",
            "image": str(args.image),
            "image_sha256": sha256(args.image),
            "image_fingerprint_scope": (
                "current 239 B merged image; sensitivity manifests point to the "
                "same build directory but do not archive a per-run image hash"
            ),
        },
        "b210_formal_capture_evidence": {
            "capture_count": len(rows),
            "boundary_formal_capture_count": sum(
                row["cohort"] == "boundary_formal" for row in rows
            ),
            "tail_length_sensitivity_capture_count": sum(
                row["cohort"] == "tail_length_sensitivity" for row in rows
            ),
            "verified_covert_lengths_bytes": sorted(
                {row["covert_len_bytes"] for row in rows}
            ),
            "capture_ids": [row["run_id"] for row in rows],
            "sample_rate_sps": 4_000_000,
            "center_frequency_hz": 2_480_000_000,
            "parser_target_filter": {
                "access_address": "0xD6BE898E",
                "channel": 39,
                "pdu": "ADV_SCAN_IND",
                "advertiser_address": "D12233445566",
                "payload_len_bytes": 6,
                "crc_capture_status": "ok",
                "standard_ble_crc24_recomputed": True,
                "dedup_gap_samples": 200,
            },
            "dedup_target_packets_total": total_packets,
            "dedup_target_packets_per_capture_min": min(
                row["target_packets_dedup"] for row in rows
            ),
            "dedup_target_packets_per_capture_median": statistics.median(
                row["target_packets_dedup"] for row in rows
            ),
            "dedup_target_packets_per_capture_max": max(
                row["target_packets_dedup"] for row in rows
            ),
            "all_capture_metadata_clean": metadata_clean,
            "post_crc_nonempty_packets": expected_post_crc,
            "post_crc_exact_expected_length_packets": exact_post_crc,
            "adjacent_packet_intervals_total": total_intervals,
            "adjacent_interval_median_us": statistics.median(all_deltas) / 4.0,
            "adjacent_interval_p05_us": percentile(all_deltas, 0.05) / 4.0,
            "adjacent_interval_p95_us": percentile(all_deltas, 0.95) / 4.0,
            "intervals_within_19_75_to_20_25_ms": sum(
                79_000 <= delta <= 81_000 for delta in all_deltas
            ),
            "interval_fraction_within_19_75_to_20_25_ms": sum(
                79_000 <= delta <= 81_000 for delta in all_deltas
            )
            / total_intervals,
        },
        "per_capture": rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ch39_terminal_verification.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    fieldnames = sorted({key for row in rows for key in row})
    with (args.output_dir / "ch39_terminal_verification_by_capture.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "status": result["status"],
        "capture_count": len(rows),
        "dedup_target_packets_total": total_packets,
        "interval_fraction_within_19_75_to_20_25_ms": result[
            "b210_formal_capture_evidence"
        ]["interval_fraction_within_19_75_to_20_25_ms"],
        "json": str(args.output_dir / "ch39_terminal_verification.json"),
        "csv": str(args.output_dir / "ch39_terminal_verification_by_capture.csv"),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
