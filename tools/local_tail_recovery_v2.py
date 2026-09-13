#!/usr/bin/env python3
"""Opt-in local multi-hypothesis Phantom tail recovery.

The decoder reads parser observations and nearby IQ from a PhantomChannel run.
It does not modify BLE_encrypt_check.  The decoder's winner is chosen only by
the known BLE prefix and the self-contained Phantom frame format/integrity;
RTT ground truth is loaded only into a separate audit CSV after decoding.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import match_rtt_sdr_results as matcher  # noqa: E402
import phantom_postprocess_scorer as postprocess  # noqa: E402


BLIND_FIELDS = [
    "run_id",
    "failure_stage_at_input",
    "observation_index",
    "candidate_sample",
    "candidate_timestamp_us",
    "access_address",
    "channel",
    "payload_len",
    "tail_length_bytes",
    "frequency_hz",
    "extracted_frame_hex",
    "extracted_seq",
    "extracted_payload_hex",
    "extracted_integrity_ok",
    "extracted_frame_len_bytes",
    "frame_len_matches",
    "known_bit_errors",
    "known_bit_error_rate",
    "lowpass_hz",
    "samples_per_bit",
    "cfo_offset_hz",
    "bit_start_sample",
    "threshold",
    "hypothesis_count",
    "hypotheses_json",
    "notes",
]


AUDIT_FIELDS = [
    "run_id",
    "failure_stage_at_input",
    "observation_index",
    "tx_attempt_id",
    "seq",
    "channel",
    "blind_extracted_seq",
    "blind_extracted_integrity_ok",
    "blind_extracted_frame_len_bytes",
    "blind_frame_len_matches",
    "rtt_guided_exact",
    "rtt_guided_data_exact",
    "tx_payload_len_bytes",
    "decoded_payload_len_bytes",
    "known_bit_errors",
    "notes",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def int_or_none(value: Any) -> int | None:
    parsed = matcher.int_or_empty(value)
    return parsed if isinstance(parsed, int) else None


def float_or_none(value: Any) -> float | None:
    parsed = matcher.float_or_none(value)
    return parsed if parsed is not None else None


def parse_float_grid(values: list[str]) -> tuple[float, ...]:
    result: list[float] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                result.append(float(item))
    if not result:
        raise ValueError("at least one numeric hypothesis value is required")
    return tuple(result)


def load_tx_by_seq(run_root: Path) -> dict[str, dict[str, Any]]:
    ground_truth = matcher.read_csv(run_root / "ground_truth" / "rtt_ground_truth.csv")
    ll_rows = matcher.read_csv(run_root / "ground_truth" / "rtt_ll_tx.csv")
    tx_rows = matcher.joined_rtt_tx_rows(ground_truth, ll_rows)
    result: dict[str, dict[str, Any]] = {}
    for row in tx_rows:
        seq = matcher.normalize_seq(row.get("seq"))
        if seq and seq not in result:
            result[seq] = row
    return result


def decode_observation(
    *,
    run_root: Path,
    sdr_row: dict[str, str],
    stage: str,
    tail_length_bytes: int,
    lowpass_hz_values: tuple[float, ...],
    samples_per_bit_values: tuple[float, ...],
    cfo_offset_hz_values: tuple[float, ...],
    search_us: float,
    max_hypotheses: int,
) -> dict[str, Any]:
    metadata = json.loads((run_root / "iq" / "metadata.json").read_text(encoding="utf-8"))
    sample_rate_hz = float(metadata.get("actual_sample_rate_sps") or metadata["sample_rate_sps"])
    center_frequency_hz = float(
        metadata.get("actual_center_frequency_hz") or metadata["center_frequency_hz"]
    )
    observation_index = int_or_none(sdr_row.get("_observation_index"))
    packet_start = int_or_none(sdr_row.get("wideband_sample_index"))
    channel = matcher.normalize_seq(sdr_row.get("channel"))
    channel_i = int_or_none(channel)
    payload_len = int_or_none(sdr_row.get("payload_len"))
    frequency_hz = matcher.ble_data_channel_frequency_hz(str(channel)) if channel else None
    notes: list[str] = []
    extraction: dict[str, Any] = {"hypotheses": []}
    if packet_start is None:
        notes.append("missing_wideband_sample_index")
    if channel_i is None:
        notes.append("missing_channel")
    if payload_len is None:
        notes.append("missing_payload_len")
    if frequency_hz is None:
        notes.append("missing_frequency")
    if not matcher.normalize_hex(sdr_row.get("dewhitened_pdu_hex")):
        notes.append("missing_dewhitened_pdu")
    if not matcher.normalize_hex(sdr_row.get("captured_crc_hex")):
        notes.append("missing_captured_crc")

    if not notes:
        extraction = postprocess.extract_post_crc_hypotheses_from_iq(
            run_root / "iq" / "capture.sc16",
            sample_rate_hz=sample_rate_hz,
            center_frequency_hz=center_frequency_hz,
            packet_start_sample=packet_start or 0,
            packet_frequency_hz=float(frequency_hz),
            access_address_text=sdr_row.get("access_address", ""),
            dewhitened_pdu_hex=sdr_row.get("dewhitened_pdu_hex", ""),
            captured_crc_hex=sdr_row.get("captured_crc_hex", ""),
            channel=channel_i or 0,
            tail_len_bytes=tail_length_bytes,
            lowpass_hz_values=lowpass_hz_values,
            samples_per_bit_values=samples_per_bit_values,
            cfo_offset_hz_values=cfo_offset_hz_values,
            expected_phantom_us=postprocess.ble_1m_airtime_us(payload_len or 0, tail_length_bytes),
            search_us=search_us,
            max_hypotheses=max_hypotheses,
        )

    hypotheses = extraction.get("hypotheses", [])
    return {
        "run_id": run_root.name,
        "failure_stage_at_input": stage,
        "observation_index": observation_index if observation_index is not None else "",
        "candidate_sample": sdr_row.get("wideband_sample_index", ""),
        "candidate_timestamp_us": sdr_row.get("timestamp_us", ""),
        "access_address": sdr_row.get("access_address", ""),
        "channel": channel,
        "payload_len": payload_len if payload_len is not None else "",
        "tail_length_bytes": tail_length_bytes,
        "frequency_hz": frequency_hz if frequency_hz is not None else "",
        "extracted_frame_hex": extraction.get("extracted_frame_hex", ""),
        "extracted_seq": extraction.get("extracted_seq", ""),
        "extracted_payload_hex": extraction.get("extracted_payload_hex", ""),
        "extracted_integrity_ok": extraction.get("extracted_integrity_ok", ""),
        "extracted_frame_len_bytes": extraction.get("extracted_frame_len_bytes", ""),
        "frame_len_matches": extraction.get("frame_len_matches", ""),
        "known_bit_errors": extraction.get("known_bit_errors", ""),
        "known_bit_error_rate": extraction.get("known_bit_error_rate", ""),
        "lowpass_hz": extraction.get("lowpass_hz", ""),
        "samples_per_bit": extraction.get("samples_per_bit", ""),
        "cfo_offset_hz": extraction.get("cfo_offset_hz", ""),
        "bit_start_sample": extraction.get("bit_start_sample", ""),
        "threshold": extraction.get("threshold", ""),
        "hypothesis_count": extraction.get("hypothesis_count", 0),
        "hypotheses_json": json.dumps(hypotheses, separators=(",", ":"), sort_keys=True),
        "notes": ";".join([*notes, str(extraction.get("notes", ""))]).strip(";"),
    }


def make_audit_row(
    blind: dict[str, Any],
    funnel: dict[str, str],
    tx_by_seq: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    seq = matcher.normalize_seq(funnel.get("seq"))
    tx = tx_by_seq.get(seq, {})
    expected_payload = matcher.normalize_hex(tx.get("payload", ""))
    decoded_payload = matcher.normalize_hex(blind.get("extracted_payload_hex", ""))
    decoded_seq = matcher.normalize_seq(blind.get("extracted_seq"))
    integrity_ok = str(blind.get("extracted_integrity_ok", "")) == "1"
    exact = bool(integrity_ok and decoded_seq == seq and decoded_payload == expected_payload)
    marker = matcher.normalize_hex(tx.get("marker", ""))
    decoded_data = matcher.payload_data_hex(decoded_payload, marker) if decoded_payload else ""
    expected_data = matcher.normalize_hex(tx.get("data_payload", ""))
    data_exact = bool(exact and decoded_data == expected_data)
    notes: list[str] = []
    if not tx:
        notes.append("no_rtt_tx_for_seq")
    if decoded_seq and decoded_seq != seq:
        notes.append("decoded_seq_differs_from_rtt")
    return {
        "run_id": blind.get("run_id", ""),
        "failure_stage_at_input": funnel.get("failure_stage", ""),
        "observation_index": blind.get("observation_index", ""),
        "tx_attempt_id": funnel.get("tx_attempt_id", ""),
        "seq": seq,
        "channel": funnel.get("channel", ""),
        "blind_extracted_seq": blind.get("extracted_seq", ""),
        "blind_extracted_integrity_ok": blind.get("extracted_integrity_ok", ""),
        "blind_extracted_frame_len_bytes": blind.get("extracted_frame_len_bytes", ""),
        "blind_frame_len_matches": blind.get("frame_len_matches", ""),
        "rtt_guided_exact": int(exact),
        "rtt_guided_data_exact": int(data_exact),
        "tx_payload_len_bytes": len(expected_payload) // 2 if expected_payload else "",
        "decoded_payload_len_bytes": len(decoded_payload) // 2 if decoded_payload else "",
        "known_bit_errors": blind.get("known_bit_errors", ""),
        "notes": ";".join(notes),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    funnel_rows = read_csv(args.funnel.resolve())
    sdr_path = (args.sdr_csv or (run_root / "sdr" / "ble_packets.csv")).resolve()
    sdr_rows = read_csv(sdr_path)
    tx_by_seq = load_tx_by_seq(run_root)
    stages = set(args.stages)
    targets: list[tuple[dict[str, str], dict[str, str]]] = []
    for funnel in funnel_rows:
        stage = funnel.get("failure_stage", "")
        if stage not in stages:
            continue
        observation_index = int_or_none(funnel.get("observation_index"))
        if observation_index is None or not (0 <= observation_index < len(sdr_rows)):
            continue
        row = dict(sdr_rows[observation_index])
        row["_observation_index"] = str(observation_index)
        targets.append((row, funnel))

    lowpass_values = parse_float_grid(args.lowpass_hz)
    samples_per_bit_values = parse_float_grid(args.samples_per_bit)
    cfo_values = parse_float_grid(args.cfo_offset_hz)
    if args.blind_input:
        blind_rows = read_csv(args.blind_input.resolve())
        if len(blind_rows) != len(targets):
            raise ValueError(
                f"blind input row count {len(blind_rows)} does not match target count {len(targets)}"
            )
    else:
        blind_rows = []
        for sdr_row, funnel in targets:
            blind_rows.append(
                decode_observation(
                    run_root=run_root,
                    sdr_row=sdr_row,
                    stage=funnel.get("failure_stage", ""),
                    tail_length_bytes=args.tail_length_bytes,
                    lowpass_hz_values=lowpass_values,
                    samples_per_bit_values=samples_per_bit_values,
                    cfo_offset_hz_values=cfo_values,
                    search_us=args.search_us,
                    max_hypotheses=args.max_hypotheses,
                )
            )
    audit_rows: list[dict[str, Any]] = [
        make_audit_row(blind, funnel, tx_by_seq)
        for blind, (_sdr_row, funnel) in zip(blind_rows, targets)
    ]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "blind_tail_candidates.csv", blind_rows, BLIND_FIELDS)
    write_csv(output_dir / "rtt_guided_audit.csv", audit_rows, AUDIT_FIELDS)
    summary = {
        "schema_version": 1,
        "run_id": run_root.name,
        "mode": "blind_candidate_decode_plus_rtt_guided_audit",
        "input_funnel": str(args.funnel.resolve()),
        "input_sdr_csv": str(sdr_path),
        "input_candidate_count": len(targets),
        "stages": sorted(stages),
        "tail_length_bytes": args.tail_length_bytes,
        "lowpass_hz": list(lowpass_values),
        "samples_per_bit": list(samples_per_bit_values),
        "cfo_offset_hz": list(cfo_values),
        "search_us": args.search_us,
        "max_hypotheses": args.max_hypotheses,
        "blind_integrity_valid_frames": sum(
            str(row.get("extracted_integrity_ok", "")) == "1" for row in blind_rows
        ),
        "blind_exact_tail_length_frames": sum(
            str(row.get("frame_len_matches", "")) == "1" for row in blind_rows
        ),
        "rtt_guided_exact": sum(int(row["rtt_guided_exact"]) for row in audit_rows),
        "rtt_guided_data_exact": sum(int(row["rtt_guided_data_exact"]) for row in audit_rows),
        "notes": [
            "This investigation decoder does not use RTT seq or payload to select a hypothesis.",
            "rtt_guided_audit.csv is scoring-only and must not be reported as blind recovery.",
            "The existing phantom_postprocess_scorer default extractor is unchanged.",
            "BLE_encrypt_check is an external read-only dependency and is not modified.",
        ],
    }
    write_json(output_dir / "local_tail_recovery_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--funnel", type=Path, required=True)
    parser.add_argument("--sdr-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blind-input", type=Path, default=None)
    parser.add_argument("--stages", nargs="+", default=["F5_FRAME_INVALID", "F6_VALID_NOT_ASSIGNED"])
    parser.add_argument("--tail-length-bytes", type=int, default=237)
    parser.add_argument("--lowpass-hz", nargs="+", default=["700000,900000,1100000"])
    parser.add_argument("--samples-per-bit", nargs="+", default=["99.7,100.0,100.3"])
    parser.add_argument("--cfo-offset-hz", nargs="+", default=["-100000,0,100000"])
    parser.add_argument("--search-us", type=float, default=2.0)
    parser.add_argument("--max-hypotheses", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
