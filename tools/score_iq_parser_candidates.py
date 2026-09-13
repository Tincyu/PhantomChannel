#!/usr/bin/env python3
"""Score PhantomChannel parser candidates without RTT ground truth.

The input is only IQ metadata and the blind one-stage parser CSV.  A candidate
is a parser row containing a complete ``PC`` frame after the standard BLE CRC,
with a valid Phantom marker.  The default rate counts the complete post-CRC
PC frame: its fixed 6-byte header/check area plus the declared covert payload.
The XOR/check byte is reported, but it is not a gate for the default rate.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FIELDS = [
    "candidate_index",
    "source_row_index",
    "sample_index",
    "timestamp_us",
    "channel",
    "access_address_hex",
    "dominant_access_address_hex",
    "aa_hamming_distance",
    "frame_seq",
    "frame_hex",
    "pc_frame_bytes",
    "covert_length_bytes",
    "payload_hex",
    "marker_hex",
    "data_hex",
    "payload_data_bytes",
    "payload_data_bytes_excluding_marker",
    "integrity_ok",
    "crc_capture_status",
    "standard_crc_ok",
    "rssi",
    "cfo_hz",
    "confidence_score",
    "raw_duplicate_count",
    "dedup_reason",
    "post_crc_hex",
    "pattern_exact",
    "pattern_correct_bytes",
    "pattern_compared_bytes",
    "pattern_byte_accuracy",
    "pattern_bit_errors",
    "pattern_bits_compared",
    "pattern_ber",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def clean_hex(value: Any) -> str:
    text = str(value or "").strip().replace("0x", "").replace("0X", "")
    text = "".join(text.split()).replace(":", "").replace("-", "")
    if len(text) % 2 or any(char not in "0123456789abcdefABCDEF" for char in text):
        return ""
    return text.lower()


def parse_int(value: Any, default: int | None = None) -> int | None:
    text = str(value or "").strip()
    if not text:
        return default
    try:
        return int(text, 0)
    except ValueError:
        try:
            return int(float(text))
        except ValueError:
            return default


def parse_float(value: Any, default: float | None = None) -> float | None:
    text = str(value or "").strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def increment_pattern_payload(marker_byte: int, length: int) -> bytes:
    """Build the fixed Phantom covert pattern: marker + 0x01..0xff increment.

    This mirrors ``fill_covert_payload()`` in
    ``phantomchannel_hrs_peripheral/src/main.c``: byte 0 is the marker and
    bytes 1.. are ((i-1) % 255) + 1, so 0x01..0xff wraps at 255.
    """

    if length < 1:
        return b""
    return bytes([marker_byte]) + bytes(((i - 1) % 255) + 1 for i in range(1, length))


def annotate_pattern_candidate(candidate: dict[str, Any], marker_byte: int = 0xA5) -> dict[str, Any]:
    """Compare a candidate payload against the predicted increment pattern."""

    payload = bytes.fromhex(candidate["payload_hex"])
    expected = increment_pattern_payload(marker_byte, len(payload))
    exact = payload == expected
    compared_bytes = len(payload)
    correct_bytes = sum(left == right for left, right in zip(payload, expected))
    bit_errors = sum((left ^ right).bit_count() for left, right in zip(payload, expected))
    compared_bits = compared_bytes * 8
    candidate = dict(candidate)
    candidate.update({
        "pattern_exact": int(exact),
        "pattern_correct_bytes": correct_bytes,
        "pattern_compared_bytes": compared_bytes,
        "pattern_byte_accuracy": correct_bytes / compared_bytes if compared_bytes else 0.0,
        "pattern_bit_errors": bit_errors,
        "pattern_bits_compared": compared_bits,
        "pattern_ber": bit_errors / compared_bits if compared_bits else 0.0,
    })
    return candidate


def pattern_summary(
    candidates: list[dict[str, Any]],
    *,
    enabled: bool,
    outlier_ber_threshold: float = 0.10,
) -> dict[str, Any]:
    """Aggregate per-candidate pattern metrics into window-level metrics.

    Global BER is dominated by a small tail of collision-corrupted packets
    (per-packet BER above ``outlier_ber_threshold``).  Besides the raw global
    BER, the summary therefore reports the BER computed over the remaining
    (non-collision) packets plus the outlier packet count/error share, so the
    collision contribution is reported explicitly instead of silently
    inflating the BER.
    """

    disabled_fields = {
        "pattern_mode": "disabled",
        "pattern_spec": "unavailable",
        "pattern_candidates_compared": 0,
        "pattern_exact_packets": "unavailable",
        "psr_exact": "unavailable",
        "pattern_byte_recovery": "unavailable",
        "pattern_ber": "unavailable",
        "pattern_outlier_ber_threshold": outlier_ber_threshold,
        "pattern_outlier_packets": "unavailable",
        "pattern_outlier_fraction": "unavailable",
        "pattern_outlier_bit_errors": "unavailable",
        "pattern_outlier_error_share": "unavailable",
        "pattern_ber_excluding_outliers": "unavailable",
    }
    if not enabled:
        return disabled_fields

    per_candidate = []
    for item in candidates:
        errors = int(item.get("pattern_bit_errors", 0))
        bits = int(item.get("pattern_bits_compared", 0))
        per_candidate.append({
            "errors": errors,
            "bits": bits,
            "ber": errors / bits if bits else 0.0,
            "exact": int(item.get("pattern_exact", 0)),
            "compared_bytes": int(item.get("pattern_compared_bytes", 0)),
            "correct_bytes": int(item.get("pattern_correct_bytes", 0)),
        })
    exact = sum(item["exact"] for item in per_candidate)
    compared_bytes = sum(item["compared_bytes"] for item in per_candidate)
    correct_bytes = sum(item["correct_bytes"] for item in per_candidate)
    compared_bits = sum(item["bits"] for item in per_candidate)
    bit_errors = sum(item["errors"] for item in per_candidate)
    outlier_packets = sum(1 for item in per_candidate if item["ber"] > outlier_ber_threshold)
    outlier_bit_errors = sum(
        item["errors"] for item in per_candidate if item["ber"] > outlier_ber_threshold
    )
    clean_bits = sum(
        item["bits"] for item in per_candidate if not item["ber"] > outlier_ber_threshold
    )
    clean_errors = bit_errors - outlier_bit_errors
    return {
        "pattern_mode": "a5_increment",
        "pattern_spec": {"marker_hex": "a5", "increment_start": 1, "increment_wrap": 255},
        "pattern_candidates_compared": len(candidates),
        "pattern_exact_packets": exact,
        "psr_exact": exact / len(candidates) if candidates else 0.0,
        "pattern_byte_recovery": correct_bytes / compared_bytes if compared_bytes else 0.0,
        "pattern_ber": bit_errors / compared_bits if compared_bits else 0.0,
        "pattern_outlier_ber_threshold": outlier_ber_threshold,
        "pattern_outlier_packets": outlier_packets,
        "pattern_outlier_fraction": outlier_packets / len(candidates) if candidates else 0.0,
        "pattern_outlier_bit_errors": outlier_bit_errors,
        "pattern_outlier_error_share": (
            outlier_bit_errors / bit_errors if bit_errors else 0.0
        ),
        "pattern_ber_excluding_outliers": (
            clean_errors / clean_bits if clean_bits else 0.0
        ),
    }


def normalize_aa(value: Any) -> str:
    text = clean_hex(value)
    return text[-8:].zfill(8) if text else ""


def hamming_hex(left: str, right: str) -> int:
    if not left or not right:
        return 999
    return (int(left, 16) ^ int(right, 16)).bit_count()


def standard_crc_ok(row: dict[str, str]) -> bool:
    status = str(row.get("crc_capture_status", "")).strip().lower()
    if status and status not in {"ok", "true", "1", "yes"}:
        return False
    crc_ok = str(row.get("crc_ok", "")).strip().lower()
    if crc_ok and crc_ok not in {"ok", "true", "1", "yes"}:
        return False
    return True


def xor_check(frame: bytes) -> bool:
    if len(frame) < 1:
        return False
    value = 0
    for byte in frame[:-1]:
        value ^= byte
    return value == frame[-1]


def extract_pc_frame(
    row: dict[str, str],
    *,
    magic_hex: str = "5043",
    marker_hex: str = "a5",
    inline_frame_source: bool = False,
) -> dict[str, Any] | None:
    """Extract the first complete marker-bearing PC frame from a parser row."""

    marker_hex = clean_hex(marker_hex)
    sources = [
        ("post_crc_hex", clean_hex(row.get("post_crc_hex"))),
        ("covert_frame_hex", clean_hex(row.get("covert_frame_hex"))),
        ("rx_frame_hex", clean_hex(row.get("rx_frame_hex"))),
    ]
    if inline_frame_source:
        # Opt-in fallback for firmware variants that embed the PC frame inside
        # the notification payload (de-whitened PDU) instead of appending it
        # after the BLE CRC via the controller post-CRC rewrite hook.  Only
        # useful when the exported PDU is long enough to contain the full
        # frame; the X310 parser exports at most 253 PDU bytes, so 236 B
        # covert frames do not fit and must not be scored this way.
        sources.append(("dewhitened_pdu_hex", clean_hex(row.get("dewhitened_pdu_hex"))))
    # Some future parser exports may already split the Phantom payload.
    payload_sources = [
        ("covert_hex", clean_hex(row.get("covert_hex"))),
        ("rx_covert_hex", clean_hex(row.get("rx_covert_hex"))),
        ("covert_payload_hex", clean_hex(row.get("covert_payload_hex"))),
    ]

    for source_name, text in sources:
        if not text:
            continue
        for offset in range(0, max(0, len(text) - 9), 2):
            if text[offset : offset + 4] != magic_hex:
                continue
            declared = int(text[offset + 8 : offset + 10], 16)
            frame_bytes = 6 + declared
            end = offset + frame_bytes * 2
            if declared < 1 or end > len(text):
                continue
            frame_hex = text[offset:end]
            payload_start = offset + 10
            payload_end = payload_start + declared * 2
            payload_hex = text[payload_start:payload_end]
            if not payload_hex.startswith(marker_hex):
                continue
            frame = bytes.fromhex(frame_hex)
            data_hex = payload_hex[len(marker_hex) :]
            return {
                "frame_source": source_name,
                "frame_hex": frame_hex,
                "pc_frame_bytes": frame_bytes,
                "payload_hex": payload_hex,
                "marker_hex": marker_hex,
                "data_hex": data_hex,
                "covert_length_bytes": declared,
                "integrity_ok": xor_check(frame),
            }

    for source_name, payload_hex in payload_sources:
        if not payload_hex or not payload_hex.startswith(marker_hex):
            continue
        return {
            "frame_source": source_name,
            "frame_hex": "",
            "pc_frame_bytes": 6 + len(payload_hex) // 2,
            "payload_hex": payload_hex,
            "marker_hex": marker_hex,
            "data_hex": payload_hex[len(marker_hex) :],
            "covert_length_bytes": len(payload_hex) // 2,
            "integrity_ok": str(row.get("covert_integrity_ok", "")).strip().lower()
            in {"1", "true", "ok", "yes"},
        }
    return None


def materialize_candidate(
    row: dict[str, str],
    row_index: int,
    frame: dict[str, Any],
    dominant_aa: str,
    aa_hamming_tolerance: int,
) -> dict[str, Any] | None:
    sample = parse_int(row.get("wideband_sample_index"))
    if sample is None:
        sample = parse_int(row.get("sample_index"))
    if sample is None or sample < 0:
        return None
    aa = normalize_aa(row.get("access_address"))
    if dominant_aa and aa and hamming_hex(aa, dominant_aa) > aa_hamming_tolerance:
        return None
    channel = str(row.get("channel", "")).strip()
    payload_hex = frame["payload_hex"]
    data_hex = frame["data_hex"]
    return {
        "source_row_index": row_index,
        "sample_index": sample,
        "timestamp_us": row.get("timestamp_us", ""),
        "channel": channel,
        "access_address_hex": aa,
        "dominant_access_address_hex": dominant_aa,
        "aa_hamming_distance": hamming_hex(aa, dominant_aa) if aa and dominant_aa else "",
        "frame_seq": int.from_bytes(bytes.fromhex(frame["frame_hex"])[2:4], "little")
        if frame["frame_hex"]
        else "",
        "frame_hex": frame["frame_hex"],
        "pc_frame_bytes": frame["pc_frame_bytes"],
        "covert_length_bytes": frame["covert_length_bytes"],
        "payload_hex": payload_hex,
        "marker_hex": frame["marker_hex"],
        "data_hex": data_hex,
        # Keep payload-only byte counts for diagnostics. The default rate uses
        # pc_frame_bytes, which also includes the fixed 6-byte PC header/check.
        "payload_data_bytes": len(payload_hex) // 2,
        "payload_data_bytes_excluding_marker": len(data_hex) // 2,
        "integrity_ok": int(bool(frame["integrity_ok"])),
        "crc_capture_status": row.get("crc_capture_status", ""),
        "standard_crc_ok": int(standard_crc_ok(row)),
        "rssi": row.get("rssi", ""),
        "cfo_hz": row.get("cfo_hz", ""),
        "confidence_score": row.get("confidence_score", ""),
        "raw_duplicate_count": 1,
        "dedup_reason": "selected_after_sample_channel_cluster",
        "post_crc_hex": clean_hex(row.get("post_crc_hex")),
    }


def choose_representative(cluster: list[dict[str, Any]]) -> dict[str, Any]:
    # Integrity is diagnostic only, but it is the best way to select one row
    # when overlap produced different hypotheses for the same physical burst.
    def score(candidate: dict[str, Any]) -> tuple[int, int, float, int]:
        confidence = parse_float(candidate.get("confidence_score"), -1.0) or -1.0
        distance = parse_int(candidate.get("aa_hamming_distance"), 999) or 999
        return (
            int(candidate.get("integrity_ok", 0)),
            -distance,
            confidence,
            -int(candidate["source_row_index"]),
        )

    selected = max(cluster, key=score)
    selected = dict(selected)
    selected["raw_duplicate_count"] = len(cluster)
    selected["dedup_reason"] = (
        "single_parser_row"
        if len(cluster) == 1
        else "sample_channel_cluster_integrity_confidence_preferred"
    )
    return selected


def deduplicate_candidates(
    candidates: list[dict[str, Any]],
    *,
    dedup_gap_samples: int,
) -> list[dict[str, Any]]:
    ordered = sorted(candidates, key=lambda item: (int(item["sample_index"]), int(item["source_row_index"])))
    clusters: list[list[dict[str, Any]]] = []
    for candidate in ordered:
        if not clusters:
            clusters.append([candidate])
            continue
        previous = clusters[-1][-1]
        same_channel = candidate.get("channel", "") == previous.get("channel", "")
        close_sample = int(candidate["sample_index"]) - int(previous["sample_index"]) <= dedup_gap_samples
        if same_channel and close_sample:
            clusters[-1].append(candidate)
        else:
            clusters.append([candidate])
    return [choose_representative(cluster) for cluster in clusters]


def seq_span_stats(seqs: list[int], *, run_gap_tolerance: int = 4) -> dict[str, Any]:
    """Theoretical window length from the recovered 16-bit frame_seq series.

    The PC header carries a 16-bit per-notification counter (frame_seq,
    0x0000..0xffff, reset on connect).  Without RTT ground truth, the first and
    last recovered seq values bound the transmitted window, but a few packets
    can have corrupted seq bytes (demodulation artifacts at the post-CRC tail
    start), so the theoretical span is taken from the dominant consecutive run
    instead of the raw min/max.  The 16-bit counter can also wrap at 0xffff
    (a connection longer than ~22 min at 20 ms), so the series is first
    unwrapped at its largest cyclic gap.  The result is an estimate: if the
    window's true first/last transmission was not recovered, the span is a
    lower bound on the number of transmitted notifications in the window.
    """

    unique = sorted(set(seqs))
    if not unique:
        return {
            "seq_first": None,
            "seq_last": None,
            "seq_span_raw_packets": None,
            "seq_main_first": None,
            "seq_main_last": None,
            "seq_span_theoretical_packets": None,
            "recovered_within_span_fraction": None,
            "seq_run_gap_tolerance": run_gap_tolerance,
        }
    first_raw, last_raw = unique[0], unique[-1]
    raw_span = (last_raw - first_raw + 1) if last_raw >= first_raw else (
        65536 - first_raw + last_raw + 1
    )

    # Unwrap the 16-bit series at its largest cyclic gap (handles 0xffff wrap).
    if len(unique) > 1:
        cyclic_gaps = [
            (unique[(index + 1) % len(unique)] - unique[index])
            % 65536
            for index in range(len(unique))
        ]
        break_index = cyclic_gaps.index(max(cyclic_gaps))
        unwrapped: list[int] = []
        for index in range(1, len(unique) + 1):
            value = unique[(break_index + index) % len(unique)]
            if unwrapped and value < unwrapped[-1]:
                value += 65536
            unwrapped.append(value)
    else:
        unwrapped = list(unique)

    positive_deltas = sorted(
        right - left for left, right in zip(unwrapped, unwrapped[1:]) if right > left
    )
    median_delta = positive_deltas[len(positive_deltas) // 2] if positive_deltas else 1
    tolerance = max(run_gap_tolerance, 8 * median_delta)
    runs: list[list[int]] = []
    current = [unwrapped[0]]
    for left, right in zip(unwrapped, unwrapped[1:]):
        if right - left <= tolerance:
            current.append(right)
        else:
            runs.append(current)
            current = [right]
    runs.append(current)
    main = max(runs, key=len)
    main_span = main[-1] - main[0] + 1
    return {
        "seq_first": first_raw,
        "seq_last": last_raw,
        "seq_span_raw_packets": raw_span,
        "seq_main_first": main[0] % 65536,
        "seq_main_last": main[-1] % 65536,
        "seq_span_theoretical_packets": main_span,
        "recovered_within_span_fraction": len(unique) / main_span if main_span else None,
        "seq_run_gap_tolerance": tolerance,
    }


def sequence_stats(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    seqs = [int(item["frame_seq"]) for item in candidates if str(item.get("frame_seq", "")) != ""]
    unique = list(dict.fromkeys(seqs))
    transitions = [b - a for a, b in zip(unique, unique[1:])]
    gaps = sum(max(0, delta - 1) for delta in transitions if delta > 0)
    stats: dict[str, Any] = {
        "seq_unique_count": len(set(seqs)),
        "seq_duplicate_count": len(seqs) - len(set(seqs)),
        "seq_gap_count": gaps,
        "seq_nonconsecutive_transitions": sum(delta != 1 for delta in transitions),
    }
    stats.update(seq_span_stats(seqs))
    return stats


def unique_sequence_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one representative candidate per Phantom frame sequence number."""

    grouped: dict[int, list[dict[str, Any]]] = {}
    for candidate in candidates:
        sequence = candidate.get("frame_seq")
        if sequence is None or str(sequence).strip() == "":
            continue
        sequence = int(sequence)
        grouped.setdefault(sequence, []).append(candidate)

    selected: list[dict[str, Any]] = []
    for cluster in grouped.values():
        representative = choose_representative(cluster)
        representative["dedup_reason"] = (
            "unique_frame_seq"
            if len(cluster) == 1
            else "unique_frame_seq_integrity_confidence_preferred"
        )
        selected.append(representative)
    return sorted(selected, key=lambda item: int(item["sample_index"]))


def score_candidates(
    metadata: dict[str, Any],
    parser_rows: list[dict[str, str]],
    *,
    aa_hamming_tolerance: int = 2,
    dedup_gap_samples: int = 200,
    magic_hex: str = "5043",
    marker_hex: str = "a5",
    pattern_enabled: bool = False,
    inline_frame_source: bool = False,
    pattern_outlier_ber_threshold: float = 0.10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_frames: list[tuple[int, dict[str, str], dict[str, Any]]] = []
    discarded_crc = 0
    discarded_no_frame = 0
    for index, row in enumerate(parser_rows):
        if not standard_crc_ok(row):
            discarded_crc += 1
            continue
        frame = extract_pc_frame(
            row,
            magic_hex=clean_hex(magic_hex),
            marker_hex=clean_hex(marker_hex),
            inline_frame_source=inline_frame_source,
        )
        if frame is None:
            discarded_no_frame += 1
            continue
        raw_frames.append((index, row, frame))

    aa_counts = Counter(normalize_aa(row.get("access_address")) for _, row, _ in raw_frames)
    aa_counts.pop("", None)
    dominant_aa = aa_counts.most_common(1)[0][0] if aa_counts else ""
    candidates: list[dict[str, Any]] = []
    for index, row, frame in raw_frames:
        candidate = materialize_candidate(row, index, frame, dominant_aa, aa_hamming_tolerance)
        if candidate is not None:
            candidates.append(candidate)
    selected = deduplicate_candidates(candidates, dedup_gap_samples=dedup_gap_samples)
    selected.sort(key=lambda item: int(item["sample_index"]))
    for index, item in enumerate(selected):
        item["candidate_index"] = index
    if pattern_enabled:
        selected = [annotate_pattern_candidate(item) for item in selected]

    samples = parse_float(metadata.get("samples"), 0.0) or 0.0
    sample_rate = parse_float(metadata.get("actual_sample_rate_sps"), 0.0) or 0.0
    duration_s = samples / sample_rate if sample_rate > 0 else 0.0
    sequence_selected = unique_sequence_candidates(selected)
    payload_bytes_all = sum(int(item["payload_data_bytes"]) for item in selected)
    data_bytes_excluding_marker_all = sum(
        int(item["payload_data_bytes_excluding_marker"]) for item in selected
    )
    pc_frame_bytes_all = sum(int(item["pc_frame_bytes"]) for item in selected)
    payload_bytes = sum(int(item["payload_data_bytes"]) for item in sequence_selected)
    data_bytes_excluding_marker = sum(
        int(item["payload_data_bytes_excluding_marker"]) for item in sequence_selected
    )
    pc_frame_bytes = sum(int(item["pc_frame_bytes"]) for item in sequence_selected)
    marker_inclusive_bytes = sum(int(item["covert_length_bytes"]) for item in sequence_selected)
    rate_bps = pc_frame_bytes * 8.0 / duration_s if duration_s > 0 else 0.0
    payload_rate_bps = payload_bytes * 8.0 / duration_s if duration_s > 0 else 0.0
    integrity_valid = sum(int(item["integrity_ok"]) for item in sequence_selected)
    summary: dict[str, Any] = {
        "schema_version": 2,
        "scorer": "score_iq_parser_candidates",
        "input_parser_rows": len(parser_rows),
        "parser_candidate_rows_raw": len(raw_frames),
        "parser_candidates_after_dominant_aa_filter": len(candidates),
        "parser_candidates_deduplicated": len(selected),
        "parser_rows_discarded_standard_crc": discarded_crc,
        "parser_rows_without_complete_pc_frame": discarded_no_frame,
        "dominant_access_address_hex": dominant_aa,
        "dominant_access_address_raw_count": aa_counts.get(dominant_aa, 0),
        "aa_hamming_tolerance": aa_hamming_tolerance,
        "dedup_gap_samples": dedup_gap_samples,
        "rate_denominator": "seq_unique_count",
        "rate_candidates": len(sequence_selected),
        "iq_samples": int(samples),
        "iq_capture_duration_s": duration_s,
        "actual_sample_rate_sps": sample_rate,
        "phantom_pc_frame_overhead_bytes": 6,
        "parser_candidate_pc_frame_bytes": pc_frame_bytes,
        "parser_candidate_pc_frame_bytes_all_deduplicated_candidates": pc_frame_bytes_all,
        "parser_candidate_covert_data_bytes": payload_bytes,
        "parser_candidate_covert_data_bytes_all_deduplicated_candidates": payload_bytes_all,
        "parser_candidate_covert_data_bytes_excluding_marker": data_bytes_excluding_marker,
        "parser_candidate_covert_data_bytes_excluding_marker_all_deduplicated_candidates": data_bytes_excluding_marker_all,
        "parser_candidate_marker_inclusive_bytes": marker_inclusive_bytes,
        "iq_window_parser_candidate_payload_data_bps": payload_rate_bps,
        "iq_window_parser_candidate_payload_data_kbps": payload_rate_bps / 1000.0,
        "iq_window_parser_candidate_data_bps": rate_bps,
        "iq_window_parser_candidate_data_kbps": rate_bps / 1000.0,
        "marker_valid_candidates": len(selected),
        "integrity_valid_candidates": integrity_valid,
        "integrity_invalid_candidates": len(selected) - integrity_valid,
        "channel_distribution": dict(Counter(item["channel"] for item in selected)),
        "unique_seq_channel_distribution": dict(
            Counter(item["channel"] for item in sequence_selected)
        ),
        "candidate_length_distribution": dict(
            Counter(int(item["payload_data_bytes"]) for item in selected)
        ),
        "pc_frame_length_distribution": dict(
            Counter(int(item["pc_frame_bytes"]) for item in selected)
        ),
        "rate_pc_frame_length_distribution": dict(
            Counter(int(item["pc_frame_bytes"]) for item in sequence_selected)
        ),
        "raw_duplicate_rows_removed": len(candidates) - len(selected),
        "rtt_metrics": "unavailable",
        "rtt_exact_packets": "unavailable",
        "bit_error_rate_against_rtt": "unavailable",
        "covert_recovery_rate": "unavailable",
        "time_aligned_capture_recovered_data_bps": "unavailable",
        "notes": [
            "No RTT, BlueZ, J-Link, or fixed payload pattern was used.",
            "The Phantom XOR/check byte is diagnostic only and does not gate the default rate.",
            "The default rate counts one complete post-CRC PC frame per unique frame_seq.",
            "Each PC frame contributes 6 fixed header/check bytes plus its declared covert payload.",
            "The fixed A5 marker is already part of the declared covert payload and is counted once.",
            "Repeated frame_seq candidates are retained in physical-candidate diagnostics but excluded from the default rate.",
            "Payload-only and marker-excluded byte rates are retained as diagnostic fields.",
        ],
    }
    summary.update(sequence_stats(selected))
    summary.update(
        pattern_summary(
            sequence_selected,
            enabled=pattern_enabled,
            outlier_ber_threshold=pattern_outlier_ber_threshold,
        )
    )
    if pattern_enabled:
        summary["notes"].append(
            "Pattern comparison assumes the fixed A5 marker plus 0x01..0xff increment payload "
            "from fill_covert_payload(); PSR_exact/BER/byte recovery use unique-seq candidates. "
            "pattern_ber_excluding_outliers drops packets whose per-packet BER exceeds "
            "pattern_outlier_ber_threshold (collision outliers)."
        )
    return selected, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--parser-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--aa-hamming-tolerance", type=int, default=2)
    parser.add_argument("--dedup-gap-samples", type=int, default=200)
    parser.add_argument("--magic-hex", default="5043")
    parser.add_argument("--marker-hex", default="a5")
    parser.add_argument(
        "--pattern",
        action="store_true",
        help="compare payloads against the fixed A5 + 0x01..0xff increment pattern "
        "(PSR_exact / BER / byte recovery)",
    )
    parser.add_argument(
        "--inline-frame-source",
        action="store_true",
        help="also search dewhitened_pdu_hex for the PC frame (only for short "
        "covert payloads fully exported by the parser)",
    )
    parser.add_argument(
        "--pattern-outlier-ber-threshold",
        type=float,
        default=0.10,
        help="per-packet BER above which a packet is treated as a collision "
        "outlier and excluded from pattern_ber_excluding_outliers (default: 0.10)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metadata_path = args.metadata.expanduser().resolve()
    parser_csv = args.parser_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    parser_rows = read_csv(parser_csv)
    candidates, summary = score_candidates(
        metadata,
        parser_rows,
        aa_hamming_tolerance=args.aa_hamming_tolerance,
        dedup_gap_samples=args.dedup_gap_samples,
        magic_hex=args.magic_hex,
        marker_hex=args.marker_hex,
        pattern_enabled=args.pattern,
        inline_frame_source=args.inline_frame_source,
        pattern_outlier_ber_threshold=args.pattern_outlier_ber_threshold,
    )
    summary.update({
        "metadata_path": str(metadata_path),
        "parser_csv": str(parser_csv),
        "output_dir": str(output_dir),
    })
    write_csv(output_dir / "parser_candidate_packets.csv", candidates, DEFAULT_FIELDS)
    write_json(output_dir / "parser_candidate_rate.json", summary)
    write_json(output_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
