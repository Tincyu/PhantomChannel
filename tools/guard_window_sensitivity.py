#!/usr/bin/env python3
"""Recompute boundary-detector AUCs for a guard/window sensitivity grid.

The advertising observations are rescored from the retained 4 MS/s B210 IQ.
The PIP observations are rescored from the corrected, global-coordinate 2M
X310 feature rows and their retained 100 MS/s IQ.  This is a sensitivity
analysis only; it does not replace the frozen 4 us / 64 us formal detector.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from tools.analyze_boundary_detector import auc_and_tpr
from tools.score_boundary_packet_detector import extract_run


DEFAULT_GRID = [(0.5, 4.0), (1.0, 4.0), (1.0, 6.0), (1.0, 8.0), (2.0, 6.0), (4.0, 64.0)]
EXTRA_GRID = [(4.0, 4.0), (8.0, 4.0), (12.0, 4.0), (16.0, 4.0), (16.0, 6.0), (16.0, 8.0), (20.0, 4.0), (24.0, 8.0)]
NOISE_START_US = 1_000.0
NOISE_END_US = 250.0
ADV_DEDUP_GAP_SAMPLES = 200
PIP_DEDUP_GAP_SAMPLES = 100_000


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def value(value: Any, default: float | None = None) -> float | None:
    try:
        parsed = float(str(value).strip())
        return parsed if math.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


def auc(scores: list[float], labels: list[int]) -> float | None:
    if not scores or len(set(labels)) < 2:
        return None
    result, _ = auc_and_tpr(np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int8))
    return result


def run_ids_from_features(path: Path, *, labels: set[str] | None = None) -> list[str]:
    rows = read_csv(path)
    run_ids = sorted({str(row["run_id"]) for row in rows if labels is None or row.get("label") in labels})
    return run_ids


def score_adv_run(root: Path, label: int, guard_us: float, window_us: float) -> list[dict[str, Any]]:
    rows, _ = extract_run(
        root,
        label,
        guard_us=guard_us,
        window_us=window_us,
        noise_start_us=NOISE_START_US,
        noise_end_us=NOISE_END_US,
        dedup_gap_samples=ADV_DEDUP_GAP_SAMPLES,
    )
    return [row for row in rows if int(row.get("collision_ambiguous", 0)) == 0]


def adv_scores(
    roots: list[tuple[Path, int]], guard_us: float, window_us: float
) -> tuple[list[float], list[int]]:
    scores: list[float] = []
    labels: list[int] = []
    for root, label in roots:
        for row in score_adv_run(root, label, guard_us, window_us):
            parsed = value(row.get("tail_energy_db"))
            if parsed is not None:
                scores.append(parsed)
                labels.append(label)
    return scores, labels


def pipsession_parser_csv(session_root: Path, condition: str) -> Path:
    sdr = session_root / "sdr"
    candidates = (
        [
            sdr / "two_stage_known_fake_aa_patched_v2/ble_packets.csv",
            sdr / "two_stage_known_fake_aa_patched/ble_packets.csv",
            sdr / "two_stage_known_fake_aa/ble_packets.csv",
        ]
        if condition == "pip"
        else [sdr / "two_stage_known_aa/ble_packets.csv"]
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"no corrected top-level parser CSV for {session_root}")


def has_activity_fast(starts: list[int], packet_start: int, left: int, right: int, gap: int) -> bool:
    """Equivalent to the PIP scorer's activity test using sorted-index lookup."""

    lo = bisect.bisect_left(starts, left)
    hi = bisect.bisect_right(starts, right)
    for candidate in starts[lo:hi]:
        if abs(candidate - packet_start) > gap:
            return True
    return False


def score_boundary_fast(
    iq: np.memmap,
    starts: list[int],
    packet_start: int,
    boundary: int,
    sample_rate_sps: float,
    guard_us: float,
    window_us: float,
) -> dict[str, Any]:
    """Score one boundary with the frozen PIP formula and fast activity lookup."""

    guard = int(round(guard_us * sample_rate_sps / 1_000_000.0))
    window = int(round(window_us * sample_rate_sps / 1_000_000.0))
    noise_start = int(round(NOISE_START_US * sample_rate_sps / 1_000_000.0))
    noise_end = int(round(NOISE_END_US * sample_rate_sps / 1_000_000.0))
    tail_left = boundary + guard
    tail_right = tail_left + window
    noise_left = packet_start - noise_start
    noise_right = packet_start - noise_end
    result: dict[str, Any] = {
        "score_db": None,
        "valid": 0,
        "ambiguous": 0,
        "window_complete": 0,
    }
    if noise_left < 0 or tail_right > len(iq) or noise_right <= noise_left:
        return result
    result["window_complete"] = 1
    if has_activity_fast(starts, packet_start, tail_left - guard, tail_right, PIP_DEDUP_GAP_SAMPLES):
        result["ambiguous"] = 1
    if has_activity_fast(starts, packet_start, noise_left, noise_right, PIP_DEDUP_GAP_SAMPLES):
        result["ambiguous"] = 1
    noise_values = np.asarray(iq[noise_left:noise_right], dtype=np.float32)
    tail_values = np.asarray(iq[tail_left:tail_right], dtype=np.float32)
    if not noise_values.size or not tail_values.size:
        return result
    noise_power = float(np.median(noise_values[:, 0] ** 2 + noise_values[:, 1] ** 2))
    tail_power = float(np.mean(tail_values[:, 0] ** 2 + tail_values[:, 1] ** 2))
    result["score_db"] = float(10.0 * np.log10(max(tail_power, 1e-12) / max(noise_power, 1e-12)))
    result["valid"] = 1
    return result


def pipsession_scores(
    session_root: Path,
    feature_path: Path,
    guard_us: float,
    window_us: float,
) -> tuple[list[float], list[int], dict[str, Any]]:
    feature_rows = read_csv(feature_path)
    if not feature_rows:
        return [], [], {"session_id": session_root.name, "condition": "", "rows": 0}
    condition = str(feature_rows[0].get("condition", "")).strip().lower()
    parser_rows = read_csv(pipsession_parser_csv(session_root, condition))
    starts = sorted(
        int(value(row.get("wideband_sample_index", row.get("sample_index")), -1) or -1)
        for row in parser_rows
        if value(row.get("wideband_sample_index", row.get("sample_index"))) is not None
    )
    metadata = json.loads((session_root / "iq/metadata.json").read_text(encoding="utf-8"))
    sample_rate = float(metadata.get("actual_sample_rate_sps") or metadata.get("sample_rate_sps"))
    iq_path = session_root / "iq/capture.sc16"
    samples = int(metadata.get("samples") or (iq_path.stat().st_size // 4))
    iq = np.memmap(iq_path, dtype="<i2", mode="r", shape=(samples, 2))

    scores: list[float] = []
    labels: list[int] = []
    valid_count = 0
    ambiguous_count = 0
    for row in feature_rows:
        packet_start = int(value(row.get("sample_start"), -1) or -1)
        boundary_real = int(value(row.get("boundary_real"), -1) or -1)
        boundary_fake = int(value(row.get("boundary_fake"), -1) or -1)
        boundary = boundary_fake if condition == "pip" else boundary_real
        result = score_boundary_fast(
            iq,
            starts,
            packet_start,
            boundary,
            sample_rate,
            guard_us,
            window_us,
        )
        valid = bool(result["valid"])
        if condition == "pip" and row.get("mapping_ok") not in {"1", "true", "True"}:
            valid = False
        if not valid:
            continue
        valid_count += 1
        if result["ambiguous"]:
            ambiguous_count += 1
            continue
        score = value(result.get("score_db"))
        if score is not None:
            scores.append(score)
            labels.append(1 if condition == "pip" else 0)
    return scores, labels, {
        "session_id": session_root.name,
        "condition": condition,
        "rows": len(feature_rows),
        "valid": valid_count,
        "ambiguous": ambiguous_count,
    }


def pip_metric(
    sessions: list[tuple[Path, Path]],
    positive_condition: str,
    guard_us: float,
    window_us: float,
) -> tuple[float | None, int, int, list[dict[str, Any]]]:
    scores: list[float] = []
    labels: list[int] = []
    session_summaries: list[dict[str, Any]] = []
    for root, feature_path in sessions:
        condition = read_csv(feature_path)[0].get("condition", "")
        local_scores, local_labels, summary = pipsession_scores(root, feature_path, guard_us, window_us)
        # The feature files are the corrected target selections.  For PIP
        # outer, the score function above uses fake boundary; for direct and
        # benign it uses the real/ordinary boundary.
        scores.extend(local_scores)
        labels.extend([1 if condition == positive_condition else 0 for _ in local_scores])
        session_summaries.append(summary)
    return auc(scores, labels), sum(label == 0 for label in labels), sum(label == 1 for label in labels), session_summaries


def build_roots(testdata: Path, detector_root: Path, pip_root: Path) -> tuple[list[tuple[Path, int]], list[tuple[Path, Path]]]:
    formal_features = detector_root / "boundary_formal_main/boundary_packet_features.csv"
    len8_features = detector_root / "boundary_sensitivity_len8/boundary_packet_features.csv"
    formal_ids = run_ids_from_features(formal_features)
    len8_ids = run_ids_from_features(len8_features, labels={"1"})
    adv_239b = [
        (testdata / run_id, 0 if "_benign_" in run_id else 1)
        for run_id in formal_ids
        if "_g35_" in run_id
    ]
    adv_8b = [
        (testdata / run_id, 0)
        for run_id in formal_ids
        if "_benign_g35_" in run_id
    ] + [(testdata / run_id, 1) for run_id in len8_ids]
    if not all(root.exists() for root, _ in adv_239b + adv_8b):
        missing = [str(root) for root, _ in adv_239b + adv_8b if not root.exists()]
        raise FileNotFoundError("missing advertising run roots: " + ", ".join(missing))

    pip_sessions: list[tuple[Path, Path]] = []
    sessions_root = pip_root / "sessions"
    for condition, prefix in (("benign", "pip_auc_benign_rep"), ("direct_tail", "pip_auc_direct_tail_rep"), ("pip", "pip_auc_pip_rep")):
        for index in range(1, 6):
            session_root = sessions_root / f"{prefix}{index}"
            feature = session_root / "sdr/pip_boundary_features_global_corrected/pip_boundary_packet_features.csv"
            if not session_root.exists() or not feature.exists():
                raise FileNotFoundError(f"missing corrected PIP input: {session_root} / {feature}")
            pip_sessions.append((session_root, feature))
    return adv_239b, adv_8b, pip_sessions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testdata", type=Path, default=Path("/path/to/PhantomChannel/testdata"))
    parser.add_argument("--detector-root", type=Path, default=Path("/path/to/PhantomChannel/experiments/figure/detector_roc_20260808"))
    parser.add_argument("--pip-root", type=Path, default=Path("/path/to/PhantomChannel/experiments/figure/pip_boundary_auc_20260808"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-extra", action="store_true", help="also test guard values around 16 us")
    args = parser.parse_args()
    grid = DEFAULT_GRID + (EXTRA_GRID if args.include_extra else [])
    adv_239b, adv_8b, pip_sessions = build_roots(args.testdata, args.detector_root, args.pip_root)
    direct_sessions = [item for item in pip_sessions if item[0].name.startswith("pip_auc_benign_rep") or item[0].name.startswith("pip_auc_direct_tail_rep")]
    outer_sessions = [item for item in pip_sessions if item[0].name.startswith("pip_auc_benign_rep") or item[0].name.startswith("pip_auc_pip_rep")]

    output_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for guard_us, window_us in grid:
        adv239_scores, adv239_labels = adv_scores(adv_239b, guard_us, window_us)
        adv8_scores, adv8_labels = adv_scores(adv_8b, guard_us, window_us)
        direct_auc, direct_n0, direct_n1, direct_details = pip_metric(direct_sessions, "direct_tail", guard_us, window_us)
        outer_auc, outer_n0, outer_n1, outer_details = pip_metric(outer_sessions, "pip", guard_us, window_us)
        row = {
            "guard_us": guard_us,
            "window_us": window_us,
            "adv_239b_auc": auc(adv239_scores, adv239_labels),
            "adv_8b_auc": auc(adv8_scores, adv8_labels),
            "direct_2b_auc": direct_auc,
            "pip_outer_auc": outer_auc,
            "adv_239b_benign_packets": sum(label == 0 for label in adv239_labels),
            "adv_239b_covert_packets": sum(label == 1 for label in adv239_labels),
            "adv_8b_benign_packets": sum(label == 0 for label in adv8_labels),
            "adv_8b_covert_packets": sum(label == 1 for label in adv8_labels),
            "direct_2b_benign_packets": direct_n0,
            "direct_2b_covert_packets": direct_n1,
            "pip_outer_benign_packets": outer_n0,
            "pip_outer_covert_packets": outer_n1,
            "preferred": int(guard_us == 1.0 and window_us == 6.0),
            "note": "sensitivity_only; 4/64 formal detector remains frozen",
        }
        output_rows.append(row)
        for detail in direct_details + outer_details:
            detail_rows.append({"guard_us": guard_us, "window_us": window_us, **detail})
        print(json.dumps({"guard_us": guard_us, "window_us": window_us, **{key: row[key] for key in ("adv_239b_auc", "adv_8b_auc", "direct_2b_auc", "pip_outer_auc")}}, ensure_ascii=False))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "guard_window_sensitivity.csv", output_rows)
    write_csv(args.output_dir / "guard_window_sensitivity_session_counts.csv", detail_rows)
    report = {
        "schema_version": 1,
        "detector": "advertising_and_pip_boundary_tail_energy",
        "grid": [{"guard_us": g, "window_us": w} for g, w in grid],
        "preferred": {"guard_us": 1.0, "window_us": 6.0},
        "sources": {
            "adv_239b": "boundary_formal_main, gain=35 dB; 5 benign + 5 covert sessions",
            "adv_8b": "boundary_sensitivity_len8, gain=35 dB; 5 benign + 3 covert sessions",
            "direct_2b": "corrected global-coordinate PIP formal benign vs direct_tail; 5 + 5 sessions",
            "pip_outer": "corrected global-coordinate PIP formal benign vs pip; 5 + 5 sessions",
        },
        "scoring": {
            "score": "10log10(mean(power[boundary+guard:boundary+guard+W])/median(independent_idle_noise))",
            "noise_start_us": NOISE_START_US,
            "noise_end_us": NOISE_END_US,
            "advertising_dedup_gap_samples": ADV_DEDUP_GAP_SAMPLES,
            "pip_dedup_gap_samples": PIP_DEDUP_GAP_SAMPLES,
            "auc_direction": "higher_score_more_boundary_anomaly_no_auto_flip",
        },
        "caveat": "This grid is an offline sensitivity analysis and does not redefine the frozen 4 us / 64 us formal detector.",
        "results_csv": str(args.output_dir / "guard_window_sensitivity.csv"),
    }
    (args.output_dir / "guard_window_sensitivity.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
