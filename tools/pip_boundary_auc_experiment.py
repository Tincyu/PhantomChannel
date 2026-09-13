#!/usr/bin/env python3
"""Extract and aggregate the matched 2M PIP boundary-evasion experiment.

The extractor deliberately accepts the raw stage-2 CSV that was used to make
``pip_x310_audit.py``'s JSON.  The merged CSV is not accepted as an audit
companion because its row indices no longer identify the audit records.

The score is the frozen normalized post-boundary tail energy:

    10 log10(mean(tail power) / median(independent idle-noise power))

For PIP packets the extractor emits both the parser-visible fake/outer view
and the mapped real/inner diagnostic view.  It never drops a formal session
because the audit or decoder failed; failures become coverage or an explicit
exclusion reason.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from tools.analyze_boundary_detector import auc_and_tpr, fixed_threshold


DEFAULT_GUARD_US = 4.0
DEFAULT_WINDOW_US = 64.0
DEFAULT_NOISE_START_US = 1000.0
DEFAULT_NOISE_END_US = 250.0
DEFAULT_DEDUP_GAP_SAMPLES = 5000
SC16_BYTES_PER_COMPLEX_SAMPLE = 4

CONDITIONS = {"benign", "direct_tail", "pip"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def number(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(str(value).strip())
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def integer(value: Any, default: int | None = None) -> int | None:
    result = number(value)
    return int(result) if result is not None else default


def normalize_aa(value: Any) -> str:
    text = str(value or "").strip().replace(":", "").replace("-", "")
    text = text[2:] if text.lower().startswith("0x") else text
    if not text:
        return ""
    try:
        return f"{int(text, 16):08X}"
    except ValueError:
        return ""


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_ledger(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".csv":
        rows = read_csv(path)
        if len(rows) != 1:
            raise ValueError("a session ledger CSV must contain exactly one session row")
        return dict(rows[0])
    value = load_json(path)
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("a session ledger JSON list must contain exactly one session")
        value = value[0]
    if not isinstance(value, dict):
        raise ValueError("session ledger must be a JSON object or one-row CSV")
    return value


def load_audit(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    value = load_json(path)
    records = value.get("records", []) if isinstance(value, dict) else value
    if not isinstance(records, list):
        raise ValueError("audit JSON records must be a list")
    output: dict[int, dict[str, Any]] = {}
    for record in records:
        index = integer(record.get("row_index")) if isinstance(record, dict) else None
        if index is not None:
            output[index] = record
    return output


def parser_sample_index(row: dict[str, Any]) -> int | None:
    """Use full-rate X310 coordinates; fall back to parser sample_index for tests."""

    wide = integer(row.get("wideband_sample_index"))
    if wide is not None:
        return wide
    return integer(row.get("sample_index"))


def phy_rate_sps(phy: str) -> float:
    value = phy.lower().replace(" ", "")
    if value in {"2m", "2mphy", "2"}:
        return 2_000_000.0
    if value in {"1m", "1mphy", "1"}:
        return 1_000_000.0
    raise ValueError(f"unsupported PHY {phy!r}; expected 1M or 2M")


def preamble_bytes(phy: str) -> int:
    return 2 if phy_rate_sps(phy) == 2_000_000.0 else 1


def boundary_sample_index(
    packet_start: int,
    declared_length_bytes: int,
    sample_rate_sps: float,
    phy: str,
) -> int:
    """Return the standard outer packet boundary after BLE CRC."""

    if declared_length_bytes < 0:
        raise ValueError("declared packet length must be non-negative")
    total_bytes = preamble_bytes(phy) + 4 + 2 + declared_length_bytes + 3
    return packet_start + int(round(total_bytes * 8.0 * sample_rate_sps / phy_rate_sps(phy)))


def pip_real_boundary_sample_index(
    packet_start: int,
    real_length_bytes: int,
    sample_rate_sps: float,
    phy: str,
) -> int:
    """Boundary after the mapped inner real CRC in the FORCE-PIP burst.

    The inner packet has no second physical preamble.  The path is:
    fake preamble + fake AA + fake header + real AA + real header + real
    payload + real CRC.
    """

    if real_length_bytes < 0:
        raise ValueError("real packet length must be non-negative")
    total_bytes = preamble_bytes(phy) + 4 + 2 + 4 + 2 + real_length_bytes + 3
    return packet_start + int(round(total_bytes * 8.0 * sample_rate_sps / phy_rate_sps(phy)))


def deduplicate_rows(
    rows: Iterable[dict[str, Any]],
    gap_samples: int,
) -> list[dict[str, Any]]:
    ordered = sorted(
        (row for row in rows if parser_sample_index(row) is not None),
        key=lambda row: int(parser_sample_index(row)),
    )
    selected: list[dict[str, Any]] = []
    for row in ordered:
        if not selected:
            selected.append(row)
            continue
        current = int(parser_sample_index(row))
        previous = int(parser_sample_index(selected[-1]))
        if current - previous <= gap_samples:
            current_confidence = number(row.get("confidence_score"), -1.0) or -1.0
            previous_confidence = number(selected[-1].get("confidence_score"), -1.0) or -1.0
            if current_confidence > previous_confidence:
                selected[-1] = row
        else:
            selected.append(row)
    return selected


def power_slice(iq: np.memmap, start: int, end: int) -> np.ndarray:
    if start < 0 or end <= start or end > len(iq):
        return np.empty(0, dtype=np.float32)
    values = np.asarray(iq[start:end], dtype=np.float32)
    return values[:, 0] * values[:, 0] + values[:, 1] * values[:, 1]


def has_activity(
    starts: list[int],
    start: int,
    left: int,
    right: int,
    dedup_gap_samples: int,
) -> bool:
    for candidate in starts:
        if abs(candidate - start) <= dedup_gap_samples:
            continue
        if left <= candidate <= right:
            return True
        if candidate > right:
            break
    return False


def score_boundary(
    iq: np.memmap,
    starts: list[int],
    packet_start: int,
    boundary: int | None,
    sample_rate_sps: float,
    guard_us: float,
    window_us: float,
    noise_start_us: float,
    noise_end_us: float,
    dedup_gap_samples: int,
) -> dict[str, Any]:
    result = {
        "score_db": None,
        "valid": 0,
        "ambiguous": 0,
        "reason": "",
        "window_complete": 0,
    }
    if boundary is None:
        result["reason"] = "boundary_unavailable"
        return result
    guard = int(round(guard_us * sample_rate_sps / 1_000_000.0))
    window = int(round(window_us * sample_rate_sps / 1_000_000.0))
    noise_start = int(round(noise_start_us * sample_rate_sps / 1_000_000.0))
    noise_end = int(round(noise_end_us * sample_rate_sps / 1_000_000.0))
    tail_left = boundary + guard
    tail_right = tail_left + window
    noise_left = packet_start - noise_start
    noise_right = packet_start - noise_end
    if noise_left < 0 or tail_right > len(iq) or noise_right <= noise_left:
        result["reason"] = "iq_window_incomplete"
        return result
    result["window_complete"] = 1
    if has_activity(starts, packet_start, tail_left - guard, tail_right, dedup_gap_samples):
        result["ambiguous"] = 1
        result["reason"] = "overlapping_parser_activity"
    if has_activity(starts, packet_start, noise_left, noise_right, dedup_gap_samples):
        result["ambiguous"] = 1
        result["reason"] = (
            "noise_reference_activity"
            if not result["reason"]
            else result["reason"] + ";noise_reference_activity"
        )
    noise = power_slice(iq, noise_left, noise_right)
    tail = power_slice(iq, tail_left, tail_right)
    if not noise.size or not tail.size:
        result["reason"] = "empty_power_window"
        return result
    noise_power = float(np.median(noise))
    tail_power = float(np.mean(tail))
    result["score_db"] = float(10.0 * np.log10(max(tail_power, 1e-12) / max(noise_power, 1e-12)))
    result["valid"] = 1
    return result


def _ledger_value(ledger: dict[str, Any], name: str, default: Any = "") -> Any:
    value = ledger.get(name, default)
    return default if value is None else value


def extract_session(
    *,
    iq_path: Path,
    metadata_path: Path,
    parser_csv: Path,
    audit_json: Path | None,
    ledger_path: Path,
    output_dir: Path,
    guard_us: float = DEFAULT_GUARD_US,
    window_us: float = DEFAULT_WINDOW_US,
    noise_start_us: float = DEFAULT_NOISE_START_US,
    noise_end_us: float = DEFAULT_NOISE_END_US,
    dedup_gap_samples: int = DEFAULT_DEDUP_GAP_SAMPLES,
) -> dict[str, Any]:
    metadata = load_json(metadata_path)
    ledger = load_ledger(ledger_path)
    session_id = str(_ledger_value(ledger, "session_id", ledger_path.stem))
    condition = str(_ledger_value(ledger, "condition", "")).strip().lower()
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {sorted(CONDITIONS)}, got {condition!r}")
    phy = str(_ledger_value(ledger, "phy", "2m"))
    sample_rate_sps = float(metadata.get("actual_sample_rate_sps") or metadata.get("sample_rate_sps"))
    if sample_rate_sps <= 0:
        raise ValueError("metadata must provide actual_sample_rate_sps")
    real_aa = normalize_aa(_ledger_value(ledger, "real_aa"))
    fake_aa = normalize_aa(_ledger_value(ledger, "fake_aa"))
    if not real_aa:
        raise ValueError("ledger must provide real_aa; do not infer or hard-code historical AAs")
    if condition == "pip" and not fake_aa:
        raise ValueError("PIP ledger must provide fake_aa")

    parser_rows = read_csv(parser_csv)
    audit_by_index = load_audit(audit_json)
    indexed_rows: list[dict[str, Any]] = []
    all_starts: list[int] = []
    for index, row in enumerate(parser_rows):
        start = parser_sample_index(row)
        if start is None:
            continue
        row = dict(row)
        row["_parser_row_index"] = index
        row["_sample_start"] = start
        indexed_rows.append(row)
        all_starts.append(start)
    all_starts.sort()

    target_aa = fake_aa if condition == "pip" else real_aa
    target_payload_len = integer(_ledger_value(ledger, "target_payload_len", ""))
    candidates = [
        row
        for row in indexed_rows
        if normalize_aa(row.get("access_address")) == target_aa
        and str(row.get("packet_type", "")).upper() == "BLE_CONN"
        and (
            condition == "pip"
            or target_payload_len is None
            or integer(row.get("payload_len")) == target_payload_len
        )
    ]
    candidates = deduplicate_rows(candidates, dedup_gap_samples)
    iq_samples = int(metadata.get("samples") or (iq_path.stat().st_size // SC16_BYTES_PER_COMPLEX_SAMPLE))
    iq = np.memmap(iq_path, dtype="<i2", mode="r", shape=(iq_samples, 2))
    features: list[dict[str, Any]] = []
    for row in candidates:
        row_index = int(row["_parser_row_index"])
        audit = audit_by_index.get(row_index, {})
        start = int(row["_sample_start"])
        fake_len = integer(audit.get("fake_len")) or integer(row.get("payload_len"))
        real_len = integer(audit.get("real_len"))
        if condition != "pip":
            real_len = integer(row.get("payload_len"))
            fake_len = real_len
        mapping_real_aa = normalize_aa(audit.get("real_aa_from_mapping_canonical"))
        mapping_ok = condition != "pip" or mapping_real_aa == real_aa
        boundary_fake = (
            boundary_sample_index(start, fake_len, sample_rate_sps, phy)
            if fake_len is not None
            else None
        )
        boundary_real = (
            boundary_sample_index(start, real_len, sample_rate_sps, phy)
            if condition != "pip" and real_len is not None
            else (
                pip_real_boundary_sample_index(start, real_len, sample_rate_sps, phy)
                if real_len is not None and mapping_ok
                else None
            )
        )
        fake_score = score_boundary(
            iq, all_starts, start, boundary_fake, sample_rate_sps,
            guard_us, window_us, noise_start_us, noise_end_us, dedup_gap_samples,
        )
        real_score = score_boundary(
            iq, all_starts, start, boundary_real, sample_rate_sps,
            guard_us, window_us, noise_start_us, noise_end_us, dedup_gap_samples,
        )
        reasons = []
        if condition == "pip" and not audit:
            reasons.append("audit_row_unavailable")
        if condition == "pip" and audit and not mapping_ok:
            reasons.append("mapping_mismatch")
        for score in (fake_score, real_score):
            if score["reason"]:
                reasons.append(score["reason"])
        features.append(
            {
                "session_id": session_id,
                "condition": condition,
                "tx_index": audit.get("tx_index", ledger.get("tx_index", "")),
                "parser_row_index": row_index,
                "real_aa": real_aa,
                "fake_aa": fake_aa,
                "boundary_real": boundary_real if boundary_real is not None else "",
                "boundary_fake": boundary_fake if boundary_fake is not None else "",
                "E_tail_real_db": real_score["score_db"] if real_score["valid"] else "",
                "E_tail_fake_db": fake_score["score_db"] if fake_score["valid"] else "",
                "E_tail_outer_db": (
                    fake_score["score_db"] if condition == "pip" and fake_score["valid"]
                    else real_score["score_db"] if real_score["valid"] else ""
                ),
                "real_score_valid": int(real_score["valid"] and mapping_ok),
                "fake_score_valid": int(fake_score["valid"]),
                "ambiguous": int(real_score["ambiguous"] or fake_score["ambiguous"]),
                "real_ambiguous": real_score["ambiguous"],
                "fake_ambiguous": fake_score["ambiguous"],
                "iq_window_complete": int(real_score["window_complete"] or fake_score["window_complete"]),
                "mapping_ok": int(mapping_ok),
                "audit_decode_status": audit.get("decode_status", ""),
                "fake_len": fake_len if fake_len is not None else "",
                "real_len": real_len if real_len is not None else "",
                "sample_start": start,
                "channel": row.get("channel", ""),
                "rssi": row.get("rssi", ""),
                "exclusion_reason": ";".join(dict.fromkeys(reasons)),
            }
        )
    tx_count = integer(
        _ledger_value(ledger, "tx_done_count", _ledger_value(ledger, "tx_count", ""))
    )
    real_valid = [row for row in features if row["real_score_valid"]]
    fake_valid = [row for row in features if row["fake_score_valid"]]
    summary = {
        "session_id": session_id,
        "condition": condition,
        "phy": phy,
        "tx_count": tx_count if tx_count is not None else "",
        "parser_target_count": len(features),
        "eligible_real": sum(not row["ambiguous"] for row in real_valid),
        "eligible_fake": sum(not row["ambiguous"] for row in fake_valid),
        "C_real": (len(real_valid) / tx_count) if tx_count else None,
        "C_fake": (len(fake_valid) / tx_count) if tx_count else None,
        "ambiguous_fraction": (
            sum(row["ambiguous"] for row in features) / len(features) if features else None
        ),
        "measured_snr_db": _ledger_value(ledger, "measured_snr_db", ""),
        "sample_rate_sps": sample_rate_sps,
        "guard_us": guard_us,
        "window_us": window_us,
        "score_direction": "higher_score_more_boundary_anomaly",
        "parser_csv": str(parser_csv),
        "audit_json": str(audit_json) if audit_json else "",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "pip_boundary_packet_features.csv", features)
    write_csv(output_dir / "pip_boundary_session_summary.csv", [summary])
    (output_dir / "pip_boundary_extraction_config.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "session_id": session_id,
                "condition": condition,
                "phy": phy,
                "sample_rate_sps": sample_rate_sps,
                "guard_us": guard_us,
                "window_us": window_us,
                "noise_start_us": noise_start_us,
                "noise_end_us": noise_end_us,
                "dedup_gap_samples": dedup_gap_samples,
                "target_payload_len": target_payload_len if target_payload_len is not None else "",
                "merged_csv_warning": "use raw stage2 CSV paired with the audit JSON",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def read_feature_files(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_csv(path))
    return rows


def numeric_score(row: dict[str, Any], field: str) -> float | None:
    if str(row.get("ambiguous", "0")) in {"1", "true", "True"}:
        return None
    valid_field = "real_score_valid" if field == "E_tail_real_db" else "fake_score_valid"
    if str(row.get(valid_field, "0")) not in {"1", "true", "True"}:
        return None
    return number(row.get(field))


def cluster_bootstrap(
    records: list[dict[str, Any]],
    score_field: str,
    negative_condition: str,
    positive_condition: str,
    threshold: float | None,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    usable = []
    for row in records:
        score = numeric_score(row, score_field)
        if score is None or row.get("condition") not in {negative_condition, positive_condition}:
            continue
        usable.append({**row, "_score": score})
    labels = np.asarray([int(row["condition"] == positive_condition) for row in usable], dtype=np.int8)
    scores = np.asarray([row["_score"] for row in usable], dtype=np.float64)
    auc, tpr5 = auc_and_tpr(scores, labels)
    fpr_threshold, tpr_threshold = fixed_threshold(scores, labels, threshold) if threshold is not None else (None, None)
    by_session: dict[str, list[dict[str, Any]]] = {}
    for row in usable:
        by_session.setdefault(str(row.get("session_id", "")), []).append(row)
    negative_sessions = [s for s, rows in by_session.items() if rows[0]["condition"] == negative_condition]
    positive_sessions = [s for s, rows in by_session.items() if rows[0]["condition"] == positive_condition]
    rng = np.random.default_rng(seed)
    boot_auc: list[float] = []
    boot_tpr5: list[float] = []
    boot_fpr: list[float] = []
    boot_tpr: list[float] = []
    for _ in range(replicates):
        chosen = []
        for pool in (negative_sessions, positive_sessions):
            if not pool:
                chosen = []
                break
            chosen.extend(rng.choice(pool, size=len(pool), replace=True).tolist())
        if not chosen:
            continue
        sampled = [row for session in chosen for row in by_session[session]]
        boot_scores = np.asarray([row["_score"] for row in sampled], dtype=np.float64)
        boot_labels = np.asarray([int(row["condition"] == positive_condition) for row in sampled], dtype=np.int8)
        b_auc, b_tpr5 = auc_and_tpr(boot_scores, boot_labels)
        b_fpr, b_tpr = fixed_threshold(boot_scores, boot_labels, threshold) if threshold is not None else (None, None)
        if None not in (b_auc, b_tpr5):
            boot_auc.append(float(b_auc))
            boot_tpr5.append(float(b_tpr5))
        if b_fpr is not None and b_tpr is not None:
            boot_fpr.append(float(b_fpr))
            boot_tpr.append(float(b_tpr))

    def ci(values: list[float]) -> list[float] | None:
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))] if values else None

    return {
        "score_field": score_field,
        "negative_condition": negative_condition,
        "positive_condition": positive_condition,
        "auc": auc,
        "tpr_at_fpr_0.05": tpr5,
        "theta_5_pip": threshold,
        "fpr_theta_5": fpr_threshold,
        "tpr_theta_5": tpr_threshold,
        "negative_sessions": len(negative_sessions),
        "positive_sessions": len(positive_sessions),
        "negative_packets": int(np.sum(labels == 0)),
        "positive_packets": int(np.sum(labels == 1)),
        "cluster_bootstrap_replicates": len(boot_auc),
        "auc_ci95": ci(boot_auc),
        "tpr_at_fpr_0.05_ci95": ci(boot_tpr5),
        "fpr_theta_5_ci95": ci(boot_fpr),
        "tpr_theta_5_ci95": ci(boot_tpr),
        "score_direction": "higher_score_more_boundary_anomaly_no_auto_flip",
    }


def choose_theta_5(benign_scores: list[float]) -> float | None:
    """Choose the lowest observed threshold whose calibration FPR is <= 5%."""

    if not benign_scores:
        return None
    for threshold in sorted(set(benign_scores)):
        if sum(score >= threshold for score in benign_scores) / len(benign_scores) <= 0.05:
            return float(threshold)
    # With fewer than 20 independent calibration packets, one observed score
    # already exceeds the requested 5% empirical FPR.  Do not manufacture an
    # infinite threshold and report a misleading all-zero test operating point.
    return None


def aggregate_features(
    feature_paths: list[Path],
    output_dir: Path,
    *,
    calibration_paths: list[Path] | None = None,
    seed: int = 20260808,
    replicates: int = 2000,
) -> dict[str, Any]:
    records = read_feature_files(feature_paths)
    calibration = read_feature_files(calibration_paths or feature_paths)
    calibration_scores = [
        score
        for row in calibration
        if row.get("condition") == "benign"
        for score in [numeric_score(row, "E_tail_outer_db")]
        if score is not None
    ]
    threshold = choose_theta_5(calibration_scores)
    metrics = {
        "AUC_direct": cluster_bootstrap(
            records, "E_tail_real_db", "benign", "direct_tail", threshold, seed, replicates
        ),
        "AUC_pip_outer": cluster_bootstrap(
            records, "E_tail_outer_db", "benign", "pip", threshold, seed + 1, replicates
        ),
        "AUC_pip_real": cluster_bootstrap(
            records, "E_tail_real_db", "benign", "pip", threshold, seed + 2, replicates
        ),
    }
    report = {
        "schema_version": 1,
        "detector": "pip_boundary_evasion_tail_energy",
        "phy": "2m",
        "guard_us": DEFAULT_GUARD_US,
        "window_us": DEFAULT_WINDOW_US,
        "theta_5_pip": threshold,
        "calibration_score_count": len(calibration_scores),
        "theta_5_pip_status": (
            "available" if threshold is not None else "unavailable_insufficient_empirical_resolution"
        ),
        "metrics": metrics,
        "score_direction": "higher_score_more_boundary_anomaly_no_auto_flip",
        "source_features": [str(path) for path in feature_paths],
        "calibration_features": [str(path) for path in (calibration_paths or feature_paths)],
        "notes": [
            "AUC_pip_outer is the primary PIP evasion result.",
            "AUC_pip_real is mapped/oracle-assisted diagnostic in FORCE mode.",
            "AUC is never replaced by max(AUC, 1-AUC).",
            "Bootstrap resamples complete sessions within each condition.",
            "Formal session inclusion is not conditioned on audit/decode success.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pip_boundary_auc.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv(
        output_dir / "pip_boundary_auc.csv",
        [
            {"metric": name, **value}
            for name, value in metrics.items()
        ],
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iq-path", type=Path)
    parser.add_argument("--metadata-path", type=Path)
    parser.add_argument("--parser-csv", type=Path, help="raw stage2 CSV paired with --audit-json")
    parser.add_argument("--audit-json", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--features-csv", action="append", type=Path, default=[])
    parser.add_argument("--calibration-features-csv", action="append", type=Path, default=[])
    parser.add_argument("--guard-us", type=float, default=DEFAULT_GUARD_US)
    parser.add_argument("--window-us", type=float, default=DEFAULT_WINDOW_US)
    parser.add_argument("--noise-start-us", type=float, default=DEFAULT_NOISE_START_US)
    parser.add_argument("--noise-end-us", type=float, default=DEFAULT_NOISE_END_US)
    parser.add_argument("--dedup-gap-samples", type=int, default=DEFAULT_DEDUP_GAP_SAMPLES)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args()

    if args.features_csv:
        report = aggregate_features(
            args.features_csv,
            args.output_dir,
            calibration_paths=args.calibration_features_csv or None,
            seed=args.seed,
            replicates=args.bootstrap,
        )
        print(json.dumps({"mode": "aggregate", "metrics": list(report["metrics"]), "output_dir": str(args.output_dir)}))
        return 0
    required = (args.iq_path, args.metadata_path, args.parser_csv, args.ledger)
    if any(value is None for value in required):
        parser.error("extraction requires --iq-path, --metadata-path, --parser-csv, and --ledger")
    summary = extract_session(
        iq_path=args.iq_path,
        metadata_path=args.metadata_path,
        parser_csv=args.parser_csv,
        audit_json=args.audit_json,
        ledger_path=args.ledger,
        output_dir=args.output_dir,
        guard_us=args.guard_us,
        window_us=args.window_us,
        noise_start_us=args.noise_start_us,
        noise_end_us=args.noise_end_us,
        dedup_gap_samples=args.dedup_gap_samples,
    )
    print(json.dumps({"mode": "extract", "summary": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
