#!/usr/bin/env python3
"""Run the minimal session-level internal validation for the redesigned PIP window.

The five folds hold out one benign, one direct-tail, and one PIP session.  Window
selection is deliberately performed without reading any PIP scores: it uses
the advertising references, the training benign/direct-tail sessions, physical
window validity, and stability of the short-window neighborhood.  PIP scores
are computed only after the fold parameter has been selected.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from tools.analyze_boundary_detector import auc_and_tpr
from tools.guard_window_sensitivity import (
    EXTRA_GRID,
    adv_scores,
    build_roots,
    pipsession_scores,
    read_csv,
    score_adv_run,
    value,
)
from tools.score_boundary_packet_detector import extract_run


REFERENCE = (4.0, 64.0)
ADV239_MIN_AUC = 0.99
ADV8_MIN_AUC = 0.99
DIRECT_MIN_AUC = 0.95
MAX_ADV_INVALID_WINDOW_FRACTION = 0.01
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260809

# The short-window neighborhood is intentionally defined by the same guard
# with adjacent available W values.  This makes 16/{4,6,8} a three-point
# stability neighborhood without optimizing on PIP AUC or calling any point a
# best parameter.
SHORT_GRID = [candidate for candidate in EXTRA_GRID if candidate != REFERENCE]


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


def finite_float(value_: Any) -> float | None:
    try:
        parsed = float(str(value_).strip())
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def metric(scores: list[float], labels: list[int]) -> float | None:
    if not scores or len(set(labels)) < 2:
        return None
    result, _ = auc_and_tpr(np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int8))
    return result


def candidate_key(candidate: tuple[float, float]) -> str:
    return f"{candidate[0]:g}/{candidate[1]:g}"


def cluster_bootstrap_auc(records: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    scores = np.asarray([float(row["score_db"]) for row in records], dtype=np.float64)
    labels = np.asarray([int(row["label"]) for row in records], dtype=np.int8)
    auc_value, tpr5 = auc_and_tpr(scores, labels)
    by_label: dict[int, dict[str, list[float]]] = {0: {}, 1: {}}
    for row in records:
        label = int(row["label"])
        cluster = str(row["session_id"])
        by_label[label].setdefault(cluster, []).append(float(row["score_db"]))

    rng = np.random.default_rng(seed)
    boot: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_scores: list[float] = []
        sampled_labels: list[int] = []
        for label in (0, 1):
            clusters = sorted(by_label[label])
            sampled = rng.choice(clusters, size=len(clusters), replace=True)
            for cluster in sampled:
                sampled_scores.extend(by_label[label][str(cluster)])
                sampled_labels.extend([label] * len(by_label[label][str(cluster)]))
        boot_auc, _ = auc_and_tpr(
            np.asarray(sampled_scores, dtype=np.float64),
            np.asarray(sampled_labels, dtype=np.int8),
        )
        if boot_auc is not None:
            boot.append(float(boot_auc))

    return {
        "auc": auc_value,
        "tpr_at_fpr_05": tpr5,
        "session_count": len({str(row["session_id"]) for row in records}),
        "benign_session_count": len(by_label[0]),
        "positive_session_count": len(by_label[1]),
        "packet_count": len(records),
        "benign_packet_count": int(np.sum(labels == 0)),
        "positive_packet_count": int(np.sum(labels == 1)),
        "cluster_bootstrap_replicates": len(boot),
        "auc_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if boot else None,
    }


def load_sessions(pip_root: Path) -> dict[str, list[tuple[Path, Path]]]:
    sessions_root = pip_root / "sessions"
    result: dict[str, list[tuple[Path, Path]]] = {"benign": [], "direct_tail": [], "pip": []}
    for condition, prefix in (
        ("benign", "pip_auc_benign_rep"),
        ("direct_tail", "pip_auc_direct_tail_rep"),
        ("pip", "pip_auc_pip_rep"),
    ):
        for index in range(1, 6):
            session_root = sessions_root / f"{prefix}{index}"
            feature_path = session_root / "sdr/pip_boundary_features_global_corrected/pip_boundary_packet_features.csv"
            if not session_root.exists() or not feature_path.exists():
                raise FileNotFoundError(f"missing corrected PIP input: {session_root} / {feature_path}")
            result[condition].append((session_root, feature_path))
    return result


def cache_pip_scores(
    sessions: list[tuple[Path, Path]],
    grid: list[tuple[float, float]],
) -> dict[tuple[str, float, float], dict[str, Any]]:
    """Cache only benign/direct scores before selection; PIP is deferred."""

    cache: dict[tuple[str, float, float], dict[str, Any]] = {}
    for root, feature_path in sessions:
        for guard_us, window_us in grid:
            scores, _, summary = pipsession_scores(root, feature_path, guard_us, window_us)
            cache[(root.name, guard_us, window_us)] = {
                "scores": scores,
                "summary": summary,
            }
    return cache


def cache_adv_scores(
    roots: list[tuple[Path, int]],
    grid: list[tuple[float, float]],
) -> dict[tuple[float, float], dict[str, Any]]:
    cache: dict[tuple[float, float], dict[str, Any]] = {}
    for guard_us, window_us in grid:
        rows: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        for root, label in roots:
            extracted, summary = extract_run(
                root,
                label,
                guard_us=guard_us,
                window_us=window_us,
                noise_start_us=1000.0,
                noise_end_us=250.0,
                dedup_gap_samples=200,
            )
            rows.extend(row for row in extracted if int(row.get("collision_ambiguous", 0)) == 0)
            summaries.append(summary)
        cache[(guard_us, window_us)] = {"rows": rows, "summaries": summaries}
    return cache


def physical_validity(
    adv_cache: dict[tuple[float, float], dict[str, Any]],
    pip_cache: dict[tuple[str, float, float], dict[str, Any]],
    candidate: tuple[float, float],
    training_sessions: list[tuple[Path, Path]],
) -> tuple[bool, dict[str, Any]]:
    adv_summaries = adv_cache[candidate]["summaries"]
    adv_ok = all(
        int(summary.get("target_crc_valid_dedup", 0)) > 0
        and int(summary.get("eligible_packet_count", 0)) > 0
        and int(summary.get("ambiguous_packet_count", 0)) == 0
        and (
            int(summary.get("invalid_window_count", 0))
            / max(int(summary.get("target_crc_valid_dedup", 0)), 1)
            <= MAX_ADV_INVALID_WINDOW_FRACTION
        )
        for summary in adv_summaries
    )
    session_details: list[dict[str, Any]] = []
    session_ok = True
    for root, _ in training_sessions:
        summary = pip_cache[(root.name, *candidate)]["summary"]
        rows = int(summary.get("rows", 0))
        valid = int(summary.get("valid", 0))
        ambiguous = int(summary.get("ambiguous", 0))
        mapping_ok = valid == rows and rows > 0 and ambiguous == 0
        session_ok = session_ok and mapping_ok
        session_details.append(
            {
                "session_id": root.name,
                "rows": rows,
                "valid": valid,
                "ambiguous": ambiguous,
                "valid_fraction": (valid / rows) if rows else None,
                "physical_valid": mapping_ok,
            }
        )
    return adv_ok and session_ok, {
        "adv_physical_valid": adv_ok,
        "train_session_physical_valid": session_ok,
        "train_session_details": session_details,
    }


def base_candidate_rows(
    adv239_cache: dict[tuple[float, float], dict[str, Any]],
    adv8_cache: dict[tuple[float, float], dict[str, Any]],
    direct_benign_cache: dict[tuple[str, float, float], dict[str, Any]],
    adv239_roots: list[tuple[Path, int]],
    adv8_roots: list[tuple[Path, int]],
    train_sessions: list[tuple[Path, Path]],
    candidate: tuple[float, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    adv239_rows = adv239_cache[candidate]["rows"]
    adv8_rows = adv8_cache[candidate]["rows"]
    adv239_auc = metric(
        [float(row["tail_energy_db"]) for row in adv239_rows],
        [int(row["label"]) for row in adv239_rows],
    )
    adv8_auc = metric(
        [float(row["tail_energy_db"]) for row in adv8_rows],
        [int(row["label"]) for row in adv8_rows],
    )
    direct_scores: list[float] = []
    direct_labels: list[int] = []
    for root, _ in train_sessions:
        local = direct_benign_cache[(root.name, *candidate)]["scores"]
        direct_scores.extend(local)
        direct_labels.extend([1 if root.name.startswith("pip_auc_direct_tail_rep") else 0] * len(local))
    direct_auc = metric(direct_scores, direct_labels)
    row = {
        "candidate": candidate_key(candidate),
        "guard_us": candidate[0],
        "window_us": candidate[1],
        "role": "advertising_scale_reference" if candidate == REFERENCE else "redesign_candidate",
        "adv239_auc": adv239_auc,
        "adv8_auc": adv8_auc,
        "direct_train_auc": direct_auc,
        "adv239_benign_packets": sum(int(row["label"]) == 0 for row in adv239_rows),
        "adv239_positive_packets": sum(int(row["label"]) == 1 for row in adv239_rows),
        "adv8_benign_packets": sum(int(row["label"]) == 0 for row in adv8_rows),
        "adv8_positive_packets": sum(int(row["label"]) == 1 for row in adv8_rows),
        "direct_train_benign_packets": sum(label == 0 for label in direct_labels),
        "direct_train_positive_packets": sum(label == 1 for label in direct_labels),
        "adv239_threshold_pass": bool(adv239_auc is not None and adv239_auc >= ADV239_MIN_AUC),
        "adv8_threshold_pass": bool(adv8_auc is not None and adv8_auc >= ADV8_MIN_AUC),
        "direct_threshold_pass": bool(direct_auc is not None and direct_auc >= DIRECT_MIN_AUC),
        "adv239_reference_session_count": len(adv239_roots),
        "adv8_reference_session_count": len(adv8_roots),
    }
    return row, {"direct_scores": direct_scores, "direct_labels": direct_labels}


def select_fold(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    for row in rows:
        candidate = (float(row["guard_us"]), float(row["window_us"]))
        row["reference_excluded"] = candidate == REFERENCE
        row["physical_valid"] = bool(row["physical_valid"])
        row["base_pass"] = bool(
            candidate != REFERENCE
            and row["adv239_threshold_pass"]
            and row["adv8_threshold_pass"]
            and row["direct_threshold_pass"]
            and row["physical_valid"]
        )
        row["fallback_base_pass"] = bool(
            candidate != REFERENCE
            and row["adv239_threshold_pass"]
            and row["adv8_threshold_pass"]
            and row["physical_valid"]
        )

    for row in rows:
        candidate = (float(row["guard_us"]), float(row["window_us"]))
        neighbors = []
        if row["role"] == "redesign_candidate":
            same_guard = sorted(
                [
                    other
                    for other in rows
                    if float(other["guard_us"]) == candidate[0]
                    and other["role"] == "redesign_candidate"
                ],
                key=lambda item: float(item["window_us"]),
            )
            position = next(
                index
                for index, other in enumerate(same_guard)
                if str(other["candidate"]) == str(row["candidate"])
            )
            if position > 0:
                neighbors.append(same_guard[position - 1])
            if position + 1 < len(same_guard):
                neighbors.append(same_guard[position + 1])
        row["neighbor_candidates"] = ",".join(str(other["candidate"]) for other in neighbors)
        row["neighbor_count"] = len(neighbors)
        row["neighbor_base_pass_count"] = sum(bool(other["base_pass"]) for other in neighbors)
        row["neighbor_stable"] = bool(neighbors) and all(bool(other["base_pass"]) for other in neighbors)
        row["fallback_neighbor_stable"] = bool(neighbors) and all(bool(other["fallback_base_pass"]) for other in neighbors)
        row["selection_eligible"] = bool(row["base_pass"] and row["neighbor_stable"])

    eligible = [row for row in rows if row["selection_eligible"]]
    selection_status = "strict_threshold_selection"
    if not eligible:
        # Do not silently discard a fold.  This fallback is not a successful
        # threshold-gated selection; it only lets us compute a complete OOF
        # diagnostic.  It still excludes the advertising-scale reference,
        # requires physical validity and neighborhood stability, and never
        # reads PIP AUC.
        fallback = [
            row
            for row in rows
            if row["role"] == "redesign_candidate"
            and row["adv239_threshold_pass"]
            and row["adv8_threshold_pass"]
            and row["physical_valid"]
            and row["fallback_neighbor_stable"]
        ]
        if not fallback:
            raise RuntimeError("no physically valid stable redesigned candidate for diagnostic fallback")
        eligible = fallback
        selection_status = "fallback_no_candidate_met_all_thresholds"
    eligible.sort(
        key=lambda row: (
            -int(row["neighbor_stable"]),
            -int(row["neighbor_base_pass_count"]),
            -int(row["neighbor_count"]),
            float(row["guard_us"]),
            float(row["window_us"]),
        )
    )
    selected = eligible[0]
    for row in rows:
        row["selected"] = str(row["candidate"]) == str(selected["candidate"])
        row["selection_status"] = selection_status
    return selected, rows, selection_status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testdata", type=Path, default=Path("/path/to/PhantomChannel/testdata"))
    parser.add_argument("--detector-root", type=Path, default=Path("/path/to/PhantomChannel/experiments/figure/detector_roc_20260808"))
    parser.add_argument("--pip-root", type=Path, default=Path("/path/to/PhantomChannel/experiments/figure/pip_boundary_auc_20260808"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    grid = [REFERENCE] + SHORT_GRID
    adv239_roots, adv8_roots, _ = build_roots(args.testdata, args.detector_root, args.pip_root)
    sessions = load_sessions(args.pip_root)
    benign_direct = sessions["benign"] + sessions["direct_tail"]

    print(json.dumps({"stage": "cache_advertising", "candidate_count": len(grid)}, ensure_ascii=False))
    adv239_cache = cache_adv_scores(adv239_roots, grid)
    adv8_cache = cache_adv_scores(adv8_roots, grid)
    print(json.dumps({"stage": "cache_training_benign_direct", "session_count": len(benign_direct)}, ensure_ascii=False))
    pip_train_cache = cache_pip_scores(benign_direct, grid)

    fold_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    selected_candidates: list[str] = []
    pip_test_cache: dict[tuple[str, float, float], dict[str, Any]] = {}

    for fold_index in range(1, 6):
        train_sessions = [
            item
            for condition in ("benign", "direct_tail")
            for item_index, item in enumerate(sessions[condition], start=1)
            if item_index != fold_index
        ]
        fold_candidates: list[dict[str, Any]] = []
        for candidate in grid:
            row, _ = base_candidate_rows(
                adv239_cache,
                adv8_cache,
                pip_train_cache,
                adv239_roots,
                adv8_roots,
                train_sessions,
                candidate,
            )
            valid, validity = physical_validity(adv239_cache, pip_train_cache, candidate, train_sessions)
            row["fold"] = fold_index
            row["physical_valid"] = valid
            row["adv_physical_valid"] = validity["adv_physical_valid"]
            row["train_session_physical_valid"] = validity["train_session_physical_valid"]
            row["train_session_validity"] = json.dumps(validity["train_session_details"], separators=(",", ":"))
            fold_candidates.append(row)
        selected, fold_candidates, selection_status = select_fold(fold_candidates)
        selected_candidates.append(str(selected["candidate"]))
        parameter_rows.extend(fold_candidates)

        heldout_benign_root, heldout_benign_feature = sessions["benign"][fold_index - 1]
        heldout_direct_root, heldout_direct_feature = sessions["direct_tail"][fold_index - 1]
        heldout_pip_root, heldout_pip_feature = sessions["pip"][fold_index - 1]
        guard_us = float(selected["guard_us"])
        window_us = float(selected["window_us"])

        def get_scores(root: Path, feature_path: Path) -> dict[str, Any]:
            key = (root.name, guard_us, window_us)
            if key not in pip_train_cache and key not in pip_test_cache:
                scores, _, summary = pipsession_scores(root, feature_path, guard_us, window_us)
                pip_test_cache[key] = {"scores": scores, "summary": summary}
            return pip_train_cache.get(key, pip_test_cache.get(key))  # type: ignore[return-value]

        # Direct OOF view: held-out benign versus held-out direct-tail.
        direct_test_sessions = [
            ("benign", heldout_benign_root, heldout_benign_feature, 0),
            ("direct_tail", heldout_direct_root, heldout_direct_feature, 1),
        ]
        for condition, root, feature_path, label in direct_test_sessions:
            result = get_scores(root, feature_path)
            summary = result["summary"]
            for score in result["scores"]:
                oof_rows.append(
                    {
                        "fold": fold_index,
                        "view": "direct",
                        "session_id": root.name,
                        "condition": condition,
                        "label": label,
                        "score_db": score,
                        "guard_us": guard_us,
                        "window_us": window_us,
                        "rows": summary.get("rows"),
                        "valid": summary.get("valid"),
                        "ambiguous": summary.get("ambiguous"),
                    }
                )

        # PIP-outer OOF view: the same held-out benign session versus the
        # held-out PIP session.  The PIP session is scored only now, after the
        # fold selection above has completed.
        outer_test_sessions = [
            ("benign", heldout_benign_root, heldout_benign_feature, 0),
            ("pip", heldout_pip_root, heldout_pip_feature, 1),
        ]
        for condition, root, feature_path, label in outer_test_sessions:
            result = get_scores(root, feature_path)
            summary = result["summary"]
            for score in result["scores"]:
                oof_rows.append(
                    {
                        "fold": fold_index,
                        "view": "pip_outer",
                        "session_id": root.name,
                        "condition": condition,
                        "label": label,
                        "score_db": score,
                        "guard_us": guard_us,
                        "window_us": window_us,
                        "rows": summary.get("rows"),
                        "valid": summary.get("valid"),
                        "ambiguous": summary.get("ambiguous"),
                    }
                )

        fold_rows.append(
            {
                "fold": fold_index,
                "heldout_benign": heldout_benign_root.name,
                "heldout_direct_tail": heldout_direct_root.name,
                "heldout_pip": heldout_pip_root.name,
                "selected_guard_us": guard_us,
                "selected_window_us": window_us,
                "selected_candidate": selected["candidate"],
                "selected_direct_train_auc": selected["direct_train_auc"],
                "selected_adv239_auc": selected["adv239_auc"],
                "selected_adv8_auc": selected["adv8_auc"],
                "selected_neighbor_candidates": selected["neighbor_candidates"],
                "selected_neighbor_base_pass_count": selected["neighbor_base_pass_count"],
                "selected_neighbor_count": selected["neighbor_count"],
                "selected_neighbor_stable": selected["neighbor_stable"],
                "selected_fallback_neighbor_stable": selected["fallback_neighbor_stable"],
                "selection_status": selection_status,
                "strict_selection_gate_pass": selection_status == "strict_threshold_selection",
                "stable_candidate_count": sum(bool(row["selection_eligible"]) for row in fold_candidates),
                "base_passing_candidate_count": sum(bool(row["base_pass"]) for row in fold_candidates),
                "selection_uses_pip_auc": False,
            }
        )
        print(json.dumps({"stage": "fold_complete", **fold_rows[-1]}, ensure_ascii=False))

    direct_records = [row for row in oof_rows if row["view"] == "direct"]
    outer_records = [row for row in oof_rows if row["view"] == "pip_outer"]
    summary = {
        "schema_version": 1,
        "validation": "5-fold session-level internal validation",
        "fold_definition": "each fold holds out one benign, one direct-tail, and one PIP session; all other sessions train the selection",
        "selection_rule": {
            "uses_pip_auc": False,
            "reference_window": {"guard_us": REFERENCE[0], "window_us": REFERENCE[1], "role": "advertising_scale_reference_excluded_from_redesign_selection"},
            "adv239_auc_min": ADV239_MIN_AUC,
            "adv8_auc_min": ADV8_MIN_AUC,
            "direct_train_auc_min": DIRECT_MIN_AUC,
            "physical_validity": "advertising target windows have <=1% incomplete edge windows, no ambiguous eligible packets, and positive coverage; all training benign/direct session boundary windows are valid, mapped, and non-ambiguous",
            "neighbor_stability": "same-guard adjacent short-window candidates must pass; prefer the candidate with the largest stable neighborhood, then neutral guard/window ordering",
            "candidate_grid": [{"guard_us": g, "window_us": w} for g, w in grid],
        },
        "selection_stability": {
            "fold_count": 5,
            "selected_by_fold": selected_candidates,
            "frequency": {candidate: selected_candidates.count(candidate) for candidate in sorted(set(selected_candidates))},
            "all_folds_same": len(set(selected_candidates)) == 1,
            "strict_selection_gate_pass_by_fold": [bool(row["strict_selection_gate_pass"]) for row in fold_rows],
            "strict_selection_gate_pass_count": sum(bool(row["strict_selection_gate_pass"]) for row in fold_rows),
            "fallback_fold_count": sum(not bool(row["strict_selection_gate_pass"]) for row in fold_rows),
        },
        "oof": {
            "direct": cluster_bootstrap_auc(direct_records, BOOTSTRAP_SEED),
            "pip_outer": cluster_bootstrap_auc(outer_records, BOOTSTRAP_SEED + 1),
        },
        "data": {
            "benign_sessions": [root.name for root, _ in sessions["benign"]],
            "direct_tail_sessions": [root.name for root, _ in sessions["direct_tail"]],
            "pip_sessions": [root.name for root, _ in sessions["pip"]],
            "advertising_239_reference_sessions": [root.name for root, _ in adv239_roots],
            "advertising_8_reference_sessions": [root.name for root, _ in adv8_roots],
        },
        "interpretation": "A fold using the fallback means the predeclared Direct2B >= 0.95 gate was not met in that training split. OOF values are retained as a diagnostic, but parameter selection is not called stable and no independent-held-out claim is made.",
        "bootstrap": {"unit": "complete session cluster", "replicates_requested": BOOTSTRAP_REPLICATES, "seed_direct": BOOTSTRAP_SEED, "seed_pip_outer": BOOTSTRAP_SEED + 1},
    }

    write_csv(output / "validation_folds.csv", fold_rows)
    write_csv(output / "validation_parameter_grid.csv", parameter_rows)
    write_csv(output / "validation_oof_scores.csv", oof_rows)
    (output / "validation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "complete", "output_dir": str(output), "summary": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
