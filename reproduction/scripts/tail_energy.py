#!/usr/bin/env python3
"""Parse and reproduce the archived cross-layer tail-energy results."""

from __future__ import annotations

import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rank_auc(positive: list[float], negative: list[float]) -> float:
    if not positive or not negative:
        raise ValueError("AUC requires both positive and negative samples")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def tpr_at_fpr(positive: list[float], negative: list[float], target: float = 0.05) -> tuple[float, float, float]:
    best = (0.0, 0.0, math.inf)
    for threshold in sorted(set(positive + negative), reverse=True):
        tp = sum(score >= threshold for score in positive)
        fp = sum(score >= threshold for score in negative)
        fpr = fp / len(negative)
        tpr = tp / len(positive)
        if fpr <= target and (tpr > best[0] or (tpr == best[0] and fpr > best[1])):
            best = (tpr, fpr, threshold)
    return best


def formal_roc(root: Path, output: Path) -> dict[str, object]:
    feature_path = root / "data/tail_energy/advertising/formal_main/boundary_packet_features.csv"
    report_path = root / "data/tail_energy/advertising/formal_main/boundary_roc_report.json"
    features = read_csv(feature_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    grouped: dict[float, list[tuple[float, int]]] = defaultdict(list)
    for row in features:
        if not (
            int(row["standard_crc_valid"])
            and int(row["iq_window_complete"])
            and not int(row["collision_ambiguous"])
        ):
            continue
        grouped[float(row["gain_db"])].append((float(row["tail_energy_db"]), int(row["label"])))

    report_by_gain = {float(row["gain_db"]): row for row in report["results"]}
    rows: list[dict[str, object]] = []
    for gain in sorted(grouped):
        samples = grouped[gain]
        positive = [score for score, label in samples if label == 1]
        negative = [score for score, label in samples if label == 0]
        auc = rank_auc(positive, negative)
        tpr, fpr, threshold = tpr_at_fpr(positive, negative)
        expected = report_by_gain[gain]
        if abs(auc - float(expected["auc"])) > 1e-12:
            raise AssertionError(f"tail-energy AUC mismatch at gain {gain}: {auc} != {expected['auc']}")
        if abs(tpr - float(expected["tpr_at_fpr_05"])) > 1e-12:
            raise AssertionError(f"tail-energy TPR mismatch at gain {gain}: {tpr} != {expected['tpr_at_fpr_05']}")
        rows.append(
            {
                "gain_db": gain,
                "eligible_packets": len(samples),
                "benign_packets": len(negative),
                "covert_packets": len(positive),
                "computed_auc": auc,
                "reported_auc": expected["auc"],
                "computed_tpr_at_fpr_05": tpr,
                "reported_tpr_at_fpr_05": expected["tpr_at_fpr_05"],
                "operating_fpr": fpr,
                "operating_threshold_db": threshold,
                "reported_tpr_theta_5": expected["tpr_theta_5"],
            }
        )
    write_csv(
        output / "advertising_formal_roc.csv",
        [
            "gain_db",
            "eligible_packets",
            "benign_packets",
            "covert_packets",
            "computed_auc",
            "reported_auc",
            "computed_tpr_at_fpr_05",
            "reported_tpr_at_fpr_05",
            "operating_fpr",
            "operating_threshold_db",
            "reported_tpr_theta_5",
        ],
        rows,
    )
    return {"feature_rows": len(features), "output_rows": len(rows), "path": str((output / "advertising_formal_roc.csv").relative_to(root))}


def advertising_sensitivity(root: Path, output: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    captures: list[dict[str, object]] = []
    for length in (8, 16, 64, 128):
        base = root / f"data/tail_energy/advertising/tail_sensitivity/len{length}"
        report = json.loads((base / "boundary_roc_report.json").read_text(encoding="utf-8"))
        for item in report["results"]:
            rows.append(
                {
                    "tail_length_bytes": length,
                    "gain_db": item["gain_db"],
                    "eligible_packets": item["eligible_packet_count"],
                    "benign_packets": item["benign_packet_count"],
                    "covert_packets": item["covert_packet_count"],
                    "auc": item["auc"],
                    "tpr_at_fpr_05": item["tpr_at_fpr_05"],
                    "fpr_theta_5": item["fpr_theta_5"],
                    "tpr_theta_5": item["tpr_theta_5"],
                }
            )
        for capture in read_csv(base / "boundary_capture_summary.csv"):
            captures.append({"tail_length_bytes": length, **capture})
    write_csv(
        output / "advertising_tail_sensitivity.csv",
        [
            "tail_length_bytes",
            "gain_db",
            "eligible_packets",
            "benign_packets",
            "covert_packets",
            "auc",
            "tpr_at_fpr_05",
            "fpr_theta_5",
            "tpr_theta_5",
        ],
        rows,
    )
    capture_fields = ["tail_length_bytes"] + [key for key in captures[0] if key != "tail_length_bytes"]
    write_csv(output / "advertising_tail_capture_summary.csv", capture_fields, captures)
    return {"tail_lengths": [8, 16, 64, 128], "roc_rows": len(rows), "capture_rows": len(captures), "path": str((output / "advertising_tail_sensitivity.csv").relative_to(root))}


def connected_auc(root: Path, output: Path) -> dict[str, object]:
    source = root / "data/tail_energy/connected/final/pip_boundary_auc.csv"
    rows = read_csv(source)
    if {"metric", "auc", "tpr_at_fpr_0.05"} - set(rows[0]):
        raise ValueError("connected tail-energy result is missing required columns")
    write_csv(output / "connected_pip_auc.csv", list(rows[0]), rows)
    return {"metric_rows": len(rows), "path": str((output / "connected_pip_auc.csv").relative_to(root))}


def connected_session_oof(root: Path, output: Path) -> dict[str, object]:
    base = root / "data/tail_energy/connected/session_level_internal_validation"
    rows = read_csv(base / "validation_oof_scores.csv")
    expected = json.loads((base / "validation_summary.json").read_text(encoding="utf-8"))["oof"]
    result: list[dict[str, object]] = []
    for view in ("direct", "pip_outer"):
        samples = [
            (float(row["score_db"]), int(row["label"]))
            for row in rows
            if row["view"] == view and int(row["valid"]) and not int(row["ambiguous"])
        ]
        positive = [score for score, label in samples if label == 1]
        negative = [score for score, label in samples if label == 0]
        auc = rank_auc(positive, negative)
        tpr, fpr, threshold = tpr_at_fpr(positive, negative)
        if abs(auc - float(expected[view]["auc"])) > 1e-12:
            raise AssertionError(f"connected OOF AUC mismatch for {view}: {auc} != {expected[view]['auc']}")
        if abs(tpr - float(expected[view]["tpr_at_fpr_05"])) > 1e-12:
            raise AssertionError(f"connected OOF TPR mismatch for {view}: {tpr} != {expected[view]['tpr_at_fpr_05']}")
        result.append(
            {
                "view": view,
                "valid_samples": len(samples),
                "benign_samples": len(negative),
                "positive_samples": len(positive),
                "computed_auc": auc,
                "reported_auc": expected[view]["auc"],
                "computed_tpr_at_fpr_05": tpr,
                "reported_tpr_at_fpr_05": expected[view]["tpr_at_fpr_05"],
                "operating_fpr": fpr,
                "operating_threshold_db": threshold,
            }
        )
    write_csv(
        output / "connected_session_oof.csv",
        [
            "view",
            "valid_samples",
            "benign_samples",
            "positive_samples",
            "computed_auc",
            "reported_auc",
            "computed_tpr_at_fpr_05",
            "reported_tpr_at_fpr_05",
            "operating_fpr",
            "operating_threshold_db",
        ],
        result,
    )
    return {"sample_rows": len(rows), "metric_rows": len(result), "path": str((output / "connected_session_oof.csv").relative_to(root))}


def copy_plots(root: Path, output: Path) -> list[str]:
    destination = output / "plots"
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for source in sorted((root / "data/tail_energy/plots").iterdir()):
        target = destination / source.name
        shutil.copy2(source, target)
        signature = target.read_bytes()[:8]
        if source.suffix == ".pdf" and signature[:5] != b"%PDF-":
            raise AssertionError(f"invalid tail-energy PDF: {target}")
        if source.suffix == ".png" and signature != b"\x89PNG\r\n\x1a\n":
            raise AssertionError(f"invalid tail-energy PNG: {target}")
        copied.append(str(target.relative_to(root)))
    return copied


def run(root: Path) -> dict[str, object]:
    output = root / "outputs/tail_energy"
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "advertising_formal": formal_roc(root, output),
        "advertising_sensitivity": advertising_sensitivity(root, output),
        "connected": connected_auc(root, output),
        "connected_session_oof": connected_session_oof(root, output),
        "plots": copy_plots(root, output),
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    print(json.dumps(run(root), indent=2))
