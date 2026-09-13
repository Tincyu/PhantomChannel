#!/usr/bin/env python3
"""Plot the three held-out PIP boundary ROC views from packet features."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.pip_boundary_auc_experiment import numeric_score, read_csv


def roc_points(rows: list[dict[str, Any]], field: str, positive: str) -> tuple[list[dict[str, float]], float | None]:
    usable = []
    for row in rows:
        if row.get("condition") not in {"benign", positive}:
            continue
        score = numeric_score(row, field)
        if score is not None:
            usable.append((score, int(row["condition"] == positive)))
    positives = np.asarray([score for score, label in usable if label], dtype=np.float64)
    negatives = np.asarray([score for score, label in usable if not label], dtype=np.float64)
    if not positives.size or not negatives.size:
        return [], None
    thresholds = sorted(set(float(value) for value, _ in usable), reverse=True)
    points = [{"fpr": 0.0, "tpr": 0.0, "threshold": float("inf")}]
    for threshold in thresholds:
        points.append(
            {
                "fpr": float(np.mean(negatives >= threshold)),
                "tpr": float(np.mean(positives >= threshold)),
                "threshold": threshold,
            }
        )
    points.sort(key=lambda item: (item["fpr"], item["tpr"]))
    pair_score = sum(pos > neg for pos in positives for neg in negatives)
    pair_score += 0.5 * sum(pos == neg for pos in positives for neg in negatives)
    auc = float(pair_score / (positives.size * negatives.size))
    return points, auc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-csv", action="append", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in args.features_csv:
        rows.extend(read_csv(path))
    definitions = {
        "AUC_direct": ("E_tail_real_db", "direct_tail"),
        "AUC_pip_outer": ("E_tail_outer_db", "pip"),
        "AUC_pip_real": ("E_tail_real_db", "pip"),
    }
    curves: dict[str, Any] = {}
    output_rows: list[dict[str, Any]] = []
    metrics = json.loads(args.metrics_json.read_text(encoding="utf-8"))["metrics"]
    for name, (field, positive) in definitions.items():
        points, auc = roc_points(rows, field, positive)
        curves[name] = {"score_field": field, "positive_condition": positive, "auc": auc, "points": points}
        for point in points:
            output_rows.append({"metric": name, **point})

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "pip_boundary_auc_roc.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "fpr", "tpr", "threshold"])
        writer.writeheader()
        writer.writerows(output_rows)

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    colors = {"AUC_direct": "#1f77b4", "AUC_pip_outer": "#d62728", "AUC_pip_real": "#2ca02c"}
    for name, curve in curves.items():
        points = curve["points"]
        if not points:
            continue
        auc = float(curve["auc"])
        ci = metrics[name].get("auc_ci95")
        ci_text = f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
        axis.plot(
            [point["fpr"] for point in points],
            [point["tpr"] for point in points],
            linewidth=1.5,
            label=f"{name}={auc:.3f}{ci_text}",
            color=colors[name],
        )
    axis.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="False-positive rate", ylabel="True-positive rate")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=7, loc="lower right")
    axis.set_title("Matched 2M PIP boundary-evasion ROC")
    figure.tight_layout()
    figure.savefig(output / "pip_boundary_auc.png", dpi=220)
    figure.savefig(output / "pip_boundary_auc.pdf")
    plt.close(figure)
    (output / "pip_boundary_auc_plot.json").write_text(
        json.dumps({"curves": curves, "source_metrics": str(args.metrics_json)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output), "curves": list(curves)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
