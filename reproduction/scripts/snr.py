"""Recompute the SNR/BER curves from the archived point-level table."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PDF_METADATA = {"Creator": "PhantomChannel artifact release", "CreationDate": None, "ModDate": None}


def run(root: Path) -> None:
    rows = []
    with (root / "data" / "snr_measurements.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    plotted = [r for r in rows if r["plotted"].lower() == "true"]
    groups = ["adv", "conn", "pip"]
    labels = {"adv": "Adv. state Embedding", "conn": "Conn. state Embedding", "pip": "PIP Embedding"}
    styles = {"adv": ("o-", 14), "conn": ("s--", 14), "pip": ("d:", 14)}
    x_src = sorted({int(r["snr_db"]) for r in plotted})
    x = np.arange(0, 21, dtype=float)

    out = root / "plots"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    plt.rcParams.update({"font.family": "Times New Roman"})
    for group in groups:
        points = {int(r["snr_db"]): float(r["ber"]) for r in plotted if r["group"] == group}
        y_src = np.array([points[v] for v in x_src], dtype=float)
        y = np.interp(x, np.array(x_src, dtype=float), y_src)
        style, size = styles[group]
        ax.plot(x, y, style, label=labels[group], markersize=size)
    ax.set_ylim(0, 0.5)
    ax.set_xlim(0, 20)
    ax.set_xticks(np.arange(0, 21, 1))
    ax.set_xlabel("SNR (dB)", fontsize=20)
    ax.set_ylabel("BER", fontsize=20)
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.tick_params(axis="both", direction="in")
    ax.legend(fontsize=20)
    fig.tight_layout()
    fig.savefig(out / "snrresult.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "snrresult.pdf", bbox_inches="tight", metadata=PDF_METADATA)
    plt.close(fig)

    with (root / "outputs" / "snr_interpolated.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["snr_db", *groups])
        curves = {}
        for group in groups:
            points = {int(r["snr_db"]): float(r["ber"]) for r in plotted if r["group"] == group}
            curves[group] = np.interp(x, np.array(x_src, dtype=float), np.array([points[v] for v in x_src]))
        for i, snr_db in enumerate(x.astype(int)):
            writer.writerow([snr_db, *(f"{curves[g][i]:.12f}" for g in groups)])
