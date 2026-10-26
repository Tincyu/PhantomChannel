"""Recompute the CRC/whitening overhead curves from measurement inputs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PDF_METADATA = {"Creator": "PhantomChannel artifact release", "CreationDate": None, "ModDate": None}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run(root: Path) -> dict[str, float]:
    rows = _read_rows(root / "data" / "overhead" / "measurements.csv")
    nrf = [r for r in rows if r["platform"] == "nRF52840"]
    da = [r for r in rows if r["platform"] == "DA14695"]

    x_nrf = np.array([float(r["payload_length_B"]) for r in nrf])
    crc_nrf = np.array([float(r["crc_us"]) for r in nrf])
    whiten_nrf = np.array([float(r["whitening_us"]) for r in nrf])
    x = np.arange(1, 31, dtype=float)
    crc_fit = np.polyval(np.polyfit(x_nrf, crc_nrf, 1), x)
    crc_k, crc_b = np.polyfit(x_nrf, crc_nrf, 1)
    whiten_k, whiten_b = np.polyfit(x_nrf, whiten_nrf, 1)
    basic_nrf = float(nrf[0]["basic_us"])
    whiten_effective = whiten_k * (2 * x + 6) + whiten_b
    nrf_conn = basic_nrf + crc_fit
    nrf_pip = nrf_conn + whiten_effective

    da_x = np.array([float(r["payload_length_B"]) for r in da])
    da_crc = np.array([float(r["crc_us"]) for r in da])
    da_whiten = np.array([float(r["whitening_us"]) for r in da])
    da_basic = float(da[0]["basic_us"])
    da_conn = da_basic + da_crc
    da_pip = da_conn + da_whiten

    out = root / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Times New Roman"})
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_ylim(0, 150)
    ax.plot(x, nrf_conn, marker="o", label="nRF52840 Conn.", markersize=4)
    ax.plot(x, nrf_pip, marker="d", label="nRF52840 PIP", markersize=4)
    ax.plot(da_x, da_conn, marker="^", label="da14695 Conn.", markersize=4)
    ax.plot(da_x, da_pip, marker="s", label="da14695 PIP", markersize=4)
    ax.set_xlabel("Size payload (Bytes)", fontsize=20)
    ax.set_ylabel("Time overhead (us)", fontsize=20)
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.tick_params(axis="both", direction="in")
    ax.legend(loc="upper left", fontsize=20)
    fig.tight_layout()
    fig.savefig(out / "overhead.png", dpi=300)
    fig.savefig(out / "overhead.pdf", metadata=PDF_METADATA)
    plt.close(fig)

    with (root / "outputs" / "overhead_fit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["payload_length_B", "nrf_conn_us", "nrf_pip_us", "da_conn_us", "da_pip_us"])
        for i, length in enumerate(x.astype(int)):
            da_i = length - 1
            writer.writerow([length, f"{nrf_conn[i]:.12f}", f"{nrf_pip[i]:.12f}", f"{da_conn[da_i]:.12f}", f"{da_pip[da_i]:.12f}"])
    return {"nrf_crc_slope": float(crc_k), "nrf_crc_intercept": float(crc_b), "nrf_whitening_slope": float(whiten_k)}
