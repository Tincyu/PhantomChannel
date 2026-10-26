"""Recompute frequency-domain similarity and the published spectrum figure."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from scipy.spatial.distance import cosine

FS = 4e6
NFFT = 1024
PDF_METADATA = {"Creator": "PhantomChannel artifact release", "CreationDate": None, "ModDate": None}


def normalized_psd(path: Path, nperseg: int) -> np.ndarray:
    iq = np.load(path, allow_pickle=False)
    _, pxx = welch(iq, fs=FS, nperseg=nperseg, nfft=NFFT, return_onesided=False)
    db = 10 * np.log10(pxx + 1e-12)
    span = db.max() - db.min()
    return (db - db.min()) / span if span else np.zeros_like(db)


def pair_similarity(a: Path, b: Path) -> float:
    nperseg = min(NFFT, len(np.load(a, mmap_mode="r", allow_pickle=False)), len(np.load(b, mmap_mode="r", allow_pickle=False)))
    return float(1 - cosine(normalized_psd(a, nperseg), normalized_psd(b, nperseg)))


def run(root: Path) -> dict[str, float]:
    data = root / "data" / "frequency"
    adv_pair = (data / "freq_adv/iq_packet_0022.npy", data / "freq_adv/iq_packet_0049.npy")
    data_pair = (data / "freq_data/iq_packet_0002.npy", data / "freq_data/iq_packet_0000.npy")
    pip_paths = [data / "freq_pip" / f"iq_packet_{i:04d}.npy" for i in range(10)]
    psds = [normalized_psd(p, NFFT) for p in pip_paths]
    matrix = np.eye(10)
    for i in range(10):
        for j in range(i + 1, 10):
            matrix[i, j] = matrix[j, i] = 1 - cosine(psds[i], psds[j])
    cross = [matrix[i, j] for i in range(6) for j in range(6, 10)]
    results = {
        "freq_adv": pair_similarity(*adv_pair),
        "freq_data": pair_similarity(*data_pair),
        "PIP_NORMAL_cross_mean": float(np.mean(cross)),
    }
    with (root / "outputs" / "frequency_similarity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["metric", "value"])
        writer.writerows([[k, f"{v:.12f}"] for k, v in results.items()])
    with (root / "outputs" / "frequency_pip_pairwise.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["file", *[f"{i:04d}" for i in range(10)]])
        for i, row in enumerate(matrix): writer.writerow([f"{i:04d}", *[f"{v:.12f}" for v in row]])

    def estimate_peak(iq: np.ndarray) -> float:
        spectrum = np.fft.fftshift(np.fft.fft(iq)); freqs = np.fft.fftshift(np.fft.fftfreq(len(iq), 1 / FS))
        return float(freqs[np.argmax(np.abs(spectrum))])
    def load_spectrum(path: Path, shift: float) -> tuple[np.ndarray, np.ndarray]:
        iq = np.load(path, allow_pickle=False); offset = estimate_peak(iq)
        corrected = iq * np.exp(-1j * 2 * np.pi * offset * np.arange(len(iq)) / FS)
        spectrum = np.fft.fftshift(np.fft.fft(corrected)); freqs = np.fft.fftshift(np.fft.fftfreq(len(iq), 1 / FS))
        return freqs + shift, 20 * np.log10(np.abs(spectrum) + 1e-12) - 90
    panels = [
        ("Frequency spectrum: Adv", data / "freq_adv/iq_packet_0022.npy", data / "freq_adv/iq_packet_0049.npy", 0, 300000, "Basic connection channel embedding", "Packet with no covert data"),
        ("Frequency spectrum: Data", data / "freq_data/iq_packet_0002.npy", data / "freq_data/iq_packet_0000.npy", 333333, 333333, "Basic connection channel embedding", "Packet with no covert data"),
        ("Frequency spectrum: PIP", data / "freq_data/iq_packet_0002.npy", data / "freq_data/iq_packet_0000.npy", 333333, 333333, "PIP embedding", "Packet with no covert data"),
    ]
    plt.rcParams.update({"font.family": "Times New Roman"})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    for ax, (title, f1, f2, s1, s2, l1, l2) in zip(axes, panels):
        x1, y1 = load_spectrum(f1, s1); x2, y2 = load_spectrum(f2, s2)
        ax.plot(x1 / 1e6, y1, label=l1, color="blue", linewidth=1.0); ax.plot(x2 / 1e6, y2, label=l2, color="orange", linewidth=1.0)
        ax.set_title(title, fontsize=13); ax.set_xlim(-1.4, 1.4); ax.set_ylim(-105, -40); ax.set_xticks([-1, 0, 1]); ax.set_yticks(np.arange(-100, -39, 20)); ax.tick_params(axis="both", labelsize=10, direction="in"); ax.grid(True, linestyle="--", alpha=0.25); ax.legend(loc="upper center", fontsize=8, framealpha=0.85); ax.set_xlabel("Frequency (MHz)", fontsize=11)
    axes[0].set_ylabel("Magnitude (dB)", fontsize=11); fig.subplots_adjust(left=0.06, right=0.995, bottom=0.15, top=0.88, wspace=0.08)
    out = root / "plots"; out.mkdir(parents=True, exist_ok=True); fig.savefig(out / "freq_nocte.png", dpi=300, bbox_inches="tight"); fig.savefig(out / "freq_nocte.pdf", bbox_inches="tight", metadata=PDF_METADATA); plt.close(fig)
    return results
