#!/usr/bin/env python3
"""Draw the eight independent candidate figures from revision_figure_design.md.

The script deliberately consumes only the frozen result files named by the
design document.  It validates the document's sanity-check values before
writing any PDF and records input hashes and plotted event selections.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties, findfont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "revision_figures"
DEFAULT_PAPER_PDF_DIR = Path("/path/to/work/paper/PhantomChannel_ccs/images")
DESIGN = ROOT / "docs" / "revision_figure_design.md"
REDESIGN = ROOT / "docs" / "detector_roc_experiment_redesign.md"

PSSD = Path("/path/to/PhantomChannel/experiments")
FIG = PSSD / "figure"
BOUNDARY = FIG / "detector_roc_20260808"
TIMING = FIG / "detector_roc_20260810" / "timing_formal"
BEHAVIOR = PSSD / "pip_cis_behavior" / "behavior_timeline_20260809"

INPUTS = {
    "design_document": DESIGN,
    "active_experiment_design": REDESIGN,
    "bandwidth": FIG / "phonehrs_los_d3m_bandwidth_summary.csv",
    "boundary_packet_features": BOUNDARY / "boundary_formal_main" / "boundary_packet_features.csv",
    "boundary_roc_by_gain": BOUNDARY / "boundary_formal_main" / "analysis" / "boundary_roc_by_gain.csv",
    "boundary_roc_report": BOUNDARY / "boundary_formal_main" / "analysis" / "boundary_roc_report.json",
    "boundary_detector_summary": BOUNDARY / "boundary_detector_summary.csv",
    "boundary_coverage": BOUNDARY / "boundary_coverage_by_condition.csv",
    "boundary_snr": BOUNDARY / "boundary_snr_by_gain.csv",
    "pip_validation_summary": FIG / "pip_boundary_auc_20260808" / "results_final" / "session_level_internal_validation" / "validation_summary.json",
    "pip_validation_oof": FIG / "pip_boundary_auc_20260808" / "results_final" / "session_level_internal_validation" / "validation_oof_scores.csv",
    "advertising_timing_auc": TIMING / "environmental_benign" / "environmental_diagnostic_auc.csv",
    "event_interval_auc": TIMING / "advertising_gap_audit" / "audited_timing_roc.csv",
    "central_tifs_auc": TIMING / "connected_timing" / "central-side-8B_tifs_auc_summary.csv",
    "central_tifs_sessions": TIMING / "connected_timing" / "central-side-8B_tifs_session_summary.csv",
    "peripheral_tifs_auc": TIMING / "connected_timing" / "peripheral-side-240B_tifs_auc_summary.csv",
    "peripheral_tifs_sessions": TIMING / "connected_timing" / "peripheral-side-240B_tifs_session_summary.csv",
    "pip_tifs_auc": TIMING / "pip_timing" / "pip_timing_auc_summary.csv",
    "pip_tifs_sessions": TIMING / "pip_timing" / "pip_timing_session_summary.csv",
    "behavior_events": BEHAVIOR / "pip_cis_behavior_events.csv",
    "behavior_table": BEHAVIOR / "pip_cis_behavior_table.csv",
    "behavior_manifest": BEHAVIOR / "pip_cis_behavior_manifest.json",
    "cis_trace_summary": PSSD / "pip_cis_behavior" / "20260809T_cis_x310_complete_r3" / "cis_trace_summary.json",
}

COLORS = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "amber": "#E69F00",
    "gray": "#555555",
    "lightgray": "#BDBDBD",
    "black": "#111111",
}


def configure_matplotlib() -> str:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.15,
            "lines.markersize": 4.2,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # Keep SVG labels as <text> elements so they remain editable in
            # Inkscape/Illustrator instead of converting every glyph to paths.
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    return findfont(FontProperties(family="serif"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_inputs() -> None:
    missing = [f"{key}: {path}" for key, path in INPUTS.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen input(s):\n" + "\n".join(missing))


def assert_close(name: str, actual, expected, atol=1e-6) -> None:
    if not np.allclose(np.asarray(actual), np.asarray(expected), atol=atol, rtol=0):
        raise ValueError(f"Sanity-check mismatch for {name}: actual={actual!r}, expected={expected!r}")


def parse_interval(value: str | list[float]) -> tuple[float, float]:
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    return float(parsed[0]), float(parsed[1])


def mean_rank_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positive = labels == 1
    negative = labels == 0
    if not positive.any() or not negative.any():
        raise ValueError("AUC requires both positive and negative observations")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    p = positive.sum()
    n = negative.sum()
    return float((ranks[positive].sum() - p * (p + 1) / 2.0) / (p * n))


def roc_points(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positive_count = float((labels == 1).sum())
    negative_count = float((labels == 0).sum())
    thresholds = np.unique(scores)[::-1]
    fpr = [0.0]
    tpr = [0.0]
    for threshold in thresholds:
        selected = scores >= threshold
        fpr.append(float(np.sum(selected & (labels == 0)) / negative_count))
        tpr.append(float(np.sum(selected & (labels == 1)) / positive_count))
    return np.asarray(fpr), np.asarray(tpr)


def style_axis(ax, grid=True) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", pad=2)
    if grid:
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.45)


def panel_tag(ax, text: str) -> None:
    ax.text(0.0, 1.04, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5)


def save_figure(fig, filename: str, overwrite: bool) -> Path:
    path = OUT / filename
    svg_path = path.with_suffix(".svg")
    for output_path in (path, svg_path):
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists; use --overwrite to replace: {output_path}")
    metadata = {"Creator": "draw_revision_figures.py"}
    fig.savefig(path, format="pdf", metadata=metadata)
    fig.savefig(svg_path, format="svg", metadata=metadata)
    plt.close(fig)
    return path


def sync_pdf_outputs(pdf_outputs: list[Path], destination: Path) -> list[Path]:
    """Copy only generated PDFs to the paper image directory.

    SVGs remain project-local under ``artifacts/revision_figures`` and are
    intentionally not copied to the paper repository.
    """
    destination.mkdir(parents=True, exist_ok=True)
    synced = []
    for source in pdf_outputs:
        if source.suffix.lower() != ".pdf":
            raise ValueError(f"PDF sync received a non-PDF output: {source}")
        target = destination / source.name
        shutil.copy2(source, target)
        synced.append(target)
    return synced


def validate_data() -> dict:
    bandwidth = pd.read_csv(INPUTS["bandwidth"])
    assert_close("bandwidth coverage (%)", bandwidth["C_bw_mean"].to_numpy() * 100, [2.70, 6.36, 12.74, 29.71, 67.53, 100.00], atol=0.03)
    assert_close("bandwidth goodput (kbps)", bandwidth["G_e2e_kbps_mean"], [2.49, 5.86, 11.73, 27.34, 62.11, 91.98], atol=0.03)

    roc = pd.read_csv(INPUTS["boundary_roc_by_gain"])
    roc = roc.sort_values("gain_db")
    assert_close("boundary AUC by gain", roc["auc"], [0.9994, 0.9967, 0.9936], atol=0.00005)
    assert_close("boundary TPR@5% FPR by gain", roc["tpr_at_fpr_05"], [1.0, 0.9930, 0.9903], atol=0.00005)

    detector = pd.read_csv(INPUTS["boundary_detector_summary"])
    selected = detector[(detector["gain_db"] == 35.0) & detector["tail_length_bytes"].isin([8, 16, 64, 128])].sort_values("tail_length_bytes")
    main_239 = detector[(detector["experiment"] == "main_roc") & (detector["gain_db"] == 35.0) & (detector["tail_length_bytes"] == 239)]
    tail_auc = list(selected["auc"]) + [float(main_239.iloc[0]["auc"])]
    assert_close("tail sensitivity AUC", tail_auc, [0.9985, 0.9932, 0.9987, 0.9979, 0.9967], atol=0.00005)

    coverage = pd.read_csv(INPUTS["boundary_coverage"])
    coverage = coverage.sort_values(["gain_db", "condition"])
    for condition, expected in [("benign", [0.8772, 0.9880, 0.9916]), ("covert", [0.9840, 0.9728, 0.9932])]:
        values = coverage[coverage["condition"] == condition].sort_values("gain_db")["parser_coverage_mean"]
        assert_close(f"{condition} parser coverage", values, expected, atol=1e-6)

    snr = pd.read_csv(INPUTS["boundary_snr"]).sort_values("gain_db")
    assert_close("session-balanced legal SNR median", snr["snr_db_session_balanced_median"], [35.20, 43.86, 49.45], atol=0.02)

    validation = json.loads(INPUTS["pip_validation_summary"].read_text())
    assert_close("PIP direct OOF AUC", validation["oof"]["direct"]["auc"], 0.9627, atol=0.00005)
    assert_close("PIP outer OOF AUC", validation["oof"]["pip_outer"]["auc"], 0.5200, atol=0.00005)
    if validation["selection_stability"]["selected_by_fold"] != ["16/6"] * 5:
        raise ValueError("PIP validation selected window is not 16/6 in all five folds")

    advertising = pd.read_csv(INPUTS["advertising_timing_auc"])
    gap = advertising[
        (advertising["analysis_scope"] == "device-balanced")
        & (advertising["weighting"] == "device-balanced")
        & (advertising["family"] == "all")
        & (advertising["feature"] == "gap_score")
        & advertising["positive_condition"].isin(["append-last-239B", "append-every-239B"])
    ].set_index("positive_condition")["auc"]
    assert_close("advertising gap AUC", [gap["append-last-239B"], gap["append-every-239B"]], [0.4010, 0.9985], atol=0.00005)

    event_interval = pd.read_csv(INPUTS["event_interval_auc"])
    event_interval = event_interval[(event_interval["crc_policy"] == "all-target-rows") & (event_interval["feature"] == "event_score")].set_index("positive_condition")["auc"]
    assert_close("advertising event-interval AUC", [event_interval["append-last-239B"], event_interval["append-every-239B"]], [0.4252, 0.4293], atol=0.00005)

    behavior_manifest = json.loads(INPUTS["behavior_manifest"].read_text())
    if behavior_manifest["event_counts"] != {"ACL": 2288, "CIS-partial-context": 2, "PIP": 1223}:
        raise ValueError(f"Behavior event-count mismatch: {behavior_manifest['event_counts']}")
    cis_summary = json.loads(INPUTS["cis_trace_summary"].read_text())
    if cis_summary["uart"]["central"]["cis_connected_count"] != 1 or cis_summary["uart"]["peripheral"]["cis_connected_count"] != 1:
        raise ValueError("CIS controller ground-truth lane is not supported by both UART traces")
    if cis_summary["pcap"]["ll_cis_req_count"] != 1 or cis_summary["pcap"]["ll_cis_ind_count"] != 1 or cis_summary["pcap"]["ll_cis_rsp_count"] != 0:
        raise ValueError("CIS passive control-event counts do not match the required partial-context evidence")
    return {"validation_summary": validation, "behavior_manifest": behavior_manifest, "cis_summary": cis_summary}


def draw_bandwidth(bandwidth: pd.DataFrame, overwrite: bool) -> Path:
    x = np.arange(len(bandwidth))
    labels = [f"{int(v)}" for v in bandwidth["bandwidth_mhz"]]
    fig = plt.figure(figsize=(7.0, 2.7))
    ax1 = fig.add_axes([0.085, 0.25, 0.40, 0.64])
    ax2 = fig.add_axes([0.575, 0.25, 0.36, 0.64])
    for ax in (ax1, ax2):
        style_axis(ax)
        ax.set_xticks(x, labels)
        ax.set_xlabel("Receiver bandwidth (MHz)")
    ax1.errorbar(x, bandwidth["C_bw_mean"] * 100, yerr=bandwidth["C_bw_std"] * 100, color=COLORS["blue"], marker="o", mfc="white", capsize=2.2)
    ax1.set_ylabel("In-band opportunity coverage (%)")
    ax1.set_ylim(0, 110)
    ax1.set_yticks([0, 25, 50, 75, 100])
    ax2.errorbar(x, bandwidth["G_e2e_kbps_mean"], yerr=bandwidth["G_e2e_kbps_std"], color=COLORS["orange"], marker="s", mfc="white", capsize=2.2)
    ax2.set_ylabel("End-to-end goodput (kbps)")
    ax2.set_ylim(0, 105)
    ax2.set_yticks([0, 25, 50, 75, 100])
    panel_tag(ax1, "(a)")
    panel_tag(ax2, "(b)")
    return save_figure(fig, "revision_bandwidth_goodput.pdf", overwrite)


def draw_boundary_roc(features: pd.DataFrame, roc_summary: pd.DataFrame, overwrite: bool) -> Path:
    fig = plt.figure(figsize=(3.35, 2.8))
    ax = fig.add_axes([0.18, 0.18, 0.78, 0.73])
    style_axis(ax)
    ax.plot([0, 1], [0, 1], color=COLORS["lightgray"], linestyle=(0, (3, 2)), linewidth=0.8, label="Chance")
    specs = [(30.0, COLORS["blue"], "-"), (35.0, COLORS["orange"], "--"), (50.0, COLORS["green"], "-.")]
    for gain, color, linestyle in specs:
        subset = features[features["gain_db"] == gain]
        labels = subset["label"].to_numpy()
        scores = subset["tail_energy_db"].to_numpy()
        fpr, tpr = roc_points(labels, scores)
        row = roc_summary[roc_summary["gain_db"] == gain].iloc[0]
        ax.step(fpr, tpr, where="post", color=color, linestyle=linestyle, linewidth=1.15, label=f"{int(gain)} dB gain, AUC {row['auc']:.3f}")
        ax.plot(0.05, row["tpr_at_fpr_05"], marker="o", markersize=3.2, color=color, markeredgecolor="white", markeredgewidth=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(loc="lower right", frameon=False, handlelength=2.2, borderpad=0.2, labelspacing=0.3)
    return save_figure(fig, "revision_boundary_roc_by_gain.pdf", overwrite)


def draw_tail_sensitivity(detector: pd.DataFrame, overwrite: bool) -> Path:
    rows = []
    for length in [8, 16, 64, 128]:
        rows.append(detector[(detector["experiment"] == "tail_sensitivity") & (detector["gain_db"] == 35.0) & (detector["tail_length_bytes"] == length)].iloc[0])
    rows.append(detector[(detector["experiment"] == "main_roc") & (detector["gain_db"] == 35.0) & (detector["tail_length_bytes"] == 239)].iloc[0])
    data = pd.DataFrame(rows)
    x = data["tail_length_bytes"].to_numpy(float)
    fig = plt.figure(figsize=(3.35, 2.5))
    ax1 = fig.add_axes([0.16, 0.22, 0.35, 0.66])
    ax2 = fig.add_axes([0.60, 0.22, 0.35, 0.66])
    for ax in (ax1, ax2):
        style_axis(ax)
        ax.set_xlabel("Tail length (B; log scale)")
        ax.set_xscale("log")
        ax.set_xlim(7, 270)
        ax.set_xticks(x, [str(int(v)) for v in x])
        ax.set_ylim(0.98, 1.001)
        ax.set_yticks([0.98, 0.99, 1.00])
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
    auc_low, auc_high = zip(*(parse_interval(v) for v in data["auc_ci95"]))
    tpr_low, tpr_high = zip(*(parse_interval(v) for v in data["tpr_at_fpr_05_ci95"]))
    for ax, values, low, high, ylabel in [
        (ax1, data["auc"].to_numpy(float), np.asarray(auc_low), np.asarray(auc_high), "ROC-AUC"),
        (ax2, data["tpr_at_fpr_05"].to_numpy(float), np.asarray(tpr_low), np.asarray(tpr_high), "TPR at 5% FPR"),
    ]:
        ax.fill_between(x, low, high, color=COLORS["blue"], alpha=0.16, linewidth=0)
        ax.plot(x, values, color=COLORS["blue"], marker="o", mfc="white", linewidth=1.0)
        ax.set_ylabel(ylabel)
    ax1.text(0.02, 0.03, "95% CI", transform=ax1.transAxes, fontsize=7, color=COLORS["gray"])
    ax2.text(0.02, 0.03, "95% CI", transform=ax2.transAxes, fontsize=7, color=COLORS["gray"])
    panel_tag(ax1, "(a)")
    panel_tag(ax2, "(b)")
    return save_figure(fig, "revision_boundary_tail_sensitivity.pdf", overwrite)


def draw_coverage_snr(coverage: pd.DataFrame, snr: pd.DataFrame, overwrite: bool) -> Path:
    gains = np.array([30.0, 35.0, 50.0])
    positions = np.arange(len(gains))
    fig = plt.figure(figsize=(6.6, 2.5))
    ax1 = fig.add_axes([0.09, 0.24, 0.38, 0.63])
    ax2 = fig.add_axes([0.58, 0.24, 0.35, 0.63])
    for ax in (ax1, ax2):
        style_axis(ax)
        ax.set_xticks(positions, [str(int(v)) for v in gains])
        ax.set_xlabel("B210 receiver gain (dB)")
    benign = coverage[coverage["condition"] == "benign"].sort_values("gain_db")
    covert = coverage[coverage["condition"] == "covert"].sort_values("gain_db")
    width = 0.32
    ax1.bar(positions - width / 2, benign["parser_coverage_mean"] * 100, width, color=COLORS["blue"], label="Benign")
    ax1.bar(positions + width / 2, covert["parser_coverage_mean"] * 100, width, color=COLORS["orange"], label="Covert")
    ax1.set_ylabel("Parser coverage (%)")
    ax1.set_ylim(0, 105)
    ax1.set_yticks([0, 25, 50, 75, 100])
    ax1.legend(frameon=False, loc="lower right", ncol=1, handlelength=1.2, borderpad=0.2)
    snr = snr.sort_values("gain_db")
    med = snr["snr_db_session_balanced_median"].to_numpy(float)
    low = med - snr["snr_db_session_median_p05"].to_numpy(float)
    high = snr["snr_db_session_median_p95"].to_numpy(float) - med
    ax2.errorbar(positions, med, yerr=np.vstack([low, high]), color=COLORS["green"], marker="o", mfc="white", capsize=2.5, linewidth=1.0, label="Median; p05–p95")
    ax2.set_ylabel("Legal-region SNR (dB)")
    ax2.set_ylim(30, 55)
    ax2.set_yticks([30, 35, 40, 45, 50, 55])
    ax2.legend(frameon=False, loc="lower right", handlelength=1.2, borderpad=0.2)
    ax2.text(0.02, 0.03, "post-CRC/tail excluded", transform=ax2.transAxes, fontsize=7, color=COLORS["gray"])
    panel_tag(ax1, "(a)")
    panel_tag(ax2, "(b)")
    return save_figure(fig, "revision_boundary_coverage_snr.pdf", overwrite)


def draw_pip_boundary_roc(oof: pd.DataFrame, validation: dict, overwrite: bool) -> Path:
    fig = plt.figure(figsize=(3.35, 2.8))
    ax = fig.add_axes([0.18, 0.18, 0.78, 0.73])
    style_axis(ax)
    ax.plot([0, 1], [0, 1], color=COLORS["lightgray"], linestyle=(0, (3, 2)), linewidth=0.8, label="Chance")
    specs = [
        ("direct", COLORS["blue"], "-", "Direct tail", validation["oof"]["direct"]),
        ("pip_outer", COLORS["purple"], "--", "PIP outer boundary", validation["oof"]["pip_outer"]),
    ]
    for view, color, linestyle, label, summary in specs:
        subset = oof[oof["view"] == view]
        fpr, tpr = roc_points(subset["label"].to_numpy(), subset["score_db"].to_numpy())
        ax.step(fpr, tpr, where="post", color=color, linestyle=linestyle, linewidth=1.15, label=f"{label}, AUC {summary['auc']:.3f}")
        ax.plot(0.05, summary["tpr_at_fpr_05"], marker="o", markersize=3.2, color=color, markeredgecolor="white", markeredgewidth=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(loc="lower right", frameon=False, handlelength=2.2, borderpad=0.2, labelspacing=0.35)
    return save_figure(fig, "revision_pip_boundary_oof_roc.pdf", overwrite)


def draw_advertising_timing(advertising: pd.DataFrame, event_interval: pd.DataFrame, overwrite: bool) -> Path:
    gap = advertising[
        (advertising["analysis_scope"] == "device-balanced")
        & (advertising["weighting"] == "device-balanced")
        & (advertising["family"] == "all")
        & (advertising["feature"] == "gap_score")
        & advertising["positive_condition"].isin(["append-last-239B", "append-every-239B"])
    ].set_index("positive_condition")["auc"]
    event_interval = event_interval[(event_interval["crc_policy"] == "all-target-rows") & (event_interval["feature"] == "event_score")].set_index("positive_condition")["auc"]
    values = {
        "Intra-event gap": [float(gap["append-last-239B"]), float(gap["append-every-239B"])],
        "Inter-event interval": [float(event_interval["append-last-239B"]), float(event_interval["append-every-239B"])],
    }
    fig = plt.figure(figsize=(3.35, 2.5))
    ax = fig.add_axes([0.18, 0.24, 0.78, 0.64])
    style_axis(ax)
    ax.axhline(0.5, color=COLORS["lightgray"], linestyle=(0, (3, 2)), linewidth=0.8)
    offsets = [-0.10, 0.10]
    markers = ["o", "s"]
    colors = [COLORS["blue"], COLORS["orange"]]
    labels = ["Append-last", "Append-every"]
    for i, (cue, pair) in enumerate(values.items()):
        for j, value in enumerate(pair):
            ax.scatter(i + offsets[j], value, s=28, marker=markers[j], color=colors[j], edgecolor="white", linewidth=0.55, zorder=3, label=labels[j] if i == 0 else None)
            dy = 0.045 if value < 0.93 else -0.065
            ax.text(i + offsets[j], value + dy, f"{value:.3f}", ha="center", va="center", fontsize=7, color=colors[j])
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 1], ["Intra-event\ngap", "Inter-event\ninterval"])
    ax.set_ylabel("AUC")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(frameon=False, loc="upper right", handletextpad=0.4, borderpad=0.2, labelspacing=0.3)
    return save_figure(fig, "revision_advertising_timing_auc.pdf", overwrite)


def parse_values(value: str | list[float]) -> list[float]:
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    return [float(x) for x in parsed]


def draw_connected_pip_tifs(
    central_auc: pd.DataFrame,
    central_sessions: pd.DataFrame,
    peripheral_auc: pd.DataFrame,
    peripheral_sessions: pd.DataFrame,
    pip_auc: pd.DataFrame,
    pip_sessions: pd.DataFrame,
    overwrite: bool,
) -> Path:
    conditions = ["Central 8 B", "Peripheral 240 B", "PIP clean", "PIP recoverable"]
    x = np.arange(len(conditions))
    session_auc = [
        float(central_auc[(central_auc.sample_unit == "session-median") & (central_auc.feature == "tifs_us")].iloc[0].auc),
        float(peripheral_auc[(peripheral_auc.sample_unit == "session-median") & (peripheral_auc.feature == "tifs_us")].iloc[0].auc),
        float(pip_auc[(pip_auc.pip_scope == "clean_uart_pip") & (pip_auc.pair_filter == "all_pairs") & (pip_auc.sample_unit == "session-median") & (pip_auc.feature == "tifs_us")].iloc[0].auc),
        float(pip_auc[(pip_auc.pip_scope == "all_observed_pip_pcap") & (pip_auc.pair_filter == "all_pairs") & (pip_auc.sample_unit == "session-median") & (pip_auc.feature == "tifs_us")].iloc[0].auc),
    ]
    pair_auc = [
        float(central_auc[(central_auc.sample_unit == "pair-pooled") & (central_auc.feature == "tifs_us")].iloc[0].auc),
        float(peripheral_auc[(peripheral_auc.sample_unit == "pair-pooled") & (peripheral_auc.feature == "tifs_us")].iloc[0].auc),
        float(pip_auc[(pip_auc.pip_scope == "clean_uart_pip") & (pip_auc.pair_filter == "all_pairs") & (pip_auc.sample_unit == "pair-pooled") & (pip_auc.feature == "tifs_us")].iloc[0].auc),
        float(pip_auc[(pip_auc.pip_scope == "all_observed_pip_pcap") & (pip_auc.pair_filter == "all_pairs") & (pip_auc.sample_unit == "pair-pooled") & (pip_auc.feature == "tifs_us")].iloc[0].auc),
    ]
    assert_close("connected/PIP session-median AUC", session_auc, [0.5, 0.5, 0.5, 0.5], atol=1e-8)
    assert_close("connected/PIP pair-pooled AUC", pair_auc, [0.5255, 0.6667, 0.4871, 0.5342], atol=0.00005)

    fig = plt.figure(figsize=(7.0, 2.8))
    ax1 = fig.add_axes([0.085, 0.23, 0.40, 0.64])
    ax2 = fig.add_axes([0.58, 0.23, 0.36, 0.64])
    style_axis(ax1)
    style_axis(ax2)
    ax1.axhline(0.5, color=COLORS["lightgray"], linestyle=(0, (3, 2)), linewidth=0.8)
    ax1.scatter(x, session_auc, s=36, marker="o", color=COLORS["blue"], edgecolor=COLORS["blue"], label="Session-median (main)", zorder=3)
    ax1.scatter(x, pair_auc, s=33, marker="s", facecolor="white", edgecolor=COLORS["orange"], linewidth=1.0, label="Pair-pooled (diagnostic)", zorder=3)
    for xi, value in zip(x, pair_auc):
        ax1.text(xi + 0.07, value + 0.012, f"{value:.3f}", fontsize=7, color=COLORS["orange"], ha="left", va="center")
    ax1.set_xlim(-0.5, 3.5)
    ax1.set_ylim(0.45, 0.72)
    ax1.set_xticks(x, ["Central\n8 B", "Peripheral\n240 B", "PIP\nclean", "PIP\nrecoverable"])
    ax1.set_ylabel("TIFS AUC")
    ax1.set_yticks([0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    ax1.legend(frameon=False, loc="upper left", handletextpad=0.4, borderpad=0.2, labelspacing=0.3)
    panel_tag(ax1, "(a)")

    strip_groups = []
    for frame in [central_sessions, peripheral_sessions]:
        normal = frame["normal_tifs_median_us"].dropna().to_numpy(float)
        covert = frame["covert_candidate_tifs_median_us"].dropna().to_numpy(float)
        strip_groups.append((normal, covert))
    pip = pip_sessions.copy()
    benign = pip[pip["mode"] == "benign"]["tifs_median_us"].dropna().to_numpy(float)
    clean = pip[(pip["mode"] == "pip") & (pip["clean_uart_pip"] == True)]["tifs_median_us"].dropna().to_numpy(float)
    recoverable = pip[pip["mode"] == "pip"]["tifs_median_us"].dropna().to_numpy(float)
    strip_groups.extend([(benign, clean), (benign, recoverable)])
    for i, (negative, positive) in enumerate(strip_groups):
        neg_offsets = np.linspace(-0.07, 0.07, len(negative)) if len(negative) > 1 else np.array([0.0])
        pos_offsets = np.linspace(-0.07, 0.07, len(positive)) if len(positive) > 1 else np.array([0.0])
        ax2.scatter(np.full(len(negative), x[i] - 0.16) + neg_offsets, negative, marker="o", s=24, color=COLORS["blue"], edgecolor="white", linewidth=0.5, label="Benign / normal" if i == 0 else None, zorder=3)
        ax2.scatter(np.full(len(positive), x[i] + 0.16) + pos_offsets, positive, marker="D", s=24, color=COLORS["orange"], edgecolor="white", linewidth=0.5, label="Covert / PIP" if i == 0 else None, zorder=3)
    ax2.set_xlim(-0.5, 3.5)
    ax2.set_ylim(148.4, 151.6)
    ax2.set_xticks(x, ["Central\n8 B", "Peripheral\n240 B", "PIP\nclean", "PIP\nrecoverable"])
    ax2.set_ylabel("Session-median TIFS (us)")
    ax2.set_yticks([149, 150, 151])
    ax2.legend(frameon=False, loc="upper left", handletextpad=0.4, borderpad=0.2, labelspacing=0.3)
    ax2.text(0.98, 0.06, "PIP: exploratory", transform=ax2.transAxes, ha="right", va="bottom", fontsize=7, color=COLORS["gray"])
    panel_tag(ax2, "(b)")
    return save_figure(fig, "revision_connected_pip_tifs.pdf", overwrite)


def sampled_timeline_events(events: pd.DataFrame) -> pd.DataFrame:
    """Select contiguous trace windows and retain source order by frame.

    The source captures contain timestamp-origin wrap/reset artifacts in
    ``t_rel_s``.  ``frame`` is the monotone packet order within each capture,
    so it is used only to establish order.  The resulting PDF deliberately
    lays events out schematically and never maps spacing to time.
    """
    marker_map = {
        "AA_acl": "circle",
        "LL_CIS_REQ": "diamond",
        "LL_CIS_IND": "diamond",
        "AA_f/H_f": "triangle",
    }
    display_map = {
        "AA_acl": r"AA$_{\mathrm{ACL}}$",
        "LL_CIS_REQ": "LL_CIS_REQ",
        "LL_CIS_IND": "LL_CIS_IND",
        "AA_f/H_f": r"AA$_f$/H$_f$",
    }
    windows = {
        "ACL": {"kind": "first_contiguous_frames", "start": 0, "stop": 5},
        "CIS-partial-context": {"kind": "complete_trace", "start": None, "stop": None},
        "PIP": {"kind": "contiguous_fake_aa_window", "start": 1197, "stop": 1210},
    }
    rows = []
    for trace, source in events.groupby("trace", sort=False):
        source = source.sort_values(["frame", "timestamp_epoch"], kind="stable").reset_index(drop=True)
        if not source["frame"].is_monotonic_increasing:
            raise ValueError(f"Source frame order is not monotone for {trace}")
        spec = windows[trace]
        if spec["kind"] == "first_contiguous_frames":
            selected = source.iloc[spec["start"] : spec["stop"]].copy()
        elif spec["kind"] == "complete_trace":
            selected = source.copy()
        else:
            selected = source[(source["frame"] >= spec["start"]) & (source["frame"] <= spec["stop"])].copy()
        if selected.empty:
            raise ValueError(f"Selected timeline window is empty for {trace}")
        if not selected["frame"].is_monotonic_increasing:
            raise ValueError(f"Selected frame order is not monotone for {trace}")
        if not selected["t_rel_s"].is_monotonic_increasing:
            raise ValueError(f"Selected window has timestamp reversal for {trace}: choose a shorter window")
        selected["event_order_index"] = np.arange(len(selected), dtype=int)
        selected["order_basis"] = "monotone source frame order; display spacing is schematic, not time-scaled"
        selected["window_kind"] = spec["kind"]
        selected["window_start_frame"] = int(selected["frame"].iloc[0])
        selected["window_end_frame"] = int(selected["frame"].iloc[-1])
        selected["sampled_for_plot"] = True
        selected["marker"] = selected["event"].map(marker_map).fillna("unknown")
        selected["display_label"] = selected["event"].map(display_map).fillna(selected["event"])
        rows.append(selected)

    retained = pd.concat(rows, ignore_index=True)
    counts = retained.groupby("trace").size().to_dict()
    expected = {"ACL": 5, "CIS-partial-context": 2, "PIP": 14}
    if counts != expected:
        raise ValueError(f"Timeline window counts changed: {counts}; expected {expected}")
    fake_count = int(((retained["trace"] == "PIP") & (retained["event"] == "AA_f/H_f")).sum())
    if fake_count != 10:
        raise ValueError(f"PIP local window must retain all 10 fake-AA hits, got {fake_count}")
    return retained


def _timeline_axis(fig, bounds, panel_label, row_label):
    ax = fig.add_axes(bounds)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(-0.045, 0.5, f"{panel_label} {row_label}", transform=ax.transAxes, ha="right", va="center", fontsize=8.2, clip_on=False)
    return ax


def _order_arrow(ax, y, x0=0.06, x1=0.94):
    ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops={"arrowstyle": "->", "color": COLORS["gray"], "lw": 0.7})


def _rule_box(fig, y, unseen, context):
    fig.text(
        0.755,
        y,
        f"Unseen-AA rule: {unseen}\nContext-aware rule: {context}",
        ha="left",
        va="center",
        fontsize=7.1,
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F7F7F7", "edgecolor": "#BDBDBD", "linewidth": 0.65},
    )


def draw_behavior_timeline(events: pd.DataFrame, behavior_manifest: dict, overwrite: bool) -> tuple[Path, pd.DataFrame]:
    retained = sampled_timeline_events(events)
    event_csv = OUT / "revision_pip_behavior_timeline_events.csv"
    retained.to_csv(event_csv, index=False)
    if not event_csv.is_file():
        raise RuntimeError("Failed to write the plotted event CSV")

    # This is a behavioral-trace schematic, not a timing plot.  One shared
    # left axis keeps the three observed rows aligned; the interpretation
    # matrix on the right uses passive evidence only. Controller state is
    # shown separately as offline ground truth, not as a monitor input.
    fig = plt.figure(figsize=(7.0, 2.5))
    trace_ax = fig.add_axes([0.22, 0.13, 0.50, 0.75])
    trace_ax.set_xlim(0, 1)
    trace_ax.set_ylim(0, 3.25)
    trace_ax.set_xticks([])
    trace_ax.set_yticks([])
    for spine in trace_ax.spines.values():
        spine.set_visible(False)
    trace_ax.text(0.0, 3.18, "Observed trace evidence", ha="left", va="center", fontsize=8.1, color=COLORS["gray"])
    blue = COLORS["blue"]
    req_color = "#C44E52"
    ind_color = COLORS["amber"]
    green = COLORS["green"]
    purple = COLORS["purple"]

    # (a) Normal ACL: primary AA only.
    normal_y = 2.55
    trace_ax.text(-0.045, normal_y, "(a) Normal ACL", transform=trace_ax.get_yaxis_transform(), ha="right", va="center", fontsize=8.2, clip_on=False)
    _order_arrow(trace_ax, normal_y)
    normal_x = np.linspace(0.17, 0.83, 5)
    trace_ax.scatter(normal_x, np.full(5, normal_y), s=30, marker="o", color=blue, edgecolor="white", linewidth=0.5, zorder=3)
    trace_ax.text(0.50, normal_y + 0.22, r"$AA_{\mathrm{ACL}}$", ha="center", va="center", fontsize=7.7, color=blue)
    trace_ax.text(0.50, normal_y - 0.26, "primary AA only; no secondary AA observed", ha="center", va="center", fontsize=7.1, color=COLORS["gray"])

    # (b) One passive lane plus explicitly separated offline ground truth.
    cis_y = 1.58
    trace_ax.text(-0.045, cis_y, "(b) Legitimate CIS\npartial passive context", transform=trace_ax.get_yaxis_transform(), ha="right", va="center", fontsize=8.2, clip_on=False)
    _order_arrow(trace_ax, cis_y, 0.06, 0.94)
    trace_ax.scatter([0.12], [cis_y], s=30, marker="o", color=blue, edgecolor="white", linewidth=0.5, zorder=3)
    trace_ax.text(0.12, cis_y + 0.22, r"$AA_{\mathrm{ACL}}$", ha="center", va="center", fontsize=7.3, color=blue)
    trace_ax.scatter([0.30], [cis_y], s=38, marker="D", color=req_color, edgecolor="white", linewidth=0.5, zorder=3)
    trace_ax.text(0.30, cis_y + 0.22, "LL_CIS_REQ", ha="center", va="center", fontsize=7.0, color=req_color)
    trace_ax.scatter([0.70], [cis_y], s=38, marker="D", color=ind_color, edgecolor="white", linewidth=0.5, zorder=3)
    trace_ax.text(0.70, cis_y + 0.22, "LL_CIS_IND", ha="center", va="center", fontsize=7.0, color="#9A6500")
    trace_ax.text(
        0.70,
        cis_y + 0.45,
        "Controller GT: CIS established",
        ha="center",
        va="center",
        fontsize=6.8,
        color=green,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "#F1F8F5", "edgecolor": green, "linewidth": 0.6},
    )
    trace_ax.annotate(
        "secondary $AA_{\\mathrm{CIS}}$ observed $\\times4$",
        xy=(0.70, cis_y - 0.03),
        xytext=(0.39, cis_y - 0.25),
        ha="left",
        va="center",
        fontsize=6.8,
        color=green,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#F1F8F5", "edgecolor": green, "linewidth": 0.65},
        arrowprops={"arrowstyle": "-", "color": green, "lw": 0.65},
    )

    # (c) A contiguous local PIP window retains every fake-AA hit in the cluster.
    pip_y = 0.57
    trace_ax.text(-0.045, pip_y, "(c) PIP", transform=trace_ax.get_yaxis_transform(), ha="right", va="center", fontsize=8.2, clip_on=False)
    _order_arrow(trace_ax, pip_y)
    pip_window = retained[retained["trace"] == "PIP"]
    positions = np.linspace(0.13, 0.87, len(pip_window))
    for event, marker, color in [("AA_acl", "o", blue), ("AA_f/H_f", "^", purple)]:
        subset = pip_window[pip_window["event"] == event]
        indices = subset["event_order_index"].to_numpy(int)
        trace_ax.scatter(positions[indices], np.full(len(indices), pip_y), s=31 if event == "AA_f/H_f" else 27, marker=marker, color=color, edgecolor="white", linewidth=0.5, zorder=3)
    fake_indices = pip_window[pip_window["event"] == "AA_f/H_f"]["event_order_index"].to_numpy(int)
    cluster_start = positions[fake_indices[0]] - 0.025
    cluster_end = positions[fake_indices[-1]] + 0.025
    trace_ax.plot([cluster_start, cluster_end], [pip_y + 0.25, pip_y + 0.25], color=purple, linewidth=0.8)
    trace_ax.plot([cluster_start, cluster_start], [pip_y + 0.19, pip_y + 0.25], color=purple, linewidth=0.8)
    trace_ax.plot([cluster_end, cluster_end], [pip_y + 0.19, pip_y + 0.25], color=purple, linewidth=0.8)
    trace_ax.text((cluster_start + cluster_end) / 2, pip_y + 0.38, "protocol-visible fake AA + header-like hits (10 observed hits)", ha="center", va="center", fontsize=7.0, color=purple)
    trace_ax.text(0.15, pip_y + 0.22, r"$AA_{\mathrm{ACL}}$", ha="center", va="center", fontsize=7.0, color=blue)
    trace_ax.text(0.85, pip_y + 0.22, r"$AA_{\mathrm{ACL}}$", ha="center", va="center", fontsize=7.0, color=blue)
    trace_ax.text(0.50, pip_y - 0.26, "no passive establishment context observed", ha="center", va="center", fontsize=7.2, color=purple)

    # Compact 3 x 2 interpretation matrix based only on passive evidence.
    matrix_ax = fig.add_axes([0.765, 0.20, 0.215, 0.63])
    matrix_ax.axis("off")
    matrix_ax.text(0.5, 1.05, "Passive interpretation", ha="center", va="bottom", fontsize=8.1, color=COLORS["gray"])
    table = matrix_ax.table(
        cellText=[["ACL", "No", "—"], ["CIS", "Yes", "Yes"], ["PIP", "Yes", "None observed"]],
        colLabels=["Trace", "Secondary-\nAA cue", "Passive\nestablishment\ncontext"],
        cellLoc="center",
        colLoc="center",
        colWidths=[0.22, 0.30, 0.48],
        bbox=[0.0, 0.08, 1.0, 0.82],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.0)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#C8C8C8")
        cell.set_linewidth(0.55)
        if row == 0:
            cell.set_height(0.28)
            cell.set_facecolor("#F1F1F1")
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_fontsize(6.2)
        else:
            cell.set_height(0.18)
            cell.set_facecolor("white" if row % 2 else "#FAFAFA")
            if col == 0:
                cell.get_text().set_fontweight("bold")
            elif cell.get_text().get_text() == "Yes":
                cell.get_text().set_fontweight("bold")
                cell.get_text().set_color("#9A4E00" if col == 1 else green)
            else:
                cell.get_text().set_color(COLORS["gray"])
    fig.text(0.50, 0.12, "Secondary-AA observations are anomaly cues, not sufficient evidence of PIP.", ha="center", va="center", fontsize=7.4, color=COLORS["gray"])
    return save_figure(fig, "revision_pip_behavior_timeline.pdf", overwrite), retained


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="replace existing candidate outputs")
    parser.add_argument(
        "--paper-pdf-dir",
        type=Path,
        default=DEFAULT_PAPER_PDF_DIR,
        help=f"copy generated PDFs only to this directory (default: {DEFAULT_PAPER_PDF_DIR})",
    )
    args = parser.parse_args()
    font_path = configure_matplotlib()
    require_inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    validation = validate_data()

    bandwidth = pd.read_csv(INPUTS["bandwidth"])
    features = pd.read_csv(INPUTS["boundary_packet_features"])
    roc_summary = pd.read_csv(INPUTS["boundary_roc_by_gain"])
    detector = pd.read_csv(INPUTS["boundary_detector_summary"])
    coverage = pd.read_csv(INPUTS["boundary_coverage"])
    snr = pd.read_csv(INPUTS["boundary_snr"])
    pip_oof = pd.read_csv(INPUTS["pip_validation_oof"])
    advertising = pd.read_csv(INPUTS["advertising_timing_auc"])
    event_interval = pd.read_csv(INPUTS["event_interval_auc"])
    central_auc = pd.read_csv(INPUTS["central_tifs_auc"])
    central_sessions = pd.read_csv(INPUTS["central_tifs_sessions"])
    peripheral_auc = pd.read_csv(INPUTS["peripheral_tifs_auc"])
    peripheral_sessions = pd.read_csv(INPUTS["peripheral_tifs_sessions"])
    pip_auc = pd.read_csv(INPUTS["pip_tifs_auc"])
    pip_sessions = pd.read_csv(INPUTS["pip_tifs_sessions"])
    events = pd.read_csv(INPUTS["behavior_events"])
    behavior_manifest = validation["behavior_manifest"]

    outputs = [
        draw_bandwidth(bandwidth, args.overwrite),
        draw_boundary_roc(features, roc_summary, args.overwrite),
        draw_tail_sensitivity(detector, args.overwrite),
        draw_coverage_snr(coverage, snr, args.overwrite),
        draw_pip_boundary_roc(pip_oof, validation["validation_summary"], args.overwrite),
        draw_advertising_timing(advertising, event_interval, args.overwrite),
        draw_connected_pip_tifs(central_auc, central_sessions, peripheral_auc, peripheral_sessions, pip_auc, pip_sessions, args.overwrite),
    ]
    behavior_output, plotted_events = draw_behavior_timeline(events, validation["behavior_manifest"], args.overwrite)
    outputs.append(behavior_output)
    svg_outputs = [path.with_suffix(".svg") for path in outputs]
    missing_svg = [str(path) for path in svg_outputs if not path.is_file()]
    if missing_svg:
        raise RuntimeError("Missing SVG output(s):\n" + "\n".join(missing_svg))
    synced_pdfs = sync_pdf_outputs(outputs, args.paper_pdf_dir)

    input_hashes = {key: {"path": str(path), "sha256": sha256(path)} for key, path in INPUTS.items()}
    output_hashes = {path.name: {"path": str(path), "sha256": sha256(path)} for path in outputs}
    output_hashes.update({path.name: {"path": str(path), "sha256": sha256(path)} for path in svg_outputs})
    output_hashes["revision_pip_behavior_timeline_events.csv"] = {"path": str(OUT / "revision_pip_behavior_timeline_events.csv"), "sha256": sha256(OUT / "revision_pip_behavior_timeline_events.csv")}
    script_path = Path(__file__).resolve()
    manifest = {
        "generator": str(script_path),
        "generator_sha256": sha256(script_path),
        "font_file": font_path,
        "font_family": "serif; Times New Roman if installed, otherwise Liberation Serif or DejaVu Serif",
        "formats": ["one-page vector PDF", "editable SVG"],
        "svg_text_mode": "text elements retained (svg.fonttype=none)",
        "sanity_checks": {
            "boundary_gain_auc": roc_summary[["gain_db", "auc"]].to_dict(orient="records"),
            "pip_selected_window_by_fold": validation["validation_summary"]["selection_stability"]["selected_by_fold"],
            "pip_strict_gate_pass_count": validation["validation_summary"]["selection_stability"]["strict_selection_gate_pass_count"],
        },
        "behavior_timeline": {
            "source_event_count": int(len(events)),
            "retained_event_count": int(len(plotted_events)),
            "retained_event_csv": str(OUT / "revision_pip_behavior_timeline_events.csv"),
            "order_basis": "source frame order within each trace; horizontal spacing is schematic and not time-scaled",
            "selected_windows": plotted_events.groupby("trace").agg(
                event_count=("event", "size"),
                start_frame=("frame", "min"),
                end_frame=("frame", "max"),
                t_rel_monotonic=("t_rel_s", lambda values: bool(values.is_monotonic_increasing)),
            ).reset_index().to_dict(orient="records"),
            "pip_fake_aa_hit_count": int(((plotted_events["trace"] == "PIP") & (plotted_events["event"] == "AA_f/H_f")).sum()),
            "controller_ground_truth": {
                "central_cis_connected_count": validation["cis_summary"]["uart"]["central"]["cis_connected_count"],
                "peripheral_cis_connected_count": validation["cis_summary"]["uart"]["peripheral"]["cis_connected_count"],
            },
            "aa_cis_policy": "X310 AA_cis hits are reported as a time-unlocked annotation; no timed AA_cis marker is synthesized.",
            "ll_cis_rsp_policy": "not observed; no synthetic marker or cross-session merge",
        },
        "inputs": input_hashes,
        "outputs": output_hashes,
        "paper_pdf_sync": {
            "destination": str(args.paper_pdf_dir),
            "svg_copied": False,
            "files": [{"path": str(path), "sha256": sha256(path)} for path in synced_pdfs],
        },
    }
    manifest_path = OUT / "revision_figures_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists; use --overwrite to replace: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Generated {len(outputs)} PDFs and {len(svg_outputs)} SVGs in {OUT}")
    for path in outputs:
        print(path)
    for path in svg_outputs:
        print(path)
    print(f"Synced {len(synced_pdfs)} PDFs only to {args.paper_pdf_dir}")
    for path in synced_pdfs:
        print(path)
    print(OUT / "revision_pip_behavior_timeline_events.csv")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
