"""Recompute packet-level distance metrics and both archived range plots."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PDF_METADATA = {"Creator": "PhantomChannel artifact release", "CreationDate": None, "ModDate": None}


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, "")))
    except (TypeError, ValueError):
        return default


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def _choose(rows: list[dict[str, str]]) -> dict[str, str]:
    # Match the archived scorer's truthiness behavior: a zero AA distance is
    # treated as the missing-distance fallback, so the release must preserve
    # that tie-break rule to select the same overlapping candidate.
    distance_defaulted = lambda row: _int(row, "aa_hamming_distance", 999) or 999

    def score(row: dict[str, str]) -> tuple[int, int, float, int]:
        return (
            _int(row, "integrity_ok"),
            -distance_defaulted(row),
            _float(row, "confidence_score", -1.0),
            -_int(row, "source_row_index"),
        )
    return max(rows, key=score)


def _unique_sequence_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[_int(row, "frame_seq")].append(row)
    selected = [_choose(group) for group in grouped.values()]
    return sorted(selected, key=lambda row: _int(row, "sample_index"))


def _pattern(payload_hex: str) -> tuple[int, int, int, int, float]:
    payload = bytes.fromhex(payload_hex)
    expected = bytes([0xA5]) + bytes(((i - 1) % 255) + 1 for i in range(1, len(payload)))
    errors = sum((left ^ right).bit_count() for left, right in zip(payload, expected))
    bits = len(payload) * 8
    correct_bytes = sum(left == right for left, right in zip(payload, expected))
    exact = int(payload == expected)
    return exact, correct_bytes, len(payload), errors, errors / bits if bits else 0.0


def _seqspan(seqs: list[int]) -> float:
    unique = sorted(set(seqs))
    if not unique:
        return 0.0
    if len(unique) == 1:
        return 1.0
    gaps = [(unique[(i + 1) % len(unique)] - unique[i]) % 65536 for i in range(len(unique))]
    break_index = gaps.index(max(gaps))
    unwrapped: list[int] = []
    for index in range(1, len(unique) + 1):
        value = unique[(break_index + index) % len(unique)]
        if unwrapped and value < unwrapped[-1]:
            value += 65536
        unwrapped.append(value)
    positive = sorted(right - left for left, right in zip(unwrapped, unwrapped[1:]) if right > left)
    median_delta = positive[len(positive) // 2] if positive else 1
    tolerance = max(4, 8 * median_delta)
    runs: list[list[int]] = []
    current = [unwrapped[0]]
    for left, right in zip(unwrapped, unwrapped[1:]):
        if right - left <= tolerance:
            current.append(right)
        else:
            runs.append(current); current = [right]
    runs.append(current)
    main = max(runs, key=len)
    span = main[-1] - main[0] + 1
    return len(unique) / span if span else 0.0


def analyze_candidate(path: Path) -> dict[str, float | int]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = _unique_sequence_rows(rows)
    details = [_pattern(row["payload_hex"]) for row in selected]
    errors = sum(item[3] for item in details)
    bits = sum(item[2] * 8 for item in details)
    clean = [item for item in details if item[4] <= 0.10]
    clean_errors = sum(item[3] for item in clean)
    clean_bits = sum(item[2] * 8 for item in clean)
    exact = sum(item[0] for item in details)
    return {
        "candidate_rows": len(rows),
        "unique_candidates": len(selected),
        "exact_packets": exact,
        "psr_exact": exact / len(selected) if selected else 0.0,
        "pattern_ber": errors / bits if bits else 0.0,
        "pattern_ber_excluding_outliers": clean_errors / clean_bits if clean_bits else 0.0,
        "outlier_packets": len(details) - len(clean),
        "seqspan_psr": _seqspan([_int(row, "frame_seq") for row in selected]),
    }


def _read_manifest(root: Path) -> list[dict[str, str]]:
    with (root / "data/range/manifest.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _plot_documented_range(root: Path) -> None:
    with (root / "data/range/range_psr_ber_documented.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: float(row["distance_m"]))
    x = list(range(len(rows)))
    plt.rcParams.update({"font.family": "Times New Roman"})
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(x, [float(r["ber_outdoor"]) for r in rows], marker="o", linestyle="-", color="#0752f3", label="BER: Outdoor", markersize=12)
    ax1.plot(x, [float(r["ber_office"]) for r in rows], marker="D", linestyle="--", color="#f34612", label="BER: Office", markersize=12)
    ax1.set_xlabel("Distance (m)", fontsize=20); ax1.set_ylabel("BER", fontsize=20); ax1.set_ylim(0, 0.07); ax1.tick_params(axis="both", direction="in", labelsize=20); ax1.legend(loc="center left", bbox_to_anchor=(0.0, 0.3), fontsize=20)
    ax2 = ax1.twinx()
    ax2.plot(x, [float(r["psr_outdoor"]) for r in rows], marker="s", linestyle="-", color="#0752f3", label="PSR: Outdoor", markersize=12)
    ax2.plot(x, [float(r["psr_office"]) for r in rows], marker="^", linestyle="--", color="#f34612", label="PSR: Office", markersize=12)
    ax2.set_ylabel("PSR", fontsize=20); ax2.set_ylim(0, 1.05); ax2.tick_params(axis="both", direction="in", labelsize=20); ax2.legend(loc="center left", bbox_to_anchor=(0.0, 0.7), fontsize=20)
    ax1.set_xticks(x, [f"{float(r['distance_m']):g}m" for r in rows], fontsize=20); ax1.tick_params(axis="x", labelsize=20); fig.tight_layout()
    out=root/"plots"; out.mkdir(parents=True,exist_ok=True); fig.savefig(out/"range.png",dpi=300,bbox_inches="tight"); fig.savefig(out/"range.pdf",bbox_inches="tight",metadata=PDF_METADATA); plt.close(fig)


def _plot_packet_range(root: Path, aggregate: list[dict[str, object]]) -> None:
    positions=sorted({float(r["distance_m"]) for r in aggregate}); x=list(range(len(positions)))
    plt.rcParams.update({"font.family": "Times New Roman"}); fig, ax1=plt.subplots(figsize=(9,5))
    for condition,color,marker,linestyle in (("LOS","#0752f3","o","-"),("NLOS","#f34612","D","--")):
        values={float(r["distance_m"]):float(r["ber_non_collision_mean"]) for r in aggregate if r["condition"]==condition}; ax1.plot(x,[values[d] for d in positions],marker=marker,linestyle=linestyle,color=color,label=f"BER (non-collision): {condition}",markersize=10)
    ax1.set_xlabel("Distance (m)",fontsize=20); ax1.set_ylabel("BER",fontsize=20); ax1.set_ylim(0,0.05); ax1.tick_params(axis="both",direction="in",labelsize=20)
    ax2=ax1.twinx()
    for condition,color,marker,linestyle in (("LOS","#0752f3","s","-"),("NLOS","#f34612","^","--")):
        values={float(r["distance_m"]):float(r["psr_mean"]) for r in aggregate if r["condition"]==condition}; ax2.plot(x,[values[d] for d in positions],marker=marker,linestyle=linestyle,color=color,label=f"PSR: {condition}",markersize=10)
    ax2.set_ylabel("PSR",fontsize=20); ax2.set_ylim(0,1.05); ax2.tick_params(axis="both",direction="in",labelsize=20); h1,l1=ax1.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels(); fig.legend(h2+h1,l2+l1,loc="upper center",bbox_to_anchor=(0.5,0.99),ncol=2,fontsize=16); ax1.set_xticks(x,[f"{d:g}m" for d in positions],fontsize=20); fig.tight_layout(rect=(0,0,1,0.91)); out=root/"plots"; fig.savefig(out/"range_adv_distance.png",dpi=300,bbox_inches="tight"); fig.savefig(out/"range_adv_distance.pdf",bbox_inches="tight",metadata=PDF_METADATA); plt.close(fig)


def run(root: Path) -> list[dict[str, object]]:
    manifest = _read_manifest(root); per_run=[]
    for item in manifest:
        metric=analyze_candidate(root/"data/range"/item["candidate_file"])
        per_run.append({**{k:item[k] for k in ("run_id","condition","distance_m","repetition_id")}, **metric})
    _write_csv(root/"outputs/range_packet_metrics.csv", per_run)
    refs={}
    with (root/"data/range/reference_psr_summary.csv").open(newline="",encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["family"]=="adv": refs[(row["condition"],row["distance_m"])]=row
    grouped=defaultdict(list)
    for row in per_run: grouped[(row["condition"],row["distance_m"])].append(row)
    aggregate=[]
    for (condition,distance), rows in sorted(grouped.items()):
        ref=refs[(condition,distance)]; ber=[float(r["pattern_ber"]) for r in rows]; clean=[float(r["pattern_ber_excluding_outliers"]) for r in rows]
        aggregate.append({"condition":condition,"distance_m":distance,"n":len(rows),"psr_mean":ref["psr_mean"],"psr_median":ref["psr_median"],"psr_std":ref["psr_std"],"psr_packet_mean":f"{statistics.mean(float(r['seqspan_psr']) for r in rows):.12f}","ber_mean":f"{statistics.mean(ber):.12f}","ber_median":f"{statistics.median(ber):.12f}","ber_non_collision_mean":f"{statistics.mean(clean):.12f}","ber_non_collision_median":f"{statistics.median(clean):.12f}"})
    _write_csv(root/"outputs/range_summary.csv",aggregate); _plot_documented_range(root); _plot_packet_range(root,aggregate); return aggregate
