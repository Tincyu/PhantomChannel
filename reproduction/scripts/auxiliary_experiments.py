#!/usr/bin/env python3
"""Recompute the two auxiliary experiments shipped with the artifact."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from statistics import pstdev
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for row in rows:
        for name in row:
            if name not in names:
                names.append(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def sanitize_runtime_paths(directory: Path, root: Path) -> None:
    """Make scorer manifests independent of the evaluator's checkout path."""

    for generated in directory.rglob("*"):
        if not generated.is_file() or generated.suffix.lower() not in {".json", ".md", ".txt", ".csv", ".log"}:
            continue
        text = generated.read_text(encoding="utf-8", errors="replace")
        sanitized = text.replace(str(root), "<PACKAGE_ROOT>")
        sanitized = sanitized.replace("/tmp/", "<TMP_ROOT>/")
        if sanitized != text:
            generated.write_text(sanitized, encoding="utf-8")


def run_bpl(root: Path, output: Path) -> dict[str, object]:
    archive = root / "data" / "auxiliary_experiments" / "bpl_caf_nlos_formal_records_20260919"
    formal = archive / "formal_result"
    method = archive / "method" / "analyze_bpl_caf_failures.py"
    replay_dir = output / "bpl_caf_nlos_formal_recomputed"
    replay_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(method), "--input-dir", str(formal), "--output-dir", str(replay_dir)],
        check=True,
        cwd=method.parent,
    )
    # The bundled method writes an audit manifest with its runtime paths.
    # Keep the manifest useful while making the release independent of the
    # evaluator's checkout location.
    generated_manifest = replay_dir / "manifest.json"
    if generated_manifest.exists():
        text = generated_manifest.read_text(encoding="utf-8")
        text = text.replace(str(formal), "<PACKAGE_ROOT>/formal_result")
        text = text.replace(str(replay_dir), "<PACKAGE_ROOT>/outputs/auxiliary_experiments/bpl_caf_nlos_formal_recomputed")
        generated_manifest.write_text(text, encoding="utf-8")
    for generated in replay_dir.rglob("*"):
        if not generated.is_file() or generated.suffix.lower() not in {".json", ".md", ".txt", ".csv"}:
            continue
        text = generated.read_text(encoding="utf-8", errors="replace")
        sanitized = text.replace(str(formal), "<PACKAGE_ROOT>/formal_result").replace(
            str(replay_dir), "<PACKAGE_ROOT>/outputs/auxiliary_experiments/bpl_caf_nlos_formal_recomputed"
        )
        if sanitized != text:
            generated.write_text(sanitized, encoding="utf-8")

    aggregate = read_csv(replay_dir / "aggregate_metrics.csv")
    nlos_modes = [
        row
        for row in aggregate
        if row.get("condition") == "NLOS" and row.get("mode") in {"bpl_caf", "caf_only"}
    ]
    if len(nlos_modes) != 6:
        raise AssertionError(f"expected six NLOS BPL/CAF rows, found {len(nlos_modes)}")
    write_csv(output / "bpl_caf_nlos_summary.csv", nlos_modes)

    bpl_rows = read_csv(replay_dir / "bpl_metrics.csv")
    nlos_aggregate = next((row for row in bpl_rows if row.get("row_type") == "condition" and row.get("condition") == "NLOS"), None)
    if nlos_aggregate is None:
        raise AssertionError("missing NLOS aggregate BPL row")
    summary = {
        "nlos_bpl_accepted_rate": float(nlos_aggregate["accepted_rate_run_mean"]),
        "nlos_bpl_rows_total": int(nlos_aggregate["bpl_rows_total"]),
        "nlos_window_rows": len(nlos_modes),
        "source": "formal_result/stage_events.csv and bpl_candidates.csv",
    }
    (output / "bpl_caf_nlos_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def run_antenna(root: Path, output: Path) -> dict[str, object]:
    archive = root / "data" / "auxiliary_experiments" / "antenna_60m_directional_omni_records_20260919_with_psr_scripts"
    summary_file = archive / "summary" / "antenna_peak_by_model.csv"
    rows = read_csv(summary_file)
    stale_prescan = output / "antenna_prescan_summary.csv"
    if stale_prescan.exists():
        stale_prescan.unlink()
    labels = {row.get("label") for row in rows}
    if labels != {"default", "sas571"}:
        raise AssertionError(f"unexpected antenna labels: {labels}")

    # Re-score every retained common-threshold parser record.  This keeps the
    # published PSR/goodput values derived from the analysis-ready records,
    # instead of copying a table snapshot from the paper.
    scorer = archive / "scripts" / "score_iq_parser_candidates.py"
    formal_rel = Path("results") / "directional_common_thr_0p015" / "thr_0p015" / "parser" / "ble_packets.csv"
    run_rows: list[dict[str, object]] = []
    replay_root = output / "antenna_60m_psr_recomputed"
    for run_dir in sorted((archive / "runs").iterdir()):
        if not run_dir.is_dir() or "_d60mlos_" not in run_dir.name:
            continue
        label = "sas571" if "_sas571_" in run_dir.name else "default"
        parser_csv = run_dir / formal_rel
        metadata = run_dir / "iq" / "metadata.json"
        if not parser_csv.is_file() or not metadata.is_file():
            raise AssertionError(f"missing formal parser input for {run_dir.name}")
        run_output = replay_root / run_dir.name
        subprocess.run(
            [
                sys.executable,
                str(scorer),
                "--metadata",
                str(metadata),
                "--parser-csv",
                str(parser_csv),
                "--output-dir",
                str(run_output),
                "--pattern",
            ],
            check=True,
            cwd=scorer.parent,
        )
        rate = json.loads((run_output / "parser_candidate_rate.json").read_text(encoding="utf-8"))
        duration_s = float(rate["iq_capture_duration_s"])
        recovered = int(rate["seq_unique_count"])
        pc_frame_bytes = int(rate["parser_candidate_pc_frame_bytes"])
        run_rows.append(
            {
                "run_id": run_dir.name,
                "label": label,
                "nominal_opportunities": 200,
                "recovered_unique_frames": recovered,
                "psr": recovered / 200.0,
                "pc_frame_bytes": pc_frame_bytes,
                "duration_s": duration_s,
                "goodput_kbps": pc_frame_bytes * 8.0 / duration_s / 1000.0,
                "parser_candidate_rate": f"antenna_60m_psr_recomputed/{run_dir.name}/parser_candidate_rate.json",
            }
        )

    if len(run_rows) != 19:
        raise AssertionError(f"expected 19 formal antenna runs, found {len(run_rows)}")
    sanitize_runtime_paths(replay_root, root)
    write_csv(output / "antenna_60m_psr_recomputed_per_run.csv", run_rows)

    psr_summary: list[dict[str, object]] = []
    for label in ("default", "sas571"):
        selected = [row for row in run_rows if row["label"] == label]
        total_frames = sum(int(row["recovered_unique_frames"]) for row in selected)
        total_nominal = sum(int(row["nominal_opportunities"]) for row in selected)
        total_bytes = sum(int(row["pc_frame_bytes"]) for row in selected)
        total_duration = sum(float(row["duration_s"]) for row in selected)
        psr_values = [float(row["psr"]) for row in selected]
        psr_summary.append(
            {
                "label": label,
                "run_count": len(selected),
                "recovered_unique_frames": total_frames,
                "nominal_opportunities": total_nominal,
                "psr_mean": total_frames / total_nominal,
                "psr_mean_percent": 100.0 * total_frames / total_nominal,
                "psr_std_percentage_points": 100.0 * pstdev(psr_values),
                "pc_frame_bytes": total_bytes,
                "duration_s": total_duration,
                "goodput_kbps": total_bytes * 8.0 / total_duration / 1000.0,
                "threshold": 0.015,
                "source": "runs/*/results/directional_common_thr_0p015/thr_0p015/parser/ble_packets.csv",
            }
        )
    write_csv(output / "antenna_60m_psr_recomputed_summary.csv", psr_summary)
    (output / "antenna_60m_psr_recomputed_summary.json").write_text(
        json.dumps(psr_summary, indent=2) + "\n", encoding="utf-8"
    )

    default = next(row for row in rows if row["label"] == "default")
    sas571 = next(row for row in rows if row["label"] == "sas571")
    default_psr = next(row for row in psr_summary if row["label"] == "default")
    sas571_psr = next(row for row in psr_summary if row["label"] == "sas571")
    combined_rows: list[dict[str, object]] = []
    for peak_row in rows:
        combined = dict(peak_row)
        metrics = next(item for item in psr_summary if item["label"] == peak_row["label"])
        combined.update(
            {
                "psr_run_count": metrics["run_count"],
                "recovered_unique_frames": metrics["recovered_unique_frames"],
                "nominal_opportunities": metrics["nominal_opportunities"],
                "psr_mean_percent": metrics["psr_mean_percent"],
                "psr_std_percentage_points": metrics["psr_std_percentage_points"],
                "goodput_kbps": metrics["goodput_kbps"],
                "threshold": metrics["threshold"],
            }
        )
        combined_rows.append(combined)
    write_csv(output / "antenna_60m_directional_omni_summary.csv", combined_rows)
    result = {
        "default_runs": int(default["run_count"]),
        "sas571_runs": int(sas571["run_count"]),
        "peak_delta_vs_default_db": float(sas571["peak_delta_vs_default_db"]),
        "snr_delta_vs_default_db": float(sas571["snr_delta_vs_default_db"]),
        "default_psr_percent": float(default_psr["psr_mean_percent"]),
        "sas571_psr_percent": float(sas571_psr["psr_mean_percent"]),
        "default_goodput_kbps": float(default_psr["goodput_kbps"]),
        "sas571_goodput_kbps": float(sas571_psr["goodput_kbps"]),
        "psr_gain_percentage_points": float(sas571_psr["psr_mean_percent"]) - float(default_psr["psr_mean_percent"]),
        "goodput_gain_kbps": float(sas571_psr["goodput_kbps"]) - float(default_psr["goodput_kbps"]),
        "source": "summary/antenna_peak_by_model.csv and common-threshold parser CSVs",
    }
    (output / "antenna_60m_directional_omni_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run(root: Path) -> dict[str, object]:
    output = root / "outputs" / "auxiliary_experiments"
    output.mkdir(parents=True, exist_ok=True)
    bpl = run_bpl(root, output)
    antenna = run_antenna(root, output)
    print("[OK] BPL/CAF NLOS formal replay")
    print(
        "[OK] 60 m antenna PSR/goodput replay "
        f"(default {antenna['default_psr_percent']:.1f}%, {antenna['default_goodput_kbps']:.2f} kbps; "
        f"SAS-571 {antenna['sas571_psr_percent']:.1f}%, {antenna['sas571_goodput_kbps']:.2f} kbps)"
    )
    return {"bpl_caf_nlos": bpl, "antenna_60m": antenna}


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[1])
