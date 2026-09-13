#!/usr/bin/env python3
"""Aggregate BPL/CAF ablation metrics and the closed stage-wise funnel."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


STAGE_ORDER = (
    "S0_PHYSICAL_BURST",
    "S1_BPL_ACCEPTED",
    "S2_SYMBOLS_AVAILABLE",
    "S3_AA_RECOVERED",
    "S4_LENGTH_VALID",
    "S5_CRC_OR_STRUCTURE_VALID",
    "S6_CAF_ACCEPTED",
    "S7_TARGET_EXACT",
)
MODES = ("baseline", "bpl_only", "caf_only", "bpl_caf")
BPL_MODES = {"bpl_only", "bpl_caf"}
CAF_MODES = {"caf_only", "bpl_caf"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    names: list[str] = []
    for row in rows:
        for name in row:
            if name not in names:
                names.append(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean(values: list[float], seed: int, replicates: int = 2000) -> tuple[float | None, list[float] | None]:
    if not values:
        return None, None
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(replicates, array.size), replace=True).mean(axis=1)
    return float(np.mean(array)), [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def stage_index(value: str) -> int:
    try:
        return STAGE_ORDER.index(value)
    except ValueError:
        return 0


def aggregate_funnel(stage_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    # burst_id is local to a source run, so condition/mode/window alone would
    # merge identically named burst_000000 rows across repetitions and inflate
    # the physical denominator.  Keep the run as an explicit grouping key.
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in stage_rows:
        grouped[(row.get("run_id", ""), row.get("condition", ""), row.get("mode", ""), row.get("window", ""))].append(row)
    output: list[dict[str, Any]] = []
    for (run_id, condition, mode, window), rows in sorted(grouped.items()):
        physical = len({row.get("burst_id", "") for row in rows})
        max_stage = max((stage_index(row.get("stage", "")) for row in rows), default=0)
        mode_rows = len(rows)
        for index, stage in enumerate(STAGE_ORDER):
            # The ablation switches make some gates intentionally
            # non-applicable: baseline has neither BPL nor CAF, CAF-only has
            # no BPL gate, and BPL-only has no CAF/S7 gate.  Marking these as
            # zero would falsely imply that the mode failed that detector;
            # report them as N/A while keeping the shared S0 denominator.
            stage_applicable = not (
                (index == 1 and mode not in BPL_MODES)
                or (index in {6, 7} and mode not in CAF_MODES)
            )
            if not stage_applicable:
                output.append(
                    {
                        "run_id": run_id,
                        "condition": condition,
                        "mode": mode,
                        "window": window,
                        "stage": stage,
                        "stage_index": index,
                        "stage_applicable": False,
                        "physical_bursts": physical,
                        "mode_rows": mode_rows,
                        "cumulative_count": None,
                        "cumulative_rate_physical": None,
                        "conditional_rate_previous_stage": None,
                        "funnel_closes": True,
                        "max_observed_stage_index": max_stage,
                    }
                )
                continue
            cumulative = sum(stage_index(row.get("stage", "")) >= index for row in rows)
            previous = (
                physical
                if index == 0 or (index == 2 and mode not in BPL_MODES)
                else sum(stage_index(row.get("stage", "")) >= index - 1 for row in rows)
            )
            output.append(
                {
                    "run_id": run_id,
                    "condition": condition,
                    "mode": mode,
                    "window": window,
                    "stage": stage,
                    "stage_index": index,
                    "stage_applicable": True,
                    "physical_bursts": physical,
                    "mode_rows": mode_rows,
                    "cumulative_count": cumulative,
                    "cumulative_rate_physical": cumulative / physical if physical else None,
                    "conditional_rate_previous_stage": cumulative / previous if previous else None,
                    "funnel_closes": bool(cumulative <= previous if index > 0 else cumulative == physical),
                    "max_observed_stage_index": max_stage,
                }
            )
    return output


def aggregate_metrics(metric_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in metric_rows:
        grouped[(row.get("condition", ""), row.get("mode", ""), row.get("window", ""))].append(row)
    result: list[dict[str, Any]] = []
    for (condition, mode, window), rows in sorted(grouped.items()):
        def number(name: str) -> list[float]:
            values: list[float] = []
            for row in rows:
                try:
                    values.append(float(row[name]))
                except (KeyError, TypeError, ValueError):
                    pass
            return values

        physical = number("physical_bursts")
        exact = number("posthoc_target_exact")
        aa = number("aa_conditional_on_mode")
        caf = number("caf_conditional_on_candidates")
        out: dict[str, Any] = {
            "condition": condition,
            "mode": mode,
            "window": window,
            "run_count": len(rows),
            "physical_bursts_total": int(sum(physical)),
            "posthoc_exact_total": int(sum(exact)),
            "weighted_exact_rate": sum(exact) / sum(physical) if physical else None,
        }
        for name, values in (("aa_conditional_on_mode", aa), ("caf_conditional_on_candidates", caf)):
            mean, ci = bootstrap_mean(values, seed=20260809 + len(result))
            out[f"{name}_run_mean"] = mean
            out[f"{name}_run_bootstrap_ci95"] = ci
        result.append(out)
    return result


def recompute_mode_metrics(stage_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Rebuild run/window metrics from winner rows with explicit denominators.

    This deliberately does not trust a legacy ``per_run_metrics.csv``: the
    CAF conditional denominator is mode-dependent.  In ``bpl_caf`` it is the
    number of BPL-accepted physical bursts; in ``caf_only`` it is all physical
    bursts because every burst enters the CAF offset scan.
    """

    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in stage_rows:
        grouped[(row.get("run_id", ""), row.get("condition", ""), row.get("window", ""), row.get("mode", ""))].append(row)
    output: list[dict[str, Any]] = []
    for (run_id, condition, window, mode), selected in sorted(grouped.items()):
        physical = len({row.get("burst_id", "") for row in selected})
        count = len(selected)

        def n(key: str) -> int:
            return sum(_as_bool(row.get(key)) for row in selected)

        bpl_count = n("bpl_accepted") if mode in BPL_MODES else 0
        caf_candidate_count = bpl_count if mode == "bpl_caf" else count if mode == "caf_only" else 0
        output.append(
            {
                "run_id": run_id,
                "condition": condition,
                "window": window,
                "mode": mode,
                "physical_bursts": physical,
                "mode_rows": count,
                "bpl_accepted": bpl_count if mode in BPL_MODES else "",
                "symbols_available": n("symbols_available"),
                "aa_recovered": n("aa_recovered"),
                "length_valid": n("length_valid"),
                "structure_valid": n("structure_valid"),
                "caf_accepted": n("caf_accepted") if mode in CAF_MODES else "",
                "caf_candidate_count": caf_candidate_count if mode in CAF_MODES else "",
                "posthoc_target_exact": sum(_as_bool(row.get("posthoc_target_exact")) for row in selected),
                "physical_to_mode_rate": count / physical if physical else None,
                "aa_conditional_on_mode": n("aa_recovered") / count if count else None,
                "caf_conditional_on_candidates": n("caf_accepted") / caf_candidate_count if caf_candidate_count else "",
            }
        )
    return output


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _float_values(rows: list[dict[str, str]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return values


def _median_iqr(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    array = np.asarray(values, dtype=np.float64)
    return float(np.median(array)), float(np.percentile(array, 75) - np.percentile(array, 25))


def summarize_bpl_candidates(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Summarize BPL quality by run, then aggregate conditions by run."""

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("run_id", ""), row.get("condition", ""))].append(row)
    run_rows: list[dict[str, Any]] = []
    for (run_id, condition), group in sorted(grouped.items()):
        accepted = sum(_as_bool(row.get("accepted")) for row in group)
        offsets: list[float] = []
        for row in group:
            try:
                offsets.append(float(row["packet_start_sample"]) - float(row["burst_start_narrow_sample"]))
            except (KeyError, TypeError, ValueError):
                pass
        row: dict[str, Any] = {
            "row_type": "run",
            "run_id": run_id,
            "condition": condition,
            "bpl_rows": len(group),
            "accepted_count": accepted,
            "accepted_rate": accepted / len(group) if group else None,
        }
        for source, target in (
            ("normalized_peak", "peak"),
            ("peak_to_second_ratio", "peak_ratio"),
            ("peak_width_samples", "peak_width"),
            ("competing_peak_count", "competing_peaks"),
            ("bpl_localization_runtime_ms", "runtime_ms"),
        ):
            median, iqr = _median_iqr(_float_values(group, source))
            row[f"{target}_median"] = median
            row[f"{target}_iqr"] = iqr
        median, iqr = _median_iqr(offsets)
        row["start_offset_median"] = median
        row["start_offset_iqr"] = iqr
        run_rows.append(row)

    condition_rows: list[dict[str, Any]] = []
    for condition in sorted({str(row["condition"]) for row in run_rows}):
        group = [row for row in run_rows if row["condition"] == condition]
        out: dict[str, Any] = {
            "row_type": "condition",
            "condition": condition,
            "run_count": len(group),
            "bpl_rows_total": sum(int(row["bpl_rows"]) for row in group),
        }
        for key in (
            "accepted_rate",
            "peak_median",
            "peak_ratio_median",
            "peak_width_median",
            "competing_peaks_median",
            "start_offset_median",
            "runtime_ms_median",
        ):
            values = [float(row[key]) for row in group if row.get(key) is not None]
            mean, ci = bootstrap_mean(values, seed=20260809 + len(condition_rows))
            out[f"{key}_run_mean"] = mean
            out[f"{key}_run_bootstrap_ci95"] = ci
        condition_rows.append(out)

    failures: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        failures[str(row.get("condition", ""))][str(row.get("failure_code", ""))] += 1
    return {
        "run_level": run_rows,
        "condition_level": condition_rows,
        "failure_code_counts": {condition: dict(values) for condition, values in sorted(failures.items())},
    }


def summarize_start_perturbation(candidate_rows: list[dict[str, str]], *, samples_per_symbol: int = 4) -> dict[str, Any]:
    """Measure structure survival at +/-1 and +/-2 symbol offsets around BPL."""

    offsets = (-2 * samples_per_symbol, -samples_per_symbol, 0, samples_per_symbol, 2 * samples_per_symbol)
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        if row.get("mode") == "bpl_caf":
            groups[(row.get("run_id", ""), row.get("condition", ""), row.get("window", ""), row.get("burst_id", ""))].append(row)
    run_offset: dict[tuple[str, str, int], list[int]] = defaultdict(lambda: [0, 0])
    for (run_id, condition, _window, _burst_id), rows in groups.items():
        by_offset: dict[int, dict[str, str]] = {}
        for row in rows:
            try:
                by_offset[int(row["phase_offset_samples"])] = row
            except (KeyError, TypeError, ValueError):
                pass
        if not all(offset in by_offset for offset in offsets):
            continue
        for offset in offsets:
            run_offset[(run_id, condition, offset)][0] += 1
            run_offset[(run_id, condition, offset)][1] += int(_as_bool(by_offset[offset].get("structure_valid")))
    run_rows: list[dict[str, Any]] = []
    for (run_id, condition, offset), (count, valid) in sorted(run_offset.items()):
        run_rows.append(
            {
                "row_type": "run",
                "run_id": run_id,
                "condition": condition,
                "offset_samples": offset,
                "offset_symbols": offset / samples_per_symbol,
                "candidate_bursts": count,
                "structure_valid_count": valid,
                "structure_valid_rate": valid / count if count else None,
            }
        )
    condition_rows: list[dict[str, Any]] = []
    for condition in sorted({str(row["condition"]) for row in run_rows}):
        for offset in offsets:
            group = [row for row in run_rows if row["condition"] == condition and row["offset_samples"] == offset]
            values = [float(row["structure_valid_rate"]) for row in group if row.get("structure_valid_rate") is not None]
            mean, ci = bootstrap_mean(values, seed=20260890 + offset + len(condition_rows))
            condition_rows.append(
                {
                    "row_type": "condition",
                    "condition": condition,
                    "offset_samples": offset,
                    "offset_symbols": offset / samples_per_symbol,
                    "run_count": len(group),
                    "candidate_bursts_total": sum(int(row["candidate_bursts"]) for row in group),
                    "structure_valid_rate_run_mean": mean,
                    "structure_valid_rate_run_bootstrap_ci95": ci,
                }
            )
    return {"run_level": run_rows, "condition_level": condition_rows}


def build_window_manifest(input_dir: Path) -> list[dict[str, Any]]:
    inventory = {row["run_id"]: row for row in read_csv(input_dir / "dataset_inventory.csv")}
    summaries = json.loads((input_dir / "run_summaries.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for item in summaries:
        summary = item.get("summary", item)
        source = inventory[summary["run_id"]]
        sample_offset = int(summary["sample_offset"])
        sample_count = int(summary["sample_count"])
        rows.append(
            {
                "run_id": summary["run_id"],
                "condition": summary["condition"],
                "window": summary["window"],
                "source_root": source["source_root"],
                "iq_path": source["iq_path"],
                "metadata_path": source["metadata_path"],
                "iq_sha256": source["iq_sha256"],
                "actual_sample_rate_sps": source["actual_sample_rate_sps"],
                "sample_offset": sample_offset,
                "sample_count": sample_count,
                "byte_offset": sample_offset * 4,
                "byte_count": sample_count * 4,
            }
        )
    return rows


def _format_number(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def report_markdown(
    summary: dict[str, Any],
    metrics: list[dict[str, Any]],
    funnel: list[dict[str, Any]],
    bpl_summary: dict[str, Any],
    perturbation_summary: dict[str, Any],
) -> str:
    key_stages = ("S0_PHYSICAL_BURST", "S1_BPL_ACCEPTED", "S3_AA_RECOVERED", "S5_CRC_OR_STRUCTURE_VALID", "S6_CAF_ACCEPTED", "S7_TARGET_EXACT")
    grouped_funnel: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in funnel:
        grouped_funnel[(row["condition"], row["mode"])][row["stage"]].append(row)

    lines = [
        "# BPL/CAF ablation report",
        "",
        "This report is post-hoc replay of existing X310 IQ. Physical burst IDs are generated by the local wideband-power detector; parser rows are compatibility references only after mode generation.",
        "",
        "## Scope and guard",
        "",
        f"- BLE_encrypt_check read-only guard: `{summary.get('guard_status')}`",
        f"- Funnel rows: `{summary.get('funnel_rows')}`; aggregate metric rows: `{summary.get('metric_rows')}`",
        f"- Funnel violations: `{summary.get('funnel_violations')}`; non-applicable gate rows: `{summary.get('funnel_na_rows')}`",
        "- Phase-derived timing is disabled; BPL uses discrete correlation peak plus parabolic interpolation.",
        "",
        "## Metrics not identifiable from this replay",
        "",
        "Absolute packet-start error, BER, byte recovery, precision and false-positive rate are `N/A` here because these crops do not contain an independent transmitter timing/label stream. Per-candidate runtime was not captured in this historical replay; the current runner exports `bpl_localization_runtime_ms` for future replays. `S7_TARGET_EXACT` remains a post-hoc parser compatibility field and is not treated as an independent truth source.",
        "",
        "## BPL peak diagnostics",
        "",
        "Peak statistics below are first summarized per run (median/IQR), then averaged across runs. Start offset is relative to the shared physical burst start in the 4 MS/s coordinate system; it is not an absolute timing truth.",
        "",
        "| condition | BPL rows | accepted rate | peak | peak/second | width | competing | start offset | runtime ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in bpl_summary.get("condition_level", []):
        lines.append(
            f"| {row['condition']} | {row['bpl_rows_total']} | {_format_number(row.get('accepted_rate_run_mean'))} | "
            f"{_format_number(row.get('peak_median_run_mean'))} | {_format_number(row.get('peak_ratio_median_run_mean'))} | "
            f"{_format_number(row.get('peak_width_median_run_mean'))} | {_format_number(row.get('competing_peaks_median_run_mean'))} | "
            f"{_format_number(row.get('start_offset_median_run_mean'))} | {_format_number(row.get('runtime_ms_median_run_mean'))} |"
        )
    lines.extend(
        [
            "",
            "Failure-code counts are preserved in `bpl_failure_codes.json`; no threshold was retuned by condition on the evaluation windows.",
            "",
            "## BPL-start perturbation stability",
            "",
            "This diagnostic uses the existing `bpl_caf` offset candidates at 0, +/-1 and +/-2 symbols. It reports structure-valid survival, not parser truth or an absolute timing error.",
            "",
            "| condition | offset (symbols) | candidate bursts | run mean | bootstrap 95% CI |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in perturbation_summary.get("condition_level", []):
        ci = row.get("structure_valid_rate_run_bootstrap_ci95")
        ci_text = "N/A" if not ci else f"{_format_number(ci[0], 3)}–{_format_number(ci[1], 3)}"
        lines.append(
            f"| {row['condition']} | {_format_number(row['offset_symbols'], 1)} | {row['candidate_bursts_total']} | "
            f"{_format_number(row.get('structure_valid_rate_run_mean'))} | {ci_text} |"
        )
    lines.extend(
        [
            "",
        "## Aggregate metrics by window",
        "",
        "These are run-level means with run-cluster bootstrap intervals where a conditional metric is defined. `posthoc exact` is a compatibility audit, not a selection criterion.",
        "",
        "| condition | mode | window | physical | exact rate | AA/mode | CAF/candidate |",
        "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metrics:
        lines.append(
            f"| {row['condition']} | {row['mode']} | {row['window']} | {row['physical_bursts_total']} | "
            f"{_format_number(row['weighted_exact_rate'])} | {_format_number(row['aa_conditional_on_mode_run_mean'])} | "
            f"{_format_number(row['caf_conditional_on_candidates_run_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Cumulative funnel rates",
            "",
            "`N/A` denotes a gate intentionally bypassed by that ablation mode, not a zero result.",
            "",
            "| condition | mode | S0 | S1 BPL | S3 AA | S5 structure | S6 CAF | S7 exact |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for (condition, mode), stage_groups in sorted(grouped_funnel.items()):
        cells: list[str] = []
        for stage in key_stages:
            values = stage_groups.get(stage, [])
            applicable = [row for row in values if row.get("stage_applicable") is not False]
            if not applicable:
                cells.append("N/A")
                continue
            denominator = sum(int(row["physical_bursts"]) for row in applicable)
            numerator = sum(int(row["cumulative_count"]) for row in applicable)
            cells.append(_format_number(numerator / denominator if denominator else None, 3))
        lines.append(f"| {condition} | {mode} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The four modes share the physical-burst denominator and differ only in the declared BPL/CAF switches. Any LOS/NLOS difference is an observation about this receiver path; it is consistent with multipath-sensitive synchronization or early demodulation only when the early funnel stages show the corresponding loss. It is not a causal identification of multipath as the unique cause.",
            "",
            "Parser rows are read only after all four mode winners are generated. They are therefore limited to the post-hoc compatibility audit and do not define physical bursts, choose BPL peaks, or alter mode winners.",
            "",
            "## Conservative paper wording",
            "",
            "Using a locally implemented BLE preamble correlator and a shared physical-burst denominator, the NLOS condition showed lower BPL survival and lower downstream BPL+CAF cumulative survival than LOS. Among bursts that passed BPL, the remaining CAF conditional survival was also lower in NLOS, while CAF-only retained a much larger cumulative survival. These observations are consistent with a multipath-sensitive synchronization and early-demodulation failure path in the receiver. They do not establish multipath as the unique causal explanation, and the post-hoc parser exact field is not used as independent ground truth.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output = (args.output_dir or input_dir).expanduser().resolve()
    stage_rows = read_csv(input_dir / "stage_events.csv")
    bpl_rows = read_csv(input_dir / "bpl_candidates.csv")
    candidate_rows = read_csv(input_dir / "candidate_events.csv")
    funnel = aggregate_funnel(stage_rows)
    metric_rows = recompute_mode_metrics(stage_rows)
    metrics = aggregate_metrics(metric_rows)
    bpl_summary = summarize_bpl_candidates(bpl_rows)
    perturbation_summary = summarize_start_perturbation(candidate_rows)
    window_manifest = build_window_manifest(input_dir)
    guard = json.loads((input_dir / "ble_encrypt_check_guard.json").read_text(encoding="utf-8"))
    summary = {
        "schema_version": 1,
        "input_dir": str(input_dir),
        "guard_status": guard.get("status"),
        "guard_unchanged": guard.get("unchanged"),
        "funnel_rows": len(funnel),
        "metric_rows": len(metrics),
        "funnel_violations": sum(not bool(row["funnel_closes"]) for row in funnel),
        "funnel_na_rows": sum(row.get("stage_applicable") is False for row in funnel),
        "bpl_rows": len(bpl_rows),
        "bpl_condition_rows": len(bpl_summary.get("condition_level", [])),
        "window_manifest_rows": len(window_manifest),
        "stage_order": list(STAGE_ORDER),
        "modes": list(MODES),
        "parser_reference_role": "posthoc_compatibility_only; not a physical denominator or BPL peak selector",
        "phase_derived_timing": False,
    }
    write_csv(output / "stage_funnel.csv", funnel)
    write_csv(output / "per_run_metrics.csv", metric_rows)
    write_csv(output / "aggregate_metrics.csv", metrics)
    write_csv(output / "window_manifest.csv", window_manifest)
    write_csv(output / "bpl_metrics.csv", bpl_summary.get("run_level", []) + bpl_summary.get("condition_level", []))
    write_json(output / "bpl_failure_codes.json", bpl_summary.get("failure_code_counts", {}))
    write_csv(
        output / "bpl_perturbation_stability.csv",
        perturbation_summary.get("run_level", []) + perturbation_summary.get("condition_level", []),
    )
    write_json(
        output / "aggregate_metrics.json",
        {
            "summary": summary,
            "metrics": metrics,
            "bpl_summary": bpl_summary,
            "bpl_start_perturbation": perturbation_summary,
        },
    )
    manifest_path = input_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        code_paths = {
            "bpl_sync": input_dir.parent.parent.parent / "tools/bpl_sync.py",
            "run_bpl_caf_ablation": input_dir.parent.parent.parent / "tools/run_bpl_caf_ablation.py",
            "analyze_bpl_caf_failures": input_dir.parent.parent.parent / "tools/analyze_bpl_caf_failures.py",
        }
        manifest.update(
            {
                "window_manifest": str(output / "window_manifest.csv"),
                "window_manifest_rows": len(window_manifest),
                "run_split_policy": "post_hoc_run_level; two repetitions per condition are insufficient for a three-way split; no condition-specific evaluation retuning",
                "posthoc_audit_code_hashes": {
                    name: sha256_file(path) for name, path in code_paths.items() if path.exists()
                },
            }
        )
        write_json(output / "manifest.json", manifest)
    (output / "report.md").write_text(
        report_markdown(summary, metrics, funnel, bpl_summary, perturbation_summary),
        encoding="utf-8",
    )
    if not guard.get("unchanged", False):
        raise RuntimeError("BLE_encrypt_check guard failed; analysis is invalid")
    print(json.dumps({"status": "completed", **summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
