"""Recompute the appendix's strict 60-m metrics after reproduce_all.py.

Usage: python3 verify_antenna_correctness.py /path/to/extracted/experiment
Only reads experiment data; prints a JSON summary to stdout.
"""
import argparse
import collections
import csv
import importlib.util
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("experiment_root", type=Path)
args = parser.parse_args()
root = args.experiment_root.resolve()
scorer = root / "data/auxiliary_experiments/antenna_60m_directional_omni_records_20260919_with_psr_scripts/scripts/score_iq_parser_candidates.py"
spec = importlib.util.spec_from_file_location("antenna_scorer", scorer)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
groups = collections.defaultdict(list)
for folder in sorted((root / "outputs/auxiliary_experiments/antenna_60m_psr_recomputed").iterdir()):
    if not folder.is_dir():
        continue
    summary = json.loads((folder / "run_summary.json").read_text())
    with (folder / "parser_candidate_packets.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("integrity_ok", "source_row_index", "sample_index"):
            row[key] = int(row[key])
    chosen = module.unique_sequence_candidates(rows)
    strict = [row for row in chosen if int(row["pattern_exact"]) and int(row["integrity_ok"])]
    groups["SAS-571" if "sas571" in folder.name else "default"].append({
        "duration_s": float(summary["iq_capture_duration_s"]),
        "candidates": len(chosen),
        "strict": len(strict),
        "useful_bytes": sum(int(row["payload_data_bytes_excluding_marker"]) for row in strict),
    })
out = {}
for label, runs in groups.items():
    opportunities = 200 * len(runs)
    out[label] = {
        "runs": len(runs),
        "nominal_opportunities": opportunities,
        "candidate_recovery_percent": 100 * sum(r["candidates"] for r in runs) / opportunities,
        "exact_and_integrity_percent": 100 * sum(r["strict"] for r in runs) / opportunities,
        "exact_and_integrity_useful_kbps": 8 * sum(r["useful_bytes"] for r in runs) / sum(r["duration_s"] for r in runs) / 1000,
    }
print(json.dumps(out, indent=2))
