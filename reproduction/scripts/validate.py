"""Data-level validation for the standalone release directory."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np


def run(root: Path) -> dict[str, int]:
    npy_files = sorted((root / "data/frequency").glob("**/*.npy"))
    if not npy_files:
        raise AssertionError("no frequency NPY files")
    npy_count = 0
    for path in npy_files:
        array = np.load(path, allow_pickle=False)
        if array.dtype == object or array.ndim != 1 or not np.iscomplexobj(array):
            raise AssertionError(f"unexpected IQ array: {path}: {array.dtype} {array.shape}")
        npy_count += 1

    range_manifest = list(csv.DictReader((root / "data/range/manifest.csv").open(newline="", encoding="utf-8")))
    range_count = 0
    for item in range_manifest:
        path = root / "data/range" / item["candidate_file"]
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != int(item["candidate_rows"]):
            raise AssertionError(f"range row-count mismatch: {path}")
        required = {"sample_index", "frame_seq", "frame_hex", "payload_hex", "integrity_ok", "confidence_score"}
        if not required.issubset(rows[0]):
            raise AssertionError(f"range candidate fields missing: {path}")
        range_count += len(rows)

    timing_manifest = list(csv.DictReader((root / "data/advertising_timing/manifest.csv").open(newline="", encoding="utf-8")))
    timing_count = 0
    for item in timing_manifest:
        path = root / "data/advertising_timing" / item["table_file"]
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != int(item["retained_records"]):
            raise AssertionError(f"timing row-count mismatch: {path}")
        timing_count += len(rows)

    forbidden = re.compile(r"/(?:home|media|mnt)/|[0-9a-f]{2}(?::[0-9a-f]{2}){5}|\bU\d{6}\b|D12233445566", re.IGNORECASE)
    text_files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".md", ".py", ".csv", ".json", ".txt", ".yml", ".yaml"}]
    privacy_hits = []
    for path in text_files:
        if path.name == "validate.py":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_no, line in enumerate(text.splitlines(), 1):
            if forbidden.search(line):
                privacy_hits.append(f"{path.relative_to(root)}:{line_no}")
    if privacy_hits:
        raise AssertionError("privacy scan hits: " + ", ".join(privacy_hits[:8]))
    return {"npy_files": npy_count, "range_candidate_rows": range_count, "timing_packet_rows": timing_count, "privacy_hits": 0}
