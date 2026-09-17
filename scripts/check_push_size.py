#!/usr/bin/env python3
"""Reject an unexpectedly large outbound Git history before a public push."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_BLOB_BYTES = 8 * 1024 * 1024
MAX_OUTBOUND_BLOB_BYTES = 64 * 1024 * 1024
MAX_OUTBOUND_BLOBS = 5_000


def git(*args: str, input_text: str | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        input=input_text,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def default_base() -> str:
    try:
        return git(
            "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise ValueError("no upstream configured; pass --base explicitly") from exc


def outbound_blobs(base: str) -> list[tuple[int, str]]:
    git("rev-parse", "--verify", f"{base}^{{commit}}")
    objects = git("rev-list", "--objects", f"{base}..HEAD")
    if not objects.strip():
        return []
    records = git(
        "cat-file",
        "--batch-check=%(objecttype) %(objectsize) %(rest)",
        input_text=objects,
    )
    blobs: list[tuple[int, str]] = []
    for line in records.splitlines():
        object_type, size_text, *path_parts = line.split(" ", 2)
        if object_type != "blob":
            continue
        path = path_parts[0] if path_parts else "<unknown>"
        blobs.append((int(size_text), path))
    return blobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        help="remote base ref to compare with; defaults to the branch upstream",
    )
    args = parser.parse_args()
    try:
        base = args.base or default_base()
        blobs = outbound_blobs(base)
    except (ValueError, subprocess.CalledProcessError) as exc:
        print(f"Outbound size check FAILED: {exc}")
        return 2

    total = sum(size for size, _ in blobs)
    problems: list[str] = []
    if len(blobs) > MAX_OUTBOUND_BLOBS:
        problems.append(
            f"{len(blobs)} outbound blobs exceeds limit {MAX_OUTBOUND_BLOBS}"
        )
    if total > MAX_OUTBOUND_BLOB_BYTES:
        problems.append(
            f"{total} outbound blob bytes exceeds limit {MAX_OUTBOUND_BLOB_BYTES}"
        )
    for size, path in sorted(blobs, reverse=True):
        if size > MAX_BLOB_BYTES:
            problems.append(f"{path}: outbound blob is {size} bytes")

    if problems:
        print(f"Outbound size check FAILED against {base}:")
        for problem in problems:
            print(f"- {problem}")
        print("Remove the large content from every unpushed commit before pushing.")
        return 1

    print(
        f"Outbound size check passed against {base}: "
        f"{len(blobs)} unique blobs, {total} bytes."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
