#!/usr/bin/env python3
"""Fail closed on obvious non-source and machine-private publication mistakes.

This is a release preflight, not a substitute for manual copyright or secret review.
It scans the files Git would see now (tracked plus non-ignored untracked files).
"""

from __future__ import annotations

import re
import subprocess
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_EXTENSIONS = {
    ".c", ".conf", ".cpp", ".example", ".h", ".hpp", ".json",
    ".md", ".patch", ".py", ".rst", ".sh", ".txt", ".yaml", ".yml",
}
FORBIDDEN_SUFFIXES = {".bin", ".elf", ".gz", ".hex", ".iq", ".key", ".p12", ".pem", ".sc16", ".so", ".tar", ".tgz", ".zip"}
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_GIT_VISIBLE_FILES = 5_000
PRIVATE_PATH = re.compile(r"/(?:home/|srv/embedded-lab/|media/|opt/uhd-)")
DEVICE_ID = re.compile(r"\b(?:001050[0-9]{6}|U[0-9]{6})\b")
SECRET = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\bgh[pousr]_[A-Za-z0-9]{20,}\b|\bAKIA[0-9A-Z]{16}\b")


def git_paths() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"],
        cwd=ROOT, check=True, capture_output=True,
    )
    return sorted({ROOT / part.decode() for part in result.stdout.split(b"\0") if part})


def check() -> list[str]:
    problems: list[str] = []
    paths = git_paths()
    total_bytes = 0
    regular_files = 0
    for path in paths:
        rel = path.relative_to(ROOT)
        if path.is_symlink():
            problems.append(f"{rel}: symlink requires manual review")
            continue
        if not path.is_file():
            continue
        regular_files += 1
        total_bytes += path.stat().st_size
        if rel.name == "local.env" or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"{rel}: private/generated file type")
        if path.stat().st_size > MAX_FILE_BYTES:
            problems.append(f"{rel}: exceeds {MAX_FILE_BYTES} bytes")
        if path.suffix.lower() not in TEXT_EXTENSIONS or rel == Path("scripts/check_public.py"):
            continue
        try:
            data = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            problems.append(f"{rel}: non-UTF-8 text requires review")
            continue
        for label, pattern in (("machine path", PRIVATE_PATH), ("device identifier", DEVICE_ID), ("secret-like token", SECRET)):
            if pattern.search(data):
                problems.append(f"{rel}: {label}")
    manifest_root = ROOT / "receiver"
    manifest = manifest_root / "SOURCE_MANIFEST.sha256"
    if not manifest.is_file():
        problems.append("receiver: source manifest missing")
    else:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            digest, _, filename = line.partition("  ")
            source = manifest_root / filename
            if not re.fullmatch(r"[0-9a-f]{64}", digest) or not filename or not source.is_file():
                problems.append(f"receiver manifest entry invalid: {filename}")
                continue
            if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                problems.append(f"receiver manifest hash mismatch: {filename}")
    if regular_files > MAX_GIT_VISIBLE_FILES:
        problems.append(
            f"Git-visible file count {regular_files} exceeds {MAX_GIT_VISIBLE_FILES}"
        )
    if total_bytes > MAX_TOTAL_BYTES:
        problems.append(
            f"Git-visible content {total_bytes} bytes exceeds {MAX_TOTAL_BYTES} bytes"
        )
    return problems


def main() -> int:
    problems = check()
    if problems:
        print("Publication preflight FAILED:")
        for problem in problems:
            print(f"- {problem}")
        return 1
    paths = git_paths()
    total_bytes = sum(path.stat().st_size for path in paths if path.is_file())
    print(
        "Publication preflight passed "
        f"({len(paths)} Git-visible files, {total_bytes} bytes)."
    )
    print("Manual license, history, and dependency review is still required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
