#!/usr/bin/env python3
"""Run this checkout's analysis against the released offline experiment bundle."""

from __future__ import annotations

import argparse
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path, help="Released experiment ZIP")
    parser.add_argument("--output-dir", required=True, type=Path, help="New workspace for inputs and generated results")
    args = parser.parse_args()
    bundle = args.bundle.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not bundle.is_file():
        parser.error(f"experiment bundle does not exist: {bundle}")
    if output.exists():
        parser.error(f"output directory already exists; choose a new directory: {output}")

    # Validate all members before extraction so the bundle cannot write outside
    # the selected workspace or introduce links to external data.
    with zipfile.ZipFile(bundle) as archive:
        roots = set()
        for item in archive.infolist():
            path = PurePosixPath(item.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in item.filename or not path.parts:
                parser.error(f"unsafe archive member: {item.filename}")
            mode = item.external_attr >> 16
            if stat.S_ISLNK(mode):
                parser.error(f"archive contains a symbolic link: {item.filename}")
            roots.add(path.parts[0])
        if len(roots) != 1:
            parser.error("expected a single experiment-package root in the archive")
        package_name = next(iter(roots))
        required = ("data/", "checks/", "reproduce_all.py")
        names = archive.namelist()
        if not all(any(name.startswith(f"{package_name}/{part}") for name in names) for part in required):
            parser.error("archive is missing experiment data, regression checks or the runner")
        output.mkdir(parents=True)
        archive.extractall(output)

    package = output / package_name
    source = Path(__file__).resolve().parent
    shutil.copy2(source / "reproduce_all.py", package / "reproduce_all.py")
    shutil.copytree(source / "scripts", package / "scripts", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    print(f"Reproduction workspace: {package}", flush=True)
    return subprocess.run([sys.executable, "reproduce_all.py"], cwd=package).returncode


if __name__ == "__main__":
    raise SystemExit(main())
