#!/usr/bin/env python3
"""Collect independent PIP timing pcaps with the documented PIP workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERIPHERAL_FLASH = PROJECT_ROOT / "tools/program_pip_hrs_peripheral_firmware.py"
CENTRAL_FLASH = PROJECT_ROOT / "tools/program_pip_hrs_central_52833_firmware.py"
CAPTURE = PROJECT_ROOT / "tools/run_cis_x310_evidence_capture.py"
PERIPHERAL_IMAGE = PROJECT_ROOT / "artifacts/firmware/phantomchannel_pip_hrs_peripheral/merged.hex"
CENTRAL_IMAGE = PROJECT_ROOT / "artifacts/firmware/phantomchannel_pip_hrs_central_52833/merged.hex"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=PROJECT_ROOT, text=True, check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--run-prefix", default="timing_formal_pip")
    parser.add_argument("--target-address", default="c0:de:52:84:00:41")
    parser.add_argument("--nvme-root", type=Path, default=PROJECT_ROOT / "testdata")
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("/path/to/PhantomChannel/experiments/figure/detector_roc_20260809/event_timing/pip_supplemental/formal"),
    )
    parser.add_argument("--sniffer-timeout-ms", type=int, default=5000)
    args = parser.parse_args()
    if args.repetitions <= 0 or args.duration_s <= 0:
        raise SystemExit("repetitions and duration-s must be positive")
    args.nvme_root = args.nvme_root.expanduser().resolve()
    args.archive_root = args.archive_root.expanduser().resolve()
    if not PERIPHERAL_IMAGE.is_file() or not CENTRAL_IMAGE.is_file():
        raise SystemExit("stable HRS/PIP firmware images are missing")

    results: list[dict[str, object]] = []
    for repetition in range(1, args.repetitions + 1):
        run_id = f"{args.run_prefix}_rep{repetition}"
        peripheral = run([sys.executable, str(PERIPHERAL_FLASH)])
        if peripheral.returncode != 0:
            results.append({"run_id": run_id, "stage": "flash_peripheral", "returncode": peripheral.returncode})
            break
        central = run([sys.executable, str(CENTRAL_FLASH)])
        if central.returncode != 0:
            results.append({"run_id": run_id, "stage": "flash_central", "returncode": central.returncode})
            break
        capture = run([
            sys.executable, str(CAPTURE),
            "--trace-kind", "pip",
            "--duration-s", str(args.duration_s),
            "--target-address", args.target_address,
            "--sniffer-only",
            "--reset-boards",
            "--sniffer-timeout-ms", str(args.sniffer_timeout_ms),
            "--run-id", run_id,
            "--nvme-root", str(args.nvme_root),
            "--archive-root", str(args.archive_root),
            "--setup-s", "1",
        ])
        results.append({
            "run_id": run_id,
            "stage": "capture",
            "returncode": capture.returncode,
            "archive_run": str(args.archive_root / run_id),
        })
        if capture.returncode != 0:
            break

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "trace_kind": "pip",
        "repetitions_requested": args.repetitions,
        "duration_s": args.duration_s,
        "target_address": args.target_address,
        "flash_method": "pip_cis_52840_52833_firmware_plan.md via nrfutil",
        "peripheral_image": str(PERIPHERAL_IMAGE),
        "peripheral_sha256": sha256(PERIPHERAL_IMAGE),
        "central_image": str(CENTRAL_IMAGE),
        "central_sha256": sha256(CENTRAL_IMAGE),
        "nvme_root": str(args.nvme_root),
        "archive_root": str(args.archive_root),
        "results": results,
        "completed_repetitions": sum(item.get("stage") == "capture" and item.get("returncode") == 0 for item in results),
    }
    args.archive_root.mkdir(parents=True, exist_ok=True)
    (args.archive_root / "batch_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0 if manifest["completed_repetitions"] == args.repetitions else 2


if __name__ == "__main__":
    raise SystemExit(main())
