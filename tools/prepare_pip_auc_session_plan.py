#!/usr/bin/env python3
"""Create the pre-registered 15-session ledger for the PIP AUC run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


PSSD_ROOT = Path("/path/to/PhantomChannel")
DEFAULT_ROOT = PSSD_ROOT / "experiments/figure/pip_boundary_auc_20260808"
CONDITIONS = ("benign", "direct_tail", "pip")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--firmware-manifest",
        type=Path,
        default=Path("artifacts/firmware/pip_auc_20260808/manifest.json"),
    )
    args = parser.parse_args()
    manifest = json.loads(args.firmware_manifest.read_text(encoding="utf-8"))
    by_condition = {item["condition"]: item for item in manifest["variants"]}
    rows = []
    for condition in CONDITIONS:
        for repetition in range(1, 6):
            session_id = f"pip_auc_{condition}_rep{repetition}"
            firmware = by_condition[condition]
            rows.append(
                {
                    "session_id": session_id,
                    "condition": condition,
                    "repetition": repetition,
                    "status": "planned",
                    "formal_inclusion": "yes",
                    "independent_session": "yes",
                    "phy": "2m",
                    "covert_len_bytes": 0 if condition == "benign" else 2,
                    "covert_payload_hex": "" if condition == "benign" else "a501",
                    "notification_interval_ms": 1000,
                    "x310_center_hz": 2440000000,
                    "x310_requested_rate_sps": 80000000,
                    "x310_expected_actual_rate_sps": 100000000,
                    "x310_bandwidth_hz": 80000000,
                    "x310_gain_db": 50,
                    "x310_antenna": "RX2",
                    "sample_format": "sc16",
                    "capture_seconds": 15,
                    "nvme_stage": "required",
                    "iq_path": "",
                    "metadata_path": "",
                    "parser_csv_raw_stage2": "",
                    "audit_json_raw_stage2": "",
                    "real_aa": "",
                    "fake_aa": "",
                    "crc_init": "",
                    "tx_done_count": "",
                    "measured_snr_db": "",
                    "iq_start_utc": "",
                    "iq_end_utc": "",
                    "firmware_name": firmware["name"],
                    "firmware_sha256": firmware["sha256"],
                    "output_dir": str(args.output_dir / "sessions" / session_id),
                    "notes": "Fill dynamic AA/CRCInit/TX_DONE/SNR/time after the session; do not use pilot as formal repetition.",
                }
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = args.output_dir / "pip_formal_session_plan.csv"
    with ledger_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    preflight = {
        "schema_version": 1,
        "experiment": "pip_boundary_auc_20260808",
        "formal_sessions": 15,
        "sessions_per_condition": 5,
        "conditions": list(CONDITIONS),
        "capture": {
            "phy": "2m",
            "center_hz": 2440000000,
            "requested_rate_sps": 80000000,
            "expected_actual_rate_sps": 100000000,
            "format": "sc16",
            "gain_db": 50,
            "antenna": "RX2",
            "segment_seconds": 15,
            "write_order": "NVMe first, then direct copy to PSSD",
            "nvme_delete_after_copy": "required after capture/metadata/UART/hash checks",
        },
        "frozen_detector": {
            "guard_us": 4,
            "window_us": 64,
            "score_direction": "higher_score_more_boundary_anomaly",
            "boundary_phy": "2m-aware",
        },
        "old_pilot_excluded": "/path/to/PhantomChannel/testdata/20260806_171601_pip_2m_x310_acceptance/",
        "firmware_manifest": str(args.firmware_manifest),
        "session_ledger": str(ledger_path),
        "x310_required_before_capture": True,
    }
    (args.output_dir / "pip_formal_preflight.json").write_text(
        json.dumps(preflight, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"ledger": str(ledger_path), "sessions": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
