#!/usr/bin/env python3
"""Prepare the firmware matrix required by detector redesign §6.

The script builds isolated overlay configurations; it never flashes a board.
The formal matrix has six traffic conditions, while connected central-side
testing needs a matched benign central image and a normal 52833 sink as
supporting images.  Consequently the build set contains eight images:

* advertising: normal, append-last, append-every;
* connected peripheral: normal, historical 240-byte on-air tail;
* connected central: normal, 8-byte tail;
* the fixed 52833 GATT sink used by both central images.

Use ``--dry-run`` to print the matrix, ``--build`` to compile it, and keep
flashing as a separate, explicit operation after reviewing the manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
from portable_paths import NCS_TOOLCHAIN as TOOLCHAIN_ROOT, NCS_WORKSPACE as SDK_ROOT  # noqa: E402

WEST_PYTHON = TOOLCHAIN_ROOT / "usr/local/bin/python3"
ZEPHYR_SDK = TOOLCHAIN_ROOT / "opt/zephyr-sdk"
OUTPUT_ROOT = PROJECT_ROOT / "artifacts/firmware/event_timing"
CONFIG_ROOT = PROJECT_ROOT / "configs/event_timing"


MATRIX: dict[str, dict[str, Any]] = {
    "adv_normal": {
        "traffic": "advertising",
        "condition": "normal",
        "sample": "phantomchannel_adv_broadcaster",
        "source": "bluetooth/phantomchannel_adv_broadcaster",
        "board": "nrf52840dk/nrf52840",
        "overlay": "adv_normal.conf",
        "config": {
            "CONFIG_PHANTOMCHANNEL_ADV_BENIGN": "y",
            "CONFIG_PHANTOMCHANNEL_ADV_COVERT_LEN": "239",
            "CONFIG_PHANTOMCHANNEL_ADV_INTERVAL_MS": "20",
            "CONFIG_PHANTOMCHANNEL_ADV_EMBED_EVERY": "n",
            "CONFIG_BT_CTLR_ADV_DELAY_ZERO": "y",
        },
        "flash": {"board": "nrf52840dk", "role": "advertiser"},
    },
    "adv_append_last": {
        "traffic": "advertising",
        "condition": "append-last-239B",
        "sample": "phantomchannel_adv_broadcaster",
        "source": "bluetooth/phantomchannel_adv_broadcaster",
        "board": "nrf52840dk/nrf52840",
        "overlay": "adv_append_last.conf",
        "config": {
            "CONFIG_PHANTOMCHANNEL_ADV_BENIGN": "n",
            "CONFIG_PHANTOMCHANNEL_ADV_COVERT_LEN": "239",
            "CONFIG_PHANTOMCHANNEL_ADV_INTERVAL_MS": "20",
            "CONFIG_PHANTOMCHANNEL_ADV_EMBED_EVERY": "n",
            "CONFIG_BT_CTLR_ADV_DELAY_ZERO": "y",
        },
        "flash": {"board": "nrf52840dk", "role": "advertiser"},
    },
    "adv_append_every": {
        "traffic": "advertising",
        "condition": "append-every-239B",
        "sample": "phantomchannel_adv_broadcaster",
        "source": "bluetooth/phantomchannel_adv_broadcaster",
        "board": "nrf52840dk/nrf52840",
        "overlay": "adv_append_every.conf",
        "config": {
            "CONFIG_PHANTOMCHANNEL_ADV_BENIGN": "n",
            "CONFIG_PHANTOMCHANNEL_ADV_COVERT_LEN": "239",
            "CONFIG_PHANTOMCHANNEL_ADV_INTERVAL_MS": "20",
            "CONFIG_PHANTOMCHANNEL_ADV_EMBED_EVERY": "y",
            "CONFIG_BT_CTLR_ADV_DELAY_ZERO": "y",
        },
        "flash": {"board": "nrf52840dk", "role": "advertiser"},
    },
    "conn_peripheral_normal": {
        "traffic": "connection",
        "condition": "normal",
        "sample": "phantomchannel_hrs_peripheral",
        "source": "bluetooth/phantomchannel_hrs_peripheral",
        "board": "nrf52840dk/nrf52840",
        "overlay": "conn_normal.conf",
        "config": {
            "CONFIG_PHANTOMCHANNEL_PERIPHERAL_TX": "n",
            "CONFIG_PHANTOMCHANNEL_EMBED_ENABLE": "n",
            "CONFIG_PHANTOMCHANNEL_NOTIFY_INTERVAL_MS": "20",
        },
        "flash": {"board": "nrf52840dk", "role": "hrs-peripheral"},
    },
    "conn_peripheral_240b": {
        "traffic": "connection",
        "condition": "peripheral-side-240B",
        "sample": "phantomchannel_hrs_peripheral",
        "source": "bluetooth/phantomchannel_hrs_peripheral",
        "board": "nrf52840dk/nrf52840",
        "overlay": "conn_peripheral_240b.conf",
        "config": {
            "CONFIG_PHANTOMCHANNEL_PERIPHERAL_TX": "y",
            "CONFIG_PHANTOMCHANNEL_EMBED_ENABLE": "y",
            "CONFIG_PHANTOMCHANNEL_COVERT_LEN": "231",
            "CONFIG_PHANTOMCHANNEL_NOTIFY_INTERVAL_MS": "20",
        },
        "flash": {"board": "nrf52840dk", "role": "hrs-peripheral"},
        "air_tail_bytes": 240,
        "application_covert_len": 231,
    },
    "conn_central_normal": {
        "traffic": "connection",
        "condition": "normal",
        "sample": "phantomchannel_central_gatt_write",
        "source": "bluetooth/phantomchannel_central_gatt_write",
        "board": "nrf52840dk/nrf52840",
        "overlay": "conn_central_normal.conf",
        "config": {
            "CONFIG_PHANTOMCHANNEL_CENTRAL_TX": "y",
            "CONFIG_PHANTOMCHANNEL_COVERT_LEN": "0",
            "CONFIG_PHANTOMCHANNEL_WRITE_INTERVAL_MS": "20",
            "CONFIG_PHANTOMCHANNEL_EMBED_ENABLE": "n",
            "CONFIG_PHANTOMCHANNEL_DYNAMIC_TIMING": "n",
            "CONFIG_PHANTOMCHANNEL_FORCE_2M": "y",
        },
        "flash": {"board": "nrf52840dk", "role": "central"},
    },
    "conn_central_normal_cal_30ms": {
        "traffic": "connection-calibration",
        "condition": "normal-calibration-30ms",
        "sample": "phantomchannel_central_gatt_write",
        "source": "bluetooth/phantomchannel_central_gatt_write",
        "board": "nrf52840dk/nrf52840",
        "overlay": "conn_central_normal_cal_30ms.conf",
        "config": {
            "CONFIG_PHANTOMCHANNEL_CENTRAL_TX": "y",
            "CONFIG_PHANTOMCHANNEL_COVERT_LEN": "0",
            "CONFIG_PHANTOMCHANNEL_WRITE_INTERVAL_MS": "20",
            "CONFIG_PHANTOMCHANNEL_CONN_INTERVAL_UNITS": "24",
            "CONFIG_PHANTOMCHANNEL_EMBED_ENABLE": "n",
            "CONFIG_PHANTOMCHANNEL_DYNAMIC_TIMING": "n",
            "CONFIG_PHANTOMCHANNEL_FORCE_2M": "y",
        },
        "flash": {"board": "nrf52840dk", "role": "central-calibration"},
    },
    "conn_central_normal_cal_30ms_1m": {
        "traffic": "connection-calibration",
        "condition": "normal-calibration-30ms-1M",
        "sample": "phantomchannel_central_gatt_write",
        "source": "bluetooth/phantomchannel_central_gatt_write",
        "board": "nrf52840dk/nrf52840",
        "overlay": "conn_central_normal_cal_30ms_1m.conf",
        "config": {
            "CONFIG_PHANTOMCHANNEL_CENTRAL_TX": "y",
            "CONFIG_PHANTOMCHANNEL_COVERT_LEN": "0",
            "CONFIG_PHANTOMCHANNEL_WRITE_INTERVAL_MS": "20",
            "CONFIG_PHANTOMCHANNEL_CONN_INTERVAL_UNITS": "24",
            "CONFIG_PHANTOMCHANNEL_EMBED_ENABLE": "n",
            "CONFIG_PHANTOMCHANNEL_DYNAMIC_TIMING": "n",
            "CONFIG_PHANTOMCHANNEL_FORCE_2M": "n",
            "CONFIG_PHANTOMCHANNEL_FORCE_1M": "n",
            "CONFIG_BT_AUTO_PHY_UPDATE": "n",
        },
        "flash": {"board": "nrf52840dk", "role": "central-calibration-1m"},
    },
    "conn_central_8b": {
        "traffic": "connection",
        "condition": "central-side-8B",
        "sample": "phantomchannel_central_gatt_write",
        "source": "bluetooth/phantomchannel_central_gatt_write",
        "board": "nrf52840dk/nrf52840",
        "overlay": "conn_central_8b.conf",
        "config": {
            "CONFIG_PHANTOMCHANNEL_CENTRAL_TX": "y",
            "CONFIG_PHANTOMCHANNEL_COVERT_LEN": "8",
            "CONFIG_PHANTOMCHANNEL_WRITE_INTERVAL_MS": "20",
            "CONFIG_PHANTOMCHANNEL_EMBED_ENABLE": "y",
            "CONFIG_PHANTOMCHANNEL_DYNAMIC_TIMING": "y",
            "CONFIG_PHANTOMCHANNEL_FORCE_2M": "y",
        },
        "flash": {"board": "nrf52840dk", "role": "central"},
    },
    "conn_sink_52833": {
        "traffic": "connection-support",
        "condition": "sink",
        "sample": "phantomchannel_sink_52833",
        "source": "bluetooth/phantomchannel_sink_52833",
        "board": "nrf52833dk/nrf52833",
        "overlay": "conn_sink_52833.conf",
        "config": {
            "CONFIG_BT_PERIPHERAL": "y",
            "CONFIG_PHANTOMCHANNEL_EMBED_ENABLE": "n",
        },
        "flash": {"board": "nrf52833dk", "role": "gatt-sink"},
    },
    "conn_hrs_central_52833": {
        "traffic": "connection-support",
        "condition": "hrs-central-support",
        "sample": "phantomchannel_pip_hrs_central_52833",
        "source": "bluetooth/phantomchannel_pip_hrs_central_52833",
        "board": "nrf52833dk/nrf52833",
        "overlay": "conn_hrs_central_52833.conf",
        "config": {
            "CONFIG_PHANTOMCHANNEL_HRS_TARGET_ADDRESS": "\"C0:DE:52:84:00:03\"",
            "CONFIG_PHANTOMCHANNEL_HRS_STATS_INTERVAL_MS": "1000",
            "CONFIG_PHANTOMCHANNEL_HRS_REQUEST_2M": "y",
        },
        "flash": {"board": "nrf52833dk", "role": "hrs-central-support"},
    },
}


def build_environment() -> dict[str, str]:
    env = os.environ.copy()
    env["ZEPHYR_SDK_INSTALL_DIR"] = str(ZEPHYR_SDK)
    env["PATH"] = os.pathsep.join(
        [str(TOOLCHAIN_ROOT / "usr/local/bin"), "/usr/bin", "/bin"]
    )
    env["PYTHONPATH"] = str(TOOLCHAIN_ROOT / "usr/local/lib/python3.12/site-packages")
    return env


def config_path(build_dir: Path, sample: str) -> Path:
    return build_dir / sample / "zephyr/.config"


def read_config_value(text: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}=(.*)$", text, flags=re.MULTILINE)
    if match:
        return match.group(1)
    if re.search(rf"^# {re.escape(name)} is not set$", text, flags=re.MULTILINE):
        return "n"
    return ""


def build_one(name: str, entry: dict[str, Any]) -> dict[str, Any]:
    build_dir = OUTPUT_ROOT / name
    overlay = CONFIG_ROOT / str(entry["overlay"])
    source = SDK_ROOT / "zephyr/samples" / str(entry["source"])
    if not source.is_dir():
        raise FileNotFoundError(f"sample directory not found: {source}")
    if not overlay.is_file():
        raise FileNotFoundError(f"overlay not found: {overlay}")
    command = [
        str(WEST_PYTHON),
        "-m",
        "west",
        "build",
        "-p",
        "always",
        "-b",
        str(entry["board"]),
        "-d",
        str(build_dir),
        str(source),
        "--",
        f"-DZEPHYR_SDK_INSTALL_DIR={ZEPHYR_SDK}",
        f"-DOVERLAY_CONFIG={overlay}",
    ]
    subprocess.run(command, cwd=SDK_ROOT, env=build_environment(), check=True)
    image = build_dir / "merged.hex"
    if not image.is_file() or image.stat().st_size == 0:
        raise RuntimeError(f"build produced no merged.hex: {image}")
    generated = config_path(build_dir, str(entry["sample"]))
    if not generated.is_file():
        raise RuntimeError(f"generated Kconfig not found: {generated}")
    config_text = generated.read_text(encoding="utf-8", errors="replace")
    observed: dict[str, str] = {}
    for symbol, expected in entry["config"].items():
        actual = read_config_value(config_text, symbol)
        # Kconfig may omit a symbol whose dependency is false instead of
        # emitting the usual "is not set" line.  For an expected negative
        # value, absence is still the correct resolved value.
        if not actual and expected == "n":
            actual = "n"
        observed[symbol] = actual
        if actual != expected:
            raise RuntimeError(
                f"{name}: {symbol} expected {expected!r}, generated {actual!r}"
            )
    return {
        "name": name,
        "sample": entry["sample"],
        "board": entry["board"],
        "condition": entry["condition"],
        "traffic": entry["traffic"],
        "build_dir": str(build_dir),
        "image": str(image),
        "config": observed,
        "air_tail_bytes": entry.get("air_tail_bytes"),
        "application_covert_len": entry.get("application_covert_len"),
        "flash": entry["flash"],
        "verified": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", action="append", choices=tuple(MATRIX), help="build only this image; repeatable")
    parser.add_argument("--build", action="store_true", help="compile and verify images")
    parser.add_argument("--dry-run", action="store_true", help="print the matrix without building")
    args = parser.parse_args(argv)
    selected = args.condition or list(MATRIX)
    if not args.build and not args.dry_run:
        args.dry_run = True
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "detector_roc_experiment_redesign.md §6 event-level timing detector",
        "formal_conditions": [
            "advertising/normal",
            "advertising/append-last-239B",
            "advertising/append-every-239B",
            "connection/normal",
            "connection/central-side-8B",
            "connection/peripheral-side-240B",
        ],
        "window_policy": {
            "advertising_events_per_window": 30,
            "connection_events_per_window": 50,
            "overlap": "none",
            "capture_weight": "equal",
        },
        "images": [],
    }
    if args.dry_run:
        for name in selected:
            entry = MATRIX[name]
            manifest["images"].append({
                "name": name,
                "traffic": entry["traffic"],
                "condition": entry["condition"],
                "sample": entry["sample"],
                "board": entry["board"],
                "overlay": str(CONFIG_ROOT / str(entry["overlay"])),
                "build_dir": str(OUTPUT_ROOT / name),
                "flash": entry["flash"],
                "air_tail_bytes": entry.get("air_tail_bytes"),
                "application_covert_len": entry.get("application_covert_len"),
                "verified": False,
            })
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0
    try:
        for name in selected:
            print(f"Building event-timing image: {name}", flush=True)
            manifest["images"].append(build_one(name, MATRIX[name]))
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        manifest_path = OUTPUT_ROOT / "firmware_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Verified {len(selected)} images; no board was flashed.")
        print(f"manifest={manifest_path}")
        return 0
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
