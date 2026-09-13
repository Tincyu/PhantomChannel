#!/usr/bin/env python3
"""Build the matched 2M firmware set for the formal PIP AUC experiment.

The historical acceptance build directory is deliberately not reused: its
generated configuration was later used for other PIP variants.  This tool
builds four immutable, condition-labelled images into separate directories
and writes a manifest containing the resolved Kconfig values and SHA-256
hashes.  It never flashes a board.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from pip_program_common import PROJECT_ROOT, SDK_ROOT, build_image


OLD_HRS_SAMPLE = SDK_ROOT / "zephyr/samples/bluetooth/phantomchannel_hrs_peripheral"
PIP_HRS_SAMPLE = SDK_ROOT / "zephyr/samples/bluetooth/phantomchannel_pip_hrs_peripheral"
HRS_CENTRAL_SAMPLE = SDK_ROOT / "zephyr/samples/bluetooth/phantomchannel_pip_hrs_central_52833"
BOARD_52840 = "nrf52840dk/nrf52840"
BOARD_52833 = "nrf52833dk/nrf52833"
OUT_ROOT = PROJECT_ROOT / "artifacts/firmware/pip_auc_20260808"
PERIPHERAL_ADDRESS = "C0:DE:52:84:00:41"
INTERVAL_MS = 1000
COVERT_LEN = 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_variant(name: str, sample: Path, board: str, settings: list[str]) -> Path:
    output = OUT_ROOT / name
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "artifacts/pip_build_configs").mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f"pip_auc_{name}_",
        suffix=".conf",
        dir=PROJECT_ROOT / "artifacts/pip_build_configs",
        delete=False,
    ) as overlay:
        overlay_path = Path(overlay.name)
        overlay.write("\n".join(settings) + "\n")
    try:
        firmware = build_image(sample, output, board, overlay_config=overlay_path)
    finally:
        overlay_path.unlink(missing_ok=True)
    return firmware


def verify_config(build_dir: Path, expected: dict[str, str]) -> None:
    configs = list(build_dir.glob("*/zephyr/.config"))
    if len(configs) != 1:
        raise RuntimeError(f"expected one generated .config below {build_dir}, got {configs}")
    config = configs[0].read_text(encoding="utf-8")
    for key, value in expected.items():
        setting = f"CONFIG_{key}={value}"
        disabled = f"# CONFIG_{key} is not set"
        if value == "n":
            # Some Kconfig symbols disappear entirely when their parent
            # subsystem is disabled; absence is equivalent to disabled here.
            if disabled not in config and setting in config:
                raise RuntimeError(f"missing disabled setting {disabled} in {configs[0]}")
        elif setting not in config:
            raise RuntimeError(f"missing setting {setting} in {configs[0]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=OUT_ROOT / "manifest.json")
    args = parser.parse_args()

    common = [
        'CONFIG_PHANTOMCHANNEL_STATIC_ADDRESS="C0:DE:52:84:00:41"',
        f"CONFIG_PHANTOMCHANNEL_NOTIFY_INTERVAL_MS={INTERVAL_MS}",
        "CONFIG_BT_PHY_UPDATE=y",
        "CONFIG_BT_USER_PHY_UPDATE=y",
        "CONFIG_BT_AUTO_PHY_UPDATE=n",
        "CONFIG_USE_SEGGER_RTT=n",
        "CONFIG_RTT_CONSOLE=n",
        "CONFIG_UART_CONSOLE=y",
    ]
    variants = [
        (
            "benign_2m_peripheral",
            OLD_HRS_SAMPLE,
            BOARD_52840,
            common
            + [
                "CONFIG_PHANTOMCHANNEL_PERIPHERAL_TX=n",
                "CONFIG_PHANTOMCHANNEL_EMBED_ENABLE=y",
            ],
            {
                "PHANTOMCHANNEL_PERIPHERAL_TX": "n",
                "PHANTOMCHANNEL_EMBED_ENABLE": "y",
                "PHANTOMCHANNEL_NOTIFY_INTERVAL_MS": str(INTERVAL_MS),
                "BT_PHY_UPDATE": "y",
                "USE_SEGGER_RTT": "n",
                "RTT_CONSOLE": "n",
                "UART_CONSOLE": "y",
            },
            "benign",
        ),
        (
            "direct_tail_2m_peripheral",
            OLD_HRS_SAMPLE,
            BOARD_52840,
            common
            + [
                "CONFIG_PHANTOMCHANNEL_PERIPHERAL_TX=y",
                "CONFIG_PHANTOMCHANNEL_EMBED_ENABLE=y",
                f"CONFIG_PHANTOMCHANNEL_COVERT_LEN={COVERT_LEN}",
            ],
            {
                "PHANTOMCHANNEL_PERIPHERAL_TX": "y",
                "PHANTOMCHANNEL_EMBED_ENABLE": "y",
                "PHANTOMCHANNEL_COVERT_LEN": str(COVERT_LEN),
                "PHANTOMCHANNEL_NOTIFY_INTERVAL_MS": str(INTERVAL_MS),
                "BT_PHY_UPDATE": "y",
                "USE_SEGGER_RTT": "n",
                "RTT_CONSOLE": "n",
                "UART_CONSOLE": "y",
            },
            "direct_tail",
        ),
        (
            "pip_2m_peripheral",
            PIP_HRS_SAMPLE,
            BOARD_52840,
            common
            + [
                "CONFIG_PHANTOMCHANNEL_PIP_PERIPHERAL_TX=y",
                "CONFIG_PHANTOMCHANNEL_PIP_TIMING=y",
                "CONFIG_PHANTOMCHANNEL_PIP_FORCE_EVERY_NOTIFY=y",
                "CONFIG_PHANTOMCHANNEL_PIP_RELEASE_ON_TX=y",
                "CONFIG_PHANTOMCHANNEL_EMBED_ENABLE=y",
                "CONFIG_PHANTOMCHANNEL_PERIPHERAL_TX=y",
                f"CONFIG_PHANTOMCHANNEL_COVERT_LEN={COVERT_LEN}",
            ],
            {
                "PHANTOMCHANNEL_PIP_PERIPHERAL_TX": "y",
                "PHANTOMCHANNEL_PIP_FORCE_EVERY_NOTIFY": "y",
                "PHANTOMCHANNEL_PIP_RELEASE_ON_TX": "y",
                "PHANTOMCHANNEL_EMBED_ENABLE": "y",
                "PHANTOMCHANNEL_COVERT_LEN": str(COVERT_LEN),
                "PHANTOMCHANNEL_NOTIFY_INTERVAL_MS": str(INTERVAL_MS),
                "BT_PHY_UPDATE": "y",
            },
            "pip",
        ),
        (
            "hrs_2m_central_52833",
            HRS_CENTRAL_SAMPLE,
            BOARD_52833,
            [
                "CONFIG_PHANTOMCHANNEL_HRS_REQUEST_2M=y",
                "CONFIG_BT_PHY_UPDATE=y",
                "CONFIG_BT_USER_PHY_UPDATE=y",
                "CONFIG_BT_AUTO_PHY_UPDATE=n",
            ],
            {
                "PHANTOMCHANNEL_HRS_REQUEST_2M": "y",
                "BT_PHY_UPDATE": "y",
            },
            "central",
        ),
    ]

    manifest: dict[str, object] = {
        "schema_version": 1,
        "experiment": "pip_boundary_auc_20260808",
        "phy": "2m",
        "covert_len_bytes": COVERT_LEN,
        "interval_ms": INTERVAL_MS,
        "peripheral_address": PERIPHERAL_ADDRESS,
        "flashed": False,
        "variants": [],
    }
    for name, sample, board, settings, expected, condition in variants:
        firmware = build_variant(name, sample, board, settings)
        verify_config(OUT_ROOT / name, expected)
        item = {
            "name": name,
            "condition": condition,
            "sample": str(sample),
            "board": board,
            "firmware": str(firmware),
            "sha256": sha256(firmware),
            "config": expected,
        }
        manifest["variants"].append(item)  # type: ignore[union-attr]

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
