#!/usr/bin/env python3
"""Run the complete standalone artifact reproduction."""

from __future__ import annotations

import shutil
import struct
from pathlib import Path

import matplotlib

matplotlib.set_loglevel("error")

from scripts import advertising_timing, auxiliary_experiments, frequency, overhead, range_analysis, recompute_tables, regression, snr, tail_energy, validate


NUMBERED_FIGURES = {
    "figure10a.pdf": "overhead.pdf",
    "figure10b.pdf": "range.pdf",
    "figure10c.pdf": "snrresult.pdf",
    "figure11.pdf": "freq_nocte.pdf",
    "figure12a.png": "gap_median_compact_single.png",
    "figure12b.png": "advevent_median_compact_single.png",
}


def verify_plot_artifacts(root: Path, aliases: list[str]) -> dict[str, int]:
    for name in aliases:
        root_file = root / name
        plot_file = root / "plots" / name
        if not root_file.is_file() or root_file.stat().st_size == 0:
            raise AssertionError(f"missing or empty target output: {root_file}")
        if root_file.read_bytes() != plot_file.read_bytes():
            raise AssertionError(f"root/plots output differs: {name}")
        if name.endswith(".pdf") and root_file.read_bytes()[:5] != b"%PDF-":
            raise AssertionError(f"invalid PDF signature: {name}")
        if name.endswith(".png"):
            header = root_file.read_bytes()[:24]
            if header[:8] != b"\x89PNG\r\n\x1a\n":
                raise AssertionError(f"invalid PNG signature: {name}")
            width, height = struct.unpack(">II", header[16:24])
            if (width, height) != (1035, 870):
                raise AssertionError(f"unexpected PNG geometry {name}: {(width, height)}")
    return {"required_files": len(aliases), "png_geometry_checks": sum(name.endswith(".png") for name in aliases)}


def main() -> None:
    root = Path(__file__).resolve().parent
    (root / "outputs").mkdir(exist_ok=True)
    validate_result = validate.run(root)
    overhead.run(root)
    snr.run(root)
    frequency.run(root)
    range_analysis.run(root)
    advertising_timing.run(root)
    table_result = recompute_tables.main()
    tail_energy_result = tail_energy.run(root)
    auxiliary_result = auxiliary_experiments.run(root)
    regression_result = regression.run(root)

    aliases = ["overhead.pdf", "range.pdf", "snrresult.pdf", "freq_nocte.pdf", "gap_median_compact_single.png", "advevent_median_compact_single.png"]
    for name in aliases:
        source = root / "plots" / name
        if not source.exists():
            raise RuntimeError(f"missing target output: {source}")
        shutil.copy2(source, root / name)
    for numbered_name, source_name in NUMBERED_FIGURES.items():
        source = root / "plots" / source_name
        if not source.exists():
            raise RuntimeError(f"missing numbered figure source: {source}")
        shutil.copy2(source, root / numbered_name)
        shutil.copy2(source, root / "plots" / numbered_name)

    print("[OK] overhead analysis and plot")
    print("[OK] range analysis and plot")
    print("[OK] SNR analysis and plot")
    print("[OK] frequency analysis from NPY")
    print("[OK] advertising gap from filtered packets")
    print("[OK] advertising event interval from filtered packets")
    print("[OK] recomputed Table 3, Table 4 and Table 5 from analysis-ready inputs")
    print("[OK] tail-energy detection analysis")
    # auxiliary_experiments.run() emits the two auxiliary [OK] lines.
    plot_result = verify_plot_artifacts(root, aliases + list(NUMBERED_FIGURES))
    print("[OK] numbered paper figures")
    print("[OK] standalone reproduction")
    print("[OK] graph-level plot checks")
    print("[OK] regression checks")
    print("[OK] privacy and metadata scan")
    print({"data": validate_result, "tables": table_result, "tail_energy": tail_energy_result, "auxiliary_experiments": auxiliary_result, "regression": regression_result, "plots": plot_result})


if __name__ == "__main__":
    main()
