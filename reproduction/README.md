# Offline experiment reproduction

This directory contains the analysis source used by the experiment release.
The released `PhantomChannel-experiment-reproduction-2026.09.19-ae.zip`
supplies the analysis-ready data and regression references. Install Python
3.10+ and the requirements, then run from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r reproduction/requirements.txt
.venv/bin/python reproduction/run.py \
  --bundle /path/to/PhantomChannel-experiment-reproduction-2026.09.19-ae.zip \
  --output-dir artifacts/offline-reproduction
```

Choose a new output directory for each run. The command extracts the bundle,
uses this checkout's analysis scripts, and runs `reproduce_all.py` in that
workspace. The standalone ZIP also supports `python3 reproduce_all.py` after
installation of its `requirements.txt`.

## Analysis-ready experimental data

The bundle provides IQ windows, packet records, timing vectors, trial
outcomes and detector scores used by the quantitative evaluation. Device
identifiers, host paths and receiver serials are replaced by placeholders.
Continuous IQ and raw PCAP files can contain information about unrelated
devices; the supplied records retain the fields needed by the analyses.

The 60 m comparison invokes the bundled scorer on all 19 common-threshold
parser CSVs. PSR is the unique recovered frame count divided by 200 nominal
opportunities per capture. Goodput is recovered PC-frame bytes times eight,
divided by capture duration. The aggregate results are:

| Antenna | Runs | PSR | Goodput |
| --- | ---: | ---: | ---: |
| Default | 11 | 55.0% | 53.94 kbps |
| SAS-571 | 8 | 88.6% | 86.79 kbps |

## Outputs and paper correspondence

Outputs are inside the extracted experiment directory beneath `--output-dir`.

| Paper result | Output |
| --- | --- |
| Figure 10(a), overhead | `figure10a.pdf` |
| Figure 10(b), range | `figure10b.pdf` |
| Figure 10(c), SNR | `figure10c.pdf` |
| Figure 11, frequency | `figure11.pdf` |
| Figure 12(a), advertising gap | `figure12a.png` |
| Figure 12(b), advertising event interval | `figure12b.png` |
| Tables 3–5 | `outputs/recomputed_tables/` |
| Tail-energy detection | `outputs/tail_energy/` |
| BPL/CAF NLOS study | `outputs/auxiliary_experiments/bpl_caf_nlos_summary.csv` |
| 60 m antenna PSR and goodput | `outputs/auxiliary_experiments/antenna_60m_psr_recomputed_summary.csv` |
| 60 m per-run scoring | `outputs/auxiliary_experiments/antenna_60m_psr_recomputed_per_run.csv` |

Successful reproduction prints all `[OK]` checks, including:

```text
[OK] 60 m antenna PSR/goodput replay (default 55.0%, 53.94 kbps; SAS-571 88.6%, 86.79 kbps)
[OK] standalone reproduction
[OK] regression checks
[OK] privacy and metadata scan
```

See [DEPENDENCY_MAP.md](DEPENDENCY_MAP.md) for the input-to-output map.
Offline reproduction requires no SDR or development board. Optional hardware
evaluation is described in [the firmware guide](../firmware/images/FIRMWARE_GUIDE.md).
