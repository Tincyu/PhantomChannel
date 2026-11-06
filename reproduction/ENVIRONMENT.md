# Verified offline analysis environment

The artifact appendix describes replay of the archived experiment ZIP. The ZIP
fixes the replay inputs and scripts; the repository supplies the supplemental
environment files. Record the commit used for these files. The repository's
`run.py` is a separate entry point that uses the checkout's analysis scripts.

## Tested platform and dependencies

- Ubuntu 24.04.4 LTS, x86-64
- CPython 3.12.3
- NumPy 2.5.4, SciPy 1.18.1, Matplotlib 3.11.2
- Exact installed analysis package versions: `requirements-verified.txt`

The lock file pins the complete analysis environment used for replay. Keep the
source-test environment separate and install its `requirements-test.txt`.
Hardware SDKs, CUDA and UHD are not needed for offline replay.

## Archived-package replay

1. Download the source and experiment ZIPs ending in `2026.09.19-ae` from
   [the archived artifact](https://doi.org/10.5281/zenodo.22842761), together
   with the record's `SHA256SUMS`.
2. In the download directory, run `sha256sum --ignore-missing -c SHA256SUMS`.
   Each downloaded file listed in that manifest must pass.
3. Extract each ZIP into its own directory. Copy this directory's
   `requirements-verified.txt` into the extracted experiment root.
4. From that experiment root, run:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-verified.txt
.venv/bin/python -m pip check
.venv/bin/python reproduce_all.py
```

Installation requires access to package downloads or a compatible wheel cache.
Once dependencies are installed, the replay uses only bundled inputs. Success
means normal exit, `[OK]` checks and generated output files. Inspect the CSVs
and dependency map to interpret each result; regression agreement with the
archive is not proof of agreement with every historical paper label.

## Capture the actual run environment

Run these commands in the experiment root, using its virtual environment:

```bash
mkdir -p environment-record
.venv/bin/python --version > environment-record/python-version.txt
.venv/bin/python -m pip freeze > environment-record/requirements.txt
.venv/bin/python -m pip check > environment-record/pip-check.txt
uname -srmo > environment-record/platform.txt
cat /etc/os-release > environment-record/os-release.txt
```

Record downloaded archive checksums and the supplemental-file repository
commit alongside this directory. Do not include hostnames or device serials
when sharing environment records.

## Inspection scope

- Frequency validation uses `outputs/frequency_similarity.csv` and the PIP
  pairwise matrix. A spectrum overlay does not establish the aggregate PIP
  metric or authenticate its input mapping.
- Candidate recovery, full-frame rate and correct useful-payload delivery are
  distinct measures. The appendix's antenna helper applies known-pattern and
  integrity checks to the archived scorer's selected sequence representatives.
- Table 4 replay covers the Data portion; it does not reconstruct advertising
  ablation counts.
- Table 5 summaries describe retained timing vectors. Preserve `Input_dataset`
  and selection rules; a condition alias alone is not firmware-length evidence.
- Advertising feature replay and direct-tail/PIP outer-boundary held-out scores
  are executable evidence. Real-boundary tables are archived diagnostics;
  independently recovering their packet scores and boundary mappings is outside
  the appendix's reproduction claim.

Changing analysis code, inputs, parameter selection, or condition mapping should
be documented as a new analysis rather than silently replacing archived replay.
