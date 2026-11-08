# PhantomChannel artifact guide

This is the maintained entry point linked from the Artifact Appendix. It
collects setup, execution, output interpretation, and compatibility information.
Environment and platform details can be updated here and in the linked guides
while the paper appendix retains a compact workflow description.

## Choose a workflow and record its version

| Workflow | Code and inputs | Entry point |
| --- | --- | --- |
| Archived offline replay | Source and experiment bundles deposited with the paper | [Permanent Zenodo record](https://doi.org/10.5281/zenodo.22842761) and the instructions below |
| Maintained analysis | This repository's scripts with the released experiment inputs | [Reproduction README](../reproduction/README.md) and `reproduction/run.py` |
| Optional live evaluation | Supported development boards, firmware and receiver runtime | [Firmware guide](../firmware/images/FIRMWARE_GUIDE.md), [receiver README](../receiver/README.md), and [setup notes](setup.md) |

The DOI identifies an archived snapshot. A repository checkout may contain
maintenance changes or new analyses; it is not a replacement for identifying
the archived inputs. Save archive checksums, the commit used for supplemental
files, installed dependencies and generated outputs with every run. When code,
input mapping, scoring or parameters change, document the change and its version
instead of silently treating new results as the archived baseline.

## Environment and installation

Start with the [environment guide](../reproduction/ENVIRONMENT.md). It records
the tested Python/OS combination, exact analysis dependencies, installation
commands, and commands for capturing the actual environment. The maintained
analysis pins are in [requirements-verified.txt](../reproduction/requirements-verified.txt).
Source tests have a separate [requirements-test.txt](../requirements-test.txt).
Keep their virtual environments separate.

For the archived path:

1. Download the source and experiment ZIPs, the record README, and `SHA256SUMS`
   from the Zenodo record. Verify downloaded files against that manifest.
2. Extract each ZIP into its own directory. Use the source bundle's
   `requirements-test.txt` for source checks.
3. Follow the environment guide to copy the supplemental analysis lock into
   the experiment root and install it in that directory's virtual environment.
4. Run `pip check` in the analysis environment and retain an installed-package
   manifest. Dependency installation needs package downloads or a compatible
   wheel cache; subsequent offline replay uses bundled inputs.

SDKs, UHD and CUDA are not required for offline replay. The live setup has
additional hardware and runtime requirements; consult the platform guides
before flashing or transmitting.

## Basic checks

In the source root with its test environment:

```bash
.venv/bin/python -m pytest -q tests
```

Successful execution exits without failed tests. The number of tests can differ
between an archive and a maintained checkout.

In the experiment root with its analysis environment:

```bash
.venv/bin/python scripts/recompute_tables.py
```

Successful execution produces CSVs under `outputs/recomputed_tables/`. These
are readiness checks, not independent validation of all historical paper labels.

## Full offline replay

In the extracted experiment root:

```bash
.venv/bin/python reproduce_all.py
```

Expect normal exit, `[OK]` messages, plots and CSV/JSON summaries. Outputs are
written inside that experiment directory. The archive's `DEPENDENCY_MAP.md`
maps inputs to results. The repository also provides a
[browsable dependency map](../reproduction/DEPENDENCY_MAP.md).

As a planning allowance, reserve about half an hour for initial setup, a few
person-minutes for inspection and several compute-minutes for each experiment
group on a conventional workstation. These estimates exclude package downloads
and vary with host performance. Hardware acquisition is a separate activity.

## Receiver and auxiliary replay (E1)

Inspect the overhead, range and SNR plots, `outputs/recomputed_tables/`, and
`outputs/auxiliary_experiments/`. Bandwidth results aggregate supplied run-level
records. The ablation replay covers the Data portion of Table 4; the archived
bundle does not independently regenerate advertising-ablation counts.

Keep these measures separate:

| Measure | Interpretation |
| --- | --- |
| Candidate recovery | Unique sequence representatives divided by nominal opportunities under the recorded scoring rule |
| Full-frame rate | Recovered complete frame bytes over capture duration, including framing; integrity acceptance is not necessarily required |
| Correct useful-payload rate | Known-pattern agreement plus integrity acceptance, excluding framing and the fixed marker, over capture duration |

The archived scorer selects representatives among duplicate sequences. The
optional [antenna correctness helper](../appendix/verify_antenna_correctness.py)
uses that same selection and reads the replay's generated candidate records.
From a repository checkout, invoke it after the archived replay:

```bash
python3 appendix/verify_antenna_correctness.py /path/to/extracted-experiment
```

The helper only reads inputs and prints a JSON summary. Nominal opportunities
are not independent transmitter ground truth. This stricter correctness check
applies to the antenna records and does not redefine every bandwidth/range
experiment's scoring.

## Timing replay (E2)

Inspect the advertising timing plots and
`outputs/recomputed_tables/table5_conn_timing.csv`. Tables alone use the basic
table command above. Preserve packet grouping, timing vectors and the archived
eligibility/filtering rules.

Use the `Input_dataset` column to trace output conditions. The archive's
short-tail output alias maps to a differently named input dataset; the label
alone does not certify transmitted payload length. Summaries describe the
retained selected samples. Reconstructing historical timing means or confirming
the transmitter configuration requires its own provenance evidence.

## Spectral and detector replay (E3)

For spectral metrics, inspect `outputs/frequency_similarity.csv` and
`outputs/frequency_pip_pairwise.csv`. Similarity uses Welch PSDs converted to dB
and independently min-max normalized before cosine comparison. PIP evidence is
the cross-group pairwise statistic using the archived normal/PIP group labels.
The archived spectrum plot's PIP panel reuses Data inputs; that illustrative
panel is not a validated PIP overlay and must not be used to substantiate the
aggregate PIP statistic. A corrected mapping or newly plotted figure should
identify its inputs and code revision.

For detector-only execution in the experiment root:

```bash
.venv/bin/python scripts/tail_energy.py
```

Inspect the advertising feature-based ROC replay and connected direct-tail/PIP
outer-boundary held-out scores under `outputs/tail_energy/`. The separate
`connected_pip_auc.csv` is an archived diagnostic summary, not a new packet-score
reconstruction. Real-boundary diagnostics must retain their guard/window,
boundary mapping, sample eligibility and validation context. The archived
workflow does not independently recover the missing real-boundary score chain.

## Platform support and updates

Use the [firmware guide](../firmware/images/FIRMWARE_GUIDE.md) for board/image
selection, build inputs, programming commands and expected acknowledgements.
Receiver operation is described in the [receiver README](../receiver/README.md)
and [broadcast guide](broadcast-realtime.md). Source notices and vendor terms
are in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

Update tested dependency combinations and lock files together. Keep archived
and maintained-workflow instructions distinguishable, document revised input
or metric mappings, and retain previous run manifests for comparisons.
Questions and replication reports can be filed in the
[issue tracker](https://github.com/Tincyu/PhantomChannel/issues).
