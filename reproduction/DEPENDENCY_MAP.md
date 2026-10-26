# Target dependency map

This map connects each target output to its release inputs and recomputation
rule.

| Target output | Entry point | Release inputs | Recomputed quantities |
|---|---|---|---|
| `overhead.pdf` | `scripts/overhead.py` | `data/overhead/measurements.csv` | nRF CRC/whitening linear fit, DA14695 measured curves, Conn/PIP overhead |
| `range.pdf` | `scripts/range_analysis.py` | `data/range/range_psr_ber_documented.csv` | measured outdoor/office PSR and BER curve |
| `range_adv_distance.pdf` | `scripts/range_analysis.py` | 52 files in `data/range/candidates/`, `reference_psr_summary.csv` | sequence selection, pattern BER, outlier-excluded BER, packet PSR and LOS/NLOS aggregate |
| `snrresult.pdf` | `scripts/snr.py` | `data/snr_measurements.csv` | interpolation of the measured Adv/Conn/PIP SNR-BER points |
| `freq_nocte.pdf` | `scripts/frequency.py` | 14 files in `data/frequency/` | Welch PSD, spectrum panels, Adv/Data/PIP cosine similarity |
| `gap_median_compact_single.png` | `scripts/advertising_timing.py` | `data/advertising_timing/packets/*.csv` | per-group packet gaps, medians, benign population distribution |
| `advevent_median_compact_single.png` | `scripts/advertising_timing.py` | same timing packet tables | 10 ms event clustering, event-start intervals and medians |
| `outputs/tail_energy/advertising_formal_roc.csv` | `scripts/tail_energy.py` | `data/tail_energy/advertising/formal_main/boundary_packet_features.csv` and the retained ROC report | formal advertising tail-energy AUC, TPR@5% FPR and operating point by gain |
| `outputs/tail_energy/advertising_tail_sensitivity.csv` | `scripts/tail_energy.py` | four retained tail-length ROC reports under `data/tail_energy/advertising/tail_sensitivity/` | AUC and TPR@5% FPR for 8/16/64/128-byte tails |
| `outputs/tail_energy/connected_pip_auc.csv` | `scripts/tail_energy.py` | `data/tail_energy/connected/final/pip_boundary_auc.csv` | connected direct/PIP tail-energy AUC rows and TPR@5% FPR |
| `outputs/tail_energy/connected_session_oof.csv` | `scripts/tail_energy.py` | `data/tail_energy/connected/session_level_internal_validation/validation_oof_scores.csv` and `validation_summary.json` | five-fold session-level direct-tail and PIP outer-boundary AUC/TPR |
| `outputs/auxiliary_experiments/bpl_caf_nlos_summary.csv` | `scripts/auxiliary_experiments.py` plus bundled BPL method | `data/auxiliary_experiments/bpl_caf_nlos_formal_records_20260919/formal_result/{stage_events,bpl_candidates,candidate_events}.csv` | formal NLOS BPL/CAF stage funnel and exact-rate rows |
| `outputs/auxiliary_experiments/antenna_60m_directional_omni_summary.csv` | `scripts/auxiliary_experiments.py` | `data/auxiliary_experiments/antenna_60m_directional_omni_records_20260919_with_psr_scripts/summary/antenna_peak_by_model.csv` plus common-threshold parser CSVs | default-versus-SAS-571 burst-peak/SNR, PSR and goodput |
| `outputs/auxiliary_experiments/antenna_60m_psr_recomputed_summary.csv` | `scripts/auxiliary_experiments.py` and bundled scorer | `data/auxiliary_experiments/antenna_60m_directional_omni_records_20260919_with_psr_scripts/runs/*/results/directional_common_thr_0p015/thr_0p015/parser/ble_packets.csv` | per-antenna PSR and end-to-end goodput recomputed from all 19 parser records |

## Table calculations

| Paper label | Output | Source in release | Status |
| --- | --- | --- | --- |
| `tab:bandwidth_goodput` | `outputs/recomputed_tables/table3_bandwidth.csv` | `data/recomputed_tables/table3/bandwidth_per_run.csv` | Mean/std across three X310 runs |
| `tab:ablation` | `outputs/recomputed_tables/table4_ablation.csv` | `data/recomputed_tables/table4/ablation_results.csv` | Four BPL/CAF ablation rows |
| `tab:conn_timing` | `outputs/recomputed_tables/table5_conn_timing.csv` | `data/recomputed_tables/table5/*_samples_longform.csv` | Mean/median from connected-state timing vectors |

Run `python3 scripts/recompute_tables.py` to regenerate these outputs.

Run `python3 scripts/tail_energy.py` to regenerate the tail-energy outputs and
copy the retained result plots into `outputs/tail_energy/plots/`.

The timing tables retain the fields actually consumed by the source analyzer:
relative time, anonymous group, packet counter, delta time, channel, PHY,
length and packet type. `advertising_timing.py` keeps a synthetic numerical
anchor only to reproduce the source code's epoch-float subtraction path; it is
not a captured wall-clock timestamp. `checks/reference_timing_sample_checks.csv`
stores packet/event counts and complete `T_gap`/`T_advEvent` vector digests at
1-us precision. `checks/reference_range_metrics.csv` stores the 52 per-run
scorer references used by the metric regression.

## Data package

The archive contains analysis-ready auxiliary BPL/CAF and antenna records in
addition to the six target figures and tail-energy inputs. Stable labels and
relative timing provide a portable interface for the offline analysis.
