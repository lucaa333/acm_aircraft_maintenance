# Output manifest

`bash run_project.sh --full` creates every file below in one pass. Paths are relative to `outputs/results/`; no other files are written there. Tables and figures are named by content, not by number, because the numbering of the paper changes between versions. Section numbers `§` refer to `UEBERGABE_PRAEEMPTION.md`.

Common settings: paper configuration from `config/scenarios/stable.json` (preemptive resume, 0.5 h set-up time, X = 14 h, escalated PM at the end of the HP queue). Seeds 42 + i, 30 replications unless stated otherwise, 95 % Student-t intervals, not capped at zero. Horizons include the 100 h warm-up.

## Analytical models

| Content | Output file | Script | Configuration |
| --- | --- | --- | --- |
| Single-class models M/M/1, M/M/2, M/G/1, M/G/2 (§5.1) | `baseline/stable_aircraft_maintenance_all_results.csv`, rows `Single Priority` | `scripts/reproduction/generate_baseline_results.py` | baseline rates |
| Preemptive-resume priority models (§5.2) | `baseline/stable_aircraft_maintenance_all_results.csv`, rows `Multi Priority` | same | baseline rates |
| The M/G/1 models alone | `baseline/stable_aircraft_maintenance_mg1_single_results.csv`, `baseline/stable_aircraft_maintenance_mg1_priority_results.csv` | same | baseline rates |
| Input quantities per class and pooled: lambda, E[S], sd, E[S²], Var, CV² | `verification/analytical_inputs.csv` | `scripts/analysis/run_verification.py` | baseline rates |

## Verification

| Content | Output file | Script | Configuration |
| --- | --- | --- | --- |
| DES against exact references and approximations, coupling, escalation and set-up time off (§5.3) | `verification/verification.csv` | `scripts/analysis/run_verification.py` | 30 x 5000 h and 30 x 100000 h, 16 engineers |
| Exact M/M/2 preemptive model against the Bondi-Buzen approximation (§5.3) | `verification/mm2_priority_exact_vs_approximation.csv` | same | analytical, baseline and rho 0.50 to 0.90 |

## DES baseline

| Content | Output file | Script | Configuration |
| --- | --- | --- | --- |
| Baseline KPIs with intervals (§5.4) | `high_precision/confidence_intervals.csv` | `scripts/reproduction/high_precision_evaluation.py` | 30 x 5000 h |
| Baseline KPIs, distribution per KPI (min, quartiles, max) | `high_precision/summary_statistics.csv` | same | 30 x 5000 h |
| Baseline KPIs, one row with all means and intervals | `high_precision/aggregated_kpi_results.csv` | same | 30 x 5000 h |
| Baseline KPIs per replication | `high_precision/raw_replication_results.csv` | same | 30 x 5000 h |
| Escalations summed over all replications | `high_precision/escalation_counts.csv` | same | 30 x 5000 h |

## Analytical models against the DES

| Content | Output file | Script | Configuration |
| --- | --- | --- | --- |
| M/G/2 FCFS and M/G/2 preemptive against four DES variants, per load (§5.5, §5.7) | `analytical_vs_des/analytical_vs_des.csv` | `scripts/analysis/run_analytical_vs_des.py` | baseline 30 x 5000 h, rho 0.25 to 0.90 30 x 50000 h |

## Load sensitivity

| Content | Output file | Script | Configuration |
| --- | --- | --- | --- |
| All KPIs with intervals per target rho (§5.6) | `target_rho_sensitivity/confidence_intervals_all_kpis.csv` | `scripts/reproduction/run_high_precision_target_rho_sensitivity_analysis.py` | rho 0.25, 0.50, 0.75, 0.85, 0.90, 30 x 50000 h |
| Means per target rho, escalations summed | `target_rho_sensitivity/aggregated_scenario_summaries.csv` | same | same |
| KPIs per replication | `target_rho_sensitivity/raw_replication_results.csv` | same | same |
| Arrival rates and theoretical utilizations per target rho | `target_rho_sensitivity/scenario_configurations.csv`, `target_rho_sensitivity/theoretical_rho_values.csv` | same | same |
| Run settings | `target_rho_sensitivity/run_configuration.json`, `target_rho_sensitivity/run_metadata.txt` | same | same |
| Figure: waiting time over load (total, CM, PM) | `target_rho_sensitivity/waiting_time_by_target_rho.png` | same | same |
| Figure: queue length over load | `target_rho_sensitivity/queue_length_by_target_rho.png` | same | same |
| Figure: MTTR over load | `target_rho_sensitivity/mttr_by_target_rho.png` | same | same |
| Figure: bay and engineer utilization over load | `target_rho_sensitivity/resource_utilization_by_target_rho.png` | same | same |
| Figure: escalation rate over load | `target_rho_sensitivity/escalation_rate_by_target_rho.png` | same | same |

## Variants

| Content | Output file | Script | Configuration |
| --- | --- | --- | --- |
| Ablation 15/16 engineers x with/without escalation, paired differences (§5.8) | `ablation/ablation_2x2.csv` | `scripts/analysis/run_ablation.py` | baseline 30 x 5000 h, rho 0.75 and 0.90 30 x 50000 h |
| 16th engineer and third bay, paired differences (§5.11) | `capacity_variants/capacity_variants.csv` | `scripts/analysis/run_capacity_variants.py` | baseline, 30 x 5000 h |
| Placement of escalated PM: end of HP queue, front, behind every CM, none (§5.14, §5.15) | `escalation_variants/escalation_variants.csv` | `scripts/analysis/run_escalation_variants.py` | rho 0.25 to 0.90, 30 x 50000 h |
| Escalation threshold X from 0 h to none, with blocked bay time, preemption rate and cascades (§5.9, §5.15) | `threshold_sweep/threshold_sweep.csv` | `scripts/analysis/run_threshold_sweep.py` | baseline 30 x 5000 h, rho 0.50 to 0.90 30 x 50000 h |
| Figure: Wq total, CM and PM over X with 95 % bands | `threshold_sweep/threshold_sweep.png` | same | rho 0.75, 0.85, 0.90 |

## Warm-up, run length and horizon

| Content | Output file | Script | Configuration |
| --- | --- | --- | --- |
| Horizons 5000, 50000, 200000 h with paired differences against 50000 h (§5.10) | `horizon_check/horizon_check.csv` | `scripts/analysis/run_horizon_check.py` | rho 0.75, 0.85, 0.90, 30 replications |
| Warm-up 0 to 1,000 h with a fixed 4,000 h window (§5.12) | `validation/warmup_analysis.csv` | `scripts/analysis/simulation_configuration_analysis.py` | baseline, 300 replications |
| Figure: KPIs over the warm-up | `validation/warmup_analysis.png` | same | same |
| Run length up to 10000 h (§5.12) | `validation/run_length_analysis.csv` | same | baseline, 300 replications |
| Figure: KPIs over the run length | `validation/run_length_analysis.png` | same | same |
| MSER-5 truncation point and Wq with that warm-up (§5.13) | `validation/mser5.csv` | `scripts/analysis/run_mser5.py` | baseline and rho 0.90, 30 x 50000 h from the empty system |
| Warm-up 100, 2000, 5000 h paired against 100 h | `validation/warmup_check_high_load.csv` | `scripts/analysis/run_warmup_check.py` | rho 0.90, 30 x 50000 h |
