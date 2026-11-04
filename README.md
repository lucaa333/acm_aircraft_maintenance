# Discrete Event Simulation for Aircraft Maintenance

This project compares analytical queueing models with a discrete-event simulation (DES) of an aircraft-maintenance hangar. The focus is on priority handling with preemption, coupled resources, escalation, and the limits of analytical queueing models.

## Start Here

Run the complete pipeline from the repository root:

```bash
bash run_project.sh          # fast tests, then every analysis
bash run_project.sh --full   # also the slow long-horizon verification tests
```

It creates a local `.venv` if needed, installs the pinned dependencies and the project itself (editable), runs the tests, executes every analysis once and writes the results to `outputs/results/`. `outputs/MANIFEST.md` maps every output file to its content and script.

## Requirements

- Python `>= 3.12`, tested with `3.14`
- `bash`
- Python packages pinned in `requirements.txt`; its last line installs this project with `pip install -e .`

Manual setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 run_project.py            # add --skip-tests or --full
python3 -m pytest -m "not slow"   # tests only
```

## Project Structure

```text
.
├── config/scenarios/stable.json   model input, the paper configuration
├── outputs/
│   ├── MANIFEST.md                which file holds which result
│   └── results/                   all generated results
├── scripts/
│   ├── analysis/                  verification, variants, sweeps, checks
│   └── reproduction/              baseline tables, DES baseline, load sensitivity
├── src/desim/
│   ├── cli.py                     run one model or all models for a scenario
│   ├── domain/                    scenario, distributions, result fields
│   ├── experiments/               model registry, replications, pipeline paths
│   ├── models/analytical/         single-class models (M/M/c, M/G/c)
│   ├── models/priority_queue/     preemptive-resume priority models
│   ├── models/simulation/         the DES
│   └── output/                    tables for the console and CSV export
├── tests/
├── run_project.py                 pipeline orchestration
└── run_project.sh                 one-command start
```

## Model

Two task classes compete for two coupled resources:

| Parameter | Value |
| --- | ---: |
| Bays (hangar slots) | `2` |
| Engineers | `15` |
| PM: arrival rate, engineers, duration | `0.08 /h`, `6`, normal (5 h, sd 1 h), cut at 0.001 h |
| CM: arrival rate, engineers, duration | `0.03 /h`, `8`, triangular (`min_value` 1.5, `mode` 3.5, `max_value` 6.0 h), mean 11/3 h |
| Escalation threshold X | `14.0 h` |
| Set-up time after a preemption | `0.5 h` |
| Baseline DES | 30 replications of 5000 h, the first 100 h are warm-up |

A task starts only if a bay and enough engineers are free at the same time. Because 8 + 8 > 15, two CM tasks never run together, so a bay can stay idle while a CM waits.

**Dispatching.** There are two FIFO queues: HP holds CM tasks and escalated PM tasks, LP holds never-escalated PM tasks. After every event only the head of the HP queue is considered; if it cannot start, nothing starts. Only an empty HP queue lets the head of the LP queue start. There is no backfilling.

**Escalation.** A PM task still waiting at arrival + X moves to the end of the HP queue. Tasks in service do not escalate. An escalated PM keeps its duration and its 6 engineers. `escalation.queue_position` selects this rule (`fifo`) or the alternatives `front` (ahead of waiting CM) and `behind_cm` (behind every CM, also later ones); `run_escalation_variants.py` compares them.

**Preemption (preemptive resume).** If the HP head cannot start, it interrupts a running never-escalated PM task, but only if that lets it start. The interrupted aircraft leaves its bay and releases its engineers, keeps its remaining work, and needs `preemption_setup_time` more hours when it resumes. It waits at the front of the LP queue; if its deadline (arrival + X) has already passed, it moves to the end of the HP queue at once and counts as escalated. If several PM tasks could be interrupted, the interrupting task takes one whose deadline has not yet passed, and among those the one started last; a PM past its deadline is interrupted only if there is no other. A preemption by an escalated PM that was itself interrupted in the same instant counts as a cascade (`preemption_cascades`). CM tasks and escalated PM tasks are never interrupted. A CM waiting behind a running CM cannot be helped by preemption.

The order of first starts is arrival time plus an allowance, 0 for CM and X for PM. A very large X is strict priority. X = 0 is not FCFS: every waiting PM escalates at once and may interrupt PM tasks that started without waiting.

```json
"priority": {
  "mode": "priority",
  "escalate_threshold": 14.0,
  "preemption_setup_time": 0.5
},
"escalation": {"queue_position": "fifo"}
```

`mode: "fcfs"` serves all tasks in one queue in arrival order, without escalation or preemption. It is used only for verification.

All replications use common random numbers: each replication draws from one stream (`Random(42 + i)`), only at arrivals, so variants with the same arrival rates process the same tasks.

## Analytical Models

| Model | Method |
| --- | --- |
| M/M/1, M/M/2, single class | Erlang C with one exponential of the pooled mean |
| M/G/1, single class | Pollaczek-Khinchine with the variance of the class mixture |
| M/G/2, single class | Erlang C times (1 + CV²)/2, label `whitt-eq70` |
| M/M/1, M/G/1 priority | exact preemptive-resume result |
| M/M/2 priority | exact, the model of Gail, Hantler and Taylor solved as a quasi-birth-death process |
| M/G/2 priority | approximation of Bondi and Buzen |

HP corresponds to CM, LP to PM. The waiting time of a class is its time in the system minus its mean service time, so it includes waiting to resume. No analytical model represents the coupling of bays and engineers, escalation or the set-up time.

## Main Metrics

All class statistics use the origin class: an escalated PM stays a PM in the reports.

| Metric | Meaning |
| --- | --- |
| `waiting_time_total` | Time in the system not in service, over all tasks; includes waiting to resume, the set-up time counts as service |
| `waiting_time_hp`, `queue_length_hp` | Original CM tasks (DES) or the HP class (analytical) |
| `waiting_time_lp`, `queue_length_lp` | All PM tasks (DES) or the LP class (analytical) |
| `*_pm_regular`, `*_pm_escalated` | Never-escalated and escalated PM tasks (DES) |
| `mttr_cm` | Arrival-to-completion time of original CM tasks |
| `escalation_rate` | Share of PM tasks that escalate |
| `hangar_utilization`, `mechanics_utilization` | Time-averaged busy bays and engineers, set-up time included; for the analytical models rho per server |
| `preemption_rate` | Share of PM tasks interrupted at least once |
| `preemptions_total`, `preemptions_by_cm`, `preemptions_by_escalated_pm` | Interruptions after warm-up, by trigger |
| `preemptions_after_deadline` | Interrupted PM tasks already past their deadline |
| `preemption_cascades` | Interruptions by an escalated PM that was itself interrupted in the same instant |
| `setup_hours` | Set-up time added by interruptions |
| `blocked_bay_share`, `blocked_cm_behind_cm_share` | Share of time a bay is free while tasks wait, and the part where a CM waits behind a running CM |

Customer averages use tasks that arrive after the warm-up and finish before the horizon; time averages use the window from the warm-up to the horizon.

## Reproduction Pipeline

`run_project.py` runs the tests, then these scripts once each. Each script writes into its own folder under `outputs/results/`.

| Step | Script | Output folder |
| --- | --- | --- |
| 1 | `scripts/reproduction/generate_baseline_results.py` | `baseline/` |
| 2 | `scripts/analysis/run_verification.py` | `verification/` |
| 3 | `scripts/reproduction/high_precision_evaluation.py` | `high_precision/` |
| 4 | `scripts/analysis/run_analytical_vs_des.py` | `analytical_vs_des/` |
| 5 | `scripts/reproduction/run_high_precision_target_rho_sensitivity_analysis.py` | `target_rho_sensitivity/` |
| 6 | `scripts/analysis/run_ablation.py` | `ablation/` |
| 7 | `scripts/analysis/run_capacity_variants.py` | `capacity_variants/` |
| 8 | `scripts/analysis/run_escalation_variants.py` | `escalation_variants/` |
| 9 | `scripts/analysis/run_threshold_sweep.py` | `threshold_sweep/` |
| 10 | `scripts/analysis/run_horizon_check.py` | `horizon_check/` |
| 11 | `scripts/analysis/simulation_configuration_analysis.py` | `validation/` |
| 12 | `scripts/analysis/run_mser5.py`, `scripts/analysis/run_warmup_check.py` | `validation/` |

The pipeline stops on errors, including a broken runtime invariant. Failed plausibility checks are logged as warnings. A second run changes no file under `outputs/`.

## Manual Runs

```bash
python3 -m desim.cli --scenario config/scenarios/stable.json --model all
python3 -m desim.cli --scenario config/scenarios/stable.json --model mg2_priority
```

Model keys: `mm1_single`, `mm2_single`, `mg1_single`, `mg2_single`, `mm1_priority`, `mm2_priority`, `mg1_priority`, `mg2_priority`, `des`.
