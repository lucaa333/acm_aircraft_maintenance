from __future__ import annotations

from dataclasses import replace
import math

import pandas as pd

from desim.domain.metrics import ORIGIN_FIELDS, t_critical_95
from desim.experiments.pipeline_context import RESULTS_DIR, load_base_scenario
from desim.experiments.runner import ExperimentRunner


OUTPUT_DIR = RESULTS_DIR / "high_precision"
RAW_RESULTS_CSV = OUTPUT_DIR / "raw_replication_results.csv"
AGGREGATED_RESULTS_CSV = OUTPUT_DIR / "aggregated_kpi_results.csv"
CONFIDENCE_INTERVALS_CSV = OUTPUT_DIR / "confidence_intervals.csv"
SUMMARY_STATISTICS_CSV = OUTPUT_DIR / "summary_statistics.csv"

HIGH_PRECISION_RUN_LENGTH = 5000.0
HIGH_PRECISION_WARMUP = 100.0
HIGH_PRECISION_REPLICATIONS = 30

METRICS = [
    ("waiting_time_total", "mean waiting time"),
    ("queue_length_total", "mean queue length"),
    ("waiting_time_hp", "mean waiting time, original CM"),
    ("waiting_time_lp", "mean waiting time, all PM"),
    ("queue_length_hp", "mean queue length, original CM"),
    ("queue_length_lp", "mean queue length, all PM"),
    ("mttr_cm", "mean MTTR, original CM"),
    ("waiting_time_pm_regular", "mean waiting time, never-escalated PM"),
    ("waiting_time_pm_escalated", "mean waiting time, escalated PM"),
    ("queue_length_pm_regular", "mean queue length, never-escalated PM"),
    ("queue_length_pm_escalated", "mean queue length, escalated PM"),
    ("cycle_time_pm_regular", "mean cycle time, never-escalated PM"),
    ("cycle_time_pm_escalated", "mean cycle time, escalated PM"),
    ("escalations_total", "escalations per replication"),
    ("escalation_rate", "escalation rate"),
    ("hangar_utilization", "hangar utilization"),
    ("mechanics_utilization", "mechanics utilization"),
    ("preemption_rate", "share of PM tasks preempted"),
    ("preemptions_total", "preemptions per replication"),
    ("preemptions_by_cm", "preemptions by CM per replication"),
    ("preemptions_by_escalated_pm", "preemptions by escalated PM per replication"),
    ("preemptions_after_deadline", "preempted PM past their deadline per replication"),
    ("preemption_cascades", "preemptions by an escalated PM itself preempted in that instant, per replication"),
    ("setup_hours", "set-up hours per replication"),
    ("blocked_bay_share", "share of time a bay is blocked"),
    ("blocked_cm_behind_cm_share", "share of time a CM waits behind a CM with a free bay"),
]


def build_high_precision_scenario(base_scenario):
    return replace(
        base_scenario,
        name=f"{base_scenario.name}_high_precision",
        simulation=replace(
            base_scenario.simulation,
            run_length=HIGH_PRECISION_RUN_LENGTH,
            warmup=HIGH_PRECISION_WARMUP,
            replications=HIGH_PRECISION_REPLICATIONS,
        ),
    )


def result_to_row(result, replication: int | None = None) -> dict[str, float | str | int | None]:
    row: dict[str, float | str | int | None] = {
        "model_name": result.model_name,
        "scenario_name": result.scenario_name,
        "waiting_time_total": result.waiting_time_total,
        "queue_length_total": result.queue_length_total,
        "waiting_time_hp": result.waiting_time_hp,
        "waiting_time_lp": result.waiting_time_lp,
        "queue_length_hp": result.queue_length_hp,
        "queue_length_lp": result.queue_length_lp,
        "mttr_cm": result.mttr_cm,
        "hangar_utilization": result.hangar_utilization,
        "mechanics_utilization": result.mechanics_utilization,
        "escalation_rate": result.escalation_rate,
        **{name: getattr(result, name) for name in ORIGIN_FIELDS},
    }
    if replication is not None:
        row["replication"] = replication
    row.update(result.extra)
    return row


def metric_summary(dataframe: pd.DataFrame, metric: str) -> dict[str, float]:
    series = dataframe[metric].astype(float).dropna()
    count = int(series.count())
    mean_value = float(series.mean())
    std_value = float(series.std(ddof=1)) if count > 1 else 0.0
    ci_margin = t_critical_95(count - 1) * std_value / math.sqrt(count) if count > 1 else 0.0
    return {
        "metric": metric,
        "n": count,
        "mean": mean_value,
        "std": std_value,
        "min": float(series.min()),
        "p25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "p75": float(series.quantile(0.75)),
        "max": float(series.max()),
        "ci_lower": mean_value - ci_margin,
        "ci_upper": mean_value + ci_margin,
        "relative_half_width": (ci_margin / mean_value) if mean_value else 0.0,
    }


def build_summary_tables(raw_results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    ci_rows = []
    for metric, _ in METRICS:
        summary = metric_summary(raw_results, metric)
        summary_rows.append({k: summary[k] for k in ["metric", "n", "mean", "std", "min", "p25", "median", "p75", "max"]})
        ci_rows.append({k: summary[k] for k in ["metric", "n", "mean", "std", "ci_lower", "ci_upper", "relative_half_width"]})
    return pd.DataFrame(summary_rows), pd.DataFrame(ci_rows)


def format_percent(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.2%}"


def print_results_table(ci_results: pd.DataFrame) -> None:
    print("High-precision DES experiment")
    print("=" * 72)
    for _, row in ci_results.iterrows():
        print(
            f"{row['metric']}: mean={row['mean']:.6f}, "
            f"95% CI=[{row['ci_lower']:.6f}, {row['ci_upper']:.6f}], "
            f"relative half-width={format_percent(float(row['relative_half_width']))}"
        )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_scenario = load_base_scenario()
    scenario = build_high_precision_scenario(base_scenario)

    runner = ExperimentRunner(scenario)
    model = runner.available_models()["des"]

    raw_replications = model.run_replications(scenario)
    raw_dataframe = pd.DataFrame([result_to_row(result, replication=index + 1) for index, result in enumerate(raw_replications)])
    raw_dataframe.to_csv(RAW_RESULTS_CSV, index=False)

    aggregated_result = runner.run_one("des")
    aggregated_dataframe = pd.DataFrame([result_to_row(aggregated_result)])
    aggregated_dataframe.to_csv(AGGREGATED_RESULTS_CSV, index=False)

    summary_dataframe, ci_dataframe = build_summary_tables(raw_dataframe)
    summary_dataframe.to_csv(SUMMARY_STATISTICS_CSV, index=False)
    ci_dataframe.to_csv(CONFIDENCE_INTERVALS_CSV, index=False)

    escalation_counts = raw_dataframe[["escalations_total"]].sum()
    escalation_counts.rename("sum_over_all_replications").to_csv(OUTPUT_DIR / "escalation_counts.csv", index_label="count")

    print_results_table(ci_dataframe)
    print("\nExports")
    print("=" * 72)
    print(f"Raw replication results: {RAW_RESULTS_CSV.resolve()}")
    print(f"Aggregated KPI results: {AGGREGATED_RESULTS_CSV.resolve()}")
    print(f"Confidence intervals: {CONFIDENCE_INTERVALS_CSV.resolve()}")
    print(f"Summary statistics: {SUMMARY_STATISTICS_CSV.resolve()}")


if __name__ == "__main__":
    main()
