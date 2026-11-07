from __future__ import annotations

from dataclasses import asdict, replace
import json
import math
from pathlib import Path

import pandas as pd

from desim.experiments.pipeline_context import BASE_SCENARIO_LABEL, RESULTS_DIR, load_base_scenario

OUTPUT_DIR = RESULTS_DIR / "target_rho_sensitivity"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from desim.domain.metrics import ORIGIN_FIELDS, ModelResult, t_critical_95
from desim.domain.scenario import Scenario, arrival_rates_for_target_rho
from desim.models.simulation.des_model import DiscreteEventSimulationModel


RAW_RESULTS_CSV = OUTPUT_DIR / "raw_replication_results.csv"
SUMMARY_CSV = OUTPUT_DIR / "aggregated_scenario_summaries.csv"
CONFIDENCE_INTERVALS_CSV = OUTPUT_DIR / "confidence_intervals_all_kpis.csv"
SCENARIO_CONFIG_CSV = OUTPUT_DIR / "scenario_configurations.csv"
THEORETICAL_RHO_CSV = OUTPUT_DIR / "theoretical_rho_values.csv"
RUN_CONFIG_JSON = OUTPUT_DIR / "run_configuration.json"
RUN_METADATA_TXT = OUTPUT_DIR / "run_metadata.txt"

PLOTS = {
    # The main figure of the paper: two panels, no title and no footnote inside the image.
    "waiting_and_escalation": OUTPUT_DIR / "waiting_and_escalation_by_target_rho.png",
    "waiting_time": OUTPUT_DIR / "waiting_time_by_target_rho.png",
    "queue_length": OUTPUT_DIR / "queue_length_by_target_rho.png",
    "escalation_rate": OUTPUT_DIR / "escalation_rate_by_target_rho.png",
    "mttr": OUTPUT_DIR / "mttr_by_target_rho.png",
    "resource_utilization": OUTPUT_DIR / "resource_utilization_by_target_rho.png",
}

# 5000 h is biased low at high load (rho = 0.90: 16.2 h against 21.1 h); 50000 h matches a 200000 h check.
HIGH_PRECISION_RUN_LENGTH = 50000.0
HIGH_PRECISION_WARMUP = 100.0
HIGH_PRECISION_REPLICATIONS = 30

TARGET_RHOS = [0.25, 0.50, 0.75, 0.85, 0.90]

# Categorical slots 1-3 of the validated palette; the marker shape is the second encoding.
WAITING_SERIES = [
    ("waiting_time_total", "Total", "#2a78d6", "o"),
    ("waiting_time_hp", "Original CM", "#eb6834", "s"),
    ("waiting_time_lp", "All PM", "#1baf7a", "^"),
]
ESCALATION_SERIES = ("escalation_rate", "Escalation rate", "#2a78d6", "o")
# Both columns of the ACM template: 7.0 in is the full text width, so the figure needs no scaling.
# The point sizes are set for that width, where the matplotlib defaults would come out oversized.
MAIN_FIGURE_SIZE = (7.0, 2.8)
MAIN_FIGURE_FONT = {"tick": 7, "label": 8, "legend": 7}

# The image carries no title and no footnote: this text belongs in the caption of the paper,
# see outputs/MANIFEST.md.
MAIN_FIGURE_CAPTION = (
    "Left: mean waiting time over the load, split into all tasks, the original CM tasks and all PM "
    "tasks. Right: share of PM tasks that escalate. Bands: 95 % Student-t intervals over 30 "
    "replications of 50,000 h."
)

METRICS = [
    "waiting_time_total",
    "queue_length_total",
    "waiting_time_hp",
    "waiting_time_lp",
    "queue_length_hp",
    "queue_length_lp",
    "mttr_cm",
    "escalation_rate",
    "mechanics_utilization",
    "hangar_utilization",
    *ORIGIN_FIELDS,
]


def target_rho_scenarios(base_scenario: Scenario) -> list[tuple[float, float, float]]:
    return [(rho, *arrival_rates_for_target_rho(base_scenario, rho)) for rho in TARGET_RHOS]


def rho_hangar(lambda_pm: float, lambda_cm: float, scenario: Scenario) -> float:
    return (
        lambda_pm * scenario.pm_service.moment_mean + lambda_cm * scenario.cm_service.moment_mean
    ) / scenario.resources.hangar_slots


def rho_engineers(lambda_pm: float, lambda_cm: float, scenario: Scenario) -> float:
    return (
        lambda_pm * scenario.pm_mechanics * scenario.pm_service.moment_mean
        + lambda_cm * scenario.cm_mechanics * scenario.cm_service.moment_mean
    ) / scenario.resources.mechanics


def build_scenario(base_scenario: Scenario, target_rho: float, lambda_pm: float, lambda_cm: float) -> Scenario:
    return replace(
        base_scenario,
        name=f"{base_scenario.name}_target_rho_{target_rho:.2f}",
        arrival_rate_pm=lambda_pm,
        arrival_rate_cm=lambda_cm,
        simulation=replace(
            base_scenario.simulation,
            run_length=HIGH_PRECISION_RUN_LENGTH,
            warmup=HIGH_PRECISION_WARMUP,
            replications=HIGH_PRECISION_REPLICATIONS,
        ),
    )


def result_to_row(
    scenario: Scenario,
    target_rho: float,
    result: ModelResult,
    replication: int,
    seed: int,
) -> dict[str, float | int | str | None]:
    theoretical_rho_hangar = rho_hangar(scenario.arrival_rate_pm, scenario.arrival_rate_cm, scenario)
    theoretical_rho_engineers = rho_engineers(scenario.arrival_rate_pm, scenario.arrival_rate_cm, scenario)
    return {
        "scenario_name": scenario.name,
        "target_rho": target_rho,
        "lambda_pm": scenario.arrival_rate_pm,
        "lambda_cm": scenario.arrival_rate_cm,
        "theoretical_rho_hangar": theoretical_rho_hangar,
        "theoretical_rho_engineers": theoretical_rho_engineers,
        "theoretical_rho_max": max(theoretical_rho_hangar, theoretical_rho_engineers),
        "replication": replication,
        "seed": seed,
        "run_length_hours": scenario.simulation.run_length,
        "warmup_hours": scenario.simulation.warmup,
        "replications_configured": scenario.simulation.replications,
        "model_name": result.model_name,
        "waiting_time_total": result.waiting_time_total,
        "queue_length_total": result.queue_length_total,
        "waiting_time_hp": result.waiting_time_hp,
        "waiting_time_lp": result.waiting_time_lp,
        "queue_length_hp": result.queue_length_hp,
        "queue_length_lp": result.queue_length_lp,
        "mttr_cm": result.mttr_cm,
        "escalation_rate": result.escalation_rate,
        **{name: getattr(result, name) for name in ORIGIN_FIELDS},
        "mechanics_utilization": result.mechanics_utilization,
        "hangar_utilization": result.hangar_utilization,
        "observed_rho": result.hangar_utilization,
    }


def summarize_metric(group: pd.DataFrame, metric: str) -> dict[str, float | int | str]:
    values = group[metric].astype(float).dropna()
    count = int(values.count())
    mean_value = float(values.mean())
    std_value = float(values.std(ddof=1)) if count > 1 else 0.0
    ci_margin = t_critical_95(count - 1) * std_value / math.sqrt(count) if count > 1 else 0.0
    return {
        "metric": metric,
        "n": count,
        "mean": mean_value,
        "std": std_value,
        "ci_lower": mean_value - ci_margin,
        "ci_upper": mean_value + ci_margin,
        "relative_half_width": (ci_margin / mean_value) if mean_value else 0.0,
    }


def build_summary_tables(raw_results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, float | int | str]] = []
    ci_rows: list[dict[str, float | int | str]] = []
    scenario_columns = [
        "scenario_name",
        "target_rho",
        "lambda_pm",
        "lambda_cm",
        "theoretical_rho_hangar",
        "theoretical_rho_engineers",
        "theoretical_rho_max",
    ]

    for scenario_key, group in raw_results.groupby(scenario_columns, sort=True):
        base = dict(zip(scenario_columns, scenario_key, strict=True))
        summary_rows.append(
            {
                **base,
                "observed_rho_mean": float(group["observed_rho"].mean()),
                "observed_mechanics_utilization_mean": float(group["mechanics_utilization"].mean()),
                "Wq": float(group["waiting_time_total"].mean()),
                "Lq": float(group["queue_length_total"].mean()),
                "WaitingTime_HP": float(group["waiting_time_hp"].mean()),
                "WaitingTime_LP": float(group["waiting_time_lp"].mean()),
                "WaitingTime_PM_Regular": float(group["waiting_time_pm_regular"].mean()),
                "WaitingTime_PM_Escalated": float(group["waiting_time_pm_escalated"].mean()),
                "QueueLength_HP": float(group["queue_length_hp"].mean()),
                "QueueLength_LP": float(group["queue_length_lp"].mean()),
                "QueueLength_PM_Regular": float(group["queue_length_pm_regular"].mean()),
                "QueueLength_PM_Escalated": float(group["queue_length_pm_escalated"].mean()),
                "MTTR_CM": float(group["mttr_cm"].mean()),
                "EscalationRate": float(group["escalation_rate"].mean()),
                "Escalations_Sum": int(group["escalations_total"].sum()),
                "MechanicsUtilization": float(group["mechanics_utilization"].mean()),
                "HangarUtilization": float(group["hangar_utilization"].mean()),
            }
        )
        for metric in METRICS:
            ci_rows.append({**base, **summarize_metric(group, metric)})

    summary = pd.DataFrame(summary_rows).sort_values(by="target_rho", ignore_index=True)
    confidence_intervals = pd.DataFrame(ci_rows).sort_values(
        by=["target_rho", "metric"],
        ignore_index=True,
    )
    return summary, confidence_intervals


def plot_line(
    summary: pd.DataFrame,
    columns: list[str],
    labels: list[str],
    ylabel: str,
    title: str,
    target: Path,
) -> None:
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for column, label in zip(columns, labels, strict=True):
        ax.plot(summary["target_rho"], summary[column], marker="o", linewidth=2, label=label)
    ax.set_xlabel(r"Target utilization $\rho$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(target, dpi=300, bbox_inches="tight")
    plt.close(fig)


def band_series(confidence_intervals: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean and 95 % interval of one metric over the load, ordered by target rho."""
    rows = confidence_intervals[confidence_intervals["metric"] == metric]
    return rows.sort_values("target_rho")


def plot_main_figure(confidence_intervals: pd.DataFrame, target: Path) -> None:
    """Waiting time and escalation rate over the load, two panels with 95 % bands.

    No title and no footnote inside the image: both belong in the caption of the paper.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=MAIN_FIGURE_SIZE, constrained_layout=True)
    for metric, label, color, marker in WAITING_SERIES:
        rows = band_series(confidence_intervals, metric)
        left.fill_between(rows["target_rho"], rows["ci_lower"], rows["ci_upper"], color=color, alpha=0.25, linewidth=0)
        left.plot(rows["target_rho"], rows["mean"], color=color, marker=marker, markersize=5, linewidth=1.8, label=label)
    left.set_ylabel("Mean waiting time $W_q$ [h]", fontsize=MAIN_FIGURE_FONT["label"])
    left.legend(loc="upper left", frameon=False, fontsize=MAIN_FIGURE_FONT["legend"])

    metric, label, color, marker = ESCALATION_SERIES
    rows = band_series(confidence_intervals, metric)
    right.fill_between(rows["target_rho"], rows["ci_lower"], rows["ci_upper"], color=color, alpha=0.25, linewidth=0)
    right.plot(rows["target_rho"], rows["mean"], color=color, marker=marker, markersize=5, linewidth=1.8, label=label)
    right.set_ylabel("Escalation rate [-]", fontsize=MAIN_FIGURE_FONT["label"])

    for ax in (left, right):
        ax.set_xlabel(r"Target utilization $\rho$", fontsize=MAIN_FIGURE_FONT["label"])
        # The load axis stays linear, so the curvature in rho is not flattened. That puts 0.85 and
        # 0.90 close together, so their labels are turned to keep every design point readable.
        ax.set_xticks(TARGET_RHOS, [f"{rho:.2f}" for rho in TARGET_RHOS], rotation=45, ha="right")
        ax.tick_params(labelsize=MAIN_FIGURE_FONT["tick"])
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#e4e3dd", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(target, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_plots(summary: pd.DataFrame, confidence_intervals: pd.DataFrame) -> None:
    plot_main_figure(confidence_intervals, PLOTS["waiting_and_escalation"])
    plot_line(
        summary,
        ["Wq", "WaitingTime_HP", "WaitingTime_LP"],
        ["Total", "Original CM", "All PM"],
        "Waiting time [h]",
        "Waiting Time by Target Utilization",
        PLOTS["waiting_time"],
    )
    plot_line(
        summary,
        ["Lq", "QueueLength_HP", "QueueLength_LP"],
        ["Total", "Original CM", "All PM"],
        "Queue length [jobs]",
        "Queue Length by Target Utilization",
        PLOTS["queue_length"],
    )
    plot_line(
        summary,
        ["EscalationRate"],
        ["Escalation rate"],
        "Escalation rate [-]",
        "Escalation Rate by Target Utilization",
        PLOTS["escalation_rate"],
    )
    plot_line(
        summary,
        ["MTTR_CM"],
        ["MTTR CM"],
        "MTTR [h]",
        "CM MTTR by Target Utilization",
        PLOTS["mttr"],
    )
    plot_line(
        summary,
        ["HangarUtilization", "MechanicsUtilization"],
        ["Hangar", "Engineers"],
        "Observed utilization [-]",
        "Resource Utilization by Target Utilization",
        PLOTS["resource_utilization"],
    )


def scenario_configurations(base_scenario: Scenario) -> pd.DataFrame:
    rows = []
    for target_rho, lambda_pm, lambda_cm in target_rho_scenarios(base_scenario):
        rh = rho_hangar(lambda_pm, lambda_cm, base_scenario)
        re = rho_engineers(lambda_pm, lambda_cm, base_scenario)
        rows.append(
            {
                "scenario_name": f"{base_scenario.name}_target_rho_{target_rho:.2f}",
                "target_rho": target_rho,
                "lambda_pm": lambda_pm,
                "lambda_cm": lambda_cm,
                "pm_cm_ratio": lambda_pm / lambda_cm,
                "theoretical_rho_hangar": rh,
                "theoretical_rho_engineers": re,
                "theoretical_rho_max": max(rh, re),
                "run_length_hours": HIGH_PRECISION_RUN_LENGTH,
                "warmup_hours": HIGH_PRECISION_WARMUP,
                "replications": HIGH_PRECISION_REPLICATIONS,
                "base_seed": base_scenario.simulation.seed,
            }
        )
    return pd.DataFrame(rows)


def write_run_configuration(base_scenario: Scenario, scenarios: pd.DataFrame) -> None:
    config = {
        "methodology": "Target-utilization sensitivity analysis using true service-time means",
        "base_scenario_path": BASE_SCENARIO_LABEL,
        "base_scenario": asdict(base_scenario),
        "simulation": {
            "warmup_hours": HIGH_PRECISION_WARMUP,
            "run_length_hours": HIGH_PRECISION_RUN_LENGTH,
            "replications": HIGH_PRECISION_REPLICATIONS,
            "seed_logic": "base seed + replication index",
            "base_seed": base_scenario.simulation.seed,
        },
        "target_rho_scenarios": scenarios.to_dict(orient="records"),
    }
    RUN_CONFIG_JSON.write_text(json.dumps(config, indent=2) + "\n")

    lines = [
        "High-precision target-rho DES sensitivity analysis",
        "=" * 72,
        "Methodology: arrival rates solved from true service-time means, PM:CM ratio kept",
        f"Base scenario: {BASE_SCENARIO_LABEL}",
        f"Warm-up period: {HIGH_PRECISION_WARMUP:.1f} h",
        f"Simulation horizon: {HIGH_PRECISION_RUN_LENGTH:.1f} h",
        f"Independent replications: {HIGH_PRECISION_REPLICATIONS}",
        f"Base seed: {base_scenario.simulation.seed}",
        "Seed strategy: base seed + replication index",
        "Utilization definition: rho_hangar = (lambda_pm * E[S_PM] + lambda_cm * E[S_CM]) / hangar_slots",
        f"Baseline PM:CM ratio: {base_scenario.arrival_rate_pm / base_scenario.arrival_rate_cm:.7f}",
    ]
    RUN_METADATA_TXT.write_text("\n".join(lines) + "\n")


def run_analysis() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_scenario = load_base_scenario()
    scenarios = scenario_configurations(base_scenario)
    scenarios.to_csv(SCENARIO_CONFIG_CSV, index=False)
    scenarios[
        [
            "target_rho",
            "lambda_pm",
            "lambda_cm",
            "theoretical_rho_hangar",
            "theoretical_rho_engineers",
            "theoretical_rho_max",
        ]
    ].to_csv(THEORETICAL_RHO_CSV, index=False)
    write_run_configuration(base_scenario, scenarios)

    model = DiscreteEventSimulationModel()
    raw_rows: list[dict[str, float | int | str | None]] = []
    for target_rho, lambda_pm, lambda_cm in target_rho_scenarios(base_scenario):
        scenario = build_scenario(base_scenario, target_rho, lambda_pm, lambda_cm)
        for index, result in enumerate(model.run_replications(scenario)):
            raw_rows.append(result_to_row(scenario, target_rho, result, index + 1, scenario.simulation.seed + index))

    raw_results = pd.DataFrame(raw_rows)
    summary, confidence_intervals = build_summary_tables(raw_results)

    raw_results.to_csv(RAW_RESULTS_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    confidence_intervals.to_csv(CONFIDENCE_INTERVALS_CSV, index=False)
    write_plots(summary, confidence_intervals)
    return raw_results, summary, confidence_intervals


def main() -> None:
    raw_results, summary, confidence_intervals = run_analysis()
    print("High-precision target-rho DES sensitivity analysis")
    print("=" * 72)
    print(summary.to_string(index=False))
    print()
    print(f"Raw replication rows: {len(raw_results)}")
    print(f"Confidence interval rows: {len(confidence_intervals)}")
    print()
    print(f"Caption of {PLOTS['waiting_and_escalation'].name}: {MAIN_FIGURE_CAPTION}")
    print()
    print("Exports")
    print("=" * 72)
    for path in [
        RAW_RESULTS_CSV,
        SUMMARY_CSV,
        CONFIDENCE_INTERVALS_CSV,
        SCENARIO_CONFIG_CSV,
        THEORETICAL_RHO_CSV,
        RUN_CONFIG_JSON,
        RUN_METADATA_TXT,
        *PLOTS.values(),
    ]:
        print(path)


if __name__ == "__main__":
    main()
