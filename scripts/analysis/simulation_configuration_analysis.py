from __future__ import annotations

import math
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from desim.domain.metrics import ModelResult, t_critical_95
from desim.experiments.pipeline_context import RESULTS_DIR, load_base_scenario
from desim.models.simulation.des_model import DiscreteEventSimulationModel

OUTPUT_DIR = RESULTS_DIR / "validation"
WARMUP_PLOT = OUTPUT_DIR / "warmup_analysis.png"
RUN_LENGTH_PLOT = OUTPUT_DIR / "run_length_analysis.png"

METRICS = {
    "waiting_time_total": "Waiting time",
    "mttr_cm": "MTTR",
    "queue_length_total": "Queue length",
    "queue_length_lp": "PM queue length",
}

WARMUP_VALUES = [0.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
OBSERVATION_WINDOW_HOURS = 4000.0
RUN_LENGTH_VALUES = [1000.0, 2000.0, 5000.0, 10000.0]
# 30 replications or fewer only show noise here: nested paths give false trends in the mean.
ANALYSIS_REPLICATIONS = 300
RUN_LENGTH_REPLICATIONS = ANALYSIS_REPLICATIONS
RUN_LENGTH_CSV = OUTPUT_DIR / "run_length_analysis.csv"
WARMUP_CSV = OUTPUT_DIR / "warmup_analysis.csv"


def build_scenario(base_scenario, *, run_length: float, warmup: float, replications: int, seed: int) -> object:
    return replace(
        base_scenario,
        name=f"{base_scenario.name}_analysis",
        simulation=replace(
            base_scenario.simulation,
            run_length=run_length,
            warmup=warmup,
            replications=replications,
            seed=seed,
        ),
    )


def run_replications(scenario) -> list[ModelResult]:
    return DiscreteEventSimulationModel().run_replications(scenario)


def summarize_results(results: list[ModelResult]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for metric_name in METRICS:
        values = np.array([getattr(result, metric_name) for result in results], dtype=float)
        mean_val = float(np.mean(values))
        std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        t_value = t_critical_95(len(values) - 1) if len(values) > 1 else 0.0
        ci_margin = t_value * std_val / math.sqrt(len(values)) if len(values) > 0 else 0.0
        summary[f"{metric_name}_mean"] = mean_val
        summary[f"{metric_name}_std"] = std_val
        summary[f"{metric_name}_ci_lower"] = mean_val - ci_margin
        summary[f"{metric_name}_ci_upper"] = mean_val + ci_margin
    return summary


def intervals_overlap(results: dict[float, dict[str, float]], reference: float) -> bool:
    """True when each setting's interval overlaps the reference interval for all KPIs."""
    return all(
        summary[f"{name}_ci_lower"] <= results[reference][f"{name}_ci_upper"]
        and summary[f"{name}_ci_upper"] >= results[reference][f"{name}_ci_lower"]
        for summary in results.values()
        for name in METRICS
    )


def plot_metric_vs_warmup(warmup_results: dict[float, dict[str, float]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for idx, (metric_name, label) in enumerate(METRICS.items()):
        ax = axes[idx]
        warmups = sorted(warmup_results)
        values = [warmup_results[w][f"{metric_name}_mean"] for w in warmups]
        lower = [warmup_results[w][f"{metric_name}_ci_lower"] for w in warmups]
        upper = [warmup_results[w][f"{metric_name}_ci_upper"] for w in warmups]
        ax.plot(warmups, values, marker="o", linewidth=2)
        ax.fill_between(warmups, lower, upper, alpha=0.2, label="95 % CI")
        ax.legend(loc="best")
        ax.set_title(label)
        ax.set_xlabel("Warm-up (hours)")
        ax.set_ylabel("Mean estimate")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Warm-up sensitivity of DES KPIs")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(WARMUP_PLOT, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric_vs_run_length(run_length_results: dict[float, dict[str, float]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for idx, (metric_name, label) in enumerate(METRICS.items()):
        ax = axes[idx]
        lengths = sorted(run_length_results)
        values = [run_length_results[length][f"{metric_name}_mean"] for length in lengths]
        lower = [run_length_results[length][f"{metric_name}_ci_lower"] for length in lengths]
        upper = [run_length_results[length][f"{metric_name}_ci_upper"] for length in lengths]
        ax.plot(lengths, values, marker="s", linewidth=2)
        ax.fill_between(lengths, lower, upper, alpha=0.2, label="95 % CI")
        ax.legend(loc="best")
        ax.set_title(label)
        ax.set_xlabel("Run length (hours)")
        ax.set_ylabel("Mean estimate")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Run-length convergence of DES KPIs")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(RUN_LENGTH_PLOT, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_scenario = load_base_scenario()

    # Warm-up analysis with a fixed post-warm-up observation window.
    warmup_results: dict[float, dict[str, float]] = {}
    for warmup in WARMUP_VALUES:
        scenario = build_scenario(
            base_scenario,
            run_length=warmup + OBSERVATION_WINDOW_HOURS,
            warmup=warmup,
            replications=ANALYSIS_REPLICATIONS,
            seed=base_scenario.simulation.seed,
        )
        warmup_results[warmup] = summarize_results(run_replications(scenario))
    pd.DataFrame(
        [
            {"warmup_hours": warmup, "metric": name, **{key: summary[f"{name}_{key}"] for key in ("mean", "std", "ci_lower", "ci_upper")}}
            for warmup, summary in warmup_results.items()
            for name in METRICS
        ]
    ).to_csv(WARMUP_CSV, index=False)
    plot_metric_vs_warmup(warmup_results)

    # Run-length analysis using the requested run lengths and 30 independent replications per setting.
    run_length_results: dict[float, dict[str, float]] = {}
    run_length_rows = []
    for run_length in RUN_LENGTH_VALUES:
        scenario = build_scenario(base_scenario, run_length=run_length, warmup=100.0, replications=RUN_LENGTH_REPLICATIONS, seed=base_scenario.simulation.seed)
        summary = summarize_results(run_replications(scenario))
        run_length_results[run_length] = summary
        for metric_name in METRICS:
            run_length_rows.append({
                "run_length_hours": run_length,
                "metric": metric_name,
                "mean": summary[f"{metric_name}_mean"],
                "std": summary[f"{metric_name}_std"],
                "ci_lower": summary[f"{metric_name}_ci_lower"],
                "ci_upper": summary[f"{metric_name}_ci_upper"],
            })
    pd.DataFrame(run_length_rows).to_csv(RUN_LENGTH_CSV, index=False)
    plot_metric_vs_run_length(run_length_results)

    # Compare each run length to the 10000 h baseline and identify the shortest run that stays within 5%.
    run_length_10000 = run_length_results[10000.0]
    relative_change_vs_10000 = {
        run_length: {
            metric_name: abs(run_length_results[run_length][f"{metric_name}_mean"] - run_length_10000[f"{metric_name}_mean"]) / max(1e-9, run_length_10000[f"{metric_name}_mean"])
            for metric_name in METRICS
        }
        for run_length in RUN_LENGTH_VALUES
    }
    max_relative_change = {run_length: max(values.values()) for run_length, values in relative_change_vs_10000.items()}
    shortest_within_5 = next((run_length for run_length in RUN_LENGTH_VALUES if max_relative_change[run_length] <= 0.05), None)

    warmup_sufficient = intervals_overlap(warmup_results, 100.0)

    print("Simulation configuration analysis")
    print("=" * 80)
    print(f"Warm-up sweep with {ANALYSIS_REPLICATIONS} replications; intervals overlap the 100 h setting: {warmup_sufficient}")
    print(f"Run-length analysis used {RUN_LENGTH_REPLICATIONS} independent replications per run length.")
    print("Run-length percentage change relative to the 10000 h baseline:")
    for run_length in RUN_LENGTH_VALUES:
        print(f"  {run_length:.0f} h -> max relative change: {max_relative_change[run_length]:.2%}")
    print(f"Shortest run length within 5% of the 10000 h baseline: {shortest_within_5:.0f} h" if shortest_within_5 is not None else "Shortest run length within 5% of the 10000 h baseline: none")
    print()
    print("Summary")
    print(f"- Is warm-up = 100 h sufficient? {'Yes' if warmup_sufficient else 'No'}")
    print("  Evidence: fixed 4000 h observation window, 95 % intervals per warm-up; see run_mser5.py for the truncation point.")
    print()
    print(f"Run-length CSV export: {RUN_LENGTH_CSV.resolve()}")
    print(f"Warm-up plot: {WARMUP_PLOT.resolve()}")
    print(f"Run-length plot: {RUN_LENGTH_PLOT.resolve()}")


if __name__ == "__main__":
    main()
