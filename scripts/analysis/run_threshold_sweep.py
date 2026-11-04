"""Escalation threshold X against load: from X = 0 (FCFS only without preemption) to no threshold, which is strict priority."""
from __future__ import annotations

from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from desim.experiments.pipeline_context import RESULTS_DIR, load_base_scenario
from desim.experiments.replications import build_variant, interval_columns
from desim.models.simulation.des_model import DiscreteEventSimulationModel

OUTPUT_CSV = RESULTS_DIR / "threshold_sweep/threshold_sweep.csv"
OUTPUT_PNG = RESULTS_DIR / "threshold_sweep/threshold_sweep.png"
LOADS = [(None, 5000.0), (0.50, 50000.0), (0.75, 50000.0), (0.85, 50000.0), (0.90, 50000.0)]
THRESHOLDS = [0.0, 3.5, 7.0, 14.0, 28.0, 56.0, None]
PLOT_LOADS = ["0.75", "0.85", "0.90"]
PAPER_THRESHOLD = 14.0
# Categorical slots 1-3 of the validated palette; the marker shape is the second encoding.
SERIES = [("Wq", "Total", "#2a78d6", "o"), ("Wq_CM", "CM", "#eb6834", "s"), ("Wq_PM", "PM", "#1baf7a", "^")]


def plot(table: pd.DataFrame) -> None:
    """Wq over X per load, one panel each with its own scale, 95 % band per class."""
    labels = ["none" if x is None else f"{x:g}" for x in THRESHOLDS]
    positions = range(len(labels))
    fig, axes = plt.subplots(1, len(PLOT_LOADS), figsize=(12, 4), constrained_layout=True)
    for ax, load in zip(axes, PLOT_LOADS, strict=True):
        rows = table[table["load"] == load]
        for column, label, color, marker in SERIES:
            ax.fill_between(positions, rows[f"{column}_ci_lower"], rows[f"{column}_ci_upper"], color=color, alpha=0.15, linewidth=0)
            ax.plot(positions, rows[column], color=color, marker=marker, markersize=6, linewidth=2, label=label)
        ax.axvline(THRESHOLDS.index(PAPER_THRESHOLD), color="#8a8984", linestyle="--", linewidth=1)
        ax.set_title(f"rho = {load}", color="#1a1a19")
        ax.set_xticks(list(positions), labels)
        ax.set_xlabel("Escalation threshold X [h]")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#e4e3dd", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean waiting time Wq [h]")
    axes[0].legend(loc="upper left", frameon=False)
    fig.text(0.5, -0.04, "Dashed line: X = 14 h of the paper. Bands: 95 % t intervals over 30 replications.", ha="center", color="#5f5e5a", fontsize=9)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    base = load_base_scenario()
    model = DiscreteEventSimulationModel()
    rows = []
    for rho, horizon in LOADS:
        for threshold in THRESHOLDS:
            scenario = build_variant(base, rho=rho, run_length=horizon)
            scenario = replace(scenario, priority=replace(scenario.priority, escalate_threshold=threshold))
            runs = model.run_replications(scenario)
            rows.append(
                {
                    "load": "baseline" if rho is None else f"{rho:.2f}",
                    "threshold_hours": "none" if threshold is None else threshold,
                    "run_length_hours": horizon,
                    **interval_columns(runs, "waiting_time_total", "Wq"),
                    **interval_columns(runs, "waiting_time_hp", "Wq_CM"),
                    **interval_columns(runs, "waiting_time_lp", "Wq_PM"),
                    **interval_columns(runs, "mttr_cm", "MTTR_CM"),
                    **interval_columns(runs, "escalation_rate", "EscalationRate"),
                    **interval_columns(runs, "blocked_bay_share", "BlockedBayShare"),
                    **interval_columns(runs, "preemption_rate", "PreemptionRate"),
                    **interval_columns(runs, "preemption_cascades", "PreemptionCascades"),
                }
            )
    table = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_CSV, index=False)
    plot(table)
    print(table[["load", "threshold_hours", "Wq", "Wq_CM", "Wq_PM", "MTTR_CM", "EscalationRate"]].round(3).to_string(index=False))
    print(f"\nCSV export: {OUTPUT_CSV}\nFigure: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
