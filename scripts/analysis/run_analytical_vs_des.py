"""Analytical models against the DES per load level. Numbers only, no interpretation."""
from __future__ import annotations

from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from desim.experiments.pipeline_context import RESULTS_DIR, load_base_scenario
from desim.experiments.replications import build_variant, confidence_interval, field_values
from desim.models.analytical.common import mg1_metrics
from desim.models.analytical.single_priority import MGSinglePriorityModel
from desim.models.priority_queue.multi_priority import MGPriorityModel
from desim.models.simulation.des_model import DiscreteEventSimulationModel

OUTPUT_CSV = RESULTS_DIR / "analytical_vs_des/analytical_vs_des.csv"
OUTPUT_PNG = RESULTS_DIR / "analytical_vs_des/decomposition_ratio.png"

# None is the baseline with its own rates and horizon (rho = 0.255); the others match the sensitivity.
LOADS = [(None, 5000.0), (0.25, 50000.0), (0.50, 50000.0), (0.75, 50000.0), (0.85, 50000.0), (0.90, 50000.0)]

# (identifier, label, build_variant options). The variants form a chain: V1 meets the assumptions of
# the analytical models (no coupling, no escalation, no set-up time after a preemption), and each
# next variant adds one mechanism of the paper configuration: V2 the set-up time, V3 the coupling of
# bays and engineers (15 instead of 16 engineers), V4 the escalation. The run order does not affect
# the numbers: seeds depend only on the replication index (common random numbers).
DES_VARIANTS = [
    ("V1", "no escalation, 16 engineers, no set-up time", {"escalation": False, "mechanics": 16, "setup": 0.0}),
    ("V2", "no escalation, 16 engineers", {"escalation": False, "mechanics": 16}),
    ("V3", "no escalation, 15 engineers", {"escalation": False}),
    ("V4", "paper configuration", {}),
]
# Legend of the figure: the mechanism each variant adds, so the figure reads without the identifiers.
LEGEND_LABELS = {
    "V1": "analytical assumptions",
    "V2": "+ set-up time",
    "V3": "+ coupling",
    "V4": "+ escalation (paper)",
}

# (label, DES field, analytical priority field, whether the FCFS model has a value, field of the
# CM-only M/G/1). The CM-only reference is a single bay serving the CM stream alone, so it speaks
# only about the CM/HP class and stays empty for the pooled and the PM rows.
KPIS = [
    ("Wq total", "waiting_time_total", "waiting_time_total", True, None),
    ("Wq CM/HP", "waiting_time_hp", "waiting_time_hp", False, "wq"),
    ("Wq PM/LP", "waiting_time_lp", "waiting_time_lp", False, None),
    ("Lq total", "queue_length_total", "queue_length_total", True, None),
    ("Lq CM/HP", "queue_length_hp", "queue_length_hp", False, "lq"),
    ("Lq PM/LP", "queue_length_lp", "queue_length_lp", False, None),
]

# Categorical slots 1-4 of the validated palette; the marker shape is the second encoding.
# Validated for the adjacent pairlist in light mode, the CSV is the table view of the figure.
FIGURE_STYLE = {"V1": ("#2a78d6", "o"), "V2": ("#eb6834", "s"), "V3": ("#1baf7a", "^"), "V4": ("#eda100", "D")}
# One column of the ACM template, with point sizes set for that width.
FIGURE_SIZE = (3.33, 2.7)
FIGURE_FONT = {"tick": 7, "label": 8, "legend": 7}
FIGURE_RHOS = [rho for rho, _ in LOADS if rho is not None]

# The image carries no footnote: this text belongs in the caption of the paper, see outputs/MANIFEST.md.
FIGURE_CAPTION = (
    "Ratio of the DES mean waiting time to the M/G/2 preemptive-resume approximation over the load, "
    "per DES variant. 1.0 is agreement, above 1.0 the DES waits longer than the analytical model. "
    "Each variant adds one mechanism to the previous one: analytical assumptions (16 engineers, no "
    "escalation, no set-up time), + set-up time, + coupling (15 engineers), + escalation (paper "
    "configuration). Up to rho = 0.50 the last two coincide, since hardly any PM task escalates. "
    "The baseline load is left out because its rates are not on the target-rho grid. 30 replications "
    "of 50,000 h per point, common random numbers."
)


def cm_only_mg1(scenario) -> dict[str, float]:
    """M/G/1 fed by the CM stream alone: one bay, lambda_CM, the CM service distribution.

    This is the Pollaczek-Khinchine reference of section 3.1 for a CM task that never shares its
    bay with a PM task, so it bounds how much of the CM wait the PM stream contributes.
    """
    _, wq, lq = mg1_metrics(
        scenario.arrival_rate_cm,
        scenario.cm_service.moment_mean,
        scenario.cm_service.moment_stddev**2,
    )
    return {"wq": wq, "lq": lq}


def plot_decomposition(table: pd.DataFrame) -> None:
    """Ratio DES / M/G/2 preemptive over the load, one line per DES variant, reference line at 1."""
    rows = table[(table["kpi"] == "Wq total") & (table["load"] != "baseline")].copy()
    rows["rho"] = rows["load"].astype(float)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    ax.axhline(1.0, color="#8a8984", linestyle="--", linewidth=1)
    for identifier, _, _ in DES_VARIANTS:
        series = rows[rows["des_variant_id"] == identifier].sort_values("rho")
        color, marker = FIGURE_STYLE[identifier]
        ax.plot(
            series["rho"],
            series["ratio_des_to_priority"],
            color=color,
            marker=marker,
            markersize=5,
            linewidth=1.8,
            label=LEGEND_LABELS[identifier],
        )
    ax.set_xlabel(r"Target utilization $\rho$", fontsize=FIGURE_FONT["label"])
    ax.set_ylabel("DES / M/G/2 preemptive [-]", fontsize=FIGURE_FONT["label"])
    # The reference of a ratio is 1.0, not 0, so the axis is not forced to the origin: that would
    # squash the whole spread of the variants into the top fifth of the panel.
    ax.set_xticks(FIGURE_RHOS, [f"{rho:.2f}" for rho in FIGURE_RHOS], rotation=45, ha="right")
    ax.tick_params(labelsize=FIGURE_FONT["tick"])
    ax.grid(axis="y", color="#e4e3dd", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=FIGURE_FONT["legend"])
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    base = load_base_scenario()
    model = DiscreteEventSimulationModel()
    rows = []
    for rho, horizon in LOADS:
        analytical_scenario = build_variant(base, rho=rho)
        fcfs = MGSinglePriorityModel(2).run(analytical_scenario)
        priority = MGPriorityModel(2).run(analytical_scenario)
        cm_only = cm_only_mg1(analytical_scenario)
        rho_cm_only = analytical_scenario.arrival_rate_cm * analytical_scenario.cm_service.moment_mean
        for identifier, label, options in DES_VARIANTS:
            options = dict(options)
            setup = options.pop("setup", None)
            scenario = build_variant(base, rho=rho, run_length=horizon, **options)
            if setup is not None:
                scenario = replace(scenario, priority=replace(scenario.priority, preemption_setup_time=setup))
            runs = model.run_replications(scenario)
            utilization = confidence_interval(field_values(runs, "hangar_utilization"))
            setup_hours = confidence_interval(field_values(runs, "setup_hours"))
            for kpi, des_field, analytical_field, has_fcfs, cm_only_field in KPIS:
                mean, lower, upper = confidence_interval(field_values(runs, des_field))
                fcfs_value = getattr(fcfs, des_field) if has_fcfs else None
                priority_value = getattr(priority, analytical_field)
                cm_only_value = cm_only[cm_only_field] if cm_only_field else None
                rows.append(
                    {
                        "load": "baseline" if rho is None else f"{rho:.2f}",
                        "rho_hangar": priority.hangar_utilization,
                        "run_length_hours": horizon,
                        "des_variant": label,
                        "kpi": kpi,
                        "mg2_fcfs": fcfs_value,
                        "mg2_priority_approximation": priority_value,
                        "des_mean": mean,
                        "des_ci_lower": lower,
                        "des_ci_upper": upper,
                        "ratio_des_to_fcfs": mean / fcfs_value if fcfs_value else None,
                        "ratio_des_to_priority": mean / priority_value,
                        "des_variant_id": identifier,
                        "mg1_cm_only": cm_only_value,
                        "rho_mg1_cm_only": rho_cm_only if cm_only_field else None,
                        "ratio_des_to_mg1_cm_only": mean / cm_only_value if cm_only_value else None,
                        "des_hangar_utilization": utilization[0],
                        "des_hangar_utilization_ci_lower": utilization[1],
                        "des_hangar_utilization_ci_upper": utilization[2],
                        "des_setup_hours": setup_hours[0],
                        "des_setup_hours_ci_lower": setup_hours[1],
                        "des_setup_hours_ci_upper": setup_hours[2],
                    }
                )
    table = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_CSV, index=False)
    plot_decomposition(table)
    print(table[table.kpi == "Wq total"].drop(columns=["kpi", "run_length_hours"]).round(4).to_string(index=False))
    print()
    print("M/G/1 with the CM stream alone, against the DES CM wait")
    print(
        table[table.kpi == "Wq CM/HP"][
            ["load", "des_variant_id", "rho_mg1_cm_only", "mg1_cm_only", "des_mean", "ratio_des_to_mg1_cm_only"]
        ].round(4).to_string(index=False)
    )
    print(f"\nCSV export: {OUTPUT_CSV}\nFigure: {OUTPUT_PNG}")
    print(f"Figure caption: {FIGURE_CAPTION}")


if __name__ == "__main__":
    main()
