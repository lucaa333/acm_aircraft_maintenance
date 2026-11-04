"""Analytical models against the DES per load level. Numbers only, no interpretation."""
from __future__ import annotations

from dataclasses import replace

import pandas as pd

from desim.experiments.pipeline_context import RESULTS_DIR, load_base_scenario
from desim.experiments.replications import build_variant, confidence_interval, field_values
from desim.models.analytical.single_priority import MGSinglePriorityModel
from desim.models.priority_queue.multi_priority import MGPriorityModel
from desim.models.simulation.des_model import DiscreteEventSimulationModel

OUTPUT_CSV = RESULTS_DIR / "analytical_vs_des/analytical_vs_des.csv"

# None is the baseline with its own rates and horizon (rho = 0.255); the others match the sensitivity.
LOADS = [(None, 5000.0), (0.25, 50000.0), (0.50, 50000.0), (0.75, 50000.0), (0.85, 50000.0), (0.90, 50000.0)]

DES_VARIANTS = [
    ("no escalation, 15 engineers", {"escalation": False}),
    ("no escalation, 16 engineers", {"escalation": False, "mechanics": 16}),
    # Closest to the analytical assumptions: no coupling, no escalation and no set-up time after a preemption.
    ("no escalation, 16 engineers, no set-up time", {"escalation": False, "mechanics": 16, "setup": 0.0}),
    ("paper configuration", {}),
]

# (label, DES field, analytical priority field, whether the FCFS model has a value)
KPIS = [
    ("Wq total", "waiting_time_total", "waiting_time_total", True),
    ("Wq CM/HP", "waiting_time_hp", "waiting_time_hp", False),
    ("Wq PM/LP", "waiting_time_lp", "waiting_time_lp", False),
    ("Lq total", "queue_length_total", "queue_length_total", True),
    ("Lq CM/HP", "queue_length_hp", "queue_length_hp", False),
    ("Lq PM/LP", "queue_length_lp", "queue_length_lp", False),
]


def main() -> None:
    base = load_base_scenario()
    model = DiscreteEventSimulationModel()
    rows = []
    for rho, horizon in LOADS:
        analytical_scenario = build_variant(base, rho=rho)
        fcfs = MGSinglePriorityModel(2).run(analytical_scenario)
        priority = MGPriorityModel(2).run(analytical_scenario)
        for label, options in DES_VARIANTS:
            options = dict(options)
            setup = options.pop("setup", None)
            scenario = build_variant(base, rho=rho, run_length=horizon, **options)
            if setup is not None:
                scenario = replace(scenario, priority=replace(scenario.priority, preemption_setup_time=setup))
            runs = model.run_replications(scenario)
            for kpi, des_field, analytical_field, has_fcfs in KPIS:
                mean, lower, upper = confidence_interval(field_values(runs, des_field))
                fcfs_value = getattr(fcfs, des_field) if has_fcfs else None
                priority_value = getattr(priority, analytical_field)
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
                    }
                )
    table = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_CSV, index=False)
    print(table[table.kpi == "Wq total"].drop(columns=["kpi", "run_length_hours"]).round(4).to_string(index=False))
    print(f"\nCSV export: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
