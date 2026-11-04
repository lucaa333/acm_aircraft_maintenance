"""Does the simulation horizon still move the estimates at high load? Paired by seed against 50,000 h."""
from __future__ import annotations

import pandas as pd

from desim.experiments.pipeline_context import RESULTS_DIR, load_base_scenario
from desim.experiments.replications import build_variant, confidence_interval, field_values, interval_columns
from desim.models.simulation.des_model import DiscreteEventSimulationModel

OUTPUT_CSV = RESULTS_DIR / "horizon_check/horizon_check.csv"
TARGET_RHOS = [0.75, 0.85, 0.90]
HORIZONS = [5000.0, 50000.0, 200000.0]
REFERENCE_HORIZON = 50000.0  # the horizon of the load sensitivity


def main() -> None:
    base = load_base_scenario()
    model = DiscreteEventSimulationModel()
    rows = []
    for rho in TARGET_RHOS:
        runs_by_horizon = {horizon: model.run_replications(build_variant(base, rho=rho, run_length=horizon)) for horizon in HORIZONS}
        reference = field_values(runs_by_horizon[REFERENCE_HORIZON], "waiting_time_total")
        for horizon, runs in runs_by_horizon.items():
            # Same seeds: the paths agree up to the shorter horizon, so the difference is paired.
            waits = field_values(runs, "waiting_time_total")
            difference, lower, upper = confidence_interval([wait - ref for wait, ref in zip(waits, reference, strict=True)])
            rows.append(
                {
                    "target_rho": rho,
                    "run_length_hours": horizon,
                    **interval_columns(runs, "waiting_time_total", "Wq"),
                    **interval_columns(runs, "waiting_time_hp", "Wq_CM"),
                    **interval_columns(runs, "waiting_time_lp", "Wq_PM"),
                    **interval_columns(runs, "escalation_rate", "EscalationRate"),
                    "paired_difference_Wq_vs_50000h": difference,
                    "paired_difference_ci_lower": lower,
                    "paired_difference_ci_upper": upper,
                }
            )
    table = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_CSV, index=False)
    columns = ["target_rho", "run_length_hours", "Wq", "paired_difference_Wq_vs_50000h", "paired_difference_ci_lower", "paired_difference_ci_upper"]
    print(table[columns].round(4).to_string(index=False))
    print(f"\nCSV export: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
