"""2x2 ablation with common random numbers: coupling (15 or 16 engineers) and escalation (on or off)."""
from __future__ import annotations

import pandas as pd

from desim.experiments.pipeline_context import RESULTS_DIR, load_base_scenario
from desim.experiments.replications import build_variant, confidence_interval, field_values, interval_columns
from desim.models.simulation.des_model import DiscreteEventSimulationModel

OUTPUT_CSV = RESULTS_DIR / "ablation/ablation_2x2.csv"
LOADS = [(None, 5000.0), (0.75, 50000.0), (0.90, 50000.0)]
DESIGN = [(15, True), (15, False), (16, True), (16, False)]  # the first row is the reference


def main() -> None:
    base = load_base_scenario()
    model = DiscreteEventSimulationModel()
    rows = []
    for rho, horizon in LOADS:
        reference: list[float] | None = None
        for mechanics, escalation in DESIGN:
            runs = model.run_replications(
                build_variant(base, rho=rho, run_length=horizon, mechanics=mechanics, escalation=escalation)
            )
            waits = field_values(runs, "waiting_time_total")
            reference = reference or waits
            difference, lower, upper = confidence_interval([wait - ref for wait, ref in zip(waits, reference, strict=True)])
            rows.append(
                {
                    "load": "baseline" if rho is None else f"{rho:.2f}",
                    "rho_hangar": model_rho(base, rho),
                    "run_length_hours": horizon,
                    "engineers": mechanics,
                    "escalation": escalation,
                    **interval_columns(runs, "waiting_time_total", "Wq"),
                    **interval_columns(runs, "waiting_time_hp", "Wq_CM"),
                    **interval_columns(runs, "waiting_time_pm_regular", "Wq_PM_Regular"),
                    **(interval_columns(runs, "waiting_time_pm_escalated", "Wq_PM_Escalated") if escalation else {}),
                    **interval_columns(runs, "mttr_cm", "MTTR_CM"),
                    **interval_columns(runs, "escalation_rate", "EscalationRate"),
                    **interval_columns(runs, "blocked_bay_share", "BlockedBayShare"),
                    **interval_columns(runs, "preemption_rate", "PreemptionRate"),
                    **interval_columns(runs, "preemption_cascades", "PreemptionCascades"),
                    "paired_difference_Wq": difference,
                    "paired_difference_ci_lower": lower,
                    "paired_difference_ci_upper": upper,
                }
            )
    table = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_CSV, index=False)
    print(table[["load", "engineers", "escalation", "Wq", "Wq_CM", "MTTR_CM", "EscalationRate", "paired_difference_Wq"]].round(4).to_string(index=False))
    print(f"\nCSV export: {OUTPUT_CSV}")


def model_rho(base, rho: float | None) -> float:
    scenario = build_variant(base, rho=rho)
    work = scenario.arrival_rate_pm * scenario.pm_service.moment_mean + scenario.arrival_rate_cm * scenario.cm_service.moment_mean
    return work / scenario.resources.hangar_slots


if __name__ == "__main__":
    main()
