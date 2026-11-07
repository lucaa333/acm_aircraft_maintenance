"""2x2 ablation with common random numbers: coupling (15 or 16 engineers) and escalation (on or off)."""
from __future__ import annotations

import pandas as pd

from desim.experiments.pipeline_context import RESULTS_DIR, load_base_scenario
from desim.experiments.replications import (
    build_variant,
    confidence_interval,
    estimate_columns,
    field_values,
    interval_columns,
)
from desim.models.simulation.des_model import DiscreteEventSimulationModel

OUTPUT_CSV = RESULTS_DIR / "ablation/ablation_2x2.csv"
INTERACTION_CSV = RESULTS_DIR / "ablation/ablation_interaction.csv"
LOADS = [(None, 5000.0), (0.75, 50000.0), (0.90, 50000.0)]
DESIGN = [(15, True), (15, False), (16, True), (16, False)]  # the first row is the reference


def main() -> None:
    base = load_base_scenario()
    model = DiscreteEventSimulationModel()
    rows = []
    interaction_rows = []
    for rho, horizon in LOADS:
        # All four cells of one load first: the factorial effects below need every cell. Seeds are
        # the base seed plus the replication index, so collecting first changes no number.
        runs_by_cell = {}
        waits_by_cell = {}
        for mechanics, escalation in DESIGN:
            runs = model.run_replications(
                build_variant(base, rho=rho, run_length=horizon, mechanics=mechanics, escalation=escalation)
            )
            runs_by_cell[(mechanics, escalation)] = runs
            waits_by_cell[(mechanics, escalation)] = field_values(runs, "waiting_time_total")
        reference = waits_by_cell[DESIGN[0]]
        label = "baseline" if rho is None else f"{rho:.2f}"
        for mechanics, escalation in DESIGN:
            runs = runs_by_cell[(mechanics, escalation)]
            waits = waits_by_cell[(mechanics, escalation)]
            difference, lower, upper = confidence_interval([wait - ref for wait, ref in zip(waits, reference, strict=True)])
            rows.append(
                {
                    "load": label,
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
        interaction_rows.append(
            factorial_effects(label, model_rho(base, rho), horizon, waits_by_cell)
        )
    table = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_CSV, index=False)
    interaction = pd.DataFrame(interaction_rows)
    interaction.to_csv(INTERACTION_CSV, index=False)
    print(table[["load", "engineers", "escalation", "Wq", "Wq_CM", "MTTR_CM", "EscalationRate", "paired_difference_Wq"]].round(4).to_string(index=False))
    print()
    print("Factorial effects on Wq, paired per replication, 95 % intervals")
    print(
        interaction[
            [
                "load",
                "main_effect_escalation",
                "main_effect_16th_engineer",
                "effect_escalation_at_15_engineers",
                "effect_escalation_at_16_engineers",
                "interaction_escalation_x_engineers",
                "interaction_escalation_x_engineers_ci_lower",
                "interaction_escalation_x_engineers_ci_upper",
            ]
        ].round(4).to_string(index=False)
    )
    print(f"\nCSV export: {OUTPUT_CSV}\nCSV export: {INTERACTION_CSV}")


def factorial_effects(
    label: str, rho_hangar: float, horizon: float, waits: dict[tuple[int, bool], list[float]]
) -> dict[str, float | str]:
    """Main effects and the interaction of the 2x2 design on Wq, differenced per replication.

    Sign convention: an escalation effect is "escalation on minus off" and the engineer effect is
    "16 minus 15 engineers", so a negative value means the change lowers the waiting time. The
    interaction is the escalation effect at 15 engineers minus the one at 16; it is negative when
    escalation lowers Wq more strongly while the engineer pool still couples the two bays, because
    8 + 8 > 15 keeps two CM tasks from running together.

    All four cells share the seeds, so every effect is a within-replication contrast.
    """
    at_15 = [on - off for on, off in zip(waits[(15, True)], waits[(15, False)], strict=True)]
    at_16 = [on - off for on, off in zip(waits[(16, True)], waits[(16, False)], strict=True)]
    engineer_with = [sixteen - fifteen for sixteen, fifteen in zip(waits[(16, True)], waits[(15, True)], strict=True)]
    engineer_without = [sixteen - fifteen for sixteen, fifteen in zip(waits[(16, False)], waits[(15, False)], strict=True)]
    return {
        "load": label,
        "rho_hangar": rho_hangar,
        "run_length_hours": horizon,
        **estimate_columns([(a + b) / 2 for a, b in zip(at_15, at_16, strict=True)], "main_effect_escalation"),
        **estimate_columns(
            [(a + b) / 2 for a, b in zip(engineer_with, engineer_without, strict=True)], "main_effect_16th_engineer"
        ),
        **estimate_columns(at_15, "effect_escalation_at_15_engineers"),
        **estimate_columns(at_16, "effect_escalation_at_16_engineers"),
        **estimate_columns([a - b for a, b in zip(at_15, at_16, strict=True)], "interaction_escalation_x_engineers"),
    }


def model_rho(base, rho: float | None) -> float:
    scenario = build_variant(base, rho=rho)
    work = scenario.arrival_rate_pm * scenario.pm_service.moment_mean + scenario.arrival_rate_cm * scenario.cm_service.moment_mean
    return work / scenario.resources.hangar_slots


if __name__ == "__main__":
    main()
