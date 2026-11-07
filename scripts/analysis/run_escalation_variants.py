"""Escalation order against load: overtaking, queueing behind CM, no escalation, and a 16th engineer."""
from __future__ import annotations

import pandas as pd

from desim.experiments.pipeline_context import RESULTS_DIR, load_base_scenario
from desim.experiments.replications import (
    build_variant,
    field_values,
    interval_columns,
    paired_difference_columns,
)
from desim.models.simulation.des_model import DiscreteEventSimulationModel

OUTPUT_CSV = RESULTS_DIR / "escalation_variants/escalation_variants.csv"
TARGET_RHOS = [0.25, 0.50, 0.75, 0.85, 0.90]
RUN_LENGTH = 50000.0

VARIANTS = [
    ("escalated PM joins HP queue end", {"queue_position": "fifo"}),
    ("escalated PM overtakes CM", {"queue_position": "front"}),
    ("escalated PM behind every CM", {"queue_position": "behind_cm"}),
    ("no escalation", {"escalation": False}),
    ("16 engineers, configured order", {"mechanics": 16}),
]

# The rule of the paper: escalated PM at the end of the HP queue, 15 engineers. Every other
# placement is differenced against it, paired per replication.
PAPER_VARIANT = "escalated PM joins HP queue end"


def main() -> None:
    base = load_base_scenario()
    model = DiscreteEventSimulationModel()
    rows = []
    for rho in TARGET_RHOS:
        # All placements of one load first, so each row can be differenced against the paper rule.
        # Seeds are the base seed plus the replication index, so the run order changes no number.
        runs_by_variant = {
            label: model.run_replications(build_variant(base, rho=rho, run_length=RUN_LENGTH, **options))
            for label, options in VARIANTS
        }
        paper_waits = field_values(runs_by_variant[PAPER_VARIANT], "waiting_time_total")
        for label, _ in VARIANTS:
            runs = runs_by_variant[label]
            rows.append(
                {
                    "target_rho": rho,
                    "variant": label,
                    "run_length_hours": RUN_LENGTH,
                    **interval_columns(runs, "waiting_time_total", "Wq"),
                    **interval_columns(runs, "waiting_time_hp", "Wq_CM"),
                    **interval_columns(runs, "waiting_time_lp", "Wq_PM"),
                    **interval_columns(runs, "mttr_cm", "MTTR_CM"),
                    **interval_columns(runs, "queue_length_total", "Lq"),
                    **interval_columns(runs, "escalation_rate", "EscalationRate"),
                    **interval_columns(runs, "blocked_bay_share", "BlockedBayShare"),
                    **interval_columns(runs, "preemption_rate", "PreemptionRate"),
                    **interval_columns(runs, "preemption_cascades", "PreemptionCascades"),
                    "Escalations_Sum": int(sum(field_values(runs, "escalations_total"))),
                    **paired_difference_columns(
                        field_values(runs, "waiting_time_total"), paper_waits, "paired_difference_Wq_vs_paper_rule"
                    ),
                }
            )
    table = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_CSV, index=False)
    print(table[["target_rho", "variant", "Wq", "Wq_CM", "Wq_PM", "MTTR_CM", "EscalationRate"]].round(4).to_string(index=False))
    print()
    print(f"Paired differences of Wq against '{PAPER_VARIANT}', 95 % intervals over the same seeds")
    print(
        table[
            [
                "target_rho",
                "variant",
                "paired_difference_Wq_vs_paper_rule",
                "paired_difference_Wq_vs_paper_rule_ci_lower",
                "paired_difference_Wq_vs_paper_rule_ci_upper",
            ]
        ].round(4).to_string(index=False)
    )
    print(f"\nCSV export: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
