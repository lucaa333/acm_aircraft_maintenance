"""Is 100 h warm-up enough at high load? Longer warm-ups on the same seeds, paired against 100 h.

The horizon includes the warm-up, so a longer warm-up shortens the observation window.
"""
from __future__ import annotations

import pandas as pd

from desim.experiments.pipeline_context import RESULTS_DIR, load_base_scenario
from desim.experiments.replications import build_variant, confidence_interval, field_values, interval_columns
from desim.models.simulation.des_model import DiscreteEventSimulationModel

OUTPUT_CSV = RESULTS_DIR / "validation/warmup_check_high_load.csv"
TARGET_RHO = 0.90
RUN_LENGTH = 50000.0
WARMUPS = [100.0, 2000.0, 5000.0]  # the first one is the configured warm-up and the reference


def main() -> None:
    base = load_base_scenario()
    model = DiscreteEventSimulationModel()
    rows = []
    reference: list[float] | None = None
    for warmup in WARMUPS:
        runs = model.run_replications(build_variant(base, rho=TARGET_RHO, run_length=RUN_LENGTH, warmup=warmup))
        waits = field_values(runs, "waiting_time_total")
        reference = reference or waits
        # Same seeds give the same sample path; only the deleted start differs, so the difference is paired.
        difference, lower, upper = confidence_interval([wait - ref for wait, ref in zip(waits, reference, strict=True)])
        rows.append(
            {
                "target_rho": TARGET_RHO,
                "run_length_hours": RUN_LENGTH,
                "warmup_hours": warmup,
                "replications": len(runs),
                **interval_columns(runs, "waiting_time_total", "Wq"),
                **interval_columns(runs, "waiting_time_hp", "Wq_CM"),
                **interval_columns(runs, "waiting_time_lp", "Wq_PM"),
                "paired_difference_Wq_vs_100h": difference,
                "paired_difference_ci_lower": lower,
                "paired_difference_ci_upper": upper,
            }
        )
    table = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_CSV, index=False)
    columns = ["warmup_hours", "Wq", "paired_difference_Wq_vs_100h", "paired_difference_ci_lower", "paired_difference_ci_upper"]
    print(table[columns].round(4).to_string(index=False))
    print(f"\nCSV export: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
