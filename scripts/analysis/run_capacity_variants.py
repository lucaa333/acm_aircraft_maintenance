"""Capacity what-ifs on the baseline: base case against a 16th engineer and a third bay.

All variants share seeds, so waiting-time differences are paired per replication.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from desim.experiments.pipeline_context import RESULTS_DIR, load_base_scenario
from desim.experiments.replications import build_variant, confidence_interval, interval_columns
from desim.models.simulation.des_model import DiscreteEventSimulationModel

OUTPUT_CSV = RESULTS_DIR / "capacity_variants/capacity_variants.csv"

VARIANTS = [
    ("15 engineers, 2 bays", 15, 2),
    ("16 engineers, 2 bays", 16, 2),
    ("15 engineers, 3 bays", 15, 3),
]


def main() -> None:
    base = load_base_scenario()
    model = DiscreteEventSimulationModel()
    base_waits: np.ndarray | None = None
    rows = []
    for label, mechanics, bays in VARIANTS:
        runs = model.run_replications(build_variant(base, mechanics=mechanics, bays=bays))
        waits = np.array([run.waiting_time_total for run in runs])
        if base_waits is None:
            base_waits = waits
        difference, difference_lower, difference_upper = confidence_interval(waits - base_waits)
        rows.append(
            {
                "variant": label,
                **interval_columns(runs, "waiting_time_total", "Wq"),
                **interval_columns(runs, "waiting_time_hp", "Wq_CM"),
                **interval_columns(runs, "waiting_time_lp", "Wq_PM"),
                **interval_columns(runs, "mttr_cm", "MTTR_CM"),
                "difference_to_base": difference,
                "difference_ci_lower": difference_lower,
                "difference_ci_upper": difference_upper,
                "relative_change_pct": 100.0 * difference / float(base_waits.mean()),
            }
        )
    table = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_CSV, index=False)
    print(table.to_string(index=False))
    print(f"\nCSV export: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
