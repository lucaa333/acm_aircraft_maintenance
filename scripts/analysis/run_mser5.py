"""MSER-5 truncation point of the waiting-time sequence, starting from the empty system."""
from __future__ import annotations

import statistics

import numpy as np
import pandas as pd

from desim.experiments.pipeline_context import RESULTS_DIR, load_base_scenario
from desim.experiments.replications import build_variant, interval_columns, mser5_truncation
from desim.models.simulation.des_model import DiscreteEventSimulationModel

OUTPUT_CSV = RESULTS_DIR / "validation/mser5.csv"
LOADS = [("baseline", None), ("0.90", 0.90)]
RUN_LENGTH = 50000.0
REPLICATIONS = 30
CONFIGURED_WARMUP = 100.0


def main() -> None:
    base = load_base_scenario()
    model = DiscreteEventSimulationModel()
    rows = []
    for label, rho in LOADS:
        scenario = build_variant(base, rho=rho, run_length=RUN_LENGTH, warmup=0.0, replications=REPLICATIONS)
        traces = []
        for index in range(REPLICATIONS):
            trace: list = []
            model._run_once(scenario, scenario.simulation.seed + index, trace=trace)
            traces.append(trace)
        length = min(len(trace) for trace in traces)
        waits = np.array([[start[2] for start in trace[:length]] for trace in traces])
        times = np.array([[start[0] for start in trace[:length]] for trace in traces]).mean(axis=0)
        ensemble_cut = mser5_truncation(waits.mean(axis=0))
        single_cuts = [times[mser5_truncation(row)] for row in waits]
        truncation_hours = float(times[ensemble_cut])
        configured = model.run_replications(build_variant(base, rho=rho, run_length=RUN_LENGTH, warmup=CONFIGURED_WARMUP))
        truncated = model.run_replications(build_variant(base, rho=rho, run_length=RUN_LENGTH, warmup=truncation_hours))
        rows.append(
            {
                "load": label,
                "run_length_hours": RUN_LENGTH,
                "replications": REPLICATIONS,
                "tasks_per_replication": length,
                "ensemble_truncation_tasks": ensemble_cut,
                "ensemble_truncation_hours": truncation_hours,
                "single_replication_median_hours": statistics.median(single_cuts),
                "single_replication_max_hours": max(single_cuts),
                "configured_warmup_hours": CONFIGURED_WARMUP,
                "configured_warmup_covers_ensemble_truncation": truncation_hours <= CONFIGURED_WARMUP,
                **interval_columns(configured, "waiting_time_total", "Wq_configured_warmup"),
                **interval_columns(truncated, "waiting_time_total", "Wq_truncated_warmup"),
            }
        )
    table = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_CSV, index=False)
    print(table.round(2).to_string(index=False))
    print(f"\nCSV export: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
