"""DES against analytical references, as a table for the paper; tests/test_verification.py asserts the exact ones.

The priority references follow the paper configuration (preemptive-resume). Coupling, escalation and the
set-up time are switched off, because no analytical model contains them.
"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd

from desim.domain.scenario import PriorityPolicy, ServiceDistribution, arrival_rates_for_target_rho
from desim.experiments.pipeline_context import RESULTS_DIR, load_base_scenario
from desim.experiments.replications import build_variant, confidence_interval, field_values
from desim.models.analytical.single_priority import MGSinglePriorityModel, MMSinglePriorityModel, analytical_inputs
from desim.models.priority_queue.multi_priority import MGPriorityModel, MMPriorityModel
from desim.models.simulation.des_model import DiscreteEventSimulationModel

OUTPUT_DIR = RESULTS_DIR / "verification"
HORIZONS = [5000.0, 100000.0]
NO_COUPLING_ENGINEERS = 16
EXACT_MM2_LOADS = [None, 0.50, 0.75, 0.85, 0.90]


def main() -> None:
    base = load_base_scenario()
    model = DiscreteEventSimulationModel()
    fcfs = PriorityPolicy(mode="fcfs", escalate_threshold=None)
    pooled_mean = (
        base.arrival_rate_pm * base.pm_service.moment_mean + base.arrival_rate_cm * base.cm_service.moment_mean
    ) / base.combined_arrival_rate
    pooled = ServiceDistribution(kind="exponential", mean=pooled_mean)
    no_setup = replace(base.priority, escalate_threshold=None, preemption_setup_time=0.0)

    def scenario(run_length: float, bays: int, *, policy: PriorityPolicy | None = None):
        variant = build_variant(base, run_length=run_length, bays=bays, mechanics=NO_COUPLING_ENGINEERS, escalation=False)
        return replace(variant, priority=policy or no_setup)

    # (label, exact?, DES scenario builder, DES field, reference value)
    reference = scenario(5000.0, 1)
    two_bays = scenario(5000.0, 2)
    mg1 = MGPriorityModel(1).run(reference)
    mg2 = MGPriorityModel(2).run(two_bays)
    exponential_pm = ServiceDistribution(kind="exponential", mean=base.pm_service.moment_mean)
    exponential_cm = ServiceDistribution(kind="exponential", mean=base.cm_service.moment_mean)

    def class_exponential(run_length: float):
        return replace(scenario(run_length, 2), pm_service=exponential_pm, cm_service=exponential_cm)

    mm2 = MMPriorityModel(2).run(two_bays)
    tests = [
        ("FCFS c=1 vs M/G/1 Pollaczek-Khinchine", True, lambda t: scenario(t, 1, policy=fcfs), "waiting_time_total", MGSinglePriorityModel(1).run(reference).waiting_time_total),
        ("FCFS exponential c=2 vs M/M/2 Erlang-C", True, lambda t: replace(scenario(t, 2, policy=fcfs), pm_service=pooled, cm_service=pooled), "waiting_time_total", MMSinglePriorityModel(2).run(two_bays).waiting_time_total),
        ("Preemptive c=1 vs exact M/G/1, HP", True, lambda t: scenario(t, 1), "waiting_time_hp", mg1.waiting_time_hp),
        ("Preemptive c=1 vs exact M/G/1, LP", True, lambda t: scenario(t, 1), "waiting_time_lp", mg1.waiting_time_lp),
        ("Preemptive exponential c=2 vs exact M/M/2, HP", True, class_exponential, "waiting_time_hp", mm2.waiting_time_hp),
        ("Preemptive exponential c=2 vs exact M/M/2, LP", True, class_exponential, "waiting_time_lp", mm2.waiting_time_lp),
        ("FCFS c=2 vs M/G/2 approximation", False, lambda t: scenario(t, 2, policy=fcfs), "waiting_time_total", MGSinglePriorityModel(2).run(two_bays).waiting_time_total),
        ("Preemptive c=2 vs Bondi-Buzen approximation, HP", False, lambda t: scenario(t, 2), "waiting_time_hp", mg2.waiting_time_hp),
        ("Preemptive c=2 vs Bondi-Buzen approximation, LP", False, lambda t: scenario(t, 2), "waiting_time_lp", mg2.waiting_time_lp),
    ]
    rows = []
    for label, exact, build, field, expected in tests:
        for horizon in HORIZONS:
            mean, lower, upper = confidence_interval(field_values(model.run_replications(build(horizon)), field))
            rows.append(
                {
                    "test": label,
                    "reference_is_exact": exact,
                    "reference": expected,
                    "run_length_hours": horizon,
                    "des_mean": mean,
                    "des_ci_lower": lower,
                    "des_ci_upper": upper,
                    "reference_in_ci": bool(lower < expected < upper),
                    "deviation_pct": 100.0 * (mean - expected) / expected,
                }
            )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_DIR / "verification.csv", index=False)

    exact_rows = []
    for rho in EXACT_MM2_LOADS:
        lambda_pm, lambda_cm = (base.arrival_rate_pm, base.arrival_rate_cm) if rho is None else arrival_rates_for_target_rho(base, rho)
        load = replace(base, arrival_rate_pm=lambda_pm, arrival_rate_cm=lambda_cm)
        exact = MMPriorityModel(2).run(load)
        approx = MMPriorityModel(2, exact=False).run(load)
        exact_rows.append(
            {
                "rho_hangar": exact.hangar_utilization,
                "exact_hp": exact.waiting_time_hp,
                "exact_lp": exact.waiting_time_lp,
                "approx_hp": approx.waiting_time_hp,
                "approx_lp": approx.waiting_time_lp,
                "deviation_hp_pct": 100.0 * (approx.waiting_time_hp - exact.waiting_time_hp) / exact.waiting_time_hp,
                "deviation_lp_pct": 100.0 * (approx.waiting_time_lp - exact.waiting_time_lp) / exact.waiting_time_lp,
            }
        )
    exact_table = pd.DataFrame(exact_rows)
    exact_table.to_csv(OUTPUT_DIR / "mm2_priority_exact_vs_approximation.csv", index=False)
    print(exact_table.round(4).to_string(index=False))

    inputs = pd.DataFrame(analytical_inputs(base))
    inputs.to_csv(OUTPUT_DIR / "analytical_inputs.csv", index=False)
    print(inputs.round(4).to_string(index=False))

    print(table.round(4).to_string(index=False))
    print(f"\nCSV export: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
