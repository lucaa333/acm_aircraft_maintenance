from __future__ import annotations

from dataclasses import replace
import math

import numpy as np

from desim.domain.metrics import ModelResult, t_critical_95
from desim.domain.scenario import EscalationConfig, Scenario, arrival_rates_for_target_rho


def confidence_interval(values: list[float] | np.ndarray) -> tuple[float, float, float]:
    """Mean and 95 % t interval, not capped at zero so that the output stays an honest interval."""
    data = np.asarray(values, dtype=float)
    mean = float(data.mean())
    margin = t_critical_95(len(data) - 1) * float(data.std(ddof=1)) / math.sqrt(len(data)) if len(data) > 1 else 0.0
    return mean, mean - margin, mean + margin


def build_variant(
    base: Scenario,
    *,
    rho: float | None = None,
    run_length: float = 5000.0,
    warmup: float = 100.0,
    replications: int = 30,
    mechanics: int | None = None,
    bays: int | None = None,
    escalation: bool = True,
    queue_position: str | None = None,
) -> Scenario:
    """Derive an experiment from the base scenario; ``rho=None`` keeps the baseline arrival rates."""
    lambda_pm, lambda_cm = (
        (base.arrival_rate_pm, base.arrival_rate_cm) if rho is None else arrival_rates_for_target_rho(base, rho)
    )
    return replace(
        base,
        arrival_rate_pm=lambda_pm,
        arrival_rate_cm=lambda_cm,
        resources=replace(
            base.resources,
            mechanics=base.resources.mechanics if mechanics is None else mechanics,
            hangar_slots=base.resources.hangar_slots if bays is None else bays,
        ),
        priority=base.priority if escalation else replace(base.priority, escalate_threshold=None),
        escalation=EscalationConfig(queue_position) if queue_position else base.escalation,
        simulation=replace(base.simulation, run_length=run_length, warmup=warmup, replications=replications),
    )


def field_values(results: list[ModelResult], field: str) -> list[float]:
    return [getattr(result, field) for result in results if getattr(result, field) is not None]


def estimate_columns(values: list[float] | np.ndarray, label: str) -> dict[str, float]:
    """Mean and 95 % interval of an already derived per-replication series."""
    mean, lower, upper = confidence_interval(values)
    return {label: mean, f"{label}_ci_lower": lower, f"{label}_ci_upper": upper}


def interval_columns(results: list[ModelResult], field: str, label: str) -> dict[str, float]:
    return estimate_columns(field_values(results, field), label)


def paired_difference_columns(
    values: list[float], reference: list[float] | None, label: str
) -> dict[str, float | None]:
    """Mean of ``values - reference`` per replication and its 95 % interval.

    Common random numbers make the pairing valid: both arms ran on the same seeds, so differencing
    per replication removes the shared arrival noise and leaves a much tighter interval than two
    independent means would give. ``reference=None`` writes empty columns, which keeps the row
    shape identical for an arm that has no partner.
    """
    if reference is None:
        return {label: None, f"{label}_ci_lower": None, f"{label}_ci_upper": None}
    return estimate_columns(
        [value - ref for value, ref in zip(values, reference, strict=True)], label
    )


def mser5_truncation(series: np.ndarray) -> int:
    """Number of leading observations MSER-5 deletes; the scan covers the first half of the batches."""
    batches = series[: len(series) // 5 * 5].reshape(-1, 5).mean(axis=1)
    count = len(batches)
    scores = [batches[cut:].var() / (count - cut) for cut in range(count // 2)]
    return int(np.argmin(scores)) * 5
