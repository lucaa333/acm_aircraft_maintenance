from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import math

import numpy as np


# Per-replication KPIs of the three origin classes that have no HP/LP equivalent.
ORIGIN_FIELDS = (
    "waiting_time_pm_regular",
    "waiting_time_pm_escalated",
    "queue_length_pm_regular",
    "queue_length_pm_escalated",
    "cycle_time_pm_regular",
    "cycle_time_pm_escalated",
    "tasks_cm",
    "tasks_pm_regular",
    "tasks_pm_escalated",
    "escalations_total",
)

# Per-replication KPIs of preemption and of the resource coupling.
PREEMPTION_FIELDS = (
    "preemption_rate",  # share of PM tasks (arrived after warm-up) preempted at least once
    "preemptions_total",  # preemptions after warm-up
    "preemptions_by_cm",
    "preemptions_by_escalated_pm",
    "preemptions_after_deadline",  # preempted PM already past its escalation deadline, moved to the HP queue at once
    "preemption_cascades",  # preemptions by an escalated PM that was itself preempted in the same instant
    "setup_hours",  # set-up time added by preemptions after warm-up
    "blocked_bay_share",  # share of time a bay is free while tasks wait (head-of-queue blocking)
    "blocked_cm_behind_cm_share",  # part of it where the waiting head is a CM and a CM runs (8 + 8 > 15)
)

# All DES detail fields that scripts copy and average beyond the HP/LP summary.
ORIGIN_FIELDS = ORIGIN_FIELDS + PREEMPTION_FIELDS


@lru_cache
def t_critical_95(df: int) -> float:
    """Two-sided 95 % Student-t critical value, integrated numerically to avoid a scipy dependency."""
    x = np.linspace(0.0, 13.0, 130001)  # t(1) = 12.71 is the largest value we can be asked for
    scale = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
    pdf = scale * (1.0 + x**2 / df) ** (-(df + 1) / 2)
    cdf = 0.5 + np.concatenate(([0.0], np.cumsum((pdf[1:] + pdf[:-1]) / 2 * (x[1] - x[0]))))
    return float(np.interp(0.975, cdf, x))


@dataclass(slots=True)
class ModelResult:
    # Common result schema for all model families.
    # Some fields stay empty for simpler analytical models.
    model_name: str
    scenario_name: str
    waiting_time_total: float
    queue_length_total: float
    waiting_time_hp: float | None = None
    waiting_time_lp: float | None = None
    queue_length_hp: float | None = None
    queue_length_lp: float | None = None
    mttr_cm: float | None = None
    hangar_utilization: float | None = None  # per bay; for the analytical models rho per server
    mechanics_utilization: float | None = None
    escalation_rate: float | None = None
    waiting_time_pm_regular: float | None = None
    waiting_time_pm_escalated: float | None = None
    queue_length_pm_regular: float | None = None
    queue_length_pm_escalated: float | None = None
    cycle_time_pm_regular: float | None = None
    cycle_time_pm_escalated: float | None = None
    tasks_cm: float | None = None
    tasks_pm_regular: float | None = None
    tasks_pm_escalated: float | None = None
    escalations_total: float | None = None
    preemption_rate: float | None = None
    preemptions_total: float | None = None
    preemptions_by_cm: float | None = None
    preemptions_by_escalated_pm: float | None = None
    preemptions_after_deadline: float | None = None
    preemption_cascades: float | None = None
    setup_hours: float | None = None
    blocked_bay_share: float | None = None
    blocked_cm_behind_cm_share: float | None = None
    extra: dict[str, float | str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float | str | None]:
        # Output formatting works on dictionaries, so rounding and optional
        # values are normalized in a single place.
        return {
            "model_name": self.model_name,
            "scenario_name": self.scenario_name,
            "waiting_time_total": round(self.waiting_time_total, 4),
            "queue_length_total": round(self.queue_length_total, 4),
            "waiting_time_hp": _round_or_none(self.waiting_time_hp),
            "waiting_time_lp": _round_or_none(self.waiting_time_lp),
            "queue_length_hp": _round_or_none(self.queue_length_hp),
            "queue_length_lp": _round_or_none(self.queue_length_lp),
            "mttr_cm": _round_or_none(self.mttr_cm),
            "hangar_utilization": _round_or_none(self.hangar_utilization),
            "mechanics_utilization": _round_or_none(self.mechanics_utilization),
            "escalation_rate": _round_or_none(self.escalation_rate),
            **{name: _round_or_none(getattr(self, name)) for name in ORIGIN_FIELDS},
            **self.extra,
        }


def _round_or_none(value: float | None) -> float | None:
    # Keeps optional metrics printable without forcing fake zeros.
    if value is None:
        return None
    return round(value, 4)
