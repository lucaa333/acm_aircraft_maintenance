from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass(slots=True)
class ServiceDistribution:
    """Parameters describing a task's service-time distribution.

    ``mean`` (and ``stddev``) describe exponential, normal and deterministic durations. A triangular
    duration is given by ``min_value``, ``mode`` and ``max_value``. Analytical models and utilization
    figures always use ``moment_mean`` and ``moment_stddev``, the true distribution moments.
    """

    kind: str
    mean: float | None = None
    stddev: float = 0.0
    min_value: float | None = None
    max_value: float | None = None
    mode: float | None = None  # most likely value of a triangular duration

    def __post_init__(self) -> None:
        if self.kind == "triangular":
            if self.mean is not None or None in (self.min_value, self.mode, self.max_value):
                raise ValueError("A triangular duration takes min_value, mode and max_value; its mean follows from them.")
            if not self.min_value <= self.mode <= self.max_value:
                raise ValueError("A triangular duration needs min_value <= mode <= max_value.")
        elif self.mean is None:
            raise ValueError(f"A {self.kind} duration needs a mean.")

    @property
    def moment_mean(self) -> float:
        if self.kind == "triangular":
            return (self.min_value + self.mode + self.max_value) / 3
        return self.mean

    @property
    def moment_stddev(self) -> float:
        if self.kind == "triangular":
            a, m, b = self.min_value, self.mode, self.max_value
            return ((a * a + m * m + b * b - a * m - a * b - m * b) / 18) ** 0.5
        return self.stddev

    @property
    def scv(self) -> float:
        # Squared coefficient of variation is a common queueing-theory input.
        if self.moment_mean <= 0:
            return 0.0
        return (self.moment_stddev / self.moment_mean) ** 2


@dataclass(slots=True)
class PriorityPolicy:
    # "priority": CM and escalated PM form the HP class and interrupt running never-escalated PM
    # (preemptive resume, the interrupted aircraft leaves its bay). "fcfs": one arrival-ordered queue
    # without classes, escalation or preemption, used only for verification.
    mode: str
    escalate_threshold: float | None
    # Extra work a preempted PM needs when it resumes (securing, moving the aircraft, set-up), in hours.
    preemption_setup_time: float = 0.0

    def __post_init__(self) -> None:
        if self.preemption_setup_time < 0:
            raise ValueError("preemption_setup_time must not be negative.")
        if self.mode not in ("priority", "fcfs"):
            raise ValueError(f"Unknown priority mode: {self.mode}")
        if self.mode == "fcfs" and self.escalate_threshold is not None:
            raise ValueError("FCFS has no priority classes, so escalation must be off.")


@dataclass(slots=True)
class EscalationConfig:
    # Where an escalated PM joins the HP queue: "fifo" at its end, like any HP task, so the whole
    # dispatch order is arrival + allowance (0 for CM, X for PM). "front" overtakes waiting CM,
    # "behind_cm" stays behind every CM, also later arrivals.
    queue_position: str = "fifo"

    def __post_init__(self) -> None:
        if self.queue_position not in ("fifo", "front", "behind_cm"):
            raise ValueError(f"Unknown escalation queue position: {self.queue_position}")


@dataclass(slots=True)
class ResourceConfig:
    # Used by the DES; the analytical models take their server count from the model class.
    hangar_slots: int
    mechanics: int


@dataclass(slots=True)
class SimulationConfig:
    # Parameters for replicated DES runs.
    run_length: float
    warmup: float
    replications: int
    seed: int
    check_invariants: bool = True  # stop a run on a broken invariant; switch off only for speed


@dataclass(slots=True)
class Scenario:
    # A scenario bundles all assumptions for one experiment run.
    # Every model receives the same scenario so results stay comparable.
    name: str
    description: str
    arrival_rate_pm: float
    arrival_rate_cm: float
    pm_service: ServiceDistribution
    cm_service: ServiceDistribution
    pm_mechanics: int
    cm_mechanics: int
    resources: ResourceConfig
    priority: PriorityPolicy
    simulation: SimulationConfig
    escalation: EscalationConfig = field(default_factory=EscalationConfig)

    @property
    def combined_arrival_rate(self) -> float:
        # Single-priority models collapse PM and CM into one arrival stream.
        return self.arrival_rate_pm + self.arrival_rate_cm


def arrival_rates_for_target_rho(scenario: Scenario, target: float) -> tuple[float, float]:
    """Scale both arrival rates so hangar utilization hits ``target``, keeping the PM:CM ratio."""
    ratio = scenario.arrival_rate_pm / scenario.arrival_rate_cm
    lambda_cm = target * scenario.resources.hangar_slots / (
        ratio * scenario.pm_service.moment_mean + scenario.cm_service.moment_mean
    )
    return ratio * lambda_cm, lambda_cm


def load_scenario(path: str | Path) -> Scenario:
    # JSON is converted into typed dataclasses here so the rest of the code
    # can work with a stable in-memory structure instead of raw dictionaries.
    data = json.loads(Path(path).read_text())
    return Scenario(
        name=data["name"],
        description=data.get("description", ""),
        arrival_rate_pm=data["arrival_rate_pm"],
        arrival_rate_cm=data["arrival_rate_cm"],
        pm_service=ServiceDistribution(**data["pm_service"]),
        cm_service=ServiceDistribution(**data["cm_service"]),
        pm_mechanics=data["pm_mechanics"],
        cm_mechanics=data["cm_mechanics"],
        resources=ResourceConfig(**data["resources"]),
        priority=PriorityPolicy(**data["priority"]),
        simulation=SimulationConfig(**data["simulation"]),
        escalation=EscalationConfig(**data.get("escalation", {})),
    )
