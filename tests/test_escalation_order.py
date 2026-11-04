"""Dispatch and escalation in the paper configuration (preemptive resume).

A PM can start several times; the order "arrival + allowance" holds for the first start of each task.
"""
from dataclasses import replace

import pytest

from desim.domain.scenario import EscalationConfig, PriorityPolicy, arrival_rates_for_target_rho, load_scenario
from desim.experiments.replications import confidence_interval
from desim.models.simulation.des_model import DiscreteEventSimulationModel

THRESHOLD = 14.0


def high_load(position: str = "fifo", **priority):
    """rho = 0.90 in the paper configuration, with the given escalated-PM placement."""
    scenario = load_scenario("config/scenarios/stable.json")
    lambda_pm, lambda_cm = arrival_rates_for_target_rho(scenario, 0.9)
    return replace(
        scenario,
        arrival_rate_pm=lambda_pm,
        arrival_rate_cm=lambda_cm,
        priority=replace(scenario.priority, **priority),
        escalation=EscalationConfig(queue_position=position),
        simulation=replace(scenario.simulation, run_length=20000.0),
    )


def run(scenario):
    trace: list = []
    result = DiscreteEventSimulationModel()._run_once(scenario, 42, trace=trace)
    return result, trace


def first_starts(trace) -> list[tuple]:
    seen: set[int] = set()
    firsts = []
    for start in trace:
        if start[4] not in seen:
            seen.add(start[4])
            firsts.append(start)
    return firsts


def cm_waiting_at_escalated_starts(position: str) -> list[int]:
    _, trace = run(high_load(position))
    return [waiting_cm for _, group, _, waiting_cm, _ in trace if group == "PM_escalated"]


def test_behind_cm_never_starts_escalated_pm_while_cm_waits() -> None:
    starts = cm_waiting_at_escalated_starts("behind_cm")
    assert starts and max(starts) == 0


def test_front_lets_escalated_pm_overtake_waiting_cm() -> None:
    assert max(cm_waiting_at_escalated_starts("front")) > 0


def test_default_is_fifo() -> None:
    assert load_scenario("config/scenarios/stable.json").escalation.queue_position == "fifo"


def test_unknown_position_is_rejected() -> None:
    with pytest.raises(ValueError):
        EscalationConfig(queue_position="middle")


def test_only_waiting_pm_escalate() -> None:
    """R8: a PM escalates only if it still waits at arrival + X; a task in service never does."""
    result, trace = run(high_load())
    assert result.escalations_total > 0
    firsts = first_starts(trace)
    # Escalated before its first start: it waited at least X.
    assert min(wait for _, group, wait, *_ in firsts if group == "PM_escalated") >= THRESHOLD
    # Started as a regular PM: before its deadline, so it never waited X.
    assert max(wait for _, group, wait, *_ in firsts if group == "PM") < THRESHOLD


def test_first_starts_follow_arrival_plus_allowance() -> None:
    """R7: head-of-queue dispatch; a task that had arrived and has the earlier key never starts later."""
    _, trace = run(high_load())
    starts = [(start, start - wait, start - wait + (0.0 if group == "CM" else THRESHOLD)) for start, group, wait, *_ in first_starts(trace)]
    for index, (start, _, key) in enumerate(starts):
        for _, later_arrival, later_key in starts[index + 1 : index + 60]:
            assert later_arrival > start or later_key >= key - 1e-9


def test_threshold_zero_is_not_fcfs() -> None:
    """Escalated PM interrupt PM that started without waiting, which FCFS never does."""
    zero, zero_trace = run(high_load(escalate_threshold=0.0))
    fcfs, fcfs_trace = run(high_load(mode="fcfs", escalate_threshold=None))
    assert zero.preemptions_by_escalated_pm > 0 and fcfs.preemptions_total == 0
    assert [start[0] for start in zero_trace] != [start[0] for start in fcfs_trace]


def test_huge_threshold_is_strict_priority() -> None:
    _, huge = run(high_load(escalate_threshold=1e9))
    _, strict = run(high_load(escalate_threshold=None))
    assert [start[0] for start in huge] == pytest.approx([start[0] for start in strict])


def test_behind_cm_differs_from_no_escalation() -> None:
    """Escalated PM interrupt regular PM and are never interrupted, so the label changes the schedule."""
    _, behind = run(high_load("behind_cm"))
    _, plain = run(high_load("behind_cm", escalate_threshold=None))
    assert [start[0] for start in behind] != [start[0] for start in plain]


def test_fcfs_mode_takes_no_escalation() -> None:
    with pytest.raises(ValueError):
        PriorityPolicy(mode="fcfs", escalate_threshold=THRESHOLD)


def test_confidence_interval_is_not_capped_at_zero() -> None:
    _, lower, _ = confidence_interval([0.0, 0.0, 0.0, 0.01])
    assert lower < 0.0
