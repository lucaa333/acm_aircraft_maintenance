"""Preemption rules of the paper configuration in the DES."""
from collections import defaultdict
from dataclasses import replace

import pytest

from desim.domain.scenario import arrival_rates_for_target_rho, load_scenario
from desim.models.simulation.des_model import DiscreteEventSimulationModel

THRESHOLD = 14.0


def scenario_at(rho: float | None = 0.90, run_length: float = 20000.0, **priority):
    scenario = load_scenario("config/scenarios/stable.json")
    if rho is not None:
        lambda_pm, lambda_cm = arrival_rates_for_target_rho(scenario, rho)
        scenario = replace(scenario, arrival_rate_pm=lambda_pm, arrival_rate_cm=lambda_cm)
    return replace(
        scenario,
        priority=replace(scenario.priority, **priority),
        simulation=replace(scenario.simulation, run_length=run_length),
    )


def starts_per_task(scenario, seed: int = 42):
    trace: list = []
    result = DiscreteEventSimulationModel()._run_once(scenario, seed, trace=trace)
    starts: dict[int, list[tuple[float, str, float]]] = defaultdict(list)
    for time, group, wait, _, task_id in trace:
        starts[task_id].append((time, group, wait))
    return result, starts


def test_paper_configuration() -> None:
    priority = load_scenario("config/scenarios/stable.json").priority
    assert priority.mode == "priority"
    assert priority.preemption_setup_time == 0.5
    assert priority.escalate_threshold == THRESHOLD


def test_cm_and_escalated_pm_are_never_interrupted() -> None:
    result, starts = starts_per_task(scenario_at())
    assert result.preemptions_total > 0
    for runs in starts.values():
        groups = [group for _, group, _ in runs]
        if groups[0] == "CM":
            assert groups == ["CM"]
        if "PM_escalated" in groups:
            # Once escalated, a PM starts once more and then runs to completion.
            assert groups.count("PM_escalated") == 1 and groups[-1] == "PM_escalated"


def test_escalated_pm_preempts_regular_pm() -> None:
    result, _ = starts_per_task(scenario_at())
    assert result.preemptions_by_cm > 0
    assert result.preemptions_by_escalated_pm > 0


def test_regular_pm_never_starts_after_its_deadline() -> None:
    """A PM still waiting at arrival + X escalates; one preempted after that point goes to the HP queue at once."""
    result, starts = starts_per_task(scenario_at())
    assert result.preemptions_after_deadline > 0
    for runs in starts.values():
        arrival = runs[0][0] - runs[0][2]
        for time, group, _ in runs:
            if group == "PM":
                assert time < arrival + THRESHOLD + 1e-9


def test_setup_time_is_added_per_preemption() -> None:
    with_setup, _ = starts_per_task(scenario_at())
    assert with_setup.setup_hours == pytest.approx(0.5 * with_setup.preemptions_total)
    without_setup, _ = starts_per_task(scenario_at(preemption_setup_time=0.0))
    assert without_setup.setup_hours == 0.0
    assert with_setup.hangar_utilization > without_setup.hangar_utilization


def test_negative_setup_time_is_rejected() -> None:
    with pytest.raises(ValueError):
        replace(load_scenario("config/scenarios/stable.json").priority, preemption_setup_time=-1.0)


def test_preemption_rate_is_a_share_of_pm_tasks() -> None:
    result, _ = starts_per_task(scenario_at())
    assert 0.0 < result.preemption_rate < 1.0


def test_blocked_bay_share_vanishes_with_sixteen_engineers() -> None:
    scenario = scenario_at()
    blocked, _ = starts_per_task(scenario)
    assert blocked.blocked_bay_share > 0.0
    assert blocked.blocked_cm_behind_cm_share == pytest.approx(blocked.blocked_bay_share)
    roomy, _ = starts_per_task(replace(scenario, resources=replace(scenario.resources, mechanics=16)))
    assert roomy.blocked_bay_share == 0.0


def interrupted_run(scenario, seed: int = 42):
    """Starts per task and the interruption log: (time, trigger id, victim id, ids of the interruptible PM)."""
    trace: list = []
    interruptions: list = []
    result = DiscreteEventSimulationModel()._run_once(scenario, seed, trace=trace, interruptions=interruptions)
    starts: dict[int, list[tuple[float, str, float]]] = defaultdict(list)
    for time, group, wait, _, task_id in trace:
        starts[task_id].append((time, group, wait))
    assert len(interruptions) >= result.preemptions_total > 0
    return result, starts, interruptions


def test_r10_r12_trigger_starts_at_the_interruption() -> None:
    """R10, R12: an interruption happens only if it lets the HP head start, which it then does at once."""
    _, starts, interruptions = interrupted_run(scenario_at())
    for time, trigger, _, _ in interruptions:
        assert any(start == time for start, _, _ in starts[trigger])


def test_r11_only_regular_pm_are_interrupted() -> None:
    """R11: the victim ran as a never-escalated PM; CM and escalated PM are never interrupted."""
    _, starts, interruptions = interrupted_run(scenario_at())
    for time, _, victim, candidates in interruptions:
        assert victim in candidates
        # Strictly before: a victim past its deadline may resume as an escalated PM in the same instant.
        last_group = [group for start, group, _ in starts[victim] if start < time][-1]
        assert last_group == "PM"


def test_r14_interrupted_pm_resumes_first_among_regular_pm() -> None:
    """R14: the interrupted PM waits at the front of the LP queue."""
    _, starts, interruptions = interrupted_run(scenario_at())
    victims_after = sorted((time, victim) for time, _, victim, _ in interruptions)
    regular_starts = sorted((start, task) for task, runs in starts.items() for start, group, _ in runs if group == "PM")
    for time, victim in victims_after:
        later_regular = [(start, task) for start, task in regular_starts if start > time]
        if not later_regular or victim not in {task for _, task in later_regular}:
            continue  # completed as an escalated PM or still waiting at the horizon
        first_start, first_task = later_regular[0]
        # Either the victim itself, or a PM interrupted after it, which then stands in front of it.
        assert first_task == victim or any(time < t <= first_start and v == first_task for t, v in victims_after)


def test_r15_interrupted_pm_past_deadline_moves_to_hp() -> None:
    """R15: past its deadline the victim resumes as an escalated PM, and the trigger still starts first."""
    result, starts, interruptions = interrupted_run(scenario_at())
    late = 0
    for time, trigger, victim, _ in interruptions:
        arrival = starts[victim][0][0] - starts[victim][0][2]
        later = [group for start, group, _ in starts[victim] if start > time]
        if time >= arrival + THRESHOLD and later:
            late += 1
            assert later[0] == "PM_escalated"
            assert any(start == time for start, _, _ in starts[trigger])
    assert late > 0 and result.preemptions_after_deadline > 0


def victim_choices(scenario):
    """Per interruption: the victim and, per interruptible PM, (deadline passed, last start before it)."""
    _, starts, interruptions = interrupted_run(scenario)
    threshold = scenario.priority.escalate_threshold
    for time, _, victim, candidates in interruptions:
        state = {}
        for task in candidates:
            arrival = starts[task][0][0] - starts[task][0][2]
            passed = threshold is not None and time >= arrival + threshold
            state[task] = (passed, max(start for start, _, _ in starts[task] if start < time))
        yield victim, state


def test_ep7_pm_within_deadline_is_interrupted_before_one_past_it() -> None:
    """EP7: with one PM past its deadline and one within, the one within is interrupted (the later started, if several)."""
    mixed = 0
    for victim, state in victim_choices(scenario_at()):
        within = {task: start for task, (passed, start) in state.items() if not passed}
        if within and len(within) < len(state):
            mixed += 1
            assert victim in within and within[victim] == max(within.values())
    assert mixed > 0


def test_ep7_all_past_deadline_takes_the_latest_started() -> None:
    """EP7: if every interruptible PM is past its deadline, the one started last is interrupted."""
    all_past = 0
    for victim, state in victim_choices(scenario_at()):
        if len(state) > 1 and all(passed for passed, _ in state.values()):
            all_past += 1
            assert state[victim][1] == max(start for _, start in state.values())
    assert all_past > 0


def test_ep7_without_escalation_takes_the_latest_started() -> None:
    """Without escalation no deadline passes: the PM started last, as before EP7, so the verification is unchanged."""
    with_choice = 0
    for victim, state in victim_choices(scenario_at(escalate_threshold=None)):
        assert state[victim][1] == max(start for _, start in state.values())
        with_choice += len(state) > 1
    assert with_choice > 0


def test_preemption_cascades_count_escalated_pm_interrupted_in_the_same_instant() -> None:
    scenario = scenario_at()
    result, _, interruptions = interrupted_run(scenario)
    victims_at = {(time, victim) for time, _, victim, _ in interruptions}
    cascades = sum(
        (time, trigger) in victims_at for time, trigger, _, _ in interruptions if time >= scenario.simulation.warmup
    )
    assert result.preemption_cascades == cascades
    assert 0 < result.preemption_cascades <= result.preemptions_by_escalated_pm
    no_escalation, _ = starts_per_task(scenario_at(escalate_threshold=None))
    assert no_escalation.preemption_cascades == 0


def test_r24_set_up_time_counts_as_service_and_resume_waits_as_waiting() -> None:
    """R24: for PM with a fixed 5 h duration, time in system minus wait is 5 h plus 0.5 h per interruption."""
    base = scenario_at(run_length=50000.0)
    fixed = replace(base, pm_service=replace(base.pm_service, kind="deterministic", stddev=0.0))
    result, _, interruptions = interrupted_run(fixed)
    completed = result.tasks_pm_regular + result.tasks_pm_escalated
    in_service = sum(
        tasks * (cycle - wait)
        for tasks, cycle, wait in [
            (result.tasks_pm_regular, result.cycle_time_pm_regular, result.waiting_time_pm_regular),
            (result.tasks_pm_escalated, result.cycle_time_pm_escalated, result.waiting_time_pm_escalated),
        ]
    )
    setups = (in_service - 5.0 * completed) / 0.5
    # A whole number of set-ups: waiting to resume is waiting, the set-up time is service. The slack covers
    # the 0.001 h floor a PM keeps when it is interrupted in the instant it would finish.
    assert setups == pytest.approx(round(setups), abs=0.01)
    # Interruptions of tasks that arrived in the warm-up or are unfinished at the horizon are the only gap.
    assert len(interruptions) - 20 <= round(setups) <= len(interruptions)
