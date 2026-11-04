"""DES against exact queueing results. Criterion: the exact value lies in the 95 % t interval."""
from dataclasses import replace

import pytest

from desim.domain.metrics import ModelResult
from desim.domain.scenario import PriorityPolicy, Scenario, ServiceDistribution, arrival_rates_for_target_rho, load_scenario
from desim.experiments.replications import confidence_interval
from desim.models.priority_queue.multi_priority import MGPriorityModel, MMPriorityModel
from desim.models.simulation.des_model import DiscreteEventSimulationModel

HORIZONS = [
    pytest.param(5000.0, id="fast"),
    pytest.param(100000.0, id="long", marks=pytest.mark.slow),
]


def setup(run_length: float, *, mode: str = "priority", bays: int = 1, mechanics: int = 16) -> Scenario:
    """Baseline without coupling, escalation and set-up time; 16 engineers never block two tasks."""
    scenario = load_scenario("config/scenarios/stable.json")
    return replace(
        scenario,
        priority=PriorityPolicy(mode=mode, escalate_threshold=None),
        resources=replace(scenario.resources, hangar_slots=bays, mechanics=mechanics),
        simulation=replace(scenario.simulation, run_length=run_length, warmup=100.0, replications=30),
    )


def replications(scenario: Scenario) -> list[ModelResult]:
    model = DiscreteEventSimulationModel()
    return [model._run_once(scenario, scenario.simulation.seed + index) for index in range(scenario.simulation.replications)]


def assert_in_interval(values: list[float], expected: float) -> None:
    _, lower, upper = confidence_interval(values)
    assert lower < expected < upper, f"{expected} outside [{lower}, {upper}]"


@pytest.mark.parametrize("run_length", HORIZONS)
def test_fcfs_one_bay_matches_pollaczek_khinchine(run_length: float) -> None:
    runs = replications(setup(run_length, mode="fcfs"))
    assert_in_interval([run.waiting_time_total for run in runs], 2.5599)


@pytest.mark.parametrize("run_length", HORIZONS)
def test_fcfs_two_bays_matches_erlang_c(run_length: float) -> None:
    scenario = setup(run_length, mode="fcfs", bays=2)
    # One exponential with the arrival-weighted mean of both classes.
    pooled_mean = (
        scenario.arrival_rate_pm * scenario.pm_service.moment_mean + scenario.arrival_rate_cm * scenario.cm_service.moment_mean
    ) / scenario.combined_arrival_rate
    pooled = ServiceDistribution(kind="exponential", mean=pooled_mean)
    scenario = replace(scenario, pm_service=pooled, cm_service=pooled)
    assert_in_interval([run.waiting_time_total for run in replications(scenario)], 0.3224)


@pytest.mark.parametrize("run_length", HORIZONS)
def test_preemptive_one_bay_matches_exact_mg1(run_length: float) -> None:
    scenario = setup(run_length)
    exact = MGPriorityModel(1).run(scenario)
    runs = replications(scenario)
    assert_in_interval([run.waiting_time_hp for run in runs], exact.waiting_time_hp)
    assert_in_interval([run.waiting_time_lp for run in runs], exact.waiting_time_lp)


@pytest.mark.parametrize("run_length", HORIZONS)
def test_preemptive_two_bays_exponential_matches_exact_qbd(run_length: float) -> None:
    scenario = replace(
        setup(run_length, bays=2),
        pm_service=ServiceDistribution(kind="exponential", mean=5.0),
        cm_service=ServiceDistribution(kind="exponential", mean=11 / 3),
    )
    exact = MMPriorityModel(2).run(scenario)
    runs = replications(scenario)
    assert_in_interval([run.waiting_time_hp for run in runs], exact.waiting_time_hp)
    assert_in_interval([run.waiting_time_lp for run in runs], exact.waiting_time_lp)


@pytest.mark.parametrize("rho", [None, 0.75, 0.90], ids=["baseline", "rho_0.75", "rho_0.90"])
def test_little_law_per_origin_class(rho: float | None) -> None:
    scenario = load_scenario("config/scenarios/stable.json")
    if rho is not None:
        lambda_pm, lambda_cm = arrival_rates_for_target_rho(scenario, rho)
        scenario = replace(scenario, arrival_rate_pm=lambda_pm, arrival_rate_cm=lambda_cm)
    # Long horizon: at baseline load the escalated class only has a few tasks per 5000 h.
    run_length = 200000.0 if rho is None else 50000.0
    scenario = replace(scenario, simulation=replace(scenario.simulation, run_length=run_length, warmup=100.0, replications=30))
    observed = run_length - 100.0
    runs = replications(scenario)
    for tasks, wait, queue in [
        ("tasks_cm", "waiting_time_hp", "queue_length_hp"),
        ("tasks_pm_regular", "waiting_time_pm_regular", "queue_length_pm_regular"),
        ("tasks_pm_escalated", "waiting_time_pm_escalated", "queue_length_pm_escalated"),
    ]:
        throughput_wait = sum(getattr(run, tasks) * (getattr(run, wait) or 0.0) / observed for run in runs) / len(runs)
        lq = sum(getattr(run, queue) for run in runs) / len(runs)
        assert abs(lq - throughput_wait) / lq < 0.05, (tasks, lq, throughput_wait)
