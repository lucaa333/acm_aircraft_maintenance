from dataclasses import replace
import logging

import pytest

from desim.domain.scenario import load_scenario
from desim.models.simulation.des_model import DiscreteEventSimulationModel, InvariantViolation, post_check_warnings


def scenario():
    return load_scenario("config/scenarios/stable.json")


def test_baseline_runs_without_violation() -> None:
    DiscreteEventSimulationModel().run(scenario())


def test_task_that_can_never_start_is_a_standstill() -> None:
    too_few = replace(scenario(), resources=replace(scenario().resources, mechanics=7))
    with pytest.raises(InvariantViolation, match="standstill"):
        DiscreteEventSimulationModel()._run_once(too_few, 42)


def test_invariants_can_be_switched_off() -> None:
    base = replace(scenario(), resources=replace(scenario().resources, mechanics=7))
    silent = replace(base, simulation=replace(base.simulation, check_invariants=False))
    DiscreteEventSimulationModel()._run_once(silent, 42)


def test_post_check_flags_a_wrong_theoretical_value() -> None:
    base = scenario()
    results = DiscreteEventSimulationModel().run_replications(base)
    assert post_check_warnings(replace(base, arrival_rate_pm=0.16), results)


def test_post_check_warnings_reach_the_log(caplog: pytest.LogCaptureFixture) -> None:
    class Shifted(DiscreteEventSimulationModel):
        def _run_once(self, scenario, seed, trace=None):
            return replace(super()._run_once(scenario, seed), hangar_utilization=0.9)

    base = scenario()
    with caplog.at_level(logging.WARNING):
        Shifted().run_replications(replace(base, simulation=replace(base.simulation, run_length=1000.0)))
    assert "hangar utilization" in caplog.text
