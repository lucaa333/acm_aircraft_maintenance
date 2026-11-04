"""Analytical preemptive-resume priority models: exact M/G/1, exact M/M/c (Gail, Hantler and Taylor), Bondi-Buzen."""
from dataclasses import replace

import numpy as np
import pytest

from desim.domain.scenario import arrival_rates_for_target_rho, load_scenario
from desim.experiments.runner import ExperimentRunner
from desim.models.analytical.common import mm_s_metrics
from desim.models.priority_queue.exact_mm_preemptive import mmc_preemptive_waits
from desim.models.priority_queue.multi_priority import MGPriorityModel, MMPriorityModel, _preemptive_resume_mg1, _preemptive_waits

HP_MEAN, LP_MEAN = 11 / 3, 5.0
HP_RATE, LP_RATE = 0.03, 0.08


def rates(rho: float | None) -> tuple[float, float]:
    base = load_scenario("config/scenarios/stable.json")
    if rho is None:
        return base.arrival_rate_cm, base.arrival_rate_pm
    lambda_pm, lambda_cm = arrival_rates_for_target_rho(base, rho)
    return lambda_cm, lambda_pm


def brute_force(hp_rate, lp_rate, hp_mean, lp_mean, servers, hp_cut, lp_cut):
    """Independent check: the full two-dimensional chain, both queues cut, solved directly."""
    index = lambda hp, lp: hp * (lp_cut + 1) + lp  # noqa: E731
    size = (hp_cut + 1) * (lp_cut + 1)
    generator = np.zeros((size, size))
    for hp in range(hp_cut + 1):
        for lp in range(lp_cut + 1):
            state = index(hp, lp)
            if hp < hp_cut:
                generator[state, index(hp + 1, lp)] += hp_rate
            if lp < lp_cut:
                generator[state, index(hp, lp + 1)] += lp_rate
            if hp:
                generator[state, index(hp - 1, lp)] += min(hp, servers) / hp_mean
            free = max(servers - hp, 0)
            if lp and free:
                generator[state, index(hp, lp - 1)] += min(lp, free) / lp_mean
    generator -= np.diag(generator.sum(axis=1))
    system = generator.T.copy()
    system[-1] = 1.0
    right = np.zeros(size)
    right[-1] = 1.0
    probabilities = np.linalg.solve(system, right)
    hp_counts = np.repeat(np.arange(hp_cut + 1), lp_cut + 1)
    lp_counts = np.tile(np.arange(lp_cut + 1), hp_cut + 1)
    return probabilities @ hp_counts / hp_rate - hp_mean, probabilities @ lp_counts / lp_rate - lp_mean


def test_paper_configuration_uses_preemptive_models() -> None:
    scenario = load_scenario("config/scenarios/stable.json")
    assert MMPriorityModel(2).run(scenario).extra["approximation"] == "exact-qbd-preemptive"
    assert MGPriorityModel(2).run(scenario).extra["approximation"] == "bondi-buzen-approximation"
    assert MGPriorityModel(1).run(scenario).extra["approximation"] == "preemptive-resume-mg1"
    assert "Preemptive-Resume" in MGPriorityModel(2).run(scenario).model_name


def test_table_4_priority_values_preemptive() -> None:
    runner = ExperimentRunner(load_scenario("config/scenarios/stable.json"))
    for key, hp, lp, total in [
        ("mm1_priority", 0.4532, 6.1289, 4.5810),
        ("mm2_priority", 0.0111, 0.4211, 0.3093),
        ("mg1_priority", 0.2409, 3.4943, 2.6070),
        ("mg2_priority", 0.0059, 0.2466, 0.1809),
    ]:
        result = runner.run_one(key)
        assert (result.waiting_time_hp, result.waiting_time_lp, result.waiting_time_total) == pytest.approx((hp, lp, total), abs=1e-4)


def test_one_server_qbd_reproduces_the_mg1_formula() -> None:
    hp_time, lp_time = _preemptive_resume_mg1(HP_RATE, LP_RATE, HP_MEAN, LP_MEAN, 2 * HP_MEAN**2, 2 * LP_MEAN**2)
    assert mmc_preemptive_waits(HP_RATE, LP_RATE, HP_MEAN, LP_MEAN, 1) == pytest.approx((hp_time - HP_MEAN, lp_time - LP_MEAN))


@pytest.mark.parametrize("servers", [2, 3])
def test_equal_service_rates_reproduce_erlang_c(servers: int) -> None:
    hp_rate, lp_rate, mean = 0.05 * servers, 0.15 * servers, 4.0
    hp_wait, lp_wait = mmc_preemptive_waits(hp_rate, lp_rate, mean, mean, servers, hp_limit=60)
    _, all_wait, _ = mm_s_metrics(hp_rate + lp_rate, mean, servers)
    _, hp_alone, _ = mm_s_metrics(hp_rate, mean, servers)
    assert hp_wait == pytest.approx(hp_alone)
    assert (hp_rate * hp_wait + lp_rate * lp_wait) / (hp_rate + lp_rate) == pytest.approx(all_wait)


@pytest.mark.parametrize("rho, lp_cut", [(None, 120), (0.50, 200), (0.75, 500)])
def test_two_servers_match_brute_force_chain(rho: float | None, lp_cut: int) -> None:
    hp_rate, lp_rate = rates(rho)
    expected = brute_force(hp_rate, lp_rate, HP_MEAN, LP_MEAN, 2, 25, lp_cut)
    assert mmc_preemptive_waits(hp_rate, lp_rate, HP_MEAN, LP_MEAN, 2) == pytest.approx(expected, abs=1e-6)


def test_hp_limit_does_not_matter() -> None:
    hp_rate, lp_rate = rates(0.90)
    assert mmc_preemptive_waits(hp_rate, lp_rate, HP_MEAN, LP_MEAN, 2, hp_limit=120) == pytest.approx(
        mmc_preemptive_waits(hp_rate, lp_rate, HP_MEAN, LP_MEAN, 2, hp_limit=40)
    )


def test_hp_wait_ignores_lp_tasks() -> None:
    hp_rate, lp_rate = rates(0.75)
    assert mmc_preemptive_waits(hp_rate, lp_rate, HP_MEAN, LP_MEAN, 2)[0] == pytest.approx(
        mmc_preemptive_waits(hp_rate, lp_rate / 4, HP_MEAN, LP_MEAN, 2)[0]
    )


def test_bondi_buzen_is_exact_for_one_server() -> None:
    rate = HP_RATE + LP_RATE
    mean = (HP_RATE * HP_MEAN + LP_RATE * LP_MEAN) / rate
    second = (HP_RATE * 2 * HP_MEAN**2 + LP_RATE * 2 * LP_MEAN**2) / rate
    approx = _preemptive_waits(HP_RATE, LP_RATE, HP_MEAN, LP_MEAN, 2 * HP_MEAN**2, 2 * LP_MEAN**2, mean, second - mean**2, 1)
    assert approx == pytest.approx(mmc_preemptive_waits(HP_RATE, LP_RATE, HP_MEAN, LP_MEAN, 1))


@pytest.mark.parametrize("rho", [None, 0.50, 0.75, 0.85, 0.90])
def test_bondi_buzen_close_to_exact_for_exponential_service(rho: float | None) -> None:
    scenario = load_scenario("config/scenarios/stable.json")
    if rho is not None:
        lambda_pm, lambda_cm = arrival_rates_for_target_rho(scenario, rho)
        scenario = replace(scenario, arrival_rate_pm=lambda_pm, arrival_rate_cm=lambda_cm)
    exact = MMPriorityModel(2).run(scenario)
    approx = MMPriorityModel(2, exact=False).run(scenario)
    assert approx.extra["approximation"] == "bondi-buzen-approximation"
    assert approx.waiting_time_hp == pytest.approx(exact.waiting_time_hp, rel=0.01)
    assert approx.waiting_time_lp == pytest.approx(exact.waiting_time_lp, rel=0.01)
