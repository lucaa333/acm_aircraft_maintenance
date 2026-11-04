from __future__ import annotations

from dataclasses import replace

from desim.domain.metrics import ModelResult
from desim.domain.scenario import Scenario
from desim.models.base import Model
from desim.models.analytical.common import mg1_metrics, mgs_approx_metrics
from desim.models.priority_queue.exact_mm_preemptive import mmc_preemptive_waits


class MMPriorityModel(Model):
    # Same basic M/M/s queue as above, but now with separate HP and LP classes, served preemptive-resume.
    # Exact by default; exact=False gives the Bondi-Buzen approximation that M/G/c has to use.
    def __init__(self, servers: int, exact: bool = True) -> None:
        self.servers = servers
        self.exact = exact
        self.name = f"M/M/{servers} Multi Priority Preemptive-Resume (class-specific exp.)"

    def run(self, scenario: Scenario) -> ModelResult:
        result = _priority_result(scenario, self.name, self.servers, exponential=True)
        if not self.exact or self.servers == 1 or result.hangar_utilization >= 1:
            return result
        hp_rate, lp_rate = scenario.arrival_rate_cm, scenario.arrival_rate_pm
        hp_wait, lp_wait = mmc_preemptive_waits(
            hp_rate, lp_rate, scenario.cm_service.moment_mean, scenario.pm_service.moment_mean, self.servers
        )
        return replace(
            result,
            waiting_time_total=(hp_rate * hp_wait + lp_rate * lp_wait) / (hp_rate + lp_rate),
            queue_length_total=hp_rate * hp_wait + lp_rate * lp_wait,
            waiting_time_hp=hp_wait,
            waiting_time_lp=lp_wait,
            queue_length_hp=hp_rate * hp_wait,
            queue_length_lp=lp_rate * lp_wait,
            extra={"approximation": "exact-qbd-preemptive"},
        )


class MGPriorityModel(Model):
    # M/G/s priority queue with separate HP and LP classes, served preemptive-resume.
    # Use this when service-time variability matters and you still want an analytical/queueing view.
    def __init__(self, servers: int) -> None:
        self.servers = servers
        self.name = f"M/G/{servers} Multi Priority Preemptive-Resume"

    def run(self, scenario: Scenario) -> ModelResult:
        return _priority_result(scenario, self.name, self.servers, exponential=False)


def _priority_result(scenario: Scenario, name: str, servers: int, exponential: bool) -> ModelResult:
    """Preemptive-resume class waits: exact M/G/1 result for one server, Bondi and Buzen for several.

    HP corresponds to CM, LP corresponds to PM. These models do not represent explicit hangar
    slots, mechanic pools, escalation or the set-up time after a preemption; the DES is the
    reference for those.
    """
    if scenario.priority.mode != "priority":
        raise ValueError("The analytical priority models need two priority classes, not FCFS.")

    hp_rate = scenario.arrival_rate_cm
    lp_rate = scenario.arrival_rate_pm
    total_rate = hp_rate + lp_rate
    if total_rate <= 0:
        return ModelResult(
            model_name=name,
            scenario_name=scenario.name,
            waiting_time_total=0.0,
            queue_length_total=0.0,
            hangar_utilization=0.0,
            waiting_time_hp=0.0,
            waiting_time_lp=0.0,
            queue_length_hp=0.0,
            queue_length_lp=0.0,
        )

    hp_mean = scenario.cm_service.moment_mean
    lp_mean = scenario.pm_service.moment_mean
    hp_second = 2.0 * hp_mean**2 if exponential else scenario.cm_service.moment_stddev**2 + hp_mean**2
    lp_second = 2.0 * lp_mean**2 if exponential else scenario.pm_service.moment_stddev**2 + lp_mean**2

    mean = (hp_rate * hp_mean + lp_rate * lp_mean) / total_rate
    variance = (hp_rate * hp_second + lp_rate * lp_second) / total_rate - mean**2

    rho = total_rate * mean / servers
    if rho >= 1:
        hp_wait = lp_wait = float("inf")
    else:
        hp_wait, lp_wait = _preemptive_waits(
            hp_rate, lp_rate, hp_mean, lp_mean, hp_second, lp_second, mean, variance, servers
        )

    queue_hp = hp_rate * hp_wait
    queue_lp = lp_rate * lp_wait
    return ModelResult(
        model_name=name,
        scenario_name=scenario.name,
        waiting_time_total=(queue_hp + queue_lp) / total_rate,
        queue_length_total=queue_hp + queue_lp,
        hangar_utilization=rho,
        waiting_time_hp=hp_wait,
        waiting_time_lp=lp_wait,
        queue_length_hp=queue_hp,
        queue_length_lp=queue_lp,
        extra={"approximation": "bondi-buzen-approximation" if servers > 1 else "preemptive-resume-mg1"},
    )


def _preemptive_resume_mg1(
    hp_rate: float,
    lp_rate: float,
    hp_mean: float,
    lp_mean: float,
    hp_second: float,
    lp_second: float,
) -> tuple[float, float]:
    """Exact M/G/1 preemptive-resume mean times in system (HP, LP) from class rates and service moments.

    Kleinrock, Queueing Systems, Vol. 2; Shortle et al. An HP task sees only HP tasks; an LP task waits
    for all HP work, including HP tasks that arrive during its own service.
    """
    rho_hp = hp_rate * hp_mean
    rho = rho_hp + lp_rate * lp_mean
    hp_time = hp_mean + hp_rate * hp_second / (2.0 * (1.0 - rho_hp))
    lp_time = lp_mean / (1.0 - rho_hp) + (hp_rate * hp_second + lp_rate * lp_second) / (2.0 * (1.0 - rho_hp) * (1.0 - rho))
    return hp_time, lp_time


def _fcfs_ratio(rate: float, mean: float, second: float, servers: int) -> float:
    """FCFS wait of M/G/c over the wait of one c-times-faster server, both from the scenario's approximations."""
    if servers == 1:
        return 1.0
    scv = (second - mean**2) / mean**2
    _, slow, _ = mgs_approx_metrics(rate, mean, scv, servers)
    _, fast, _ = mg1_metrics(rate, mean / servers, (second - mean**2) / servers**2)
    return slow / fast


def _preemptive_waits(
    hp_rate: float,
    lp_rate: float,
    hp_mean: float,
    lp_mean: float,
    hp_second: float,
    lp_second: float,
    mean: float,
    variance: float,
    servers: int,
) -> tuple[float, float]:
    """Preemptive-resume class waits after Bondi and Buzen (1984); exact for one server.

    The c servers are replaced by one server c times as fast, where the M/G/1 preemptive-resume
    result is exact. The waiting time of the top p classes taken together is then scaled by the
    FCFS ratio of the c-server system to the fast server, computed for those p classes. The LP
    time follows from the totals, R_LP = (Lambda R(2) - lambda_HP R(1)) / lambda_LP. A wait is the
    time in system minus the service time, so it includes waiting to resume after a preemption.
    """
    fast_hp, fast_lp = _preemptive_resume_mg1(
        hp_rate, lp_rate, hp_mean / servers, lp_mean / servers, hp_second / servers**2, lp_second / servers**2
    )
    rate = hp_rate + lp_rate
    second = variance + mean**2
    # Top class alone: on the fast server its wait is fast_hp - hp_mean / c.
    hp_time = hp_mean + _fcfs_ratio(hp_rate, hp_mean, hp_second, servers) * (fast_hp - hp_mean / servers)
    # Both classes together, arrival-weighted.
    fast_both = (hp_rate * fast_hp + lp_rate * fast_lp) / rate
    both_time = mean + _fcfs_ratio(rate, mean, second, servers) * (fast_both - mean / servers)
    lp_time = (rate * both_time - hp_rate * hp_time) / lp_rate
    return hp_time - hp_mean, lp_time - lp_mean
