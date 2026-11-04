from __future__ import annotations

from desim.domain.metrics import ModelResult
from desim.domain.scenario import Scenario
from desim.models.base import Model

from .common import mg1_metrics, mgs_approx_metrics, mm_s_metrics


class MMSinglePriorityModel(Model):
    # M/M/s:
    # First M = exponential interarrival times.
    # Second M = exponential service times.
    # s = number of parallel servers.
    # "Single Priority" means PM and CM are merged into one aggregated stream.
    def __init__(self, servers: int) -> None:
        self.servers = servers
        self.name = f"M/M/{servers} Single Priority (pooled exp.)"

    def run(self, scenario: Scenario) -> ModelResult:
        mean_service = _weighted_mean_service(scenario)
        rho, wq, lq = mm_s_metrics(scenario.combined_arrival_rate, mean_service, self.servers)
        return ModelResult(
            model_name=self.name,
            scenario_name=scenario.name,
            waiting_time_total=wq,
            queue_length_total=lq,
            hangar_utilization=rho,
        )


class MGSinglePriorityModel(Model):
    # M/G/s:
    # First M = exponential interarrival times.
    # G = general service-time distribution instead of exponential.
    # s = number of parallel servers.
    # This captures variability better than M/M/s, but still without explicit HP/LP behavior.
    def __init__(self, servers: int) -> None:
        self.servers = servers
        self.name = f"M/G/{servers} Single Priority"

    def run(self, scenario: Scenario) -> ModelResult:
        mean_service = _weighted_mean_service(scenario)
        variance = _weighted_variance(scenario)
        if self.servers == 1:
            rho, wq, lq = mg1_metrics(scenario.combined_arrival_rate, mean_service, variance)
        else:
            scv = variance / (mean_service**2) if mean_service else 0.0
            rho, wq, lq = mgs_approx_metrics(scenario.combined_arrival_rate, mean_service, scv, self.servers)
        return ModelResult(
            model_name=self.name,
            scenario_name=scenario.name,
            waiting_time_total=wq,
            queue_length_total=lq,
            hangar_utilization=rho,
            # Exact for one server; for several, Erlang C times (1 + CV^2) / 2 after Whitt, QNA, Eq. (70).
            extra={"approximation": "whitt-eq70" if self.servers > 1 else "pollaczek-khinchine"},
        )


def _weighted_mean_service(scenario: Scenario) -> float:
    # Single-priority models need one aggregate mean service time.
    total = scenario.combined_arrival_rate
    if total == 0:
        return 0.0
    return (
        scenario.arrival_rate_pm * scenario.pm_service.moment_mean
        + scenario.arrival_rate_cm * scenario.cm_service.moment_mean
    ) / total


def _weighted_variance(scenario: Scenario) -> float:
    # Variance is mixed across PM and CM, including the gap between their means.
    total = scenario.combined_arrival_rate
    if total == 0:
        return 0.0
    mean = _weighted_mean_service(scenario)
    pm_weight = scenario.arrival_rate_pm / total
    cm_weight = scenario.arrival_rate_cm / total
    pm_var = scenario.pm_service.moment_stddev**2 + (scenario.pm_service.moment_mean - mean) ** 2
    cm_var = scenario.cm_service.moment_stddev**2 + (scenario.cm_service.moment_mean - mean) ** 2
    return pm_weight * pm_var + cm_weight * cm_var


def analytical_inputs(scenario: Scenario) -> list[dict[str, float | str]]:
    """Inputs of the analytical models per class and for the pooled stream, the mixture of both classes."""
    classes = [
        ("PM", scenario.arrival_rate_pm, scenario.pm_service.moment_mean, scenario.pm_service.moment_stddev**2),
        ("CM", scenario.arrival_rate_cm, scenario.cm_service.moment_mean, scenario.cm_service.moment_stddev**2),
        ("combined", scenario.combined_arrival_rate, _weighted_mean_service(scenario), _weighted_variance(scenario)),
    ]
    return [
        {
            "class": label,
            "arrival_rate": rate,
            "mean_service": mean,
            "sd_service": variance**0.5,
            "second_moment": variance + mean**2,
            "variance": variance,
            "scv": variance / mean**2,
        }
        for label, rate, mean, variance in classes
    ]
