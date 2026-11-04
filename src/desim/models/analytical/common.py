from __future__ import annotations

import math


def erlang_c(arrival_rate: float, service_rate: float, servers: int) -> tuple[float, float]:
    # Erlang-C gives utilization and probability of waiting in an M/M/s system.
    offered = arrival_rate / service_rate
    rho = arrival_rate / (servers * service_rate)
    if rho >= 1:
        return rho, 1.0
    series = sum((offered**n) / math.factorial(n) for n in range(servers))
    tail = (offered**servers) / math.factorial(servers) * (1.0 / (1.0 - rho))
    p0 = 1.0 / (series + tail)
    pw = tail * p0
    return rho, pw


def mm_s_metrics(arrival_rate: float, mean_service: float, servers: int) -> tuple[float, float, float]:
    # Exact queue metrics for the M/M/s models used as the analytical baseline (Erlang C; Shortle et al.).
    service_rate = 1.0 / mean_service
    rho, pw = erlang_c(arrival_rate, service_rate, servers)
    if rho >= 1:
        return rho, float("inf"), float("inf")
    wq = pw / (servers * service_rate - arrival_rate)
    lq = arrival_rate * wq
    return rho, wq, lq


def mg1_metrics(arrival_rate: float, mean_service: float, variance_service: float) -> tuple[float, float, float]:
    # Pollaczek-Khinchine formula for the single-server M/G/1 case (Shortle et al.).
    rho = arrival_rate * mean_service
    if rho >= 1:
        return rho, float("inf"), float("inf")
    second_moment = variance_service + mean_service**2
    wq = arrival_rate * second_moment / (2.0 * (1.0 - rho))
    lq = arrival_rate * wq
    return rho, wq, lq


def mgs_approx_metrics(arrival_rate: float, mean_service: float, scv: float, servers: int) -> tuple[float, float, float]:
    # For M/G/s there is no simple exact closed form, so the M/M/s wait is scaled by (1 + CV^2) / 2
    # (Whitt, QNA, Eq. (70)); the output label is whitt-eq70.
    rho, wq_mm, _ = mm_s_metrics(arrival_rate, mean_service, servers)
    if rho >= 1:
        return rho, float("inf"), float("inf")
    variability_factor = (1.0 + scv) / 2.0
    wq = variability_factor * wq_mm
    lq = arrival_rate * wq
    return rho, wq, lq
