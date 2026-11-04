from __future__ import annotations

from random import Random

from .scenario import ServiceDistribution


def sample_service_time(distribution: ServiceDistribution, rng: Random) -> float:
    """Draw one service duration for a simulated task."""
    kind = distribution.kind
    if kind == "exponential":
        return rng.expovariate(1.0 / distribution.mean)
    if kind == "normal":
        # Negative maintenance times are not meaningful; clamp at 0.001 h. Rare for S_PM: P(S <= 0) = 2.9e-7.
        value = rng.gauss(distribution.mean, distribution.stddev)
        return max(0.001, value)
    if kind == "triangular":
        return rng.triangular(distribution.min_value, distribution.max_value, distribution.mode)
    if kind == "deterministic":
        return distribution.mean
    raise ValueError(f"Unsupported distribution kind: {distribution.kind}")
