"""Exact M/M/c preemptive-resume priority queue with two classes and class-specific service rates.

This is the model of Gail, Hantler and Taylor (1992). Here it is solved numerically as a
quasi-birth-death process: the level is the number of LP tasks in the system, the phase is
the number of HP tasks in the system. HP tasks see an M/M/c queue of their own; LP tasks use
the servers HP tasks leave free. From level c upward the transitions no longer depend on the
level, so the levels above c follow the matrix-geometric form with the rate matrix R. Only the
HP count is cut, at a length with negligible probability mass.
"""
from __future__ import annotations

import numpy as np


def mmc_preemptive_waits(
    hp_rate: float,
    lp_rate: float,
    hp_mean: float,
    lp_mean: float,
    servers: int,
    hp_limit: int = 80,
) -> tuple[float, float]:
    """Mean waits (HP, LP) of a stable M/M/c preemptive-resume priority queue.

    The wait of a class is its mean time in the system minus its mean service time, so the LP
    wait includes the time an interrupted task spends waiting to resume.
    """
    hp_service, lp_service = 1.0 / hp_mean, 1.0 / lp_mean
    phases = hp_limit + 1

    def local(level: int) -> np.ndarray:
        # HP arrivals and completions within one level, with the diagonal set to all outflows.
        block = np.zeros((phases, phases))
        for hp in range(phases):
            if hp < hp_limit:
                block[hp, hp + 1] = hp_rate
            if hp > 0:
                block[hp, hp - 1] = min(hp, servers) * hp_service
        out = block.sum(axis=1) + lp_rate + np.diag(down(level))
        return block - np.diag(out)

    def down(level: int) -> np.ndarray:
        # LP completions: LP tasks use the servers that HP tasks leave free.
        return np.diag([min(level, max(servers - hp, 0)) * lp_service for hp in range(phases)])

    up = lp_rate * np.eye(phases)
    # From level c on, every free server holds an LP task, so the blocks no longer change.
    rate_matrix = _rate_matrix(up, local(servers), down(servers))

    # Balance equations of levels 0..c; level c also carries all levels above it through R.
    size = (servers + 1) * phases
    system = np.zeros((size, size))

    def block(row_level: int, col_level: int, matrix: np.ndarray) -> None:
        rows = slice(row_level * phases, (row_level + 1) * phases)
        cols = slice(col_level * phases, (col_level + 1) * phases)
        system[rows, cols] += matrix

    for level in range(servers + 1):
        own = local(level) if level < servers else local(servers) + rate_matrix @ down(servers)
        block(level, level, own)
        if level < servers:
            block(level, level + 1, up)
        if level > 0:
            block(level, level - 1, down(level))

    geometric = np.linalg.inv(np.eye(phases) - rate_matrix)
    normalize = np.concatenate([np.ones(servers * phases), geometric @ np.ones(phases)])
    equations = system.T.copy()
    equations[-1] = normalize
    right = np.zeros(size)
    right[-1] = 1.0
    probabilities = np.linalg.solve(equations, right)
    levels = [probabilities[level * phases : (level + 1) * phases] for level in range(servers + 1)]

    hp_counts = np.arange(phases, dtype=float)
    boundary_hp = sum(levels[level] @ hp_counts for level in range(servers))
    hp_in_system = boundary_hp + levels[servers] @ geometric @ hp_counts
    boundary_lp = sum(level * levels[level].sum() for level in range(servers))
    # Levels c, c+1, ...: sum over k >= 0 of (c + k) * pi_c R^k.
    lp_in_system = boundary_lp + levels[servers] @ (servers * geometric + rate_matrix @ geometric @ geometric) @ np.ones(phases)
    return hp_in_system / hp_rate - hp_mean, lp_in_system / lp_rate - lp_mean


def _rate_matrix(up: np.ndarray, middle: np.ndarray, down: np.ndarray) -> np.ndarray:
    """Matrix R of the QBD, from G by logarithmic reduction."""
    identity = np.eye(len(up))
    base = np.linalg.inv(-middle)
    high, low = base @ up, base @ down
    first_passage, transient = low.copy(), high.copy()
    for _ in range(100):
        step = identity - (high @ low + low @ high)
        high = np.linalg.solve(step, high @ high)
        low = np.linalg.solve(step, low @ low)
        increment = transient @ low
        first_passage += increment
        transient = transient @ high
        # Stop on the increment: the row sums of G cannot get closer to 1 than the rounding error.
        if np.abs(increment).max() < 1e-14:
            return up @ np.linalg.inv(-(middle + up @ first_passage))
    raise RuntimeError("Logarithmic reduction did not converge; is the queue stable?")
