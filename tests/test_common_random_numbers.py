"""R19: common random numbers. Variants with the same arrival rates draw exactly the same random numbers."""
from dataclasses import replace
from random import Random

import pytest

from desim.domain.scenario import arrival_rates_for_target_rho, load_scenario
from desim.models.simulation import des_model


class RecordingRandom(Random):
    streams: list[list[float]] = []

    def __init__(self, seed: int) -> None:
        self.draws: list[float] = []
        RecordingRandom.streams.append(self.draws)
        super().__init__(seed)

    def random(self) -> float:
        value = super().random()
        self.draws.append(value)
        return value


def draws(scenario, monkeypatch: pytest.MonkeyPatch) -> list[float]:
    RecordingRandom.streams = []
    monkeypatch.setattr(des_model, "Random", RecordingRandom)
    des_model.DiscreteEventSimulationModel()._run_once(scenario, 42)
    (stream,) = RecordingRandom.streams
    return stream


def test_variants_draw_the_same_random_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    base = load_scenario("config/scenarios/stable.json")
    lambda_pm, lambda_cm = arrival_rates_for_target_rho(base, 0.9)
    base = replace(base, arrival_rate_pm=lambda_pm, arrival_rate_cm=lambda_cm, simulation=replace(base.simulation, run_length=20000.0))
    variants = {
        "16 engineers": replace(base, resources=replace(base.resources, mechanics=16)),
        "3 bays": replace(base, resources=replace(base.resources, hangar_slots=3)),
        "no escalation": replace(base, priority=replace(base.priority, escalate_threshold=None)),
        "X = 0": replace(base, priority=replace(base.priority, escalate_threshold=0.0)),
        "X = 56 h": replace(base, priority=replace(base.priority, escalate_threshold=56.0)),
    }
    reference = draws(base, monkeypatch)
    assert len(reference) > 1000
    for label, variant in variants.items():
        assert draws(variant, monkeypatch) == reference, label
