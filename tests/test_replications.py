import numpy as np

from desim.domain.scenario import load_scenario
from desim.experiments.replications import build_variant, confidence_interval, mser5_truncation


def test_mser5_removes_an_initial_transient() -> None:
    rng = np.random.default_rng(1)
    series = np.concatenate((np.full(500, 50.0), rng.normal(0.0, 1.0, 5000)))
    assert 450 <= mser5_truncation(series) <= 550


def test_mser5_keeps_a_stationary_series() -> None:
    assert mser5_truncation(np.random.default_rng(2).normal(0.0, 1.0, 5000)) < 2500


def test_build_variant_changes_only_what_is_asked() -> None:
    base = load_scenario("config/scenarios/stable.json")
    variant = build_variant(base, mechanics=16, escalation=False, queue_position="behind_cm")
    assert variant.resources.mechanics == 16 and variant.resources.hangar_slots == 2
    assert variant.priority.escalate_threshold is None
    assert variant.escalation.queue_position == "behind_cm"
    assert variant.arrival_rate_pm == base.arrival_rate_pm


def test_confidence_interval_for_one_value_has_no_width() -> None:
    assert confidence_interval([2.0]) == (2.0, 2.0, 2.0)
