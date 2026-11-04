"""Where the pipeline reads its scenario and writes its results."""
from __future__ import annotations

from pathlib import Path

from desim.domain.scenario import Scenario, load_scenario

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_SCENARIO_LABEL = "config/scenarios/stable.json"  # recorded in metadata, so outputs do not depend on the checkout
BASE_SCENARIO_PATH = PROJECT_ROOT / BASE_SCENARIO_LABEL
RESULTS_DIR = PROJECT_ROOT / "outputs/results"


def load_base_scenario() -> Scenario:
    return load_scenario(BASE_SCENARIO_PATH)
