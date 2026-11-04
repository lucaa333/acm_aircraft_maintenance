from __future__ import annotations

from dataclasses import dataclass

from desim.domain.metrics import ModelResult
from desim.domain.scenario import Scenario
from desim.models.analytical.single_priority import MGSinglePriorityModel, MMSinglePriorityModel
from desim.models.base import Model
from desim.models.priority_queue.multi_priority import MGPriorityModel, MMPriorityModel
from desim.models.simulation.des_model import DiscreteEventSimulationModel


@dataclass(slots=True)
class ExperimentRunner:
    # Central registry for all executable model variants.
    scenario: Scenario

    def available_models(self) -> dict[str, Model]:
        # The keys here are the CLI-facing identifiers.
        models: dict[str, Model] = {
            "mm1_single": MMSinglePriorityModel(servers=1),
            "mm2_single": MMSinglePriorityModel(servers=2),
            "mg1_single": MGSinglePriorityModel(servers=1),
            "mg2_single": MGSinglePriorityModel(servers=2),
            "mm1_priority": MMPriorityModel(servers=1),
            "mm2_priority": MMPriorityModel(servers=2),
            "mg1_priority": MGPriorityModel(servers=1),
            "mg2_priority": MGPriorityModel(servers=2),
            "des": DiscreteEventSimulationModel(),
        }
        return models

    def run_one(self, model_key: str) -> ModelResult:
        # Runs exactly one selected model.
        models = self.available_models()
        return models[model_key].run(self.scenario)

    def run_many(self, model_keys: list[str] | None = None) -> list[ModelResult]:
        # Used for full comparison tables across several model families.
        models = self.available_models()
        keys = model_keys or list(models.keys())
        return [models[key].run(self.scenario) for key in keys]
