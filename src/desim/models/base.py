from __future__ import annotations

from abc import ABC, abstractmethod

from desim.domain.metrics import ModelResult
from desim.domain.scenario import Scenario


class Model(ABC):
    name: str

    @abstractmethod
    def run(self, scenario: Scenario) -> ModelResult:
        raise NotImplementedError
