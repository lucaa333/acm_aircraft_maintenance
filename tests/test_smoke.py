import unittest

from desim.domain.scenario import load_scenario
from desim.domain.scenario import PriorityPolicy, ResourceConfig, Scenario, ServiceDistribution, SimulationConfig
from desim.experiments.runner import ExperimentRunner
from desim.models.simulation.des_model import DiscreteEventSimulationModel
from desim.output.dataframes import results_to_dataframe


class SmokeTest(unittest.TestCase):
    def test_smoke_all_models(self) -> None:
        scenario = load_scenario("config/scenarios/stable.json")
        runner = ExperimentRunner(scenario)
        results = runner.run_many(["mm1_single", "mg1_single", "mm1_priority", "des"])
        self.assertEqual(len(results), 4)
        self.assertTrue(all(result.model_name for result in results))

    def test_des_handles_zero_arrivals(self) -> None:
        scenario = Scenario(
            name="zero",
            description="No arrivals should produce zero-valued metrics.",
            arrival_rate_pm=0.0,
            arrival_rate_cm=0.0,
            pm_service=ServiceDistribution(kind="deterministic", mean=1.0),
            cm_service=ServiceDistribution(kind="deterministic", mean=1.0),
            pm_mechanics=1,
            cm_mechanics=1,
            resources=ResourceConfig(hangar_slots=1, mechanics=2),
            priority=PriorityPolicy(mode="priority", escalate_threshold=5.0),
            simulation=SimulationConfig(run_length=20.0, warmup=5.0, replications=1, seed=1),
        )
        result = DiscreteEventSimulationModel().run(scenario)
        self.assertEqual(result.waiting_time_total, 0.0)
        self.assertEqual(result.queue_length_total, 0.0)
        self.assertEqual(result.escalation_rate, 0.0)

    def test_des_warmup_excludes_pre_warmup_arrivals(self) -> None:
        scenario = Scenario(
            name="warmup",
            description="A long pre-warmup job should not contaminate observed metrics.",
            arrival_rate_pm=0.2,
            arrival_rate_cm=0.0,
            pm_service=ServiceDistribution(kind="deterministic", mean=50.0),
            cm_service=ServiceDistribution(kind="deterministic", mean=1.0),
            pm_mechanics=1,
            cm_mechanics=1,
            resources=ResourceConfig(hangar_slots=1, mechanics=2),
            priority=PriorityPolicy(mode="priority", escalate_threshold=None),
            simulation=SimulationConfig(run_length=20.0, warmup=10.0, replications=1, seed=2),
        )
        result = DiscreteEventSimulationModel().run(scenario)
        self.assertEqual(result.waiting_time_hp, 0.0)
        self.assertEqual(result.mttr_cm, 0.0)

    def test_results_can_be_converted_to_dataframe(self) -> None:
        scenario = load_scenario("config/scenarios/stable.json")
        result = ExperimentRunner(scenario).run_one("des")
        dataframe = results_to_dataframe([result])
        self.assertEqual(len(dataframe), 1)
        self.assertIn("model_name", dataframe.columns)


if __name__ == "__main__":
    unittest.main()
