"""Checks of the paper configuration: moments, single-class formulas and class statistics."""
from dataclasses import replace
import unittest

from desim.domain.metrics import t_critical_95
from desim.domain.scenario import ServiceDistribution, arrival_rates_for_target_rho, load_scenario
from desim.experiments.runner import ExperimentRunner
from desim.models.analytical.single_priority import analytical_inputs
from desim.models.priority_queue.multi_priority import MGPriorityModel, MMPriorityModel
from desim.models.simulation.des_model import DiscreteEventSimulationModel


def base_scenario(replications: int = 30):
    scenario = load_scenario("config/scenarios/stable.json")
    return replace(
        scenario,
        simulation=replace(scenario.simulation, run_length=5000.0, warmup=100.0, replications=replications),
    )


class AnalyticalTest(unittest.TestCase):
    def test_triangular_moments(self) -> None:
        """R4: the config gives the mode; analytics and rho use the moments."""
        cm = base_scenario().cm_service
        self.assertEqual((cm.min_value, cm.mode, cm.max_value, cm.mean), (1.5, 3.5, 6.0, None))
        self.assertAlmostEqual(cm.moment_mean, 11 / 3)
        self.assertAlmostEqual(cm.moment_stddev**2, 15.25 / 18)
        with self.assertRaises(ValueError):
            ServiceDistribution(kind="triangular", mean=3.5, min_value=1.5, max_value=6.0)

    def test_analytical_inputs(self) -> None:
        """R40: per class and pooled; the pooled CV^2 includes the gap between the class means."""
        rows = {row["class"]: row for row in analytical_inputs(base_scenario())}
        self.assertAlmostEqual(rows["combined"]["arrival_rate"], 0.11)
        self.assertAlmostEqual(rows["combined"]["mean_service"], 4.6364, places=4)
        self.assertAlmostEqual(rows["combined"]["scv"], 0.0610, places=4)
        self.assertAlmostEqual(rows["CM"]["variance"], 15.25 / 18)
        self.assertAlmostEqual(rows["PM"]["scv"], 0.04)

    def test_t_critical_values(self) -> None:
        for df, expected in [(1, 12.7062), (4, 2.7764), (29, 2.0452), (299, 1.9679)]:
            self.assertAlmostEqual(t_critical_95(df), expected, places=3)

    def test_model_names_separate_the_two_exponential_models(self) -> None:
        self.assertIn("pooled exp.", ExperimentRunner(base_scenario()).run_one("mm1_single").model_name)
        self.assertIn("class-specific exp.", MMPriorityModel(1).run(base_scenario()).model_name)

    def test_table_3_single_class_values(self) -> None:
        runner = ExperimentRunner(base_scenario())
        for key, wait, queue in [
            ("mm1_single", 4.8256, 0.5308),
            ("mm2_single", 0.3224, 0.0355),
            ("mg1_single", 2.5599, 0.2816),
            ("mg2_single", 0.1711, 0.0188),
        ]:
            result = runner.run_one(key)
            self.assertAlmostEqual(result.waiting_time_total, wait, places=4)
            self.assertAlmostEqual(result.queue_length_total, queue, places=4)

    def test_total_wait_is_arrival_weighted_mean_of_classes(self) -> None:
        scenario = base_scenario()
        result = MGPriorityModel(2).run(scenario)
        weighted = (
            scenario.arrival_rate_cm * result.waiting_time_hp + scenario.arrival_rate_pm * result.waiting_time_lp
        ) / scenario.combined_arrival_rate
        self.assertAlmostEqual(result.waiting_time_total, weighted)

    def test_exponential_service_changes_class_waits(self) -> None:
        scenario = base_scenario()
        self.assertGreater(MMPriorityModel(1).run(scenario).waiting_time_hp, MGPriorityModel(1).run(scenario).waiting_time_hp)

    def test_overload_is_infinite(self) -> None:
        scenario = base_scenario()
        overloaded = replace(scenario, arrival_rate_pm=1.0)
        self.assertEqual(MGPriorityModel(2).run(overloaded).waiting_time_hp, float("inf"))

    def test_fcfs_scenario_is_rejected(self) -> None:
        scenario = base_scenario()
        fcfs = replace(scenario.priority, mode="fcfs", escalate_threshold=None)
        with self.assertRaises(ValueError):
            MGPriorityModel(1).run(replace(scenario, priority=fcfs))

    def test_target_rho_rates_use_true_means(self) -> None:
        scenario = base_scenario()
        lambda_pm, lambda_cm = arrival_rates_for_target_rho(scenario, 0.9)
        rho = (lambda_pm * 5.0 + lambda_cm * 11 / 3) / 2
        self.assertAlmostEqual(rho, 0.9)
        self.assertAlmostEqual(lambda_pm / lambda_cm, 0.08 / 0.03)


class SimulationTest(unittest.TestCase):
    def test_little_law_holds_per_origin_class(self) -> None:
        scenario = base_scenario()
        lambda_pm, lambda_cm = arrival_rates_for_target_rho(scenario, 0.75)
        scenario = replace(scenario, arrival_rate_pm=lambda_pm, arrival_rate_cm=lambda_cm)
        result = DiscreteEventSimulationModel().run(scenario)
        escalated = result.escalation_rate
        self.assertGreater(escalated, 0.01)
        for rate, wait, queue in [
            (lambda_cm, result.waiting_time_hp, result.queue_length_hp),
            (lambda_pm, result.waiting_time_lp, result.queue_length_lp),
            (lambda_pm * (1 - escalated), result.waiting_time_pm_regular, result.queue_length_pm_regular),
            (lambda_pm * escalated, result.waiting_time_pm_escalated, result.queue_length_pm_escalated),
        ]:
            self.assertAlmostEqual(rate * wait / queue, 1.0, delta=0.05)

    def test_mttr_counts_original_cm_only(self) -> None:
        scenario = base_scenario(replications=5)
        scenario = replace(scenario, arrival_rate_pm=0.16, arrival_rate_cm=0.06)
        result = DiscreteEventSimulationModel().run(scenario)
        self.assertGreater(result.escalation_rate, 0.0)
        self.assertAlmostEqual(result.mttr_cm - result.waiting_time_hp, 11 / 3, delta=0.1)


if __name__ == "__main__":
    unittest.main()
