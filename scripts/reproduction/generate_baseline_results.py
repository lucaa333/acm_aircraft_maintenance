from __future__ import annotations


from desim.experiments.pipeline_context import RESULTS_DIR, load_base_scenario
from desim.experiments.runner import ExperimentRunner
from desim.output.dataframes import export_dataframe, results_to_dataframe


OUTPUT_DIR = RESULTS_DIR / "baseline"


def export_model_result(runner: ExperimentRunner, scenario_name: str, model_key: str) -> None:
    dataframe = results_to_dataframe([runner.run_one(model_key)])
    export_dataframe(dataframe, OUTPUT_DIR / f"{scenario_name}_{model_key}_results.csv")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scenario = load_base_scenario()
    runner = ExperimentRunner(scenario)

    # Analytical models only; the DES baseline is Tab. 5 (high_precision_evaluation.py).
    analytical = [key for key in runner.available_models() if key != "des"]
    export_dataframe(results_to_dataframe(runner.run_many(analytical)), OUTPUT_DIR / f"{scenario.name}_all_results.csv")

    for model_key in ["mg1_single", "mg1_priority"]:
        export_model_result(runner, scenario.name, model_key)

    print(f"Baseline results exported to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
