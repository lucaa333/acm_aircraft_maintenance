from __future__ import annotations

import argparse
from pathlib import Path

from desim.domain.scenario import load_scenario
from desim.experiments.runner import ExperimentRunner
from desim.output.formatting import format_results


def build_parser() -> argparse.ArgumentParser:
    # Minimal CLI: choose a scenario file and either one model or all models.
    parser = argparse.ArgumentParser(description="Run aircraft maintenance model variants.")
    parser.add_argument("--scenario", required=True, help="Path to scenario JSON file.")
    parser.add_argument(
        "--model",
        default="all",
        help="Model key to run (e.g. mm1_single, mg2_priority, des) or 'all'.",
    )
    return parser


def main() -> None:
    # The CLI stays intentionally thin; orchestration happens in ExperimentRunner.
    parser = build_parser()
    args = parser.parse_args()
    scenario = load_scenario(Path(args.scenario))
    runner = ExperimentRunner(scenario)
    if args.model == "all":
        results = runner.run_many()
    else:
        results = [runner.run_one(args.model)]
    print(format_results(results))


if __name__ == "__main__":
    main()
