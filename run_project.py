from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

REQUIRED_IMPORTS = {
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pandas": "pandas",
    "pytest": "pytest",
    "desim": "this project (pip install -e .)",
}


def command(label: str, args: list[str]) -> tuple[str, list[str]]:
    return label, args


REPRODUCTION_PIPELINE = [
    command("Analytical models (Tab. 3, Tab. 4)", ["scripts/reproduction/generate_baseline_results.py"]),
    command("Verification against analytical references", ["scripts/analysis/run_verification.py"]),
    command("DES baseline (Tab. 5)", ["scripts/reproduction/high_precision_evaluation.py"]),
    command("Analytical models against the DES", ["scripts/analysis/run_analytical_vs_des.py"]),
    command("Load sensitivity", ["scripts/reproduction/run_high_precision_target_rho_sensitivity_analysis.py"]),
    command("Ablation engineers x escalation", ["scripts/analysis/run_ablation.py"]),
    command("Capacity variants", ["scripts/analysis/run_capacity_variants.py"]),
    command("Placement of escalated PM", ["scripts/analysis/run_escalation_variants.py"]),
    command("Escalation threshold sweep", ["scripts/analysis/run_threshold_sweep.py"]),
    command("Horizon check", ["scripts/analysis/run_horizon_check.py"]),
    command("Warm-up and run length", ["scripts/analysis/simulation_configuration_analysis.py"]),
    command("MSER-5", ["scripts/analysis/run_mser5.py"]),
    command("Warm-up check at high load", ["scripts/analysis/run_warmup_check.py"]),
]


def missing_dependencies() -> list[str]:
    missing = []
    for import_name, package_name in REQUIRED_IMPORTS.items():
        if importlib.util.find_spec(import_name) is None:
            missing.append(package_name)
    return missing


def run_step(label: str, args: list[str]) -> None:
    """Run one pipeline command from the project root and stop if it fails."""
    print(f"\n== {label} ==", flush=True)
    subprocess.run([sys.executable, *args], cwd=PROJECT_ROOT, check=True)


def clean_runtime_caches() -> None:
    """Remove the bytecode caches Python leaves next to the sources."""
    for pattern in ("__pycache__",):
        for path in PROJECT_ROOT.rglob(pattern):
            if path.is_dir():
                shutil.rmtree(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete DES reproduction pipeline from one entry point.")
    parser.add_argument("--skip-tests", action="store_true", help="Skip all tests.")
    parser.add_argument("--full", action="store_true", help="Also run the slow long-horizon verification tests.")
    return parser.parse_args()


def main() -> None:
    """Check dependencies, run the reproduction steps, then clean runtime caches."""
    args = parse_args()
    missing = missing_dependencies()
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(
            f"Missing Python packages: {joined}\n"
            "Install dependencies first with:\n"
            "  python3 -m pip install -r requirements.txt\n"
            "or run:\n"
            "  bash run_project.sh"
        )

    if not args.skip_tests:
        run_step("Run tests", ["-m", "pytest", "-q", *([] if args.full else ["-m", "not slow"])])

    for label, command_args in REPRODUCTION_PIPELINE:
        run_step(label, command_args)

    clean_runtime_caches()
    print("\nPipeline completed successfully.", flush=True)
    print("Generated outputs are under outputs/.", flush=True)


if __name__ == "__main__":
    main()
