#!/usr/bin/env bash
set -euo pipefail

# Resolve paths from the repository root, even when this script is called elsewhere.
cd "$(dirname "$0")"

# Create the project-local environment on the first run; keep it on later runs.
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

# Install declared dependencies, then run every reproduction step with this Python.
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_project.py "$@"
