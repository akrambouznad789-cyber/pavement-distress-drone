#!/usr/bin/env bash
# Run Airflow against this repo's DAG, e.g.:
#   scripts/airflow.sh standalone                       # web UI on http://localhost:8080
#   scripts/airflow.sh dags test pavement_pipeline      # one full run in the terminal, no UI
#
# Assumes Airflow is installed in ~/airflow-venv (see the README's Orchestration section).
# On Windows, run this from WSL: Airflow only runs on Linux/macOS.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export AIRFLOW_HOME="${AIRFLOW_HOME:-$HOME/airflow}"          # Airflow's own config, database and logs
export AIRFLOW__CORE__DAGS_FOLDER="$REPO/dags"                 # where Airflow looks for DAG files
export AIRFLOW__CORE__LOAD_EXAMPLES=False                      # hide Airflow's ~50 tutorial DAGs
export PAVEMENT_PROJECT_DIR="$REPO"

# The Python that runs the pipeline steps: the project's own virtualenv. From WSL, a
# Windows venv (.venv/Scripts/python.exe) works too, thanks to WSL's Windows interop.
if [ -z "${PAVEMENT_PYTHON:-}" ]; then
  if   [ -x "$REPO/.venv/bin/python" ];         then PAVEMENT_PYTHON="$REPO/.venv/bin/python"
  elif [ -f "$REPO/.venv/Scripts/python.exe" ]; then PAVEMENT_PYTHON="$REPO/.venv/Scripts/python.exe"
  else PAVEMENT_PYTHON=python
  fi
fi
export PAVEMENT_PYTHON

AIRFLOW_BIN="${AIRFLOW_BIN:-$HOME/airflow-venv/bin/airflow}"
# Put Airflow's venv first on PATH, like activating it: `airflow standalone` starts its
# components (API server, scheduler...) as separate `airflow ...` processes found via PATH.
export PATH="$(dirname "$AIRFLOW_BIN"):$PATH"
"$AIRFLOW_BIN" "$@"
