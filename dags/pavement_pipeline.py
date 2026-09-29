"""
# Pavement distress pipeline

Airflow DAG that runs the whole pipeline, from drone images to the SQLite
database, in the right order. Trigger it manually from the UI (or with
`airflow dags trigger pavement_pipeline`).

What's **not** in the DAG: training the YOLO detector. It needs a GPU and runs
once on Colab (`src/train/train.py`); the DAG uses the resulting weights.

Two branches run in parallel, then meet at the condition model:

- **survey branch**: dataset prep -> simulated survey -> YOLO detection ->
  georeferencing -> DBSCAN hotspots + segment scores
- **city-data branch**: download Montréal's PCI survey -> enrich it with
  pavement age

Every task is safe to re-run (idempotent): downloads are skipped when the file
is already there, and every other step overwrites its own outputs.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

# The repo root is one level above this dags/ folder. Both values can be
# overridden with environment variables, so the same DAG runs on any machine.
PROJECT_DIR = os.environ.get("PAVEMENT_PROJECT_DIR", str(Path(__file__).resolve().parent.parent))
# The Python that has the project's dependencies (ultralytics, geopandas...).
# Airflow itself lives in a separate environment: an orchestrator only *launches*
# the steps, it doesn't need their libraries.
PYTHON = os.environ.get("PAVEMENT_PYTHON", "python")
CONFIG = "configs/config.yaml"


def step(task_id: str, module: str, args: str = f"--config {CONFIG}", skip_if_exists: str | None = None,
         doc: str = "") -> BashOperator:
    """One pipeline step = one shell command: `python -m src.<module> <args>`, run from the repo root.

    skip_if_exists makes a download step idempotent: if its output file is already
    there, the task succeeds immediately instead of downloading it again.
    """
    command = f"{PYTHON} -m {module} {args}".strip()
    if skip_if_exists:
        command = f'if [ -e "{skip_if_exists}" ]; then echo "Already there, skipping: {skip_if_exists}"; else {command}; fi'
    return BashOperator(
        task_id=task_id,
        bash_command=command,
        cwd=PROJECT_DIR,
        # Force UTF-8 output (street names like "Montréal"). WSLENV forwards the variable
        # when PAVEMENT_PYTHON is a Windows python.exe called from WSL; harmless elsewhere.
        env={"PYTHONUTF8": "1", "WSLENV": "PYTHONUTF8"},
        append_env=True,  # keep the rest of the environment (PATH, etc.)
        doc_md=doc,
    )


with DAG(
    dag_id="pavement_pipeline",
    description="Drone images -> distress detections -> segment priority -> SQLite",
    schedule=None,  # manual trigger; in production: each time a new flight is uploaded
    start_date=datetime(2026, 9, 1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=1)},  # retry once (e.g. a network hiccup)
    tags=["pavement", "geospatial", "ml"],
    doc_md=__doc__,
) as dag:
    # --- Survey branch ---------------------------------------------------------
    prepare_dataset = step(
        "prepare_dataset", "src.data.prepare_dataset", skip_if_exists="data/yolo/images/test",
        doc="Convert UAV-PDD2023 annotations to YOLO format (train/val/test split).",
    )
    fetch_road_network = step(
        "fetch_road_network", "src.geo.fetch_osm_network", args="", skip_if_exists="data/road_segments.gpkg",
        doc="Download the real street network (32 segments) from OpenStreetMap.",
    )
    simulate_survey = step(
        "simulate_survey", "src.data.generate_synthetic_survey",
        doc="Stamp each test image with a simulated GPS position on a real street (the only simulated part).",
    )
    detect = step("detect", "src.inference.detect", doc="Run the trained YOLO detector on the survey images.")
    georeference = step("georeference", "src.geo.georeference", doc="Read each image's GPS and place its detections on the map.")
    aggregate = step(
        "aggregate_to_segments", "src.geo.aggregate_to_segments",
        doc="DBSCAN hotspots, snap detections to streets, compute each segment's priority score.",
    )

    # --- City-data branch ------------------------------------------------------
    fetch_pci = step(
        "fetch_pci_data", "src.model.fetch_montreal_pci_data", skip_if_exists="data/montreal_pci_segments.gpkg",
        doc="Download the City of Montréal's pavement condition survey (~16,000 segments).",
    )
    enrich_pci = step(
        "enrich_pci_with_age", "src.model.enrich_with_voirie_age",
        doc="Add real pavement age and material (spatial join with the Voirie survey).",
    )

    # --- Where the branches meet, and the outputs --------------------------------
    train_model = step(
        "train_condition_model", "src.model.segment_model",
        doc="Train the scikit-learn PCI model and compare it with the drone score and the real PCI.",
    )
    export_qgis = step("export_qgis", "src.geo.export_qgis", doc="Write the QGIS GeoPackage (priority + hotspots layers).")
    load_sqlite = step("load_to_sqlite", "src.db.load_to_sqlite", doc="Load segments, detections and scores into SQLite.")
    make_figures = step("make_figures", "src.reporting.make_figures", args="", doc="Regenerate the README figures.")

    # --- Dependencies: ">>" reads "must finish before" -----------------------------
    [prepare_dataset, fetch_road_network] >> simulate_survey >> detect >> georeference >> aggregate
    fetch_pci >> enrich_pci
    [aggregate, enrich_pci] >> train_model >> load_sqlite
    aggregate >> export_qgis
    [train_model, export_qgis] >> make_figures
