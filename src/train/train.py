"""
Train a YOLO detector on the prepared pavement-distress dataset.

Uses ultralytics. Reads all hyperparameters from configs/config.yaml so runs
are reproducible and easy to tweak without editing code.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str):
    cfg = load_config(config_path)
    t = cfg["train"]
    data_yaml = Path(cfg["data"]["yolo_dir"]) / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"{data_yaml} not found. Run src.data.prepare_dataset first."
        )

    model = YOLO(t["model"])
    model.train(
        data=str(data_yaml),
        imgsz=t["imgsz"],
        epochs=t["epochs"],
        batch=t["batch"],
        device=t["device"],
        project=t["project_dir"],
        name=t["run_name"],
        seed=cfg["project"]["seed"],
    )

    # Evaluate on the held-out test split and print the honest numbers.
    metrics = model.val(split="test")
    print("Test mAP@50:", round(float(metrics.box.map50), 4))
    print("Test mAP@50-95:", round(float(metrics.box.map), 4))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    main(args.config)
