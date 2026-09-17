"""
Generate a synthetic road-segment attribute table to train the condition model.

Why: the segment condition model (src.model.segment_model) needs many segments
with (road_class, traffic, pavement_age) attributes and a known priority score
to learn from. We only have 8 real segments from one 800m drone survey -- far
too few to train/validate a model. So we simulate a larger, plausible dataset:
the target (priority_score_true) is generated from a simple, documented formula
plus noise, mimicking how older/busier/heavier-traffic roads tend to score worse.
This is a clearly synthetic training set, not measured data.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str):
    cfg = load_config(config_path)
    mcfg = cfg["model"]
    rng = np.random.default_rng(cfg["project"]["seed"])

    n = mcfg["n_synthetic_segments"]
    road_classes = rng.choice(
        ["residential", "collector", "arterial"], size=n, p=[0.5, 0.3, 0.2]
    )
    # Arterial roads carry more traffic; add class-dependent noise.
    base_traffic = {"residential": 2000, "collector": 8000, "arterial": 20000}
    traffic_aadt = np.array([base_traffic[c] for c in road_classes]) * rng.uniform(
        0.6, 1.4, size=n
    )
    pavement_age_years = rng.uniform(0, 30, size=n)
    num_lanes = np.array(
        [{"residential": 2, "collector": 2, "arterial": 4}[c] for c in road_classes]
    )

    # Documented synthetic ground truth: older, busier, wider roads score worse,
    # plus random noise so the relationship isn't perfectly learnable (realistic).
    priority_score_true = (
        0.8 * pavement_age_years
        + 0.004 * traffic_aadt
        + 3.0 * (num_lanes - 2)
        + rng.normal(0, 5, size=n)
    ).clip(min=0)

    df = pd.DataFrame(
        {
            "seg_id": np.arange(n),
            "road_class": road_classes,
            "traffic_aadt": traffic_aadt.round(0),
            "pavement_age_years": pavement_age_years.round(1),
            "num_lanes": num_lanes,
            "priority_score_true": priority_score_true.round(2),
        }
    )
    df.to_csv(mcfg["features_path"], index=False)
    print(f"{n} synthetic segments -> {mcfg['features_path']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    main(args.config)
