"""
Export the scored road segments as a QGIS-ready GeoPackage.

The layer carries `priority_score`, `priority_rank`, `n_detections`, and
`worst_class`, so in QGIS you can apply a graduated symbology on
`priority_score` (e.g. green -> red) to see where to intervene first.

A QGIS layer-style (.qml) is optional; here we just guarantee a clean GeoPackage.
Styling in QGIS: right-click layer -> Properties -> Symbology -> Graduated ->
value = priority_score.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str):
    cfg = load_config(config_path)
    geo = cfg["geo"]

    scored = gpd.read_file(Path("outputs") / "segments_scored.gpkg")
    out = Path(geo["out_layer"])
    out.parent.mkdir(parents=True, exist_ok=True)

    scored.to_crs(geo["output_crs"]).to_file(out, layer="segment_priority", driver="GPKG")
    print(f"QGIS layer written -> {out} (layer: segment_priority)")
    print("Open in QGIS and apply a graduated symbology on 'priority_score'.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    main(args.config)
