"""
Load the pipeline's file outputs (GeoPackage / CSV) into a SQLite database,
so the results are queryable with SQL instead of only readable as files.

This is a *loading* step, not a computation step -- everything here already
exists in outputs/ and data/; this script just reshapes it into three related
tables (see sql/schema.sql):
    segments        -- one row per road segment (reference data)
    detections      -- one row per detected distress instance (raw facts)
    segment_scores  -- one row per segment, the final priority result

Run after the full pipeline (through aggregate_to_segments.py and
segment_model.py) has produced its outputs.
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_segments(cfg: dict) -> pd.DataFrame:
    """Geometry comes from road_segments.gpkg (seg_id + shape only); the real
    attributes (arrondissement, length, roughness, pavement age/material)
    live in surveyed_vs_predicted.csv -- matched in from two separate City
    of Montreal surveys in segment_model.py. Join them on seg_id.
    """
    geom = gpd.read_file(cfg["geo"]["segments_path"])
    attrs = pd.read_csv(Path(cfg["model"]["out_dir"]) / "surveyed_vs_predicted.csv")
    attrs = attrs[
        ["seg_id", "arrondissement", "longueur_m", "indice_iri", "pavement_age_years", "materiau_chaussee"]
    ]

    segments = geom.merge(attrs, on="seg_id", how="left")
    segments["geometry_wkt"] = segments["geometry"].apply(lambda g: g.wkt)
    return segments[
        ["seg_id", "arrondissement", "longueur_m", "indice_iri", "pavement_age_years", "materiau_chaussee", "geometry_wkt"]
    ]


def build_detections(cfg: dict) -> pd.DataFrame:
    det = pd.read_csv(Path(cfg["inference"]["out_dir"]) / "detections_segmented.csv")
    det = det.rename(columns={"conf": "confidence"})
    # seg_id arrives as float (NaN-capable from the join); SQLite stores it fine either way,
    # but keep it a clean nullable int for anyone reading the CSV/DB directly.
    det["seg_id"] = det["seg_id"].astype("Int64")
    return det[
        ["image", "class_name", "confidence", "xmin", "ymin", "xmax", "ymax", "lat", "lon", "seg_id"]
    ]


def build_segment_scores(cfg: dict) -> pd.DataFrame:
    scored = gpd.read_file("outputs/segments_scored.gpkg")
    predicted = pd.read_csv(Path(cfg["model"]["out_dir"]) / "surveyed_vs_predicted.csv")

    scores = scored[["seg_id", "n_detections", "worst_class", "priority_score", "priority_rank"]].merge(
        predicted[["seg_id", "predicted_score", "real_indice_pci"]], on="seg_id", how="left"
    )
    return scores[
        ["seg_id", "n_detections", "worst_class", "priority_score", "predicted_score", "real_indice_pci", "priority_rank"]
    ]


def main(config_path: str):
    cfg = load_config(config_path)
    db_path = Path(cfg["db"]["path"])
    db_path.parent.mkdir(parents=True, exist_ok=True)

    segments = build_segments(cfg)
    detections = build_detections(cfg)
    segment_scores = build_segment_scores(cfg)

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(Path(cfg["db"]["schema_path"]).read_text(encoding="utf-8"))

        # Delete-then-insert (not DROP TABLE) so the schema/constraints persist
        # across runs; order matters because detections/segment_scores reference segments.
        conn.execute("DELETE FROM segment_scores")
        conn.execute("DELETE FROM detections")
        conn.execute("DELETE FROM segments")

        segments.to_sql("segments", conn, if_exists="append", index=False)
        detections.to_sql("detections", conn, if_exists="append", index=False)
        segment_scores.to_sql("segment_scores", conn, if_exists="append", index=False)
        conn.commit()

        print(f"Loaded into {db_path}:")
        print(f"  segments        {len(segments)} rows")
        print(f"  detections      {len(detections)} rows")
        print(f"  segment_scores  {len(segment_scores)} rows")

        print("\nSanity check -- top 3 segments by drone-based priority, straight from SQL:")
        top3 = conn.execute(
            "SELECT seg_id, arrondissement, priority_score, real_indice_pci "
            "FROM segments NATURAL JOIN segment_scores "
            "ORDER BY priority_score DESC LIMIT 3"
        ).fetchall()
        for row in top3:
            print(f"  seg_id={row[0]}  {row[1]:<38}  priority={row[2]:.1f}  real_pci={row[3]:.1f}")
    finally:
        conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    main(args.config)
