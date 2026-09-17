"""
Aggregate georeferenced detections onto road segments and compute a per-segment
maintenance-priority score. Also runs DBSCAN over the raw detection points to
find distress "hotspots" -- these are independent of segment boundaries, so a
segment that looks fine on average can still contain one dense hotspot.

Priority score (simple, explainable — refine as you go):
    score = sum over detections on the segment of ( severity_weight[class] * conf )

Each detection is snapped to the nearest road segment within `snap_distance_m`
(measured in a projected/metric CRS for correct distances).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from shapely.geometry import MultiPoint, Point
from sklearn.cluster import DBSCAN


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_hotspots(pts: gpd.GeoDataFrame, eps_m: float, min_samples: int) -> tuple[pd.Series, gpd.GeoDataFrame]:
    """Cluster detection points that sit close together in space (DBSCAN,
    unsupervised -- no labels needed). `pts` must already be in a metric CRS
    so that `eps_m` is a real distance in metres, not degrees.

    DBSCAN labels every point either with a cluster id (0, 1, 2, ...) or -1
    for "noise" (an isolated detection with fewer than `min_samples`
    neighbours within `eps_m` -- not part of any hotspot).

    Returns the per-point cluster labels, and one summary row per hotspot
    (centroid + convex hull, so QGIS can draw a zone around each cluster).
    """
    coords = np.column_stack([pts.geometry.x, pts.geometry.y])
    labels = DBSCAN(eps=eps_m, min_samples=min_samples).fit_predict(coords)

    hotspots = []
    for hotspot_id in sorted(set(labels) - {-1}):
        members = pts.loc[labels == hotspot_id]
        hull = MultiPoint(list(members.geometry)).convex_hull
        hotspots.append(
            {
                "hotspot_id": hotspot_id,
                "n_detections": len(members),
                "dominant_class": members["class_name"].value_counts().idxmax(),
                "geometry": hull.centroid if hull.geom_type == "Point" else hull,
            }
        )
    hotspots_gdf = gpd.GeoDataFrame(hotspots, geometry="geometry", crs=pts.crs)
    return pd.Series(labels, index=pts.index, name="hotspot_id"), hotspots_gdf


def main(config_path: str):
    cfg = load_config(config_path)
    geo = cfg["geo"]

    det = pd.read_csv(Path(cfg["inference"]["out_dir"]) / "detections_geo.csv")
    det = det.dropna(subset=["lat", "lon"])

    # Build a GeoDataFrame of detection points, in the metric CRS for distances.
    pts = gpd.GeoDataFrame(
        det,
        geometry=[Point(lon, lat) for lon, lat in zip(det["lon"], det["lat"])],
        crs="EPSG:4326",
    ).to_crs(geo["metric_crs"])

    segments = gpd.read_file(geo["segments_path"]).to_crs(geo["metric_crs"])
    if "seg_id" not in segments.columns:
        segments = segments.reset_index(names="seg_id")

    # Hotspots are found on the raw points, before snapping to segments --
    # a dense cluster of damage can straddle a segment boundary.
    pts["hotspot_id"], hotspots = find_hotspots(
        pts, eps_m=geo["hotspot_eps_m"], min_samples=geo["hotspot_min_samples"]
    )
    hotspots_out = Path("outputs") / "hotspots.gpkg"
    if len(hotspots):
        hotspots.to_crs(geo["output_crs"]).to_file(hotspots_out, driver="GPKG")
    print(f"Found {len(hotspots)} distress hotspot(s) (DBSCAN, eps={geo['hotspot_eps_m']}m) -> {hotspots_out}")

    # Snap each detection to the nearest segment within the distance threshold.
    joined = gpd.sjoin_nearest(
        pts, segments[["seg_id", "geometry"]],
        how="left", max_distance=geo["snap_distance_m"], distance_col="dist_m",
    )

    weights = geo["severity_weights"]
    joined["weight"] = joined["class_name"].map(weights).fillna(1)
    joined["contribution"] = joined["weight"] * joined["conf"]

    agg = (
        joined.dropna(subset=["seg_id"])
        .groupby("seg_id")
        .agg(
            priority_score=("contribution", "sum"),
            n_detections=("class_name", "count"),
            worst_class=("weight", "idxmax"),
        )
        .reset_index()
    )
    agg["worst_class"] = joined.loc[agg["worst_class"], "class_name"].values

    result = segments.merge(agg, on="seg_id", how="left")
    result["priority_score"] = result["priority_score"].fillna(0)
    result["n_detections"] = result["n_detections"].fillna(0).astype(int)
    result["priority_rank"] = (
        result["priority_score"].rank(ascending=False, method="min").astype(int)
    )

    out = Path("outputs") / "segments_scored.gpkg"
    result.to_crs(geo["output_crs"]).to_file(out, driver="GPKG")
    top = result.sort_values("priority_score", ascending=False).head(10)
    print(f"Scored {len(result)} segments -> {out}")
    print("Top segments by priority:")
    print(top[["seg_id", "priority_score", "n_detections", "worst_class"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    main(args.config)
