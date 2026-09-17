"""
Generate a synthetic road-segments layer for src.geo.aggregate_to_segments.

Why: aggregate_to_segments.py needs a real road network (data/road_segments.gpkg)
to snap detections onto, and we have none. We split a real street's geometry
(fetched from OpenStreetMap by src.geo.fetch_osm_route -- run that first) into
N successive segments, each with an id and geometry -- exactly the shape
(seg_id, geometry) a real municipal road-segments table would have. The route
shape is real (an actual street's curvature and intersections); only the
"survey happened here" and "these are the segment boundaries" framing is
synthetic -- there was no real drone flight or municipal segment table to use.
"""
from __future__ import annotations

import argparse
import json

import geopandas as gpd
from shapely.geometry import LineString


def load_config(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_route(route_path: str) -> LineString:
    with open(route_path, "r", encoding="utf-8") as f:
        geojson = json.load(f)
    return LineString(geojson["geometry"]["coordinates"])


def main(config_path: str, route_path: str, n_segments: int):
    cfg = load_config(config_path)
    geo = cfg["geo"]

    # Split by real distance (metres), not by raw point count, so segments
    # are roughly equal length regardless of how densely OSM sampled the curve.
    route = gpd.GeoSeries([load_route(route_path)], crs="EPSG:4326").to_crs(geo["metric_crs"]).iloc[0]

    segments = []
    for i in range(n_segments):
        d0 = route.length * i / n_segments
        d1 = route.length * (i + 1) / n_segments
        # Sample a few intermediate points so curved streets keep their shape
        # instead of being flattened into a straight chord per segment.
        n_samples = 5
        pts = [route.interpolate(d0 + (d1 - d0) * t / n_samples) for t in range(n_samples + 1)]
        segments.append({"seg_id": i, "geometry": LineString(pts)})

    gdf = gpd.GeoDataFrame(segments, geometry="geometry", crs=geo["metric_crs"]).to_crs("EPSG:4326")
    out_path = geo["segments_path"]
    gdf.to_file(out_path, driver="GPKG")
    print(f"{n_segments} synthetic road segments (real OSM street shape) -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--route", default="data/road_route.geojson")
    ap.add_argument("--n-segments", type=int, default=8)
    args = ap.parse_args()
    main(args.config, args.route, args.n_segments)
