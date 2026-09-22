"""
Fetch a small real street network from OpenStreetMap (Overpass API) for a
bounding box, and save each way as its own road segment.

Why a network instead of one named street: real municipal segment tables
(including the City of Montréal's own PCI survey the condition model trains
on) are defined block-to-block -- each stretch between two intersections is
one segment. Using individual OSM ways directly gives that same granularity
for free, for a whole neighbourhood instead of one street divided
arbitrarily. This also removes a synthetic step that used to exist here:
segment boundaries are now real OSM way breaks, not an equal-length division
we invented.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import requests
from shapely.geometry import LineString

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# Overpass rejects requests with no identifying User-Agent.
HEADERS = {"User-Agent": "pavement-distress-drone (student project; contact via GitHub)"}

# Real, drivable road types -- excludes footways, cycleways, service roads,
# parking aisles, steps, etc., so the network looks like streets a drone
# survey of a neighbourhood would actually cover.
ROAD_TYPES = ["residential", "tertiary", "secondary", "primary", "unclassified", "living_street"]


def fetch_ways(bbox: tuple[float, float, float, float]) -> list[dict]:
    """Query Overpass for real road ways in bbox=(south, west, north, east)."""
    south, west, north, east = bbox
    type_filter = "|".join(ROAD_TYPES)
    query = f"""
    [out:json][timeout:25];
    way["highway"~"^({type_filter})$"]({south},{west},{north},{east});
    out geom;
    """
    resp = requests.post(OVERPASS_URL, data={"data": query.encode("utf-8")}, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()["elements"]


def main(bbox: tuple[float, float, float, float], out_path: str):
    elements = fetch_ways(bbox)
    segments = []
    for el in elements:
        coords = [(pt["lon"], pt["lat"]) for pt in el["geometry"]]
        if len(coords) < 2:
            continue
        segments.append(
            {
                "seg_id": len(segments),
                "osm_way_id": el["id"],
                "name": el.get("tags", {}).get("name", ""),
                "geometry": LineString(coords),
            }
        )

    gdf = gpd.GeoDataFrame(segments, geometry="geometry", crs="EPSG:4326")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GPKG")

    street_names = sorted({n for n in gdf["name"] if n})
    print(f"{len(gdf)} real street segments (OSM ways) -> {out_path}")
    print(f"Streets ({len(street_names)}): {', '.join(street_names)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # bbox = south, west, north, east -- a small neighbourhood near Polytechnique
    # Montréal (~32 segments across 15 real streets, incl. Chemin de la
    # Côte-Sainte-Catherine and Boulevard Édouard-Montpetit).
    ap.add_argument("--bbox", nargs=4, type=float, default=[45.5025, -73.6195, 45.5095, -73.6085])
    ap.add_argument("--out", default="data/road_segments.gpkg")
    args = ap.parse_args()
    main(tuple(args.bbox), args.out)
