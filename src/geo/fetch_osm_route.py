"""
Fetch a real street's geometry from OpenStreetMap (Overpass API) and stitch
its way segments into one ordered polyline, saved as GeoJSON.

Why: the synthetic demo route used to be a straight line between two lat/lon
points, which doesn't look like a real road (no curvature, no intersections).
OSM stores a street as many short "way" segments (split at intersections),
sharing endpoint nodes with their neighbours. This script chains them back
into a single continuous line, which the synthetic segments/survey generators
then use instead of a straight-line interpolation -- same "clearly synthetic,
honestly labeled" demo, but following an actual street shape.
"""
from __future__ import annotations

import argparse
import json

import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# Overpass rejects requests with no identifying User-Agent.
HEADERS = {"User-Agent": "pavement-distress-drone (student project; contact via GitHub)"}


def fetch_ways(name: str, bbox: tuple[float, float, float, float]) -> list[list[tuple[float, float]]]:
    """Query Overpass for named highway ways in bbox=(south, west, north, east).
    Returns each way as a list of (lon, lat) points, in the order OSM stored them."""
    south, west, north, east = bbox
    query = f"""
    [out:json][timeout:25];
    way["highway"]["name"="{name}"]({south},{west},{north},{east});
    out geom;
    """
    resp = requests.post(OVERPASS_URL, data={"data": query.encode("utf-8")}, headers=HEADERS)
    resp.raise_for_status()
    elements = resp.json()["elements"]
    return [[(pt["lon"], pt["lat"]) for pt in el["geometry"]] for el in elements]


def chain_ways(ways: list[list[tuple[float, float]]]) -> list[tuple[float, float]]:
    """Stitch way segments into one polyline by matching shared endpoints.
    OSM ways for one street are split at intersections but share the exact
    node coordinate at each split, so exact-match endpoint lookup works."""
    remaining = [list(w) for w in ways]

    # Start from a way touching an endpoint that no other way touches
    # (a true end of the corridor within the query bbox), so we chain in one
    # direction instead of starting mid-street.
    endpoint_counts: dict[tuple[float, float], int] = {}
    for w in remaining:
        endpoint_counts[w[0]] = endpoint_counts.get(w[0], 0) + 1
        endpoint_counts[w[-1]] = endpoint_counts.get(w[-1], 0) + 1

    start_idx = next(
        (i for i, w in enumerate(remaining) if endpoint_counts[w[0]] == 1 or endpoint_counts[w[-1]] == 1),
        0,
    )
    first = remaining.pop(start_idx)
    if endpoint_counts[first[-1]] == 1 and endpoint_counts[first[0]] != 1:
        first = first[::-1]  # walk away from the free end, not into it
    chain = first
    seen = set(chain)

    # Greedily extend the chain at its current tail, in either direction.
    # A candidate whose new points already appear in the chain would create a
    # loop/dead-end spur (e.g. two OSM ways covering the same two endpoints
    # in opposite directions) -- skip those rather than looping in place.
    extended = True
    while extended and remaining:
        extended = False
        for i, w in enumerate(remaining):
            if w[0] == chain[-1]:
                new_pts = w[1:]
            elif w[-1] == chain[-1]:
                new_pts = w[-2::-1]
            elif w[-1] == chain[0]:
                new_pts = w[:-1]
            elif w[0] == chain[0]:
                new_pts = w[::-1][:-1]
            else:
                continue
            if seen.intersection(new_pts):
                continue
            if w[-1] == chain[0] or w[0] == chain[0]:
                chain = new_pts + chain
            else:
                chain = chain + new_pts
            seen.update(new_pts)
            remaining.pop(i)
            extended = True
            break

    return chain


def main(name: str, bbox: tuple[float, float, float, float], out_path: str):
    ways = fetch_ways(name, bbox)
    if not ways:
        raise SystemExit(f"No OSM ways found for {name!r} in bbox {bbox}. Check the name/bbox.")
    route = chain_ways(ways)
    geojson = {
        "type": "Feature",
        "properties": {"name": name, "source": "OpenStreetMap (Overpass API)"},
        "geometry": {"type": "LineString", "coordinates": [[lon, lat] for lon, lat in route]},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f)
    print(f"Chained {len(ways)} OSM way segments into a {len(route)}-point route -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="Chemin de la Côte-Sainte-Catherine")
    # bbox = south, west, north, east -- covers the stretch near Polytechnique Montréal.
    ap.add_argument("--bbox", nargs=4, type=float, default=[45.500, -73.625, 45.514, -73.610])
    ap.add_argument("--out", default="data/road_route.geojson")
    args = ap.parse_args()
    main(args.name, tuple(args.bbox), args.out)
