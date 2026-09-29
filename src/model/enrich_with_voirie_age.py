"""
Enrich the real Montréal PCI segments with real pavement age, by spatially
joining them to the City of Montréal's road-asset survey (Voirie), which
carries construction/resurfacing dates -- something the PCI survey itself
doesn't record.

Why a separate join, not a shared key: the two City of Montréal datasets use
different identifiers (PCI segments: ID_TRC; Voirie: ID_VOI_VOIRIE_AGR) and
different geometry types (PCI: LineString road centrelines; Voirie: Polygon
pavement footprints). There is no common key, so each PCI segment's
geometry is matched to the Voirie polygon it actually falls inside (or
nearest, if none) -- a real spatial relationship, not an invented one.

Source: Ville de Montréal open data, "Voirie" (CC-BY 4.0).
https://donnees.montreal.ca/dataset/voirie-actif
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
import yaml
from shapely import wkt

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
# Age is measured at the time the city measured the PCI (the 2024 survey file in
# config.yaml), not "today": that's the age that explains the measured condition, and it
# keeps the result identical whatever year the pipeline is re-run.
SURVEY_YEAR = 2024

# Real, drivable pavement categories in the Voirie survey (excludes
# sidewalks, islands, etc., which aren't in this dataset's CATEGORIECHAUSSEE
# column at all, but the column also holds a majority of NaN rows for
# non-roadway assets we filter out here).
ROAD_CATEGORIES = ["Rue", "Ruelle", "Autoroute", "Autoroute accotement", "Bretelle", "Bretelle accotement"]


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _year_from_yyyymmdd_float(value: float) -> int | None:
    """Dates arrive as e.g. 19580101000000.0 (YYYYMMDDHHMMSS). Just need the year."""
    if pd.isna(value):
        return None
    return int(str(int(value))[:4])


def download_voirie(url: str, cache_csv: Path) -> Path:
    if cache_csv.exists():
        return cache_csv
    print(f"Downloading {url} ...")
    resp = requests.get(url, headers=HEADERS, timeout=180)
    resp.raise_for_status()
    zip_path = cache_csv.with_suffix(".zip")
    zip_path.write_bytes(resp.content)
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(name) as src, open(cache_csv, "wb") as dst:
            dst.write(src.read())
    zip_path.unlink()
    return cache_csv


def load_voirie_streets(csv_path: Path, metric_crs: str) -> gpd.GeoDataFrame:
    cols = ["CATEGORIECHAUSSEE_REF", "DATECONSTRUCTION", "DATERESURFACAGE", "MATERIAUCHAUSSEE_REF", "GEOMETRY"]
    # A couple of rows have a geometry field long enough to exceed the C
    # parser's field-size limit; the python engine handles those instead of
    # dropping the whole file.
    df = pd.read_csv(csv_path, usecols=cols, on_bad_lines="skip", engine="python")
    df = df[df["CATEGORIECHAUSSEE_REF"].isin(ROAD_CATEGORIES)].copy()

    df["year_built"] = df["DATECONSTRUCTION"].apply(_year_from_yyyymmdd_float)
    df["year_resurfaced"] = df["DATERESURFACAGE"].apply(_year_from_yyyymmdd_float)
    df["pavement_age_years"] = SURVEY_YEAR - df[["year_built", "year_resurfaced"]].max(axis=1)

    df["geometry"] = df["GEOMETRY"].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(
        df[["pavement_age_years", "MATERIAUCHAUSSEE_REF", "geometry"]].rename(
            columns={"MATERIAUCHAUSSEE_REF": "materiau_chaussee"}
        ),
        geometry="geometry",
        crs="EPSG:4326",
    )
    return gdf.to_crs(metric_crs)


def match_age_to_segments(segments: gpd.GeoDataFrame, voirie: gpd.GeoDataFrame) -> pd.DataFrame:
    """For each PCI segment (LineString), find the Voirie polygon it falls
    inside (its midpoint, to avoid edge-of-polygon ambiguity at
    intersections); fall back to the nearest polygon within 25m for
    segments that land in a small gap between mapped polygons."""
    midpoints = gpd.GeoDataFrame(
        {"seg_idx": segments.index}, geometry=segments.geometry.interpolate(0.5, normalized=True), crs=segments.crs
    )

    within = gpd.sjoin(midpoints, voirie, how="left", predicate="within")
    missing = within[within["pavement_age_years"].isna()]["seg_idx"]

    if len(missing):
        nearest = gpd.sjoin_nearest(
            midpoints[midpoints["seg_idx"].isin(missing)], voirie, how="left", max_distance=25, distance_col="_d"
        )
        # sjoin_nearest can return several equidistant ties per point -- keep one.
        nearest = nearest.drop_duplicates(subset="seg_idx").set_index("seg_idx")
        within = within.set_index("seg_idx")
        within.update(nearest[["pavement_age_years", "materiau_chaussee"]])
        within = within.reset_index()

    return within.drop_duplicates(subset="seg_idx").set_index("seg_idx")[["pavement_age_years", "materiau_chaussee"]]


def main(config_path: str):
    cfg = load_config(config_path)
    mcfg = cfg["model"]
    metric_crs = cfg["geo"]["metric_crs"]

    cache_csv = Path("data") / "voirie_streets_raw.csv"
    csv_path = download_voirie(mcfg["voirie_source_url"], cache_csv)

    print("Loading and filtering Voirie street polygons...")
    voirie = load_voirie_streets(csv_path, metric_crs)
    print(f"{len(voirie)} real street pavement polygons")

    segments_path = Path(mcfg["features_path"])
    segments = gpd.read_file(segments_path).to_crs(metric_crs)

    print("Matching each PCI segment to its real pavement age...")
    matched = match_age_to_segments(segments, voirie)
    # Drop the columns a previous run already added, so this step can be re-run safely
    # (idempotent) -- an orchestrator like Airflow retries and re-runs steps routinely.
    segments = segments.drop(columns=["pavement_age_years", "materiau_chaussee"], errors="ignore").join(matched)

    n_matched = segments["pavement_age_years"].notna().sum()
    print(f"{n_matched}/{len(segments)} segments matched a real construction/resurfacing date")

    segments.to_crs("EPSG:4326").to_file(segments_path, driver="GPKG")
    print(f"Enriched {segments_path} with pavement_age_years / materiau_chaussee")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    main(args.config)
