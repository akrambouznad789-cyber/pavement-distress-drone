"""
Convert pixel-space detections into geographic points.

Two supported strategies (pick the one matching your data):

  A) EXIF geotag per image (simplest): each drone photo carries a GPS position
     in its EXIF. We approximate every detection in that image at the image's
     GPS point. Coarse, but fine for a first end-to-end version.

  B) Orthomosaic (accurate): detections are run on tiles of a georeferenced
     GeoTIFF orthomosaic. The affine transform maps pixel (col, row) -> map (x, y)
     exactly. Use rasterio to read the transform.

This script implements strategy A and leaves B as a documented hook.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dms_to_deg(dms, ref):
    d, m, s = [float(x) for x in dms]
    deg = d + m / 60 + s / 3600
    if ref in ("S", "W"):
        deg = -deg
    return deg


def exif_latlon(image_path: Path):
    """Return (lat, lon) from an image's EXIF GPS, or None if absent."""
    img = Image.open(image_path)
    exif = img._getexif()
    if not exif:
        return None
    tagged = {TAGS.get(k, k): v for k, v in exif.items()}
    gps = tagged.get("GPSInfo")
    if not gps:
        return None
    gps = {GPSTAGS.get(k, k): v for k, v in gps.items()}
    try:
        lat = _dms_to_deg(gps["GPSLatitude"], gps["GPSLatitudeRef"])
        lon = _dms_to_deg(gps["GPSLongitude"], gps["GPSLongitudeRef"])
        return lat, lon
    except KeyError:
        return None


def main(config_path: str):
    cfg = load_config(config_path)
    inf = cfg["inference"]
    images_dir = Path(inf["images_dir"])
    det_csv = Path(inf["out_dir"]) / "detections.csv"

    df = pd.read_csv(det_csv)

    lats, lons = [], []
    for img_name in df["image"]:
        coords = exif_latlon(images_dir / img_name)
        lats.append(coords[0] if coords else None)
        lons.append(coords[1] if coords else None)

    df["lat"] = lats
    df["lon"] = lons
    missing = df["lat"].isna().sum()
    if missing:
        print(f"WARNING: {missing} detections have no GPS (image lacked EXIF geotag).")

    out = Path(inf["out_dir"]) / "detections_geo.csv"
    df.to_csv(out, index=False)
    print(f"Georeferenced detections -> {out}")

    # --- Strategy B hook (orthomosaic) -------------------------------------
    # import rasterio
    # with rasterio.open(ortho_path) as src:
    #     x, y = src.transform * (col_pixel, row_pixel)   # affine transform
    #     # then reproject (x, y) from src.crs to EPSG:4326 with pyproj.


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    main(args.config)
