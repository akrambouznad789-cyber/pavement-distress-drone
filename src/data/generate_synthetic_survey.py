"""
Generate a synthetic drone survey by copying test images into
data/new_survey/ and stamping each one with a simulated GPS EXIF tag.

Why: the UAV-PDD2023 images have no GPS geotags (they're research-dataset
crops, not real flight photos). To honestly demonstrate the georeferencing
step (src.geo.georeference, strategy A: EXIF geotag per image) without
faking a real field survey, we simulate a drone surveying a real
neighbourhood's street network (fetched from OpenStreetMap by
src.geo.fetch_osm_network -- run that first): each image is assigned a
random point on a random real segment, segments weighted by length so a
longer street gets proportionally more images. This is clearly a synthetic
demo (no real flight happened), but the streets it "flies over" are real.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import piexif
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _deg_to_dms_rational(deg: float):
    """EXIF GPS stores degrees as (deg, min, sec), each a (numerator, denominator) pair."""
    deg = abs(deg)
    d = int(deg)
    m_float = (deg - d) * 60
    m = int(m_float)
    s = round((m_float - m) * 60 * 100)
    return [(d, 1), (m, 1), (s, 100)]


def write_gps_exif(image_path: Path, lat: float, lon: float):
    gps_ifd = {
        piexif.GPSIFD.GPSLatitudeRef: "N" if lat >= 0 else "S",
        piexif.GPSIFD.GPSLatitude: _deg_to_dms_rational(lat),
        piexif.GPSIFD.GPSLongitudeRef: "E" if lon >= 0 else "W",
        piexif.GPSIFD.GPSLongitude: _deg_to_dms_rational(lon),
    }
    try:
        exif_dict = piexif.load(str(image_path))
    except Exception:
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    exif_dict["GPS"] = gps_ifd
    piexif.insert(piexif.dump(exif_dict), str(image_path))


def main(config_path: str):
    cfg = load_config(config_path)
    src_dir = Path(cfg["data"]["yolo_dir"]) / "images" / "test"
    out_dir = Path(cfg["inference"]["images_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_crs = cfg["geo"]["metric_crs"]
    seed = cfg["project"]["seed"]

    # Work in a metric CRS so "weight by length" and "a random point along
    # this segment" both use real distances, not raw lon/lat degrees.
    segments = gpd.read_file(cfg["geo"]["segments_path"]).to_crs(metric_crs)
    weights = segments.geometry.length / segments.geometry.length.sum()

    images = sorted(src_dir.glob("*.jpg"))
    rng = np.random.default_rng(seed)
    # Longer segments get proportionally more images -- the same logic a real
    # flight plan would use to cover more of a longer street.
    chosen = rng.choice(segments.index, size=len(images), p=weights)
    frac_along = rng.uniform(0, 1, size=len(images))

    for img_path, seg_idx, t in zip(images, chosen, frac_along):
        line = segments.geometry.iloc[seg_idx]
        point_metric = line.interpolate(line.length * t)
        point = gpd.GeoSeries([point_metric], crs=metric_crs).to_crs("EPSG:4326").iloc[0]
        dst = out_dir / img_path.name
        shutil.copy(img_path, dst)
        write_gps_exif(dst, lat=point.y, lon=point.x)

    print(f"{len(images)} images copied to {out_dir}, GPS spread across {len(segments)} real street segments.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    main(args.config)
