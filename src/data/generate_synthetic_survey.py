"""
Generate a synthetic drone survey by copying test images into
data/new_survey/ and stamping each one with a simulated GPS EXIF tag.

Why: the UAV-PDD2023 images have no GPS geotags (they're research-dataset
crops, not real flight photos). To honestly demonstrate the georeferencing
step (src.geo.georeference, strategy A: EXIF geotag per image) without
faking a real field survey, we simulate a drone flying along a real street's
geometry (fetched from OpenStreetMap by src.geo.fetch_osm_route -- run that
first) and assign each image one point along that route, in order, spaced by
real distance. This is clearly a synthetic demo (no real flight happened),
but the road it "flies over" is a real, correctly-shaped street.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import geopandas as gpd
import piexif
import yaml
from shapely.geometry import LineString


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_route(route_path: str) -> LineString:
    with open(route_path, "r", encoding="utf-8") as f:
        geojson = json.load(f)
    return LineString(geojson["geometry"]["coordinates"])


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


def main(config_path: str, route_path: str):
    cfg = load_config(config_path)
    src_dir = Path(cfg["data"]["yolo_dir"]) / "images" / "test"
    out_dir = Path(cfg["inference"]["images_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_crs = cfg["geo"]["metric_crs"]

    # Work in a metric CRS so images are spaced evenly by real distance along
    # the route's actual curve, not by raw longitude/latitude interpolation.
    route = gpd.GeoSeries([load_route(route_path)], crs="EPSG:4326").to_crs(metric_crs).iloc[0]

    images = sorted(src_dir.glob("*.jpg"))
    n = len(images)

    for i, img_path in enumerate(images):
        t = i / (n - 1) if n > 1 else 0.0  # 0 -> start of route, 1 -> end
        point_metric = route.interpolate(route.length * t)
        point = gpd.GeoSeries([point_metric], crs=metric_crs).to_crs("EPSG:4326").iloc[0]
        dst = out_dir / img_path.name
        shutil.copy(img_path, dst)
        write_gps_exif(dst, lat=point.y, lon=point.x)

    print(f"{n} images copied to {out_dir} with simulated GPS along the real OSM route.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--route", default="data/road_route.geojson")
    args = ap.parse_args()
    main(args.config, args.route)
