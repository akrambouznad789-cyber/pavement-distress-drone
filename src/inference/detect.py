"""
Run the trained detector on new drone imagery and save detections to CSV.

Output CSV columns:
    image, class_name, conf, xmin, ymin, xmax, ymax   (bbox in pixel coords)

Georeferencing (pixel -> lat/lon) happens in the next step (src.geo.georeference),
which needs either per-image EXIF geotags or an orthomosaic transform.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from ultralytics import YOLO


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str, images_override: str | None = None):
    cfg = load_config(config_path)
    inf = cfg["inference"]
    class_names = cfg["data"]["classes"]

    images_dir = Path(images_override or inf["images_dir"])
    out_dir = Path(inf["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(inf["weights"])
    results = model.predict(source=str(images_dir), conf=inf["conf"], stream=True)

    rows = []
    for r in results:
        img_name = Path(r.path).name
        for b in r.boxes:
            cls_idx = int(b.cls.item())
            xmin, ymin, xmax, ymax = b.xyxy[0].tolist()
            rows.append(
                {
                    "image": img_name,
                    "class_name": class_names[cls_idx],
                    "conf": round(float(b.conf.item()), 4),
                    "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
                }
            )

    df = pd.DataFrame(rows)
    out_csv = out_dir / "detections.csv"
    df.to_csv(out_csv, index=False)
    print(f"{len(df)} detections across {df['image'].nunique()} images -> {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--images", default=None, help="override the images dir from config")
    args = ap.parse_args()
    main(args.config, args.images)
