"""
Prepare the raw UAV pavement-distress dataset for YOLO training.

Steps:
  1. Read the raw images and their annotations.
  2. Convert bounding boxes to YOLO format (class cx cy w h, normalized).
  3. Split into train / val / test and write a data.yaml for ultralytics.

The exact parsing in `load_raw_annotations` depends on the dataset's annotation
format (Pascal VOC XML for UAV-PDD2023 / RDD2022). That function is the one
concrete piece you fill in for your chosen dataset — everything downstream is
format-agnostic.
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def voc_bbox_to_yolo(box, img_w, img_h):
    """Convert (xmin, ymin, xmax, ymax) in pixels to normalized YOLO (cx, cy, w, h)."""
    xmin, ymin, xmax, ymax = box
    cx = ((xmin + xmax) / 2) / img_w
    cy = ((ymin + ymax) / 2) / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return cx, cy, w, h


def load_raw_annotations(raw_dir: Path, class_to_idx: dict):
    """
    TODO: implement for your dataset's annotation format.

    Should yield tuples of:
        (image_path: Path, width: int, height: int, boxes: list[(cls_idx, xmin, ymin, xmax, ymax)])

    For UAV-PDD2023 / RDD2022 (Pascal VOC), parse each .xml with xml.etree,
    read <size> for width/height and each <object> for <name> and <bndbox>.
    """
    raise NotImplementedError(
        "Implement load_raw_annotations() for your dataset's annotation format."
    )


def write_split(records, yolo_dir: Path, split_name: str, class_names):
    img_out = yolo_dir / "images" / split_name
    lbl_out = yolo_dir / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for img_path, w, h, boxes in records:
        shutil.copy(img_path, img_out / img_path.name)
        label_lines = []
        for cls_idx, xmin, ymin, xmax, ymax in boxes:
            cx, cy, bw, bh = voc_bbox_to_yolo((xmin, ymin, xmax, ymax), w, h)
            label_lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (lbl_out / f"{img_path.stem}.txt").write_text("\n".join(label_lines))


def main(config_path: str):
    cfg = load_config(config_path)
    random.seed(cfg["project"]["seed"])

    class_names = cfg["data"]["classes"]
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    raw_dir = Path(cfg["data"]["raw_dir"])
    yolo_dir = Path(cfg["data"]["yolo_dir"])

    records = list(load_raw_annotations(raw_dir, class_to_idx))
    random.shuffle(records)

    n = len(records)
    r_train, r_val, _ = cfg["data"]["split"]
    i_train, i_val = int(n * r_train), int(n * (r_train + r_val))

    write_split(records[:i_train], yolo_dir, "train", class_names)
    write_split(records[i_train:i_val], yolo_dir, "val", class_names)
    write_split(records[i_val:], yolo_dir, "test", class_names)

    data_yaml = {
        "path": str(yolo_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(class_names)},
    }
    with open(yolo_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)

    print(f"Prepared {n} images -> {yolo_dir} (train/val/test written).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    main(args.config)
