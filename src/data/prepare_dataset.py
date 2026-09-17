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

import argparse   # lit les arguments passés en ligne de commande (--config ...)
import random      # pour mélanger les images avant de les répartir en train/val/test
import shutil      # pour copier les fichiers image vers leur dossier de split
import xml.etree.ElementTree as ET  # pour lire (parser) les fichiers .xml
from pathlib import Path  # manipulation de chemins de fichiers, multiplateforme

import yaml  # pour lire configs/config.yaml et écrire data/yolo/data.yaml


def load_config(path: str) -> dict:
    """Charge configs/config.yaml et le transforme en dictionnaire Python."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def voc_bbox_to_yolo(box, img_w, img_h):
    """Convert (xmin, ymin, xmax, ymax) in pixels to normalized YOLO (cx, cy, w, h).

    Entrée (format VOC) : coin haut-gauche + coin bas-droite, en pixels absolus.
    Sortie (format YOLO) : centre de la boîte (cx, cy) + largeur/hauteur (w, h),
    chaque valeur divisée par la taille de l'image -> toujours entre 0 et 1.
    Diviser par img_w/img_h ne modifie pas l'image ni le XML d'origine : ça ne
    fait que ré-exprimer les mêmes coordonnées dans un autre système.
    """
    xmin, ymin, xmax, ymax = box
    cx = ((xmin + xmax) / 2) / img_w
    cy = ((ymin + ymax) / 2) / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return cx, cy, w, h


def normalize_class_name(name: str) -> str:
    """Match the XML <name> text (e.g. "Transverse crack") to a config class
    key (e.g. "transverse_crack").

    Le XML écrit les noms de classe avec majuscule et espace ("Transverse crack"),
    alors que configs/config.yaml les écrit en minuscule avec underscore
    ("transverse_crack"). Cette fonction uniformise les deux écritures pour
    qu'on puisse les comparer.
    """
    return name.strip().lower().replace(" ", "_")


def load_raw_annotations(raw_dir: Path, class_to_idx: dict):
    """
    Parse Pascal VOC annotations (UAV-PDD2023 / RDD2022 layout):
        raw_dir/Annotations/*.xml
        raw_dir/JPEGImages/*.jpg

    Yields (image_path, width, height, boxes), where boxes is a list of
    (cls_idx, xmin, ymin, xmax, ymax) in pixel coordinates.

    C'est le coeur du script : pour chaque photo, on lit son fichier XML
    jumeau et on en extrait tout ce dont on a besoin pour construire, plus
    loin, le label YOLO correspondant.
    """
    annotations_dir = raw_dir / "Annotations"
    images_dir = raw_dir / "JPEGImages"

    # Un .xml par image -> on boucle sur tous les fichiers d'annotation.
    for xml_path in sorted(annotations_dir.glob("*.xml")):
        root = ET.parse(xml_path).getroot()  # racine de l'arbre XML (<annotation>)

        # Le XML donne le nom du fichier image correspondant (balise <filename>).
        img_path = images_dir / root.find("filename").text
        if not img_path.exists():
            print(f"WARNING: image not found for {xml_path.name}, skipping.")
            continue

        # Taille de l'image en pixels (balise <size>), nécessaire pour normaliser
        # les boîtes plus tard. .text renvoie toujours une chaîne -> int(float(...))
        # convertit en nombre (float d'abord au cas où la valeur serait "637.0").
        width = int(float(root.find("size/width").text))
        height = int(float(root.find("size/height").text))

        # Une image peut contenir plusieurs dégradations -> plusieurs <object>.
        boxes = []
        for obj in root.findall("object"):
            cls_key = normalize_class_name(obj.find("name").text)
            if cls_key not in class_to_idx:
                # Classe absente de configs/config.yaml (ex: "_background_")
                # -> on ignore juste cette boîte, sans planter tout le script.
                print(f"WARNING: unknown class '{cls_key}' in {xml_path.name}, skipping box.")
                continue
            cls_idx = class_to_idx[cls_key]
            box = obj.find("bndbox")
            xmin = int(float(box.find("xmin").text))
            ymin = int(float(box.find("ymin").text))
            xmax = int(float(box.find("xmax").text))
            ymax = int(float(box.find("ymax").text))
            boxes.append((cls_idx, xmin, ymin, xmax, ymax))

        # yield = comme "return" mais la fonction continue au tour suivant de la
        # boucle for(xml_path) au lieu de s'arrêter -> ça produit un flux
        # d'enregistrements, un par image, sans tout garder en mémoire d'un coup.
        yield img_path, width, height, boxes


def write_split(records, yolo_dir: Path, split_name: str, class_names):
    """Écrit un split (train, val ou test) sur disque :
    copie les images et écrit, pour chacune, son fichier de label YOLO .txt.
    """
    img_out = yolo_dir / "images" / split_name
    lbl_out = yolo_dir / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for img_path, w, h, boxes in records:
        shutil.copy(img_path, img_out / img_path.name)  # copie la photo telle quelle
        label_lines = []
        for cls_idx, xmin, ymin, xmax, ymax in boxes:
            # Conversion VOC -> YOLO, une boîte à la fois.
            cx, cy, bw, bh = voc_bbox_to_yolo((xmin, ymin, xmax, ymax), w, h)
            # Format attendu par ultralytics : "classe cx cy w h", une ligne par boîte.
            label_lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        # Nouveau fichier .txt, même nom que l'image mais extension différente.
        # Le XML original n'est jamais touché.
        (lbl_out / f"{img_path.stem}.txt").write_text("\n".join(label_lines))


def main(config_path: str):
    cfg = load_config(config_path)
    random.seed(cfg["project"]["seed"])  # graine fixe -> le mélange aléatoire est reproductible

    class_names = cfg["data"]["classes"]
    class_to_idx = {name: i for i, name in enumerate(class_names)}  # ex: {"pothole": 4, ...}
    raw_dir = Path(cfg["data"]["raw_dir"])   # data/UAV-PDD2023
    yolo_dir = Path(cfg["data"]["yolo_dir"])  # data/yolo (dossier de sortie)

    # On charge TOUTES les annotations en mémoire (list() consomme le générateur).
    records = list(load_raw_annotations(raw_dir, class_to_idx))
    random.shuffle(records)  # mélange avant de découper, pour éviter tout biais d'ordre

    n = len(records)
    r_train, r_val, _ = cfg["data"]["split"]  # ex: [0.8, 0.1, 0.1]
    i_train, i_val = int(n * r_train), int(n * (r_train + r_val))

    # Découpage simple par indices : [0:i_train] train, [i_train:i_val] val, le reste test.
    write_split(records[:i_train], yolo_dir, "train", class_names)
    write_split(records[i_train:i_val], yolo_dir, "val", class_names)
    write_split(records[i_val:], yolo_dir, "test", class_names)

    # data.yaml : fichier que YOLO (ultralytics) lira directement à l'entraînement,
    # pour savoir où sont les images de chaque split et quel indice correspond
    # à quelle classe.
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
    # Ce bloc ne s'exécute que si on lance CE fichier directement
    # (python -m src.data.prepare_dataset), pas si on l'importe ailleurs.
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    main(args.config)
