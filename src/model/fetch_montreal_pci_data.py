"""
Download the City of Montreal's real pavement condition survey (Auscultation
chaussée) and save it locally. Replaces the old synthetic segment generator:
this is a real, measured Pavement Condition Index (PCI) per street segment,
city-wide (~17,000 segments), collected by an auscultation vehicle -- not a
formula.

Source: Ville de Montréal open data, "Indicateurs de condition des chaussées
du réseau routier" (CC-BY 4.0).
https://donnees.montreal.ca/dataset/condition-chaussees-reseau-routier
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import requests
import yaml

# The server rejects requests with no / a non-browser User-Agent.
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str):
    cfg = load_config(config_path)
    mcfg = cfg["model"]
    out_path = Path(mcfg["features_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {mcfg['pci_source_url']} ...")
    resp = requests.get(mcfg["pci_source_url"], headers=HEADERS, timeout=60)
    resp.raise_for_status()
    tmp = out_path.with_suffix(".download.gpkg")
    tmp.write_bytes(resp.content)

    gdf = gpd.read_file(tmp)
    tmp.unlink()

    # Keep only the columns the model needs, in snake_case, and drop segments
    # missing a measured PCI or IRI (a handful of rows in every real survey).
    gdf = gdf.rename(
        columns={
            "ID_TRC": "seg_id_montreal",
            "Rue": "rue",
            "Longueur": "longueur_m",
            "Arrondissement": "arrondissement",
            "Indice_PCI": "indice_pci",
            "Indice_IRI": "indice_iri",
        }
    )[["seg_id_montreal", "rue", "longueur_m", "arrondissement", "indice_pci", "indice_iri", "geometry"]]
    gdf = gdf.dropna(subset=["indice_pci", "indice_iri"])

    gdf.to_file(out_path, driver="GPKG")
    print(f"{len(gdf)} real segments (PCI + IRI, city-wide) -> {out_path}")
    print(f"Indice_PCI: mean={gdf['indice_pci'].mean():.1f}  min={gdf['indice_pci'].min():.0f}  max={gdf['indice_pci'].max():.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    main(args.config)
