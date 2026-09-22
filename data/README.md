# Data

Datasets are **not** committed (see `.gitignore`). Download and extract here.

## UAV-PDD2023 (recommended starting point)
Small UAV pavement-distress dataset (~2,440 drone images, ~11k instances, 6 classes).
Search "UAV-PDD2023 dataset" — the data article links the download.
Extract so that images and Pascal VOC annotations sit under `data/UAV-PDD2023/`.

## RDD2022 (larger, optional)
Multi-national road-damage dataset that includes a UAV/drone subset
(`RDD2022_*_Drone`). Available via the arXiv paper 2209.08538 and its repository.

## Road segments (for aggregation)
`data/road_segments.gpkg` is downloaded automatically by `src.geo.fetch_osm_network` from
OpenStreetMap (Overpass API) — every real drivable street in a small Montréal neighbourhood.
No manual step needed unless you want to survey a different area (pass `--bbox`).

## Montréal pavement condition (real training data for the segment model)
`data/montreal_pci_segments.gpkg` is downloaded automatically by
`src.model.fetch_montreal_pci_data` from the City of Montréal's open data portal
(~16,000 real segments with a measured Pavement Condition Index), then enriched in place by
`src.model.enrich_with_voirie_age` with real pavement age (construction/resurfacing date) and
material, spatially joined from Montréal's "Voirie" road-asset survey (no shared key between
the two datasets — see that script's docstring). Sources (both CC-BY 4.0):
- https://donnees.montreal.ca/dataset/condition-chaussees-reseau-routier
- https://donnees.montreal.ca/dataset/voirie-actif

`data/voirie_streets_raw.csv` (~340 MB) is cached here after the first download so re-running
`enrich_with_voirie_age` doesn't re-download it.
