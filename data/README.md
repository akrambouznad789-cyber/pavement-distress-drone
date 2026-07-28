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
Provide a line-geometry layer of the road network you surveyed as
`data/road_segments.gpkg` (GeoPackage) or update `geo.segments_path` in the config.
For Québec roads, MTQ / open-data road-network layers work well.
