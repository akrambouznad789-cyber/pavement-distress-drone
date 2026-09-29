-- Reference table: the road segments themselves. Attributes change rarely.
-- arrondissement/longueur_m/indice_iri/pavement_age_years/materiau_chaussee
-- are matched in from two separate City of Montreal surveys (PCI + Voirie,
-- no shared key -- see segment_model.py and enrich_with_voirie_age.py) --
-- real, measured values, not invented. NULL means no matching real Montreal
-- segment was found nearby (real coverage gap, not an error) -- happens for
-- a handful of the 32 demo segments.
CREATE TABLE IF NOT EXISTS segments (
    seg_id              INTEGER PRIMARY KEY,
    arrondissement      TEXT,
    longueur_m          REAL,
    indice_iri          REAL,
    pavement_age_years  REAL,
    materiau_chaussee   TEXT,
    geometry_wkt        TEXT  -- segment shape, as WKT text (SQLite has no native geometry type)
);

-- Fact table: one row per detected distress instance. Grows with every drone survey.
CREATE TABLE IF NOT EXISTS detections (
    detection_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    image         TEXT NOT NULL,
    class_name    TEXT NOT NULL,
    confidence    REAL NOT NULL,
    xmin REAL, ymin REAL, xmax REAL, ymax REAL,
    lat  REAL,
    lon  REAL,
    seg_id        INTEGER REFERENCES segments(seg_id)  -- NULL if beyond snap_distance_m
);

-- Result table: one row per segment, the pipeline's final output.
-- Snapshot only (latest run overwrites) -- this project isn't re-run on a schedule.
CREATE TABLE IF NOT EXISTS segment_scores (
    seg_id           INTEGER PRIMARY KEY REFERENCES segments(seg_id),
    n_detections     INTEGER,
    worst_class      TEXT,
    priority_score   REAL,  -- computed from real drone detections
    predicted_score  REAL,  -- PCI predicted by the scikit-learn condition model (0-100, higher = better)
    real_indice_pci  REAL,  -- the City of Montreal's own measured PCI for this stretch
    priority_rank    INTEGER
);
