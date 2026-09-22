-- Reference table: the road segments themselves. Attributes change rarely.
CREATE TABLE IF NOT EXISTS segments (
    seg_id              INTEGER PRIMARY KEY,
    road_class          TEXT NOT NULL,
    traffic_aadt        REAL,
    pavement_age_years  REAL,
    num_lanes           INTEGER,
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
    priority_score   REAL,  -- computed from real detections
    predicted_score  REAL,  -- estimated by the scikit-learn condition model
    priority_rank    INTEGER
);
