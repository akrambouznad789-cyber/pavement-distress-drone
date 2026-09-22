"""
Train a scikit-learn model that predicts a road segment's real Pavement
Condition Index (PCI) from attributes available city-wide (segment length,
borough, roughness, and now real pavement age/material), so segments the
drone never surveyed can still get an estimated condition.

Steps:
  1. Load the real Montreal PCI segment table (see
     src.model.fetch_montreal_pci_data, further enriched with real pavement
     age by src.model.enrich_with_voirie_age) -- ~16,000 real, measured
     segments.
  2. Pipeline: ColumnTransformer (one-hot arrondissement + material, scale
     numeric features) -> regressor.
  3. Compare RandomForestRegressor vs GradientBoostingRegressor with
     cross-validation, then tune the winner with GridSearchCV.
  4. Evaluate the tuned model on a held-out test split (MAE / RMSE / R2) and
     plot feature importance.
  5. Spatially match our 8 drone-surveyed demo segments to the real Montreal
     segments that overlap them (same street, real geometry), so they inherit
     real attributes -- then apply the trained model and compare three
     values: our drone-based priority_score, the model's predicted_score,
     and the city's own real indice_pci for that stretch of road.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["longueur_m", "indice_iri", "pavement_age_years"]
CATEGORICAL_FEATURES = ["arrondissement", "materiau_chaussee"]
TARGET = "indice_pci"


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def real_features_for_surveyed_segments(
    segments_path: str, montreal_pci_path: str, metric_crs: str, match_distance_m: float
) -> gpd.GeoDataFrame:
    """Match each of our 8 demo segments to the real Montreal segments whose
    geometry falls within `match_distance_m` of it (there are several real,
    short block-to-block segments per demo segment), and aggregate their real
    attributes -- length-weighted for PCI/IRI, most common borough. This
    replaces inventing attributes: it's the same street, so the real segments
    directly under our demo segments are the honest ground truth.
    """
    demo = gpd.read_file(segments_path).to_crs(metric_crs)
    real = gpd.read_file(montreal_pci_path).to_crs(metric_crs)

    demo_buffered = demo.copy()
    demo_buffered["geometry"] = demo.geometry.buffer(match_distance_m)

    joined = gpd.sjoin(real, demo_buffered[["seg_id", "geometry"]], how="inner", predicate="intersects")

    def weighted_avg(g: pd.DataFrame, col: str) -> float:
        return float(np.average(g[col], weights=g["longueur_m"]))

    rows = []
    for seg_id, g in joined.groupby("seg_id"):
        g_age = g.dropna(subset=["pavement_age_years"])
        rows.append(
            {
                "seg_id": seg_id,
                "longueur_m": weighted_avg(g, "longueur_m"),
                "indice_iri": weighted_avg(g, "indice_iri"),
                "pavement_age_years": weighted_avg(g_age, "pavement_age_years") if len(g_age) else np.nan,
                "arrondissement": g["arrondissement"].mode().iat[0],
                "materiau_chaussee": g["materiau_chaussee"].mode().iat[0],
                "real_indice_pci": weighted_avg(g, "indice_pci"),
                "n_real_segments_matched": len(g),
            }
        )
    return pd.DataFrame(rows)


def main(config_path: str):
    cfg = load_config(config_path)
    mcfg = cfg["model"]
    seed = cfg["project"]["seed"]
    out_dir = Path(mcfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = gpd.read_file(mcfg["features_path"]).dropna(subset=NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=mcfg["test_size"], random_state=seed
    )

    candidates = {
        "random_forest": RandomForestRegressor(random_state=seed),
        "gradient_boosting": GradientBoostingRegressor(random_state=seed),
    }
    cv_scores = {}
    for name, reg in candidates.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("reg", reg)])
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2")
        cv_scores[name] = scores.mean()
        print(f"{name}: CV R2 = {scores.mean():.3f} (+/- {scores.std():.3f})")

    best_name = max(cv_scores, key=cv_scores.get)
    print(f"Best candidate: {best_name}")

    # Tune the winning model's hyperparameters.
    param_grid = (
        {"reg__n_estimators": [100, 300], "reg__max_depth": [None, 5, 10]}
        if best_name == "random_forest"
        else {"reg__n_estimators": [100, 300], "reg__learning_rate": [0.05, 0.1]}
    )
    pipe = Pipeline([("prep", build_preprocessor()), ("reg", candidates[best_name])])
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="r2")
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    print(f"Best params: {grid.best_params_}")

    y_pred = best_model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": mean_squared_error(y_test, y_pred) ** 0.5,
        "r2": r2_score(y_test, y_pred),
    }
    print(f"Test set: MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}  R2={metrics['r2']:.3f}")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"cv_r2": cv_scores, "best_params": grid.best_params_, "test": metrics}, f, indent=2)

    # Feature importance (tree-based models expose feature_importances_).
    prep = best_model.named_steps["prep"]
    feature_names = NUMERIC_FEATURES + list(
        prep.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
    )
    importances = best_model.named_steps["reg"].feature_importances_
    order = np.argsort(importances)[::-1][:15]  # city-wide boroughs = many one-hot columns; show top 15

    plt.figure(figsize=(7, 5))
    plt.barh([feature_names[i] for i in order][::-1], importances[order][::-1])
    plt.xlabel("Feature importance")
    model_abbrev = {"random_forest": "Random Forest", "gradient_boosting": "GBM"}[best_name]
    plt.title(f"Segment condition model ({model_abbrev})", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance.png")
    print(f"Feature importance plot -> {out_dir / 'feature_importance.png'}")

    # Match our 8 demo segments to real Montreal segments, apply the model,
    # and compare against both our drone-based score and the city's real PCI.
    scored_path = Path("outputs") / "segments_scored.gpkg"
    if scored_path.exists():
        geo = cfg["geo"]
        real_matched = real_features_for_surveyed_segments(
            geo["segments_path"], mcfg["features_path"], geo["metric_crs"], mcfg["match_distance_m"]
        )
        real_matched["predicted_score"] = best_model.predict(
            real_matched[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        )

        surveyed = gpd.read_file(scored_path)
        comparison = real_matched.merge(
            surveyed[["seg_id", "priority_score"]].rename(columns={"priority_score": "drone_priority_score"}),
            on="seg_id",
        )
        comparison_path = out_dir / "surveyed_vs_predicted.csv"
        comparison.to_csv(comparison_path, index=False)
        print(f"Surveyed segments: drone score vs model prediction vs real city PCI -> {comparison_path}")
        print(
            comparison[
                ["seg_id", "arrondissement", "n_real_segments_matched", "drone_priority_score", "predicted_score", "real_indice_pci"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    main(args.config)
