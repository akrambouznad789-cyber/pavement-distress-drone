"""
Train a scikit-learn model that predicts a road segment's condition
(priority_score) from tabular attributes (road class, traffic, pavement age),
so segments the drone never surveyed can still get an estimated priority.

Steps:
  1. Load the synthetic segment-features training set (see
     src.model.generate_synthetic_segment_features).
  2. Pipeline: ColumnTransformer (one-hot road_class, scale numeric features)
     -> regressor.
  3. Compare RandomForestRegressor vs GradientBoostingRegressor with
     cross-validation, then tune the winner with GridSearchCV.
  4. Evaluate the tuned model on a held-out test split (MAE / RMSE / R2) and
     plot feature importance.
  5. Apply the trained model to the 8 real drone-surveyed segments (assigning
     them synthetic attributes, since we have no real road registry) and
     compare its prediction to the actual priority_score computed from real
     detections in src.geo.aggregate_to_segments -- this is the "estimate
     condition on segments the drone never flew over" workflow, checked
     against segments where we *do* have ground truth.
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

NUMERIC_FEATURES = ["traffic_aadt", "pavement_age_years", "num_lanes"]
CATEGORICAL_FEATURES = ["road_class"]
TARGET = "priority_score_true"


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


def synthetic_features_for_surveyed_segments(seg_ids, seed: int) -> pd.DataFrame:
    """Assign plausible road attributes to the real surveyed segments (no real
    road registry exists), so the trained model can be applied to them."""
    rng = np.random.default_rng(seed)
    n = len(seg_ids)
    road_classes = rng.choice(["residential", "collector", "arterial"], size=n, p=[0.5, 0.3, 0.2])
    base_traffic = {"residential": 2000, "collector": 8000, "arterial": 20000}
    traffic_aadt = np.array([base_traffic[c] for c in road_classes]) * rng.uniform(0.6, 1.4, size=n)
    pavement_age_years = rng.uniform(0, 30, size=n)
    num_lanes = np.array([{"residential": 2, "collector": 2, "arterial": 4}[c] for c in road_classes])
    return pd.DataFrame(
        {
            "seg_id": seg_ids,
            "road_class": road_classes,
            "traffic_aadt": traffic_aadt.round(0),
            "pavement_age_years": pavement_age_years.round(1),
            "num_lanes": num_lanes,
        }
    )


def main(config_path: str):
    cfg = load_config(config_path)
    mcfg = cfg["model"]
    seed = cfg["project"]["seed"]
    out_dir = Path(mcfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(mcfg["features_path"])
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
    order = np.argsort(importances)[::-1]

    plt.figure(figsize=(6, 4))
    plt.barh([feature_names[i] for i in order], importances[order])
    plt.xlabel("Feature importance")
    plt.title(f"Segment condition model ({best_name})")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance.png")
    print(f"Feature importance plot -> {out_dir / 'feature_importance.png'}")

    # Apply to the real drone-surveyed segments and compare to actual scores.
    scored_path = Path("outputs") / "segments_scored.gpkg"
    if scored_path.exists():
        surveyed = gpd.read_file(scored_path)
        surveyed_features = synthetic_features_for_surveyed_segments(surveyed["seg_id"], seed=seed + 1)
        surveyed_features["predicted_score"] = best_model.predict(
            surveyed_features[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        )
        comparison = surveyed_features.merge(
            surveyed[["seg_id", "priority_score"]].rename(columns={"priority_score": "actual_score"}),
            on="seg_id",
        )
        comparison_path = out_dir / "surveyed_vs_predicted.csv"
        comparison.to_csv(comparison_path, index=False)
        print(f"Surveyed segments: predicted vs actual -> {comparison_path}")
        print(comparison[["seg_id", "road_class", "actual_score", "predicted_score"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    main(args.config)
