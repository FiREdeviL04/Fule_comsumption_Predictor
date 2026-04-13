"""Model training pipeline for the fuel prediction application."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .utils import (
    FEATURE_COLUMNS,
    MODEL_PATHS,
    ModelMetrics,
    TARGET_COLUMNS,
    clean_dataset,
    ensure_directories,
    load_dataset,
    prepare_features_and_targets,
)

RANDOM_STATE = 42
PARAM_GRID = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [5, 10, 15],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2, 4],
}


@dataclass(frozen=True)
class TrainedModelBundle:
    """Container for a trained model and its evaluation metrics."""

    model_name: str
    target_column: str
    model_path: str
    metrics: ModelMetrics
    best_estimator: Pipeline

    def to_disk_bundle(self) -> Dict[str, object]:
        return {
            "model_name": self.model_name,
            "target_column": self.target_column,
            "feature_columns": FEATURE_COLUMNS,
            "trained_at": datetime.utcnow().isoformat(timespec="seconds"),
            "metrics": self.metrics.to_dict(),
            "best_model": self.best_estimator,
        }


def build_pipeline() -> Pipeline:
    """Create the preprocessing and model pipeline used for all targets."""

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestRegressor(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_single_model(model_name: str, features: pd.DataFrame, target: pd.Series) -> TrainedModelBundle:
    """Train, tune, evaluate, and persist a single model."""

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    search = GridSearchCV(
        estimator=build_pipeline(),
        param_grid=PARAM_GRID,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    best_estimator = search.best_estimator_
    test_predictions = best_estimator.predict(X_test)

    metrics = ModelMetrics(
        r2_score=float(r2_score(y_test, test_predictions)),
        mae=float(mean_absolute_error(y_test, test_predictions)),
        cv_mean_score=float(search.best_score_),
        best_params=search.best_params_,
        train_rows=int(len(X_train)),
        test_rows=int(len(X_test)),
    )

    bundle = TrainedModelBundle(
        model_name=model_name,
        target_column=TARGET_COLUMNS[model_name],
        model_path=str(MODEL_PATHS[model_name]),
        metrics=metrics,
        best_estimator=best_estimator,
    )
    joblib.dump(bundle.to_disk_bundle(), MODEL_PATHS[model_name])
    return bundle


def train_all_models(force_retrain: bool = False) -> Dict[str, TrainedModelBundle]:
    """Train all target models or reuse existing persisted artifacts."""

    ensure_directories()
    raw_dataset = load_dataset()
    dataset = clean_dataset(raw_dataset)
    features, targets = prepare_features_and_targets(dataset)

    trained_models: Dict[str, TrainedModelBundle] = {}
    for model_name in TARGET_COLUMNS:
        model_path = MODEL_PATHS[model_name]
        if model_path.exists() and not force_retrain:
            continue
        trained_models[model_name] = train_single_model(model_name, features, targets[model_name])

    return trained_models


def train_and_save_missing_models() -> None:
    """Train any missing model artifacts needed by the GUI."""

    train_all_models(force_retrain=False)
