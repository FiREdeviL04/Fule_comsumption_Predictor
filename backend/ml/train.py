"""Model training and persistence for fuel prediction."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_COLUMNS, GRID_PARAMS, MODEL_PATHS, RANDOM_STATE, TARGET_COLUMNS
from .data import clean_dataset, ensure_directories, load_dataset


def _build_pipeline() -> Pipeline:
    """Create preprocessing and model pipeline."""

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


def _split_features_targets(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """Split cleaned dataset into feature matrix and target vectors."""

    X = df[FEATURE_COLUMNS].copy()
    y = {name: df[column].copy() for name, column in TARGET_COLUMNS.items()}
    return X, y


def train_models(force_retrain: bool = False) -> Dict[str, Dict[str, object]]:
    """Train all model variants and save joblib artifacts."""

    ensure_directories()
    source = clean_dataset(load_dataset())
    X, targets = _split_features_targets(source)

    all_metrics: Dict[str, Dict[str, object]] = {}
    for model_name, target in targets.items():
        if MODEL_PATHS[model_name].exists() and not force_retrain:
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            target,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )

        search = GridSearchCV(
            estimator=_build_pipeline(),
            param_grid=GRID_PARAMS,
            cv=5,
            scoring="r2",
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)

        metrics = {
            "r2": float(r2_score(y_test, y_pred)),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "cv_score": float(search.best_score_),
            "best_params": search.best_params_,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        }

        estimator = best_model.named_steps["model"]
        feature_importance = {
            feature: float(score)
            for feature, score in zip(FEATURE_COLUMNS, estimator.feature_importances_)
        }

        bundle = {
            "model_name": model_name,
            "target_column": TARGET_COLUMNS[model_name],
            "feature_columns": FEATURE_COLUMNS,
            "trained_at": datetime.utcnow().isoformat(timespec="seconds"),
            "metrics": metrics,
            "feature_importance": feature_importance,
            "model": best_model,
        }
        joblib.dump(bundle, MODEL_PATHS[model_name])
        all_metrics[model_name] = metrics

    return all_metrics
