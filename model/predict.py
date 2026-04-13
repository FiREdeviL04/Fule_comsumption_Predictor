"""Prediction helpers for the fuel prediction application."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib

from .train import train_all_models
from .utils import FEATURE_COLUMNS, MODEL_PATHS, PredictionInput, TARGET_LABELS, ensure_directories


@dataclass(frozen=True)
class LoadedModel:
    """Represents a persisted model bundle loaded from disk."""

    model_name: str
    model: object
    metrics: Dict[str, object]
    target_column: str


class FuelPredictionService:
    """Load persisted models and provide prediction and insight utilities."""

    def __init__(self, models: Dict[str, LoadedModel]):
        self._models = models

    @classmethod
    def from_disk(cls) -> "FuelPredictionService":
        """Load persisted models, training any missing ones if required."""

        ensure_directories()
        if not all(path.exists() for path in MODEL_PATHS.values()):
            train_all_models(force_retrain=False)

        models: Dict[str, LoadedModel] = {}
        for model_name, path in MODEL_PATHS.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing trained model artifact: {path}")
            bundle = joblib.load(path)
            best_model = bundle["best_model"] if isinstance(bundle, dict) else bundle
            metrics = bundle.get("metrics", {}) if isinstance(bundle, dict) else {}
            target_column = bundle.get("target_column", "") if isinstance(bundle, dict) else ""
            models[model_name] = LoadedModel(
                model_name=model_name,
                model=best_model,
                metrics=metrics,
                target_column=target_column,
            )
        return cls(models)

    def predict(self, prediction_input: PredictionInput) -> Dict[str, float]:
        """Generate predictions for all target models."""

        feature_frame = prediction_input.to_frame()
        predictions: Dict[str, float] = {}
        for model_name, loaded_model in self._models.items():
            prediction = loaded_model.model.predict(feature_frame)[0]
            predictions[model_name] = float(prediction)
        return predictions

    def get_metrics(self) -> Dict[str, Dict[str, object]]:
        """Return all stored evaluation metrics."""

        return {model_name: loaded_model.metrics for model_name, loaded_model in self._models.items()}

    def get_feature_importances(self, model_name: str = "comb_model") -> List[Tuple[str, float]]:
        """Return feature importances for a persisted random forest model."""

        if model_name not in self._models:
            raise KeyError(f"Unknown model requested: {model_name}")

        model = self._models[model_name].model
        if not hasattr(model, "named_steps"):
            return []

        estimator = model.named_steps.get("model")
        if estimator is None or not hasattr(estimator, "feature_importances_"):
            return []

        importance_values = estimator.feature_importances_
        feature_names = FEATURE_COLUMNS
        if len(importance_values) != len(feature_names):
            feature_names = feature_names[: len(importance_values)]

        return sorted(
            zip(feature_names, importance_values),
            key=lambda item: item[1],
            reverse=True,
        )

    def get_display_rows(self) -> List[Dict[str, object]]:
        """Format model metrics for the UI tables."""

        rows: List[Dict[str, object]] = []
        for model_name, loaded_model in self._models.items():
            metrics = loaded_model.metrics
            rows.append(
                {
                    "Model": model_name,
                    "Target": TARGET_LABELS.get(model_name, model_name),
                    "R2": metrics.get("r2_score", 0.0),
                    "MAE": metrics.get("mae", 0.0),
                    "CV Mean": metrics.get("cv_mean_score", 0.0),
                    "Best Params": metrics.get("best_params", {}),
                }
            )
        return rows
