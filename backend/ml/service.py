"""Inference and insights service layer."""
from __future__ import annotations

from typing import Dict, List

import joblib
import pandas as pd

from .config import FEATURE_COLUMNS, MODEL_PATHS
from .data import append_history, ensure_directories, load_history, make_history_record
from .train import train_models


class ModelService:
    """Handles model loading, inference, metrics, and history."""

    def __init__(self, bundles: Dict[str, Dict[str, object]]) -> None:
        self.bundles = bundles

    @classmethod
    def create(cls) -> "ModelService":
        """Load models from disk and train missing artifacts."""

        ensure_directories()
        if not all(path.exists() for path in MODEL_PATHS.values()):
            train_models(force_retrain=False)

        bundles: Dict[str, Dict[str, object]] = {}
        for name, path in MODEL_PATHS.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing model artifact: {path}")
            bundles[name] = joblib.load(path)

        return cls(bundles)

    def validate_input(self, payload: Dict[str, object]) -> Dict[str, float]:
        """Validate and normalize incoming payload for prediction."""

        required = ["engine_size", "cylinders", "emissions"]
        for key in required:
            if key not in payload:
                raise ValueError(f"Missing required field: {key}")

        try:
            engine_size = float(payload["engine_size"])
            cylinders = int(payload["cylinders"])
            emissions = float(payload["emissions"])
        except (TypeError, ValueError) as exc:
            raise ValueError("engine_size, cylinders, and emissions must be numeric.") from exc

        if engine_size <= 0 or cylinders <= 0 or emissions <= 0:
            raise ValueError("All input values must be greater than zero.")

        return {
            "engine_size": engine_size,
            "cylinders": cylinders,
            "emissions": emissions,
        }

    def _build_feature_frame(self, payload: Dict[str, float]) -> pd.DataFrame:
        """Construct one-row dataframe for model inference."""

        efficiency = payload["engine_size"] / payload["cylinders"]
        return pd.DataFrame(
            [[payload["engine_size"], payload["cylinders"], payload["emissions"], efficiency]],
            columns=FEATURE_COLUMNS,
        )

    def predict(self, payload: Dict[str, object]) -> Dict[str, object]:
        """Run model inference, persist history, and return response payload."""

        clean_payload = self.validate_input(payload)
        X = self._build_feature_frame(clean_payload)

        predictions = {
            name: float(bundle["model"].predict(X)[0])
            for name, bundle in self.bundles.items()
        }

        history_row = make_history_record(clean_payload, predictions)
        append_history(history_row)

        metrics = {
            name: {
                "r2": float(bundle["metrics"].get("r2", 0.0)),
                "mae": float(bundle["metrics"].get("mae", 0.0)),
                "cv_score": float(bundle["metrics"].get("cv_score", 0.0)),
            }
            for name, bundle in self.bundles.items()
        }

        avg_metrics = {
            "r2": sum(value["r2"] for value in metrics.values()) / len(metrics),
            "mae": sum(value["mae"] for value in metrics.values()) / len(metrics),
            "cv_score": sum(value["cv_score"] for value in metrics.values()) / len(metrics),
        }

        return {
            "fuel": predictions["fuel"],
            "hwy": predictions["hwy"],
            "comb": predictions["comb"],
            "metrics": avg_metrics,
            "model_metrics": metrics,
        }

    def insights(self) -> Dict[str, object]:
        """Return model metrics and feature importance."""

        performance = {
            name: bundle["metrics"]
            for name, bundle in self.bundles.items()
        }
        feature_importance = {
            name: bundle.get("feature_importance", {})
            for name, bundle in self.bundles.items()
        }

        return {
            "performance": performance,
            "feature_importance": feature_importance,
        }

    def history(self, limit: int = 100) -> List[Dict[str, object]]:
        """Return recent history rows."""

        return load_history(limit=limit)
