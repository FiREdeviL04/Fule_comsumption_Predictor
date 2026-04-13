"""Shared utilities for the fuel prediction application."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "saved_models"
HISTORY_FILE = PROJECT_ROOT / "prediction_history.csv"

BASE_FEATURE_COLUMNS: List[str] = ["ENGINE SIZE", "CYLINDERS", "EMISSIONS"]
FEATURE_COLUMNS: List[str] = ["ENGINE SIZE", "CYLINDERS", "EMISSIONS", "EFFICIENCY"]
TARGET_COLUMNS: Dict[str, str] = {
    "fuel_model": "FUEL CONSUMPTION",
    "hwy_model": "HWY (L/100 km)",
    "comb_model": "COMB (L/100 km)",
}
MODEL_PATHS: Dict[str, Path] = {
    "fuel_model": MODEL_DIR / "fuel_model.pkl",
    "hwy_model": MODEL_DIR / "hwy_model.pkl",
    "comb_model": MODEL_DIR / "comb_model.pkl",
}
TARGET_LABELS: Dict[str, str] = {
    "fuel_model": "Fuel Consumption",
    "hwy_model": "Highway Fuel Consumption",
    "comb_model": "Combined Fuel Consumption",
}


@dataclass(frozen=True)
class PredictionInput:
    """Validated feature values used by the predictive models."""

    engine_size: float
    cylinders: int
    emissions: int

    def to_frame(self) -> pd.DataFrame:
        """Convert the input into a one-row DataFrame for model inference."""

        efficiency = self.engine_size / self.cylinders
        return pd.DataFrame(
            [[self.engine_size, self.cylinders, self.emissions, efficiency]],
            columns=FEATURE_COLUMNS,
        )


@dataclass(frozen=True)
class ModelMetrics:
    """Evaluation metrics collected during model training."""

    r2_score: float
    mae: float
    cv_mean_score: float
    best_params: Dict[str, object]
    train_rows: int
    test_rows: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "r2_score": self.r2_score,
            "mae": self.mae,
            "cv_mean_score": self.cv_mean_score,
            "best_params": self.best_params,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
        }


def ensure_directories() -> None:
    """Create the project folders used by the application."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def dataset_candidates() -> Iterable[Path]:
    """Return the possible locations for the training dataset."""

    yield DATA_DIR / "clean_fuel_model_ready.csv"
    yield DATA_DIR / "clean_fuel.csv"
    yield PROJECT_ROOT / "clean_fuel_model_ready.csv"
    yield PROJECT_ROOT / "clean_fuel.csv"


def get_dataset_path() -> Path:
    """Return the first available dataset path."""

    for candidate in dataset_candidates():
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find clean_fuel.csv. Expected it in data/clean_fuel.csv or the project root."
    )


def load_dataset() -> pd.DataFrame:
    """Load the source dataset with a friendly error message."""

    dataset_path = get_dataset_path()
    return pd.read_csv(dataset_path)


def clean_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataset and engineer additional model features."""

    required_columns = BASE_FEATURE_COLUMNS + list(TARGET_COLUMNS.values())
    missing_columns = [column for column in required_columns if column not in dataset.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    cleaned = dataset[required_columns].copy()
    cleaned = cleaned.dropna().drop_duplicates()

    cleaned = cleaned[(cleaned["ENGINE SIZE"] > 0) & (cleaned["CYLINDERS"] > 0) & (cleaned["EMISSIONS"] > 0)]
    cleaned = cleaned[(cleaned["ENGINE SIZE"] < 8) & (cleaned["EMISSIONS"] < 500)]

    cleaned["EFFICIENCY"] = cleaned["ENGINE SIZE"] / cleaned["CYLINDERS"]
    cleaned = cleaned.replace([float("inf"), float("-inf")], pd.NA).dropna()
    cleaned = cleaned.reset_index(drop=True)
    return cleaned


def load_prediction_history(limit: int = 5) -> pd.DataFrame:
    """Load recent prediction history from disk."""

    if not HISTORY_FILE.exists():
        return pd.DataFrame()

    try:
        history = pd.read_csv(HISTORY_FILE)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    return history.tail(limit)


def append_prediction_history(record: Dict[str, object]) -> None:
    """Append a prediction record to the persistent history file."""

    history = pd.DataFrame([record])
    if HISTORY_FILE.exists():
        history.to_csv(HISTORY_FILE, mode="a", header=False, index=False)
    else:
        history.to_csv(HISTORY_FILE, index=False)


def prepare_features_and_targets(dataset: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """Split the dataset into the shared feature matrix and target series."""

    features = dataset[FEATURE_COLUMNS].copy()
    targets = {model_name: dataset[target_column].copy() for model_name, target_column in TARGET_COLUMNS.items()}
    return features, targets
