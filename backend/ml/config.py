"""Configuration and constants for backend ML workflows."""
from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "saved_models"
HISTORY_PATH = BASE_DIR / "prediction_history.csv"

DATASET_CANDIDATES = [
    DATA_DIR / "clean_fuel_model_ready.csv",
    DATA_DIR / "clean_fuel.csv",
    BASE_DIR.parent / "data" / "clean_fuel_model_ready.csv",
    BASE_DIR.parent / "data" / "clean_fuel.csv",
    BASE_DIR.parent / "clean_fuel.csv",
]

TARGET_COLUMNS = {
    "fuel": "FUEL CONSUMPTION",
    "hwy": "HWY (L/100 km)",
    "comb": "COMB (L/100 km)",
}

FEATURE_COLUMNS = ["ENGINE SIZE", "CYLINDERS", "EMISSIONS", "EFFICIENCY"]
BASE_FEATURE_COLUMNS = ["ENGINE SIZE", "CYLINDERS", "EMISSIONS"]

MODEL_PATHS = {
    "fuel": MODEL_DIR / "fuel_model.pkl",
    "hwy": MODEL_DIR / "hwy_model.pkl",
    "comb": MODEL_DIR / "comb_model.pkl",
}

RANDOM_STATE = 42
GRID_PARAMS = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [5, 10, 15],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2, 4],
}
