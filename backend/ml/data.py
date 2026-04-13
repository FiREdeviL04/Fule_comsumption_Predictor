"""Dataset loading, cleaning, and history helpers."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import (
    BASE_FEATURE_COLUMNS,
    DATASET_CANDIDATES,
    HISTORY_PATH,
    MODEL_DIR,
    TARGET_COLUMNS,
)


def ensure_directories() -> None:
    """Create required output directories if they do not exist."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def get_dataset_path() -> Path:
    """Return the first available dataset path from configured candidates."""

    for candidate in DATASET_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No dataset found. Expected clean_fuel.csv in backend/data or project data folders.")


def load_dataset() -> pd.DataFrame:
    """Load raw dataset from disk."""

    return pd.read_csv(get_dataset_path())


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataset and engineer features aligned with model assumptions."""

    required = BASE_FEATURE_COLUMNS + list(TARGET_COLUMNS.values())
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    clean = df[required].copy()

    for col in required:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    clean = clean.dropna()
    clean = clean.drop_duplicates()

    clean = clean[(clean["ENGINE SIZE"] > 0) & (clean["CYLINDERS"] > 0) & (clean["EMISSIONS"] > 0)]
    clean = clean[(clean["ENGINE SIZE"] < 8) & (clean["EMISSIONS"] < 500)]

    clean["EFFICIENCY"] = clean["ENGINE SIZE"] / clean["CYLINDERS"]
    clean = clean.replace([float("inf"), float("-inf")], pd.NA).dropna()
    return clean.reset_index(drop=True)


def append_history(record: Dict[str, object]) -> None:
    """Append one prediction row to history CSV."""

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([record])
    if HISTORY_PATH.exists():
        row.to_csv(HISTORY_PATH, mode="a", header=False, index=False)
    else:
        row.to_csv(HISTORY_PATH, index=False)


def load_history(limit: int = 50) -> List[Dict[str, object]]:
    """Load prediction history rows from disk."""

    if not HISTORY_PATH.exists():
        return []

    try:
        history = pd.read_csv(HISTORY_PATH)
    except pd.errors.EmptyDataError:
        return []

    return history.tail(limit).to_dict("records")


def make_history_record(payload: Dict[str, float], predictions: Dict[str, float]) -> Dict[str, object]:
    """Build the standardized prediction history record."""

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "engine_size": payload["engine_size"],
        "cylinders": payload["cylinders"],
        "emissions": payload["emissions"],
        "fuel": round(predictions["fuel"], 4),
        "hwy": round(predictions["hwy"], 4),
        "comb": round(predictions["comb"], 4),
    }
