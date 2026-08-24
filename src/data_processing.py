"""
Data preprocessing, cleaning, feature engineering, and preparation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import logging

import config
from src.constants import SENSOR_FEATURES, map_fault_to_category, is_operational_state

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Engineered feature names
ENGINEERED_FEATURES = [
    "vel_ratio_zx",
    "acc_ratio_zx",
    "rms_acc_ratio_zx",
    "kurtosis_ratio_zx",
    "crest_ratio_zx",
    "high_freq_ratio_zx",
    "vibration_energy_z",
    "vibration_energy_x",
    "total_vibration_energy",
]

ALL_MODEL_FEATURES = SENSOR_FEATURES + ENGINEERED_FEATURES


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates domain-specific feature engineering for vibration analysis."""
    df = df.copy()
    eps = 1e-6

    df["vel_ratio_zx"] = df["z_rms_velocity_mm_s"] / (df["x_rms_velocity_mm_s"] + eps)
    df["acc_ratio_zx"] = df["z_peak_acceleration_g"] / (df["x_peak_acceleration_g"] + eps)
    df["rms_acc_ratio_zx"] = df["z_rms_acceleration_g"] / (df["x_rms_acceleration_g"] + eps)
    df["kurtosis_ratio_zx"] = df["z_kurtosis"] / (df["x_kurtosis"] + eps)
    df["crest_ratio_zx"] = df["z_crest_factor"] / (df["x_crest_factor"] + eps)
    df["high_freq_ratio_zx"] = df["z_high_freq_rms_accel_g"] / (df["x_high_freq_rms_accel_g"] + eps)

    df["vibration_energy_z"] = (df["z_rms_velocity_mm_s"] ** 2) + (df["z_rms_acceleration_g"] ** 2)
    df["vibration_energy_x"] = (df["x_rms_velocity_mm_s"] ** 2) + (df["x_rms_acceleration_g"] ** 2)
    df["total_vibration_energy"] = df["vibration_energy_z"] + df["vibration_energy_x"]

    return df


def load_and_preprocess_raw_data(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Loads banner.csv, cleans data, maps categories, and engineers features."""
    if csv_path is None:
        csv_path = config.RAW_CSV_PATH

    logger.info(f"Loading raw dataset from {csv_path}...")
    df = pd.read_csv(csv_path)

    logger.info(f"Loaded {len(df)} records. Parsing timestamps...")
    df["created_at"] = pd.to_datetime(df["created_at"])

    # Ensure numeric columns are strictly numeric
    for col in SENSOR_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Add macro category and problem indicator
    logger.info("Mapping raw fault labels to macro categories...")
    df["fault_category"] = df["fault"].apply(map_fault_to_category)
    df["is_problem"] = ~df["fault_category"].isin(["normal", "desconhecido"])

    # Extract engineered features
    logger.info("Computing engineered vibration features...")
    df = extract_features(df)

    # Sort chronologically
    df = df.sort_values("created_at").reset_index(drop=True)

    # Save processed parquet cache
    logger.info(f"Saving processed dataset to {config.PROCESSED_CSV_PATH}...")
    df.to_parquet(config.PROCESSED_CSV_PATH, index=False)

    return df


def load_processed_data() -> pd.DataFrame:
    """Loads cached processed parquet if available, otherwise generates it."""
    if config.PROCESSED_CSV_PATH.exists():
        logger.info(f"Loading cached dataset from {config.PROCESSED_CSV_PATH}...")
        return pd.read_parquet(config.PROCESSED_CSV_PATH)
    else:
        return load_and_preprocess_raw_data()


def prepare_single_event_features(event_json: dict) -> pd.DataFrame:
    """Transforms a single incoming sensor JSON dictionary into a model-ready feature DataFrame."""
    # Convert dict to single row DataFrame
    df = pd.DataFrame([event_json])

    for col in SENSOR_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df = extract_features(df)
    return df[ALL_MODEL_FEATURES]


if __name__ == "__main__":
    df = load_and_preprocess_raw_data()
    print("Dataset processed successfully!")
    print(f"Total rows: {len(df):,}")
    print("\nFault categories distribution:")
    print(df["fault_category"].value_counts())
