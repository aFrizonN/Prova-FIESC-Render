"""
Historical similarity engine for retrieving similar past machine events and operational context.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import joblib
import logging

import config
from src.data_processing import ALL_MODEL_FEATURES, load_processed_data, prepare_single_event_features

logger = logging.getLogger(__name__)

SIMILARITY_MODEL_PATH = config.MODELS_DIR / "similarity_nn.joblib"
SIMILARITY_SCALER_PATH = config.MODELS_DIR / "similarity_scaler.joblib"
HISTORICAL_DATA_PATH = config.DATA_DIR / "historical_cache.parquet"


class SimilarityEngine:
    """Finds historical occurrences with vibration profiles similar to incoming events."""

    def __init__(self, top_k: int = 10):
        self.top_k = top_k
        self.nn_model: Optional[NearestNeighbors] = None
        self.scaler: Optional[StandardScaler] = None
        self.df_history: Optional[pd.DataFrame] = None

    def fit(self, df: Optional[pd.DataFrame] = None) -> None:
        """Fits NearestNeighbors on historical dataset."""
        if df is None:
            df = load_processed_data()

        logger.info(f"Fitting Similarity Engine on {len(df):,} historical records...")
        self.df_history = df.copy().reset_index(drop=True)

        X = self.df_history[ALL_MODEL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.nn_model = NearestNeighbors(n_neighbors=self.top_k, metric="cosine", n_jobs=-1)
        self.nn_model.fit(X_scaled)

        # Save artifacts
        joblib.dump(self.nn_model, SIMILARITY_MODEL_PATH)
        joblib.dump(self.scaler, SIMILARITY_SCALER_PATH)
        self.df_history.to_parquet(HISTORICAL_DATA_PATH, index=False)
        logger.info("Similarity Engine trained and saved.")

    def load_artifacts(self) -> bool:
        """Loads cached similarity model and historical data."""
        if not (SIMILARITY_MODEL_PATH.exists() and SIMILARITY_SCALER_PATH.exists() and HISTORICAL_DATA_PATH.exists()):
            return False
        self.nn_model = joblib.load(SIMILARITY_MODEL_PATH)
        self.scaler = joblib.load(SIMILARITY_SCALER_PATH)
        self.df_history = pd.read_parquet(HISTORICAL_DATA_PATH)
        return True

    def find_similar(self, event_data: dict, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Locates historical occurrences with characteristics close to the analyzed equipment.
        Returns:
            - similar_count: int
            - top_matches: List[dict] (similarity %, id, created_at, fault, rpm, temp)
            - time_distribution: Dict (occurrences over time)
            - operational_context: Dict (RPM, temp, vibration stats of similar events)
            - frequency: Dict (estimated frequency and cadence)
        """
        if self.nn_model is None or self.df_history is None:
            if not self.load_artifacts():
                self.fit()

        k = top_k or self.top_k
        feat_df = prepare_single_event_features(event_data)
        feat_scaled = self.scaler.transform(feat_df)

        # Find neighbors (cosine distances: 0 means identical, 2 means opposite)
        distances, indices = self.nn_model.kneighbors(feat_scaled, n_neighbors=k)
        
        neighbor_indices = indices[0]
        neighbor_distances = distances[0]

        similar_rows = self.df_history.iloc[neighbor_indices].copy()
        # Convert cosine distance to percentage similarity (1 - dist) * 100
        similarities = [max(0.0, min(100.0, (1.0 - d) * 100.0)) for d in neighbor_distances]
        similar_rows["similarity_score"] = similarities

        # 1. Top matches list
        top_matches = []
        for _, row in similar_rows.iterrows():
            top_matches.append({
                "id": int(row.get("id", 0)),
                "created_at": str(row.get("created_at", "")),
                "fault": str(row.get("fault", "")),
                "fault_category": str(row.get("fault_category", "")),
                "similarity_score": round(float(row.get("similarity_score", 0.0)), 2),
                "rpm": float(row.get("rpm", 0.0)),
                "temperature_c": float(row.get("temperature_c", 0.0)),
                "z_rms_velocity_mm_s": float(row.get("z_rms_velocity_mm_s", 0.0)),
                "x_rms_velocity_mm_s": float(row.get("x_rms_velocity_mm_s", 0.0)),
            })

        # 2. Total category occurrences in dataset for broader context
        predicted_fault = top_matches[0]["fault_category"] if top_matches else "desconhecido"
        category_subset = self.df_history[self.df_history["fault_category"] == predicted_fault]
        total_category_count = len(category_subset)

        # 3. Time distribution of matching category
        if not category_subset.empty:
            cat_copy = category_subset.copy()
            cat_copy["date"] = pd.to_datetime(cat_copy["created_at"]).dt.strftime("%Y-%m-%d")
            time_dist_series = cat_copy["date"].value_counts().sort_index().to_dict()
        else:
            time_dist_series = {}

        # 4. Operational Context (from nearest similar events)
        operational_context = {
            "avg_rpm": round(float(similar_rows["rpm"].mean()), 1),
            "rpm_range": [round(float(similar_rows["rpm"].min()), 1), round(float(similar_rows["rpm"].max()), 1)],
            "avg_temp_c": round(float(similar_rows["temperature_c"].mean()), 2),
            "temp_c_range": [round(float(similar_rows["temperature_c"].min()), 2), round(float(similar_rows["temperature_c"].max()), 2)],
            "avg_z_rms_velocity": round(float(similar_rows["z_rms_velocity_mm_s"].mean()), 3),
            "avg_x_rms_velocity": round(float(similar_rows["x_rms_velocity_mm_s"].mean()), 3),
            "primary_historical_fault": str(similar_rows["fault"].mode().iloc[0]) if not similar_rows.empty else "N/A",
        }

        # 5. Frequency & Recurrence
        if len(time_dist_series) > 1:
            total_days = max(1, len(time_dist_series))
            avg_events_per_active_day = round(total_category_count / total_days, 1)
        else:
            avg_events_per_active_day = float(total_category_count)

        return {
            "similar_count": len(top_matches),
            "total_historical_occurrences": total_category_count,
            "top_matches": top_matches,
            "time_distribution": time_dist_series,
            "operational_context": operational_context,
            "frequency_summary": {
                "events_per_active_day": avg_events_per_active_day,
                "first_seen": str(category_subset["created_at"].min()) if not category_subset.empty else "N/A",
                "last_seen": str(category_subset["created_at"].max()) if not category_subset.empty else "N/A",
            }
        }


# Global similarity singleton
similarity_instance = SimilarityEngine()


def get_similarity_engine() -> SimilarityEngine:
    """Returns the singleton similarity engine instance."""
    global similarity_instance
    if similarity_instance.nn_model is None:
        similarity_instance.load_artifacts()
    return similarity_instance
