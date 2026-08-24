"""
Machine Learning fault classification model training, evaluation, and inference.
"""

import json
import logging
from typing import Dict, Any, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import config
from src.data_processing import (
    ALL_MODEL_FEATURES,
    load_processed_data,
    prepare_single_event_features,
)

logger = logging.getLogger(__name__)


class FaultClassifier:
    """End-to-end Machine Learning classifier for vibration fault diagnosis."""

    def __init__(self):
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.encoder: Optional[LabelEncoder] = None
        self.classes: list = []

    def train_and_evaluate(
        self,
        df: Optional[pd.DataFrame] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Trains the Random Forest model and saves artifacts and evaluation metrics."""
        if df is None:
            df = load_processed_data()

        logger.info(f"Preparing features from {len(df):,} samples for training...")

        # Feature matrix X and target y
        X = df[ALL_MODEL_FEATURES].copy()
        y = df["fault_category"].copy()

        # Handle any possible NaN or Inf
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Label encoding
        self.encoder = LabelEncoder()
        y_encoded = self.encoder.fit_transform(y)
        self.classes = list(self.encoder.classes_)

        # Train/Test Split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )

        # Feature Scaling
        logger.info("Fitting StandardScaler on feature matrix...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Model instantiation - tuned for speed and robustness on industrial tabular sensor metrics
        logger.info("Training Random Forest Classifier (n_estimators=100, n_jobs=-1)...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        self.model.fit(X_train_scaled, y_train)

        # Evaluation
        logger.info("Evaluating classifier performance on test set...")
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
        report = classification_report(
            y_test, y_pred, target_names=self.classes, output_dict=True
        )
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Feature Importance
        importances = dict(
            zip(ALL_MODEL_FEATURES, self.model.feature_importances_.tolist())
        )
        sorted_importances = dict(
            sorted(importances.items(), key=lambda item: item[1], reverse=True)
        )

        metrics = {
            "accuracy": float(acc),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "classes": self.classes,
            "classification_report": report,
            "confusion_matrix": cm,
            "top_features": list(sorted_importances.items())[:10],
            "total_samples": len(df),
            "test_samples": len(y_test),
        }

        # Save artifacts
        self.save_artifacts(metrics)
        logger.info(f"Model successfully trained! Test Accuracy: {acc:.4f}, F1-Macro: {f1_macro:.4f}")
        return metrics

    def save_artifacts(self, metrics: Dict[str, Any]) -> None:
        """Serializes model, scaler, encoder, and metrics to disk."""
        logger.info(f"Saving model artifacts to {config.MODELS_DIR}...")
        joblib.dump(self.model, config.CLASSIFIER_PATH)
        joblib.dump(self.scaler, config.SCALER_PATH)
        joblib.dump(self.encoder, config.ENCODER_PATH)
        with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    def load_artifacts(self) -> bool:
        """Loads trained model artifacts from disk."""
        if not (config.CLASSIFIER_PATH.exists() and config.SCALER_PATH.exists() and config.ENCODER_PATH.exists()):
            return False
        
        self.model = joblib.load(config.CLASSIFIER_PATH)
        self.scaler = joblib.load(config.SCALER_PATH)
        self.encoder = joblib.load(config.ENCODER_PATH)
        self.classes = list(self.encoder.classes_)
        return True

    def predict(self, event_data: dict) -> Tuple[str, float, Dict[str, float]]:
        """
        Infers fault category and confidence for an incoming event JSON.
        Returns:
            - predicted_category: str
            - confidence: float (0.0 to 100.0)
            - probabilities: Dict[category, probability_percentage]
        """
        if self.model is None:
            if not self.load_artifacts():
                raise RuntimeError("Classifier model artifacts not found. Run training first.")

        # Prepare features
        feat_df = prepare_single_event_features(event_data)
        feat_scaled = self.scaler.transform(feat_df)

        probas = self.model.predict_proba(feat_scaled)[0]
        top_idx = int(np.argmax(probas))
        predicted_category = self.classes[top_idx]
        confidence = float(probas[top_idx] * 100.0)

        all_probas = {
            cls_name: round(float(p * 100.0), 2)
            for cls_name, p in zip(self.classes, probas)
        }

        return predicted_category, confidence, all_probas


# Global classifier singleton
classifier_instance = FaultClassifier()


def get_classifier() -> FaultClassifier:
    """Returns the singleton classifier instance."""
    global classifier_instance
    if classifier_instance.model is None:
        classifier_instance.load_artifacts()
    return classifier_instance


if __name__ == "__main__":
    classifier = FaultClassifier()
    metrics = classifier.train_and_evaluate()
    print("Evaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4%}")
    print(f"F1 Macro: {metrics['f1_macro']:.4%}")
    print("\nTop 5 Most Important Features:")
    for feat, imp in metrics["top_features"][:5]:
        print(f" - {feat}: {imp:.4f}")
