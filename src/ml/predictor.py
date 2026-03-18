"""
ML Predictor — Loads a trained model and runs inference on live data.

The predictor is the bridge between the ML model and the trading system.
It takes a feature vector (from the indicator engine) and returns a
probability score indicating how confident the model is that a trade
will be profitable.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier

from src.config import Config, PROJECT_ROOT

logger = logging.getLogger("traderbot.ml.predictor")

MODEL_STORE = PROJECT_ROOT / "src" / "ml" / "model_store"


class Predictor:
    """
    ML prediction engine.

    Loads a trained XGBoost model and produces probability scores
    for trade signals. Includes confidence thresholds and signal
    generation logic.
    """

    def __init__(self, config: Config):
        self.config = config
        self.model: Optional[XGBClassifier] = None
        self.version: Optional[str] = None
        self.feature_names: list[str] = []
        self.metadata: dict = {}

        # Confidence thresholds from config
        ml_config = config.get("ml", {})
        self.threshold_high = ml_config.get("confidence_threshold_high", 0.65)
        self.threshold_low = ml_config.get("confidence_threshold_low", 0.55)

    def load_model(self, version: str = None) -> bool:
        """
        Load a model from the model store.

        Args:
            version: Specific version to load (e.g., "v1.0").
                     If None, loads the latest version.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        if version is None:
            version = self._get_latest_version()
            if version is None:
                logger.error("No trained model found in model store.")
                return False

        model_path = MODEL_STORE / f"model_{version}.joblib"
        meta_path = MODEL_STORE / f"model_{version}_meta.json"

        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return False

        try:
            self.model = joblib.load(model_path)
            self.version = version

            if meta_path.exists():
                with open(meta_path) as f:
                    self.metadata = json.load(f)
                self.feature_names = self.metadata.get("feature_names", [])

            logger.info(
                f"Loaded model {version} "
                f"(accuracy={self.metadata.get('metrics', {}).get('accuracy', 'N/A')})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load model {version}: {e}", exc_info=True)
            return False

    def predict(self, features: dict) -> float:
        """
        Get probability score for a single feature vector.

        Args:
            features: Dict of feature_name → value (from IndicatorEngine)

        Returns:
            Probability (0.0 to 1.0) that the trade will be profitable.
            Returns 0.0 if model not loaded or features invalid.
        """
        if self.model is None:
            logger.error("No model loaded. Call load_model() first.")
            return 0.0

        try:
            # Build feature array in correct column order
            if self.feature_names:
                feature_values = [features.get(name, 0.0) for name in self.feature_names]
            else:
                feature_values = list(features.values())

            X = np.array([feature_values])
            probability = self.model.predict_proba(X)[0][1]

            return float(probability)

        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            return 0.0

    def predict_batch(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get probability scores for a batch of feature vectors.

        Args:
            X: DataFrame where each row is a feature vector.

        Returns:
            Array of probabilities.
        """
        if self.model is None:
            logger.error("No model loaded.")
            return np.zeros(len(X))

        try:
            # Ensure column order matches training
            if self.feature_names:
                X_ordered = X[self.feature_names]
            else:
                X_ordered = X

            return self.model.predict_proba(X_ordered)[:, 1]

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}", exc_info=True)
            return np.zeros(len(X))

    def get_signal(self, features: dict, indicators_agree: bool = False) -> dict:
        """
        Generate a trading signal from features.

        If no model is loaded (or model has poor AUC), uses rules-only mode:
        indicators_agree is the sole gate.

        With a model loaded:
        - P >= threshold_high -> TRADE (high confidence)
        - P >= threshold_low AND indicators_agree -> TRADE (confirmed)
        - P < threshold_low -> SKIP
        """
        # Rules-only mode: no model loaded
        if self.model is None:
            if indicators_agree:
                return {
                    "action": "trade",
                    "probability": 0.5,
                    "confidence": "rules",
                    "reason": "Rules-only mode: indicators confirm direction",
                }
            return {
                "action": "skip",
                "probability": 0.5,
                "confidence": "low",
                "reason": "Rules-only mode: indicators do not confirm",
            }

        probability = self.predict(features)

        if probability >= self.threshold_high:
            return {
                "action": "trade",
                "probability": probability,
                "confidence": "high",
                "reason": f"ML confidence {probability:.1%} above high threshold {self.threshold_high:.1%}",
            }

        if probability >= self.threshold_low and indicators_agree:
            return {
                "action": "trade",
                "probability": probability,
                "confidence": "medium",
                "reason": (
                    f"ML confidence {probability:.1%} above low threshold "
                    f"{self.threshold_low:.1%} with indicator confirmation"
                ),
            }

        if probability >= self.threshold_low:
            return {
                "action": "skip",
                "probability": probability,
                "confidence": "medium",
                "reason": (
                    f"ML confidence {probability:.1%} above low threshold but "
                    "indicators do not confirm"
                ),
            }

        return {
            "action": "skip",
            "probability": probability,
            "confidence": "low",
            "reason": f"ML confidence {probability:.1%} below threshold {self.threshold_low:.1%}",
        }

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance from the loaded model."""
        if self.model is None:
            return {}

        if self.feature_names:
            return dict(zip(self.feature_names, self.model.feature_importances_))

        return self.metadata.get("feature_importance", {})

    def _get_latest_version(self) -> Optional[str]:
        """Get the latest model version from the store."""
        latest_path = MODEL_STORE / "latest_version.txt"
        if latest_path.exists():
            return latest_path.read_text().strip()
        return None
