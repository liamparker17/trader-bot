"""
ML Trainer — Trains and retrains XGBoost models with walk-forward validation.

Walk-forward validation:
  Instead of random train/test splits (which leak future info), we split
  chronologically: train on older data, test on newer data. This simulates
  real trading where you only know the past.

Model versioning:
  Each trained model is saved with a version number, timestamp, and
  performance metrics. Old models are kept for comparison.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

from src.config import Config, PROJECT_ROOT

logger = logging.getLogger("traderbot.ml.trainer")

MODEL_STORE = PROJECT_ROOT / "src" / "ml" / "model_store"


class TrainingResult:
    """Holds the results of a training run."""

    def __init__(
        self,
        model: XGBClassifier,
        version: str,
        metrics: dict,
        feature_names: list[str],
        feature_importance: dict[str, float],
        train_size: int,
        test_size: int,
        timestamp: str,
    ):
        self.model = model
        self.version = version
        self.metrics = metrics
        self.feature_names = feature_names
        self.feature_importance = feature_importance
        self.train_size = train_size
        self.test_size = test_size
        self.timestamp = timestamp

    def summary(self) -> str:
        """Human-readable training summary."""
        lines = [
            f"Model: {self.version}",
            f"Trained: {self.timestamp}",
            f"Train size: {self.train_size} | Test size: {self.test_size}",
            f"Accuracy: {self.metrics['accuracy']:.4f}",
            f"Precision: {self.metrics['precision']:.4f}",
            f"Recall: {self.metrics['recall']:.4f}",
            f"F1 Score: {self.metrics['f1']:.4f}",
            f"AUC-ROC: {self.metrics.get('auc_roc', 'N/A')}",
            f"",
            f"Top 5 Features:",
        ]
        sorted_features = sorted(
            self.feature_importance.items(), key=lambda x: x[1], reverse=True
        )
        for name, importance in sorted_features[:5]:
            lines.append(f"  {name}: {importance:.4f}")

        return "\n".join(lines)


class Trainer:
    """
    XGBoost model trainer with walk-forward validation.

    Handles:
    - Initial training on historical data
    - Walk-forward cross-validation (train on past, test on future)
    - Model versioning and persistence
    - Comparison between old and new models
    - Adaptive retraining triggers
    """

    def __init__(self, config: Config):
        self.config = config
        MODEL_STORE.mkdir(parents=True, exist_ok=True)

        # XGBoost hyperparameters from config
        ml_config = config.get("ml", {})
        xgb_config = ml_config.get("xgboost", {})

        early_stop = xgb_config.get("early_stopping_rounds", 50)
        self.xgb_params = {
            "n_estimators": xgb_config.get("n_estimators", 500),
            "max_depth": xgb_config.get("max_depth", 5),
            "learning_rate": xgb_config.get("learning_rate", 0.03),
            "min_child_weight": xgb_config.get("min_child_weight", 5),
            "subsample": xgb_config.get("subsample", 0.8),
            "colsample_bytree": xgb_config.get("colsample_bytree", 0.8),
            "gamma": xgb_config.get("gamma", 0.5),
            "reg_alpha": xgb_config.get("reg_alpha", 0.1),
            "reg_lambda": xgb_config.get("reg_lambda", 1.0),
            "scale_pos_weight": xgb_config.get("scale_pos_weight", 2.0),
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }
        if early_stop and early_stop > 0:
            self.xgb_params["early_stopping_rounds"] = early_stop

        self.test_months = ml_config.get("walk_forward_test_months", 2)
        self.min_samples = ml_config.get("min_training_samples", 2000)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_ratio: float = 0.2,
    ) -> TrainingResult:
        """
        Train a model using walk-forward split.

        The data is split chronologically:
        - First (1 - test_ratio) of data → training
        - Last test_ratio of data → testing

        Args:
            X: Feature DataFrame (must be time-sorted)
            y: Label Series
            test_ratio: Fraction of data for testing (default 20%)

        Returns:
            TrainingResult with model, metrics, and feature importance.
        """
        if len(X) < self.min_samples:
            logger.warning(
                f"Dataset too small ({len(X)} < {self.min_samples}). "
                "Training anyway but results may be unreliable."
            )

        # Walk-forward split (chronological, no shuffling)
        split_idx = int(len(X) * (1 - test_ratio))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        logger.info(
            f"Training: {len(X_train)} samples | Testing: {len(X_test)} samples"
        )
        logger.info(
            f"Train positive rate: {y_train.mean():.3f} | "
            f"Test positive rate: {y_test.mean():.3f}"
        )

        # Train with early stopping
        model = XGBClassifier(**self.xgb_params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        }

        try:
            metrics["auc_roc"] = float(roc_auc_score(y_test, y_prob))
        except ValueError:
            metrics["auc_roc"] = 0.0

        # Feature importance
        importance = dict(zip(X.columns, model.feature_importances_))

        # Version
        version = self._next_version()
        timestamp = datetime.now(timezone.utc).isoformat()

        result = TrainingResult(
            model=model,
            version=version,
            metrics=metrics,
            feature_names=list(X.columns),
            feature_importance=importance,
            train_size=len(X_train),
            test_size=len(X_test),
            timestamp=timestamp,
        )

        logger.info(f"Training complete:\n{result.summary()}")
        return result

    def walk_forward_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
    ) -> list[dict]:
        """
        Perform walk-forward cross-validation.

        Splits data into n_splits sequential folds. For each fold:
        - Train on all data before the fold
        - Test on the fold

        This gives a more realistic estimate of live performance than
        a single train/test split.

        Returns:
            List of metric dicts, one per fold.
        """
        fold_size = len(X) // (n_splits + 1)
        results = []

        for i in range(n_splits):
            train_end = fold_size * (i + 2)
            test_start = train_end
            test_end = min(test_start + fold_size, len(X))

            if test_end <= test_start:
                break

            X_train = X.iloc[:train_end]
            y_train = y.iloc[:train_end]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]

            if len(X_train) < 100 or len(X_test) < 50:
                continue

            params = self.xgb_params.copy()
            model = XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            fold_metrics = {
                "fold": i + 1,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            }

            try:
                fold_metrics["auc_roc"] = float(roc_auc_score(y_test, y_prob))
            except ValueError:
                fold_metrics["auc_roc"] = 0.0

            results.append(fold_metrics)
            logger.info(
                f"Fold {i + 1}: accuracy={fold_metrics['accuracy']:.3f} "
                f"precision={fold_metrics['precision']:.3f} "
                f"auc={fold_metrics['auc_roc']:.3f}"
            )

        if results:
            avg_acc = np.mean([r["accuracy"] for r in results])
            avg_auc = np.mean([r["auc_roc"] for r in results])
            logger.info(
                f"Walk-forward validation: avg accuracy={avg_acc:.3f} avg AUC={avg_auc:.3f}"
            )

        return results

    def save_model(self, result: TrainingResult) -> Path:
        """Save a trained model and its metadata to the model store."""
        model_path = MODEL_STORE / f"model_{result.version}.joblib"
        meta_path = MODEL_STORE / f"model_{result.version}_meta.json"

        # Save model
        joblib.dump(result.model, model_path)

        # Save metadata (convert numpy types to native Python for JSON)
        def _to_native(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        metadata = {
            "version": result.version,
            "timestamp": result.timestamp,
            "metrics": {k: _to_native(v) for k, v in result.metrics.items()},
            "feature_names": result.feature_names,
            "feature_importance": {k: _to_native(v) for k, v in result.feature_importance.items()},
            "train_size": result.train_size,
            "test_size": result.test_size,
            "xgb_params": {k: _to_native(v) for k, v in self.xgb_params.items()
                           if isinstance(v, (int, float, str, bool, np.integer, np.floating))},
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Update "latest" symlink/pointer
        latest_path = MODEL_STORE / "latest_version.txt"
        latest_path.write_text(result.version)

        logger.info(f"Model saved: {model_path}")
        return model_path

    def compare_models(
        self,
        current_result: TrainingResult,
        new_result: TrainingResult,
        min_improvement: float = 0.01,
    ) -> bool:
        """
        Compare a new model against the current model.

        Returns True if new model should replace current.

        Criteria:
        - New model AUC-ROC is at least min_improvement better
        - OR new model accuracy is at least min_improvement better
        - AND new model isn't significantly worse on any metric
        """
        current = current_result.metrics
        new = new_result.metrics

        auc_better = new.get("auc_roc", 0) > current.get("auc_roc", 0) + min_improvement
        acc_better = new["accuracy"] > current["accuracy"] + min_improvement
        no_worse = new["precision"] >= current["precision"] - 0.05

        should_swap = (auc_better or acc_better) and no_worse

        logger.info(
            f"Model comparison: current_auc={current.get('auc_roc', 0):.4f} "
            f"new_auc={new.get('auc_roc', 0):.4f} | "
            f"current_acc={current['accuracy']:.4f} new_acc={new['accuracy']:.4f} | "
            f"swap={'YES' if should_swap else 'NO'}"
        )

        return should_swap

    def _next_version(self) -> str:
        """Generate the next model version string."""
        latest_path = MODEL_STORE / "latest_version.txt"
        if latest_path.exists():
            current = latest_path.read_text().strip()
            try:
                major, minor = current.lstrip("v").split(".")
                return f"v{major}.{int(minor) + 1}"
            except (ValueError, AttributeError):
                pass
        return "v1.0"

    @staticmethod
    def load_model_metadata(version: str) -> Optional[dict]:
        """Load metadata for a specific model version."""
        meta_path = MODEL_STORE / f"model_{version}_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f)
        return None
