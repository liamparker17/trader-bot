"""
Gold-Specific Model Training — Trains a separate XGBoost model for XAU/USD
using cross-asset features (dollar strength, correlations, volume, multi-TF momentum).

Usage:
    python -m backtest.train_gold
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, DATA_DIR
from src.indicators.engine import IndicatorEngine
from src.ml.feature_builder import FeatureBuilder, EXCLUDE_FROM_MODEL
from src.ml.gold_features import build_gold_features
from src.ml.trainer import Trainer
from src.ml.predictor import Predictor
from backtest.simulator import BacktestSimulator


def run():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("backtest.train_gold")

    config = load_config()
    engine = IndicatorEngine(config)

    # 1. Load M15 data for all instruments
    cache = DATA_DIR / "historical"
    gold_m15 = pd.read_parquet(cache / "XAU_USD_M15.parquet")
    eur_m15 = pd.read_parquet(cache / "EUR_USD_M15.parquet")
    gbp_m15 = pd.read_parquet(cache / "GBP_USD_M15.parquet")
    jpy_m15 = pd.read_parquet(cache / "USD_JPY_M15.parquet")
    gold_h1 = pd.read_parquet(cache / "XAU_USD_H1.parquet")

    logger.info(f"Gold M15: {len(gold_m15):,} candles ({gold_m15.index.min().date()} to {gold_m15.index.max().date()})")

    # 2. Build gold-specific cross-asset features
    logger.info("Building gold-specific cross-asset features...")
    gold_extra = build_gold_features(gold_m15, eur_m15, gbp_m15, jpy_m15)
    logger.info(f"Gold extra features: {len(gold_extra.columns)} columns")

    # 3. Build standard indicator features on gold M15
    logger.info("Building standard indicator features on gold M15...")
    feature_builder = FeatureBuilder(config, engine)

    # Use per-instrument SL/TP for labeling
    inst_cfg = config.get_instrument("XAU_USD")
    feature_builder.sl_atr_mult = inst_cfg.get("sl_atr_multiplier", 1.5)
    feature_builder.tp_atr_mult = inst_cfg.get("tp_atr_multiplier", 3.5)
    feature_builder.max_hold_bars = 4  # 4 M15 bars = 60 min

    # Split: 80% train, 20% test
    split = int(len(gold_m15) * 0.8)
    gold_train = gold_m15.iloc[:split]
    h1_train = gold_h1[gold_h1.index <= gold_train.index[-1]]

    X_standard, y = feature_builder.build_dataset(gold_train, h1_train)
    logger.info(f"Standard features: {len(X_standard):,} samples, {len(X_standard.columns)} features")
    logger.info(f"Positive rate: {y.mean():.1%}")

    # 4. Merge gold-specific features into the standard feature matrix
    # Align gold_extra to the same index as X_standard
    gold_extra_aligned = gold_extra.reindex(X_standard.index)

    # Drop any gold_extra columns that are all NaN after alignment
    gold_extra_aligned = gold_extra_aligned.dropna(axis=1, how="all")

    X_combined = pd.concat([X_standard, gold_extra_aligned], axis=1)

    # Drop rows with NaN (warmup period for rolling features)
    valid = X_combined.notna().all(axis=1) & y.reindex(X_combined.index).notna()
    X_combined = X_combined[valid]
    y = y.reindex(X_combined.index)[valid]

    logger.info(f"Combined features: {len(X_combined):,} samples, {len(X_combined.columns)} columns")
    logger.info(f"Positive rate after cleanup: {y.mean():.1%}")

    # Print feature list
    logger.info("Features:")
    for i, col in enumerate(X_combined.columns):
        logger.info(f"  {i+1:2d}. {col}")

    # 5. Train gold-specific model
    logger.info("")
    logger.info("=" * 60)
    logger.info("Training gold-specific XGBoost model...")
    logger.info("=" * 60)

    trainer = Trainer(config)
    result = trainer.train(X_combined, y, test_ratio=0.2)
    logger.info(f"\n{result.summary()}")

    # Walk-forward validation
    logger.info("Walk-forward validation (5 folds)...")
    wf = trainer.walk_forward_validate(X_combined, y, n_splits=5)

    # Save model with gold-specific name
    from src.ml.trainer import MODEL_STORE
    import joblib, json

    model_path = MODEL_STORE / "model_gold_v1.0.joblib"
    meta_path = MODEL_STORE / "model_gold_v1.0_meta.json"

    joblib.dump(result.model, model_path)

    def _native(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    meta = {
        "version": "gold_v1.0",
        "timestamp": result.timestamp,
        "metrics": {k: _native(v) for k, v in result.metrics.items()},
        "feature_names": result.feature_names,
        "feature_importance": {k: _native(v) for k, v in result.feature_importance.items()},
        "train_size": result.train_size,
        "test_size": result.test_size,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Gold model saved: {model_path}")

    # 6. Analyze: what features matter?
    logger.info("")
    logger.info("=" * 60)
    logger.info("GOLD MODEL — TOP 15 FEATURES")
    logger.info("=" * 60)
    sorted_imp = sorted(result.feature_importance.items(), key=lambda x: x[1], reverse=True)
    for name, imp in sorted_imp[:15]:
        bar = "#" * int(imp * 200)
        logger.info(f"  {name:35s} {imp:.4f} {bar}")

    # Count gold-specific vs standard features in top 15
    gold_feats = set(gold_extra.columns)
    top15_names = [n for n, _ in sorted_imp[:15]]
    gold_in_top15 = sum(1 for n in top15_names if n in gold_feats)
    logger.info(f"\nGold-specific features in top 15: {gold_in_top15}/15")

    # 7. Quick backtest comparison: gold model vs rules-only vs old model
    logger.info("")
    logger.info("=" * 60)
    logger.info("BACKTEST COMPARISON")
    logger.info("=" * 60)

    gold_m1 = pd.read_parquet(cache / "XAU_USD_M1.parquet")
    m1_split = int(len(gold_m1) * 0.7)
    m1_test = gold_m1.iloc[m1_split:]
    m15_test = gold_m15[gold_m15.index >= m1_test.index[0]]

    # We'll need to check ML prediction distribution on M1 data
    # to find the right threshold
    predictor_gold = Predictor(config)
    predictor_gold.model = result.model
    predictor_gold.feature_names = result.feature_names
    predictor_gold.version = "gold_v1.0"

    # Check prediction distribution
    logger.info("Checking ML prediction distribution on gold M1 test data...")
    sim = BacktestSimulator(config, engine, predictor_gold)
    m1_ind = engine.calculate_all(m1_test)
    m1_ind["ema_55"] = m1_test["close"].ewm(span=55, adjust=False).mean()
    m15_ind = engine.calculate_all(m15_test)
    m15_trend = sim._compute_m15_trend_series(m15_ind)

    # Build cross-asset data for M1 test period
    eur_m15_test = eur_m15[eur_m15.index >= m1_test.index[0]]
    gbp_m15_test = gbp_m15[gbp_m15.index >= m1_test.index[0]]
    jpy_m15_test = jpy_m15[jpy_m15.index >= m1_test.index[0]]
    gold_m15_test = gold_m15[gold_m15.index >= m1_test.index[0]]

    gold_extra_test = build_gold_features(gold_m15_test, eur_m15_test, gbp_m15_test, jpy_m15_test)

    # Sample predictions
    min_bars = engine.get_required_candle_count()
    probs = []
    for i in range(min_bars, min(min_bars + 1000, len(m1_test))):
        ts = m1_test.index[i]
        feats = sim._build_features_at(m1_ind, m15_trend, i, ts, 30 * 0.01)
        if feats is None:
            continue
        feats["instrument_id"] = 3.0
        # Add gold-specific features (map M1 timestamp to nearest M15)
        m15_idx = gold_extra_test.index.get_indexer([ts], method="ffill")[0]
        if m15_idx >= 0 and m15_idx < len(gold_extra_test):
            for col in gold_extra_test.columns:
                val = gold_extra_test.iloc[m15_idx][col]
                feats[col] = float(val) if pd.notna(val) else 0.0
        else:
            for col in gold_extra_test.columns:
                feats[col] = 0.0

        prob = predictor_gold.predict(feats)
        probs.append(prob)

    probs = np.array(probs)
    logger.info(f"Predictions on {len(probs)} gold M1 candles:")
    logger.info(f"  Min: {probs.min():.4f}, Max: {probs.max():.4f}")
    logger.info(f"  Mean: {probs.mean():.4f}, Median: {np.median(probs):.4f}")
    for pct in [50, 75, 90, 95, 99]:
        logger.info(f"  P{pct}: {np.percentile(probs, pct):.4f}")

    print("\n" + "=" * 60)
    print("Gold model training complete!")
    print(f"AUC: {result.metrics.get('auc_roc', 0):.3f}")
    print(f"Gold-specific features in top 15: {gold_in_top15}/15")
    print(f"Prediction range on M1: {probs.min():.4f} - {probs.max():.4f}")
    print("=" * 60)


if __name__ == "__main__":
    run()
