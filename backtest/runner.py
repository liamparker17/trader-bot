"""
Backtest Runner — End-to-end script for training and backtesting.

Usage:
    python -m backtest.runner

Steps:
    1. Load M15 data (4 years) for all instruments -> ML training
    2. Load H1 data for trend bias during training
    3. Train XGBoost on combined multi-instrument M15 dataset
    4. Run backtest on M1 test data (last ~100 days)
    5. Print results
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, DATA_DIR
from src.indicators.engine import IndicatorEngine
from src.ml.feature_builder import FeatureBuilder
from src.ml.trainer import Trainer
from src.ml.predictor import Predictor
from backtest.simulator import BacktestSimulator


def _load_m15_h1_training_data(config, logger) -> dict:
    """
    Load M15 + H1 data for all instruments for rich ML training.

    M15 = entry timeframe features (100k candles per instrument, ~4 years)
    H1 = trend bias (30k candles per instrument, ~8 years)

    Returns dict of {instrument: (m15_df, h1_df)}.
    """
    cache_dir = DATA_DIR / "historical"
    instruments = config.get_enabled_instruments()
    result = {}

    for inst in instruments:
        m15_path = cache_dir / f"{inst}_M15.parquet"
        h1_path = cache_dir / f"{inst}_H1.parquet"

        if m15_path.exists() and h1_path.exists():
            m15 = pd.read_parquet(m15_path)
            h1 = pd.read_parquet(h1_path)
            if len(m15) > 5000:
                result[inst] = (m15, h1)
                logger.info(f"Training data: {inst} -- {len(m15):,} M15, {len(h1):,} H1 candles")

    return result


def _load_m1_m15_backtest_data(config, logger) -> dict:
    """Load M1 + M15 data for backtesting (last ~100 days)."""
    cache_dir = DATA_DIR / "historical"
    instruments = config.get_enabled_instruments()
    result = {}

    for inst in instruments:
        m1_path = cache_dir / f"{inst}_M1.parquet"
        m15_path = cache_dir / f"{inst}_M15.parquet"

        if m1_path.exists() and m15_path.exists():
            m1 = pd.read_parquet(m1_path)
            m15 = pd.read_parquet(m15_path)
            if len(m1) > 1000:
                result[inst] = (m1, m15)
                logger.info(f"Backtest data: {inst} -- {len(m1):,} M1, {len(m15):,} M15 candles")

    return result


def run_backtest():
    """Execute the full backtest pipeline with rich multi-instrument training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("backtest.runner")

    # 1. Load config
    config = load_config()
    logger.info("Config loaded")

    # 2. Initialize indicator engine
    engine = IndicatorEngine(config)
    feature_builder = FeatureBuilder(config, engine)

    # 3. Load M15 + H1 training data (4 years, all instruments)
    training_data = _load_m15_h1_training_data(config, logger)

    if not training_data:
        logger.error("No M15/H1 training data found. Run: python -m src.main --fetch-data")
        return

    # 4. Build combined multi-instrument training dataset
    logger.info("=" * 60)
    logger.info("Building multi-instrument M15 training dataset...")
    logger.info("=" * 60)

    all_X = []
    all_y = []

    # Save original settings, adjust for M15 timeframe
    original_hold = feature_builder.max_hold_bars
    original_sl = feature_builder.sl_atr_mult
    original_tp = feature_builder.tp_atr_mult
    feature_builder.max_hold_bars = 4  # 4 M15 bars = 60 minutes

    # Use 80% of M15 data for training, keep last 20% out
    for inst, (m15_df, h1_df) in training_data.items():
        logger.info(f"Processing {inst}...")

        # Use first 80% for training
        split = int(len(m15_df) * 0.8)
        m15_train = m15_df.iloc[:split]
        h1_train = h1_df[h1_df.index <= m15_train.index[-1]]

        # Get per-instrument SL/TP multipliers for labeling
        inst_cfg = config.get_instrument(inst)
        if inst_cfg:
            feature_builder.sl_atr_mult = inst_cfg.get("sl_atr_multiplier", 1.5)
            feature_builder.tp_atr_mult = inst_cfg.get("tp_atr_multiplier", 2.5)

        try:
            X, y = feature_builder.build_dataset(m15_train, h1_train)
            if len(X) > 500:
                # Add instrument identifier (encoded)
                inst_map = {"EUR_USD": 0, "GBP_USD": 1, "USD_JPY": 2, "XAU_USD": 3}
                X = X.copy()
                X["instrument_id"] = inst_map.get(inst, 4)
                all_X.append(X)
                all_y.append(y)
                logger.info(f"  {inst}: {len(X):,} samples, {y.mean():.1%} positive")
            else:
                logger.warning(f"  {inst}: Too few samples ({len(X)}), skipping")
        except Exception as e:
            logger.error(f"  {inst}: Failed to build dataset: {e}")
            import traceback
            traceback.print_exc()

    # Restore original settings
    feature_builder.max_hold_bars = original_hold
    feature_builder.sl_atr_mult = original_sl
    feature_builder.tp_atr_mult = original_tp

    if not all_X:
        logger.error("No training data could be built")
        return

    # Combine all instruments
    X_combined = pd.concat(all_X, ignore_index=True)
    y_combined = pd.concat(all_y, ignore_index=True)

    logger.info("")
    logger.info(f"Combined training dataset: {len(X_combined):,} samples from {len(all_X)} instruments")
    logger.info(f"Positive rate: {y_combined.mean():.1%}")
    logger.info(f"Features: {len(X_combined.columns)}")

    # 5. Train model on combined data
    logger.info("")
    logger.info("Training XGBoost on combined multi-instrument dataset...")
    trainer = Trainer(config)
    result = trainer.train(X_combined, y_combined, test_ratio=0.2)
    logger.info(f"\n{result.summary()}")

    # Walk-forward validation (5 splits for more data)
    logger.info("Running walk-forward validation (5 folds)...")
    wf_results = trainer.walk_forward_validate(X_combined, y_combined, n_splits=5)

    # Save model
    trainer.save_model(result)
    logger.info("Model saved")

    # 6. Load model into predictor
    predictor = Predictor(config)
    predictor.load_model()

    model_auc = result.metrics.get("auc_roc", 0)
    if model_auc < 0.52:
        logger.warning(f"Model AUC {model_auc:.3f} < 0.52 -- using rules-only mode")
        predictor.model = None
    else:
        logger.info(f"Model AUC {model_auc:.3f} -- ML model ENABLED")

    # 7. Run backtest on M1 test data (last ~30% of 100 days)
    logger.info("")
    logger.info("=" * 60)
    logger.info("Running M1 backtest on test data...")
    logger.info("=" * 60)

    simulator = BacktestSimulator(config, engine, predictor)
    backtest_data = _load_m1_m15_backtest_data(config, logger)
    all_results = []

    for inst, (m1_df, m15_df) in backtest_data.items():
        inst_split = int(len(m1_df) * 0.7)
        m1_test = m1_df.iloc[inst_split:]
        m15_test = m15_df[m15_df.index >= m1_test.index[0]]

        inst_config = config.get_instrument(inst)
        spread = inst_config.get("typical_spread_pips", 1.2) if inst_config else 1.2

        logger.info(f"Backtesting {inst}...")
        bt_result = simulator.run(m1_test, m15_test, instrument=inst, spread_pips=spread)
        all_results.append(bt_result)

    # 8. Print results
    total_trades = 0
    total_wins = 0
    total_pnl = 0.0
    starting = config.get("account.starting_balance", 500)

    for results in all_results:
        if "error" in results:
            continue
        inst = results.get("instrument", "?")
        print()
        print("=" * 60)
        print(f"  {inst} BACKTEST")
        print("=" * 60)
        print(f"  Trades:        {results.get('total_trades', 0)}")
        print(f"  Win Rate:      {results.get('win_rate', 0):.1%}")
        print(f"  Profit Factor: {results.get('profit_factor', 0):.2f}")
        print(f"  Total PnL:     ${results.get('total_pnl_zar', 0):.2f}")
        print(f"  Return:        {results.get('return_pct', 0):.1f}%")
        print(f"  Max Drawdown:  {results.get('max_drawdown_pct', 0):.1f}%")
        print(f"  Sharpe Ratio:  {results.get('sharpe_ratio', 0):.2f}")
        print(f"  Exit Reasons:  {results.get('exit_reasons', {})}")
        print("=" * 60)

        total_trades += results.get("total_trades", 0)
        total_wins += results.get("wins", 0)
        total_pnl += results.get("total_pnl_zar", 0)

    combined_wr = total_wins / total_trades if total_trades > 0 else 0
    combined_return = total_pnl / starting * 100

    print()
    print("=" * 60)
    print(f"  COMBINED ({len(all_results)} instruments)")
    print("=" * 60)
    print(f"  Total Trades:    {total_trades}")
    print(f"  Win Rate:        {combined_wr:.1%}")
    print(f"  Total PnL:       ${total_pnl:.2f}")
    print(f"  Combined Return: {combined_return:.1f}%")
    print("=" * 60)

    print()
    print("TRAINING DATA SUMMARY:")
    print(f"  M15 training samples: {len(X_combined):,} (from {len(all_X)} instruments)")
    print(f"  M15 data range: ~4 years (2022-2026)")
    print(f"  Model AUC: {model_auc:.3f}")
    print(f"  Mode: {'ML-assisted' if model_auc >= 0.52 else 'Rules-only'}")

    print()
    print("PASS CRITERIA CHECK (combined):")
    checks = [
        ("Win rate > 48%", combined_wr > 0.48),
        ("Return > 5%", combined_return > 5.0),
        ("Minimum 50 trades", total_trades >= 50),
    ]
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    all_passed = all(p for _, p in checks)
    all_passed_str = "ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED"
    print()
    print(f"Overall: {all_passed_str}")

    return all_results[0] if len(all_results) == 1 else {"combined": all_results}


if __name__ == "__main__":
    run_backtest()
