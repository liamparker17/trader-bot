"""
ML Feature Builder — Creates labeled training datasets from historical candle data.

Takes raw candle data, calculates all indicators, then labels each candle
with the outcome of a hypothetical trade:
  - Label 1: If entering at this candle, price hits take-profit before stop-loss
  - Label 0: If entering at this candle, price hits stop-loss first (or neither)

This creates the supervised learning dataset for the XGBoost model.
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

from src.config import Config
from src.indicators.engine import IndicatorEngine

logger = logging.getLogger("traderbot.ml.features")

# Features excluded from ML model training:
# - trend_15min: circular (used as direction gate AND feature)
# - Raw price-scale features: add noise, not normalized signal
EXCLUDE_FROM_MODEL = {
    "trend_15min",
    "ema_fast",
    "ema_slow",
    "bb_upper",
    "bb_lower",
    "bb_middle",
    "atr_value",
    "macd_value",
    "macd_signal",
    "hour_of_day",  # Replaced by cyclical session features
}


class FeatureBuilder:
    """
    Builds labeled feature datasets for ML training.

    Process:
    1. Calculate all indicators on historical M1 and M15 candle data
    2. For each candle, look forward to determine trade outcome
    3. Combine indicator features + price action + context into feature matrix
    4. Return (X, y) ready for model training
    """

    def __init__(self, config: Config, indicator_engine: IndicatorEngine):
        self.config = config
        self.engine = indicator_engine

        # Risk/reward params from config
        self.sl_atr_mult = config.get("risk.sl_atr_multiplier", 1.5)
        self.tp_atr_mult = config.get("risk.tp_atr_multiplier", 2.25)
        self.max_hold_bars = config.get("trading.max_hold_minutes", 60)

    def build_dataset(
        self,
        m1_df: pd.DataFrame,
        m15_df: pd.DataFrame,
        direction: str = "both",
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Build a labeled dataset from historical candle data.

        Args:
            m1_df: 1-minute candles with DatetimeIndex and OHLCV columns
            m15_df: 15-minute candles (for trend bias)
            direction: "buy", "sell", or "both" (labels for both directions)

        Returns:
            (X, y) where X is the feature DataFrame and y is the label Series.
            Rows with insufficient data or NaN features are dropped.
        """
        logger.info(
            f"Building dataset from {len(m1_df)} M1 candles and "
            f"{len(m15_df)} M15 candles (direction={direction})"
        )

        # Step 1: Calculate all indicators on M1 data
        m1_indicators = self.engine.calculate_all(m1_df)

        # Step 2: Calculate M15 trend bias for each M1 candle
        m15_indicators = self.engine.calculate_all(m15_df)
        trend_bias = self._map_m15_trend_to_m1(m1_df, m15_indicators)

        # Step 3: Build feature matrix
        features = self._build_feature_matrix(m1_indicators, trend_bias)

        # Step 4: Label each candle with trade outcome
        labels_buy = self._label_candles(m1_df, m1_indicators, direction="buy")
        labels_sell = self._label_candles(m1_df, m1_indicators, direction="sell")

        if direction == "buy":
            labels = labels_buy
        elif direction == "sell":
            labels = labels_sell
        else:
            # "both": for each candle, pick the direction matching trend bias
            # If trend bullish → label as buy. If bearish → label as sell.
            labels = pd.Series(0, index=m1_df.index, dtype=int)
            bullish = trend_bias >= 1
            bearish = trend_bias <= -1
            labels[bullish] = labels_buy[bullish]
            labels[bearish] = labels_sell[bearish]

            # IMPORTANT: Exclude neutral-trend candles entirely from training.
            # Previously these were labeled 0 which poisoned the model —
            # they're not "bad trades", they're "no-trade zones".
            tradeable = bullish | bearish
            labels = labels[tradeable]
            features = features.loc[tradeable]

        # Step 5: Align and clean
        X = features.loc[labels.index]

        # Drop rows where indicators haven't warmed up (NaN values)
        min_bars = self.engine.get_required_candle_count()
        X = X.iloc[min_bars:]
        labels = labels.iloc[min_bars:]

        # Drop rows that still have NaN (shouldn't be many after warmup)
        valid_mask = X.notna().all(axis=1)
        X = X[valid_mask]
        labels = labels[valid_mask]

        # Drop rows where we can't look forward enough for labeling
        # (last max_hold_bars candles can't be labeled)
        cutoff = len(X) - self.max_hold_bars
        if cutoff > 0:
            X = X.iloc[:cutoff]
            labels = labels.iloc[:cutoff]

        # Step 6: Drop features excluded from model (circular/noisy)
        cols_to_drop = [c for c in X.columns if c in EXCLUDE_FROM_MODEL]
        if cols_to_drop:
            logger.info(f"Dropping {len(cols_to_drop)} excluded features: {cols_to_drop}")
            X = X.drop(columns=cols_to_drop)

        logger.info(
            f"Dataset built: {len(X)} samples, {len(X.columns)} features, "
            f"{labels.sum()} positive ({labels.mean() * 100:.1f}% win rate), "
            f"{len(X) - labels.sum()} negative"
        )

        return X, labels

    def build_features_single(
        self,
        m1_df: pd.DataFrame,
        m15_df: pd.DataFrame,
        current_spread: float = 0.0,
    ) -> Optional[dict]:
        """
        Build a single feature vector for live prediction.
        Delegates to indicator engine's build_feature_vector.
        """
        return self.engine.build_feature_vector(m1_df, m15_df, current_spread)

    def _build_feature_matrix(
        self,
        m1_indicators: pd.DataFrame,
        trend_bias: pd.Series,
    ) -> pd.DataFrame:
        """Assemble the full feature matrix from indicator data."""
        feature_cols = []

        # All indicator columns (exclude raw OHLCV)
        ohlcv_cols = {"open", "high", "low", "close", "volume"}
        for col in m1_indicators.columns:
            if col not in ohlcv_cols:
                feature_cols.append(col)

        features = m1_indicators[feature_cols].copy()

        # Add trend bias (kept for labeling/direction, excluded from model later)
        features["trend_15min"] = trend_bias

        # Add price action features
        features["candle_body_ratio"] = self._candle_body_ratio(m1_indicators)
        features["upper_wick_ratio"] = self._upper_wick_ratio(m1_indicators)
        features["lower_wick_ratio"] = self._lower_wick_ratio(m1_indicators)
        features["close_vs_open"] = (m1_indicators["close"] > m1_indicators["open"]).astype(int)

        # Add time context
        features["hour_of_day"] = m1_indicators.index.hour.astype(float)
        features["day_of_week"] = m1_indicators.index.weekday.astype(float)

        # Spread not available in historical data — use 0 as placeholder
        features["spread_current"] = 0.0

        # --- Interaction features ---
        # Momentum matters more in certain volatility regimes
        rsi_dist = features.get("rsi_distance_50")
        vol_regime = features.get("volatility_regime")
        if rsi_dist is not None and vol_regime is not None:
            features["momentum_x_vol_regime"] = rsi_dist * vol_regime

        # Trends are stronger during session overlap
        ema_dist = features.get("ema_distance")
        sess_overlap = features.get("session_overlap")
        if ema_dist is not None and sess_overlap is not None:
            features["trend_x_session"] = ema_dist * sess_overlap

        # --- Lagged features (1-bar and 3-bar) ---
        lag_cols = ["macd_hist_accel", "rsi_roc", "momentum_consistency"]
        for col in lag_cols:
            if col in features.columns:
                features[f"{col}_lag1"] = features[col].shift(1).fillna(0)
                features[f"{col}_lag3"] = features[col].shift(3).fillna(0)

        return features

    def _map_m15_trend_to_m1(
        self,
        m1_df: pd.DataFrame,
        m15_indicators: pd.DataFrame,
    ) -> pd.Series:
        """
        Map the 15-minute trend bias to each 1-minute candle.

        For each M1 candle, find the most recent completed M15 candle
        and use its trend signals.
        """
        # Calculate trend bias per M15 candle
        m15_ema_cross = m15_indicators.get("ema_crossover", pd.Series(0, index=m15_indicators.index))
        m15_macd_cross = m15_indicators.get("macd_crossover", pd.Series(0, index=m15_indicators.index))

        m15_bias = pd.Series(0, index=m15_indicators.index, dtype=int)
        # STRICT: both must agree for training data quality
        bullish = (m15_ema_cross == 1) & (m15_macd_cross == 1)
        bearish = (m15_ema_cross == 0) & (m15_macd_cross == 0)
        m15_bias[bullish] = 1
        m15_bias[bearish] = -1

        # Forward-fill M15 bias to M1 timeframe
        # reindex to M1 timestamps, forward fill
        m1_bias = m15_bias.reindex(m1_df.index, method="ffill")
        m1_bias = m1_bias.fillna(0).astype(int)

        return m1_bias

    def _label_candles(
        self,
        m1_df: pd.DataFrame,
        m1_indicators: pd.DataFrame,
        direction: str,
    ) -> pd.Series:
        """
        Label each candle: did a hypothetical trade hit TP before SL?

        Includes a spread cost penalty to make labels realistic:
        - Buy entries are shifted up by half-spread
        - Sell entries are shifted down by half-spread
        This means the price has to move further to hit TP and less to hit SL,
        matching real trading conditions.
        """
        labels = pd.Series(0, index=m1_df.index, dtype=int)
        atr = m1_indicators.get("atr_value", pd.Series(np.nan, index=m1_df.index))

        close_values = m1_df["close"].values
        high_values = m1_df["high"].values
        low_values = m1_df["low"].values
        atr_values = atr.values
        n = len(m1_df)

        # Approximate spread as % of ATR (typically ~10-20% of ATR for EUR/USD)
        spread_atr_fraction = 0.15

        for i in range(n - self.max_hold_bars):
            atr_val = atr_values[i]
            if np.isnan(atr_val) or atr_val <= 0:
                continue

            spread_cost = atr_val * spread_atr_fraction
            entry = close_values[i]
            sl_dist = atr_val * self.sl_atr_mult
            tp_dist = atr_val * self.tp_atr_mult

            if direction == "buy":
                # Entry shifted up by spread
                effective_entry = entry + spread_cost / 2
                sl_price = effective_entry - sl_dist
                tp_price = effective_entry + tp_dist
            else:
                effective_entry = entry - spread_cost / 2
                sl_price = effective_entry + sl_dist
                tp_price = effective_entry - tp_dist

            for j in range(i + 1, min(i + self.max_hold_bars + 1, n)):
                if direction == "buy":
                    if high_values[j] >= tp_price:
                        labels.iloc[i] = 1
                        break
                    if low_values[j] <= sl_price:
                        break
                else:
                    if low_values[j] <= tp_price:
                        labels.iloc[i] = 1
                        break
                    if high_values[j] >= sl_price:
                        break

        return labels

    @staticmethod
    def _candle_body_ratio(df: pd.DataFrame) -> pd.Series:
        total_range = df["high"] - df["low"]
        body = (df["close"] - df["open"]).abs()
        return (body / total_range.replace(0, np.nan)).fillna(0)

    @staticmethod
    def _upper_wick_ratio(df: pd.DataFrame) -> pd.Series:
        total_range = df["high"] - df["low"]
        upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
        return (upper_wick / total_range.replace(0, np.nan)).fillna(0)

    @staticmethod
    def _lower_wick_ratio(df: pd.DataFrame) -> pd.Series:
        total_range = df["high"] - df["low"]
        lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
        return (lower_wick / total_range.replace(0, np.nan)).fillna(0)
