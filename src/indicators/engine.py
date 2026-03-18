"""
Indicator Engine — Orchestrates all indicator calculations.

Runs registered indicators on candle data for both timeframes (M1, M15)
and produces a unified feature dictionary for the ML model.

Also computes derived price-action features (candle body ratio, wick ratios)
and context features (hour of day, day of week, spread).
"""

import logging
from datetime import datetime, timezone

import pandas as pd
import numpy as np

from src.config import Config

# Import indicator modules to trigger registration
from src.indicators import trend, momentum, volatility, statistical, price_action, session
from src.indicators.registry import registry, BaseIndicator

logger = logging.getLogger("traderbot.indicators")


class IndicatorEngine:
    """
    Central indicator orchestrator.

    Responsibilities:
    1. Initialize all indicators from config
    2. Calculate indicators on M1 and M15 candle data
    3. Determine 15-minute trend bias
    4. Produce a complete feature vector for ML prediction
    """

    def __init__(self, config: Config):
        self.config = config
        self._setup_indicators()

    def _setup_indicators(self):
        """Instantiate all indicators from config."""
        ind_config = self.config.get("indicators", {})

        # EMA
        ema_cfg = ind_config.get("ema", {})
        registry.instantiate(
            "ema",
            fast_period=ema_cfg.get("fast_period", 8),
            slow_period=ema_cfg.get("slow_period", 21),
        )

        # RSI
        rsi_cfg = ind_config.get("rsi", {})
        registry.instantiate(
            "rsi",
            period=rsi_cfg.get("period", 14),
            overbought=rsi_cfg.get("overbought", 75),
            oversold=rsi_cfg.get("oversold", 25),
        )

        # ATR
        atr_cfg = ind_config.get("atr", {})
        registry.instantiate(
            "atr",
            period=atr_cfg.get("period", 14),
            baseline_period=atr_cfg.get("baseline_period", 50),
        )

        # MACD
        macd_cfg = ind_config.get("macd", {})
        registry.instantiate(
            "macd",
            fast_period=macd_cfg.get("fast_period", 12),
            slow_period=macd_cfg.get("slow_period", 26),
            signal_period=macd_cfg.get("signal_period", 9),
        )

        # Bollinger Bands
        bb_cfg = ind_config.get("bollinger", {})
        registry.instantiate(
            "bollingerbands",
            period=bb_cfg.get("period", 20),
            std_dev=bb_cfg.get("std_dev", 2.0),
            squeeze_percentile=bb_cfg.get("squeeze_percentile", 20),
        )

        # Momentum Quality
        registry.instantiate("momentumquality")

        # Volatility Regime
        registry.instantiate("volatilityregime")

        # Statistical Features
        registry.instantiate("statisticalfeatures")

        # Price Action
        registry.instantiate("priceaction")

        # Session Features
        registry.instantiate("sessionfeatures")

        logger.info(f"Active indicators: {registry.list_active()}")
        logger.info(f"Total features: {len(registry.get_all_feature_names())}")

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run all active indicators on a candle DataFrame.

        Args:
            df: DataFrame with columns [open, high, low, close, volume]

        Returns:
            Original DataFrame with indicator columns appended.
        """
        result = df.copy()

        for name, indicator in registry.get_active().items():
            try:
                features = indicator.calculate(df)
                for feature_name, series in features.items():
                    result[feature_name] = series
            except Exception as e:
                logger.error(f"Error calculating {name}: {e}", exc_info=True)
                # Fill with NaN so downstream can handle gracefully
                for feature_name in indicator.feature_names:
                    result[feature_name] = np.nan

        return result

    def get_trend_bias(self, m15_df: pd.DataFrame) -> int:
        """
        Determine the 15-minute trend bias.

        Returns:
            +1 = bullish (EMA fast > slow AND MACD bullish)
            -1 = bearish (EMA fast < slow AND MACD bearish)
             0 = neutral (conflicting signals)
        """
        if m15_df.empty or len(m15_df) < 30:
            return 0

        m15_with_indicators = self.calculate_all(m15_df)
        latest = m15_with_indicators.iloc[-1]

        ema_bullish = latest.get("ema_crossover", 0) == 1
        macd_bullish = latest.get("macd_crossover", 0) == 1

        if ema_bullish and macd_bullish:
            return 1
        elif not ema_bullish and not macd_bullish:
            return -1
        else:
            return 0

    def build_feature_vector(
        self,
        m1_df: pd.DataFrame,
        m15_df: pd.DataFrame,
        current_spread: float = 0.0,
    ) -> dict | None:
        """
        Build a complete feature vector for ML prediction.

        Combines:
        1. All indicator values from the latest M1 candle
        2. 15-minute trend bias
        3. Price action features (candle shape)
        4. Context features (time of day, spread)

        Args:
            m1_df: 1-minute candle DataFrame (needs 50+ candles)
            m15_df: 15-minute candle DataFrame (needs 30+ candles)
            current_spread: Current bid-ask spread in price units

        Returns:
            Dict of feature_name → value, or None if insufficient data.
        """
        if m1_df.empty or len(m1_df) < 120:
            logger.warning("Insufficient M1 data for feature vector")
            return None

        if m15_df.empty or len(m15_df) < 30:
            logger.warning("Insufficient M15 data for feature vector")
            return None

        # Calculate indicators on M1 data
        m1_with_indicators = self.calculate_all(m1_df)
        latest = m1_with_indicators.iloc[-1]

        features = {}

        # 1. All indicator features from M1
        for feature_name in registry.get_all_feature_names():
            value = latest.get(feature_name, np.nan)
            features[feature_name] = float(value) if pd.notna(value) else 0.0

        # 2. 15-minute trend bias
        features["trend_15min"] = float(self.get_trend_bias(m15_df))

        # 3. Price action features from latest M1 candle
        price_features = self._price_action_features(latest)
        features.update(price_features)

        # 4. Context features
        context_features = self._context_features(m1_df.index[-1], current_spread)
        features.update(context_features)

        # 5. Interaction features
        rsi_dist = features.get("rsi_distance_50", 0)
        vol_regime = features.get("volatility_regime", 0.5)
        features["momentum_x_vol_regime"] = rsi_dist * vol_regime

        ema_dist = features.get("ema_distance", 0)
        sess_overlap = features.get("session_overlap", 0)
        features["trend_x_session"] = ema_dist * sess_overlap

        # 6. Lagged features (from recent M1 candles)
        lag_cols = ["macd_hist_accel", "rsi_roc", "momentum_consistency"]
        if len(m1_with_indicators) >= 4:
            prev1 = m1_with_indicators.iloc[-2]
            prev3 = m1_with_indicators.iloc[-4]
            for col in lag_cols:
                val1 = prev1.get(col, 0)
                val3 = prev3.get(col, 0)
                features[f"{col}_lag1"] = float(val1) if pd.notna(val1) else 0.0
                features[f"{col}_lag3"] = float(val3) if pd.notna(val3) else 0.0
        else:
            for col in lag_cols:
                features[f"{col}_lag1"] = 0.0
                features[f"{col}_lag3"] = 0.0

        return features

    def _price_action_features(self, candle: pd.Series) -> dict:
        """
        Extract price action features from a single candle.

        These describe the shape of the candle:
        - Body ratio: how much of the candle is body vs wicks
        - Upper/lower wick ratios: rejection signals
        - Bullish/bearish flag
        """
        o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
        total_range = h - l

        if total_range == 0:
            return {
                "candle_body_ratio": 0.0,
                "upper_wick_ratio": 0.0,
                "lower_wick_ratio": 0.0,
                "close_vs_open": 0,
            }

        body = abs(c - o)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        return {
            "candle_body_ratio": body / total_range,
            "upper_wick_ratio": upper_wick / total_range,
            "lower_wick_ratio": lower_wick / total_range,
            "close_vs_open": 1 if c > o else 0,
        }

    def _context_features(self, timestamp, spread: float) -> dict:
        """
        Context features: time of day, day of week, current spread.

        Time features capture session effects:
        - London open (7–8 UTC) tends to be volatile
        - NY open (13–14 UTC) tends to be volatile
        - Asian session (0–7 UTC) tends to be quieter

        Day of week captures patterns:
        - Monday often ranges (post-weekend gap risk)
        - Friday often sees profit-taking
        """
        if isinstance(timestamp, pd.Timestamp):
            dt = timestamp.to_pydatetime()
        else:
            dt = timestamp

        if hasattr(dt, 'hour'):
            hour = dt.hour
            weekday = dt.weekday()
        else:
            hour = 12
            weekday = 2

        return {
            "hour_of_day": float(hour),
            "day_of_week": float(weekday),
            "spread_current": float(spread),
        }

    def is_pullback_entry(self, m1_df: pd.DataFrame, direction: str) -> bool:
        """
        Detect pullback entry: price pulled back to 21 EMA zone then resumed trend.

        Proven scalping strategy:
        1. Price was recently near the 21 EMA (within last 5 bars)
        2. Current candle closes in trend direction
        3. RSI confirms momentum resuming
        """
        if len(m1_df) < 10:
            return False

        m1_with_ind = self.calculate_all(m1_df)
        current = m1_with_ind.iloc[-1]
        prev = m1_with_ind.iloc[-2]

        # Current candle must close in trend direction
        if direction == "buy" and current["close"] <= current["open"]:
            return False
        if direction == "sell" and current["close"] >= current["open"]:
            return False

        ema_slow = current.get("ema_slow")
        if pd.isna(ema_slow) or ema_slow <= 0:
            return False

        # Check if price touched/crossed the 21 EMA in the last 5 bars
        touched_ema = False
        for j in range(-6, -1):
            if abs(j) > len(m1_with_ind):
                continue
            bar = m1_with_ind.iloc[j]
            bar_ema = bar.get("ema_slow")
            if pd.isna(bar_ema):
                continue
            atr = bar.get("atr_value", 0)
            if pd.isna(atr) or atr <= 0:
                continue
            ema_zone = atr * 0.5
            if direction == "buy":
                if bar["low"] <= bar_ema + ema_zone:
                    touched_ema = True
                    break
            else:
                if bar["high"] >= bar_ema - ema_zone:
                    touched_ema = True
                    break

        if not touched_ema:
            return False

        # RSI should confirm momentum resuming
        rsi = current.get("rsi_value", 50)
        prev_rsi = prev.get("rsi_value", 50)
        if pd.isna(rsi) or pd.isna(prev_rsi):
            return False

        if direction == "buy":
            if rsi < 40 or rsi < prev_rsi:
                return False
        else:
            if rsi > 60 or rsi > prev_rsi:
                return False

        return True

    def calculate_all_with_extras(self, df):
        """Calculate all indicators plus additional EMAs for per-instrument strategies."""
        result = self.calculate_all(df)
        # EMA(55) for triple-EMA stacking strategies (EUR/USD, XAU/USD)
        result["ema_55"] = df["close"].ewm(span=55, adjust=False).mean()
        return result

    def get_session_range(self, m1_df, current_time, start_hour, end_hour):
        """Get the high/low of a session range for the current day."""
        today = current_time.date()
        try:
            session_mask = (
                (m1_df.index.date == today) &
                (m1_df.index.hour >= start_hour) &
                (m1_df.index.hour < end_hour)
            )
            session_candles = m1_df[session_mask]
            if len(session_candles) < 10:
                return None, None
            return session_candles["high"].max(), session_candles["low"].min()
        except Exception:
            return None, None

    def get_required_candle_count(self) -> int:
        """Minimum number of candles needed for all indicators to produce valid output."""
        max_min_periods = 0
        for indicator in registry.get_active().values():
            max_min_periods = max(max_min_periods, indicator.min_periods)
        return max_min_periods + 10  # Extra safety buffer

    def get_feature_names(self) -> list[str]:
        """Get complete list of all feature names (indicators + price action + context + interactions + lags)."""
        names = registry.get_all_feature_names()
        names.append("trend_15min")
        names.extend(["candle_body_ratio", "upper_wick_ratio", "lower_wick_ratio", "close_vs_open"])
        names.extend(["hour_of_day", "day_of_week", "spread_current"])
        # Interaction features
        names.extend(["momentum_x_vol_regime", "trend_x_session"])
        # Lagged features
        for col in ["macd_hist_accel", "rsi_roc", "momentum_consistency"]:
            names.extend([f"{col}_lag1", f"{col}_lag3"])
        return names
