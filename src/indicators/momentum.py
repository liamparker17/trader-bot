"""
Momentum Indicators — RSI.

RSI (Relative Strength Index):
  Measures how fast and how far price has moved recently. Scale 0–100.
  Above 70–75 means price rallied hard and may reverse down ("overbought").
  Below 25–30 means price dropped hard and may bounce ("oversold").
  Also used to detect divergences — when price makes a new high but RSI
  doesn't, it suggests the move is weakening.
"""

import pandas as pd
import numpy as np

from src.indicators.registry import BaseIndicator, registry


class RSI(BaseIndicator):
    """
    Relative Strength Index indicator.

    Outputs:
      - rsi_value: RSI value (0–100)
      - rsi_overbought: 1 if RSI > overbought threshold
      - rsi_oversold: 1 if RSI < oversold threshold
      - rsi_divergence: Detects bullish/bearish divergence
        (+1 = bullish divergence, -1 = bearish divergence, 0 = none)
    """

    def __init__(self, period: int = 14, overbought: float = 75,
                 oversold: float = 25, divergence_lookback: int = 20, **kwargs):
        super().__init__(period=period)
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self.divergence_lookback = divergence_lookback

    @property
    def name(self) -> str:
        return "rsi"

    @property
    def feature_names(self) -> list[str]:
        return ["rsi_value", "rsi_overbought", "rsi_oversold", "rsi_divergence"]

    @property
    def min_periods(self) -> int:
        return self.period + self.divergence_lookback + 5

    def calculate(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        close = df["close"]

        # Calculate RSI using Wilder's smoothing method
        delta = close.diff()
        gains = delta.clip(lower=0)
        losses = (-delta).clip(lower=0)

        avg_gain = gains.ewm(alpha=1 / self.period, min_periods=self.period, adjust=False).mean()
        avg_loss = losses.ewm(alpha=1 / self.period, min_periods=self.period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)  # Default to neutral if undefined

        overbought_flag = (rsi > self.overbought).astype(int)
        oversold_flag = (rsi < self.oversold).astype(int)

        # Divergence detection
        divergence = self._detect_divergence(close, rsi)

        return {
            "rsi_value": rsi,
            "rsi_overbought": overbought_flag,
            "rsi_oversold": oversold_flag,
            "rsi_divergence": divergence,
        }

    def _detect_divergence(self, price: pd.Series, rsi: pd.Series) -> pd.Series:
        """
        Detect RSI divergence.

        Bullish divergence (+1): Price makes lower low but RSI makes higher low.
        This suggests selling pressure is weakening — potential reversal up.

        Bearish divergence (-1): Price makes higher high but RSI makes lower high.
        This suggests buying pressure is weakening — potential reversal down.

        No divergence (0): Normal conditions.
        """
        divergence = pd.Series(0, index=price.index, dtype=int)
        lookback = self.divergence_lookback

        if len(price) < lookback * 2:
            return divergence

        for i in range(lookback, len(price)):
            window_price = price.iloc[i - lookback:i + 1]
            window_rsi = rsi.iloc[i - lookback:i + 1]

            # Find local lows and highs in the window
            price_min_idx = window_price.idxmin()
            price_max_idx = window_price.idxmax()

            current_price = price.iloc[i]
            current_rsi = rsi.iloc[i]

            # Bullish divergence: price at/near low but RSI higher than at price low
            price_low = window_price.min()
            rsi_at_price_low = rsi.loc[price_min_idx] if price_min_idx in rsi.index else current_rsi

            if (current_price <= price_low * 1.002 and  # Price near or below recent low
                    current_rsi > rsi_at_price_low + 3):  # RSI meaningfully higher
                divergence.iloc[i] = 1

            # Bearish divergence: price at/near high but RSI lower than at price high
            price_high = window_price.max()
            rsi_at_price_high = rsi.loc[price_max_idx] if price_max_idx in rsi.index else current_rsi

            if (current_price >= price_high * 0.998 and  # Price near or above recent high
                    current_rsi < rsi_at_price_high - 3):  # RSI meaningfully lower
                divergence.iloc[i] = -1

        return divergence


class MomentumQuality(BaseIndicator):
    """
    Momentum quality features — measures how strong and consistent momentum is.

    Outputs:
      - macd_hist_accel: Rate of change of MACD histogram (2nd derivative of momentum)
      - rsi_roc: 3-bar RSI rate of change
      - momentum_consistency: Fraction of last 5 bars in same direction
      - rsi_distance_50: Distance from RSI neutral (conviction strength, 0-50 scale)
    """

    def __init__(self, rsi_period: int = 14, consistency_window: int = 5,
                 rsi_roc_period: int = 3, **kwargs):
        super().__init__(
            rsi_period=rsi_period,
            consistency_window=consistency_window,
            rsi_roc_period=rsi_roc_period,
        )
        self.rsi_period = rsi_period
        self.consistency_window = consistency_window
        self.rsi_roc_period = rsi_roc_period

    @property
    def name(self) -> str:
        return "momentumquality"

    @property
    def feature_names(self) -> list[str]:
        return [
            "macd_hist_accel",
            "rsi_roc",
            "momentum_consistency",
            "rsi_distance_50",
        ]

    @property
    def min_periods(self) -> int:
        return 40  # Need enough bars for MACD + derivatives

    def calculate(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        close = df["close"]

        # MACD histogram acceleration (2nd derivative)
        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        hist_accel = histogram.diff().fillna(0)
        # Normalize by ATR-like scale
        price_scale = close.rolling(20).std().replace(0, np.nan)
        macd_hist_accel = (hist_accel / price_scale).fillna(0).clip(-5, 5)

        # RSI rate of change
        delta = close.diff()
        gains = delta.clip(lower=0)
        losses = (-delta).clip(lower=0)
        avg_gain = gains.ewm(alpha=1 / self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()
        avg_loss = losses.ewm(alpha=1 / self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)
        rsi_roc = (rsi.diff(self.rsi_roc_period) / self.rsi_roc_period).fillna(0).clip(-10, 10)

        # Momentum consistency: fraction of last N bars with same close direction
        direction = np.sign(close.diff())
        consistency = direction.rolling(window=self.consistency_window).apply(
            lambda x: abs(x.sum()) / len(x),
            raw=True,
        ).fillna(0)

        # RSI distance from 50 (conviction strength)
        rsi_distance_50 = (rsi - 50).abs().fillna(0) / 50.0  # Normalize to 0-1

        return {
            "macd_hist_accel": macd_hist_accel,
            "rsi_roc": rsi_roc,
            "momentum_consistency": consistency,
            "rsi_distance_50": rsi_distance_50,
        }


# Register with global registry
registry.register_class(RSI)
registry.register_class(MomentumQuality)
