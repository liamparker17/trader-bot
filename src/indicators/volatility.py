"""
Volatility Indicators — ATR and Bollinger Bands.

ATR (Average True Range):
  Measures how much price typically moves per candle. Higher ATR = more
  volatile market. Used to set dynamic stop-losses (wider when volatile,
  tighter when calm) and to detect when markets are too wild or too dead
  to trade.

Bollinger Bands:
  A channel around price based on standard deviations from a moving average.
  Price near the upper band = stretched high, may pull back. Price near the
  lower band = stretched low, may bounce. When bands squeeze tight, a big
  move is often coming (breakout).
"""

import pandas as pd
import numpy as np

from src.indicators.registry import BaseIndicator, registry


class ATR(BaseIndicator):
    """
    Average True Range indicator.

    Outputs:
      - atr_value: Current ATR value (in price units)
      - atr_ratio: Current ATR / baseline ATR (volatility vs normal)
        >2.0 = extremely volatile, <0.3 = dead market
      - atr_pips: ATR converted to pips (for position sizing)
    """

    def __init__(self, period: int = 14, baseline_period: int = 50, **kwargs):
        super().__init__(period=period, baseline_period=baseline_period)
        self.period = period
        self.baseline_period = baseline_period

    @property
    def name(self) -> str:
        return "atr"

    @property
    def feature_names(self) -> list[str]:
        return ["atr_value", "atr_ratio"]

    @property
    def min_periods(self) -> int:
        return self.baseline_period + 5

    def calculate(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # True Range = max of:
        #   high - low (current bar range)
        #   abs(high - previous close) (gap up)
        #   abs(low - previous close) (gap down)
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR = smoothed average of True Range (Wilder's method)
        atr = true_range.ewm(alpha=1 / self.period, min_periods=self.period, adjust=False).mean()

        # Baseline ATR for ratio calculation
        atr_baseline = true_range.ewm(
            alpha=1 / self.baseline_period,
            min_periods=self.baseline_period,
            adjust=False,
        ).mean()

        # Volatility ratio: how volatile is the market right now vs. normal?
        atr_ratio = atr / atr_baseline.replace(0, np.nan)
        atr_ratio = atr_ratio.fillna(1.0)

        return {
            "atr_value": atr,
            "atr_ratio": atr_ratio,
        }


class BollingerBands(BaseIndicator):
    """
    Bollinger Bands indicator.

    Outputs:
      - bb_upper: Upper band (SMA + std_dev * std)
      - bb_lower: Lower band (SMA - std_dev * std)
      - bb_middle: Middle band (SMA)
      - bb_position: Where price is within the bands (0.0 = at lower, 1.0 = at upper)
      - bb_width: Band width normalized by middle (measures volatility)
      - bb_squeeze: 1 if width is below the squeeze_percentile threshold
    """

    def __init__(self, period: int = 20, std_dev: float = 2.0,
                 squeeze_percentile: int = 20, **kwargs):
        super().__init__(period=period)
        self.period = period
        self.std_dev = std_dev
        self.squeeze_percentile = squeeze_percentile

    @property
    def name(self) -> str:
        return "bollinger"

    @property
    def feature_names(self) -> list[str]:
        return ["bb_upper", "bb_lower", "bb_middle", "bb_position", "bb_width", "bb_squeeze"]

    @property
    def min_periods(self) -> int:
        return self.period + 20  # Extra for squeeze percentile calculation

    def calculate(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        close = df["close"]

        # Middle band = Simple Moving Average
        middle = close.rolling(window=self.period).mean()

        # Standard deviation
        std = close.rolling(window=self.period).std()

        # Upper and lower bands
        upper = middle + (self.std_dev * std)
        lower = middle - (self.std_dev * std)

        # Band position: where price sits within the bands (0.0 to 1.0)
        # 0.0 = at lower band, 0.5 = at middle, 1.0 = at upper band
        band_range = upper - lower
        position = (close - lower) / band_range.replace(0, np.nan)
        position = position.clip(0, 1).fillna(0.5)

        # Band width: normalized by middle band
        width = band_range / middle.replace(0, np.nan)
        width = width.fillna(0)

        # Squeeze detection: width below Nth percentile of recent history
        # A squeeze often precedes a breakout
        squeeze = pd.Series(0, index=df.index, dtype=int)
        if len(width.dropna()) > self.period:
            rolling_percentile = width.rolling(
                window=min(100, len(width))
            ).quantile(self.squeeze_percentile / 100)
            squeeze = (width < rolling_percentile).astype(int)

        return {
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_middle": middle,
            "bb_position": position,
            "bb_width": width,
            "bb_squeeze": squeeze,
        }


class VolatilityRegime(BaseIndicator):
    """
    Volatility regime features — detect expanding/contracting volatility.

    Outputs:
      - atr_expansion_ratio: ATR(5)/ATR(20) — expanding vs contracting
      - bb_squeeze_duration: Consecutive bars in squeeze (normalized 0-1)
      - volatility_regime: Percentile rank of ATR in last 100 bars (0-1)
      - range_to_atr: Current bar range / ATR (wide bar = momentum)
    """

    def __init__(self, atr_fast: int = 5, atr_slow: int = 20,
                 regime_window: int = 100, atr_period: int = 14,
                 bb_period: int = 20, bb_std: float = 2.0, **kwargs):
        super().__init__(
            atr_fast=atr_fast,
            atr_slow=atr_slow,
            regime_window=regime_window,
            atr_period=atr_period,
        )
        self.atr_fast = atr_fast
        self.atr_slow = atr_slow
        self.regime_window = regime_window
        self.atr_period = atr_period
        self.bb_period = bb_period
        self.bb_std = bb_std

    @property
    def name(self) -> str:
        return "volatilityregime"

    @property
    def feature_names(self) -> list[str]:
        return [
            "atr_expansion_ratio",
            "bb_squeeze_duration",
            "volatility_regime",
            "range_to_atr",
        ]

    @property
    def min_periods(self) -> int:
        return self.regime_window + 10

    def calculate(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # True range
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR expansion ratio: fast ATR / slow ATR
        atr_fast = true_range.ewm(alpha=1 / self.atr_fast, min_periods=self.atr_fast, adjust=False).mean()
        atr_slow = true_range.ewm(alpha=1 / self.atr_slow, min_periods=self.atr_slow, adjust=False).mean()
        expansion_ratio = (atr_fast / atr_slow.replace(0, np.nan)).fillna(1.0).clip(0.2, 5.0)

        # BB squeeze duration: consecutive bars where BB width is below 20th percentile
        sma = close.rolling(window=self.bb_period).mean()
        std = close.rolling(window=self.bb_period).std()
        bb_width = (2 * self.bb_std * std) / sma.replace(0, np.nan)
        bb_width = bb_width.fillna(0)
        width_threshold = bb_width.rolling(window=min(100, len(df))).quantile(0.2)
        in_squeeze = (bb_width < width_threshold).astype(int)

        # Count consecutive squeeze bars
        squeeze_values = in_squeeze.values
        duration_values = np.zeros(len(df))
        count = 0
        for i in range(len(df)):
            if squeeze_values[i] == 1:
                count += 1
            else:
                count = 0
            duration_values[i] = count

        # Normalize to 0-1 (cap at 20 bars)
        squeeze_duration = pd.Series(
            np.clip(duration_values / 20.0, 0, 1),
            index=df.index,
        )

        # Volatility regime: percentile rank of ATR in last N bars
        atr_current = true_range.ewm(alpha=1 / self.atr_period, min_periods=self.atr_period, adjust=False).mean()
        vol_regime = atr_current.rolling(window=self.regime_window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1],
            raw=False,
        ).fillna(0.5)

        # Range to ATR: current bar range / ATR
        bar_range = high - low
        range_to_atr = (bar_range / atr_current.replace(0, np.nan)).fillna(1.0).clip(0, 5.0)

        return {
            "atr_expansion_ratio": expansion_ratio,
            "bb_squeeze_duration": squeeze_duration,
            "volatility_regime": vol_regime,
            "range_to_atr": range_to_atr,
        }


# Register with global registry
registry.register_class(ATR)
registry.register_class(BollingerBands)
registry.register_class(VolatilityRegime)
