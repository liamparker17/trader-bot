"""
Trend Indicators — EMA and MACD.

EMA (Exponential Moving Average):
  Smooths price to reveal trend direction. Fast EMA (8) reacts quickly,
  slow EMA (21) reacts slowly. When fast > slow = uptrend. Crossovers
  signal trend changes.

MACD (Moving Average Convergence Divergence):
  Tracks momentum by comparing two EMAs. The MACD line is the difference
  between a fast and slow EMA. The signal line is an EMA of the MACD line.
  The histogram (MACD - signal) shows momentum strength.
"""

import pandas as pd
import numpy as np

from src.indicators.registry import BaseIndicator, registry


class EMA(BaseIndicator):
    """
    Exponential Moving Average indicator.

    Outputs:
      - ema_fast: Fast EMA values
      - ema_slow: Slow EMA values
      - ema_crossover: 1 if fast > slow (bullish), 0 otherwise
      - ema_distance: Normalized distance between fast and slow EMA
    """

    def __init__(self, fast_period: int = 8, slow_period: int = 21, **kwargs):
        super().__init__(fast_period=fast_period, slow_period=slow_period)
        self.fast_period = fast_period
        self.slow_period = slow_period

    @property
    def name(self) -> str:
        return "ema"

    @property
    def feature_names(self) -> list[str]:
        return ["ema_fast", "ema_slow", "ema_crossover", "ema_distance"]

    @property
    def min_periods(self) -> int:
        return self.slow_period + 5  # Extra buffer for stable values

    def calculate(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        close = df["close"]

        ema_fast = close.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow_period, adjust=False).mean()

        # Crossover: 1 = bullish (fast above slow), 0 = bearish
        crossover = (ema_fast > ema_slow).astype(int)

        # Normalized distance: (fast - slow) / slow
        # Positive = bullish momentum, negative = bearish
        distance = (ema_fast - ema_slow) / ema_slow

        return {
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "ema_crossover": crossover,
            "ema_distance": distance,
        }


class MACD(BaseIndicator):
    """
    Moving Average Convergence Divergence indicator.

    Outputs:
      - macd_value: MACD line (fast EMA - slow EMA)
      - macd_signal: Signal line (EMA of MACD line)
      - macd_histogram: MACD - Signal (momentum strength)
      - macd_crossover: 1 if MACD > Signal (bullish), 0 otherwise
    """

    def __init__(self, fast_period: int = 12, slow_period: int = 26,
                 signal_period: int = 9, **kwargs):
        super().__init__(
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    @property
    def name(self) -> str:
        return "macd"

    @property
    def feature_names(self) -> list[str]:
        return ["macd_value", "macd_signal", "macd_histogram", "macd_crossover"]

    @property
    def min_periods(self) -> int:
        return self.slow_period + self.signal_period + 5

    def calculate(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        close = df["close"]

        ema_fast = close.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow_period, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        # Crossover: 1 if MACD above signal (bullish momentum)
        crossover = (macd_line > signal_line).astype(int)

        return {
            "macd_value": macd_line,
            "macd_signal": signal_line,
            "macd_histogram": histogram,
            "macd_crossover": crossover,
        }


# Register with global registry
registry.register_class(EMA)
registry.register_class(MACD)
