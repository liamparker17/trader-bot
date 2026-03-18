"""
Statistical Features — Mean-reversion and regime detection signals.

These features capture where price sits relative to recent history,
helping the model detect overbought/oversold conditions and whether
the market is trending or mean-reverting.
"""

import pandas as pd
import numpy as np

from src.indicators.registry import BaseIndicator, registry


class StatisticalFeatures(BaseIndicator):
    """
    Statistical features for regime detection.

    Outputs:
      - price_zscore: (close - SMA20) / std20, clipped [-4, 4]
      - price_percentile: Rank of close in last 50 bars (0-1)
      - autocorrelation_1: Lag-1 return autocorrelation (trending vs mean-reverting)
      - close_to_high_ratio: Rolling 10-bar average of where close sits in range
    """

    def __init__(self, sma_period: int = 20, percentile_window: int = 50,
                 autocorr_window: int = 20, range_window: int = 10, **kwargs):
        super().__init__(
            sma_period=sma_period,
            percentile_window=percentile_window,
            autocorr_window=autocorr_window,
            range_window=range_window,
        )
        self.sma_period = sma_period
        self.percentile_window = percentile_window
        self.autocorr_window = autocorr_window
        self.range_window = range_window

    @property
    def name(self) -> str:
        return "statistical"

    @property
    def feature_names(self) -> list[str]:
        return [
            "price_zscore",
            "price_percentile",
            "autocorrelation_1",
            "close_to_high_ratio",
        ]

    @property
    def min_periods(self) -> int:
        return self.percentile_window + 10

    def calculate(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Price z-score: (close - SMA) / std, clipped to [-4, 4]
        sma = close.rolling(window=self.sma_period).mean()
        std = close.rolling(window=self.sma_period).std()
        zscore = ((close - sma) / std.replace(0, np.nan)).fillna(0).clip(-4, 4)

        # Price percentile: rank of close in last N bars (0 to 1)
        percentile = close.rolling(window=self.percentile_window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1],
            raw=False,
        ).fillna(0.5)

        # Lag-1 return autocorrelation (positive = trending, negative = mean-reverting)
        returns = close.pct_change()
        autocorr = returns.rolling(window=self.autocorr_window).apply(
            lambda x: x.autocorr(lag=1) if len(x.dropna()) > 2 else 0,
            raw=False,
        ).fillna(0).clip(-1, 1)

        # Close-to-high ratio: rolling average of where close sits in bar range
        bar_range = high - low
        close_position = ((close - low) / bar_range.replace(0, np.nan)).fillna(0.5)
        close_to_high = close_position.rolling(window=self.range_window).mean().fillna(0.5)

        return {
            "price_zscore": zscore,
            "price_percentile": percentile,
            "autocorrelation_1": autocorr,
            "close_to_high_ratio": close_to_high,
        }


# Register with global registry
registry.register_class(StatisticalFeatures)
