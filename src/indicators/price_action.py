"""
Price Action Indicators — Candlestick pattern detection.

These features detect specific candlestick patterns that signal
potential reversals or continuations. Uses continuous scores rather
than binary flags to give the ML model more nuanced information.
"""

import pandas as pd
import numpy as np

from src.indicators.registry import BaseIndicator, registry


class PriceAction(BaseIndicator):
    """
    Price action pattern features.

    Outputs:
      - pin_bar_score: Rejection wick strength [-1, 1] (positive = bullish rejection)
      - engulfing_score: Engulfing pattern strength [-1, 1]
      - inside_bar_tightness: Current range / previous range (compression detection)
      - consecutive_direction: Signed count of same-direction closes (exhaustion signal)
    """

    def __init__(self, max_consecutive: int = 10, **kwargs):
        super().__init__(max_consecutive=max_consecutive)
        self.max_consecutive = max_consecutive

    @property
    def name(self) -> str:
        return "priceaction"

    @property
    def feature_names(self) -> list[str]:
        return [
            "pin_bar_score",
            "engulfing_score",
            "inside_bar_tightness",
            "consecutive_direction",
        ]

    @property
    def min_periods(self) -> int:
        return 5

    def calculate(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        o = df["open"]
        h = df["high"]
        l = df["low"]
        c = df["close"]
        total_range = (h - l).replace(0, np.nan)
        body = (c - o).abs()

        # Pin bar score: measures rejection wick strength
        # Bullish pin bar: long lower wick, small body near top → positive
        # Bearish pin bar: long upper wick, small body near bottom → negative
        upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
        lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l

        bullish_pin = (lower_wick / total_range).fillna(0)  # Long lower wick
        bearish_pin = (upper_wick / total_range).fillna(0)  # Long upper wick
        pin_bar_score = (bullish_pin - bearish_pin).clip(-1, 1)

        # Engulfing score: current body engulfs previous body
        prev_body = body.shift(1)
        prev_direction = (c.shift(1) > o.shift(1)).astype(int)  # 1=bullish, 0=bearish
        curr_direction = (c > o).astype(int)

        # Bullish engulfing: prev bearish, current bullish, current body > prev body
        bullish_engulf = (
            (prev_direction == 0) & (curr_direction == 1) &
            (body > prev_body)
        ).astype(float)

        # Bearish engulfing: prev bullish, current bearish, current body > prev body
        bearish_engulf = (
            (prev_direction == 1) & (curr_direction == 0) &
            (body > prev_body)
        ).astype(float)

        # Scale by how much bigger the engulfing body is
        size_ratio = (body / prev_body.replace(0, np.nan)).fillna(1).clip(0, 3) / 3
        engulfing_score = ((bullish_engulf - bearish_engulf) * size_ratio).clip(-1, 1)

        # Inside bar tightness: current range / previous range
        prev_range = total_range.shift(1)
        inside_bar = (total_range / prev_range.replace(0, np.nan)).fillna(1).clip(0, 2)

        # Consecutive direction: signed count of same-direction closes
        direction = np.sign(c - o)
        consecutive = pd.Series(0.0, index=df.index)
        count = 0.0
        prev_dir = 0.0

        direction_values = direction.values
        consec_values = np.zeros(len(df))

        for i in range(len(df)):
            d = direction_values[i]
            if d == prev_dir and d != 0:
                count += d
            else:
                count = d
            prev_dir = d
            consec_values[i] = count

        consecutive = pd.Series(
            np.clip(consec_values / self.max_consecutive, -1, 1),
            index=df.index,
        )

        return {
            "pin_bar_score": pin_bar_score,
            "engulfing_score": engulfing_score,
            "inside_bar_tightness": inside_bar,
            "consecutive_direction": consecutive,
        }


# Register with global registry
registry.register_class(PriceAction)
