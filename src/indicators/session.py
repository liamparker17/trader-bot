"""
Session Features — Time-based market context indicators.

Forex markets behave differently during different trading sessions.
London/NY overlap (13-16 UTC) has the highest volume and volatility.
These features encode time cyclically so the model can learn session effects.
"""

import pandas as pd
import numpy as np

from src.indicators.registry import BaseIndicator, registry

# Session open times in UTC
LONDON_OPEN_HOUR = 7
NY_OPEN_HOUR = 13
OVERLAP_START = 13
OVERLAP_END = 16


class SessionFeatures(BaseIndicator):
    """
    Trading session and cyclical time features.

    Outputs:
      - minutes_since_london: Normalized distance from London open (0-1)
      - minutes_since_ny: Normalized distance from NY open (0-1)
      - session_overlap: 1.0 during London/NY overlap (13-16 UTC), 0.0 otherwise
      - hour_sin: Sine encoding of hour (captures cyclical nature)
      - hour_cos: Cosine encoding of hour (captures cyclical nature)
    """

    def __init__(self, **kwargs):
        super().__init__()

    @property
    def name(self) -> str:
        return "session"

    @property
    def feature_names(self) -> list[str]:
        return [
            "minutes_since_london",
            "minutes_since_ny",
            "session_overlap",
            "hour_sin",
            "hour_cos",
        ]

    @property
    def min_periods(self) -> int:
        return 1

    def calculate(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        if not isinstance(df.index, pd.DatetimeIndex):
            # Fall back to zeros if no datetime index
            zeros = pd.Series(0.0, index=df.index)
            return {name: zeros.copy() for name in self.feature_names}

        hours = df.index.hour
        minutes = df.index.minute

        total_minutes = hours * 60 + minutes

        # Minutes since London open, normalized to [0, 1] over 24h
        london_open_min = LONDON_OPEN_HOUR * 60
        minutes_since_london = ((total_minutes - london_open_min) % 1440) / 1440.0

        # Minutes since NY open, normalized to [0, 1] over 24h
        ny_open_min = NY_OPEN_HOUR * 60
        minutes_since_ny = ((total_minutes - ny_open_min) % 1440) / 1440.0

        # Session overlap flag (13-16 UTC)
        session_overlap = ((hours >= OVERLAP_START) & (hours < OVERLAP_END)).astype(float)

        # Cyclical hour encoding
        hour_angle = 2 * np.pi * hours / 24.0
        hour_sin = np.sin(hour_angle)
        hour_cos = np.cos(hour_angle)

        return {
            "minutes_since_london": pd.Series(minutes_since_london, index=df.index),
            "minutes_since_ny": pd.Series(minutes_since_ny, index=df.index),
            "session_overlap": pd.Series(session_overlap, index=df.index),
            "hour_sin": pd.Series(hour_sin, index=df.index),
            "hour_cos": pd.Series(hour_cos, index=df.index),
        }


# Register with global registry
registry.register_class(SessionFeatures)
