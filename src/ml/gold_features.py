"""
Gold-Specific Feature Builder — Cross-asset and macro-proxy features for XAU/USD.

Research shows gold is a macro asset driven by:
1. Dollar strength (DXY proxy from EUR/USD, GBP/USD, USD/JPY)
2. Cross-asset momentum and correlation regimes
3. Volume dynamics (tick volume surges predict breakouts)
4. Multi-timeframe momentum (15m, 1h, 4h price changes)
5. Session context (London open reference, daily range position)

These features supplement the standard technical indicators.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("traderbot.ml.gold_features")


def build_gold_features(
    gold_m15: pd.DataFrame,
    eur_m15: pd.DataFrame,
    gbp_m15: pd.DataFrame,
    jpy_m15: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build gold-specific features from cross-asset M15 data.

    All inputs must be M15 DataFrames with DatetimeIndex and OHLCV columns.
    Returns a DataFrame indexed to gold's timestamps with new features.
    """
    # Align all instruments to gold's index
    eur = eur_m15["close"].reindex(gold_m15.index, method="ffill")
    gbp = gbp_m15["close"].reindex(gold_m15.index, method="ffill")
    jpy = jpy_m15["close"].reindex(gold_m15.index, method="ffill")
    gold_close = gold_m15["close"]
    gold_vol = gold_m15["volume"]

    features = pd.DataFrame(index=gold_m15.index)

    # === 1. Dollar Strength Proxy (DXY substitute) ===
    # Dollar strength = average of USD performance vs EUR, GBP, JPY
    # EUR/USD down = dollar up, GBP/USD down = dollar up, USD/JPY up = dollar up
    eur_ret = eur.pct_change()
    gbp_ret = gbp.pct_change()
    jpy_ret = jpy.pct_change()

    # Dollar strength: positive = dollar strengthening
    features["dollar_strength_1bar"] = (-eur_ret - gbp_ret + jpy_ret) / 3

    # Dollar momentum over different windows
    for window in [4, 16, 64]:  # 1h, 4h, 16h
        eur_chg = eur.pct_change(window)
        gbp_chg = gbp.pct_change(window)
        jpy_chg = jpy.pct_change(window)
        features[f"dollar_momentum_{window}bar"] = (-eur_chg - gbp_chg + jpy_chg) / 3

    # === 2. Gold-Dollar Correlation (Regime Detection) ===
    # Rolling correlation between gold returns and dollar strength
    gold_ret = gold_close.pct_change()
    dollar_str = features["dollar_strength_1bar"]

    for window in [16, 64]:  # 4h, 16h
        features[f"gold_dollar_corr_{window}bar"] = (
            gold_ret.rolling(window).corr(dollar_str)
        )

    # === 3. Multi-Timeframe Gold Momentum ===
    # Price change normalized by ATR for comparability
    gold_atr = (gold_m15["high"] - gold_m15["low"]).rolling(14).mean()

    for window in [1, 4, 16, 64]:  # 15m, 1h, 4h, 16h
        raw_change = gold_close - gold_close.shift(window)
        features[f"gold_momentum_{window}bar"] = raw_change / gold_atr

    # Momentum acceleration (is momentum increasing or decreasing?)
    mom_4 = features["gold_momentum_4bar"]
    features["gold_momentum_accel"] = mom_4 - mom_4.shift(4)

    # === 4. Volume Features ===
    # Volume ratio (current vs average)
    vol_ma_20 = gold_vol.rolling(20).mean()
    features["volume_ratio"] = gold_vol / vol_ma_20

    # Volume momentum (is volume increasing?)
    features["volume_momentum"] = gold_vol.rolling(5).mean() / vol_ma_20

    # Volume surge (current bar vs recent)
    features["volume_surge"] = gold_vol / gold_vol.rolling(10).mean()

    # Price-volume divergence (price up but volume dropping = weak move)
    price_dir = np.sign(gold_ret)
    vol_dir = np.sign(gold_vol - gold_vol.shift(1))
    features["price_volume_agree"] = (price_dir * vol_dir).rolling(5).mean()

    # === 5. Range and Level Features ===
    # Daily range position (where is price within today's range)
    daily_high = gold_m15["high"].resample("D").transform("max")
    daily_low = gold_m15["low"].resample("D").transform("min")
    daily_range = daily_high - daily_low
    features["daily_range_position"] = (
        (gold_close - daily_low) / daily_range.replace(0, np.nan)
    ).fillna(0.5)

    # Distance from session open (London open reference)
    # Group by date and take first bar at/after 07:00 UTC
    features["london_open_distance"] = _london_open_distance(gold_m15, gold_atr)

    # Intraday range expansion ratio
    # How much of today's range has formed vs typical
    features["range_expansion"] = daily_range / gold_atr

    # === 6. Cross-Asset Momentum ===
    # When EUR/USD and GBP/USD are both falling, dollar is strong -> gold usually falls
    features["eur_momentum_4bar"] = eur.pct_change(4) / eur.pct_change(4).rolling(50).std()
    features["gbp_momentum_4bar"] = gbp.pct_change(4) / gbp.pct_change(4).rolling(50).std()

    # Gold vs silver proxy: use gold's own mean-reversion tendency
    # Gold price z-score over rolling window (overextended = likely to revert)
    gold_ma_64 = gold_close.rolling(64).mean()
    gold_std_64 = gold_close.rolling(64).std()
    features["gold_zscore_64bar"] = (gold_close - gold_ma_64) / gold_std_64

    # === 7. Volatility Context ===
    # Realized volatility ratio (short vs long term)
    rv_short = gold_ret.rolling(8).std()
    rv_long = gold_ret.rolling(64).std()
    features["rv_ratio"] = rv_short / rv_long

    # High-low range relative to recent average
    bar_range = gold_m15["high"] - gold_m15["low"]
    features["bar_range_ratio"] = bar_range / bar_range.rolling(20).mean()

    logger.info(f"Built {len(features.columns)} gold-specific features")
    return features


def _london_open_distance(gold_m15: pd.DataFrame, gold_atr: pd.Series) -> pd.Series:
    """Distance from London open price, normalized by ATR."""
    result = pd.Series(0.0, index=gold_m15.index)
    london_open_price = None
    current_date = None

    for i, (ts, row) in enumerate(gold_m15.iterrows()):
        if current_date is None or ts.date() != current_date:
            current_date = ts.date()
            london_open_price = None

        if london_open_price is None and ts.hour >= 7:
            london_open_price = row["open"]

        if london_open_price is not None:
            atr = gold_atr.iloc[i] if i < len(gold_atr) and pd.notna(gold_atr.iloc[i]) else 1.0
            if atr > 0:
                result.iloc[i] = (row["close"] - london_open_price) / atr

    return result
