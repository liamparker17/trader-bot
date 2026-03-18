"""
Historical Data Loader — Fetches and stores bulk historical candle data.

Downloads months of 1-minute candle data from MT5, processes it,
and stores it locally in Parquet format for backtesting and ML training.
"""

import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from src.config import Config, DATA_DIR
from src.data.mt5_client import MT5Client

logger = logging.getLogger("traderbot.historical")


class HistoricalLoader:
    """
    Loads historical candle data from MT5 and caches it locally.

    Handles:
    - Bulk downloading via MT5 copy_rates API
    - Data validation and cleaning
    - Parquet file storage for fast repeated access
    - Incremental updates (only fetch new data)
    """

    def __init__(self, config: Config, client: MT5Client):
        self.config = config
        self.client = client
        self.cache_dir = DATA_DIR / "historical"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_and_cache(
        self,
        instrument: str,
        granularity: str = "M1",
        months: int = None,
    ) -> pd.DataFrame:
        """
        Fetch historical data and save to Parquet cache.

        If cached data exists, only fetches new data since the last cached candle.

        Args:
            instrument: e.g., "EUR_USD"
            granularity: e.g., "M1", "M15"
            months: How many months of history (default: from config)

        Returns:
            DataFrame with full historical data.
        """
        if months is None:
            months = self.config.get("data.history_months", 12)

        cache_path = self._cache_path(instrument, granularity)

        # Check for existing cached data
        existing_df = self._load_cache(cache_path)
        if existing_df is not None and not existing_df.empty:
            # Fetch only new data since last cached candle
            last_time = existing_df.index.max()
            from_time = (last_time + timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
            to_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

            logger.info(
                f"Cache exists for {instrument} {granularity} "
                f"({len(existing_df)} candles). Fetching updates from {from_time}..."
            )

            new_candles = self.client.get_candles_batch(
                instrument=instrument,
                granularity=granularity,
                from_time=from_time,
                to_time=to_time,
            )

            if new_candles:
                new_df = self._candles_to_dataframe(new_candles)
                combined_df = pd.concat([existing_df, new_df])
                combined_df = combined_df[~combined_df.index.duplicated(keep="last")]
                combined_df.sort_index(inplace=True)
                self._save_cache(combined_df, cache_path)
                logger.info(f"Updated cache: {len(combined_df)} total candles")
                return combined_df
            else:
                logger.info("Cache is already up to date.")
                return existing_df

        # No cache — fetch full history
        to_time = datetime.now(timezone.utc)
        from_time = to_time - timedelta(days=months * 30)

        logger.info(
            f"Fetching {months} months of {granularity} data for {instrument} "
            f"({from_time.date()} to {to_time.date()})..."
        )

        candles = self.client.get_candles_batch(
            instrument=instrument,
            granularity=granularity,
            from_time=from_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            to_time=to_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

        if not candles:
            logger.warning(f"No candle data returned for {instrument}")
            return pd.DataFrame()

        df = self._candles_to_dataframe(candles)
        df = self._validate_and_clean(df, instrument, granularity)
        self._save_cache(df, cache_path)

        logger.info(f"Cached {len(df)} {granularity} candles for {instrument}")
        return df

    def load_cached(self, instrument: str, granularity: str = "M1") -> pd.DataFrame:
        """Load cached data without fetching. Returns empty DataFrame if no cache."""
        cache_path = self._cache_path(instrument, granularity)
        df = self._load_cache(cache_path)
        if df is None:
            return pd.DataFrame()
        return df

    def fetch_all_instruments(self, granularity: str = "M1") -> dict[str, pd.DataFrame]:
        """Fetch historical data for all enabled instruments."""
        instruments = self.config.get_enabled_instruments()
        results = {}

        for instrument in instruments:
            logger.info(f"--- Fetching {instrument} ---")
            results[instrument] = self.fetch_and_cache(instrument, granularity)

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _candles_to_dataframe(self, candles: list[dict]) -> pd.DataFrame:
        """Convert MT5 candle dicts to a clean DataFrame.

        MT5Client already returns OANDA-compatible dicts with:
        {time, mid: {o, h, l, c}, volume, complete}
        """
        rows = []
        for c in candles:
            if not c.get("complete", True):
                continue

            mid = c.get("mid", {})
            if not mid:
                continue

            rows.append({
                "timestamp": pd.Timestamp(c["time"]),
                "open": float(mid["o"]),
                "high": float(mid["h"]),
                "low": float(mid["l"]),
                "close": float(mid["c"]),
                "volume": int(c.get("volume", 0)),
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df.set_index("timestamp", inplace=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df.sort_index(inplace=True)
        return df

    def _validate_and_clean(self, df: pd.DataFrame, instrument: str,
                            granularity: str) -> pd.DataFrame:
        """
        Validate and clean candle data.

        Checks for:
        - Impossible OHLC values (high < low)
        - Zero volume candles
        - Duplicate timestamps
        - Large gaps (logged as warnings)
        """
        original_len = len(df)

        # Remove duplicates
        df = df[~df.index.duplicated(keep="last")]

        # Remove impossible candles
        valid_ohlc = (df["high"] >= df["low"]) & (df["high"] >= df["open"]) & \
                     (df["high"] >= df["close"]) & (df["low"] <= df["open"]) & \
                     (df["low"] <= df["close"])
        df = df[valid_ohlc]

        # Remove zero-price candles
        df = df[(df["open"] > 0) & (df["close"] > 0)]

        removed = original_len - len(df)
        if removed > 0:
            logger.warning(
                f"Removed {removed} invalid candles from {instrument} {granularity}"
            )

        # Detect and log gaps
        if granularity == "M1" and len(df) > 1:
            max_gap = self.config.get("data.max_candle_gap_minutes", 5)
            time_diffs = df.index.to_series().diff()
            large_gaps = time_diffs[time_diffs > timedelta(minutes=max_gap)]

            if not large_gaps.empty:
                logger.info(
                    f"Found {len(large_gaps)} gaps > {max_gap} min in "
                    f"{instrument} data (expected on weekends/holidays)"
                )

        return df

    def _cache_path(self, instrument: str, granularity: str) -> Path:
        """Generate cache file path for an instrument/granularity combo."""
        return self.cache_dir / f"{instrument}_{granularity}.parquet"

    def _load_cache(self, path: Path) -> pd.DataFrame | None:
        """Load a Parquet cache file. Returns None if not found."""
        if path.exists():
            try:
                df = pd.read_parquet(path)
                logger.info(f"Loaded cache: {path.name} ({len(df)} candles)")
                return df
            except Exception as e:
                logger.warning(f"Failed to read cache {path}: {e}")
                return None
        return None

    def _save_cache(self, df: pd.DataFrame, path: Path):
        """Save a DataFrame to Parquet cache."""
        df.to_parquet(path, engine="pyarrow")
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved cache: {path.name} ({len(df)} candles, {size_mb:.1f} MB)")
