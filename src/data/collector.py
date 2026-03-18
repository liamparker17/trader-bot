"""
Data Collector — Orchestrates data flow from MT5 to the trading system.

Ties together:
- MT5Client (API communication)
- HistoricalLoader (bulk data fetching + caching)
- CandleBuilder (live tick → candle conversion)
"""

import logging
import threading
from typing import Callable, Optional

import pandas as pd

from src.config import Config
from src.data.mt5_client import MT5Client
from src.data.candle_builder import CandleBuilder
from src.data.historical_loader import HistoricalLoader

logger = logging.getLogger("traderbot.collector")


class DataCollector:
    """
    Central data orchestrator.

    Responsibilities:
    1. Fetch and cache historical data for backtesting/ML training
    2. Stream live prices and build candles in real-time
    3. Provide candle data to indicator engine and ML model
    """

    def __init__(
        self,
        config: Config,
        client: MT5Client,
        on_candle_complete: Optional[Callable] = None,
    ):
        self.config = config
        self.client = client
        self.historical = HistoricalLoader(config, client)
        self.candle_builder = CandleBuilder(
            on_candle_complete=on_candle_complete,
            max_buffer_size=500,
        )

        self._stream_thread: Optional[threading.Thread] = None
        self._streaming = False
        self._error_count = 0
        self._max_errors = 5
        self._error_window_seconds = 600  # 10 minutes
        self._error_timestamps: list[float] = []

    def load_historical_data(self, granularity: str = "M1") -> dict[str, pd.DataFrame]:
        """
        Fetch and cache historical data for all enabled instruments.

        Returns dict mapping instrument name to DataFrame.
        """
        return self.historical.fetch_all_instruments(granularity)

    def warm_up_candle_builder(self, candle_count: int = 200):
        """
        Pre-load recent historical candles into the candle builder
        so indicators have data to work with at startup.
        """
        instruments = self.config.get_enabled_instruments()

        for instrument in instruments:
            for timeframe in ["M1", "M15"]:
                df = self.historical.load_cached(instrument, timeframe)

                if df.empty:
                    # Try to fetch if not cached
                    granularity = timeframe
                    df = self.historical.fetch_and_cache(instrument, granularity, months=1)

                if not df.empty:
                    recent = df.tail(candle_count)
                    self.candle_builder.load_historical(instrument, timeframe, recent)
                else:
                    logger.warning(
                        f"No historical data to warm up {instrument} {timeframe}"
                    )

    def start_streaming(self):
        """Start streaming live prices in a background thread."""
        if self._streaming:
            logger.warning("Price stream already running")
            return

        instruments = self.config.get_enabled_instruments()
        if not instruments:
            logger.error("No instruments enabled — cannot start streaming")
            return

        self._streaming = True
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            args=(instruments,),
            daemon=True,
            name="price-stream",
        )
        self._stream_thread.start()
        logger.info(f"Price stream started for {instruments}")

    def stop_streaming(self):
        """Stop the price stream."""
        self._streaming = False
        if self._stream_thread and self._stream_thread.is_alive():
            logger.info("Stopping price stream...")
            # The stream loop will exit on next iteration when _streaming is False

    def get_candles_df(self, instrument: str, timeframe: str,
                       count: Optional[int] = None) -> pd.DataFrame:
        """Get candle data as DataFrame (from candle builder buffer)."""
        return self.candle_builder.get_candles_df(instrument, timeframe, count)

    def _stream_loop(self, instruments: list[str]):
        """Background thread that consumes the price stream."""
        import time

        logger.info("Stream loop started")

        while self._streaming:
            try:
                for tick in self.client.stream_prices(instruments):
                    if not self._streaming:
                        break

                    if tick.get("type") == "PRICE":
                        self.candle_builder.on_tick(tick)
                    # HEARTBEAT ticks are silently consumed (connection keepalive)

            except Exception as e:
                if not self._streaming:
                    break

                self._error_count += 1
                now = time.time()
                self._error_timestamps.append(now)

                # Clean old errors outside window
                cutoff = now - self._error_window_seconds
                self._error_timestamps = [
                    t for t in self._error_timestamps if t > cutoff
                ]

                if len(self._error_timestamps) >= self._max_errors:
                    logger.critical(
                        f"Too many stream errors ({len(self._error_timestamps)} "
                        f"in {self._error_window_seconds}s). Stopping stream. "
                        "Trading should be paused."
                    )
                    self._streaming = False
                    break

                logger.error(f"Stream error: {e}. Will retry via client reconnect logic.")

        logger.info("Stream loop exited")
