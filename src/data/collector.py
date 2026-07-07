"""
Data Collector — Orchestrates data flow from MT5 to the trading system.

Ties together:
- MT5Client (API communication)
- HistoricalLoader (bulk data fetching + caching)
- CandleBuilder (live tick → candle conversion)
"""

import logging
import threading
import time
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
        telegram=None,
    ):
        self.config = config
        self.client = client
        self.telegram = telegram
        self.historical = HistoricalLoader(config, client)
        self.candle_builder = CandleBuilder(
            on_candle_complete=on_candle_complete,
            max_buffer_size=500,
        )

        self._stream_thread: Optional[threading.Thread] = None
        self._health_thread: Optional[threading.Thread] = None
        self._streaming = False
        self._error_count = 0
        self._max_errors = 5
        self._error_window_seconds = 600  # 10 minutes
        self._error_timestamps: list[float] = []

        # Broker connectivity state (Blockers B1/B2/B3 + I).
        # Main loop / executor should check `broker_down` before approving
        # new-entry signals; it is not enforced here since main.py wiring
        # is out of scope for this change.
        self.broker_down = False
        self._health_check_interval_seconds = 5

        # Flap debounce: require this many CONSECUTIVE agreeing health
        # checks before flipping `broker_down` (and firing its alert) in
        # either direction. Guards against a flapping connection firing
        # disconnected/reconnected alerts back-to-back on single blips.
        self._flap_debounce_count = 2
        self._consecutive_failures = 0
        self._consecutive_successes = 0

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

        self._health_thread = threading.Thread(
            target=self._health_loop,
            daemon=True,
            name="mt5-health-check",
        )
        self._health_thread.start()

        logger.info(f"Price stream started for {instruments}")

    def stop_streaming(self):
        """Stop the price stream."""
        self._streaming = False
        # Interrupt an in-progress stream_prices() generator immediately,
        # including mid-backoff-sleep, instead of waiting out its chunked
        # sleep loop naturally (see MT5Client.cancel_stream()).
        self.client.cancel_stream()
        if self._stream_thread and self._stream_thread.is_alive():
            logger.info("Stopping price stream...")
            # The stream loop will exit on next iteration when _streaming is False

    def get_candles_df(self, instrument: str, timeframe: str,
                       count: Optional[int] = None) -> pd.DataFrame:
        """Get candle data as DataFrame (from candle builder buffer)."""
        return self.candle_builder.get_candles_df(instrument, timeframe, count)

    def check_connection(self) -> bool:
        """
        Check MT5 broker connectivity and manage disconnect/reconnect state.

        On transition connected -> disconnected: sets `self.broker_down = True`
        (main loop / executor should check this before approving new-entry
        signals) and fires the `mt5.disconnected` Telegram alert.

        On transition disconnected -> connected: re-detects the instrument
        symbol suffix mapping, sets `self.broker_down = False`, and fires
        the `mt5.reconnected` Telegram alert.

        Flap debounce: a state flip only fires after `_flap_debounce_count`
        (2) CONSECUTIVE checks agree in that direction, so a single flaky
        probe result can't flap `broker_down`/alerts back and forth.

        IMPORTANT — single-writer: this method mutates `broker_down` and the
        consecutive-failure/success counters without a lock. It must only
        ever be called from the health-check thread (`_health_loop`); calling
        it concurrently from another thread is not safe.

        Returns the current connected state. Never raises.
        """
        try:
            connected = self.client.is_broker_connected()
        except Exception as e:
            logger.warning(f"Broker connectivity check failed: {e}")
            connected = False

        if connected:
            self._consecutive_successes += 1
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1
            self._consecutive_successes = 0

        if (
            not connected
            and not self.broker_down
            and self._consecutive_failures >= self._flap_debounce_count
        ):
            self.broker_down = True
            logger.critical(
                "MT5 broker connection lost. Pausing new-entry signals."
            )
            if self.telegram:
                try:
                    self.telegram.mt5_disconnected()
                except Exception as e:
                    logger.warning(f"Telegram disconnect alert failed: {e}")

        elif (
            connected
            and self.broker_down
            and self._consecutive_successes >= self._flap_debounce_count
        ):
            self._resync_symbols()
            self.broker_down = False
            logger.info("MT5 broker connection restored. Resuming.")
            if self.telegram:
                try:
                    self.telegram.mt5_reconnected()
                except Exception as e:
                    logger.warning(f"Telegram reconnect alert failed: {e}")

        return connected

    def _resync_symbols(self):
        """
        Re-detect the instrument -> MT5 symbol suffix mapping after a
        reconnect (broker terminal state, e.g. Market Watch selections,
        can reset on reconnect).

        MT5Client caches the detected suffix per instrument (see
        MT5Client._symbol_cache), so a plain _to_mt5_symbol() call would
        just return the stale cached value. invalidate_symbol_cache()
        clears it first so the loop below forces a real re-probe of the
        broker for every enabled instrument.
        """
        try:
            self.client.invalidate_symbol_cache()
        except Exception as e:
            logger.warning(f"Symbol cache invalidation failed: {e}")

        instruments = self.config.get_enabled_instruments()
        for instrument in instruments:
            try:
                self.client._to_mt5_symbol(instrument)
            except Exception as e:
                logger.warning(f"Symbol resync failed for {instrument}: {e}")

    def _health_loop(self):
        """Background thread that periodically checks broker connectivity."""
        logger.info("Health check loop started")

        while self._streaming:
            self.check_connection()

            for _ in range(int(self._health_check_interval_seconds * 10)):
                if not self._streaming:
                    break
                time.sleep(0.1)

        logger.info("Health check loop exited")

    def _stream_loop(self, instruments: list[str]):
        """Background thread that consumes the price stream."""
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
