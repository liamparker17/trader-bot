"""
Candle Builder — Converts raw price ticks into OHLC candles.

Used during live trading to build 1-minute and 15-minute candles
from the OANDA price stream.
"""

import logging
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional, Callable

import pandas as pd

logger = logging.getLogger("traderbot.candles")


@dataclass
class Candle:
    """Represents a single OHLC candle."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    complete: bool

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "complete": self.complete,
        }


class CandleBuilder:
    """
    Builds OHLC candles from incoming price ticks.

    Maintains rolling buffers of completed candles for both
    1-minute and 15-minute timeframes.

    Usage:
        builder = CandleBuilder(
            on_candle_complete=my_callback,
            max_buffer_size=500,
        )

        # Feed ticks from the price stream
        for tick in price_stream:
            builder.on_tick(tick)
    """

    def __init__(
        self,
        on_candle_complete: Optional[Callable] = None,
        max_buffer_size: int = 500,
    ):
        """
        Args:
            on_candle_complete: Callback(instrument, timeframe, candle) when a candle completes
            max_buffer_size: Max completed candles to keep in memory per instrument/timeframe
        """
        self.on_candle_complete = on_candle_complete
        self.max_buffer_size = max_buffer_size

        # Current incomplete candles: {instrument: {"M1": Candle, "M15": Candle}}
        self._current: dict[str, dict[str, Optional[Candle]]] = {}

        # Completed candle buffers: {instrument: {"M1": [Candle, ...], "M15": [...]}}
        self._completed: dict[str, dict[str, list[Candle]]] = {}

        # Count of 1-minute candles completed (for building 15-min candles)
        self._m1_count: dict[str, int] = {}

    def on_tick(self, tick: dict):
        """
        Process an incoming price tick.

        Expected tick format (from OANDA stream):
        {
            "type": "PRICE",
            "instrument": "EUR_USD",
            "time": "2026-03-05T10:23:45.123456789Z",
            "bids": [{"price": "1.08432"}],
            "asks": [{"price": "1.08445"}]
        }
        """
        if tick.get("type") != "PRICE":
            return  # Skip heartbeats

        instrument = tick["instrument"]
        timestamp = datetime.fromisoformat(tick["time"].replace("Z", "+00:00"))

        # Use mid price (average of bid and ask)
        bid = float(tick["bids"][0]["price"])
        ask = float(tick["asks"][0]["price"])
        mid = (bid + ask) / 2

        self._ensure_instrument(instrument)
        self._update_candle(instrument, "M1", timestamp, mid, minutes=1)

    def _ensure_instrument(self, instrument: str):
        """Initialize storage for an instrument if not yet seen."""
        if instrument not in self._current:
            self._current[instrument] = {"M1": None, "M15": None}
            self._completed[instrument] = {"M1": [], "M15": []}
            self._m1_count[instrument] = 0

    def _update_candle(self, instrument: str, timeframe: str,
                       timestamp: datetime, price: float, minutes: int):
        """Update current candle with new price, completing it if period elapsed."""
        current = self._current[instrument][timeframe]

        # Determine the candle start time for this tick
        candle_start = self._floor_time(timestamp, minutes)

        if current is None or candle_start > current.timestamp:
            # Complete the previous candle (if any)
            if current is not None:
                current.complete = True
                self._completed[instrument][timeframe].append(current)

                # Trim buffer
                if len(self._completed[instrument][timeframe]) > self.max_buffer_size:
                    self._completed[instrument][timeframe] = \
                        self._completed[instrument][timeframe][-self.max_buffer_size:]

                # Notify callback
                if self.on_candle_complete:
                    self.on_candle_complete(instrument, timeframe, current)

                # Track M1 completions for M15 building
                if timeframe == "M1":
                    self._m1_count[instrument] += 1
                    # Every 15 completed M1 candles, build an M15 candle
                    if self._m1_count[instrument] >= 15:
                        self._build_m15_candle(instrument)
                        self._m1_count[instrument] = 0

            # Start new candle
            self._current[instrument][timeframe] = Candle(
                timestamp=candle_start,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=1,
                complete=False,
            )
        else:
            # Update current candle
            current.high = max(current.high, price)
            current.low = min(current.low, price)
            current.close = price
            current.volume += 1

    def _build_m15_candle(self, instrument: str):
        """Build a 15-minute candle from the last 15 completed 1-minute candles."""
        m1_candles = self._completed[instrument]["M1"]

        if len(m1_candles) < 15:
            return

        last_15 = m1_candles[-15:]
        m15_candle = Candle(
            timestamp=last_15[0].timestamp,
            open=last_15[0].open,
            high=max(c.high for c in last_15),
            low=min(c.low for c in last_15),
            close=last_15[-1].close,
            volume=sum(c.volume for c in last_15),
            complete=True,
        )

        self._completed[instrument]["M15"].append(m15_candle)

        # Trim M15 buffer
        if len(self._completed[instrument]["M15"]) > self.max_buffer_size:
            self._completed[instrument]["M15"] = \
                self._completed[instrument]["M15"][-self.max_buffer_size:]

        if self.on_candle_complete:
            self.on_candle_complete(instrument, "M15", m15_candle)

    def get_candles(self, instrument: str, timeframe: str,
                    count: Optional[int] = None) -> list[Candle]:
        """
        Get completed candles for an instrument/timeframe.

        Args:
            instrument: e.g., "EUR_USD"
            timeframe: "M1" or "M15"
            count: Optional limit on number of candles (most recent)

        Returns:
            List of completed Candle objects, oldest first.
        """
        candles = self._completed.get(instrument, {}).get(timeframe, [])
        if count is not None:
            return candles[-count:]
        return candles

    def get_candles_df(self, instrument: str, timeframe: str,
                       count: Optional[int] = None) -> pd.DataFrame:
        """Get completed candles as a pandas DataFrame."""
        candles = self.get_candles(instrument, timeframe, count)
        if not candles:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame([c.to_dict() for c in candles])
        df.set_index("timestamp", inplace=True)
        df.drop(columns=["complete"], inplace=True, errors="ignore")
        return df

    def load_historical(self, instrument: str, timeframe: str,
                        candles_df: pd.DataFrame):
        """
        Pre-load historical candles into the buffer.
        Used to warm up indicators at startup.

        Args:
            instrument: e.g., "EUR_USD"
            timeframe: "M1" or "M15"
            candles_df: DataFrame with columns [open, high, low, close, volume]
                        and DatetimeIndex
        """
        self._ensure_instrument(instrument)

        for ts, row in candles_df.iterrows():
            candle = Candle(
                timestamp=ts if isinstance(ts, datetime) else pd.Timestamp(ts).to_pydatetime(),
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=int(row.get("volume", 0)),
                complete=True,
            )
            self._completed[instrument][timeframe].append(candle)

        # Trim to buffer size
        if len(self._completed[instrument][timeframe]) > self.max_buffer_size:
            self._completed[instrument][timeframe] = \
                self._completed[instrument][timeframe][-self.max_buffer_size:]

        logger.info(
            f"Loaded {len(self._completed[instrument][timeframe])} historical "
            f"{timeframe} candles for {instrument}"
        )

    @staticmethod
    def _floor_time(dt: datetime, minutes: int) -> datetime:
        """Round a datetime down to the nearest N-minute boundary."""
        total_minutes = dt.hour * 60 + dt.minute
        floored_minutes = (total_minutes // minutes) * minutes
        return dt.replace(
            hour=floored_minutes // 60,
            minute=floored_minutes % 60,
            second=0,
            microsecond=0,
        )
