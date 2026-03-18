"""
MetaTrader 5 Client for TraderBot.

Drop-in replacement for OandaClient. Uses the MetaTrader5 Python package
which communicates with the MT5 desktop terminal via IPC (Windows only).

Handles:
- MT5 terminal initialization and login
- Account information
- Historical candle data fetching
- Live tick/price retrieval
- Order placement and management (market orders with SL/TP)
- Trade modification and closing
- Reconnection logic
"""

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Generator

import MetaTrader5 as mt5
import pandas as pd
import numpy as np

from src.config import Config

logger = logging.getLogger("traderbot.mt5")


# Map our granularity strings to MT5 timeframe constants
TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D": mt5.TIMEFRAME_D1,
    "D1": mt5.TIMEFRAME_D1,
}


class MT5Error(Exception):
    """Raised when MT5 operation fails."""
    def __init__(self, message: str, error_code: int = 0):
        self.error_code = error_code
        super().__init__(f"MT5 Error ({error_code}): {message}")


class MT5Client:
    """
    Client for MetaTrader 5 via the Python bridge.

    Provides the same logical interface as OandaClient so that
    all upstream modules (collector, executor, etc.) can swap
    seamlessly with minimal changes.
    """

    def __init__(self, config: Config):
        self.config = config
        self.max_retries = config.get("broker.max_retries", 3)
        self.retry_delay = config.get("broker.retry_delay_seconds", 1)
        self._connected = False

        # MT5 credentials from config/env
        self.mt5_path = config.get("broker.mt5_terminal_path", "")
        self.login = int(config.mt5_login) if config.mt5_login else 0
        self.password = config.mt5_password or ""
        self.server = config.mt5_server or ""

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """
        Initialize MT5 terminal and log in.

        Returns True if connected successfully.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                # Initialize MT5 terminal
                init_kwargs = {}
                if self.mt5_path:
                    init_kwargs["path"] = self.mt5_path

                if not mt5.initialize(**init_kwargs):
                    error = mt5.last_error()
                    logger.warning(
                        f"MT5 init failed (attempt {attempt}/{self.max_retries}): {error}"
                    )
                    time.sleep(self.retry_delay * attempt)
                    continue

                # Login if credentials provided
                if self.login and self.password:
                    if not mt5.login(self.login, password=self.password, server=self.server):
                        error = mt5.last_error()
                        logger.warning(f"MT5 login failed: {error}")
                        mt5.shutdown()
                        time.sleep(self.retry_delay * attempt)
                        continue

                self._connected = True
                account = mt5.account_info()
                if account:
                    logger.info(
                        f"MT5 connected: {account.server} | "
                        f"Account: {account.login} | "
                        f"Balance: {account.balance} {account.currency}"
                    )
                else:
                    logger.info("MT5 connected (no account info available)")

                return True

            except Exception as e:
                logger.warning(f"MT5 connection error (attempt {attempt}): {e}")
                time.sleep(self.retry_delay * attempt)

        logger.error(f"MT5 connection failed after {self.max_retries} attempts")
        return False

    def ensure_connected(self):
        """Reconnect if connection was lost."""
        if not self._connected or mt5.terminal_info() is None:
            logger.warning("MT5 connection lost, reconnecting...")
            self._connected = False
            if not self.connect():
                raise MT5Error("Failed to reconnect to MT5")

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account_summary(self) -> dict:
        """Get account summary including balance, equity, margin."""
        self.ensure_connected()
        info = mt5.account_info()
        if info is None:
            raise MT5Error("Failed to get account info", mt5.last_error()[0])

        return {
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "marginFree": info.margin_free,
            "marginLevel": info.margin_level,
            "unrealizedPL": info.profit,
            "currency": info.currency,
            "leverage": info.leverage,
            "openTradeCount": len(mt5.positions_get() or []),
        }

    def get_account_balance(self) -> float:
        """Get current account balance."""
        summary = self.get_account_summary()
        return float(summary["balance"])

    def get_open_positions(self) -> list[dict]:
        """Get all open positions."""
        self.ensure_connected()
        positions = mt5.positions_get()
        if positions is None:
            return []

        return [
            {
                "ticket": p.ticket,
                "instrument": p.symbol,
                "type": "buy" if p.type == mt5.ORDER_TYPE_BUY else "sell",
                "volume": p.volume,
                "price_open": p.price_open,
                "price_current": p.price_current,
                "sl": p.sl,
                "tp": p.tp,
                "profit": p.profit,
                "time": datetime.fromtimestamp(p.time, tz=timezone.utc),
            }
            for p in positions
        ]

    def get_open_trades(self) -> list[dict]:
        """Get all open trades (alias for get_open_positions for compatibility)."""
        positions = self.get_open_positions()
        # Convert to OANDA-like format for compatibility
        return [
            {
                "id": str(p["ticket"]),
                "instrument": p["instrument"],
                "currentUnits": str(p["volume"] * (1 if p["type"] == "buy" else -1)),
                "price": str(p["price_open"]),
                "unrealizedPL": str(p["profit"]),
            }
            for p in positions
        ]

    # ------------------------------------------------------------------
    # Historical Data
    # ------------------------------------------------------------------

    def get_candles(
        self,
        instrument: str,
        granularity: str = "M1",
        count: int = None,
        from_time: str = None,
        to_time: str = None,
        **kwargs,
    ) -> list[dict]:
        """
        Fetch historical candle data.

        Args:
            instrument: e.g., "EURUSD" (MT5 format, no underscore)
            granularity: "M1", "M5", "M15", "H1", "D", etc.
            count: Number of candles to fetch
            from_time: Start time (RFC3339 or datetime string)
            to_time: End time (RFC3339 or datetime string)

        Returns:
            List of candle dicts with keys: time, open, high, low, close, volume, complete
            and nested mid dict for compatibility.
        """
        self.ensure_connected()
        symbol = self._to_mt5_symbol(instrument)
        tf = TIMEFRAME_MAP.get(granularity, mt5.TIMEFRAME_M1)

        if count is not None and from_time is None:
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, min(count, 99999))
        elif from_time is not None:
            dt_from = self._parse_time(from_time)
            if to_time is not None:
                dt_to = self._parse_time(to_time)
                rates = mt5.copy_rates_range(symbol, tf, dt_from, dt_to)
            else:
                rates = mt5.copy_rates_from(symbol, tf, dt_from, count or 5000)
        else:
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, 500)

        if rates is None or len(rates) == 0:
            error = mt5.last_error()
            logger.warning(f"No candle data for {symbol} {granularity}: {error}")
            return []

        candles = []
        for r in rates:
            time_str = datetime.fromtimestamp(r['time'], tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.000000000Z"
            )
            candles.append({
                "time": time_str,
                "volume": int(r['tick_volume']),
                "complete": True,
                "mid": {
                    "o": str(r['open']),
                    "h": str(r['high']),
                    "l": str(r['low']),
                    "c": str(r['close']),
                },
            })

        return candles

    def get_candles_batch(
        self,
        instrument: str,
        granularity: str,
        from_time: str,
        to_time: str,
        **kwargs,
    ) -> list[dict]:
        """
        Fetch a large date range of candles.

        For M1 data, fetches in 30-day chunks to avoid MT5's request size limits.
        """
        self.ensure_connected()
        symbol = self._to_mt5_symbol(instrument)
        tf = TIMEFRAME_MAP.get(granularity, mt5.TIMEFRAME_M1)

        dt_from = self._parse_time(from_time)
        dt_to = self._parse_time(to_time)

        logger.info(f"Fetching {symbol} {granularity} from {dt_from} to {dt_to}...")

        # For M1 data, chunk into 30-day windows to avoid MT5 limits
        all_rates = []
        if granularity == "M1":
            chunk_days = 28
            chunk_start = dt_from
            while chunk_start < dt_to:
                chunk_end = min(chunk_start + timedelta(days=chunk_days), dt_to)
                rates = mt5.copy_rates_range(symbol, tf, chunk_start, chunk_end)
                if rates is not None and len(rates) > 0:
                    all_rates.extend(rates)
                    logger.info(f"  Chunk {chunk_start.date()} to {chunk_end.date()}: {len(rates)} candles")
                chunk_start = chunk_end
        else:
            rates = mt5.copy_rates_range(symbol, tf, dt_from, dt_to)
            if rates is not None:
                all_rates = list(rates)

        if not all_rates:
            logger.warning(f"No data returned for {symbol}")
            return []

        candles = []
        for r in all_rates:
            time_str = datetime.fromtimestamp(r['time'], tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.000000000Z"
            )
            candles.append({
                "time": time_str,
                "volume": int(r['tick_volume']),
                "complete": True,
                "mid": {
                    "o": str(r['open']),
                    "h": str(r['high']),
                    "l": str(r['low']),
                    "c": str(r['close']),
                },
            })

        logger.info(f"Fetched {len(candles)} {granularity} candles for {symbol}")
        return candles

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def get_current_price(self, instrument: str) -> dict:
        """
        Get current bid/ask price for an instrument.

        Returns dict compatible with OANDA format.
        """
        self.ensure_connected()
        symbol = self._to_mt5_symbol(instrument)
        tick = mt5.symbol_info_tick(symbol)

        if tick is None:
            raise MT5Error(f"No price data for {symbol}", mt5.last_error()[0])

        return {
            "instrument": instrument,
            "time": datetime.fromtimestamp(tick.time, tz=timezone.utc).isoformat(),
            "bids": [{"price": str(tick.bid)}],
            "asks": [{"price": str(tick.ask)}],
            "bid": tick.bid,
            "ask": tick.ask,
            "tradeable": True,
        }

    def get_spread(self, instrument: str) -> float:
        """Get current spread in price units."""
        price = self.get_current_price(instrument)
        return price["ask"] - price["bid"]

    def stream_prices(self, instruments: list[str]) -> Generator[dict, None, None]:
        """
        Stream live prices by polling MT5 ticks.

        MT5 doesn't have a true streaming API via Python — we poll
        symbol_info_tick() at high frequency. This runs in a loop
        and yields tick dicts in the same format as OANDA's stream.
        """
        self.ensure_connected()
        symbols = [self._to_mt5_symbol(i) for i in instruments]
        symbol_map = dict(zip(symbols, instruments))

        # Enable symbols in Market Watch
        for sym in symbols:
            mt5.symbol_select(sym, True)

        last_ticks = {}
        logger.info(f"Price polling started for {instruments}")

        while True:
            for sym in symbols:
                try:
                    tick = mt5.symbol_info_tick(sym)
                    if tick is None:
                        continue

                    # Only yield if tick changed
                    last = last_ticks.get(sym)
                    if last and last.time == tick.time and last.bid == tick.bid:
                        continue

                    last_ticks[sym] = tick
                    instrument = symbol_map[sym]

                    yield {
                        "type": "PRICE",
                        "instrument": instrument,
                        "time": datetime.fromtimestamp(
                            tick.time, tz=timezone.utc
                        ).strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
                        "bids": [{"price": str(tick.bid)}],
                        "asks": [{"price": str(tick.ask)}],
                    }

                except Exception as e:
                    logger.warning(f"Tick error for {sym}: {e}")

            # Poll interval — ~100ms for responsive candle building
            time.sleep(0.1)

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def place_market_order(
        self,
        instrument: str,
        units: int,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
    ) -> dict:
        """
        Place a market order.

        Args:
            instrument: e.g., "EUR_USD" (will be converted to MT5 format)
            units: Positive for BUY, negative for SELL
            stop_loss_price: Server-side stop-loss price
            take_profit_price: Server-side take-profit price

        Returns:
            Order result dict with fill info.
        """
        self.ensure_connected()
        symbol = self._to_mt5_symbol(instrument)

        # Determine direction and volume
        order_type = mt5.ORDER_TYPE_BUY if units > 0 else mt5.ORDER_TYPE_SELL
        volume = self._units_to_lots(symbol, abs(units))

        # Get current price for the order
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise MT5Error(f"Cannot get price for {symbol}")

        price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

        # Build order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": 20,  # Max slippage in points
            "magic": 234000,  # Magic number to identify our bot's trades
            "comment": "TraderBot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        if stop_loss_price is not None:
            request["sl"] = round(stop_loss_price, self._get_digits(symbol))
        if take_profit_price is not None:
            request["tp"] = round(take_profit_price, self._get_digits(symbol))

        logger.info(
            f"Placing market order: {symbol} {'BUY' if units > 0 else 'SELL'} "
            f"{volume} lots | SL: {stop_loss_price} | TP: {take_profit_price}"
        )

        result = mt5.order_send(request)

        if result is None:
            raise MT5Error(f"Order send returned None: {mt5.last_error()}")

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            raise MT5Error(
                f"Order rejected: {result.comment} (retcode={result.retcode})",
                result.retcode,
            )

        logger.info(
            f"Order filled: {symbol} {volume} lots @ {result.price} | "
            f"Deal: {result.deal} | Order: {result.order}"
        )

        # Return in OANDA-compatible format
        return {
            "orderFillTransaction": {
                "price": str(result.price),
                "tradeOpened": {
                    "tradeID": str(result.order),
                },
                "id": str(result.deal),
            },
        }

    def close_trade(self, trade_id: str, units: str = "ALL") -> dict:
        """
        Close an open trade by position ticket.

        Args:
            trade_id: The MT5 position ticket (as string)
            units: "ALL" or specific volume (not used — MT5 closes full position)
        """
        self.ensure_connected()
        ticket = int(trade_id)

        # Find the position
        position = mt5.positions_get(ticket=ticket)
        if not position:
            raise MT5Error(f"Position {ticket} not found")

        pos = position[0]
        symbol = pos.symbol

        # Opposite order to close
        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(symbol)
        price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": "TraderBot close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            error = result.comment if result else str(mt5.last_error())
            raise MT5Error(f"Failed to close trade {ticket}: {error}")

        logger.info(f"Trade {ticket} closed @ {result.price}")
        return {"price": result.price, "ticket": ticket}

    def close_all_trades(self) -> list[dict]:
        """Close all open positions. Returns list of close results."""
        positions = mt5.positions_get()
        if not positions:
            return []

        results = []
        for pos in positions:
            if pos.magic == 234000:  # Only close our bot's trades
                try:
                    result = self.close_trade(str(pos.ticket))
                    results.append(result)
                except MT5Error as e:
                    logger.error(f"Failed to close {pos.ticket}: {e}")
                    results.append({"error": str(e), "ticket": pos.ticket})

        logger.info(f"Closed {len(results)} trades")
        return results

    def modify_trade(
        self,
        trade_id: str,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        **kwargs,
    ) -> dict:
        """Modify SL/TP on an existing position."""
        self.ensure_connected()
        ticket = int(trade_id)

        position = mt5.positions_get(ticket=ticket)
        if not position:
            raise MT5Error(f"Position {ticket} not found")

        pos = position[0]
        digits = self._get_digits(pos.symbol)

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": ticket,
            "sl": round(stop_loss_price, digits) if stop_loss_price is not None else pos.sl,
            "tp": round(take_profit_price, digits) if take_profit_price is not None else pos.tp,
        }

        result = mt5.order_send(request)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            error = result.comment if result else str(mt5.last_error())
            raise MT5Error(f"Failed to modify trade {ticket}: {error}")

        logger.info(f"Trade {ticket} modified: SL={request['sl']} TP={request['tp']}")
        return {"ticket": ticket, "sl": request["sl"], "tp": request["tp"]}

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def is_market_open(self, instrument: str) -> bool:
        """Check if the market is currently open for trading."""
        self.ensure_connected()
        symbol = self._to_mt5_symbol(instrument)
        info = mt5.symbol_info(symbol)
        if info is None:
            return False
        # MT5 trade mode: TRADE_MODE_FULL = 0 means trading allowed
        return info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL

    def get_server_time(self) -> str:
        """Get connectivity check."""
        self.ensure_connected()
        info = mt5.terminal_info()
        return f"connected (build {info.build})" if info else "disconnected"

    def close(self):
        """Shut down MT5 connection."""
        mt5.shutdown()
        self._connected = False
        logger.info("MT5 connection closed.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_mt5_symbol(self, instrument: str) -> str:
        """
        Convert our instrument format to MT5 symbol.

        "EUR_USD" → "EURUSD"
        "XAU_USD" → "XAUUSD"

        Some brokers use different naming (e.g., "EURUSDm" for micro).
        We check if the symbol exists and try common suffixes.
        """
        # Direct conversion: remove underscore
        symbol = instrument.replace("_", "")

        # Check if symbol exists in MT5
        info = mt5.symbol_info(symbol)
        if info is not None:
            mt5.symbol_select(symbol, True)
            return symbol

        # Try common broker suffixes
        for suffix in ["m", ".i", ".a", ".", "_", "c", "micro"]:
            test = symbol + suffix
            info = mt5.symbol_info(test)
            if info is not None:
                mt5.symbol_select(test, True)
                logger.info(f"Symbol mapped: {instrument} → {test}")
                return test

        # Fallback: return as-is and hope for the best
        logger.warning(f"Symbol {symbol} not found in MT5, using as-is")
        mt5.symbol_select(symbol, True)
        return symbol

    def _units_to_lots(self, symbol: str, units: int) -> float:
        """
        Convert position units to MT5 lot size.

        MT5 uses lots, not units:
        - Standard lot = 100,000 units
        - Mini lot = 10,000
        - Micro lot = 1,000

        For forex: 1 lot = 100,000 units
        For gold: 1 lot = 100 oz (varies by broker)
        """
        info = mt5.symbol_info(symbol)
        if info is None:
            # Default: assume standard forex contract
            contract_size = 100000
        else:
            contract_size = info.trade_contract_size

        if contract_size <= 0:
            contract_size = 100000

        lots = units / contract_size

        # Round to broker's lot step
        lot_step = info.volume_step if info else 0.01
        lots = max(lot_step, round(lots / lot_step) * lot_step)

        # Clamp to min/max
        min_lot = info.volume_min if info else 0.01
        max_lot = info.volume_max if info else 100.0
        lots = max(min_lot, min(lots, max_lot))

        return round(lots, 2)

    def _get_digits(self, symbol: str) -> int:
        """Get price decimal places for a symbol."""
        info = mt5.symbol_info(symbol)
        return info.digits if info else 5

    @staticmethod
    def _parse_time(time_str: str) -> datetime:
        """Parse a time string to datetime."""
        if isinstance(time_str, datetime):
            return time_str

        # Handle various formats
        for fmt in [
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%S+00:00",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]:
            try:
                return datetime.strptime(time_str, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue

        raise ValueError(f"Cannot parse time: {time_str}")
