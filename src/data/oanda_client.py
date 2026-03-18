"""
OANDA API Client for TraderBot.

Handles:
- REST API calls (account info, historical candles, order placement)
- Streaming price feed via HTTP streaming (OANDA uses chunked HTTP, not WebSocket)
- Authentication, error handling, rate limiting, reconnection logic
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional, Generator

import requests

from src.config import Config

logger = logging.getLogger("traderbot.oanda")


class OandaAPIError(Exception):
    """Raised when OANDA API returns an error."""

    def __init__(self, status_code: int, message: str, error_code: str = ""):
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(f"OANDA API Error {status_code}: {message}")


class OandaClient:
    """
    Client for the OANDA v20 REST API.

    Handles all communication with OANDA including:
    - Account information
    - Historical candle data fetching
    - Live price streaming
    - Order placement and management
    """

    def __init__(self, config: Config):
        self.config = config
        self.api_url = config.api_base_url
        self.stream_url = config.stream_base_url
        self.account_id = config.oanda_account_id
        self.max_retries = config.get("broker.max_retries", 3)
        self.retry_delay = config.get("broker.retry_delay_seconds", 1)
        self.timeout = config.get("broker.request_timeout_seconds", 10)

        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {config.oanda_api_key}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": "RFC3339",
        })

        self._stream_session: Optional[requests.Session] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request(self, method: str, path: str, params: dict = None,
                 data: dict = None, stream: bool = False) -> dict | requests.Response:
        """
        Make an API request with retry logic.

        Returns parsed JSON for normal requests, or raw Response for streams.
        """
        url = f"{self.stream_url if stream else self.api_url}{path}"

        for attempt in range(1, self.max_retries + 1):
            try:
                session = self._stream_session if stream else self._session
                if session is None:
                    session = self._session

                response = session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data,
                    timeout=self.timeout if not stream else None,
                    stream=stream,
                )

                if stream:
                    response.raise_for_status()
                    return response

                if response.status_code in (200, 201):
                    return response.json()

                # Parse error response
                error_body = response.json() if response.text else {}
                error_msg = error_body.get("errorMessage", response.text)
                error_code = error_body.get("errorCode", "")

                # Don't retry client errors (4xx) except rate limits
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    raise OandaAPIError(response.status_code, error_msg, error_code)

                logger.warning(
                    f"API request failed (attempt {attempt}/{self.max_retries}): "
                    f"{response.status_code} {error_msg}"
                )

            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error (attempt {attempt}/{self.max_retries}): {e}")
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt}/{self.max_retries})")
            except OandaAPIError:
                raise  # Don't retry client errors

            if attempt < self.max_retries:
                sleep_time = self.retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                logger.info(f"Retrying in {sleep_time}s...")
                time.sleep(sleep_time)

        raise OandaAPIError(0, f"Request failed after {self.max_retries} attempts")

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account_summary(self) -> dict:
        """
        Get account summary including balance, unrealized PnL, margin used.

        Returns dict with keys like:
            balance, unrealizedPL, pl, marginUsed, marginAvailable, openTradeCount
        """
        path = f"/v3/accounts/{self.account_id}/summary"
        response = self._request("GET", path)
        return response.get("account", {})

    def get_account_balance(self) -> float:
        """Get current account balance as a float."""
        summary = self.get_account_summary()
        return float(summary.get("balance", 0))

    def get_open_positions(self) -> list[dict]:
        """Get all open positions."""
        path = f"/v3/accounts/{self.account_id}/openPositions"
        response = self._request("GET", path)
        return response.get("positions", [])

    def get_open_trades(self) -> list[dict]:
        """Get all open trades with full details."""
        path = f"/v3/accounts/{self.account_id}/openTrades"
        response = self._request("GET", path)
        return response.get("trades", [])

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
        price: str = "MBA",
    ) -> list[dict]:
        """
        Fetch historical candle data.

        Args:
            instrument: e.g., "EUR_USD"
            granularity: "M1", "M5", "M15", "H1", "D", etc.
            count: Number of candles (max 5000). Use either count or from/to.
            from_time: Start time in RFC3339 format
            to_time: End time in RFC3339 format
            price: "M" (mid), "B" (bid), "A" (ask), or combo like "MBA"

        Returns:
            List of candle dicts with keys: time, volume, complete,
            and mid/bid/ask containing o, h, l, c
        """
        path = f"/v3/instruments/{instrument}/candles"
        params = {
            "granularity": granularity,
            "price": price,
        }

        if count is not None:
            params["count"] = min(count, 5000)
        if from_time is not None:
            params["from"] = from_time
        if to_time is not None:
            params["to"] = to_time

        response = self._request("GET", path, params=params)
        return response.get("candles", [])

    def get_candles_batch(
        self,
        instrument: str,
        granularity: str,
        from_time: str,
        to_time: str,
        price: str = "MBA",
    ) -> list[dict]:
        """
        Fetch a large date range of candles by paginating through 5000-candle chunks.

        Args:
            instrument: e.g., "EUR_USD"
            granularity: e.g., "M1"
            from_time: Start time (RFC3339)
            to_time: End time (RFC3339)
            price: Price component(s)

        Returns:
            All candles in the range as a flat list.
        """
        all_candles = []
        current_from = from_time

        while True:
            logger.info(
                f"Fetching {instrument} {granularity} candles from {current_from}..."
            )

            candles = self.get_candles(
                instrument=instrument,
                granularity=granularity,
                from_time=current_from,
                to_time=to_time,
                count=5000,
                price=price,
            )

            if not candles:
                break

            all_candles.extend(candles)

            # Get the time of the last candle and use it as next start
            last_time = candles[-1]["time"]

            # If last candle time hasn't advanced, we're done
            if last_time <= current_from:
                break

            current_from = last_time

            # Small delay to respect rate limits
            time.sleep(0.1)

            logger.info(f"  Fetched {len(all_candles)} candles so far...")

        # Remove duplicates (overlapping boundary candles)
        seen_times = set()
        unique_candles = []
        for c in all_candles:
            if c["time"] not in seen_times:
                seen_times.add(c["time"])
                unique_candles.append(c)

        logger.info(
            f"Fetched {len(unique_candles)} total {granularity} candles for {instrument}"
        )
        return unique_candles

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def get_current_price(self, instrument: str) -> dict:
        """
        Get current bid/ask price for an instrument.

        Returns dict with keys: instrument, time, bids, asks, closeoutBid, closeoutAsk
        """
        path = f"/v3/accounts/{self.account_id}/pricing"
        params = {"instruments": instrument}
        response = self._request("GET", path, params=params)
        prices = response.get("prices", [])
        if prices:
            return prices[0]
        raise OandaAPIError(404, f"No price data for {instrument}")

    def get_spread(self, instrument: str) -> float:
        """Get current spread in price units (not pips)."""
        price = self.get_current_price(instrument)
        bid = float(price["bids"][0]["price"])
        ask = float(price["asks"][0]["price"])
        return ask - bid

    def stream_prices(self, instruments: list[str]) -> Generator[dict, None, None]:
        """
        Stream live prices via OANDA's HTTP streaming API.

        Yields price tick dicts with keys:
            type ("PRICE" or "HEARTBEAT"), instrument, time, bids, asks

        This is a blocking generator — run in a separate thread.
        """
        path = f"/v3/accounts/{self.account_id}/pricing/stream"
        params = {"instruments": ",".join(instruments)}

        self._stream_session = requests.Session()
        self._stream_session.headers.update({
            "Authorization": f"Bearer {self.config.oanda_api_key}",
        })

        reconnect_attempts = 0
        max_reconnects = self.config.get("broker.reconnect_max_attempts", 10)
        max_backoff = self.config.get("broker.reconnect_backoff_max_seconds", 60)

        while reconnect_attempts < max_reconnects:
            try:
                response = self._request("GET", path, params=params, stream=True)
                reconnect_attempts = 0  # Reset on successful connection

                logger.info(f"Price stream connected for {instruments}")

                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        tick = json.loads(line.decode("utf-8"))
                        yield tick
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse stream data: {e}")
                        continue

            except (requests.exceptions.ConnectionError,
                    requests.exceptions.ChunkedEncodingError) as e:
                reconnect_attempts += 1
                backoff = min(2 ** reconnect_attempts, max_backoff)
                logger.warning(
                    f"Stream disconnected (attempt {reconnect_attempts}/{max_reconnects}): "
                    f"{e}. Reconnecting in {backoff}s..."
                )
                time.sleep(backoff)

            except Exception as e:
                logger.error(f"Unexpected stream error: {e}", exc_info=True)
                reconnect_attempts += 1
                time.sleep(5)

        logger.critical(
            f"Price stream failed after {max_reconnects} reconnection attempts. "
            "Trading should be paused."
        )

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
            instrument: e.g., "EUR_USD"
            units: Positive for BUY, negative for SELL
            stop_loss_price: Server-side stop-loss price
            take_profit_price: Server-side take-profit price

        Returns:
            Order fill response dict
        """
        path = f"/v3/accounts/{self.account_id}/orders"

        order_body = {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(units),
            "timeInForce": "FOK",  # Fill Or Kill
        }

        if stop_loss_price is not None:
            order_body["stopLossOnFill"] = {
                "price": f"{stop_loss_price:.5f}",
                "timeInForce": "GTC",  # Good Till Cancelled
            }

        if take_profit_price is not None:
            order_body["takeProfitOnFill"] = {
                "price": f"{take_profit_price:.5f}",
                "timeInForce": "GTC",
            }

        data = {"order": order_body}

        logger.info(
            f"Placing market order: {instrument} {units} units | "
            f"SL: {stop_loss_price} | TP: {take_profit_price}"
        )

        response = self._request("POST", path, data=data)

        # Check for fill
        if "orderFillTransaction" in response:
            fill = response["orderFillTransaction"]
            logger.info(
                f"Order filled: {instrument} {units} units @ {fill.get('price', 'N/A')}"
            )
            return response

        # Check for rejection
        if "orderCancelTransaction" in response:
            cancel = response["orderCancelTransaction"]
            reason = cancel.get("reason", "UNKNOWN")
            raise OandaAPIError(400, f"Order rejected: {reason}")

        return response

    def close_trade(self, trade_id: str, units: str = "ALL") -> dict:
        """
        Close an open trade.

        Args:
            trade_id: The OANDA trade ID
            units: Number of units to close, or "ALL"
        """
        path = f"/v3/accounts/{self.account_id}/trades/{trade_id}/close"
        data = {"units": units}

        logger.info(f"Closing trade {trade_id} ({units} units)")
        return self._request("PUT", path, data=data)

    def close_all_trades(self) -> list[dict]:
        """Close all open trades. Returns list of close responses."""
        trades = self.get_open_trades()
        results = []

        for trade in trades:
            try:
                result = self.close_trade(trade["id"])
                results.append(result)
            except OandaAPIError as e:
                logger.error(f"Failed to close trade {trade['id']}: {e}")
                results.append({"error": str(e), "trade_id": trade["id"]})

        logger.info(f"Closed {len(results)} trades")
        return results

    def modify_trade(
        self,
        trade_id: str,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        trailing_stop_distance: Optional[float] = None,
    ) -> dict:
        """
        Modify SL/TP on an existing trade.

        Use this for trailing stop adjustments.
        """
        path = f"/v3/accounts/{self.account_id}/trades/{trade_id}/orders"
        data = {}

        if stop_loss_price is not None:
            data["stopLoss"] = {
                "price": f"{stop_loss_price:.5f}",
                "timeInForce": "GTC",
            }

        if take_profit_price is not None:
            data["takeProfit"] = {
                "price": f"{take_profit_price:.5f}",
                "timeInForce": "GTC",
            }

        if trailing_stop_distance is not None:
            data["trailingStopLoss"] = {
                "distance": f"{trailing_stop_distance:.5f}",
                "timeInForce": "GTC",
            }

        return self._request("PUT", path, data=data)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_server_time(self) -> str:
        """Get current server time to check connectivity."""
        # Use a lightweight endpoint
        path = f"/v3/accounts/{self.account_id}/summary"
        response = self._request("GET", path)
        return response.get("account", {}).get("lastTransactionID", "connected")

    def is_market_open(self, instrument: str) -> bool:
        """Check if the market is currently open for trading."""
        try:
            price = self.get_current_price(instrument)
            return price.get("tradeable", False)
        except OandaAPIError:
            return False

    def close(self):
        """Clean up sessions."""
        self._session.close()
        if self._stream_session:
            self._stream_session.close()
        logger.info("OANDA client sessions closed.")
