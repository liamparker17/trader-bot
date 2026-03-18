"""
Trade Executor — Translates approved trade signals into actual MT5 orders.

This is the bridge between the bot's decision-making (indicators + ML + risk)
and the real world (broker API). It handles all the messy details:
- Pre-trade validation
- Order placement with server-side SL/TP
- Fill confirmation and slippage detection
- Position tracking
- Reconciliation with broker state
"""

import logging
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional

from src.config import Config
from src.data.mt5_client import MT5Client, MT5Error
from src.risk.manager import RiskManager, TradeRequest, TradeApproval

logger = logging.getLogger("traderbot.execution")


@dataclass
class OpenTrade:
    """Represents a currently open trade."""
    trade_id: str  # OANDA trade ID
    instrument: str
    direction: str
    units: int
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    ml_confidence: float
    risk_amount: float
    sl_pips: float
    tp_pips: float

    atr_at_entry: float = 0.0

    # Updated as trade progresses
    current_pnl: float = 0.0
    trailing_stop_active: bool = False
    breakeven_moved: bool = False
    highest_price: float = 0.0
    lowest_price: float = 0.0


@dataclass
class TradeResult:
    """Result of a completed trade."""
    trade_id: str
    instrument: str
    direction: str
    units: int
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    stop_loss: float
    take_profit: float
    pnl_pips: float
    pnl_zar: float
    ml_confidence: float
    exit_reason: str
    slippage_pips: float
    spread_at_entry: float


class Executor:
    """
    Executes trades via the MT5 API.

    Flow for each trade:
    1. Receive signal (instrument, direction, confidence)
    2. Get current price and spread
    3. Build TradeRequest for risk manager
    4. If approved: place market order with server-side SL/TP
    5. Confirm fill and check slippage
    6. Track open position
    7. Monitor for trailing stop adjustments
    8. Record result when closed
    """

    def __init__(self, config: Config, client: MT5Client, risk_manager: RiskManager):
        self.config = config
        self.client = client
        self.risk_manager = risk_manager

        # Trailing stop config
        self.trailing_enabled = config.get("risk.trailing_stop_enabled", True)
        self.trailing_activation_rr = config.get("risk.trailing_stop_activation_rr", 1.0)
        self.trailing_atr_mult = config.get("risk.trailing_stop_atr_multiplier", 1.0)
        self.min_seconds_between = config.get("trading.min_seconds_between_trades", 30)

        # State
        self.open_trades: dict[str, OpenTrade] = {}
        self.trade_history: list[TradeResult] = []
        self.last_trade_time: Optional[float] = None

    def execute_signal(
        self,
        instrument: str,
        direction: str,
        ml_confidence: float,
        atr_value: float,
        atr_ratio: float,
    ) -> Optional[OpenTrade]:
        """
        Execute a trading signal end-to-end.

        Returns the OpenTrade if successful, None if rejected or failed.
        """
        # Throttle: minimum time between trades
        if self.last_trade_time:
            elapsed = time.time() - self.last_trade_time
            if elapsed < self.min_seconds_between:
                logger.debug(f"Trade throttled: {elapsed:.0f}s < {self.min_seconds_between}s")
                return None

        # Check for existing position in same instrument
        for trade in self.open_trades.values():
            if trade.instrument == instrument:
                logger.info(f"Already have open trade in {instrument}, skipping")
                return None

        # Step 1: Get current price and spread
        try:
            price_data = self.client.get_current_price(instrument)
        except MT5Error as e:
            logger.error(f"Failed to get price for {instrument}: {e}")
            self.risk_manager.record_api_error()
            return None

        bid = float(price_data["bids"][0]["price"])
        ask = float(price_data["asks"][0]["price"])
        spread = ask - bid
        entry_price = ask if direction == "buy" else bid

        inst_config = self.config.get_instrument(instrument)
        pip_location = inst_config.get("pip_location", -4)
        pip_size = 10 ** pip_location
        spread_pips = spread / pip_size

        # Step 2: Check market is open
        if not price_data.get("tradeable", False):
            logger.info(f"{instrument} market is closed")
            return None

        # Step 3: Submit to risk manager
        request = TradeRequest(
            instrument=instrument,
            direction=direction,
            entry_price=entry_price,
            atr_value=atr_value,
            atr_ratio=atr_ratio,
            ml_confidence=ml_confidence,
            current_spread=spread,
            current_spread_pips=spread_pips,
        )

        balance = self.client.get_account_balance()
        approval = self.risk_manager.evaluate_trade(request, balance)

        if not approval.approved:
            logger.info(f"Trade rejected: {approval.reason}")
            return None

        # Step 4: Place market order with server-side SL/TP
        try:
            response = self.client.place_market_order(
                instrument=instrument,
                units=approval.units,
                stop_loss_price=approval.stop_loss,
                take_profit_price=approval.take_profit,
            )
        except MT5Error as e:
            logger.error(f"Order placement failed: {e}")
            self.risk_manager.record_api_error()
            return None

        # Step 5: Confirm fill
        fill = response.get("orderFillTransaction", {})
        if not fill:
            logger.error("No fill transaction in response")
            return None

        fill_price = float(fill.get("price", entry_price))
        trade_id = fill.get("tradeOpened", {}).get("tradeID", "")
        if not trade_id:
            # Sometimes OANDA nests it differently
            trade_id = str(fill.get("id", f"local_{int(time.time())}"))

        # Check slippage
        slippage_pips = abs(fill_price - entry_price) / pip_size
        if slippage_pips > 3:
            logger.warning(
                f"HIGH SLIPPAGE on {instrument}: {slippage_pips:.1f} pips "
                f"(expected {entry_price:.5f}, got {fill_price:.5f})"
            )

        # Step 6: Track open position
        open_trade = OpenTrade(
            trade_id=trade_id,
            instrument=instrument,
            direction=direction,
            units=approval.units,
            entry_price=fill_price,
            entry_time=datetime.now(timezone.utc),
            stop_loss=approval.stop_loss,
            take_profit=approval.take_profit,
            ml_confidence=ml_confidence,
            risk_amount=approval.risk_amount,
            sl_pips=approval.sl_pips,
            tp_pips=approval.tp_pips,
            atr_at_entry=atr_value,
            highest_price=fill_price,
            lowest_price=fill_price,
        )

        self.open_trades[trade_id] = open_trade
        self.risk_manager.record_trade_opened()
        self.last_trade_time = time.time()

        logger.info(
            f"TRADE OPENED: {instrument} {direction} {abs(approval.units)} units "
            f"@ {fill_price:.5f} | SL: {approval.stop_loss:.5f} | "
            f"TP: {approval.take_profit:.5f} | Slippage: {slippage_pips:.1f} pips"
        )

        return open_trade

    def check_and_manage_positions(self, current_prices: dict[str, dict] = None):
        """
        Check open positions for trailing stop adjustments.

        Call this periodically (e.g., every tick or every minute).

        Args:
            current_prices: Optional dict of instrument → {bid, ask} prices.
                           If None, fetches from OANDA.
        """
        if not self.open_trades:
            return

        for trade_id, trade in list(self.open_trades.items()):
            try:
                self._manage_single_trade(trade, current_prices)
            except Exception as e:
                logger.error(f"Error managing trade {trade_id}: {e}")

    def _manage_single_trade(self, trade: OpenTrade, current_prices: dict = None):
        """Manage trailing stop: breakeven at activation R:R, then trail by ATR.
        Respects per-instrument trailing_stop_enabled config."""
        inst_cfg = self.config.get_instrument(trade.instrument)
        trailing_on = inst_cfg.get("trailing_stop_enabled", self.trailing_enabled) if inst_cfg else self.trailing_enabled
        if not trailing_on:
            return

        # Get current price
        if current_prices and trade.instrument in current_prices:
            price_data = current_prices[trade.instrument]
            current_price = float(price_data.get("bid" if trade.direction == "buy" else "ask", 0))
        else:
            return

        if current_price <= 0:
            return

        inst_config = self.config.get_instrument(trade.instrument)
        pip_size = 10 ** inst_config.get("pip_location", -4)

        # Update price extremes
        if trade.direction == "buy":
            trade.highest_price = max(trade.highest_price, current_price)
            current_pips = (current_price - trade.entry_price) / pip_size
        else:
            trade.lowest_price = min(trade.lowest_price, current_price) if trade.lowest_price > 0 else current_price
            current_pips = (trade.entry_price - current_price) / pip_size

        current_rr = current_pips / trade.sl_pips if trade.sl_pips > 0 else 0
        trade.current_pnl = current_pips

        # Step 1: Move to breakeven at activation R:R
        activation_rr = inst_cfg.get("trailing_stop_activation_rr", self.trailing_activation_rr) if inst_cfg else self.trailing_activation_rr
        if current_rr >= activation_rr and not trade.breakeven_moved:
            new_sl = trade.entry_price + (1 * pip_size) if trade.direction == "buy" else trade.entry_price - (1 * pip_size)
            try:
                self.client.modify_trade(
                    trade_id=trade.trade_id,
                    stop_loss_price=new_sl,
                )
                trade.stop_loss = new_sl
                trade.breakeven_moved = True
                trade.trailing_stop_active = True
                logger.info(
                    f"Trade {trade.trade_id} SL moved to breakeven @ {new_sl:.5f}"
                )
            except MT5Error as e:
                logger.warning(f"Failed to move SL to breakeven: {e}")

        # Step 2: Trail by ATR distance from price extreme
        if trade.breakeven_moved and trade.atr_at_entry > 0:
            trail_distance = trade.atr_at_entry * (inst_cfg.get("trailing_stop_atr_multiplier", self.trailing_atr_mult) if inst_cfg else self.trailing_atr_mult)
            if trade.direction == "buy":
                new_sl = trade.highest_price - trail_distance
                if new_sl > trade.stop_loss:
                    try:
                        self.client.modify_trade(
                            trade_id=trade.trade_id,
                            stop_loss_price=new_sl,
                        )
                        trade.stop_loss = new_sl
                        logger.debug(f"Trade {trade.trade_id} trailing SL moved to {new_sl:.5f}")
                    except MT5Error as e:
                        logger.warning(f"Failed to trail SL: {e}")
            else:
                ref_price = trade.lowest_price if trade.lowest_price < 999999 else current_price
                new_sl = ref_price + trail_distance
                if new_sl < trade.stop_loss:
                    try:
                        self.client.modify_trade(
                            trade_id=trade.trade_id,
                            stop_loss_price=new_sl,
                        )
                        trade.stop_loss = new_sl
                        logger.debug(f"Trade {trade.trade_id} trailing SL moved to {new_sl:.5f}")
                    except MT5Error as e:
                        logger.warning(f"Failed to trail SL: {e}")

    def sync_with_broker(self):
        """
        Reconcile local state with OANDA account state.

        Call periodically (every 60 seconds) to detect:
        - Trades closed by SL/TP on server side
        - Trades we don't know about
        - State mismatches
        """
        try:
            broker_trades = self.client.get_open_trades()
        except MT5Error as e:
            logger.error(f"Reconciliation failed: {e}")
            self.risk_manager.record_api_error()
            return

        broker_trade_ids = {t["id"] for t in broker_trades}
        local_trade_ids = set(self.open_trades.keys())

        # Trades that closed on broker side (SL/TP hit)
        closed_ids = local_trade_ids - broker_trade_ids
        for trade_id in closed_ids:
            trade = self.open_trades.pop(trade_id, None)
            if trade:
                self._record_closed_trade(trade, exit_reason="broker_closed")

        # Trades on broker we don't know about (shouldn't happen normally)
        unknown_ids = broker_trade_ids - local_trade_ids
        if unknown_ids:
            logger.warning(f"Unknown trades on broker: {unknown_ids}")

        # Update position count
        self.risk_manager.open_position_count = len(self.open_trades)

    def close_trade(self, trade_id: str, reason: str = "manual") -> Optional[TradeResult]:
        """Close a specific trade."""
        trade = self.open_trades.get(trade_id)
        if not trade:
            logger.warning(f"Trade {trade_id} not found in open trades")
            return None

        try:
            self.client.close_trade(trade_id)
        except MT5Error as e:
            logger.error(f"Failed to close trade {trade_id}: {e}")
            return None

        return self._record_closed_trade(trade, exit_reason=reason)

    def close_all(self, reason: str = "manual_close_all") -> list[TradeResult]:
        """Close all open trades. Returns list of results."""
        results = []
        for trade_id in list(self.open_trades.keys()):
            result = self.close_trade(trade_id, reason=reason)
            if result:
                results.append(result)

        logger.info(f"Closed all: {len(results)} trades ({reason})")
        return results

    def _record_closed_trade(self, trade: OpenTrade, exit_reason: str) -> TradeResult:
        """Record a closed trade and update risk manager."""
        # Try to get actual exit price from broker
        exit_price = trade.entry_price  # Default fallback
        try:
            # Get the most recent transaction for this trade
            price_data = self.client.get_current_price(trade.instrument)
            bid = float(price_data["bids"][0]["price"])
            ask = float(price_data["asks"][0]["price"])
            exit_price = bid if trade.direction == "buy" else ask
        except Exception:
            pass

        inst_config = self.config.get_instrument(trade.instrument)
        pip_size = 10 ** inst_config.get("pip_location", -4)

        if trade.direction == "buy":
            pnl_pips = (exit_price - trade.entry_price) / pip_size
        else:
            pnl_pips = (trade.entry_price - exit_price) / pip_size

        # Approximate PnL in ZAR
        pip_value_zar = pip_size * 18.5  # Simplified; sizer has full calculation
        pnl_zar = pnl_pips * pip_value_zar * abs(trade.units)

        result = TradeResult(
            trade_id=trade.trade_id,
            instrument=trade.instrument,
            direction=trade.direction,
            units=trade.units,
            entry_price=trade.entry_price,
            exit_price=exit_price,
            entry_time=trade.entry_time,
            exit_time=datetime.now(timezone.utc),
            stop_loss=trade.stop_loss,
            take_profit=trade.take_profit,
            pnl_pips=pnl_pips,
            pnl_zar=pnl_zar,
            ml_confidence=trade.ml_confidence,
            exit_reason=exit_reason,
            slippage_pips=0.0,
            spread_at_entry=0.0,
        )

        self.trade_history.append(result)

        # Remove from open trades
        if trade.trade_id in self.open_trades:
            del self.open_trades[trade.trade_id]

        # Update risk manager
        balance = 0
        try:
            balance = self.client.get_account_balance()
        except Exception:
            pass
        self.risk_manager.record_trade_closed(pnl_zar, balance)

        logger.info(
            f"TRADE CLOSED: {trade.instrument} {trade.direction} | "
            f"PnL: {pnl_pips:+.1f} pips (R{pnl_zar:+.2f}) | "
            f"Reason: {exit_reason}"
        )

        return result

    def get_open_trades_summary(self) -> list[dict]:
        """Get summary of all open trades for monitoring."""
        return [
            {
                "trade_id": t.trade_id,
                "instrument": t.instrument,
                "direction": t.direction,
                "units": t.units,
                "entry_price": t.entry_price,
                "stop_loss": t.stop_loss,
                "take_profit": t.take_profit,
                "ml_confidence": t.ml_confidence,
                "current_pnl_pips": t.current_pnl,
                "trailing_active": t.trailing_stop_active,
                "breakeven_moved": t.breakeven_moved,
            }
            for t in self.open_trades.values()
        ]
