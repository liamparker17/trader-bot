"""
Risk Manager — Central gate that every trade request must pass through.

This is the single most important module for account survival.
It combines:
- Position sizing (how much to trade)
- Drawdown tracking (how much we've lost)
- Circuit breakers (emergency stops)

No trade can be placed without the risk manager's approval.
"""

import logging
from typing import Optional

from src.config import Config
from src.risk.position_sizer import PositionSizer
from src.risk.drawdown_tracker import DrawdownTracker
from src.risk.circuit_breaker import CircuitBreaker

logger = logging.getLogger("traderbot.risk")


class TradeRequest:
    """A request to open a trade, submitted for risk approval."""

    def __init__(
        self,
        instrument: str,
        direction: str,  # "buy" or "sell"
        entry_price: float,
        atr_value: float,
        atr_ratio: float,
        ml_confidence: float,
        current_spread: float = 0.0,
        current_spread_pips: float = 0.0,
    ):
        self.instrument = instrument
        self.direction = direction
        self.entry_price = entry_price
        self.atr_value = atr_value
        self.atr_ratio = atr_ratio
        self.ml_confidence = ml_confidence
        self.current_spread = current_spread
        self.current_spread_pips = current_spread_pips


class TradeApproval:
    """Result of risk manager evaluation."""

    def __init__(
        self,
        approved: bool,
        reason: str,
        units: int = 0,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        risk_amount: float = 0.0,
        sl_pips: float = 0.0,
        tp_pips: float = 0.0,
        adjustments: list = None,
    ):
        self.approved = approved
        self.reason = reason
        self.units = units
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.risk_amount = risk_amount
        self.sl_pips = sl_pips
        self.tp_pips = tp_pips
        self.adjustments = adjustments or []


class RiskManager:
    """
    Central risk orchestrator.

    Every trade passes through evaluate_trade() which runs a full
    checklist before approving or rejecting.

    Pre-trade checklist:
    1. Circuit breaker allows trading?
    2. Drawdown within limits?
    3. Balance above hard floor?
    4. Spread acceptable?
    5. Volatility within range?
    6. Max open positions not exceeded?
    7. Daily trade count not exceeded?
    8. Position size calculable?
    """

    def __init__(self, config: Config):
        self.config = config
        self.sizer = PositionSizer(config)
        self.drawdown = DrawdownTracker(config)
        self.circuit_breaker = CircuitBreaker(config)

        self.max_open_positions = config.get("risk.max_open_positions", 3)
        self.max_trades_per_day = config.get("trading.max_trades_per_day", 60)
        self.low_vol_ratio = config.get("risk.low_volatility_atr_ratio", 0.3)
        self.high_vol_ratio = config.get("risk.high_volatility_atr_ratio", 2.0)

        # Runtime state
        self.open_position_count: int = 0
        self.trades_today: int = 0
        self._initialized = False

    def initialize(self, balance: float):
        """Initialize with current account balance. Call once at startup."""
        self.drawdown.initialize(balance)
        self._initialized = True
        logger.info(f"Risk manager initialized with balance R{balance:.2f}")

    def evaluate_trade(
        self,
        request: TradeRequest,
        current_balance: float,
        current_equity: float = None,
    ) -> TradeApproval:
        """
        Evaluate a trade request against all risk rules.

        This is the main entry point. Returns approval or rejection
        with detailed reason.
        """
        if not self._initialized:
            return TradeApproval(False, "Risk manager not initialized")

        equity = current_equity if current_equity is not None else current_balance
        rejections = []

        # Check 1: Circuit breaker
        cb_status = self.circuit_breaker.can_trade(request.instrument)
        if not cb_status["allowed"]:
            return TradeApproval(False, cb_status["reason"])

        # Check 2: Balance above hard floor
        if not self.circuit_breaker.check_balance(current_balance):
            return TradeApproval(False, f"Below hard floor: R{current_balance:.2f}")

        # Check 3: Drawdown within limits
        dd_check = self.drawdown.check(current_balance, equity)
        if not dd_check["allowed"]:
            reasons = "; ".join(dd_check["violations"])
            return TradeApproval(False, f"Drawdown limit: {reasons}")

        # Check 4: Spread acceptable
        inst_config = self.config.get_instrument(request.instrument)
        typical_spread = inst_config.get("typical_spread_pips", 1.5) if inst_config else 1.5
        if not self.circuit_breaker.check_spread(
            request.instrument, request.current_spread_pips, typical_spread
        ):
            return TradeApproval(False, f"Spread too wide: {request.current_spread_pips:.1f} pips")

        # Check 5: Volatility within range
        if request.atr_ratio < self.low_vol_ratio:
            return TradeApproval(
                False,
                f"Volatility too low: ATR ratio {request.atr_ratio:.2f} < {self.low_vol_ratio}",
            )
        if request.atr_ratio > self.high_vol_ratio * 1.5:
            # Allow up to 1.5x high threshold (sizer will reduce size)
            # But beyond that, reject entirely
            return TradeApproval(
                False,
                f"Volatility extreme: ATR ratio {request.atr_ratio:.2f} > {self.high_vol_ratio * 1.5}",
            )

        # Check 6: Max open positions
        if self.open_position_count >= self.max_open_positions:
            return TradeApproval(
                False,
                f"Max open positions ({self.max_open_positions}) reached",
            )

        # Check 7: Daily trade count
        if self.trades_today >= self.max_trades_per_day:
            return TradeApproval(
                False,
                f"Daily trade limit ({self.max_trades_per_day}) reached",
            )

        # Check 8: Position sizing
        sizing = self.sizer.calculate(
            balance=current_balance,
            instrument=request.instrument,
            direction=request.direction,
            entry_price=request.entry_price,
            atr_value=request.atr_value,
            atr_ratio=request.atr_ratio,
            consecutive_losses=self.circuit_breaker.consecutive_losses,
            current_spread=request.current_spread,
        )

        if sizing is None:
            return TradeApproval(False, "Position sizing failed (spread or config issue)")

        # All checks passed
        logger.info(
            f"TRADE APPROVED: {request.instrument} {request.direction} | "
            f"{sizing['abs_units']} units | Risk: R{sizing['risk_amount']:.2f} | "
            f"SL: {sizing['sl_pips']:.1f} pips | Conf: {request.ml_confidence:.1%}"
        )

        return TradeApproval(
            approved=True,
            reason="All risk checks passed",
            units=sizing["units"],
            stop_loss=sizing["stop_loss"],
            take_profit=sizing["take_profit"],
            risk_amount=sizing["risk_amount"],
            sl_pips=sizing["sl_pips"],
            tp_pips=sizing["tp_pips"],
            adjustments=sizing["adjustments"],
        )

    def record_trade_opened(self):
        """Call when a trade is successfully opened."""
        self.open_position_count += 1
        self.trades_today += 1

    def record_trade_closed(self, pnl: float, current_balance: float):
        """Call when a trade is closed."""
        self.open_position_count = max(0, self.open_position_count - 1)
        won = pnl > 0
        self.circuit_breaker.record_trade_outcome(won)
        self.drawdown.update(current_balance)

    def record_api_error(self):
        """Call when an API error occurs."""
        self.circuit_breaker.record_api_error()

    def reset_daily(self, current_balance: float):
        """Call at the start of each trading day."""
        self.trades_today = 0
        self.drawdown.update(current_balance)

    def get_status(self) -> dict:
        """Get comprehensive risk status for monitoring."""
        return {
            "open_positions": self.open_position_count,
            "trades_today": self.trades_today,
            "consecutive_losses": self.circuit_breaker.consecutive_losses,
            "circuit_breaker": self.circuit_breaker.get_status(),
            "max_drawdown_pct": self.drawdown.max_drawdown_pct,
        }

    def close_all_signal(self) -> bool:
        """Check if we should close all positions (emergency)."""
        return self.circuit_breaker.is_shutdown

    def force_resume(self):
        """Manual override to resume from pause (not shutdown)."""
        self.circuit_breaker.force_resume()
