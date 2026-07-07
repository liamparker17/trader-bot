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
from datetime import datetime, timezone
from typing import Callable, Optional

from src.config import Config
from src.risk.position_sizer import PositionSizer
from src.risk.drawdown_tracker import DrawdownTracker, session_boundary
from src.risk.circuit_breaker import CircuitBreaker
from src.risk.ratchet_floor import RatchetFloor

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

    def __init__(
        self,
        config: Config,
        clock: Optional[Callable[[], datetime]] = None,
        ratchet_floor: Optional[RatchetFloor] = None,
    ):
        self.config = config

        # Injectable clock so tests can freeze/advance time across the
        # 21:00 UTC session boundary without sleeping. Defaults to real UTC.
        # Shared with the DrawdownTracker so both agree on "now".
        self.clock: Callable[[], datetime] = clock or (lambda: datetime.now(timezone.utc))

        self.sizer = PositionSizer(config)

        # One shared ratchet floor so the high-water mark used by the
        # drawdown tracker's hard-floor check and the circuit breaker's
        # kill switch always agree. Tests can inject one pointed at a
        # tmp_path state file to avoid touching data/account_state.json.
        self.ratchet_floor = ratchet_floor or RatchetFloor(
            min_floor_zar=config.get("risk.min_floor_zar", 600),
            max_total_drawdown_pct=config.get("risk.max_total_drawdown_pct", 0.35),
        )
        self.drawdown = DrawdownTracker(config, ratchet_floor=self.ratchet_floor, clock=self.clock)
        self.circuit_breaker = CircuitBreaker(config, ratchet_floor=self.ratchet_floor)

        self.max_open_positions = config.get("risk.max_open_positions", 3)
        self.max_trades_per_day = config.get("trading.max_trades_per_day", 60)
        self.low_vol_ratio = config.get("risk.low_volatility_atr_ratio", 0.3)
        self.high_vol_ratio = config.get("risk.high_volatility_atr_ratio", 2.0)

        # Session boundary (21:00 UTC) — daily drawdown block + consecutive
        # loss counter both reset here. Falls back to the pre-existing
        # risk.session_boundary_hour_utc key if the newer trading.* key
        # isn't set (see config/settings.yaml).
        self.session_reset_hour = config.get(
            "trading.session_reset_hour_utc",
            config.get("risk.session_boundary_hour_utc", 21),
        )
        self._last_daily_boundary: Optional[datetime] = None

        # Runtime state
        self.open_position_count: int = 0
        self.trades_today: int = 0
        self._initialized = False

        # Emergency block state — set when a daily-drawdown breach fires.
        # Blocks new entries until the next 21:00 UTC session boundary,
        # independent of the circuit breaker's (manual-resume) shutdown.
        self._blocked_until_boundary: bool = False
        self._block_reason: str = ""

        # Manual pause (Task 8: control queue `pause`/`resume` verbs, and
        # the main-loop's own session-boundary/drawdown-check exception
        # handler). Independent of the daily-drawdown block above and of
        # the circuit breaker's shutdown/resume — this is an operator- or
        # code-triggered "stop opening new trades" switch that persists
        # until explicitly cleared (no automatic session-boundary reset).
        self._manual_pause_reason: str = ""

    def initialize(self, balance: float):
        """Initialize with current account balance. Call once at startup."""
        self.drawdown.initialize(balance)
        self._last_daily_boundary = session_boundary(self.clock(), self.session_reset_hour)
        self._initialized = True
        logger.info(f"Risk manager initialized with balance R{balance:.2f}")

    def check_session_boundary(self, current_balance: Optional[float] = None) -> bool:
        """
        Detect whether the 21:00 UTC session boundary has been crossed since
        the last check. On crossing: resets today's trade count, lifts any
        daily-drawdown emergency block, resets the circuit breaker's
        consecutive-loss counter (the win-rate rolling window is untouched —
        it isn't session-scoped), and — when `current_balance` is supplied —
        rolls the DrawdownTracker's own daily/weekly baseline forward too
        (DrawdownTracker shares this same clock, so it detects the identical
        crossing and rebases its start-of-day/week balance).

        Call periodically (e.g. every tick) as well as at the top of
        evaluate_trade() so the reset fires even if nothing else runs.

        Returns True if a boundary crossing was handled this call.
        """
        now = self.clock()
        boundary = session_boundary(now, self.session_reset_hour)

        if self._last_daily_boundary is None:
            self._last_daily_boundary = boundary
            return False

        if boundary == self._last_daily_boundary:
            return False

        self._last_daily_boundary = boundary
        self.trades_today = 0
        self.circuit_breaker.reset_consecutive_losses()
        if current_balance is not None:
            self.drawdown.update(current_balance)
        if self._blocked_until_boundary:
            logger.info(
                f"Session boundary crossed ({boundary.isoformat()}) — "
                f"daily drawdown block lifted ({self._block_reason})"
            )
        self._blocked_until_boundary = False
        self._block_reason = ""
        return True

    def check_drawdown_emergency(
        self, current_balance: float, current_equity: float = None
    ) -> bool:
        """
        Periodic check (call every tick / loop iteration, independent of
        whether a new trade is being evaluated): advances the session
        boundary and checks for a daily-drawdown breach.

        Returns True the moment a daily-drawdown breach is first detected —
        the caller (Executor) should respond by closing all open positions
        at market and alerting. Returns False on every subsequent call while
        still blocked (so callers don't re-trigger close_all repeatedly).
        """
        self.check_session_boundary(current_balance)

        equity = current_equity if current_equity is not None else current_balance
        dd_check = self.drawdown.check(current_balance, equity)
        daily_violations = [v for v in dd_check["violations"] if "Daily drawdown" in v]

        if daily_violations and not self._blocked_until_boundary:
            self._blocked_until_boundary = True
            self._block_reason = "; ".join(daily_violations)
            logger.critical(
                f"DAILY DRAWDOWN BREACH: {self._block_reason} — closing all "
                f"positions, blocking new entries until next session boundary"
            )
            return True

        return False

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

        # Check -1: Manual pause (control queue `pause` verb, or an
        # exception in the main loop's session-boundary/drawdown checks).
        if self._manual_pause_reason:
            return TradeApproval(False, f"Manually paused: {self._manual_pause_reason}")

        # Check 0: Session boundary / daily-drawdown emergency block
        self.check_session_boundary(current_balance)
        if self._blocked_until_boundary:
            return TradeApproval(
                False,
                f"Blocked until next session boundary (daily drawdown): {self._block_reason}",
            )

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
            daily_violations = [v for v in dd_check["violations"] if "Daily drawdown" in v]
            if daily_violations and not self._blocked_until_boundary:
                self._blocked_until_boundary = True
                self._block_reason = "; ".join(daily_violations)
                logger.critical(
                    f"DAILY DRAWDOWN BREACH: {self._block_reason} — closing all "
                    f"positions, blocking new entries until next session boundary"
                )
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

        # Check 7.5: Per-instrument weight mute (Task 10). weight.<INSTRUMENT>
        # == 0.0 means the instrument has been fully muted via `tb tune` —
        # reject the entry before sizing is even attempted.
        if self.sizer.get_weight(request.instrument) == 0.0:
            return TradeApproval(False, f"muted by weight: {request.instrument}")

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
        """
        Check if we should PERMANENTLY stop the bot and close all positions
        (hard-floor breach / circuit-breaker shutdown).

        This is deliberately NOT the same thing as the resumable daily-
        drawdown block (`entries_blocked` / `_blocked_until_boundary`):
        that one lifts automatically at the next session boundary and must
        not stop the bot. Use `check_drawdown_emergency()` to detect and
        react to the daily-drawdown breach instead.
        """
        return self.circuit_breaker.is_shutdown

    @property
    def close_all_reason(self) -> str:
        """Human-readable reason for the current close_all_signal(), if any."""
        if self.circuit_breaker.is_shutdown:
            return f"circuit_breaker_shutdown: {self.circuit_breaker.shutdown_reason}"
        return ""

    @property
    def entries_blocked(self) -> bool:
        """
        True while new entries are blocked by a daily-drawdown breach that
        hasn't yet been lifted by a session boundary crossing. Resumable —
        does NOT imply the bot should stop or that open positions should be
        closed again (that already happened when the breach was detected).
        """
        return self._blocked_until_boundary

    def force_resume(self):
        """Manual override to resume from pause (not shutdown)."""
        self.circuit_breaker.force_resume()

    @property
    def manual_paused(self) -> bool:
        """True while a manual pause (control queue or exception-triggered) is active."""
        return bool(self._manual_pause_reason)

    @property
    def manual_pause_reason(self) -> str:
        return self._manual_pause_reason

    def set_manual_pause(self, reason: str):
        """Block new entries until `clear_manual_pause()` is called."""
        self._manual_pause_reason = reason
        logger.warning(f"Manual pause engaged: {reason}")

    def clear_manual_pause(self):
        """Lift a manual pause set by `set_manual_pause()`."""
        if self._manual_pause_reason:
            logger.info(f"Manual pause cleared (was: {self._manual_pause_reason})")
        self._manual_pause_reason = ""
