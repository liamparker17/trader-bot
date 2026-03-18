"""
Circuit Breaker — Emergency stop conditions for the trading bot.

Like a circuit breaker in your house, this module cuts power (stops trading)
before anything catches fire (account blows up).

Monitors multiple danger signals and responds with graduated actions:
- Reduce position size
- Pause trading temporarily
- Full emergency shutdown
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional

from src.config import Config

logger = logging.getLogger("traderbot.risk.breaker")


class CircuitBreaker:
    """
    Emergency stop system with graduated responses.

    Conditions monitored:
    1. Consecutive losses → reduce size, then pause
    2. Win rate degradation → pause + force retrain
    3. Hard floor breach → full shutdown
    4. Spread spikes → skip individual instruments
    5. API errors → pause all trading
    6. Daily/weekly drawdown → handled by DrawdownTracker
    """

    def __init__(self, config: Config):
        self.config = config

        # Thresholds from config
        self.consec_loss_reduce = config.get("risk.consecutive_loss_reduce_at", 3)
        self.consec_loss_pause = config.get("risk.consecutive_loss_pause_at", 5)
        self.pause_duration_min = config.get("risk.pause_duration_minutes", 30)
        self.min_win_rate = config.get("risk.min_win_rate_threshold", 0.45)
        self.win_rate_lookback = config.get("risk.min_win_rate_lookback", 100)
        self.hard_floor = config.get("account.hard_floor_zar", 350)
        self.max_spread_mult = config.get("risk.max_spread_multiplier", 2.0)

        # State
        self.consecutive_losses: int = 0
        self.recent_outcomes: list[bool] = []  # True=win, False=loss
        self.is_paused: bool = False
        self.pause_reason: str = ""
        self.pause_until: Optional[datetime] = None
        self.is_shutdown: bool = False
        self.shutdown_reason: str = ""

        # API error tracking
        self.api_errors: list[float] = []  # timestamps
        self.api_error_window = 600  # 10 minutes
        self.api_error_limit = 5

        # Spread tracking per instrument
        self.blocked_instruments: dict[str, str] = {}  # instrument → reason

    def record_trade_outcome(self, won: bool):
        """Record a trade outcome to update consecutive loss tracking."""
        self.recent_outcomes.append(won)
        if len(self.recent_outcomes) > self.win_rate_lookback:
            self.recent_outcomes = self.recent_outcomes[-self.win_rate_lookback:]

        if won:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            logger.info(f"Consecutive losses: {self.consecutive_losses}")

        # Check consecutive loss pause
        if self.consecutive_losses >= self.consec_loss_pause:
            self._pause(
                f"Consecutive loss pause: {self.consecutive_losses} losses in a row",
                self.pause_duration_min,
            )

    def record_api_error(self):
        """Record an API error for rate tracking."""
        now = time.time()
        self.api_errors.append(now)

        # Clean old errors
        cutoff = now - self.api_error_window
        self.api_errors = [t for t in self.api_errors if t > cutoff]

        if len(self.api_errors) >= self.api_error_limit:
            self._pause(
                f"API error limit: {len(self.api_errors)} errors in "
                f"{self.api_error_window}s",
                self.pause_duration_min,
            )

    def check_spread(self, instrument: str, current_spread_pips: float,
                     typical_spread_pips: float) -> bool:
        """
        Check if spread is acceptable for trading.

        Returns True if spread is OK, False if instrument should be skipped.
        """
        if typical_spread_pips <= 0:
            return True

        spread_ratio = current_spread_pips / typical_spread_pips

        if spread_ratio > self.max_spread_mult:
            reason = (
                f"Spread {current_spread_pips:.1f} pips > "
                f"{self.max_spread_mult}x typical ({typical_spread_pips:.1f})"
            )
            self.blocked_instruments[instrument] = reason
            logger.warning(f"{instrument} blocked: {reason}")
            return False

        # Unblock if previously blocked and spread is now OK
        if instrument in self.blocked_instruments:
            del self.blocked_instruments[instrument]
            logger.info(f"{instrument} unblocked: spread normalized")

        return True

    def check_balance(self, balance: float) -> bool:
        """Check if balance is above hard floor. Returns True if OK."""
        if balance <= self.hard_floor:
            self._shutdown(
                f"HARD FLOOR BREACH: Balance R{balance:.2f} <= R{self.hard_floor}"
            )
            return False
        return True

    def can_trade(self, instrument: str = None) -> dict:
        """
        Check if trading is currently allowed.

        Returns:
            Dict with:
                allowed: bool
                reason: str (why not allowed, or "ok")
                size_reduction: float (1.0 = full size, 0.5 = half)
                needs_retrain: bool
        """
        # Full shutdown
        if self.is_shutdown:
            return {
                "allowed": False,
                "reason": f"SHUTDOWN: {self.shutdown_reason}",
                "size_reduction": 0.0,
                "needs_retrain": False,
            }

        # Timed pause
        if self.is_paused:
            now = datetime.now(timezone.utc)
            if self.pause_until and now >= self.pause_until:
                self._unpause()
            else:
                remaining = ""
                if self.pause_until:
                    secs = (self.pause_until - now).total_seconds()
                    remaining = f" ({secs / 60:.0f} min remaining)"
                return {
                    "allowed": False,
                    "reason": f"PAUSED: {self.pause_reason}{remaining}",
                    "size_reduction": 0.0,
                    "needs_retrain": False,
                }

        # Instrument-specific block
        if instrument and instrument in self.blocked_instruments:
            return {
                "allowed": False,
                "reason": f"{instrument}: {self.blocked_instruments[instrument]}",
                "size_reduction": 0.0,
                "needs_retrain": False,
            }

        # Size reduction for consecutive losses (below pause threshold)
        size_reduction = 1.0
        if self.consecutive_losses >= self.consec_loss_reduce:
            size_reduction = 0.5

        # Win rate check
        needs_retrain = False
        if len(self.recent_outcomes) >= self.win_rate_lookback:
            win_rate = sum(self.recent_outcomes) / len(self.recent_outcomes)
            if win_rate < self.min_win_rate:
                needs_retrain = True
                self._pause(
                    f"Win rate degraded: {win_rate:.1%} < {self.min_win_rate:.1%} "
                    f"over last {len(self.recent_outcomes)} trades",
                    self.pause_duration_min * 2,  # Longer pause for win rate issues
                )
                return {
                    "allowed": False,
                    "reason": f"Win rate {win_rate:.1%} below threshold",
                    "size_reduction": 0.0,
                    "needs_retrain": True,
                }

        return {
            "allowed": True,
            "reason": "ok",
            "size_reduction": size_reduction,
            "needs_retrain": needs_retrain,
        }

    def reset(self):
        """Reset all circuit breaker state. Use after successful retrain or manual review."""
        self.consecutive_losses = 0
        self.is_paused = False
        self.pause_reason = ""
        self.pause_until = None
        self.api_errors.clear()
        self.blocked_instruments.clear()
        logger.info("Circuit breaker reset")

    def force_resume(self):
        """Force resume from pause state (manual override)."""
        if self.is_shutdown:
            logger.warning("Cannot force resume from SHUTDOWN. Manual review required.")
            return
        self._unpause()
        logger.info("Circuit breaker force-resumed by manual override")

    def _pause(self, reason: str, duration_minutes: int):
        """Activate timed pause."""
        self.is_paused = True
        self.pause_reason = reason
        self.pause_until = datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)
        logger.warning(f"CIRCUIT BREAKER PAUSE: {reason} (for {duration_minutes} min)")

    def _unpause(self):
        """Deactivate pause."""
        self.is_paused = False
        self.pause_reason = ""
        self.pause_until = None
        logger.info("Circuit breaker pause lifted")

    def _shutdown(self, reason: str):
        """Full emergency shutdown. Requires manual intervention to resume."""
        self.is_shutdown = True
        self.shutdown_reason = reason
        logger.critical(f"CIRCUIT BREAKER SHUTDOWN: {reason}")

    def get_status(self) -> dict:
        """Get current circuit breaker status for monitoring."""
        return {
            "is_paused": self.is_paused,
            "is_shutdown": self.is_shutdown,
            "pause_reason": self.pause_reason,
            "shutdown_reason": self.shutdown_reason,
            "consecutive_losses": self.consecutive_losses,
            "recent_win_rate": (
                sum(self.recent_outcomes) / len(self.recent_outcomes)
                if self.recent_outcomes else 0
            ),
            "recent_trades_count": len(self.recent_outcomes),
            "api_errors_recent": len(self.api_errors),
            "blocked_instruments": dict(self.blocked_instruments),
        }


# Need this for the timed pause
from datetime import timedelta
