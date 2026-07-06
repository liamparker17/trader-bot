"""
Drawdown Tracker — Monitors account drawdown at daily, weekly, and total levels.

Drawdown = how far the account has fallen from its peak. This is the most
important risk metric because it measures how close you are to ruin.

Daily drawdown: Compared to the balance at start of day.
Weekly drawdown: Compared to the balance at start of week.
Total drawdown: Compared to the all-time high balance (high-water mark).
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Callable, Optional

from src.config import Config
from src.risk.ratchet_floor import RatchetFloor

logger = logging.getLogger("traderbot.risk.drawdown")

# Friday, per Python's Monday=0 .. Sunday=6 weekday() convention.
FRIDAY = 4


def session_boundary(now: datetime, reset_hour: int, weekday: Optional[int] = None) -> datetime:
    """
    Return the most recent session-boundary timestamp at or before `now`.

    The boundary recurs daily at `reset_hour`:00 UTC. When `weekday` is given
    (0=Monday .. 6=Sunday), the boundary instead recurs weekly, anchored to
    that weekday at `reset_hour`:00 UTC (e.g. Friday 21:00 UTC).

    `now` must be timezone-aware (UTC).
    """
    candidate = now.replace(hour=reset_hour, minute=0, second=0, microsecond=0)
    if weekday is not None:
        days_back = (candidate.weekday() - weekday) % 7
        candidate -= timedelta(days=days_back)
    if candidate > now:
        candidate -= timedelta(days=7 if weekday is not None else 1)
    return candidate


class DrawdownTracker:
    """
    Tracks drawdown at multiple time horizons.

    Provides real-time drawdown checks that the risk manager uses
    to decide whether to pause trading.

    Daily/weekly counters reset at the trading session boundary
    (`trading.session_reset_hour_utc`, default 21:00 UTC) rather than at
    midnight — the week rolls over at Friday 21:00 UTC.
    """

    def __init__(
        self,
        config: Config,
        ratchet_floor: Optional[RatchetFloor] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ):
        self.config = config
        self.daily_limit = config.get("risk.daily_drawdown_limit_pct", 4.0) / 100
        self.weekly_limit = config.get("risk.weekly_drawdown_limit_pct", 8.0) / 100

        # Injectable clock so tests can freeze/advance time across the
        # 21:00 UTC session boundary without sleeping. Defaults to real UTC.
        self.clock: Callable[[], datetime] = clock or (lambda: datetime.now(timezone.utc))

        # Session boundary hour. Falls back to the pre-existing
        # risk.session_boundary_hour_utc key already present in
        # config/settings.yaml if the newer trading.* key isn't set.
        self.session_reset_hour = config.get(
            "trading.session_reset_hour_utc",
            config.get("risk.session_boundary_hour_utc", 21),
        )

        # Ratcheting hard floor (replaces the old fixed hard_floor_zar).
        # Same injection pattern as CircuitBreaker — share one RatchetFloor
        # instance across both when wired up by the risk manager so the
        # high-water mark stays consistent.
        self.ratchet_floor = ratchet_floor or RatchetFloor(
            min_floor_zar=config.get("risk.min_floor_zar", 600),
            max_total_drawdown_pct=config.get("risk.max_total_drawdown_pct", 0.35),
        )

        # State
        self.high_water_mark: float = 0.0
        self.daily_start_balance: float = 0.0
        self.weekly_start_balance: float = 0.0
        self.current_date: Optional[datetime] = None
        self.current_week_start: Optional[datetime] = None

        # Drawdown history for analytics
        self.daily_drawdowns: list[dict] = []
        self.max_drawdown_pct: float = 0.0

    def initialize(self, balance: float):
        """Set initial state. Call once at startup."""
        self.high_water_mark = balance
        self.daily_start_balance = balance
        self.weekly_start_balance = balance
        now = self.clock()
        self.current_date = session_boundary(now, self.session_reset_hour)
        self.current_week_start = session_boundary(now, self.session_reset_hour, weekday=FRIDAY)
        logger.info(
            f"Drawdown tracker initialized | Balance: R{balance:.2f} | "
            f"Daily limit: {self.daily_limit:.1%} | Weekly limit: {self.weekly_limit:.1%}"
        )

    def update(self, current_balance: float, current_equity: float = None):
        """
        Update with current balance/equity. Call after every trade close
        and periodically during trading.

        Args:
            current_balance: Realized account balance
            current_equity: Balance + unrealized PnL (optional, uses balance if not given)
        """
        equity = current_equity if current_equity is not None else current_balance
        now = self.clock()

        # Check for a new trading day (boundary = session_reset_hour:00 UTC)
        daily_boundary = session_boundary(now, self.session_reset_hour)
        if self.current_date is None or daily_boundary != self.current_date:
            self._handle_new_day(current_balance, now, daily_boundary)

        # Check for a new trading week (boundary = Friday session_reset_hour:00 UTC)
        weekly_boundary = session_boundary(now, self.session_reset_hour, weekday=FRIDAY)
        if self.current_week_start is None or weekly_boundary != self.current_week_start:
            self._handle_new_week(current_balance, now, weekly_boundary)

        # Update high water mark
        if current_balance > self.high_water_mark:
            self.high_water_mark = current_balance

        # Track max drawdown
        if self.high_water_mark > 0:
            total_dd = (self.high_water_mark - equity) / self.high_water_mark
            if total_dd > self.max_drawdown_pct:
                self.max_drawdown_pct = total_dd

    def check(self, current_balance: float, current_equity: float = None) -> dict:
        """
        Check all drawdown conditions.

        Returns:
            Dict with:
                allowed: bool — True if trading is allowed
                daily_drawdown_pct: float
                weekly_drawdown_pct: float
                total_drawdown_pct: float
                violations: list of strings describing any breaches
        """
        equity = current_equity if current_equity is not None else current_balance
        violations = []

        # Daily drawdown
        daily_dd = 0.0
        if self.daily_start_balance > 0:
            daily_loss = self.daily_start_balance - equity
            daily_dd = daily_loss / self.daily_start_balance
            if daily_dd >= self.daily_limit:
                violations.append(
                    f"Daily drawdown {daily_dd:.1%} >= limit {self.daily_limit:.1%}"
                )

        # Weekly drawdown
        weekly_dd = 0.0
        if self.weekly_start_balance > 0:
            weekly_loss = self.weekly_start_balance - equity
            weekly_dd = weekly_loss / self.weekly_start_balance
            if weekly_dd >= self.weekly_limit:
                violations.append(
                    f"Weekly drawdown {weekly_dd:.1%} >= limit {self.weekly_limit:.1%}"
                )

        # Total drawdown (from high water mark)
        total_dd = 0.0
        if self.high_water_mark > 0:
            total_dd = (self.high_water_mark - equity) / self.high_water_mark

        # Hard floor (ratcheting)
        floor = self.ratchet_floor.update(current_balance)
        if self.ratchet_floor.is_breached(current_balance):
            violations.append(
                f"Balance R{current_balance:.2f} <= hard floor R{floor:.2f}"
            )

        return {
            "allowed": len(violations) == 0,
            "daily_drawdown_pct": daily_dd,
            "weekly_drawdown_pct": weekly_dd,
            "total_drawdown_pct": total_dd,
            "max_drawdown_pct": self.max_drawdown_pct,
            "daily_remaining_pct": max(0, self.daily_limit - daily_dd),
            "violations": violations,
        }

    def get_daily_loss(self, current_balance: float) -> float:
        """Get today's loss in ZAR."""
        return max(0, self.daily_start_balance - current_balance)

    def get_daily_drawdown_pct(self, current_balance: float) -> float:
        """Get today's drawdown as a percentage."""
        if self.daily_start_balance <= 0:
            return 0.0
        return max(0, (self.daily_start_balance - current_balance) / self.daily_start_balance)

    def _handle_new_day(self, balance: float, now: datetime, boundary: datetime):
        """Reset daily tracking at the 21:00 UTC session boundary."""
        if self.current_date is not None and self.daily_start_balance > 0:
            # Record previous day's result
            daily_result = {
                "date": self.current_date.date().isoformat(),
                "start_balance": self.daily_start_balance,
                "end_balance": balance,
                "pnl": balance - self.daily_start_balance,
                "drawdown_pct": max(0, (self.daily_start_balance - balance) / self.daily_start_balance),
            }
            self.daily_drawdowns.append(daily_result)

        self.daily_start_balance = balance
        self.current_date = boundary
        logger.info(f"New session day (boundary {boundary.isoformat()}): daily start balance = R{balance:.2f}")

    def _handle_new_week(self, balance: float, now: datetime, boundary: datetime):
        """Reset weekly tracking at the Friday 21:00 UTC session boundary."""
        self.weekly_start_balance = balance
        self.current_week_start = boundary
        logger.info(f"New session week (boundary {boundary.isoformat()}): weekly start balance = R{balance:.2f}")
