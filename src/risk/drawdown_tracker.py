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
from typing import Optional

from src.config import Config

logger = logging.getLogger("traderbot.risk.drawdown")


class DrawdownTracker:
    """
    Tracks drawdown at multiple time horizons.

    Provides real-time drawdown checks that the risk manager uses
    to decide whether to pause trading.
    """

    def __init__(self, config: Config):
        self.config = config
        self.daily_limit = config.get("risk.daily_drawdown_limit_pct", 4.0) / 100
        self.weekly_limit = config.get("risk.weekly_drawdown_limit_pct", 8.0) / 100
        self.hard_floor = config.get("account.hard_floor_zar", 350)

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
        now = datetime.now(timezone.utc)
        self.current_date = now
        self.current_week_start = now - timedelta(days=now.weekday())
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
        now = datetime.now(timezone.utc)

        # Check for new day
        if self.current_date is None or now.date() != self.current_date.date():
            self._handle_new_day(current_balance, now)

        # Check for new week
        week_start = now - timedelta(days=now.weekday())
        if self.current_week_start is None or week_start.date() != self.current_week_start.date():
            self._handle_new_week(current_balance, now)

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

        # Hard floor
        if current_balance <= self.hard_floor:
            violations.append(
                f"Balance R{current_balance:.2f} <= hard floor R{self.hard_floor}"
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

    def _handle_new_day(self, balance: float, now: datetime):
        """Reset daily tracking."""
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
        self.current_date = now
        logger.info(f"New day: daily start balance = R{balance:.2f}")

    def _handle_new_week(self, balance: float, now: datetime):
        """Reset weekly tracking."""
        self.weekly_start_balance = balance
        self.current_week_start = now - timedelta(days=now.weekday())
        logger.info(f"New week: weekly start balance = R{balance:.2f}")
