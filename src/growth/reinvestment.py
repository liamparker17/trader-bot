"""
Growth & Reinvestment — Manages profit reinvestment and account scaling.

Two phases:
  Phase 1 (Growth): R500 → R6,000
    - 100% of profits reinvested
    - Position sizes grow automatically (percentage-based)

  Phase 2 (Harvest): R6,000+
    - 50% of monthly profits withdrawn
    - 50% reinvested for continued growth
    - If drawdown occurs, temporarily switch to 100% reinvest
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from src.config import Config

logger = logging.getLogger("traderbot.growth")


class GrowthManager:
    """
    Manages reinvestment strategy and tracks growth phase transitions.

    Since position sizing is percentage-based (1.5% of current balance),
    reinvestment happens automatically — profits increase the balance,
    which increases trade sizes. This module tracks the phases and
    calculates withdrawal amounts when in harvest mode.
    """

    def __init__(self, config: Config):
        self.config = config
        self.target_balance = config.get("growth.target_balance", 6000)
        self.reinvest_growth = config.get("growth.reinvest_pct_growth_phase", 100) / 100
        self.reinvest_harvest = config.get("growth.reinvest_pct_harvest_phase", 50) / 100

        # State
        self.current_phase: str = "growth"  # "growth" or "harvest"
        self.month_start_balance: float = 0.0
        self.total_withdrawn: float = 0.0
        self.withdrawal_history: list[dict] = []

    def initialize(self, balance: float):
        """Set initial balance. Call at startup."""
        self.month_start_balance = balance
        self.current_phase = "growth" if balance < self.target_balance else "harvest"
        logger.info(
            f"Growth manager initialized | Phase: {self.current_phase} | "
            f"Balance: R{balance:.2f} | Target: R{self.target_balance}"
        )

    def update(self, current_balance: float) -> dict:
        """
        Check for phase transitions and calculate any pending withdrawals.

        Call periodically or after each trade.

        Returns:
            Dict with current growth status.
        """
        old_phase = self.current_phase

        if current_balance >= self.target_balance:
            self.current_phase = "harvest"
        else:
            self.current_phase = "growth"

        # Detect phase transition
        if old_phase == "growth" and self.current_phase == "harvest":
            logger.info(
                f"TARGET REACHED: R{current_balance:.2f} >= R{self.target_balance} | "
                f"Switching to harvest mode (50/50 split)"
            )

        progress = (current_balance / self.target_balance) * 100
        growth_multiple = current_balance / 500 if current_balance > 0 else 0

        return {
            "phase": self.current_phase,
            "balance": current_balance,
            "target": self.target_balance,
            "progress_pct": min(progress, 100),
            "growth_multiple": growth_multiple,
            "reinvest_pct": self.reinvest_growth if self.current_phase == "growth" else self.reinvest_harvest,
            "total_withdrawn": self.total_withdrawn,
        }

    def calculate_monthly_withdrawal(self, current_balance: float) -> float:
        """
        Calculate how much can be withdrawn at month end.

        Only applies in harvest phase.
        Returns 0 in growth phase.
        """
        if self.current_phase != "harvest":
            return 0.0

        monthly_profit = current_balance - self.month_start_balance
        if monthly_profit <= 0:
            return 0.0

        withdrawal = monthly_profit * (1 - self.reinvest_harvest)

        # Don't withdraw if it would drop below target
        if current_balance - withdrawal < self.target_balance:
            withdrawal = max(0, current_balance - self.target_balance)

        return withdrawal

    def record_withdrawal(self, amount: float, current_balance: float):
        """Record a withdrawal."""
        self.total_withdrawn += amount
        self.withdrawal_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "amount": amount,
            "balance_after": current_balance - amount,
        })
        logger.info(
            f"Withdrawal recorded: R{amount:.2f} | "
            f"Total withdrawn: R{self.total_withdrawn:.2f}"
        )

    def reset_month(self, current_balance: float):
        """Reset monthly tracking. Call at start of each month."""
        self.month_start_balance = current_balance
        logger.info(f"Monthly reset | Start balance: R{current_balance:.2f}")

    def get_reinvestment_rate(self) -> float:
        """Get current reinvestment rate (0.0 to 1.0)."""
        return self.reinvest_growth if self.current_phase == "growth" else self.reinvest_harvest
