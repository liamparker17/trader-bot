"""
Milestone Tracker — Tracks and celebrates account balance milestones.

Milestones provide motivation and natural checkpoints for reviewing
strategy performance. Each milestone triggers a Telegram alert and
a log entry.
"""

import logging
from datetime import datetime, timezone
from typing import Callable, Optional

from src.config import Config

logger = logging.getLogger("traderbot.growth.milestones")


class MilestoneTracker:
    """
    Tracks balance milestones and triggers alerts when reached.

    Default milestones: R750, R1000, R2000, R3000, R6000
    """

    def __init__(self, config: Config, on_milestone: Optional[Callable] = None):
        """
        Args:
            config: Bot configuration
            on_milestone: Callback(milestone_zar, balance, message) when milestone hit
        """
        self.config = config
        self.on_milestone = on_milestone

        milestones_config = config.get("growth.milestones", [750, 1000, 2000, 3000, 6000])
        self.milestones: list[float] = sorted(milestones_config)
        self.reached: set[float] = set()
        self.milestone_history: list[dict] = []
        self.starting_balance = config.get("account.starting_balance", 500)

    def check(self, current_balance: float):
        """
        Check if any new milestones have been reached.
        Call after every trade or balance update.
        """
        for milestone in self.milestones:
            if milestone in self.reached:
                continue

            if current_balance >= milestone:
                self._milestone_reached(milestone, current_balance)

    def _milestone_reached(self, milestone: float, balance: float):
        """Handle a newly reached milestone."""
        self.reached.add(milestone)

        growth = balance / self.starting_balance
        messages = {
            750: f"50% growth achieved! R{balance:.2f} ({growth:.1f}x)",
            1000: f"Account DOUBLED! R{balance:.2f} ({growth:.1f}x)",
            2000: f"4x growth! R{balance:.2f} ({growth:.1f}x)",
            3000: f"Halfway to target! R{balance:.2f} ({growth:.1f}x)",
            6000: f"TARGET REACHED! R{balance:.2f} ({growth:.1f}x) — Switching to harvest mode",
        }

        message = messages.get(
            milestone,
            f"Milestone R{milestone:.0f} reached! Balance: R{balance:.2f} ({growth:.1f}x)"
        )

        self.milestone_history.append({
            "milestone": milestone,
            "balance": balance,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": message,
        })

        logger.info(f"MILESTONE: {message}")

        if self.on_milestone:
            self.on_milestone(milestone, balance, message)

    def get_next_milestone(self, current_balance: float) -> Optional[float]:
        """Get the next unreached milestone."""
        for milestone in self.milestones:
            if milestone > current_balance and milestone not in self.reached:
                return milestone
        return None

    def get_progress_to_next(self, current_balance: float) -> dict:
        """Get progress toward the next milestone."""
        next_ms = self.get_next_milestone(current_balance)
        if next_ms is None:
            return {"next_milestone": None, "progress_pct": 100, "remaining": 0}

        # Find the previous milestone or starting balance
        prev = self.starting_balance
        for ms in self.milestones:
            if ms < next_ms and ms <= current_balance:
                prev = ms

        total_range = next_ms - prev
        current_progress = current_balance - prev
        pct = (current_progress / total_range * 100) if total_range > 0 else 0

        return {
            "next_milestone": next_ms,
            "progress_pct": min(pct, 100),
            "remaining": max(0, next_ms - current_balance),
        }
