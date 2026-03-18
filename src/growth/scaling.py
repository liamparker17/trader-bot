"""
Scaling — Adapts trading behavior as account balance grows.

As the account grows from R500 to R6000+, several things change:
- Absolute risk per trade increases (1.5% of a larger number)
- More margin available → can hold more positions
- Can consider additional instruments
- Trade frequency may adjust based on available margin

This module provides scaling recommendations at each balance level.
"""

import logging

from src.config import Config

logger = logging.getLogger("traderbot.growth.scaling")


class ScalingAdvisor:
    """
    Provides scaling recommendations based on current account balance.

    This is advisory — the risk manager still enforces all limits.
    The scaling advisor suggests when to adjust parameters.
    """

    def __init__(self, config: Config):
        self.config = config
        self.starting_balance = config.get("account.starting_balance", 500)

    def get_recommendations(self, balance: float) -> dict:
        """
        Get scaling recommendations for current balance.

        Returns dict with recommended parameters and explanations.
        """
        growth = balance / self.starting_balance if self.starting_balance > 0 else 1

        # Base recommendations
        rec = {
            "balance": balance,
            "growth_multiple": growth,
            "max_open_positions": self._recommended_max_positions(balance),
            "max_trades_per_day": self._recommended_max_trades(balance),
            "instruments": self._recommended_instruments(balance),
            "risk_per_trade_pct": 1.5,  # Fixed — never change this
            "notes": [],
        }

        # Balance-specific notes
        if balance < 400:
            rec["notes"].append("CRITICAL: Balance near hard floor. Ultra-conservative mode.")
            rec["max_open_positions"] = 1
            rec["max_trades_per_day"] = 10

        elif balance < 750:
            rec["notes"].append("Early stage: focus on consistency over growth.")

        elif balance < 2000:
            rec["notes"].append("Growth stage: strategy is working. Stay disciplined.")

        elif balance < 6000:
            rec["notes"].append("Advanced growth: consider ML model upgrade (Phase 2).")

        else:
            rec["notes"].append("Target reached: harvest mode active.")
            rec["notes"].append("Consider adding reinforcement learning (Phase 3).")

        return rec

    def _recommended_max_positions(self, balance: float) -> int:
        """Scale max open positions with balance."""
        if balance < 400:
            return 1
        elif balance < 1000:
            return 2
        elif balance < 3000:
            return 3
        elif balance < 6000:
            return 4
        else:
            return 5

    def _recommended_max_trades(self, balance: float) -> int:
        """Scale max daily trades with balance."""
        if balance < 400:
            return 10
        elif balance < 1000:
            return 30
        elif balance < 3000:
            return 45
        else:
            return 60

    def _recommended_instruments(self, balance: float) -> list[str]:
        """Recommend which instruments to trade based on balance."""
        # Start with the most liquid, lowest spread pair
        if balance < 750:
            return ["EUR_USD"]
        elif balance < 1500:
            return ["EUR_USD", "USD_JPY"]
        elif balance < 3000:
            return ["EUR_USD", "GBP_USD", "USD_JPY"]
        else:
            return ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD"]
