"""
ML Evaluator — Tracks live model performance and triggers retraining.

Monitors the running model's predictions against actual trade outcomes.
When performance degrades beyond thresholds, flags for retraining.

Also calculates trading-specific metrics like profit factor and
expected value per trade.
"""

import json
import logging
from collections import deque
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import Config, PROJECT_ROOT

logger = logging.getLogger("traderbot.ml.evaluator")


class TradeRecord:
    """A single trade outcome for evaluation."""

    def __init__(
        self,
        prediction: float,
        predicted_action: str,
        actual_outcome: int,  # 1 = win, 0 = loss
        pnl: float,  # Profit/loss in account currency
        timestamp: str = None,
    ):
        self.prediction = prediction
        self.predicted_action = predicted_action
        self.actual_outcome = actual_outcome
        self.pnl = pnl
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()


class Evaluator:
    """
    Tracks model performance over time and triggers retraining.

    Metrics tracked:
    - Accuracy: % of predictions that were correct
    - Precision: % of "trade" signals that were actually profitable
    - Recall: % of profitable setups that the model identified
    - Profit factor: gross profit / gross loss
    - Win rate: % of executed trades that were profitable
    - Expected value: average PnL per trade
    - Calibration: does 70% confidence actually win 70% of the time?

    Retraining triggers:
    - Win rate below threshold over N trades
    - Accuracy dropped significantly from training metrics
    - N trades completed since last retrain
    - Time elapsed since last retrain
    """

    def __init__(self, config: Config):
        self.config = config

        ml_config = config.get("ml", {})
        risk_config = config.get("risk", {})

        # Retraining triggers
        self.retrain_trade_interval = ml_config.get("retrain_trade_interval", 500)
        self.retrain_time_days = ml_config.get("retrain_time_interval_days", 14)
        self.min_win_rate = risk_config.get("min_win_rate_threshold", 0.45)
        self.win_rate_lookback = risk_config.get("min_win_rate_lookback", 100)

        # Trade history
        self.trades: deque[TradeRecord] = deque(maxlen=5000)
        self.trades_since_retrain: int = 0
        self.last_retrain_time: Optional[datetime] = None
        self.model_version: Optional[str] = None

        # Persistence
        self._log_path = PROJECT_ROOT / "data" / "trade_logs" / "evaluator_state.json"

    def record_trade(self, trade: TradeRecord):
        """Record a completed trade for evaluation."""
        self.trades.append(trade)
        self.trades_since_retrain += 1

    def record_from_dict(self, trade_data: dict):
        """Record a trade from a dictionary (convenience method)."""
        self.record_trade(TradeRecord(
            prediction=trade_data.get("ml_confidence", 0.0),
            predicted_action=trade_data.get("signal_action", "trade"),
            actual_outcome=1 if trade_data.get("pnl_zar", 0) > 0 else 0,
            pnl=trade_data.get("pnl_zar", 0.0),
            timestamp=trade_data.get("timestamp_close", None),
        ))

    def get_metrics(self, last_n: int = None) -> dict:
        """
        Calculate comprehensive performance metrics.

        Args:
            last_n: Only consider last N trades. None = all trades.

        Returns:
            Dict of metric_name → value.
        """
        trades = list(self.trades)
        if last_n is not None:
            trades = trades[-last_n:]

        if not trades:
            return self._empty_metrics()

        outcomes = [t.actual_outcome for t in trades]
        predictions = [t.prediction for t in trades]
        pnls = [t.pnl for t in trades]

        wins = sum(outcomes)
        losses = len(outcomes) - wins
        win_rate = wins / len(outcomes) if outcomes else 0

        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        total_pnl = sum(pnls)
        avg_pnl = total_pnl / len(pnls) if pnls else 0
        avg_win = gross_profit / wins if wins > 0 else 0
        avg_loss = gross_loss / losses if losses > 0 else 0

        # Accuracy: did the model correctly predict win/loss?
        correct = sum(
            1 for t in trades
            if (t.prediction >= 0.5 and t.actual_outcome == 1)
            or (t.prediction < 0.5 and t.actual_outcome == 0)
        )
        accuracy = correct / len(trades) if trades else 0

        # Confidence calibration (binned)
        calibration = self._calculate_calibration(trades)

        return {
            "total_trades": len(trades),
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "accuracy": accuracy,
            "profit_factor": profit_factor,
            "total_pnl": total_pnl,
            "avg_pnl_per_trade": avg_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "calibration": calibration,
            "trades_since_retrain": self.trades_since_retrain,
        }

    def should_retrain(self) -> tuple[bool, str]:
        """
        Check if the model should be retrained.

        Returns:
            (should_retrain, reason)
        """
        # Check trade count trigger
        if self.trades_since_retrain >= self.retrain_trade_interval:
            return True, (
                f"Trade count trigger: {self.trades_since_retrain} trades "
                f"since last retrain (threshold: {self.retrain_trade_interval})"
            )

        # Check time trigger
        if self.last_retrain_time:
            days_since = (datetime.now(timezone.utc) - self.last_retrain_time).days
            if days_since >= self.retrain_time_days:
                return True, (
                    f"Time trigger: {days_since} days since last retrain "
                    f"(threshold: {self.retrain_time_days})"
                )

        # Check win rate degradation
        if len(self.trades) >= self.win_rate_lookback:
            recent = list(self.trades)[-self.win_rate_lookback:]
            recent_win_rate = sum(t.actual_outcome for t in recent) / len(recent)
            if recent_win_rate < self.min_win_rate:
                return True, (
                    f"Win rate trigger: {recent_win_rate:.1%} over last "
                    f"{self.win_rate_lookback} trades "
                    f"(threshold: {self.min_win_rate:.1%})"
                )

        return False, "No retrain needed"

    def mark_retrained(self, model_version: str):
        """Reset counters after a successful retrain."""
        self.trades_since_retrain = 0
        self.last_retrain_time = datetime.now(timezone.utc)
        self.model_version = model_version
        logger.info(f"Evaluator reset for model {model_version}")

    def get_recent_win_rate(self, n: int = 100) -> float:
        """Get win rate over last N trades."""
        recent = list(self.trades)[-n:]
        if not recent:
            return 0.0
        return sum(t.actual_outcome for t in recent) / len(recent)

    def get_consecutive_losses(self) -> int:
        """Count current consecutive losing streak."""
        count = 0
        for trade in reversed(self.trades):
            if trade.actual_outcome == 0:
                count += 1
            else:
                break
        return count

    def generate_report(self) -> str:
        """Generate a human-readable performance report."""
        all_metrics = self.get_metrics()
        recent_metrics = self.get_metrics(last_n=100)

        lines = [
            "=" * 50,
            "ML MODEL PERFORMANCE REPORT",
            "=" * 50,
            f"Model version: {self.model_version or 'unknown'}",
            f"Total trades evaluated: {all_metrics['total_trades']}",
            f"Trades since retrain: {self.trades_since_retrain}",
            "",
            "--- ALL TIME ---",
            f"Win rate: {all_metrics['win_rate']:.1%}",
            f"Accuracy: {all_metrics['accuracy']:.1%}",
            f"Profit factor: {all_metrics['profit_factor']:.2f}",
            f"Total PnL: R{all_metrics['total_pnl']:.2f}",
            f"Avg PnL/trade: R{all_metrics['avg_pnl_per_trade']:.2f}",
            f"Avg win: R{all_metrics['avg_win']:.2f}",
            f"Avg loss: R{all_metrics['avg_loss']:.2f}",
            "",
            "--- LAST 100 TRADES ---",
            f"Win rate: {recent_metrics['win_rate']:.1%}",
            f"Accuracy: {recent_metrics['accuracy']:.1%}",
            f"Profit factor: {recent_metrics['profit_factor']:.2f}",
            f"Total PnL: R{recent_metrics['total_pnl']:.2f}",
            "",
        ]

        # Retraining status
        should, reason = self.should_retrain()
        lines.append(f"Retrain needed: {'YES' if should else 'NO'}")
        if should:
            lines.append(f"Reason: {reason}")

        lines.append("=" * 50)
        return "\n".join(lines)

    def save_state(self):
        """Save evaluator state to disk."""
        state = {
            "trades_since_retrain": self.trades_since_retrain,
            "last_retrain_time": self.last_retrain_time.isoformat() if self.last_retrain_time else None,
            "model_version": self.model_version,
            "total_trades": len(self.trades),
        }
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._log_path, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self):
        """Load evaluator state from disk."""
        if self._log_path.exists():
            with open(self._log_path) as f:
                state = json.load(f)
            self.trades_since_retrain = state.get("trades_since_retrain", 0)
            if state.get("last_retrain_time"):
                self.last_retrain_time = datetime.fromisoformat(state["last_retrain_time"])
            self.model_version = state.get("model_version")

    def _calculate_calibration(self, trades: list[TradeRecord]) -> dict:
        """
        Check if model confidence correlates with actual outcomes.

        Bins predictions into buckets and checks actual win rate per bucket.
        Good calibration: 70% confidence → ~70% win rate.
        """
        bins = [(0.5, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
                (0.70, 0.80), (0.80, 1.0)]
        calibration = {}

        for low, high in bins:
            bucket_trades = [t for t in trades if low <= t.prediction < high]
            if bucket_trades:
                actual_win_rate = sum(t.actual_outcome for t in bucket_trades) / len(bucket_trades)
                calibration[f"{low:.2f}-{high:.2f}"] = {
                    "count": len(bucket_trades),
                    "actual_win_rate": actual_win_rate,
                    "expected_midpoint": (low + high) / 2,
                }

        return calibration

    @staticmethod
    def _empty_metrics() -> dict:
        return {
            "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
            "accuracy": 0, "profit_factor": 0, "total_pnl": 0,
            "avg_pnl_per_trade": 0, "avg_win": 0, "avg_loss": 0,
            "gross_profit": 0, "gross_loss": 0, "calibration": {},
            "trades_since_retrain": 0,
        }
