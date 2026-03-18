"""
Performance Tracker — Calculates comprehensive trading performance metrics.

Provides real-time and historical performance analysis for the dashboard
and decision-making.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.monitoring.trade_journal import TradeJournal

logger = logging.getLogger("traderbot.performance")


class PerformanceTracker:
    """
    Calculates trading performance metrics from the trade journal.

    Metrics:
    - Win rate, profit factor, Sharpe ratio
    - Equity curve and drawdown
    - Per-instrument breakdown
    - Per-hour and per-day analysis
    - Average R:R achieved
    """

    def __init__(self, journal: TradeJournal):
        self.journal = journal

    def get_summary(self) -> dict:
        """Get comprehensive performance summary."""
        df = self.journal.get_all_trades_df()
        completed = df[df["exit_price"].notna()].copy()

        if completed.empty:
            return self._empty_summary()

        pnls = completed["pnl_zar"].fillna(0)
        wins = completed[completed["pnl_zar"] > 0]
        losses = completed[completed["pnl_zar"] <= 0]

        gross_profit = wins["pnl_zar"].sum() if not wins.empty else 0
        gross_loss = abs(losses["pnl_zar"].sum()) if not losses.empty else 0

        return {
            "total_trades": len(completed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(completed) if len(completed) > 0 else 0,
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
            "total_pnl": float(pnls.sum()),
            "avg_pnl": float(pnls.mean()),
            "avg_win": float(wins["pnl_zar"].mean()) if not wins.empty else 0,
            "avg_loss": float(losses["pnl_zar"].mean()) if not losses.empty else 0,
            "largest_win": float(pnls.max()),
            "largest_loss": float(pnls.min()),
            "gross_profit": float(gross_profit),
            "gross_loss": float(gross_loss),
            "sharpe_ratio": self._sharpe_ratio(pnls),
            "max_drawdown_pct": self._max_drawdown_pct(completed),
            "avg_hold_minutes": self._avg_hold_time(completed),
            "best_instrument": self._best_instrument(completed),
            "exit_reasons": self._exit_reason_breakdown(completed),
        }

    def get_equity_curve(self, starting_balance: float = 500) -> pd.DataFrame:
        """Build equity curve DataFrame."""
        df = self.journal.get_all_trades_df()
        completed = df[df["exit_price"].notna()].copy()

        if completed.empty:
            return pd.DataFrame(columns=["time", "balance"])

        completed = completed.sort_values("exit_time")
        balance = starting_balance
        curve = [{"time": completed.iloc[0]["entry_time"], "balance": balance}]

        for _, trade in completed.iterrows():
            balance += trade["pnl_zar"] if pd.notna(trade["pnl_zar"]) else 0
            curve.append({"time": trade["exit_time"], "balance": balance})

        return pd.DataFrame(curve)

    def get_drawdown_series(self, starting_balance: float = 500) -> pd.DataFrame:
        """Build drawdown percentage series."""
        equity = self.get_equity_curve(starting_balance)
        if equity.empty:
            return pd.DataFrame(columns=["time", "drawdown_pct"])

        equity["peak"] = equity["balance"].expanding().max()
        equity["drawdown_pct"] = (equity["balance"] - equity["peak"]) / equity["peak"] * 100

        return equity[["time", "drawdown_pct"]]

    def get_instrument_breakdown(self) -> pd.DataFrame:
        """Performance breakdown by instrument."""
        df = self.journal.get_all_trades_df()
        completed = df[df["exit_price"].notna()].copy()

        if completed.empty:
            return pd.DataFrame()

        grouped = completed.groupby("instrument").agg(
            trades=("pnl_zar", "count"),
            wins=("pnl_zar", lambda x: (x > 0).sum()),
            total_pnl=("pnl_zar", "sum"),
            avg_pnl=("pnl_zar", "mean"),
        ).reset_index()

        grouped["win_rate"] = grouped["wins"] / grouped["trades"]
        return grouped

    def get_hourly_breakdown(self) -> pd.DataFrame:
        """Performance breakdown by hour of day."""
        df = self.journal.get_all_trades_df()
        completed = df[df["exit_price"].notna()].copy()

        if completed.empty:
            return pd.DataFrame()

        completed["hour"] = pd.to_datetime(completed["entry_time"]).dt.hour
        grouped = completed.groupby("hour").agg(
            trades=("pnl_zar", "count"),
            wins=("pnl_zar", lambda x: (x > 0).sum()),
            total_pnl=("pnl_zar", "sum"),
        ).reset_index()

        grouped["win_rate"] = grouped["wins"] / grouped["trades"]
        return grouped

    def _sharpe_ratio(self, pnls: pd.Series) -> float:
        """Calculate annualized Sharpe ratio from trade PnLs."""
        if len(pnls) < 2 or pnls.std() == 0:
            return 0.0
        # Approximate: assume ~20 trades per day, 252 trading days
        trades_per_year = 20 * 252
        return float(pnls.mean() / pnls.std() * np.sqrt(trades_per_year))

    def _max_drawdown_pct(self, df: pd.DataFrame) -> float:
        """Calculate maximum drawdown percentage."""
        if df.empty or "balance_after" not in df.columns:
            return 0.0

        balances = df["balance_after"].dropna()
        if balances.empty:
            return 0.0

        peak = balances.expanding().max()
        drawdown = (balances - peak) / peak
        return float(abs(drawdown.min())) * 100 if not drawdown.empty else 0.0

    def _avg_hold_time(self, df: pd.DataFrame) -> float:
        """Average hold time in minutes."""
        if df.empty:
            return 0.0

        try:
            entry = pd.to_datetime(df["entry_time"])
            exit_ = pd.to_datetime(df["exit_time"])
            hold = (exit_ - entry).dt.total_seconds() / 60
            return float(hold.mean())
        except Exception:
            return 0.0

    def _best_instrument(self, df: pd.DataFrame) -> str:
        """Instrument with highest total PnL."""
        if df.empty:
            return "N/A"
        by_inst = df.groupby("instrument")["pnl_zar"].sum()
        return str(by_inst.idxmax()) if not by_inst.empty else "N/A"

    def _exit_reason_breakdown(self, df: pd.DataFrame) -> dict:
        """Count of trades by exit reason."""
        if df.empty or "exit_reason" not in df.columns:
            return {}
        return df["exit_reason"].value_counts().to_dict()

    @staticmethod
    def _empty_summary() -> dict:
        return {
            "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
            "profit_factor": 0, "total_pnl": 0, "avg_pnl": 0,
            "avg_win": 0, "avg_loss": 0, "largest_win": 0, "largest_loss": 0,
            "gross_profit": 0, "gross_loss": 0, "sharpe_ratio": 0,
            "max_drawdown_pct": 0, "avg_hold_minutes": 0,
            "best_instrument": "N/A", "exit_reasons": {},
        }
