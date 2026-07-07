"""
Performance Tracker — Calculates comprehensive trading performance metrics.

Provides real-time and historical performance analysis for the dashboard
and decision-making.
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from src.monitoring.trade_journal import TradeJournal

logger = logging.getLogger("traderbot.performance")

# Sentinel "since forever" timestamp for cumulative manager-cost queries.
_EPOCH_ISO = "1970-01-01T00:00:00+00:00"


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
            "api_cost_zar": self._manager_cost(),
            "net_pnl_after_api": self.net_pnl_after_api(),
        }

    # ------------------------------------------------------------------
    # Self-funding scorecard (Task 14)
    # ------------------------------------------------------------------

    def _realized_net_pnl(self, days: Optional[int] = None) -> float:
        """Sum of net_pnl_zar (fallback: pnl_zar) over completed trades,
        optionally restricted to exits within the trailing `days` window."""
        df = self.journal.get_all_trades_df()
        completed = df[df["exit_price"].notna()].copy()
        if completed.empty:
            return 0.0
        if days is not None:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            completed = completed[completed["exit_time"] >= cutoff]
        if completed.empty:
            return 0.0
        net = completed["net_pnl_zar"].fillna(completed["pnl_zar"]).fillna(0)
        return float(net.sum())

    def _manager_cost(self, days: Optional[int] = None) -> float:
        """Sum of manager_log.cost_zar, cumulative or over trailing `days`."""
        if days is not None:
            since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        else:
            since = _EPOCH_ISO
        try:
            return float(self.journal.manager_cost_since(since))
        except Exception as e:  # manager_log may predate Task 12 schemas
            logger.warning(f"manager cost query failed: {e}")
            return 0.0

    def net_pnl_after_api(self, days: Optional[int] = None) -> float:
        """
        Self-funding headline number: realized net P&L minus the Claude
        manager's API cost, cumulative or over the trailing `days` window.
        """
        return self._realized_net_pnl(days) - self._manager_cost(days)

    def manager_stats_since(self, since: datetime) -> dict:
        """
        Manager activity since `since` (normally today's 21:00 UTC session
        boundary) for the daily Telegram summary: cycles run, adjustments
        applied, API cost.
        """
        stats = {"cycles": 0, "adjustments_applied": 0, "api_cost_zar": 0.0}
        try:
            df = self.journal.get_manager_log()
        except Exception as e:
            logger.warning(f"manager_stats_since query failed: {e}")
            return stats
        if df.empty:
            return stats
        window = df[df["ts_utc"] >= since.isoformat()]
        stats["cycles"] = int(len(window))
        stats["api_cost_zar"] = float(window["cost_zar"].fillna(0).sum())
        applied_total = 0
        for raw in window["applied_json"].fillna(""):
            try:
                entries = json.loads(raw) if raw else []
                applied_total += len(entries) if isinstance(entries, list) else 0
            except (json.JSONDecodeError, TypeError):
                pass
        stats["adjustments_applied"] = applied_total
        return stats

    def days_since_first_manager_cycle(self, now: Optional[datetime] = None) -> Optional[int]:
        """Whole days since the first manager_log row; None if no cycles yet."""
        try:
            with sqlite3.connect(self.journal.db_path) as conn:
                row = conn.execute("SELECT MIN(ts_utc) FROM manager_log").fetchone()
        except sqlite3.Error as e:
            logger.warning(f"first-manager-cycle query failed: {e}")
            return None
        if not row or not row[0]:
            return None
        first = datetime.fromisoformat(str(row[0]))
        now = now or datetime.now(timezone.utc)
        return max(0, (now - first).days)

    def justification_report(
        self,
        heuristic_baseline_pnl_zar: Optional[float] = None,
    ) -> dict:
        """
        Day-10 self-funding verdict (API budget amendment):

          SELF-FUNDING  — net-after-cost > 0 AND the P&L uplift over the
                          heuristic baseline exceeds the API cost.
          NOT JUSTIFIED — either condition fails, or no heuristic baseline
                          is available to prove uplift (conservative).
          PENDING       — no manager cycles logged yet.

        `heuristic_baseline_pnl_zar` is the same-window P&L of the
        no-API heuristic manager (from the Task 15/16 managed backtest).
        """
        days_running = self.days_since_first_manager_cycle()
        net_pnl = self._realized_net_pnl()
        api_cost = self._manager_cost()
        report = {
            "net_pnl_zar": net_pnl,
            "api_cost_zar": api_cost,
            "net_after_cost_zar": net_pnl - api_cost,
            "heuristic_baseline_pnl_zar": heuristic_baseline_pnl_zar,
            "days_since_first_cycle": days_running,
        }
        if days_running is None:
            report["verdict"] = "PENDING"
            report["reason"] = "no manager cycles logged yet"
            return report
        if report["net_after_cost_zar"] <= 0:
            report["verdict"] = "NOT JUSTIFIED"
            report["reason"] = "net-after-cost P&L is not positive"
            return report
        if heuristic_baseline_pnl_zar is None:
            report["verdict"] = "NOT JUSTIFIED"
            report["reason"] = (
                "no heuristic baseline available to prove uplift — "
                "run the managed backtest comparison (tb manager --verdict --baseline-pnl X)"
            )
            return report
        uplift = net_pnl - heuristic_baseline_pnl_zar
        report["uplift_zar"] = uplift
        if uplift > api_cost:
            report["verdict"] = "SELF-FUNDING"
            report["reason"] = (
                f"net-after-cost R{report['net_after_cost_zar']:.2f} > 0 and "
                f"uplift R{uplift:.2f} exceeds API cost R{api_cost:.2f}"
            )
        else:
            report["verdict"] = "NOT JUSTIFIED"
            report["reason"] = (
                f"uplift R{uplift:.2f} does not exceed API cost R{api_cost:.2f}"
            )
        return report

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
            "api_cost_zar": 0.0, "net_pnl_after_api": 0.0,
        }
