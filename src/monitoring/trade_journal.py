"""
Trade Journal — SQLite-backed persistent log of every trade.

Every trade is recorded with full details for:
- Performance review
- ML model retraining (feedback loop)
- Regulatory compliance
- Personal learning (what works, what doesn't)
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import Config, PROJECT_ROOT

logger = logging.getLogger("traderbot.journal")


class TradeJournal:
    """
    Persistent trade log using SQLite.

    Records every trade with full context including indicator values,
    ML confidence, risk parameters, and outcome. This data feeds
    back into ML retraining.
    """

    def __init__(self, config: Config):
        self.config = config
        db_path = config.get("monitoring.trade_journal_db", "data/trade_logs/trades.db")
        self.db_path = PROJECT_ROOT / db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    instrument TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    units INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    stop_loss REAL,
                    take_profit REAL,
                    pnl_pips REAL,
                    pnl_zar REAL,
                    ml_confidence REAL,
                    exit_reason TEXT,
                    slippage_pips REAL,
                    spread_at_entry REAL,
                    balance_after REAL,
                    model_version TEXT,
                    trend_15min INTEGER,
                    indicators_json TEXT,
                    adjustments_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_summary (
                    date TEXT PRIMARY KEY,
                    starting_balance REAL,
                    ending_balance REAL,
                    trades_count INTEGER,
                    wins INTEGER,
                    losses INTEGER,
                    total_pnl REAL,
                    max_drawdown_pct REAL,
                    notes TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    message TEXT,
                    data_json TEXT
                )
            """)

        logger.info(f"Trade journal initialized: {self.db_path}")

    def record_trade(
        self,
        trade_id: str,
        instrument: str,
        direction: str,
        units: int,
        entry_price: float,
        entry_time: datetime,
        stop_loss: float,
        take_profit: float,
        ml_confidence: float,
        model_version: str = "",
        trend_15min: int = 0,
        indicators: dict = None,
        adjustments: list = None,
        exit_price: float = None,
        exit_time: datetime = None,
        pnl_pips: float = None,
        pnl_zar: float = None,
        exit_reason: str = None,
        slippage_pips: float = 0.0,
        spread_at_entry: float = 0.0,
        balance_after: float = None,
    ):
        """Record a trade (can be called at open and updated at close)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO trades (
                    trade_id, instrument, direction, units, entry_price,
                    exit_price, entry_time, exit_time, stop_loss, take_profit,
                    pnl_pips, pnl_zar, ml_confidence, exit_reason,
                    slippage_pips, spread_at_entry, balance_after,
                    model_version, trend_15min, indicators_json, adjustments_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_id, instrument, direction, units, entry_price,
                exit_price,
                entry_time.isoformat() if isinstance(entry_time, datetime) else str(entry_time),
                exit_time.isoformat() if isinstance(exit_time, datetime) else str(exit_time) if exit_time else None,
                stop_loss, take_profit,
                pnl_pips, pnl_zar, ml_confidence, exit_reason,
                slippage_pips, spread_at_entry, balance_after,
                model_version, trend_15min,
                json.dumps(indicators) if indicators else None,
                json.dumps(adjustments) if adjustments else None,
            ))

    def record_event(self, event_type: str, message: str, data: dict = None):
        """Record a system event (circuit breaker, retrain, milestone, etc.)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO events (timestamp, event_type, message, data_json)
                VALUES (?, ?, ?, ?)
            """, (
                datetime.now(timezone.utc).isoformat(),
                event_type,
                message,
                json.dumps(data) if data else None,
            ))

    def record_daily_summary(
        self,
        date: str,
        starting_balance: float,
        ending_balance: float,
        trades_count: int,
        wins: int,
        losses: int,
        total_pnl: float,
        max_drawdown_pct: float = 0.0,
        notes: str = "",
    ):
        """Record end-of-day summary."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO daily_summary
                (date, starting_balance, ending_balance, trades_count,
                 wins, losses, total_pnl, max_drawdown_pct, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                date, starting_balance, ending_balance, trades_count,
                wins, losses, total_pnl, max_drawdown_pct, notes,
            ))

    def get_trades(
        self,
        instrument: str = None,
        direction: str = None,
        since: str = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Query trades with optional filters."""
        query = "SELECT * FROM trades WHERE 1=1"
        params = []

        if instrument:
            query += " AND instrument = ?"
            params.append(instrument)
        if direction:
            query += " AND direction = ?"
            params.append(direction)
        if since:
            query += " AND entry_time >= ?"
            params.append(since)

        query += " ORDER BY entry_time DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)

    def get_all_trades_df(self) -> pd.DataFrame:
        """Get all trades as a DataFrame."""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(
                "SELECT * FROM trades ORDER BY entry_time", conn
            )

    def get_daily_summaries(self, limit: int = 30) -> pd.DataFrame:
        """Get recent daily summaries."""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(
                "SELECT * FROM daily_summary ORDER BY date DESC LIMIT ?",
                conn, params=(limit,)
            )

    def get_events(self, event_type: str = None, limit: int = 50) -> pd.DataFrame:
        """Get system events."""
        query = "SELECT * FROM events"
        params = []
        if event_type:
            query += " WHERE event_type = ?"
            params.append(event_type)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)

    def get_trade_count(self) -> int:
        """Get total number of completed trades."""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE exit_price IS NOT NULL"
            ).fetchone()
            return result[0] if result else 0
