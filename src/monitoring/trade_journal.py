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
                    commission REAL DEFAULT 0,
                    swap REAL DEFAULT 0,
                    net_pnl_zar REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            self._migrate_fee_columns(conn)

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

            # Task 8: control queue (tb CLI) command audit trail. A row is
            # inserted with outcome='pending' when a command is received,
            # then updated in place once processed.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS control_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_utc TEXT NOT NULL,
                    verb TEXT NOT NULL,
                    args_json TEXT,
                    reason TEXT,
                    requested_by TEXT,
                    before_config_json TEXT,
                    after_config_json TEXT,
                    outcome TEXT NOT NULL DEFAULT 'pending'
                )
            """)

        logger.info(f"Trade journal initialized: {self.db_path}")

    def _migrate_fee_columns(self, conn: sqlite3.Connection):
        """
        Idempotently add fee/net-P&L columns to `trades` for DBs created
        before Task 7. Guarded by PRAGMA table_info so it's safe to run
        on every open, including already-migrated (or brand-new) DBs.
        """
        existing = {row[1] for row in conn.execute("PRAGMA table_info(trades)").fetchall()}
        if "commission" not in existing:
            conn.execute("ALTER TABLE trades ADD COLUMN commission REAL DEFAULT 0")
        if "swap" not in existing:
            conn.execute("ALTER TABLE trades ADD COLUMN swap REAL DEFAULT 0")
        if "net_pnl_zar" not in existing:
            conn.execute("ALTER TABLE trades ADD COLUMN net_pnl_zar REAL")

    @staticmethod
    def compute_net_pnl(gross_pnl: float, commission: float = 0.0, swap: float = 0.0) -> float:
        """
        Net P&L = gross + commission + swap.

        Sign convention: MT5 reports both `commission` and `swap` on deal
        objects as already-signed values, and in the overwhelming majority
        of cases both are costs and therefore negative. Adding them to the
        gross P&L yields the true net result without any extra negation.
        """
        return gross_pnl + commission + swap

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
        commission: float = 0.0,
        swap: float = 0.0,
    ):
        """Record a trade (can be called at open and updated at close)."""
        net_pnl_zar = (
            self.compute_net_pnl(pnl_zar, commission, swap)
            if pnl_zar is not None else None
        )
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO trades (
                    trade_id, instrument, direction, units, entry_price,
                    exit_price, entry_time, exit_time, stop_loss, take_profit,
                    pnl_pips, pnl_zar, ml_confidence, exit_reason,
                    slippage_pips, spread_at_entry, balance_after,
                    model_version, trend_15min, indicators_json, adjustments_json,
                    commission, swap, net_pnl_zar
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                commission, swap, net_pnl_zar,
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

    def log_control_command(
        self,
        verb: str,
        args: dict = None,
        reason: str = "",
        requested_by: str = "",
        ts_utc: str = None,
    ) -> int:
        """
        Record an incoming control-queue (`tb` CLI) command as
        outcome='pending'. Returns the new row id so the caller can later
        call `update_control_outcome()` once the command has been processed.

        `ts_utc` lets callers (ControlQueue, which has its own injectable
        clock for testing) supply the timestamp explicitly; defaults to
        wall-clock UTC now.
        """
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("""
                INSERT INTO control_log
                (ts_utc, verb, args_json, reason, requested_by, outcome)
                VALUES (?, ?, ?, ?, ?, 'pending')
            """, (
                ts_utc or datetime.now(timezone.utc).isoformat(),
                verb,
                json.dumps(args) if args else None,
                reason,
                requested_by,
            ))
            return cur.lastrowid

    def update_control_outcome(
        self,
        log_id: int,
        outcome: str,
        before_config_json: str = None,
        after_config_json: str = None,
    ):
        """Update a control_log row: pending -> applied|rejected|error."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE control_log
                SET outcome = ?,
                    before_config_json = COALESCE(?, before_config_json),
                    after_config_json = COALESCE(?, after_config_json)
                WHERE id = ?
            """, (outcome, before_config_json, after_config_json, log_id))

    def get_control_log(
        self,
        verb: str = None,
        requested_by: str = None,
        outcome: str = None,
        limit: int = 200,
    ) -> pd.DataFrame:
        """Query the control_log audit trail, most recent first."""
        query = "SELECT * FROM control_log WHERE 1=1"
        params = []
        if verb:
            query += " AND verb = ?"
            params.append(verb)
        if requested_by:
            query += " AND requested_by = ?"
            params.append(requested_by)
        if outcome:
            query += " AND outcome = ?"
            params.append(outcome)
        query += " ORDER BY id DESC LIMIT ?"
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
