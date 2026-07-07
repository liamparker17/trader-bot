"""
Task 7 (R) — Trade journal fee/swap columns + net P&L math.

Covers:
- Idempotent migration adds commission/swap/net_pnl_zar columns
- record_trade persists commission/swap and computes net_pnl_zar
- Net P&L math: gross + commission + swap (MT5 reports both as negative costs)
"""
import sqlite3
from datetime import datetime, timezone

from src.config import Config
from src.monitoring.trade_journal import TradeJournal


def _journal(tmp_path):
    config = Config(settings={
        "monitoring": {"trade_journal_db": str(tmp_path / "trades.db")}
    })
    return TradeJournal(config)


def _columns(db_path):
    with sqlite3.connect(db_path) as conn:
        return {row[1] for row in conn.execute("PRAGMA table_info(trades)").fetchall()}


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def test_fresh_db_has_fee_columns(tmp_path):
    journal = _journal(tmp_path)
    cols = _columns(journal.db_path)
    assert {"commission", "swap", "net_pnl_zar"} <= cols


def test_migration_is_idempotent_on_already_migrated_db(tmp_path):
    journal = _journal(tmp_path)
    # Re-opening an already-migrated DB must not raise (duplicate ALTER).
    journal2 = TradeJournal(journal.config)
    cols = _columns(journal2.db_path)
    assert {"commission", "swap", "net_pnl_zar"} <= cols


def test_migration_adds_columns_to_legacy_db_missing_them(tmp_path):
    db_path = tmp_path / "legacy.db"
    # Simulate a pre-Task-7 DB: trades table without fee columns.
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE trades (
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

    config = Config(settings={"monitoring": {"trade_journal_db": str(db_path)}})
    journal = TradeJournal(config)  # should ALTER TABLE in-place, no error
    cols = _columns(journal.db_path)
    assert {"commission", "swap", "net_pnl_zar"} <= cols


# ---------------------------------------------------------------------------
# record_trade + net P&L math
# ---------------------------------------------------------------------------

def test_record_trade_defaults_commission_and_swap_to_zero(tmp_path):
    journal = _journal(tmp_path)
    journal.record_trade(
        trade_id="t1", instrument="EUR_USD", direction="buy", units=1000,
        entry_price=1.1000, entry_time=datetime.now(timezone.utc),
        stop_loss=1.0950, take_profit=1.1080, ml_confidence=0.6,
        exit_price=1.1050, pnl_zar=50.0,
    )
    with sqlite3.connect(journal.db_path) as conn:
        row = conn.execute(
            "SELECT commission, swap, net_pnl_zar FROM trades WHERE trade_id = 't1'"
        ).fetchone()
    assert row == (0.0, 0.0, 50.0)


def test_record_trade_computes_net_pnl_with_fees(tmp_path):
    journal = _journal(tmp_path)
    # MT5 reports commission/swap as negative costs.
    journal.record_trade(
        trade_id="t2", instrument="EUR_USD", direction="buy", units=1000,
        entry_price=1.1000, entry_time=datetime.now(timezone.utc),
        stop_loss=1.0950, take_profit=1.1080, ml_confidence=0.6,
        exit_price=1.1050, pnl_zar=50.0,
        commission=-2.5, swap=-0.75,
    )
    with sqlite3.connect(journal.db_path) as conn:
        row = conn.execute(
            "SELECT commission, swap, net_pnl_zar FROM trades WHERE trade_id = 't2'"
        ).fetchone()
    assert row == (-2.5, -0.75, 46.75)


def test_record_trade_net_pnl_is_null_when_still_open(tmp_path):
    journal = _journal(tmp_path)
    journal.record_trade(
        trade_id="t3", instrument="EUR_USD", direction="buy", units=1000,
        entry_price=1.1000, entry_time=datetime.now(timezone.utc),
        stop_loss=1.0950, take_profit=1.1080, ml_confidence=0.6,
    )
    with sqlite3.connect(journal.db_path) as conn:
        row = conn.execute(
            "SELECT pnl_zar, net_pnl_zar FROM trades WHERE trade_id = 't3'"
        ).fetchone()
    assert row == (None, None)


def test_compute_net_pnl_static_math():
    assert TradeJournal.compute_net_pnl(50.0, commission=-2.5, swap=-0.75) == 46.75
    assert TradeJournal.compute_net_pnl(100.0) == 100.0
    assert TradeJournal.compute_net_pnl(-30.0, commission=-1.0, swap=-0.5) == -31.5
