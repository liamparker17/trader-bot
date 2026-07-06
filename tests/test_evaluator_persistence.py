"""
Task 7 (K) — ML Evaluator checkpoint persistence via the trade journal DB.

Covers:
- Evaluator persists trades_since_retrain / last_retrain_time / model_version
  to a dedicated `evaluator_state` table in the same SQLite DB as the journal.
- State round-trips across a fresh Evaluator instance (simulates restart).
- Persistence happens automatically on record_trade / mark_retrained ("every
  update"), not only on explicit save_state() calls.
- Idempotent CREATE TABLE — safe to construct repeatedly against the same DB.
"""
import sqlite3
from datetime import datetime, timezone

from src.config import Config
from src.ml.evaluator import Evaluator, TradeRecord


def _config(tmp_path):
    return Config(settings={
        "monitoring": {"trade_journal_db": str(tmp_path / "trades.db")}
    })


def test_state_table_created_idempotently(tmp_path):
    config = _config(tmp_path)
    Evaluator(config)
    Evaluator(config)  # second construction on same DB must not raise
    with sqlite3.connect(tmp_path / "trades.db") as conn:
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
    assert "evaluator_state" in tables


def test_save_and_load_state_round_trip(tmp_path):
    config = _config(tmp_path)
    ev = Evaluator(config)
    ev.trades_since_retrain = 42
    ev.last_retrain_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ev.model_version = "v2.3"
    ev.save_state()

    # Simulate restart: fresh Evaluator instance loads persisted state.
    ev2 = Evaluator(config)
    ev2.load_state()
    assert ev2.trades_since_retrain == 42
    assert ev2.last_retrain_time == datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert ev2.model_version == "v2.3"


def test_load_state_on_empty_db_is_noop(tmp_path):
    config = _config(tmp_path)
    ev = Evaluator(config)
    ev.load_state()  # no prior state row — must not raise
    assert ev.trades_since_retrain == 0
    assert ev.last_retrain_time is None
    assert ev.model_version is None


def test_record_trade_persists_state_automatically(tmp_path):
    config = _config(tmp_path)
    ev = Evaluator(config)
    ev.record_trade(TradeRecord(
        prediction=0.6, predicted_action="trade", actual_outcome=1, pnl=10.0,
    ))

    # Fresh instance should see the persisted counter without an explicit save_state().
    ev2 = Evaluator(config)
    ev2.load_state()
    assert ev2.trades_since_retrain == 1


def test_mark_retrained_persists_state_automatically(tmp_path):
    config = _config(tmp_path)
    ev = Evaluator(config)
    ev.trades_since_retrain = 500
    ev.mark_retrained("v3.0")

    ev2 = Evaluator(config)
    ev2.load_state()
    assert ev2.trades_since_retrain == 0
    assert ev2.model_version == "v3.0"
    assert ev2.last_retrain_time is not None


def test_save_state_upserts_not_duplicates(tmp_path):
    config = _config(tmp_path)
    ev = Evaluator(config)
    ev.trades_since_retrain = 1
    ev.save_state()
    ev.trades_since_retrain = 2
    ev.save_state()

    with sqlite3.connect(tmp_path / "trades.db") as conn:
        rows = conn.execute("SELECT COUNT(*) FROM evaluator_state").fetchone()
    assert rows[0] == 1
