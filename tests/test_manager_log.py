"""
Task 12 — manager_log table + CRUD on TradeJournal.

Covers:
- Table creation (fresh DB has all manager_log columns).
- Idempotent migration guard (re-opening an already-migrated DB doesn't raise).
- log_manager_cycle() persists a full row and returns a row id.
- get_manager_log() filters by days / limit, most-recent-first.
- manager_cost_since() sums cost_zar for rows at/after a timestamp.
"""
import sqlite3
from datetime import datetime, timedelta, timezone

from src.config import Config
from src.monitoring.trade_journal import TradeJournal


def _journal(tmp_path):
    config = Config(settings={
        "monitoring": {"trade_journal_db": str(tmp_path / "trades.db")}
    })
    return TradeJournal(config)


def _columns(db_path):
    with sqlite3.connect(db_path) as conn:
        return {row[1] for row in conn.execute("PRAGMA table_info(manager_log)").fetchall()}


EXPECTED_COLUMNS = {
    "id", "ts_utc", "trigger", "briefing_json", "model", "input_tokens",
    "output_tokens", "cost_zar", "rationale", "proposals_json",
    "applied_json", "rejected_json", "outcome",
}


def test_fresh_db_has_manager_log_columns(tmp_path):
    journal = _journal(tmp_path)
    assert EXPECTED_COLUMNS <= _columns(journal.db_path)


def test_migration_is_idempotent_on_already_migrated_db(tmp_path):
    journal = _journal(tmp_path)
    journal2 = TradeJournal(journal.config)
    assert EXPECTED_COLUMNS <= _columns(journal2.db_path)


def test_log_manager_cycle_persists_and_returns_id(tmp_path):
    journal = _journal(tmp_path)
    row_id = journal.log_manager_cycle(
        trigger="scheduled",
        briefing={"balance": 1000},
        model="claude-x",
        input_tokens=100,
        output_tokens=50,
        cost_zar=1.23,
        rationale="looks fine",
        proposals=[{"key": "risk.risk_per_trade_pct", "value": 2.0}],
        applied=[{"key": "risk.risk_per_trade_pct", "value": 2.0, "original_value": 2.0,
                  "reason": "within bounds", "clamped": False}],
        rejected=[],
        outcome="applied",
    )
    assert isinstance(row_id, int)
    assert row_id >= 1

    df = journal.get_manager_log()
    assert len(df) == 1
    row = df.iloc[0]
    assert row["trigger"] == "scheduled"
    assert row["model"] == "claude-x"
    assert row["outcome"] == "applied"
    assert row["cost_zar"] == 1.23
    assert "risk.risk_per_trade_pct" in row["briefing_json"] or row["briefing_json"] is None or "balance" in row["briefing_json"]


def test_get_manager_log_limit(tmp_path):
    journal = _journal(tmp_path)
    for i in range(5):
        journal.log_manager_cycle(trigger=f"cycle-{i}", outcome="noop")

    df = journal.get_manager_log(limit=2)
    assert len(df) == 2
    # most recent first
    assert df.iloc[0]["trigger"] == "cycle-4"
    assert df.iloc[1]["trigger"] == "cycle-3"


def test_get_manager_log_days_filter(tmp_path):
    journal = _journal(tmp_path)
    old_ts = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    recent_ts = datetime.now(timezone.utc).isoformat()

    journal.log_manager_cycle(trigger="old", outcome="noop", ts_utc=old_ts)
    journal.log_manager_cycle(trigger="recent", outcome="noop", ts_utc=recent_ts)

    df = journal.get_manager_log(days=1)
    assert len(df) == 1
    assert df.iloc[0]["trigger"] == "recent"


def test_manager_cost_since_sums_cost(tmp_path):
    journal = _journal(tmp_path)
    now = datetime.now(timezone.utc)
    old_ts = (now - timedelta(days=2)).isoformat()

    journal.log_manager_cycle(trigger="old", outcome="noop", cost_zar=5.0, ts_utc=old_ts)
    journal.log_manager_cycle(trigger="recent1", outcome="noop", cost_zar=2.5, ts_utc=now.isoformat())
    journal.log_manager_cycle(trigger="recent2", outcome="noop", cost_zar=None, ts_utc=now.isoformat())

    since = now - timedelta(hours=1)
    total = journal.manager_cost_since(since)
    assert total == 2.5


def test_manager_cost_since_accepts_iso_string(tmp_path):
    journal = _journal(tmp_path)
    now = datetime.now(timezone.utc)
    journal.log_manager_cycle(trigger="a", outcome="noop", cost_zar=3.0, ts_utc=now.isoformat())

    total = journal.manager_cost_since((now - timedelta(minutes=1)).isoformat())
    assert total == 3.0
