"""
Task 13 — src/manager/scheduler.py: ManagerScheduler.

Covers:
- min_gap_minutes suppresses a rapid repeated event trigger.
- Daily cap (max_cycles_per_day) stops firing after the 21st cycle.
- Budget governor: per-day cap stops cycles, total-window cap stops
  cycles, interval stretches when remaining daily budget is tight.
- Event files are consumed oldest-first (read + delete).
- Injectable clock — no wall-clock dependency.
"""
import json
from datetime import datetime, timedelta, timezone

import pytest

from src.config import Config
from src.manager.scheduler import ManagerScheduler
from src.monitoring.trade_journal import TradeJournal


def _config(tmp_path=None, **manager_overrides):
    manager = {
        "enabled": True,
        "model": "claude-opus-4-8",
        "cycle_minutes": 60,
        "min_gap_minutes": 20,
        "max_cycles_per_day": 20,
        "max_adjustments_per_cycle": 3,
        "event_triggers": {"drawdown_pct": 2.0, "consecutive_losses": 3, "circuit_breaker": True},
        "usd_zar_rate": 18.0,
        "api_budget_zar_total": 500,
        "api_budget_days": 10,
        "api_budget_zar_per_day": 50,
    }
    manager.update(manager_overrides)
    settings = {
        "manager": manager,
        "risk": {"session_boundary_hour_utc": 21},
        "trading": {
            "trading_sessions": {
                "forex": {"start_hour": 0, "end_hour": 24},
            }
        },
        "monitoring": {"trade_journal_db": str((tmp_path or _tmp()) / "trades.db")},
    }
    return Config(settings=settings)


def _tmp():
    import tempfile
    from pathlib import Path
    return Path(tempfile.mkdtemp())


def _journal(tmp_path, config=None):
    config = config or _config(tmp_path)
    return TradeJournal(config)


def _clock(start):
    box = {"now": start}

    def now_fn():
        return box["now"]

    def advance(**kwargs):
        box["now"] += timedelta(**kwargs)

    return now_fn, advance


class FakeTelegram:
    def __init__(self):
        self.messages = []

    def manager_cycle(self, summary):
        self.messages.append(summary)


def _scheduler(tmp_path, config=None, now_fn=None, telegram=None):
    config = config or _config(tmp_path)
    journal = _journal(tmp_path, config)
    events_dir = tmp_path / "control" / "manager_events"
    return ManagerScheduler(
        config=config,
        journal=journal,
        telegram=telegram or FakeTelegram(),
        now_fn=now_fn or (lambda: datetime(2026, 7, 7, 10, 0, tzinfo=timezone.utc)),
        events_dir=events_dir,
    ), journal, events_dir


# ---------------------------------------------------------------------------
# Event consumption
# ---------------------------------------------------------------------------

def test_events_consumed_oldest_first_and_deleted(tmp_path):
    scheduler, journal, events_dir = _scheduler(tmp_path)
    events_dir.mkdir(parents=True)
    (events_dir / "20260101T000000_a.json").write_text(json.dumps({"type": "drawdown"}), encoding="utf-8")
    (events_dir / "20260101T000001_b.json").write_text(json.dumps({"type": "consecutive_losses"}), encoding="utf-8")

    events = scheduler.poll_events()
    assert [e["type"] for e in events] == ["drawdown", "consecutive_losses"]
    assert list(events_dir.glob("*.json")) == []


def test_no_events_dir_returns_empty(tmp_path):
    scheduler, journal, events_dir = _scheduler(tmp_path)
    assert scheduler.poll_events() == []


# ---------------------------------------------------------------------------
# min_gap_minutes
# ---------------------------------------------------------------------------

def test_min_gap_suppresses_rapid_repeated_event(tmp_path):
    now_fn, advance = _clock(datetime(2026, 7, 7, 10, 0, tzinfo=timezone.utc))
    scheduler, journal, events_dir = _scheduler(tmp_path, now_fn=now_fn)
    events_dir.mkdir(parents=True)
    (events_dir / "e1.json").write_text(json.dumps({"type": "drawdown"}), encoding="utf-8")

    fire, trigger, reason = scheduler.should_fire()
    assert fire is True
    assert trigger == "drawdown"
    scheduler.record_cycle(now_fn())

    # A second event drops 5 minutes later — inside the 20-minute min-gap.
    advance(minutes=5)
    (events_dir / "e2.json").write_text(json.dumps({"type": "drawdown"}), encoding="utf-8")
    fire2, trigger2, reason2 = scheduler.should_fire()
    assert fire2 is False
    assert reason2 == "min_gap"

    # 20+ minutes later, it's allowed again (event file still consumed/gone
    # from the first poll — this checks a *new* one).
    advance(minutes=16)  # total 21 minutes since last cycle
    (events_dir / "e3.json").write_text(json.dumps({"type": "drawdown"}), encoding="utf-8")
    fire3, trigger3, reason3 = scheduler.should_fire()
    assert fire3 is True


# ---------------------------------------------------------------------------
# Daily cap
# ---------------------------------------------------------------------------

def test_daily_cap_stops_after_20_cycles(tmp_path):
    now = datetime(2026, 7, 7, 10, 0, tzinfo=timezone.utc)
    config = _config(tmp_path, max_cycles_per_day=20)
    journal = _journal(tmp_path, config)
    events_dir = tmp_path / "control" / "manager_events"
    scheduler = ManagerScheduler(
        config=config, journal=journal, telegram=FakeTelegram(),
        now_fn=lambda: now, events_dir=events_dir,
    )

    for i in range(20):
        journal.log_manager_cycle(trigger="timer", outcome="no_op", ts_utc=now.isoformat())

    ok, reason = scheduler.check_daily_cap(now)
    assert ok is False
    assert reason == "daily_cap_exceeded"


def test_daily_cap_allows_under_limit(tmp_path):
    now = datetime(2026, 7, 7, 10, 0, tzinfo=timezone.utc)
    config = _config(tmp_path, max_cycles_per_day=20)
    journal = _journal(tmp_path, config)
    events_dir = tmp_path / "control" / "manager_events"
    scheduler = ManagerScheduler(
        config=config, journal=journal, telegram=FakeTelegram(),
        now_fn=lambda: now, events_dir=events_dir,
    )
    for i in range(19):
        journal.log_manager_cycle(trigger="timer", outcome="no_op", ts_utc=now.isoformat())

    ok, reason = scheduler.check_daily_cap(now)
    assert ok is True


# ---------------------------------------------------------------------------
# Budget governor
# ---------------------------------------------------------------------------

def test_budget_governor_per_day_cap_skips(tmp_path):
    now = datetime(2026, 7, 7, 10, 0, tzinfo=timezone.utc)
    config = _config(tmp_path, api_budget_zar_per_day=50, api_budget_zar_total=500, api_budget_days=10)
    journal = _journal(tmp_path, config)
    events_dir = tmp_path / "control" / "manager_events"
    telegram = FakeTelegram()
    scheduler = ManagerScheduler(
        config=config, journal=journal, telegram=telegram,
        now_fn=lambda: now, events_dir=events_dir,
    )
    # Already spent R50 today.
    journal.log_manager_cycle(trigger="timer", outcome="applied", cost_zar=50.0, ts_utc=now.isoformat())

    ok, reason = scheduler.check_budget(now)
    assert ok is False
    assert reason == "daily_budget_exhausted"


def test_budget_governor_total_cap_skips(tmp_path):
    now = datetime(2026, 7, 7, 10, 0, tzinfo=timezone.utc)
    config = _config(tmp_path, api_budget_zar_per_day=50, api_budget_zar_total=100, api_budget_days=10)
    journal = _journal(tmp_path, config)
    events_dir = tmp_path / "control" / "manager_events"
    scheduler = ManagerScheduler(
        config=config, journal=journal, telegram=FakeTelegram(),
        now_fn=lambda: now, events_dir=events_dir,
    )
    # Spread over several days so daily cap isn't the trigger, but the
    # cumulative 10-day total is.
    for days_ago in (1, 2, 3):
        ts = (now - timedelta(days=days_ago)).isoformat()
        journal.log_manager_cycle(trigger="timer", outcome="applied", cost_zar=40.0, ts_utc=ts)

    ok, reason = scheduler.check_budget(now)
    assert ok is False
    assert reason == "total_budget_exhausted"


def test_budget_governor_warns_telegram_once_per_day(tmp_path):
    now = datetime(2026, 7, 7, 10, 0, tzinfo=timezone.utc)
    config = _config(tmp_path, api_budget_zar_per_day=50)
    journal = _journal(tmp_path, config)
    events_dir = tmp_path / "control" / "manager_events"
    telegram = FakeTelegram()
    scheduler = ManagerScheduler(
        config=config, journal=journal, telegram=telegram,
        now_fn=lambda: now, events_dir=events_dir,
    )
    journal.log_manager_cycle(trigger="timer", outcome="applied", cost_zar=50.0, ts_utc=now.isoformat())

    scheduler.warn_budget_exhausted_once(now)
    scheduler.warn_budget_exhausted_once(now)  # second call same day -> no duplicate
    assert len(telegram.messages) == 1


def test_budget_governor_stretches_interval_when_tight(tmp_path):
    now = datetime(2026, 7, 7, 20, 0, tzinfo=timezone.utc)  # 1 hour before 21:00 boundary
    config = _config(tmp_path, api_budget_zar_per_day=50, cycle_minutes=60)
    journal = _journal(tmp_path, config)
    events_dir = tmp_path / "control" / "manager_events"
    scheduler = ManagerScheduler(
        config=config, journal=journal, telegram=FakeTelegram(),
        now_fn=lambda: now, events_dir=events_dir,
        estimated_cost_per_cycle_zar=10.0,
    )
    # R45 already spent of R50 daily budget -> only R5 left, less than one
    # estimated cycle cost -> interval must stretch beyond cycle_minutes.
    journal.log_manager_cycle(trigger="timer", outcome="applied", cost_zar=45.0, ts_utc=now.isoformat())

    interval = scheduler.next_interval_minutes(now)
    assert interval > config.get("manager.cycle_minutes")


def test_budget_governor_normal_interval_when_budget_healthy(tmp_path):
    now = datetime(2026, 7, 7, 10, 0, tzinfo=timezone.utc)
    config = _config(tmp_path, api_budget_zar_per_day=50, cycle_minutes=60)
    journal = _journal(tmp_path, config)
    events_dir = tmp_path / "control" / "manager_events"
    scheduler = ManagerScheduler(
        config=config, journal=journal, telegram=FakeTelegram(),
        now_fn=lambda: now, events_dir=events_dir,
        estimated_cost_per_cycle_zar=1.0,
    )
    interval = scheduler.next_interval_minutes(now)
    assert interval == pytest.approx(config.get("manager.cycle_minutes"))
