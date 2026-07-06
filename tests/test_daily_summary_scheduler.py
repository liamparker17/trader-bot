"""
Task 5 — Blocker H: daily summary scheduler.

Covers:
- DailySummaryScheduler.due() fires exactly once per 21:00 UTC boundary
  crossing (config fallback chain: trading.session_reset_hour_utc ->
  risk.session_boundary_hour_utc -> 21), using an injected clock (no
  sleeps).
- Double-fire guard survives a fresh scheduler instance (simulating a
  process restart) via the persisted journal event row.
- TraderBot._maybe_send_daily_summary()/_handle_loop_exception() wiring:
  fires telegram.daily_summary() exactly once per boundary and never
  raises; the main-loop catch-all logs + rate-limits a Telegram alert
  per exception type and lets the loop continue.
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.config import Config
from src.monitoring.trade_journal import TradeJournal
from src.main import DailySummaryScheduler, TraderBot


def _journal(tmp_path):
    config = Config(settings={
        "monitoring": {"trade_journal_db": str(tmp_path / "trades.db")},
    })
    return TradeJournal(config)


def _config(**overrides):
    settings = {
        "risk": {"session_boundary_hour_utc": 21},
        "trading": {},
    }
    settings.update(overrides)
    return Config(settings=settings)


# ---------------------------------------------------------------------------
# DailySummaryScheduler
# ---------------------------------------------------------------------------

def test_due_returns_none_before_first_boundary_reached(tmp_path):
    journal = _journal(tmp_path)
    clock_box = {"now": datetime(2026, 7, 8, 10, 0, tzinfo=timezone.utc)}
    scheduler = DailySummaryScheduler(_config(), journal, clock=lambda: clock_box["now"])

    # A boundary in the past (Tue 21:00 UTC) is always "due" the first
    # time we ask, since nothing has fired yet.
    assert scheduler.due() == "2026-07-07"


def test_due_fires_exactly_once_per_boundary_crossing(tmp_path):
    journal = _journal(tmp_path)
    clock_box = {"now": datetime(2026, 7, 8, 10, 0, tzinfo=timezone.utc)}
    scheduler = DailySummaryScheduler(_config(), journal, clock=lambda: clock_box["now"])

    boundary_date = scheduler.due()
    assert boundary_date == "2026-07-07"
    scheduler.mark_fired(boundary_date)

    # Still same boundary -> not due again.
    assert scheduler.due() is None

    # Cross the boundary.
    clock_box["now"] = datetime(2026, 7, 8, 21, 0, tzinfo=timezone.utc)
    assert scheduler.due() == "2026-07-08"


def test_double_fire_guarded_across_process_restart_via_journal(tmp_path):
    journal = _journal(tmp_path)
    now = datetime(2026, 7, 8, 22, 0, tzinfo=timezone.utc)

    scheduler_a = DailySummaryScheduler(_config(), journal, clock=lambda: now)
    boundary_date = scheduler_a.due()
    assert boundary_date == "2026-07-08"
    scheduler_a.mark_fired(boundary_date)

    # Fresh scheduler instance (simulating a restart) sharing the same
    # journal must see the already-fired event and not re-fire.
    scheduler_b = DailySummaryScheduler(_config(), journal, clock=lambda: now)
    assert scheduler_b.due() is None


def test_reset_hour_uses_trading_key_before_risk_fallback(tmp_path):
    journal = _journal(tmp_path)
    config = _config(trading={"session_reset_hour_utc": 5})
    scheduler = DailySummaryScheduler(config, journal)
    assert scheduler.reset_hour == 5


def test_reset_hour_falls_back_to_risk_key_then_default(tmp_path):
    journal = _journal(tmp_path)
    scheduler = DailySummaryScheduler(_config(), journal)
    assert scheduler.reset_hour == 21

    scheduler_default = DailySummaryScheduler(_config(risk={}), journal)
    assert scheduler_default.reset_hour == 21


# ---------------------------------------------------------------------------
# TraderBot._maybe_send_daily_summary / _handle_loop_exception
# ---------------------------------------------------------------------------

def _bare_bot():
    """Construct a TraderBot without running __init__ (avoids touching
    real config/env/MT5); wire only the attributes the methods need."""
    bot = TraderBot.__new__(TraderBot)
    bot.telegram = MagicMock()
    bot.performance = MagicMock()
    bot.performance.get_summary.return_value = {
        "total_trades": 5, "wins": 3, "losses": 2,
        "win_rate": 0.6, "total_pnl": 42.0, "max_drawdown_pct": 0.02,
    }
    bot.journal = MagicMock()
    bot.daily_summary_scheduler = MagicMock()
    bot._loop_error_last_alert = {}
    bot._loop_error_cooldown_seconds = 300
    return bot


def test_maybe_send_daily_summary_fires_once_when_due():
    bot = _bare_bot()
    bot.daily_summary_scheduler.due.return_value = "2026-07-07"

    bot._maybe_send_daily_summary(balance=1234.5)

    bot.telegram.daily_summary.assert_called_once()
    kwargs = bot.telegram.daily_summary.call_args.kwargs
    assert kwargs["date"] == "2026-07-07"
    assert kwargs["balance"] == 1234.5
    assert kwargs["trades"] == 5
    bot.daily_summary_scheduler.mark_fired.assert_called_once_with("2026-07-07")


def test_maybe_send_daily_summary_noop_when_not_due():
    bot = _bare_bot()
    bot.daily_summary_scheduler.due.return_value = None

    bot._maybe_send_daily_summary(balance=1234.5)

    bot.telegram.daily_summary.assert_not_called()
    bot.daily_summary_scheduler.mark_fired.assert_not_called()


def test_maybe_send_daily_summary_never_raises_on_telegram_failure():
    bot = _bare_bot()
    bot.daily_summary_scheduler.due.return_value = "2026-07-07"
    bot.telegram.daily_summary.side_effect = RuntimeError("network down")

    bot._maybe_send_daily_summary(balance=1234.5)  # must not raise


def test_handle_loop_exception_logs_and_alerts_once():
    bot = _bare_bot()

    bot._handle_loop_exception(ValueError("boom"))

    bot.telegram.bot_error.assert_called_once()
    args = bot.telegram.bot_error.call_args.args
    assert args[0] == "ValueError"


def test_handle_loop_exception_rate_limits_same_exception_type(monkeypatch):
    bot = _bare_bot()
    clock_box = {"now": 1000.0}
    monkeypatch.setattr("src.main.time.time", lambda: clock_box["now"])

    bot._handle_loop_exception(ValueError("boom 1"))
    clock_box["now"] = 1001.0
    bot._handle_loop_exception(ValueError("boom 2"))  # within cooldown -> suppressed

    assert bot.telegram.bot_error.call_count == 1


def test_handle_loop_exception_allows_different_exception_types_independently(monkeypatch):
    bot = _bare_bot()
    monkeypatch.setattr("src.main.time.time", lambda: 1000.0)

    bot._handle_loop_exception(ValueError("boom"))
    bot._handle_loop_exception(KeyError("missing"))

    assert bot.telegram.bot_error.call_count == 2


def test_handle_loop_exception_never_raises_when_alert_fails(monkeypatch):
    bot = _bare_bot()
    monkeypatch.setattr("src.main.time.time", lambda: 1000.0)
    bot.telegram.bot_error.side_effect = RuntimeError("network down")

    bot._handle_loop_exception(ValueError("boom"))  # must not raise
