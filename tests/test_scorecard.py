"""
Task 14 — self-funding scorecard.

Covers:
- PerformanceTracker.net_pnl_after_api: realized net P&L minus manager
  API cost over a window, exposed in the perf summary dict.
- PerformanceTracker.manager_stats_since: cycles / adjustments applied /
  API cost for the daily Telegram summary.
- justification_report(): cumulative net P&L, API cost, net-after-cost,
  and the SELF-FUNDING / NOT JUSTIFIED verdict per the budget amendment.
- Telegram daily_summary carries the manager lines when provided.
- `tb manager --verdict` returns the real report once manager_log has rows.
"""
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from src.config import Config
from src.monitoring.performance import PerformanceTracker
from src.monitoring.trade_journal import TradeJournal


NOW = datetime(2026, 7, 7, 10, 0, tzinfo=timezone.utc)


def make_config(tmp_path):
    return Config(settings={
        "monitoring": {"trade_journal_db": str(tmp_path / "trades.db")},
        "account": {"starting_balance_zar": 1000},
    })


@pytest.fixture
def journal(tmp_path):
    return TradeJournal(make_config(tmp_path))


def insert_trade(journal, trade_id, pnl, net_pnl=None, exit_time=None):
    exit_time = (exit_time or NOW).isoformat()
    with sqlite3.connect(journal.db_path) as conn:
        conn.execute(
            """INSERT INTO trades
               (trade_id, instrument, direction, units, entry_price, exit_price,
                entry_time, exit_time, pnl_zar, net_pnl_zar)
               VALUES (?, 'EUR_USD', 'buy', 1000, 1.10, 1.11, ?, ?, ?, ?)""",
            (trade_id, exit_time, exit_time, pnl,
             net_pnl if net_pnl is not None else pnl),
        )


def log_cycle(journal, cost=2.5, applied=None, ts=None, outcome="applied"):
    journal.log_manager_cycle(
        trigger="timer", model="claude-opus-4-8", cost_zar=cost,
        applied=applied if applied is not None else [],
        outcome=outcome, ts_utc=(ts or NOW).isoformat(),
    )


# ---------------------------------------------------------------------------
# net_pnl_after_api
# ---------------------------------------------------------------------------

def test_net_pnl_after_api_subtracts_manager_cost(journal):
    insert_trade(journal, "t1", pnl=100.0, net_pnl=95.0)
    insert_trade(journal, "t2", pnl=-40.0, net_pnl=-42.0)
    log_cycle(journal, cost=2.5)
    log_cycle(journal, cost=1.5)

    tracker = PerformanceTracker(journal)
    # realized net = 95 - 42 = 53; api cost = 4.0
    assert tracker.net_pnl_after_api() == pytest.approx(49.0)


def test_net_pnl_after_api_windowed(journal):
    old = NOW - timedelta(days=30)
    insert_trade(journal, "t1", pnl=100.0, exit_time=old)
    insert_trade(journal, "t2", pnl=10.0, exit_time=NOW - timedelta(hours=1))
    log_cycle(journal, cost=2.0, ts=old)
    log_cycle(journal, cost=1.0, ts=NOW - timedelta(hours=1))

    tracker = PerformanceTracker(journal)
    # 7-day window: only t2 and the recent cycle count.
    assert tracker.net_pnl_after_api(days=7) == pytest.approx(9.0)


def test_summary_exposes_api_cost_and_net_after_api(journal):
    insert_trade(journal, "t1", pnl=50.0)
    log_cycle(journal, cost=3.0)

    summary = PerformanceTracker(journal).get_summary()
    assert summary["api_cost_zar"] == pytest.approx(3.0)
    assert summary["net_pnl_after_api"] == pytest.approx(47.0)


def test_summary_without_manager_log_rows_is_zero_cost(journal):
    insert_trade(journal, "t1", pnl=50.0)
    summary = PerformanceTracker(journal).get_summary()
    assert summary["api_cost_zar"] == 0.0
    assert summary["net_pnl_after_api"] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# manager_stats_since (daily Telegram summary inputs)
# ---------------------------------------------------------------------------

def test_manager_stats_since_counts_cycles_adjustments_cost(journal):
    day_start = NOW.replace(hour=0)
    log_cycle(journal, cost=2.0, applied=[{"key": "weight.XAU_USD", "value": 0.5}],
              ts=NOW - timedelta(hours=2))
    log_cycle(journal, cost=1.0,
              applied=[{"key": "risk.risk_per_trade_pct", "value": 1.2},
                       {"key": "weight.EUR_USD", "value": 1.1}],
              ts=NOW - timedelta(hours=1))
    # Before the boundary — must not count.
    log_cycle(journal, cost=9.0, applied=[{"key": "weight.EUR_USD", "value": 0.9}],
              ts=day_start - timedelta(hours=3))

    stats = PerformanceTracker(journal).manager_stats_since(day_start)
    assert stats["cycles"] == 2
    assert stats["adjustments_applied"] == 3
    assert stats["api_cost_zar"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# justification_report
# ---------------------------------------------------------------------------

def test_justification_self_funding_when_uplift_beats_cost(journal):
    insert_trade(journal, "t1", pnl=200.0)
    log_cycle(journal, cost=10.0)

    report = PerformanceTracker(journal).justification_report(
        heuristic_baseline_pnl_zar=150.0)
    # net-after-cost = 190 > 0; uplift = 200 - 150 = 50 > 10 cost.
    assert report["verdict"] == "SELF-FUNDING"
    assert report["net_pnl_zar"] == pytest.approx(200.0)
    assert report["api_cost_zar"] == pytest.approx(10.0)
    assert report["net_after_cost_zar"] == pytest.approx(190.0)


def test_justification_not_justified_when_uplift_below_cost(journal):
    insert_trade(journal, "t1", pnl=200.0)
    log_cycle(journal, cost=10.0)

    report = PerformanceTracker(journal).justification_report(
        heuristic_baseline_pnl_zar=195.0)
    # uplift = 5 < 10 cost.
    assert report["verdict"] == "NOT JUSTIFIED"


def test_justification_not_justified_when_net_after_cost_negative(journal):
    insert_trade(journal, "t1", pnl=5.0)
    log_cycle(journal, cost=10.0)

    report = PerformanceTracker(journal).justification_report(
        heuristic_baseline_pnl_zar=-100.0)
    assert report["verdict"] == "NOT JUSTIFIED"


def test_justification_conservative_without_baseline(journal):
    insert_trade(journal, "t1", pnl=200.0)
    log_cycle(journal, cost=10.0)

    report = PerformanceTracker(journal).justification_report()
    # No heuristic baseline available -> cannot prove uplift -> conservative.
    assert report["verdict"] == "NOT JUSTIFIED"
    assert "baseline" in report["reason"].lower()


def test_justification_pending_without_manager_cycles(journal):
    insert_trade(journal, "t1", pnl=200.0)
    report = PerformanceTracker(journal).justification_report()
    assert report["verdict"] == "PENDING"


def test_days_since_first_manager_cycle(journal):
    log_cycle(journal, ts=NOW - timedelta(days=9))
    log_cycle(journal, ts=NOW - timedelta(hours=1))
    tracker = PerformanceTracker(journal)
    assert tracker.days_since_first_manager_cycle(now=NOW) == 9


def test_days_since_first_manager_cycle_none_when_empty(journal):
    tracker = PerformanceTracker(journal)
    assert tracker.days_since_first_manager_cycle(now=NOW) is None


# ---------------------------------------------------------------------------
# Telegram daily summary additions
# ---------------------------------------------------------------------------

def test_daily_summary_includes_manager_lines(tmp_path):
    from src.monitoring.telegram_bot import TelegramBot

    config = Config(settings={"telegram": {"enabled": False}})
    bot = TelegramBot(config)
    sent = []
    bot._send = lambda text, parse_mode="HTML": sent.append(text)

    bot.daily_summary(
        date="2026-07-07", trades=5, wins=3, losses=2, pnl=42.0,
        balance=1042.0, win_rate=0.6, max_drawdown=0.02,
        manager_cycles=4, manager_adjustments=2, api_cost_today=8.5,
        net_after_cost_today=33.5, net_after_cost_total=120.0,
        verdict_line="SELF-FUNDING",
    )

    assert len(sent) == 1
    text = sent[0]
    assert "Manager cycles: 4" in text
    assert "2 adjustment" in text
    assert "8.5" in text
    assert "33.5" in text
    assert "120.0" in text
    assert "SELF-FUNDING" in text


def test_daily_summary_backwards_compatible_without_manager_args(tmp_path):
    from src.monitoring.telegram_bot import TelegramBot

    config = Config(settings={"telegram": {"enabled": False}})
    bot = TelegramBot(config)
    sent = []
    bot._send = lambda text, parse_mode="HTML": sent.append(text)

    bot.daily_summary(
        date="2026-07-07", trades=5, wins=3, losses=2, pnl=42.0,
        balance=1042.0, win_rate=0.6, max_drawdown=0.02,
    )
    assert len(sent) == 1
    assert "Manager" not in sent[0]


# ---------------------------------------------------------------------------
# tb manager --verdict wired to the real report
# ---------------------------------------------------------------------------

def test_tb_manager_verdict_real_report(journal, tmp_path, capsys):
    import cli.tb as tb

    insert_trade(journal, "t1", pnl=200.0)
    log_cycle(journal, cost=10.0)

    rc = tb.main(
        ["manager", "--verdict", "--baseline-pnl", "150"],
        config=journal.config, journal=journal,
        inbox_dir=tmp_path / "inbox", outbox_dir=tmp_path / "outbox",
    )
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["verdict"] == "SELF-FUNDING"
    assert out["net_after_cost_zar"] == pytest.approx(190.0)


def test_tb_manager_verdict_pending_when_no_cycles(journal, tmp_path, capsys):
    import cli.tb as tb

    rc = tb.main(
        ["manager", "--verdict"],
        config=journal.config, journal=journal,
        inbox_dir=tmp_path / "inbox", outbox_dir=tmp_path / "outbox",
    )
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["verdict"] == "PENDING"
