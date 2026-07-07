"""
Task 13 — src/manager/runner.py: ManagerRunner.

Covers:
- Full cycle happy path: briefing -> client.call -> policy validate/clamp
  -> control-queue enqueue -> manager_log row -> Telegram alert.
- API down (client raises ManagerAPIUnavailable) -> outcome=api_unavailable,
  Telegram alert sent, loop keeps going (never raises).
- A clamped proposal still flows through to the control queue with the
  post-clamp value.
- The run loop body never lets an exception escape.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.config import Config
from src.control.effective_config import EffectiveConfig
from src.manager.client import ManagerAPIUnavailable
from src.manager.runner import ManagerRunner
from src.manager.scheduler import ManagerScheduler
from src.monitoring.trade_journal import TradeJournal


class FakeTelegram:
    def __init__(self):
        self.messages = []

    def manager_cycle(self, summary):
        self.messages.append(summary)


class FakeClient:
    def __init__(self, proposals=None, rationale="ok", usage=None, exc=None):
        self.proposals = proposals if proposals is not None else []
        self.rationale = rationale
        self.usage = usage or {"input_tokens": 100, "output_tokens": 50, "cost_zar": 2.5}
        self.exc = exc
        self.model = "claude-opus-4-8"
        self.calls = []

    def call(self, briefing):
        self.calls.append(briefing)
        if self.exc is not None:
            raise self.exc
        return self.proposals, self.rationale, self.usage


class AlwaysFireScheduler:
    """Minimal scheduler stub: always fires on 'timer', budget always OK."""

    def __init__(self, now, budget_ok=True):
        self._now = now
        self.budget_ok = budget_ok
        self.recorded = []

    def now_fn(self):
        return self._now

    def should_fire(self):
        return True, "timer", "timer"

    def check_budget(self, now):
        return (self.budget_ok, None) if self.budget_ok else (False, "daily_budget_exhausted")

    def warn_budget_exhausted_once(self, now):
        pass

    def record_cycle(self, now=None):
        self.recorded.append(now)


def _config(tmp_path):
    return Config(settings={
        "manager": {"model": "claude-opus-4-8", "usd_zar_rate": 18.0},
        "risk": {"session_boundary_hour_utc": 21},
        "growth": {"milestones": [1500, 2000, 3000, 4500, 6000]},
        "monitoring": {"trade_journal_db": str(tmp_path / "trades.db")},
        "account": {"starting_balance_zar": 1000},
    })


def _journal(config):
    return TradeJournal(config)


def _effective_config():
    return EffectiveConfig(
        data={
            "risk": {"risk_per_trade_pct": 1.5},
            "ml": {"confidence_threshold_high": 0.55, "confidence_threshold_low": 0.50},
            "growth": {"milestones": [1500, 2000, 3000, 4500, 6000]},
        },
        safety_keys=set(),
    )


def _runner(tmp_path, client, budget_ok=True, effective_config=None):
    config = _config(tmp_path)
    journal = _journal(config)
    telegram = FakeTelegram()
    inbox_dir = tmp_path / "control" / "inbox"
    outbox_dir = tmp_path / "control" / "outbox"
    now = datetime(2026, 7, 7, 10, 0, tzinfo=timezone.utc)
    scheduler = AlwaysFireScheduler(now, budget_ok=budget_ok)

    runner = ManagerRunner(
        config=config,
        journal=journal,
        client=client,
        scheduler=scheduler,
        telegram=telegram,
        inbox_dir=inbox_dir,
        outbox_dir=outbox_dir,
        effective_config=effective_config or _effective_config(),
        balance_equity_fn=lambda: (1000.0, 1000.0),
        ratchet_floor=None,
    )
    return runner, journal, telegram, inbox_dir, outbox_dir, scheduler


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_full_cycle_happy_path_enqueues_and_logs(tmp_path):
    client = FakeClient(
        proposals=[
            {"key": "risk.risk_per_trade_pct", "value": 1.2, "reason": "reduce after losses"},
        ],
        rationale="conservative cycle",
    )
    runner, journal, telegram, inbox_dir, outbox_dir, scheduler = _runner(tmp_path, client)

    runner.run_once()

    # Enqueued onto the control queue inbox.
    cmd_files = list(inbox_dir.glob("*.cmd.json"))
    assert len(cmd_files) == 1
    payload = json.loads(cmd_files[0].read_text(encoding="utf-8"))
    assert payload["verb"] == "tune"
    assert payload["args"] == {"key": "risk.risk_per_trade_pct", "value": 1.2}
    assert payload["requested_by"] == "manager"
    assert payload["reason"] == "reduce after losses"

    # manager_log row written.
    df = journal.get_manager_log()
    assert len(df) == 1
    assert df.iloc[0]["outcome"] == "applied"
    assert df.iloc[0]["trigger"] == "timer"

    # Telegram summary sent.
    assert len(telegram.messages) == 1
    assert "MANAGER" in telegram.messages[0]

    # Scheduler informed a cycle ran.
    assert len(scheduler.recorded) == 1


def test_no_op_when_no_proposals(tmp_path):
    client = FakeClient(proposals=[], rationale="nothing to do")
    runner, journal, telegram, inbox_dir, outbox_dir, scheduler = _runner(tmp_path, client)

    runner.run_once()

    assert list(inbox_dir.glob("*.cmd.json")) == []
    df = journal.get_manager_log()
    assert df.iloc[0]["outcome"] == "no_op"


# ---------------------------------------------------------------------------
# Clamped proposal still flows through
# ---------------------------------------------------------------------------

def test_clamped_proposal_flows_through_with_post_clamp_value(tmp_path):
    client = FakeClient(
        proposals=[
            {"key": "risk.risk_per_trade_pct", "value": 99.0, "reason": "way too high on purpose"},
        ],
        rationale="testing clamp",
    )
    runner, journal, telegram, inbox_dir, outbox_dir, scheduler = _runner(tmp_path, client)

    runner.run_once()

    cmd_files = list(inbox_dir.glob("*.cmd.json"))
    assert len(cmd_files) == 1
    payload = json.loads(cmd_files[0].read_text(encoding="utf-8"))
    # Clamped to the static bound (2.5) since risk_ceiling_now(1000, ...) is
    # also 1.5 below the first milestone -> clamped to 1.5, whichever is
    # lower of the static bound and the growth-stage ceiling.
    assert payload["args"]["value"] == pytest.approx(1.5)
    assert payload["reason"] == "way too high on purpose"

    df = journal.get_manager_log()
    applied = json.loads(df.iloc[0]["applied_json"])
    assert applied[0]["clamped"] is True
    assert applied[0]["value"] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# API down
# ---------------------------------------------------------------------------

def test_api_unavailable_logs_and_alerts(tmp_path):
    client = FakeClient(exc=ManagerAPIUnavailable("rate limited after retries"))
    runner, journal, telegram, inbox_dir, outbox_dir, scheduler = _runner(tmp_path, client)

    runner.run_once()

    assert list(inbox_dir.glob("*.cmd.json")) == []
    df = journal.get_manager_log()
    assert len(df) == 1
    assert df.iloc[0]["outcome"] == "api_unavailable"
    assert len(telegram.messages) == 1
    assert "unavailable" in telegram.messages[0].lower()


# ---------------------------------------------------------------------------
# Budget exhausted
# ---------------------------------------------------------------------------

def test_budget_exhausted_skips_cycle_without_calling_client(tmp_path):
    client = FakeClient(proposals=[{"key": "risk.risk_per_trade_pct", "value": 1.2, "reason": "x"}])
    runner, journal, telegram, inbox_dir, outbox_dir, scheduler = _runner(tmp_path, client, budget_ok=False)

    runner.run_once()

    assert client.calls == []
    df = journal.get_manager_log()
    assert df.iloc[0]["outcome"] == "budget_exhausted"
    assert list(inbox_dir.glob("*.cmd.json")) == []


# ---------------------------------------------------------------------------
# Exception safety
# ---------------------------------------------------------------------------

def test_run_once_never_raises_on_unexpected_error(tmp_path):
    class ExplodingClient:
        def call(self, briefing):
            raise RuntimeError("boom — totally unexpected")

    runner, journal, telegram, inbox_dir, outbox_dir, scheduler = _runner(tmp_path, ExplodingClient())

    # Must not raise.
    runner.run_once()

    df = journal.get_manager_log()
    assert len(df) == 1
    assert df.iloc[0]["outcome"] == "error"


def test_run_forever_survives_repeated_errors(tmp_path):
    class ExplodingScheduler:
        def now_fn(self):
            return datetime(2026, 7, 7, 10, 0, tzinfo=timezone.utc)

        def should_fire(self):
            raise RuntimeError("scheduler exploded")

    config = _config(tmp_path)
    journal = _journal(config)
    runner = ManagerRunner(
        config=config,
        journal=journal,
        client=FakeClient(),
        scheduler=ExplodingScheduler(),
        telegram=FakeTelegram(),
        inbox_dir=tmp_path / "control" / "inbox",
        outbox_dir=tmp_path / "control" / "outbox",
        effective_config=_effective_config(),
        balance_equity_fn=lambda: (1000.0, 1000.0),
    )

    calls = {"n": 0}

    def sleep_fn(_seconds):
        calls["n"] += 1
        if calls["n"] >= 3:
            raise StopIteration  # test escape hatch

    with pytest.raises(StopIteration):
        runner.run_forever(sleep_fn=sleep_fn, poll_interval_seconds=0)

    assert calls["n"] == 3


# ---------------------------------------------------------------------------
# Fix round: API cost must never be silently lost on post-call error paths
# ---------------------------------------------------------------------------

def test_post_api_failure_still_logs_cost(tmp_path, monkeypatch):
    """client.call succeeded (cost incurred) but enqueue blows up — the
    error row must still carry the cycle's token/cost usage."""
    import src.manager.runner as runner_module

    client = FakeClient(
        proposals=[{"key": "risk.risk_per_trade_pct", "value": 1.2, "reason": "r"}],
        rationale="will fail downstream",
    )
    runner, journal, telegram, inbox_dir, outbox_dir, scheduler = _runner(tmp_path, client)

    def exploding_enqueue(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(runner_module, "_enqueue_tune", exploding_enqueue)

    runner.run_once()  # must not raise

    df = journal.get_manager_log()
    assert len(df) == 1
    row = df.iloc[0]
    assert row["outcome"] == "error"
    assert row["cost_zar"] == pytest.approx(2.5)
    assert row["input_tokens"] == 100
    assert row["output_tokens"] == 50


def test_api_unavailable_with_billed_response_logs_cost(tmp_path):
    """A billed response with no usable tool_use block raises
    ManagerAPIUnavailable carrying .usage — the log row must record it."""
    exc = ManagerAPIUnavailable("no tool_use block")
    exc.usage = {"input_tokens": 80, "output_tokens": 10, "cost_zar": 1.7}
    client = FakeClient(exc=exc)
    runner, journal, telegram, inbox_dir, outbox_dir, scheduler = _runner(tmp_path, client)

    runner.run_once()

    df = journal.get_manager_log()
    assert len(df) == 1
    row = df.iloc[0]
    assert row["outcome"] == "api_unavailable"
    assert row["cost_zar"] == pytest.approx(1.7)


def test_journal_write_failure_is_loud_but_never_raises(tmp_path, caplog):
    """If even the manager_log write fails, the loop must survive and the
    cost figure must land in a CRITICAL log line (recoverable from logs)."""
    import logging as _logging

    client = FakeClient(
        proposals=[], rationale="ok",
        usage={"input_tokens": 100, "output_tokens": 50, "cost_zar": 2.5},
    )
    runner, journal, telegram, inbox_dir, outbox_dir, scheduler = _runner(tmp_path, client)

    def exploding_log(*args, **kwargs):
        raise RuntimeError("db locked")

    journal.log_manager_cycle = exploding_log

    with caplog.at_level(_logging.CRITICAL, logger="traderbot.manager.runner"):
        runner.run_once()  # must not raise

    assert any("2.5" in rec.message or "cost" in rec.message.lower()
               for rec in caplog.records)


# ---------------------------------------------------------------------------
# Fix round: manager.enabled=false must prevent cycles end-to-end
# ---------------------------------------------------------------------------

def test_disabled_manager_runs_no_cycles(tmp_path):
    config = Config(settings={
        "manager": {"enabled": False, "model": "claude-opus-4-8", "usd_zar_rate": 18.0},
        "risk": {"session_boundary_hour_utc": 21},
        "growth": {"milestones": [1500, 2000, 3000, 4500, 6000]},
        "monitoring": {"trade_journal_db": str(tmp_path / "trades.db")},
        "account": {"starting_balance_zar": 1000},
        "trading": {"trading_sessions": {"forex": {"start_hour": 0, "end_hour": 24}}},
    })
    journal = TradeJournal(config)
    client = FakeClient(proposals=[{"key": "risk.risk_per_trade_pct", "value": 1.2, "reason": "r"}])
    scheduler = ManagerScheduler(
        config=config, journal=journal, telegram=None,
        now_fn=lambda: datetime(2026, 7, 7, 10, 0, tzinfo=timezone.utc),
        events_dir=tmp_path / "control" / "manager_events",
    )
    runner = ManagerRunner(
        config=config, journal=journal, client=client, scheduler=scheduler,
        telegram=FakeTelegram(),
        inbox_dir=tmp_path / "control" / "inbox",
        outbox_dir=tmp_path / "control" / "outbox",
        effective_config=_effective_config(),
        balance_equity_fn=lambda: (1000.0, 1000.0),
    )

    runner.run_once()

    assert client.calls == []
    assert journal.get_manager_log().empty


def test_budget_check_failure_fails_closed(tmp_path):
    """If the budget governor itself errors, skip the cycle — never spend
    with the governor blind."""
    client = FakeClient(proposals=[{"key": "risk.risk_per_trade_pct", "value": 1.2, "reason": "r"}])
    runner, journal, telegram, inbox_dir, outbox_dir, scheduler = _runner(tmp_path, client)

    def exploding_check_budget(now):
        raise RuntimeError("journal unreadable")

    scheduler.check_budget = exploding_check_budget

    runner.run_once()  # must not raise

    assert client.calls == []
    assert journal.get_manager_log().empty
