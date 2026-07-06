"""
Task 8 — file-based control queue (src/control/queue.py).

Covers, per the task-8 brief:
- Full round-trip (inbox cmd -> poll -> outbox result -> inbox file gone)
  for each verb: pause, resume, tune, revert, status_snapshot.
- Out-of-bounds tune rejected.
- Safety-locked key rejected.
- Rate limit: manual 2nd tune within 24h rejected; manager exempt.
- Revert restores the prior value.
- Atomicity: a stray `.tmp` file mid-write is never picked up.
- control_log row lifecycle: pending -> applied/rejected/error.
"""
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import src.control.effective_config as ec_module
import src.control.queue as queue_module
from src.config import Config
from src.control.effective_config import EffectiveConfig
from src.control.queue import ControlQueue
from src.monitoring.trade_journal import TradeJournal


class FakeRiskManager:
    """Minimal stand-in matching the RiskManager surface queue.py uses."""

    def __init__(self):
        self._manual_pause_reason = ""
        self.entries_blocked = False
        self.open_position_count = 0
        self.ratchet_floor = None

    @property
    def manual_paused(self) -> bool:
        return bool(self._manual_pause_reason)

    @property
    def manual_pause_reason(self) -> str:
        return self._manual_pause_reason

    def set_manual_pause(self, reason: str):
        self._manual_pause_reason = reason

    def clear_manual_pause(self):
        self._manual_pause_reason = ""


class FakeTelegram:
    def __init__(self):
        self.sent = []

    def _send(self, text, parse_mode="HTML"):
        self.sent.append(text)


def _patch_config_paths(tmp_path, monkeypatch):
    settings_path = tmp_path / "settings.yaml"
    safety_path = tmp_path / "safety_floor.yaml"
    tunes_path = tmp_path / "control_data" / "effective_config.json"

    settings_path.write_text(
        "risk:\n  risk_per_trade_pct: 1.5\n"
        "ml:\n  confidence_threshold_high: 0.65\n  confidence_threshold_low: 0.55\n",
        encoding="utf-8",
    )
    safety_path.write_text("risk:\n  min_floor_zar: 600\n", encoding="utf-8")

    monkeypatch.setattr(ec_module, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(ec_module, "SAFETY_FLOOR_PATH", safety_path)
    monkeypatch.setattr(ec_module, "TUNES_PATH", tunes_path)
    # queue.py imported SETTINGS_PATH by value; patch its local binding too
    # so status_snapshot's baseline-delta calc reads the fixture settings.
    monkeypatch.setattr(queue_module, "SETTINGS_PATH", settings_path)

    return settings_path, safety_path, tunes_path


def _make_queue(tmp_path, monkeypatch, now=None):
    _patch_config_paths(tmp_path, monkeypatch)
    effective_config = EffectiveConfig.load()

    config = Config(
        settings={"monitoring": {"trade_journal_db": str(tmp_path / "trades.db")}},
        instruments={"instruments": {"EUR_USD": {"enabled": True}, "GBP_USD": {"enabled": True}}},
    )
    journal = TradeJournal(config)
    telegram = FakeTelegram()
    risk_manager = FakeRiskManager()

    clock_box = {"now": now or datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)}

    queue = ControlQueue(
        config=config,
        journal=journal,
        telegram=telegram,
        risk_manager=risk_manager,
        effective_config=effective_config,
        inbox_dir=tmp_path / "inbox",
        outbox_dir=tmp_path / "outbox",
        clock=lambda: clock_box["now"],
    )
    return queue, journal, telegram, risk_manager, clock_box


def _write_cmd(queue, cmd_id, verb, args=None, reason="", requested_by="tester", requested_at=None):
    cmd = {
        "id": cmd_id,
        "verb": verb,
        "args": args or {},
        "reason": reason,
        "requested_at": requested_at or datetime.now(timezone.utc).isoformat(),
        "requested_by": requested_by,
    }
    tmp_path = queue.inbox_dir / f"{cmd_id}.cmd.json.tmp"
    final_path = queue.inbox_dir / f"{cmd_id}.cmd.json"
    tmp_path.write_text(json.dumps(cmd), encoding="utf-8")
    os.replace(tmp_path, final_path)
    return final_path


def _read_outbox(queue, cmd_id):
    path = queue.outbox_dir / f"{cmd_id}.result.json"
    assert path.exists(), f"expected outbox result for {cmd_id}"
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# pause / resume
# ---------------------------------------------------------------------------

def test_pause_round_trip(tmp_path, monkeypatch):
    queue, journal, telegram, risk_manager, _ = _make_queue(tmp_path, monkeypatch)
    inbox_file = _write_cmd(queue, "cmd1", "pause", reason="operator requested pause")

    processed = queue.poll_once()

    assert processed == 1
    assert not inbox_file.exists()
    result = _read_outbox(queue, "cmd1")
    assert result["outcome"] == "applied"
    assert risk_manager.manual_paused is True
    assert any("received" in t for t in telegram.sent)
    assert any("applied" in t for t in telegram.sent)


def test_pause_rejects_short_reason(tmp_path, monkeypatch):
    queue, journal, telegram, risk_manager, _ = _make_queue(tmp_path, monkeypatch)
    _write_cmd(queue, "cmd1", "pause", reason="short")

    queue.poll_once()

    result = _read_outbox(queue, "cmd1")
    assert result["outcome"] == "rejected"
    assert risk_manager.manual_paused is False


def test_resume_round_trip_clears_pause(tmp_path, monkeypatch):
    queue, journal, telegram, risk_manager, _ = _make_queue(tmp_path, monkeypatch)
    risk_manager.set_manual_pause("previously paused for testing")

    _write_cmd(queue, "cmd1", "resume", reason="operator lifted the pause")
    queue.poll_once()

    result = _read_outbox(queue, "cmd1")
    assert result["outcome"] == "applied"
    assert risk_manager.manual_paused is False


# ---------------------------------------------------------------------------
# tune
# ---------------------------------------------------------------------------

def test_tune_round_trip_applies_and_persists(tmp_path, monkeypatch):
    queue, journal, telegram, risk_manager, _ = _make_queue(tmp_path, monkeypatch)
    inbox_file = _write_cmd(
        queue, "cmd1", "tune",
        args={"key": "risk.risk_per_trade_pct", "value": 2.0},
        reason="reducing exposure this week",
    )

    queue.poll_once()

    assert not inbox_file.exists()
    result = _read_outbox(queue, "cmd1")
    assert result["outcome"] == "applied"
    assert queue.effective_config.get("risk.risk_per_trade_pct") == 2.0

    # A fresh load() reflects the tune without restart.
    reloaded = EffectiveConfig.load()
    assert reloaded.get("risk.risk_per_trade_pct") == 2.0


def test_tune_out_of_bounds_rejected(tmp_path, monkeypatch):
    queue, journal, telegram, risk_manager, _ = _make_queue(tmp_path, monkeypatch)
    _write_cmd(
        queue, "cmd1", "tune",
        args={"key": "risk.risk_per_trade_pct", "value": 5.0},
        reason="way too aggressive tune",
    )

    queue.poll_once()

    result = _read_outbox(queue, "cmd1")
    assert result["outcome"] == "rejected"
    assert "bounds" in result["detail"]
    # Unchanged.
    assert queue.effective_config.get("risk.risk_per_trade_pct") == 1.5


def test_tune_key_not_whitelisted_rejected(tmp_path, monkeypatch):
    queue, journal, telegram, risk_manager, _ = _make_queue(tmp_path, monkeypatch)
    _write_cmd(
        queue, "cmd1", "tune",
        args={"key": "trading.max_trades_per_day", "value": 100},
        reason="trying to tune a non-whitelisted key",
    )

    queue.poll_once()

    result = _read_outbox(queue, "cmd1")
    assert result["outcome"] == "rejected"
    assert "not whitelisted" in result["detail"]


def test_tune_safety_locked_key_rejected(tmp_path, monkeypatch):
    queue, journal, telegram, risk_manager, _ = _make_queue(tmp_path, monkeypatch)
    # risk.min_floor_zar isn't in TUNE_BOUNDS at all, so exercise the
    # safety-lock path via a key that IS whitelisted-shaped but also
    # happens to be safety-locked isn't possible with the fixed whitelist;
    # instead assert is_safety_locked is consulted for weight.* once an
    # instrument key is also present in safety_floor.yaml.
    settings_path, safety_path, _tunes = _patch_config_paths(tmp_path, monkeypatch)
    safety_path.write_text(
        "risk:\n  min_floor_zar: 600\nweight:\n  EUR_USD: 1.0\n", encoding="utf-8"
    )
    queue.effective_config = EffectiveConfig.load()

    _write_cmd(
        queue, "cmd1", "tune",
        args={"key": "weight.EUR_USD", "value": 0.5},
        reason="trying to tune a safety-locked weight",
    )

    queue.poll_once()

    result = _read_outbox(queue, "cmd1")
    assert result["outcome"] == "rejected"
    assert "safety-locked" in result["detail"]


def test_weight_whitelist_validates_against_instruments_yaml(tmp_path, monkeypatch):
    queue, journal, telegram, risk_manager, _ = _make_queue(tmp_path, monkeypatch)

    _write_cmd(
        queue, "cmd_ok", "tune",
        args={"key": "weight.EUR_USD", "value": 0.8},
        reason="EUR_USD is a real configured instrument",
    )
    _write_cmd(
        queue, "cmd_bad", "tune",
        args={"key": "weight.NOT_REAL", "value": 0.8},
        reason="NOT_REAL is not a configured instrument",
        requested_by="someone_else",
    )

    queue.poll_once()

    assert _read_outbox(queue, "cmd_ok")["outcome"] == "applied"
    assert _read_outbox(queue, "cmd_bad")["outcome"] == "rejected"


def test_ml_low_high_cross_check(tmp_path, monkeypatch):
    queue, journal, telegram, risk_manager, _ = _make_queue(tmp_path, monkeypatch)
    # Current effective low is 0.55; tuning high down to 0.5 would violate
    # low<=high (0.5 is within the high key's own [0.50, 0.75] bounds).
    _write_cmd(
        queue, "cmd1", "tune",
        args={"key": "ml.confidence_threshold_high", "value": 0.5},
        reason="pushing high below low on purpose",
    )

    queue.poll_once()

    result = _read_outbox(queue, "cmd1")
    assert result["outcome"] == "rejected"
    assert "must be <=" in result["detail"]


def test_manual_tune_rate_limited_second_within_24h_rejected(tmp_path, monkeypatch):
    queue, journal, telegram, risk_manager, clock_box = _make_queue(tmp_path, monkeypatch)

    _write_cmd(
        queue, "cmd1", "tune",
        args={"key": "risk.risk_per_trade_pct", "value": 1.8},
        reason="first manual tune of the day",
        requested_by="operator",
    )
    queue.poll_once()
    assert _read_outbox(queue, "cmd1")["outcome"] == "applied"

    clock_box["now"] += timedelta(hours=1)
    _write_cmd(
        queue, "cmd2", "tune",
        args={"key": "risk.risk_per_trade_pct", "value": 2.0},
        reason="second manual tune same day",
        requested_by="operator",
    )
    queue.poll_once()

    result = _read_outbox(queue, "cmd2")
    assert result["outcome"] == "rejected"
    assert "rate limit" in result["detail"]
    # First tune's value sticks.
    assert queue.effective_config.get("risk.risk_per_trade_pct") == 1.8


def test_manual_tune_allowed_after_24h_window(tmp_path, monkeypatch):
    queue, journal, telegram, risk_manager, clock_box = _make_queue(tmp_path, monkeypatch)

    _write_cmd(
        queue, "cmd1", "tune",
        args={"key": "risk.risk_per_trade_pct", "value": 1.8},
        reason="first manual tune of the day",
        requested_by="operator",
    )
    queue.poll_once()

    clock_box["now"] += timedelta(hours=25)
    _write_cmd(
        queue, "cmd2", "tune",
        args={"key": "risk.risk_per_trade_pct", "value": 2.0},
        reason="tune after rate limit window expired",
        requested_by="operator",
    )
    queue.poll_once()

    assert _read_outbox(queue, "cmd2")["outcome"] == "applied"
    assert queue.effective_config.get("risk.risk_per_trade_pct") == 2.0


def test_manager_tunes_exempt_from_rate_limit(tmp_path, monkeypatch):
    queue, journal, telegram, risk_manager, clock_box = _make_queue(tmp_path, monkeypatch)

    _write_cmd(
        queue, "cmd1", "tune",
        args={"key": "risk.risk_per_trade_pct", "value": 1.8},
        reason="manager tune number one",
        requested_by="manager",
    )
    queue.poll_once()

    clock_box["now"] += timedelta(minutes=5)
    _write_cmd(
        queue, "cmd2", "tune",
        args={"key": "risk.risk_per_trade_pct", "value": 2.0},
        reason="manager tune number two same day",
        requested_by="manager",
    )
    queue.poll_once()

    assert _read_outbox(queue, "cmd1")["outcome"] == "applied"
    assert _read_outbox(queue, "cmd2")["outcome"] == "applied"
    assert queue.effective_config.get("risk.risk_per_trade_pct") == 2.0


# ---------------------------------------------------------------------------
# revert
# ---------------------------------------------------------------------------

def test_revert_restores_prior_value(tmp_path, monkeypatch):
    queue, journal, telegram, risk_manager, clock_box = _make_queue(tmp_path, monkeypatch)

    _write_cmd(
        queue, "cmd1", "tune",
        args={"key": "risk.risk_per_trade_pct", "value": 2.0},
        reason="tuning up risk per trade",
        requested_by="manager",
    )
    queue.poll_once()
    assert queue.effective_config.get("risk.risk_per_trade_pct") == 2.0

    clock_box["now"] += timedelta(minutes=1)
    _write_cmd(queue, "cmd2", "revert", reason="undoing the risk tune from earlier")
    queue.poll_once()

    result = _read_outbox(queue, "cmd2")
    assert result["outcome"] == "applied"
    assert queue.effective_config.get("risk.risk_per_trade_pct") == 1.5

    reloaded = EffectiveConfig.load()
    assert reloaded.get("risk.risk_per_trade_pct") == 1.5


def test_revert_with_no_prior_tune_rejected(tmp_path, monkeypatch):
    queue, journal, telegram, risk_manager, _ = _make_queue(tmp_path, monkeypatch)
    _write_cmd(queue, "cmd1", "revert", reason="nothing to revert yet")

    queue.poll_once()

    result = _read_outbox(queue, "cmd1")
    assert result["outcome"] == "rejected"


# ---------------------------------------------------------------------------
# status_snapshot
# ---------------------------------------------------------------------------

def test_status_snapshot_round_trip_no_control_log_row(tmp_path, monkeypatch):
    queue, journal, telegram, risk_manager, _ = _make_queue(tmp_path, monkeypatch)
    inbox_file = _write_cmd(queue, "cmd1", "status_snapshot", reason="")

    queue.poll_once()

    assert not inbox_file.exists()
    result = _read_outbox(queue, "cmd1")
    assert result["outcome"] == "applied"
    assert isinstance(result["detail"], dict)
    assert result["detail"]["bot_up"] is True
    assert result["detail"]["entries_blocked"] is False

    # Read-only: no control_log row and no Telegram messages for this verb.
    log = journal.get_control_log()
    assert log.empty
    assert telegram.sent == []


# ---------------------------------------------------------------------------
# atomicity
# ---------------------------------------------------------------------------

def test_stray_tmp_file_never_picked_up(tmp_path, monkeypatch):
    queue, journal, telegram, risk_manager, _ = _make_queue(tmp_path, monkeypatch)

    # A command mid-write, still under its .tmp name (writer hasn't
    # os.replace()'d it into place yet).
    stray = queue.inbox_dir / "half_written.cmd.json.tmp"
    stray.write_text('{"id": "half_written", "verb": "pause"', encoding="utf-8")  # truncated / invalid

    processed = queue.poll_once()

    assert processed == 0
    assert stray.exists()  # untouched
    assert not (queue.outbox_dir / "half_written.result.json").exists()
    assert risk_manager.manual_paused is False


def test_oldest_first_ordering(tmp_path, monkeypatch):
    queue, journal, telegram, risk_manager, _ = _make_queue(tmp_path, monkeypatch)

    _write_cmd(
        queue, "second", "tune",
        args={"key": "risk.risk_per_trade_pct", "value": 2.0},
        reason="this was requested second",
        requested_at="2026-01-01T12:00:05+00:00",
        requested_by="manager",
    )
    _write_cmd(
        queue, "first", "tune",
        args={"key": "risk.risk_per_trade_pct", "value": 1.7},
        reason="this was requested first",
        requested_at="2026-01-01T12:00:00+00:00",
        requested_by="manager",
    )

    queue.poll_once()

    # Both applied, but "second" (processed last) should be the final value.
    assert queue.effective_config.get("risk.risk_per_trade_pct") == 2.0


# ---------------------------------------------------------------------------
# control_log lifecycle
# ---------------------------------------------------------------------------

def test_control_log_lifecycle_pending_then_applied(tmp_path, monkeypatch):
    queue, journal, telegram, risk_manager, _ = _make_queue(tmp_path, monkeypatch)
    _write_cmd(queue, "cmd1", "pause", reason="checking control_log lifecycle")

    queue.poll_once()

    log = journal.get_control_log(verb="pause")
    assert len(log) == 1
    row = log.iloc[0]
    assert row["outcome"] == "applied"
    assert row["reason"] == "checking control_log lifecycle"


def test_control_log_lifecycle_rejected(tmp_path, monkeypatch):
    queue, journal, telegram, risk_manager, _ = _make_queue(tmp_path, monkeypatch)
    _write_cmd(queue, "cmd1", "pause", reason="x")  # too short -> rejected

    queue.poll_once()

    log = journal.get_control_log(verb="pause")
    assert len(log) == 1
    assert log.iloc[0]["outcome"] == "rejected"


def test_control_log_lifecycle_error_when_journal_lookup_raises(tmp_path, monkeypatch):
    queue, journal, telegram, risk_manager, _ = _make_queue(tmp_path, monkeypatch)

    def boom(*args, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(risk_manager, "set_manual_pause", boom)
    _write_cmd(queue, "cmd1", "pause", reason="this pause will explode")

    queue.poll_once()

    result = _read_outbox(queue, "cmd1")
    assert result["outcome"] == "error"
    log = journal.get_control_log(verb="pause")
    assert log.iloc[0]["outcome"] == "error"
