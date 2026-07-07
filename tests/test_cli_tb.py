"""
Task 9 — `tb` CLI (cli/tb.py).

Covers, per the task-9 brief:
- `python -m cli.tb <cmd>` -> single JSON document on stdout, exit 0/1.
- Read commands: status (round-trip + degrade), trades, perf, positions,
  config, logs, model, manager (incl. --verdict stub and the
  table-existence guard before manager_log exists).
- Write commands: pause/resume/tune/revert enqueue via the Task 8 queue
  and wait (bounded) for an outbox result.
- Client-side --reason validation (fail fast, no round trip needed).
- Errors are always {"error": ...} JSON + exit 1, never a bare traceback.

Each round-trip test seeds a tmp journal DB and starts a fake outbox
responder thread that mimics ControlQueue's inbox->outbox behavior,
instead of spinning up a real bot / MT5 / network.
"""
import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import cli.tb as tb
from src.config import Config
from src.monitoring.trade_journal import TradeJournal

FAST_TIMEOUT = 1.0
FAST_POLL = 0.02


def make_config(tmp_path: Path) -> Config:
    return Config(settings={
        "monitoring": {"trade_journal_db": str(tmp_path / "trades.db")},
    }, instruments={"instruments": {"EUR_USD": {"enabled": True}}})


class FakeResponder:
    """
    Watches `inbox_dir` and writes a matching outbox result for every
    `*.cmd.json` it sees, simulating ControlQueue.poll_once() without
    depending on the real bot process.
    """

    def __init__(self, inbox_dir: Path, outbox_dir: Path, handler=None):
        self.inbox_dir = inbox_dir
        self.outbox_dir = outbox_dir
        self.handler = handler or self._default_handler
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        self.outbox_dir.mkdir(parents=True, exist_ok=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self):
        while not self._stop.is_set():
            for path in list(self.inbox_dir.glob("*.cmd.json")):
                try:
                    cmd = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                cmd_id = cmd["id"]
                outcome, detail = self.handler(cmd)
                result = {
                    "id": cmd_id, "outcome": outcome, "detail": detail,
                    "applied_at": datetime.now(timezone.utc).isoformat(),
                }
                tmp = self.outbox_dir / f"{cmd_id}.result.json.tmp"
                final = self.outbox_dir / f"{cmd_id}.result.json"
                tmp.write_text(json.dumps(result), encoding="utf-8")
                os.replace(tmp, final)
                try:
                    path.unlink()
                except OSError:
                    pass
            time.sleep(0.01)

    @staticmethod
    def _default_handler(cmd: dict):
        verb = cmd["verb"]
        if verb == "status_snapshot":
            return "applied", {
                "bot_up": True, "broker_connected": True, "open_positions": 0,
                "manual_paused": False, "balance": 1000.0, "equity": 1000.0,
            }
        if verb in ("pause", "resume", "revert"):
            return "applied", f"{verb} ok"
        if verb == "tune":
            return "applied", f"tuned {cmd['args']['key']} -> {cmd['args']['value']}"
        return "error", f"unhandled verb in fake responder: {verb}"


@pytest.fixture
def paths(tmp_path):
    return {
        "inbox": tmp_path / "control" / "inbox",
        "outbox": tmp_path / "control" / "outbox",
    }


@pytest.fixture
def journal(tmp_path):
    return TradeJournal(make_config(tmp_path))


def insert_trade(journal: TradeJournal, trade_id, instrument="EUR_USD", pnl=10.0,
                  entry_time=None, exit_price=1.1, net_pnl=None):
    entry_time = entry_time or datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(journal.db_path) as conn:
        conn.execute(
            """INSERT INTO trades
               (trade_id, instrument, direction, units, entry_price, exit_price,
                entry_time, exit_time, pnl_zar, net_pnl_zar)
               VALUES (?, ?, 'buy', 1000, 1.10, ?, ?, ?, ?, ?)""",
            (trade_id, instrument, exit_price, entry_time,
             entry_time if exit_price is not None else None,
             pnl, net_pnl if net_pnl is not None else pnl),
        )


# ---------------------------------------------------------------------
# Read commands
# ---------------------------------------------------------------------

def test_status_round_trip_bot_running(journal, paths):
    responder = FakeResponder(paths["inbox"], paths["outbox"])
    responder.start()
    try:
        rc = tb.main(
            ["status"], config=journal.config, journal=journal,
            inbox_dir=paths["inbox"], outbox_dir=paths["outbox"],
            timeout=FAST_TIMEOUT, poll_interval=FAST_POLL,
        )
    finally:
        responder.stop()
    assert rc == 0


def test_status_degrades_when_bot_not_running(journal, paths, capsys):
    rc = tb.main(
        ["status"], config=journal.config, journal=journal,
        inbox_dir=paths["inbox"], outbox_dir=paths["outbox"],
        timeout=0.3, poll_interval=FAST_POLL,
    )
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["bot_running"] is False
    assert "todays_pnl" in out


def test_trades_command(journal, paths, capsys):
    insert_trade(journal, "t1")
    insert_trade(journal, "t2")
    rc = tb.main(["trades"], config=journal.config, journal=journal,
                  inbox_dir=paths["inbox"], outbox_dir=paths["outbox"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["count"] == 2


def test_trades_days_filter_excludes_old_trades(journal, paths, capsys):
    old_time = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    insert_trade(journal, "old", entry_time=old_time)
    insert_trade(journal, "recent")
    rc = tb.main(["trades", "--days", "1"], config=journal.config, journal=journal,
                  inbox_dir=paths["inbox"], outbox_dir=paths["outbox"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["count"] == 1
    assert out["trades"][0]["trade_id"] == "recent"


def test_perf_command_computes_win_rate(journal, paths, capsys):
    insert_trade(journal, "win1", pnl=50.0)
    insert_trade(journal, "loss1", pnl=-20.0)
    rc = tb.main(["perf"], config=journal.config, journal=journal,
                  inbox_dir=paths["inbox"], outbox_dir=paths["outbox"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["total_trades"] == 2
    assert out["wins"] == 1
    assert out["losses"] == 1
    assert out["win_rate"] == pytest.approx(0.5)
    assert out["total_pnl"] == pytest.approx(30.0)


def test_perf_empty_journal(journal, paths, capsys):
    rc = tb.main(["perf"], config=journal.config, journal=journal,
                  inbox_dir=paths["inbox"], outbox_dir=paths["outbox"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["total_trades"] == 0
    assert out["profit_factor"] is None


def test_positions_command_only_open(journal, paths, capsys):
    insert_trade(journal, "closed", exit_price=1.1)
    insert_trade(journal, "open", exit_price=None)
    rc = tb.main(["positions"], config=journal.config, journal=journal,
                  inbox_dir=paths["inbox"], outbox_dir=paths["outbox"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["count"] == 1
    assert out["positions"][0]["trade_id"] == "open"


def test_config_command_shows_delta(journal, paths, monkeypatch, tmp_path, capsys):
    # Point the effective-config module at an isolated settings file +
    # tunes overlay so this test doesn't touch the real repo config.
    import src.control.effective_config as ec

    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text("risk:\n  risk_per_trade_pct: 1.5\n", encoding="utf-8")
    tunes_path = tmp_path / "effective_config.json"
    tunes_path.write_text(json.dumps({"risk": {"risk_per_trade_pct": 2.0}}), encoding="utf-8")
    safety_path = tmp_path / "safety_floor.yaml"

    monkeypatch.setattr(ec, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(ec, "SAFETY_FLOOR_PATH", safety_path)
    monkeypatch.setattr(ec, "TUNES_PATH", tunes_path)
    monkeypatch.setattr(tb, "SETTINGS_PATH", settings_path)

    rc = tb.main(["config"], config=journal.config, journal=journal,
                  inbox_dir=paths["inbox"], outbox_dir=paths["outbox"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["effective_config_delta"]["risk.risk_per_trade_pct"] == {
        "baseline": 1.5, "effective": 2.0,
    }


def test_logs_command_tail_and_level(journal, paths, monkeypatch, tmp_path, capsys):
    log_path = tmp_path / "traderbot.log"
    log_path.write_text(
        "2026-07-07 10:00:00 | traderbot | INFO | started\n"
        "2026-07-07 10:00:01 | traderbot | ERROR | boom\n"
        "2026-07-07 10:00:02 | traderbot | INFO | still running\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(tb, "LOG_PATH", log_path)

    rc = tb.main(["logs", "--level", "ERROR"], config=journal.config, journal=journal,
                  inbox_dir=paths["inbox"], outbox_dir=paths["outbox"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert len(out["lines"]) == 1
    assert "boom" in out["lines"][0]


def test_logs_command_missing_file(journal, paths, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(tb, "LOG_PATH", tmp_path / "does_not_exist.log")
    rc = tb.main(["logs"], config=journal.config, journal=journal,
                  inbox_dir=paths["inbox"], outbox_dir=paths["outbox"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["lines"] == []


def test_model_command_no_model_trained(journal, paths, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(tb, "MODEL_STORE", tmp_path / "model_store")
    rc = tb.main(["model"], config=journal.config, journal=journal,
                  inbox_dir=paths["inbox"], outbox_dir=paths["outbox"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["version"] is None


def test_model_command_reads_metadata(journal, paths, monkeypatch, tmp_path, capsys):
    store = tmp_path / "model_store"
    store.mkdir()
    (store / "latest_version.txt").write_text("v1.11", encoding="utf-8")
    (store / "model_v1.11_meta.json").write_text(
        json.dumps({"version": "v1.11", "metrics": {"accuracy": 0.6}}), encoding="utf-8",
    )
    monkeypatch.setattr(tb, "MODEL_STORE", store)

    rc = tb.main(["model"], config=journal.config, journal=journal,
                  inbox_dir=paths["inbox"], outbox_dir=paths["outbox"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["version"] == "v1.11"
    assert out["metadata"]["metrics"]["accuracy"] == 0.6
    assert out["evaluator_state"] is None  # table doesn't exist yet in this fresh journal


def test_manager_command_before_task_12_returns_empty(journal, paths, capsys):
    rc = tb.main(["manager"], config=journal.config, journal=journal,
                  inbox_dir=paths["inbox"], outbox_dir=paths["outbox"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out == {"count": 0, "entries": []}


def test_manager_command_reads_existing_manager_log(journal, paths, capsys):
    # manager_log table now created by TradeJournal schema init (Task 12);
    # this test only needs to insert a row and confirm the CLI reads it back.
    journal.log_manager_cycle(trigger="test", outcome="checked in")

    rc = tb.main(["manager"], config=journal.config, journal=journal,
                  inbox_dir=paths["inbox"], outbox_dir=paths["outbox"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["count"] == 1
    assert out["entries"][0]["outcome"] == "checked in"


def test_manager_verdict_stub(journal, paths, capsys):
    rc = tb.main(["manager", "--verdict"], config=journal.config, journal=journal,
                  inbox_dir=paths["inbox"], outbox_dir=paths["outbox"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out == {"verdict": "PENDING", "reason": "manager not yet active"}


# ---------------------------------------------------------------------
# Write commands
# ---------------------------------------------------------------------

def test_pause_round_trip_applied(journal, paths, capsys):
    responder = FakeResponder(paths["inbox"], paths["outbox"])
    responder.start()
    try:
        rc = tb.main(
            ["pause", "--reason", "manual pause for testing"],
            config=journal.config, journal=journal,
            inbox_dir=paths["inbox"], outbox_dir=paths["outbox"],
            timeout=FAST_TIMEOUT, poll_interval=FAST_POLL,
        )
    finally:
        responder.stop()
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["outcome"] == "applied"


def test_resume_round_trip_applied(journal, paths, capsys):
    responder = FakeResponder(paths["inbox"], paths["outbox"])
    responder.start()
    try:
        rc = tb.main(
            ["resume", "--reason", "manual resume for testing"],
            config=journal.config, journal=journal,
            inbox_dir=paths["inbox"], outbox_dir=paths["outbox"],
            timeout=FAST_TIMEOUT, poll_interval=FAST_POLL,
        )
    finally:
        responder.stop()
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["outcome"] == "applied"


def test_tune_round_trip_applied(journal, paths, capsys):
    responder = FakeResponder(paths["inbox"], paths["outbox"])
    responder.start()
    try:
        rc = tb.main(
            ["tune", "risk.risk_per_trade_pct=2.0", "--reason", "testing tune roundtrip"],
            config=journal.config, journal=journal,
            inbox_dir=paths["inbox"], outbox_dir=paths["outbox"],
            timeout=FAST_TIMEOUT, poll_interval=FAST_POLL,
        )
    finally:
        responder.stop()
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["outcome"] == "applied"


def test_revert_round_trip_applied(journal, paths, capsys):
    responder = FakeResponder(paths["inbox"], paths["outbox"])
    responder.start()
    try:
        rc = tb.main(
            ["revert", "--reason", "testing revert roundtrip"],
            config=journal.config, journal=journal,
            inbox_dir=paths["inbox"], outbox_dir=paths["outbox"],
            timeout=FAST_TIMEOUT, poll_interval=FAST_POLL,
        )
    finally:
        responder.stop()
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["outcome"] == "applied"


def test_pause_rejected_short_reason_fails_fast_no_round_trip(journal, paths, capsys):
    # No responder running at all — if this were a real round trip it
    # would time out; a fast client-side rejection must not depend on
    # timeout at all.
    rc = tb.main(
        ["pause", "--reason", "short"],
        config=journal.config, journal=journal,
        inbox_dir=paths["inbox"], outbox_dir=paths["outbox"],
        timeout=FAST_TIMEOUT, poll_interval=FAST_POLL,
    )
    out = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert "error" in out
    assert "reason" in out["error"]


def test_pause_missing_reason_fails_fast(journal, paths, capsys):
    rc = tb.main(
        ["pause"], config=journal.config, journal=journal,
        inbox_dir=paths["inbox"], outbox_dir=paths["outbox"],
        timeout=FAST_TIMEOUT, poll_interval=FAST_POLL,
    )
    out = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert "error" in out


def test_tune_bad_assignment_format_fails_fast(journal, paths, capsys):
    rc = tb.main(
        ["tune", "not-a-key-value-pair", "--reason", "testing bad tune format"],
        config=journal.config, journal=journal,
        inbox_dir=paths["inbox"], outbox_dir=paths["outbox"],
        timeout=FAST_TIMEOUT, poll_interval=FAST_POLL,
    )
    out = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert "error" in out


def test_pause_times_out_when_bot_not_running(journal, paths, capsys):
    # No responder started -> nothing ever drains the inbox.
    rc = tb.main(
        ["pause", "--reason", "manual pause with nobody listening"],
        config=journal.config, journal=journal,
        inbox_dir=paths["inbox"], outbox_dir=paths["outbox"],
        timeout=0.3, poll_interval=FAST_POLL,
    )
    out = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert "error" in out
    assert "timed out" in out["error"]


def test_tune_rejected_by_bot_reports_nonzero_exit(journal, paths, capsys):
    def handler(cmd):
        return "rejected", "value out of bounds"

    responder = FakeResponder(paths["inbox"], paths["outbox"], handler=handler)
    responder.start()
    try:
        rc = tb.main(
            ["tune", "risk.risk_per_trade_pct=99", "--reason", "testing rejection path"],
            config=journal.config, journal=journal,
            inbox_dir=paths["inbox"], outbox_dir=paths["outbox"],
            timeout=FAST_TIMEOUT, poll_interval=FAST_POLL,
        )
    finally:
        responder.stop()
    out = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert out["outcome"] == "rejected"


def test_inbox_file_uses_tmp_then_replace(journal, paths):
    """
    Sanity check on the writer contract itself: enqueue_command must
    never leave a bare .cmd.json.tmp visible as a .cmd.json (it should
    already have been replaced by the time the call returns).
    """
    def handler(cmd):
        return "applied", "ok"

    responder = FakeResponder(paths["inbox"], paths["outbox"], handler=handler)
    responder.start()
    try:
        result = tb.enqueue_command(
            "status_snapshot", {}, reason="", requested_by="cli",
            inbox_dir=paths["inbox"], outbox_dir=paths["outbox"],
            timeout=FAST_TIMEOUT, poll_interval=FAST_POLL,
        )
    finally:
        responder.stop()
    assert result is not None
    assert result["outcome"] == "applied"
    assert not list(paths["inbox"].glob("*.cmd.json.tmp"))
