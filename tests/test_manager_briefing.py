"""
Task 12 — src/manager/briefing.py: build().

Covers:
- Schema: required top-level keys present.
- Size guard: serialized briefing <= MAX_BRIEFING_CHARS, with truncation
  of trade-list-like fields (keep newest) when the raw data would
  otherwise overflow the cap.
- Degraded-source cases: missing model_store meta -> "unknown" version;
  no evaluator_trades table -> recent_accuracy None; empty journal ->
  zeroed/empty stats, never raises.
- growth_stage / risk_ceiling_now come from policy helpers and match
  balance vs configured milestones.
- Per-instrument weights read from effective_config (default 1.0).
"""
import json
from datetime import datetime, timedelta, timezone

import src.control.effective_config as ec_module
import src.manager.briefing as briefing_module
from src.config import Config
from src.control.effective_config import EffectiveConfig
from src.monitoring.trade_journal import TradeJournal
from src.risk.ratchet_floor import RatchetFloor


def _journal(tmp_path):
    config = Config(settings={
        "monitoring": {"trade_journal_db": str(tmp_path / "trades.db")}
    })
    return TradeJournal(config)


def _eff(monkeypatch, tmp_path, settings_yaml=None):
    settings_path = tmp_path / "settings.yaml"
    safety_path = tmp_path / "safety_floor.yaml"
    tunes_path = tmp_path / "control" / "effective_config.json"

    settings_path.write_text(
        settings_yaml or (
            "risk:\n  risk_per_trade_pct: 1.5\n  daily_drawdown_limit_pct: 4.0\n"
            "ml:\n  confidence_threshold_high: 0.65\n  confidence_threshold_low: 0.55\n"
            "growth:\n  milestones: [1500, 2000, 3000, 4500, 6000]\n"
            "weight:\n  EUR_USD: 1.2\n"
        ),
        encoding="utf-8",
    )
    safety_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(ec_module, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(ec_module, "SAFETY_FLOOR_PATH", safety_path)
    monkeypatch.setattr(ec_module, "TUNES_PATH", tunes_path)
    return EffectiveConfig.load()


def _instruments(monkeypatch, tmp_path, names=("EUR_USD", "XAU_USD")):
    instruments_path = tmp_path / "instruments.yaml"
    body = "instruments:\n" + "".join(f"  {n}:\n    enabled: true\n" for n in names)
    instruments_path.write_text(body, encoding="utf-8")
    monkeypatch.setattr(briefing_module.policy, "INSTRUMENTS_PATH", instruments_path)


def _floor(tmp_path):
    return RatchetFloor(
        min_floor_zar=600.0,
        max_total_drawdown_pct=0.30,
        state_path=tmp_path / "account_state.json",
    )


def _no_model_store(monkeypatch, tmp_path):
    monkeypatch.setattr(briefing_module, "MODEL_STORE_PATH", tmp_path / "no_such_model_store")


def _build(monkeypatch, tmp_path, balance=1000.0, equity=1000.0, extra=None, names=("EUR_USD", "XAU_USD")):
    journal = _journal(tmp_path)
    eff = _eff(monkeypatch, tmp_path)
    floor = _floor(tmp_path)
    _instruments(monkeypatch, tmp_path, names=names)
    _no_model_store(monkeypatch, tmp_path)
    return journal, eff, floor, briefing_module.build(journal, eff, floor, balance, equity, extra)


REQUIRED_KEYS = {
    "generated_at_utc", "balance", "equity", "floor", "headroom_to_floor",
    "today_pnl_zar", "drawdown_vs_cap", "instruments", "open_positions",
    "config_delta", "model_version", "recent_accuracy", "growth_stage",
    "risk_ceiling_now", "milestones", "last_manager_actions",
}


def test_schema_has_required_keys(monkeypatch, tmp_path):
    _, _, _, briefing = _build(monkeypatch, tmp_path)
    assert REQUIRED_KEYS <= set(briefing.keys())


def test_empty_journal_never_raises_and_degrades(monkeypatch, tmp_path):
    _, _, _, briefing = _build(monkeypatch, tmp_path)
    assert briefing["today_pnl_zar"] == 0.0
    assert briefing["open_positions"] == []
    assert briefing["last_manager_actions"] == []
    for instrument, stats in briefing["instruments"].items():
        assert stats["trades"] == 0
        assert stats["win_rate"] is None


def test_model_version_unknown_when_model_store_missing(monkeypatch, tmp_path):
    _, _, _, briefing = _build(monkeypatch, tmp_path)
    assert briefing["model_version"] == "unknown"


def test_model_version_read_from_latest_version_file(monkeypatch, tmp_path):
    journal = _journal(tmp_path)
    eff = _eff(monkeypatch, tmp_path)
    floor = _floor(tmp_path)
    _instruments(monkeypatch, tmp_path)

    model_store = tmp_path / "model_store"
    model_store.mkdir()
    (model_store / "latest_version.txt").write_text("v1.40", encoding="utf-8")
    monkeypatch.setattr(briefing_module, "MODEL_STORE_PATH", model_store)

    briefing = briefing_module.build(journal, eff, floor, 1000.0, 1000.0)
    assert briefing["model_version"] == "v1.40"


def test_recent_accuracy_none_when_no_evaluator_table(monkeypatch, tmp_path):
    _, _, _, briefing = _build(monkeypatch, tmp_path)
    assert briefing["recent_accuracy"] is None


def test_growth_stage_and_risk_ceiling_match_policy(monkeypatch, tmp_path):
    _, _, _, briefing = _build(monkeypatch, tmp_path, balance=2000.0, equity=2000.0)
    assert briefing["growth_stage"] == 2
    assert briefing["risk_ceiling_now"] == 2.0


def test_milestones_state_reached_flags(monkeypatch, tmp_path):
    _, _, _, briefing = _build(monkeypatch, tmp_path, balance=2000.0, equity=2000.0)
    reached = {m["milestone"]: m["reached"] for m in briefing["milestones"]}
    assert reached[1500] is True
    assert reached[2000] is True
    assert reached[3000] is False


def test_per_instrument_weight_defaults_to_one(monkeypatch, tmp_path):
    _, _, _, briefing = _build(monkeypatch, tmp_path, names=("EUR_USD", "XAU_USD"))
    assert briefing["instruments"]["EUR_USD"]["current_weight"] == 1.2
    assert briefing["instruments"]["XAU_USD"]["current_weight"] == 1.0


def test_floor_and_headroom(monkeypatch, tmp_path):
    _, _, floor, briefing = _build(monkeypatch, tmp_path, balance=1000.0, equity=1000.0)
    assert briefing["floor"] == floor.current_floor
    assert briefing["headroom_to_floor"] == 1000.0 - floor.current_floor


def test_last_manager_actions_populated(monkeypatch, tmp_path):
    journal = _journal(tmp_path)
    eff = _eff(monkeypatch, tmp_path)
    floor = _floor(tmp_path)
    _instruments(monkeypatch, tmp_path)
    _no_model_store(monkeypatch, tmp_path)

    for i in range(3):
        journal.log_manager_cycle(trigger=f"cycle-{i}", outcome="noop", rationale=f"r{i}")

    briefing = briefing_module.build(journal, eff, floor, 1000.0, 1000.0)
    assert len(briefing["last_manager_actions"]) == 3
    assert briefing["last_manager_actions"][0]["trigger"] == "cycle-2"


def test_last_manager_actions_capped_at_five(monkeypatch, tmp_path):
    journal = _journal(tmp_path)
    eff = _eff(monkeypatch, tmp_path)
    floor = _floor(tmp_path)
    _instruments(monkeypatch, tmp_path)
    _no_model_store(monkeypatch, tmp_path)

    for i in range(8):
        journal.log_manager_cycle(trigger=f"cycle-{i}", outcome="noop")

    briefing = briefing_module.build(journal, eff, floor, 1000.0, 1000.0)
    assert len(briefing["last_manager_actions"]) == 5
    assert briefing["last_manager_actions"][0]["trigger"] == "cycle-7"


def test_extra_merged_when_provided(monkeypatch, tmp_path):
    _, _, _, briefing = _build(monkeypatch, tmp_path, extra={"trigger_source": "telegram"})
    assert briefing["extra"] == {"trigger_source": "telegram"}


def test_extra_absent_when_not_provided(monkeypatch, tmp_path):
    _, _, _, briefing = _build(monkeypatch, tmp_path)
    assert "extra" not in briefing


# ---------------------------------------------------------------------------
# Size guard + truncation
# ---------------------------------------------------------------------------

def test_serialized_briefing_within_char_cap(monkeypatch, tmp_path):
    _, _, _, briefing = _build(monkeypatch, tmp_path)
    assert len(json.dumps(briefing)) <= briefing_module.MAX_BRIEFING_CHARS


def test_open_positions_truncated_keeping_newest_when_oversized(monkeypatch, tmp_path):
    journal = _journal(tmp_path)
    eff = _eff(monkeypatch, tmp_path)
    floor = _floor(tmp_path)
    _instruments(monkeypatch, tmp_path)
    _no_model_store(monkeypatch, tmp_path)

    import sqlite3
    now = datetime.now(timezone.utc)
    with sqlite3.connect(journal.db_path) as conn:
        for i in range(400):
            entry_time = (now - timedelta(minutes=i)).isoformat()
            conn.execute(
                """
                INSERT INTO trades (
                    trade_id, instrument, direction, units, entry_price,
                    entry_time, stop_loss, take_profit
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (f"open-{i}", "EUR_USD", "buy", 1000, 1.1000, entry_time, 1.0950, 1.1100),
            )

    briefing = briefing_module.build(journal, eff, floor, 1000.0, 1000.0)
    assert len(json.dumps(briefing)) <= briefing_module.MAX_BRIEFING_CHARS
    # Newest (smallest i, most recent entry_time) kept, oldest dropped.
    kept_ids = {p["trade_id"] for p in briefing["open_positions"]}
    assert "open-0" in kept_ids
    assert len(briefing["open_positions"]) < 400
