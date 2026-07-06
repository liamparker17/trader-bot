"""
Task 8 — EffectiveConfig.apply_tune().

Covers:
- apply_tune() persists a dotted-key tune to the tunes overlay JSON file
  (read-modify-write, merging with any other tuned keys already present).
- A fresh EffectiveConfig.load() after apply_tune() reflects the change
  (this is how "read through EffectiveConfig at use time" works without a
  bot restart).
- Safety-floor keys still win over a tune (apply_tune itself performs no
  validation — that's src/control/queue.py's job — but safety_floor.yaml
  is merged last in load(), so even a tuned safety key would be
  overridden on the next load()).
"""
import json

import src.control.effective_config as ec_module
from src.control.effective_config import EffectiveConfig


def _patch_paths(monkeypatch, tmp_path):
    settings_path = tmp_path / "settings.yaml"
    safety_path = tmp_path / "safety_floor.yaml"
    tunes_path = tmp_path / "control" / "effective_config.json"

    settings_path.write_text(
        "risk:\n  risk_per_trade_pct: 1.5\nml:\n  confidence_threshold_high: 0.65\n  confidence_threshold_low: 0.55\n",
        encoding="utf-8",
    )
    safety_path.write_text(
        "risk:\n  min_floor_zar: 600\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(ec_module, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(ec_module, "SAFETY_FLOOR_PATH", safety_path)
    monkeypatch.setattr(ec_module, "TUNES_PATH", tunes_path)
    return settings_path, safety_path, tunes_path


def test_apply_tune_persists_and_reloads(monkeypatch, tmp_path):
    _settings, _safety, tunes_path = _patch_paths(monkeypatch, tmp_path)

    eff = EffectiveConfig.load()
    assert eff.get("risk.risk_per_trade_pct") == 1.5

    eff.apply_tune("risk.risk_per_trade_pct", 2.0)

    # In-memory instance reflects it immediately.
    assert eff.get("risk.risk_per_trade_pct") == 2.0

    # Persisted to disk.
    assert tunes_path.exists()
    on_disk = json.loads(tunes_path.read_text(encoding="utf-8"))
    assert on_disk["risk"]["risk_per_trade_pct"] == 2.0

    # A fresh load() (simulating a different process / restart) sees it too.
    eff2 = EffectiveConfig.load()
    assert eff2.get("risk.risk_per_trade_pct") == 2.0


def test_apply_tune_merges_does_not_clobber_other_keys(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)

    eff = EffectiveConfig.load()
    eff.apply_tune("risk.risk_per_trade_pct", 2.0)
    eff.apply_tune("ml.confidence_threshold_high", 0.7)

    eff2 = EffectiveConfig.load()
    assert eff2.get("risk.risk_per_trade_pct") == 2.0
    assert eff2.get("ml.confidence_threshold_high") == 0.7
    # Untouched key still falls through to settings.yaml.
    assert eff2.get("ml.confidence_threshold_low") == 0.55


def test_safety_floor_still_wins_after_tune(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)

    eff = EffectiveConfig.load()
    # apply_tune() itself does no validation (queue.py enforces the
    # safety-lock check before calling it) — but even if a safety key were
    # tuned, safety_floor.yaml is merged last on load() and always wins.
    eff.apply_tune("risk.min_floor_zar", 50)

    eff2 = EffectiveConfig.load()
    assert eff2.get("risk.min_floor_zar") == 600
