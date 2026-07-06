from src.control.effective_config import EffectiveConfig


def test_safety_floor_overrides_settings():
    eff = EffectiveConfig.load()
    # Even if someone edits settings.yaml to weaken the floor, safety_floor wins
    assert eff.get("risk.min_floor_zar") == 600
    assert eff.get("risk.max_total_drawdown_pct") == 0.35
    assert eff.get("risk.daily_drawdown_stop_pct") == 4.0


def test_is_safety_locked_rejects_floor_keys():
    eff = EffectiveConfig.load()
    assert eff.is_safety_locked("risk.min_floor_zar") is True
    assert eff.is_safety_locked("risk.max_total_drawdown_pct") is True
    assert eff.is_safety_locked("risk.daily_drawdown_stop_pct") is True
    assert eff.is_safety_locked("risk.risk_per_trade_pct") is False  # tunable
