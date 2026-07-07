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


# ---------------------------------------------------------------------------
# Final-review fix: safety_floor.yaml must actually be consumed by
# load_config() — floor-wins over settings.yaml, with the two key renames
# to the names the risk modules read.
# ---------------------------------------------------------------------------

def test_apply_safety_floor_overrides_settings():
    from src.config import _apply_safety_floor

    settings = {
        "risk": {
            "daily_drawdown_limit_pct": 99.0,   # must lose to the floor file
            "min_floor_zar": 1,                 # must lose
            "risk_per_trade_pct": 1.5,          # untouched (not a floor key)
        },
    }
    safety = {
        "risk": {
            "daily_drawdown_stop_pct": 4.0,
            "min_floor_zar": 600,
            "max_total_drawdown_pct": 0.35,
            "max_leverage_effective": 5.0,
        },
        "circuit_breaker": {"api_error_threshold": 10},
    }
    merged = _apply_safety_floor(settings, safety)
    assert merged["risk"]["daily_drawdown_limit_pct"] == 4.0
    assert merged["risk"]["min_floor_zar"] == 600
    assert merged["risk"]["max_total_drawdown_pct"] == 0.35
    assert merged["risk"]["max_effective_leverage"] == 5.0
    assert merged["circuit_breaker"]["api_error_threshold"] == 10
    assert merged["risk"]["risk_per_trade_pct"] == 1.5


def test_apply_safety_floor_noop_when_missing():
    from src.config import _apply_safety_floor

    settings = {"risk": {"risk_per_trade_pct": 1.5}}
    assert _apply_safety_floor(settings, {}) == {"risk": {"risk_per_trade_pct": 1.5}}


def test_load_config_consumes_safety_floor_yaml():
    """End-to-end against the real repo config files: the values the live
    risk modules read must come from config/safety_floor.yaml."""
    from src.config import load_config

    config = load_config()
    assert config.get("risk.min_floor_zar") == 600
    assert config.get("risk.max_total_drawdown_pct") == 0.35
    assert config.get("risk.daily_drawdown_limit_pct") == 4.0
    assert config.get("risk.max_effective_leverage") == 5.0
    assert config.get("circuit_breaker.api_error_threshold") == 10
