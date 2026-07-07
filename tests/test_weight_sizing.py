"""
Task 10 — Per-instrument weight in position sizing.

Covers:
- `weight.<INSTRUMENT>` (read through EffectiveConfig, default 1.0) scales
  risk_amount BEFORE the size calc; existing clamps (leverage cap, min/max
  size, round DOWN) still apply after.
- weight == 0.0 -> RiskManager rejects the entry ("muted by weight") before
  sizing is attempted.
- weight == 1.5 at max risk% still respects the leverage cap (a weight > 1.0
  can never breach the leverage cap or the risk_per_trade bounds).
- Unset weight / no EffectiveConfig wired defaults to 1.0 (no behavior
  change), preserving pre-Task-10 sizing.

Uses a real EffectiveConfig against tmp settings/safety-floor/tunes files
(monkeypatched paths), mirroring tests/test_effective_config_tune.py, rather
than mocking EffectiveConfig itself.
"""
import src.control.effective_config as ec_module
from src.control.effective_config import EffectiveConfig
from src.config import Config
from src.risk.position_sizer import PositionSizer
from src.risk.manager import RiskManager, TradeRequest


EUR_USD_INSTRUMENT = {
    "pip_location": -4,
    "sl_atr_multiplier": 1.5,
    "tp_atr_multiplier": 2.25,
    "atr_sl_min_pips": 5,
    "atr_sl_max_pips": 20,
    "risk_per_trade_pct": 1.5,
    "max_spread_pips": 3.0,
    "typical_spread_pips": 1.2,
}


def _make_config(risk_overrides: dict | None = None) -> Config:
    settings = {
        "risk": {
            "risk_per_trade_pct": 1.5,
            "sl_atr_multiplier": 1.5,
            "tp_atr_multiplier": 2.25,
            "min_sl_pips": 5,
            "max_sl_pips": 20,
            "max_effective_leverage": 5.0,
            "consecutive_loss_reduce_at": 3,
            "high_volatility_atr_ratio": 2.0,
            "low_volatility_atr_ratio": 0.3,
            "min_floor_zar": 600,
            "max_total_drawdown_pct": 0.35,
            "max_open_positions": 3,
        },
        "trading": {"max_trades_per_day": 60},
    }
    if risk_overrides:
        settings["risk"].update(risk_overrides)
    instruments = {"instruments": {"EUR_USD": dict(EUR_USD_INSTRUMENT)}}
    return Config(settings=settings, instruments=instruments)


def _patch_effective_config_paths(monkeypatch, tmp_path):
    settings_path = tmp_path / "settings.yaml"
    safety_path = tmp_path / "safety_floor.yaml"
    tunes_path = tmp_path / "control" / "effective_config.json"
    settings_path.write_text("risk:\n  risk_per_trade_pct: 1.5\n", encoding="utf-8")
    safety_path.write_text("risk:\n  min_floor_zar: 600\n", encoding="utf-8")
    monkeypatch.setattr(ec_module, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(ec_module, "SAFETY_FLOOR_PATH", safety_path)
    monkeypatch.setattr(ec_module, "TUNES_PATH", tunes_path)


def _base_calc_kwargs():
    return dict(
        balance=10_000.0,
        instrument="EUR_USD",
        direction="buy",
        entry_price=1.1000,
        atr_value=0.0010,  # 10 pips
        atr_ratio=1.0,
        consecutive_losses=0,
        current_spread=0.0001,
    )


def test_weight_defaults_to_1_when_no_effective_config_wired():
    config = _make_config()
    sizer = PositionSizer(config)
    assert sizer.effective_config is None

    result = sizer.calculate(**_base_calc_kwargs())
    assert result is not None
    assert result["risk_amount"] == 10_000.0 * 0.015
    assert not any(a.startswith("weight_") for a in result["adjustments"])


def test_weight_scales_risk_amount_and_units_before_size_calc(monkeypatch, tmp_path):
    _patch_effective_config_paths(monkeypatch, tmp_path)
    eff = EffectiveConfig.load()
    eff.apply_tune("weight.EUR_USD", 0.5)

    # High leverage cap so the leverage clamp isn't the binding constraint —
    # this isolates the weight multiplier's effect on unit count.
    config = _make_config(risk_overrides={"max_effective_leverage": 50.0})
    sizer = PositionSizer(config)
    sizer.effective_config = eff

    baseline = PositionSizer(config).calculate(**_base_calc_kwargs())
    weighted = sizer.calculate(**_base_calc_kwargs())

    assert weighted is not None and baseline is not None
    assert weighted["risk_amount"] == baseline["risk_amount"] * 0.5
    assert weighted["abs_units"] < baseline["abs_units"]
    assert "weight_0.50x" in weighted["adjustments"]


def test_weight_1_5_at_max_risk_pct_still_leverage_capped(monkeypatch, tmp_path):
    """
    A weight > 1.0 combined with the max allowed risk_per_trade_pct must
    never push effective leverage above the configured cap — the leverage
    clamp (Step 8 in position_sizer.calculate) runs AFTER the weight
    multiplier and still wins.
    """
    _patch_effective_config_paths(monkeypatch, tmp_path)
    eff = EffectiveConfig.load()
    eff.apply_tune("weight.EUR_USD", 1.5)

    # Max risk_per_trade_pct per TUNE_BOUNDS is 2.5%; use the instrument
    # override (2.5%) representing the highest per-instrument risk % seen
    # in config/instruments.yaml, combined with a small balance/tight ATR
    # so the unclamped size would badly breach leverage.
    config = _make_config()
    config.instruments["instruments"]["EUR_USD"]["risk_per_trade_pct"] = 2.5

    sizer = PositionSizer(config)
    sizer.effective_config = eff

    kwargs = _base_calc_kwargs()
    kwargs["atr_value"] = 0.0002  # very tight ATR -> tiny SL -> huge raw size
    result = sizer.calculate(**kwargs)

    assert result is not None
    assert result["effective_leverage"] <= config.get("risk.max_effective_leverage", 5.0) + 1e-9
    assert any(a.startswith("leverage_capped_") for a in result["adjustments"])


def _init_risk_manager(config, effective_config=None):
    rm = RiskManager(config)
    rm.initialize(balance=10_000.0)
    if effective_config is not None:
        rm.sizer.effective_config = effective_config
    return rm


def _base_trade_request(instrument="EUR_USD"):
    return TradeRequest(
        instrument=instrument,
        direction="buy",
        entry_price=1.1000,
        atr_value=0.0010,
        atr_ratio=1.0,
        ml_confidence=0.6,
        current_spread=0.0001,
        current_spread_pips=1.0,
    )


def test_risk_manager_rejects_muted_instrument_before_sizing(monkeypatch, tmp_path):
    _patch_effective_config_paths(monkeypatch, tmp_path)
    eff = EffectiveConfig.load()
    eff.apply_tune("weight.EUR_USD", 0.0)

    config = _make_config()
    rm = _init_risk_manager(config, effective_config=eff)

    approval = rm.evaluate_trade(_base_trade_request(), current_balance=10_000.0)

    assert approval.approved is False
    assert "muted by weight" in approval.reason


def test_risk_manager_approves_normally_when_weight_unset():
    config = _make_config()
    rm = _init_risk_manager(config)

    approval = rm.evaluate_trade(_base_trade_request(), current_balance=10_000.0)

    assert approval.approved is True


def test_risk_manager_approves_with_nonzero_weight(monkeypatch, tmp_path):
    _patch_effective_config_paths(monkeypatch, tmp_path)
    eff = EffectiveConfig.load()
    eff.apply_tune("weight.EUR_USD", 0.5)

    config = _make_config()
    rm = _init_risk_manager(config, effective_config=eff)

    approval = rm.evaluate_trade(_base_trade_request(), current_balance=10_000.0)

    assert approval.approved is True
    assert approval.risk_amount == 10_000.0 * 0.015 * 0.5
