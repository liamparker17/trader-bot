"""
Task 12 — src/manager/policy.py: LEVERS, risk_ceiling_now(), growth_stage(),
validate_and_clamp().

Covers:
- LEVERS reuses src.control.queue.TUNE_BOUNDS / WEIGHT_BOUNDS (single
  source of truth, no re-defined bounds).
- risk_ceiling_now() growth-stage ladder (below/at/above each milestone).
- validate_and_clamp() clamp matrix: each lever below/at/above bounds.
- 4th+ proposal in a cycle rejected (cap of 3).
- threshold_low <= threshold_high pair-wise invariant against resulting
  config (both proposed together, one proposed vs current effective value).
- unknown key / bad instrument / safety-locked rejection.
- risk_per_trade_pct additionally capped at risk_ceiling_now.
"""
import json

import src.control.effective_config as ec_module
import src.manager.policy as policy_module
from src.control.effective_config import EffectiveConfig
from src.control.queue import TUNE_BOUNDS, WEIGHT_BOUNDS
from src.manager.policy import (
    LEVERS,
    growth_stage,
    risk_ceiling_now,
    validate_and_clamp,
)

MILESTONES = [1500, 2000, 3000, 4500, 6000]


def _eff(monkeypatch, tmp_path, settings_yaml=None, safety_yaml=None):
    settings_path = tmp_path / "settings.yaml"
    safety_path = tmp_path / "safety_floor.yaml"
    tunes_path = tmp_path / "control" / "effective_config.json"

    settings_path.write_text(
        settings_yaml or (
            "risk:\n  risk_per_trade_pct: 1.5\n"
            "ml:\n  confidence_threshold_high: 0.65\n  confidence_threshold_low: 0.55\n"
        ),
        encoding="utf-8",
    )
    safety_path.write_text(safety_yaml or "", encoding="utf-8")

    monkeypatch.setattr(ec_module, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(ec_module, "SAFETY_FLOOR_PATH", safety_path)
    monkeypatch.setattr(ec_module, "TUNES_PATH", tunes_path)
    return EffectiveConfig.load()


def _instruments(monkeypatch, tmp_path, names=("EUR_USD", "XAU_USD")):
    instruments_path = tmp_path / "instruments.yaml"
    body = "instruments:\n" + "".join(f"  {n}:\n    enabled: true\n" for n in names)
    instruments_path.write_text(body, encoding="utf-8")
    monkeypatch.setattr(policy_module, "INSTRUMENTS_PATH", instruments_path)
    return instruments_path


# ---------------------------------------------------------------------------
# LEVERS reuses queue.py constants
# ---------------------------------------------------------------------------

def test_levers_reuses_tune_bounds():
    for key, bounds in TUNE_BOUNDS.items():
        assert LEVERS[key] == bounds


def test_weight_bounds_identity():
    assert policy_module.WEIGHT_BOUNDS == WEIGHT_BOUNDS


# ---------------------------------------------------------------------------
# risk_ceiling_now ladder
# ---------------------------------------------------------------------------

def test_ceiling_below_first_milestone():
    assert risk_ceiling_now(1000, MILESTONES) == 1.5


def test_ceiling_at_1500():
    assert risk_ceiling_now(1500, MILESTONES) == 1.8


def test_ceiling_between_1500_and_2000():
    assert risk_ceiling_now(1800, MILESTONES) == 1.8


def test_ceiling_at_2000():
    assert risk_ceiling_now(2000, MILESTONES) == 2.0


def test_ceiling_at_3000():
    assert risk_ceiling_now(3000, MILESTONES) == 2.2


def test_ceiling_at_4500():
    assert risk_ceiling_now(4500, MILESTONES) == 2.5


def test_ceiling_above_4500_hard_capped():
    assert risk_ceiling_now(100000, MILESTONES) == 2.5


def test_ceiling_at_6000():
    assert risk_ceiling_now(6000, MILESTONES) == 2.5


# ---------------------------------------------------------------------------
# growth_stage
# ---------------------------------------------------------------------------

def test_growth_stage_zero_below_first_milestone():
    assert growth_stage(1000, MILESTONES) == 0


def test_growth_stage_increments_per_milestone():
    assert growth_stage(1500, MILESTONES) == 1
    assert growth_stage(2000, MILESTONES) == 2
    assert growth_stage(3000, MILESTONES) == 3
    assert growth_stage(4500, MILESTONES) == 4
    assert growth_stage(6000, MILESTONES) == 5


# ---------------------------------------------------------------------------
# validate_and_clamp: clamp matrix
# ---------------------------------------------------------------------------

def test_risk_per_trade_pct_below_bound_clamped_up(monkeypatch, tmp_path):
    eff = _eff(monkeypatch, tmp_path)
    applied, rejected = validate_and_clamp(
        [{"key": "risk.risk_per_trade_pct", "value": 0.1}], eff, risk_ceiling_now=2.5
    )
    assert rejected == []
    assert applied[0]["value"] == 0.5
    assert applied[0]["clamped"] is True
    assert applied[0]["original_value"] == 0.1


def test_risk_per_trade_pct_at_bound_not_clamped(monkeypatch, tmp_path):
    eff = _eff(monkeypatch, tmp_path)
    applied, rejected = validate_and_clamp(
        [{"key": "risk.risk_per_trade_pct", "value": 0.5}], eff, risk_ceiling_now=2.5
    )
    assert rejected == []
    assert applied[0]["value"] == 0.5
    assert applied[0]["clamped"] is False


def test_risk_per_trade_pct_above_bound_clamped_down(monkeypatch, tmp_path):
    eff = _eff(monkeypatch, tmp_path)
    applied, rejected = validate_and_clamp(
        [{"key": "risk.risk_per_trade_pct", "value": 5.0}], eff, risk_ceiling_now=2.5
    )
    assert rejected == []
    assert applied[0]["value"] == 2.5
    assert applied[0]["clamped"] is True


def test_risk_per_trade_pct_capped_by_ceiling_now(monkeypatch, tmp_path):
    eff = _eff(monkeypatch, tmp_path)
    applied, rejected = validate_and_clamp(
        [{"key": "risk.risk_per_trade_pct", "value": 2.3}], eff, risk_ceiling_now=1.8
    )
    assert rejected == []
    assert applied[0]["value"] == 1.8
    assert applied[0]["clamped"] is True


def test_ml_threshold_high_below_bound(monkeypatch, tmp_path):
    # Baseline low=0.45 so clamped high (0.50) still satisfies low<=high.
    eff = _eff(
        monkeypatch, tmp_path,
        settings_yaml=(
            "risk:\n  risk_per_trade_pct: 1.5\n"
            "ml:\n  confidence_threshold_high: 0.65\n  confidence_threshold_low: 0.45\n"
        ),
    )
    applied, rejected = validate_and_clamp(
        [{"key": "ml.confidence_threshold_high", "value": 0.3}], eff, risk_ceiling_now=2.5
    )
    assert rejected == []
    assert applied[0]["value"] == 0.50
    assert applied[0]["clamped"] is True


def test_ml_threshold_high_at_bound(monkeypatch, tmp_path):
    eff = _eff(monkeypatch, tmp_path)
    applied, rejected = validate_and_clamp(
        [{"key": "ml.confidence_threshold_high", "value": 0.75}], eff, risk_ceiling_now=2.5
    )
    assert rejected == []
    assert applied[0]["value"] == 0.75
    assert applied[0]["clamped"] is False


def test_ml_threshold_high_above_bound(monkeypatch, tmp_path):
    eff = _eff(monkeypatch, tmp_path)
    applied, rejected = validate_and_clamp(
        [{"key": "ml.confidence_threshold_high", "value": 0.99}], eff, risk_ceiling_now=2.5
    )
    assert rejected == []
    assert applied[0]["value"] == 0.75
    assert applied[0]["clamped"] is True


def test_ml_threshold_low_below_bound(monkeypatch, tmp_path):
    eff = _eff(monkeypatch, tmp_path)
    applied, rejected = validate_and_clamp(
        [{"key": "ml.confidence_threshold_low", "value": 0.1}], eff, risk_ceiling_now=2.5
    )
    assert rejected == []
    assert applied[0]["value"] == 0.45
    assert applied[0]["clamped"] is True


def test_ml_threshold_low_at_bound(monkeypatch, tmp_path):
    eff = _eff(monkeypatch, tmp_path)
    applied, rejected = validate_and_clamp(
        [{"key": "ml.confidence_threshold_low", "value": 0.45}], eff, risk_ceiling_now=2.5
    )
    assert rejected == []
    assert applied[0]["value"] == 0.45
    assert applied[0]["clamped"] is False


def test_ml_threshold_low_above_bound(monkeypatch, tmp_path):
    eff = _eff(monkeypatch, tmp_path)
    applied, rejected = validate_and_clamp(
        [{"key": "ml.confidence_threshold_low", "value": 0.9}], eff, risk_ceiling_now=2.5
    )
    assert rejected == []
    assert applied[0]["value"] == 0.65
    assert applied[0]["clamped"] is True


def test_weight_below_bound(monkeypatch, tmp_path):
    eff = _eff(monkeypatch, tmp_path)
    _instruments(monkeypatch, tmp_path)
    applied, rejected = validate_and_clamp(
        [{"key": "weight.EUR_USD", "value": -1.0}], eff, risk_ceiling_now=2.5
    )
    assert rejected == []
    assert applied[0]["value"] == 0.0
    assert applied[0]["clamped"] is True


def test_weight_at_bound(monkeypatch, tmp_path):
    eff = _eff(monkeypatch, tmp_path)
    _instruments(monkeypatch, tmp_path)
    applied, rejected = validate_and_clamp(
        [{"key": "weight.EUR_USD", "value": 1.5}], eff, risk_ceiling_now=2.5
    )
    assert rejected == []
    assert applied[0]["value"] == 1.5
    assert applied[0]["clamped"] is False


def test_weight_above_bound(monkeypatch, tmp_path):
    eff = _eff(monkeypatch, tmp_path)
    _instruments(monkeypatch, tmp_path)
    applied, rejected = validate_and_clamp(
        [{"key": "weight.EUR_USD", "value": 3.0}], eff, risk_ceiling_now=2.5
    )
    assert rejected == []
    assert applied[0]["value"] == 1.5
    assert applied[0]["clamped"] is True


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------

def test_unknown_key_rejected(monkeypatch, tmp_path):
    eff = _eff(monkeypatch, tmp_path)
    applied, rejected = validate_and_clamp(
        [{"key": "not.a.real.key", "value": 1.0}], eff, risk_ceiling_now=2.5
    )
    assert applied == []
    assert len(rejected) == 1
    assert rejected[0]["key"] == "not.a.real.key"
    assert "reason" in rejected[0] and "rejection_reason" in rejected[0]


def test_bad_instrument_rejected(monkeypatch, tmp_path):
    eff = _eff(monkeypatch, tmp_path)
    _instruments(monkeypatch, tmp_path)
    applied, rejected = validate_and_clamp(
        [{"key": "weight.NOT_REAL", "value": 1.0}], eff, risk_ceiling_now=2.5
    )
    assert applied == []
    assert len(rejected) == 1
    assert rejected[0]["key"] == "weight.NOT_REAL"


def test_safety_locked_key_rejected(monkeypatch, tmp_path):
    eff = _eff(
        monkeypatch, tmp_path,
        safety_yaml="risk:\n  risk_per_trade_pct: 1.0\n",
    )
    applied, rejected = validate_and_clamp(
        [{"key": "risk.risk_per_trade_pct", "value": 2.0}], eff, risk_ceiling_now=2.5
    )
    assert applied == []
    assert len(rejected) == 1
    assert "safety" in rejected[0]["reason"].lower()


def test_non_numeric_value_rejected(monkeypatch, tmp_path):
    eff = _eff(monkeypatch, tmp_path)
    applied, rejected = validate_and_clamp(
        [{"key": "risk.risk_per_trade_pct", "value": "not-a-number"}], eff, risk_ceiling_now=2.5
    )
    assert applied == []
    assert len(rejected) == 1


# ---------------------------------------------------------------------------
# Cycle cap: keep first 3, reject rest
# ---------------------------------------------------------------------------

def test_fourth_proposal_rejected(monkeypatch, tmp_path):
    eff = _eff(monkeypatch, tmp_path)
    _instruments(monkeypatch, tmp_path, names=("EUR_USD", "GBP_USD", "XAU_USD"))
    proposals = [
        {"key": "risk.risk_per_trade_pct", "value": 1.6},
        {"key": "weight.EUR_USD", "value": 1.1},
        {"key": "weight.GBP_USD", "value": 0.9},
        {"key": "weight.XAU_USD", "value": 1.2},
    ]
    applied, rejected = validate_and_clamp(proposals, eff, risk_ceiling_now=2.5)
    assert len(applied) == 3
    assert len(rejected) == 1
    assert rejected[0]["key"] == "weight.XAU_USD"
    assert "cycle" in rejected[0]["reason"].lower() or "limit" in rejected[0]["reason"].lower()


# ---------------------------------------------------------------------------
# Duplicate keys within one proposal set: last occurrence wins, earlier
# occurrence(s) rejected as superseded. Exactly one entry per key across
# (applied, rejected) from the accepted (<=3) slice.
# ---------------------------------------------------------------------------

def test_duplicate_key_valid_valid_last_wins(monkeypatch, tmp_path):
    eff = _eff(monkeypatch, tmp_path)
    proposals = [
        {"key": "risk.risk_per_trade_pct", "value": 1.6},
        {"key": "risk.risk_per_trade_pct", "value": 1.8},
    ]
    applied, rejected = validate_and_clamp(proposals, eff, risk_ceiling_now=2.5)
    assert len(applied) == 1
    assert applied[0]["value"] == 1.8
    assert len(rejected) == 1
    assert rejected[0]["value"] == 1.6
    assert rejected[0]["rejection_reason"] == "duplicate_key_superseded"
    # Exactly one entry per key across applied+rejected.
    keys = [e["key"] for e in applied] + [e["key"] for e in rejected]
    assert keys.count("risk.risk_per_trade_pct") == 2  # one applied, one rejected
    assert len(set(keys)) == 1


def test_duplicate_key_valid_then_invalid_last_loses(monkeypatch, tmp_path):
    # First occurrence would be valid on its own; the second (last, wins
    # positionally) is invalid (out of numeric range is fine, but here we
    # use a bad instrument to force a validation rejection independent of
    # clamping) -> last is validated (and rejected), first is superseded.
    eff = _eff(monkeypatch, tmp_path)
    _instruments(monkeypatch, tmp_path)
    proposals = [
        {"key": "weight.EUR_USD", "value": 1.1},
        {"key": "weight.EUR_USD", "value": "not-a-number"},
    ]
    applied, rejected = validate_and_clamp(proposals, eff, risk_ceiling_now=2.5)
    assert applied == []
    assert len(rejected) == 2
    reasons = {r["rejection_reason"] for r in rejected}
    assert reasons == {"duplicate_key_superseded", "non_numeric"}
    # The superseded entry carries the FIRST (valid) proposal's value; the
    # non_numeric entry carries the second (last) proposal's value.
    superseded = next(r for r in rejected if r["rejection_reason"] == "duplicate_key_superseded")
    non_numeric = next(r for r in rejected if r["rejection_reason"] == "non_numeric")
    assert superseded["value"] == 1.1
    assert non_numeric["value"] == "not-a-number"
    keys = [r["key"] for r in rejected]
    assert len(set(keys)) == 1


def test_duplicate_key_spanning_cap_boundary(monkeypatch, tmp_path):
    # Dedup happens AFTER the <=3 slice: a duplicate of an accepted key
    # appearing beyond position 3 is dropped by the cycle cap, not by
    # dedup, and must not resurrect/duplicate the earlier key.
    eff = _eff(monkeypatch, tmp_path)
    _instruments(monkeypatch, tmp_path, names=("EUR_USD", "GBP_USD", "XAU_USD"))
    proposals = [
        {"key": "risk.risk_per_trade_pct", "value": 1.6},
        {"key": "weight.EUR_USD", "value": 1.1},
        {"key": "weight.GBP_USD", "value": 0.9},
        {"key": "risk.risk_per_trade_pct", "value": 1.7},  # 4th: beyond cap
    ]
    applied, rejected = validate_and_clamp(proposals, eff, risk_ceiling_now=2.5)
    assert len(applied) == 3
    assert applied[0]["key"] == "risk.risk_per_trade_pct"
    assert applied[0]["value"] == 1.6  # first occurrence kept (only one in accepted slice)
    assert len(rejected) == 1
    assert rejected[0]["key"] == "risk.risk_per_trade_pct"
    assert rejected[0]["value"] == 1.7
    assert rejected[0]["rejection_reason"] == "cycle_limit_exceeded"
    keys = [e["key"] for e in applied] + [e["key"] for e in rejected]
    assert keys.count("risk.risk_per_trade_pct") == 2  # one applied, one cycle-limit rejected


# ---------------------------------------------------------------------------
# threshold_low <= threshold_high pair-wise invariant
# ---------------------------------------------------------------------------

def test_threshold_pair_both_proposed_violating_invariant(monkeypatch, tmp_path):
    eff = _eff(monkeypatch, tmp_path)
    proposals = [
        {"key": "ml.confidence_threshold_low", "value": 0.60},
        {"key": "ml.confidence_threshold_high", "value": 0.55},
    ]
    applied, rejected = validate_and_clamp(proposals, eff, risk_ceiling_now=2.5)
    assert applied == []
    assert len(rejected) == 2


def test_threshold_pair_both_proposed_satisfying_invariant(monkeypatch, tmp_path):
    eff = _eff(monkeypatch, tmp_path)
    proposals = [
        {"key": "ml.confidence_threshold_low", "value": 0.50},
        {"key": "ml.confidence_threshold_high", "value": 0.60},
    ]
    applied, rejected = validate_and_clamp(proposals, eff, risk_ceiling_now=2.5)
    assert rejected == []
    assert len(applied) == 2


def test_threshold_low_alone_violates_current_high(monkeypatch, tmp_path):
    # Baseline settings: high=0.65, low=0.55. Proposing low=0.70 clamps to
    # bound-max 0.65 which still exceeds current high 0.65? use a value
    # that clamps to something > current high.
    eff = _eff(
        monkeypatch, tmp_path,
        settings_yaml=(
            "risk:\n  risk_per_trade_pct: 1.5\n"
            "ml:\n  confidence_threshold_high: 0.55\n  confidence_threshold_low: 0.50\n"
        ),
    )
    # low proposed at 0.60 (within [0.45,0.65] bounds, no clamp) but current
    # high is 0.55 -> 0.60 > 0.55 violates invariant.
    applied, rejected = validate_and_clamp(
        [{"key": "ml.confidence_threshold_low", "value": 0.60}], eff, risk_ceiling_now=2.5
    )
    assert applied == []
    assert len(rejected) == 1
    assert rejected[0]["key"] == "ml.confidence_threshold_low"


def test_threshold_high_alone_ok_against_current_low(monkeypatch, tmp_path):
    eff = _eff(monkeypatch, tmp_path)  # high=0.65, low=0.55
    applied, rejected = validate_and_clamp(
        [{"key": "ml.confidence_threshold_high", "value": 0.70}], eff, risk_ceiling_now=2.5
    )
    assert rejected == []
    assert len(applied) == 1
    assert applied[0]["value"] == 0.70
