"""
Task 1 — Capital re-base for R1000 + ratcheting floor config.

Covers:
- settings.yaml re-based to R1000 starting balance / new milestones
- RatchetFloor math, persistence, corruption recovery, breach detection
"""
import json

from src.config import load_config
from src.risk.ratchet_floor import RatchetFloor


# ---------------------------------------------------------------------------
# Capital re-base (settings.yaml)
# ---------------------------------------------------------------------------

def test_settings_rebased_to_r1000():
    config = load_config()
    assert config.get("account.starting_balance_zar") == 1000
    assert config.get("account.currency") == "ZAR"


def test_growth_milestones_rebased():
    config = load_config()
    assert config.get("growth.milestones") == [1500, 2000, 3000, 4500, 6000]
    assert config.get("growth.target_balance") == 6000


def test_risk_per_trade_pct_unchanged():
    config = load_config()
    # Task 1 keeps this value as-is
    assert config.get("risk.risk_per_trade_pct") == 1.5


# ---------------------------------------------------------------------------
# RatchetFloor math
# ---------------------------------------------------------------------------

def _floor(tmp_path, hwm_seed=None):
    state_path = tmp_path / "account_state.json"
    if hwm_seed is not None:
        state_path.write_text(
            json.dumps({"high_water_mark": hwm_seed, "updated_utc": "2026-01-01T00:00:00+00:00"}),
            encoding="utf-8",
        )
    return RatchetFloor(min_floor_zar=600, max_total_drawdown_pct=0.35, state_path=str(state_path))


def test_floor_at_hwm_1000_is_650(tmp_path):
    rf = _floor(tmp_path, hwm_seed=1000)
    assert rf.current_floor == 650


def test_floor_at_hwm_2000_is_1300(tmp_path):
    rf = _floor(tmp_path, hwm_seed=2000)
    assert rf.current_floor == 1300


def test_floor_never_below_min_floor(tmp_path):
    rf = _floor(tmp_path, hwm_seed=600)
    # 600 * 0.65 = 390, below min_floor_zar of 600 -> clamps to 600
    assert rf.current_floor == 600
    rf.update(100)  # balance far below floor, HWM shouldn't move down
    assert rf.current_floor == 600


def test_floor_never_decreases_when_balance_falls(tmp_path):
    rf = _floor(tmp_path, hwm_seed=2000)
    assert rf.current_floor == 1300
    rf.update(1500)  # balance drops well below HWM
    assert rf.current_floor == 1300  # floor unchanged, HWM unchanged


def test_floor_ratchets_up_as_balance_grows(tmp_path):
    rf = _floor(tmp_path, hwm_seed=1000)
    assert rf.current_floor == 650
    new_floor = rf.update(2000)
    assert new_floor == 1300
    assert rf.current_floor == 1300


def test_default_seed_hwm_is_1000_when_no_state_file(tmp_path):
    state_path = tmp_path / "does_not_exist.json"
    rf = RatchetFloor(min_floor_zar=600, max_total_drawdown_pct=0.35, state_path=str(state_path))
    assert rf.current_floor == 650  # 1000 * 0.65


def test_seed_uses_max_of_balance_and_starting_hwm(tmp_path):
    state_path = tmp_path / "does_not_exist.json"
    rf = RatchetFloor(min_floor_zar=600, max_total_drawdown_pct=0.35, state_path=str(state_path))
    # First update with a balance below the 1000 starting HWM shouldn't lower it
    rf.update(800)
    assert rf.current_floor == 650  # still based on HWM=1000, not 800


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------

def test_state_persists_across_instances(tmp_path):
    state_path = tmp_path / "account_state.json"
    rf1 = RatchetFloor(min_floor_zar=600, max_total_drawdown_pct=0.35, state_path=str(state_path))
    rf1.update(2000)
    assert state_path.exists()

    rf2 = RatchetFloor(min_floor_zar=600, max_total_drawdown_pct=0.35, state_path=str(state_path))
    assert rf2.current_floor == 1300


def test_persisted_file_has_expected_shape(tmp_path):
    state_path = tmp_path / "account_state.json"
    rf = RatchetFloor(min_floor_zar=600, max_total_drawdown_pct=0.35, state_path=str(state_path))
    rf.update(2000)

    data = json.loads(state_path.read_text(encoding="utf-8"))
    assert data["high_water_mark"] == 2000
    assert "updated_utc" in data


# ---------------------------------------------------------------------------
# Corrupt-file recovery
# ---------------------------------------------------------------------------

def test_corrupt_state_file_reseeds_to_default(tmp_path):
    state_path = tmp_path / "account_state.json"
    state_path.write_text("{not valid json", encoding="utf-8")

    rf = RatchetFloor(min_floor_zar=600, max_total_drawdown_pct=0.35, state_path=str(state_path))
    assert rf.current_floor == 650  # reseeded to starting HWM 1000


def test_missing_high_water_mark_key_reseeds_to_default(tmp_path):
    state_path = tmp_path / "account_state.json"
    state_path.write_text(json.dumps({"updated_utc": "2026-01-01T00:00:00+00:00"}), encoding="utf-8")

    rf = RatchetFloor(min_floor_zar=600, max_total_drawdown_pct=0.35, state_path=str(state_path))
    assert rf.current_floor == 650


# ---------------------------------------------------------------------------
# Breach detection
# ---------------------------------------------------------------------------

def test_is_breached_true_when_equity_at_or_below_floor(tmp_path):
    rf = _floor(tmp_path, hwm_seed=1000)
    assert rf.is_breached(650) is True
    assert rf.is_breached(649.99) is True


def test_is_breached_false_when_equity_above_floor(tmp_path):
    rf = _floor(tmp_path, hwm_seed=1000)
    assert rf.is_breached(650.01) is False
    assert rf.is_breached(1000) is False
