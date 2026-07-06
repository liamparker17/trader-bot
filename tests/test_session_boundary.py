"""
Task 4 — Blockers D/E/F: 21:00 UTC session-boundary resets.

Covers:
- session_boundary() helper: daily (21:00 UTC) and weekly (Friday 21:00 UTC)
- DrawdownTracker resets daily/weekly counters at the session boundary,
  not at midnight, using an injected clock (no sleeps).
- CircuitBreaker.reset_consecutive_losses() resets only the consecutive
  loss counter, leaving pause/shutdown/win-rate state untouched.
- RiskManager.check_session_boundary() resets trades_today, the circuit
  breaker's consecutive-loss counter, and lifts a daily-drawdown block,
  exactly at the 21:00 UTC crossing.
"""
from datetime import datetime, timezone

from src.config import Config
from src.risk.drawdown_tracker import DrawdownTracker, session_boundary
from src.risk.circuit_breaker import CircuitBreaker
from src.risk.ratchet_floor import RatchetFloor
from src.risk.manager import RiskManager


def _config(**overrides):
    settings = {
        "risk": {
            "daily_drawdown_limit_pct": 4.0,
            "weekly_drawdown_limit_pct": 8.0,
            "min_floor_zar": 600,
            "max_total_drawdown_pct": 0.35,
            "session_boundary_hour_utc": 21,
            "max_open_positions": 3,
            "low_volatility_atr_ratio": 0.3,
            "high_volatility_atr_ratio": 2.0,
            "consecutive_loss_reduce_at": 3,
            "consecutive_loss_pause_at": 5,
            "pause_duration_minutes": 30,
            "min_win_rate_threshold": 0.45,
            "min_win_rate_lookback": 100,
            "max_spread_multiplier": 2.0,
            "risk_per_trade_pct": 1.5,
            "sl_atr_multiplier": 1.5,
            "tp_atr_multiplier": 1.8,
            "min_sl_pips": 5,
            "max_sl_pips": 20,
        },
        "trading": {
            "max_trades_per_day": 60,
        },
    }
    settings.update(overrides)
    return Config(settings=settings)


def _floor(tmp_path):
    return RatchetFloor(
        min_floor_zar=600,
        max_total_drawdown_pct=0.35,
        state_path=str(tmp_path / "account_state.json"),
    )


# ---------------------------------------------------------------------------
# session_boundary() helper
# ---------------------------------------------------------------------------

def test_daily_boundary_before_reset_hour_is_previous_day():
    # Wed 2026-07-08 10:00 UTC, reset hour 21 -> boundary is Tue 21:00 UTC
    now = datetime(2026, 7, 8, 10, 0, tzinfo=timezone.utc)
    boundary = session_boundary(now, 21)
    assert boundary == datetime(2026, 7, 7, 21, 0, tzinfo=timezone.utc)


def test_daily_boundary_after_reset_hour_is_same_day():
    # Wed 2026-07-08 22:00 UTC, reset hour 21 -> boundary is Wed 21:00 UTC
    now = datetime(2026, 7, 8, 22, 0, tzinfo=timezone.utc)
    boundary = session_boundary(now, 21)
    assert boundary == datetime(2026, 7, 8, 21, 0, tzinfo=timezone.utc)


def test_daily_boundary_exactly_at_reset_hour_is_itself():
    now = datetime(2026, 7, 8, 21, 0, tzinfo=timezone.utc)
    boundary = session_boundary(now, 21)
    assert boundary == now


def test_weekly_boundary_is_friday_21_utc():
    # 2026-07-07 is a Tuesday. Most recent Friday 21:00 UTC before it
    # is 2026-07-03.
    now = datetime(2026, 7, 7, 10, 0, tzinfo=timezone.utc)
    boundary = session_boundary(now, 21, weekday=4)
    assert boundary == datetime(2026, 7, 3, 21, 0, tzinfo=timezone.utc)
    assert boundary.weekday() == 4


def test_weekly_boundary_on_friday_before_reset_hour_is_prior_friday():
    # 2026-07-10 is a Friday. At 20:00 UTC (before the 21:00 reset), the
    # boundary should still be the *previous* Friday.
    now = datetime(2026, 7, 10, 20, 0, tzinfo=timezone.utc)
    boundary = session_boundary(now, 21, weekday=4)
    assert boundary == datetime(2026, 7, 3, 21, 0, tzinfo=timezone.utc)


def test_weekly_boundary_on_friday_after_reset_hour_is_that_friday():
    now = datetime(2026, 7, 10, 22, 0, tzinfo=timezone.utc)
    boundary = session_boundary(now, 21, weekday=4)
    assert boundary == datetime(2026, 7, 10, 21, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# DrawdownTracker — boundary-driven resets (injected clock, no sleeps)
# ---------------------------------------------------------------------------

def test_drawdown_tracker_does_not_reset_before_21utc_crossing(tmp_path):
    clock_box = {"now": datetime(2026, 7, 8, 10, 0, tzinfo=timezone.utc)}
    tracker = DrawdownTracker(_config(), ratchet_floor=_floor(tmp_path), clock=lambda: clock_box["now"])
    tracker.initialize(1000.0)

    # Move forward within the same session day (before 21:00 UTC)
    clock_box["now"] = datetime(2026, 7, 8, 20, 59, tzinfo=timezone.utc)
    tracker.update(900.0)
    assert tracker.daily_start_balance == 1000.0  # unchanged — no reset yet


def test_drawdown_tracker_resets_exactly_at_21utc_crossing(tmp_path):
    clock_box = {"now": datetime(2026, 7, 8, 10, 0, tzinfo=timezone.utc)}
    tracker = DrawdownTracker(_config(), ratchet_floor=_floor(tmp_path), clock=lambda: clock_box["now"])
    tracker.initialize(1000.0)

    # Cross the 21:00 UTC boundary
    clock_box["now"] = datetime(2026, 7, 8, 21, 0, tzinfo=timezone.utc)
    tracker.update(900.0)
    assert tracker.daily_start_balance == 900.0  # reset to new balance
    assert len(tracker.daily_drawdowns) == 1
    assert tracker.daily_drawdowns[0]["start_balance"] == 1000.0


def test_drawdown_tracker_weekly_reset_at_friday_21utc(tmp_path):
    # Start Monday 2026-07-06 10:00 UTC
    clock_box = {"now": datetime(2026, 7, 6, 10, 0, tzinfo=timezone.utc)}
    tracker = DrawdownTracker(_config(), ratchet_floor=_floor(tmp_path), clock=lambda: clock_box["now"])
    tracker.initialize(1000.0)

    # Thursday — still same trading week, no reset
    clock_box["now"] = datetime(2026, 7, 9, 12, 0, tzinfo=timezone.utc)
    tracker.update(950.0)
    assert tracker.weekly_start_balance == 1000.0

    # Friday 21:00 UTC — new trading week starts
    clock_box["now"] = datetime(2026, 7, 10, 21, 0, tzinfo=timezone.utc)
    tracker.update(950.0)
    assert tracker.weekly_start_balance == 950.0


# ---------------------------------------------------------------------------
# CircuitBreaker.reset_consecutive_losses()
# ---------------------------------------------------------------------------

def test_reset_consecutive_losses_clears_counter_only(tmp_path):
    breaker = CircuitBreaker(_config(), ratchet_floor=_floor(tmp_path))
    breaker.record_trade_outcome(False)
    breaker.record_trade_outcome(False)
    assert breaker.consecutive_losses == 2

    breaker.reset_consecutive_losses()

    assert breaker.consecutive_losses == 0
    # Win-rate rolling window is NOT session-scoped — untouched.
    assert breaker.recent_outcomes == [False, False]


def test_reset_consecutive_losses_does_not_touch_pause_state(tmp_path):
    breaker = CircuitBreaker(_config(), ratchet_floor=_floor(tmp_path))
    for _ in range(5):
        breaker.record_trade_outcome(False)
    assert breaker.is_paused is True

    breaker.reset_consecutive_losses()

    assert breaker.consecutive_losses == 0
    assert breaker.is_paused is True  # unrelated state untouched


# ---------------------------------------------------------------------------
# RiskManager — boundary-driven resets (consecutive losses + unblock)
# ---------------------------------------------------------------------------

def test_risk_manager_resets_consecutive_losses_at_boundary(tmp_path):
    clock_box = {"now": datetime(2026, 7, 8, 10, 0, tzinfo=timezone.utc)}
    rm = RiskManager(_config(), clock=lambda: clock_box["now"], ratchet_floor=_floor(tmp_path))
    rm.initialize(1000.0)

    for _ in range(3):
        rm.circuit_breaker.record_trade_outcome(False)
    assert rm.circuit_breaker.consecutive_losses == 3

    # Before the boundary: unchanged
    clock_box["now"] = datetime(2026, 7, 8, 20, 0, tzinfo=timezone.utc)
    crossed = rm.check_session_boundary()
    assert crossed is False
    assert rm.circuit_breaker.consecutive_losses == 3

    # Cross 21:00 UTC boundary
    clock_box["now"] = datetime(2026, 7, 8, 21, 0, tzinfo=timezone.utc)
    crossed = rm.check_session_boundary()
    assert crossed is True
    assert rm.circuit_breaker.consecutive_losses == 0


def test_risk_manager_lifts_daily_drawdown_block_at_boundary(tmp_path):
    clock_box = {"now": datetime(2026, 7, 8, 10, 0, tzinfo=timezone.utc)}
    rm = RiskManager(_config(), clock=lambda: clock_box["now"], ratchet_floor=_floor(tmp_path))
    rm.initialize(1000.0)

    # Force the block flag on directly (breach mechanics covered elsewhere)
    rm._blocked_until_boundary = True
    rm._block_reason = "Daily drawdown 5.0% >= limit 4.0%"
    assert rm.close_all_signal() is True

    clock_box["now"] = datetime(2026, 7, 8, 21, 0, tzinfo=timezone.utc)
    rm.check_session_boundary()

    assert rm._blocked_until_boundary is False
    assert rm.close_all_signal() is False
