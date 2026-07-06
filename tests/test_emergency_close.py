"""
Task 4 — Blockers D/E/F: daily-drawdown emergency close-all.

Covers:
- RiskManager.check_drawdown_emergency() flags a daily-drawdown breach
  exactly once and blocks new entries until the next session boundary.
- evaluate_trade() rejects trades while blocked, and while flagging a
  fresh breach.
- Executor.check_and_manage_positions() reacts to the breach/signal by
  calling close_all(), which closes every open trade via the (mocked)
  MT5Client, logs each close, and fires the injected alert callback.
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.config import Config
from src.risk.manager import RiskManager, TradeRequest
from src.risk.ratchet_floor import RatchetFloor
from src.execution.executor import Executor, OpenTrade


def _config():
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
            "trailing_stop_enabled": False,
        },
        "trading": {
            "max_trades_per_day": 60,
            "min_seconds_between_trades": 0,
        },
    }
    return Config(settings=settings, instruments={
        "instruments": {
            "EUR_USD": {"enabled": True, "pip_location": -4, "typical_spread_pips": 1.5},
        }
    })


def _floor(tmp_path):
    return RatchetFloor(
        min_floor_zar=600,
        max_total_drawdown_pct=0.35,
        state_path=str(tmp_path / "account_state.json"),
    )


def _risk_manager(tmp_path, now=None):
    now = now or datetime(2026, 7, 8, 10, 0, tzinfo=timezone.utc)
    clock_box = {"now": now}
    rm = RiskManager(_config(), clock=lambda: clock_box["now"], ratchet_floor=_floor(tmp_path))
    rm.initialize(1000.0)
    return rm, clock_box


def _trade_request():
    return TradeRequest(
        instrument="EUR_USD",
        direction="buy",
        entry_price=1.1000,
        atr_value=0.0010,
        atr_ratio=1.0,
        ml_confidence=0.6,
        current_spread=0.0001,
        current_spread_pips=1.0,
    )


# ---------------------------------------------------------------------------
# RiskManager: breach detection + blocking
# ---------------------------------------------------------------------------

def test_check_drawdown_emergency_returns_true_once_on_breach(tmp_path):
    rm, _ = _risk_manager(tmp_path)

    # Balance down 5% from the 1000 daily start -> breaches the 4% limit
    assert rm.check_drawdown_emergency(950.0) is True
    # Still breached, but already flagged -> no repeat close_all trigger
    assert rm.check_drawdown_emergency(950.0) is False


def test_daily_drawdown_breach_blocks_new_entries(tmp_path):
    rm, _ = _risk_manager(tmp_path)

    assert rm.check_drawdown_emergency(950.0) is True

    approval = rm.evaluate_trade(_trade_request(), current_balance=950.0)
    assert approval.approved is False
    assert "session boundary" in approval.reason.lower()

    assert rm.close_all_signal() is True
    assert rm.close_all_reason == "daily_drawdown"


def test_evaluate_trade_flags_breach_directly(tmp_path):
    rm, _ = _risk_manager(tmp_path)

    # No prior periodic check — evaluate_trade's own Check 3 must flag it.
    approval = rm.evaluate_trade(_trade_request(), current_balance=950.0)
    assert approval.approved is False
    assert "drawdown limit" in approval.reason.lower()
    assert rm.close_all_signal() is True


def test_block_lifts_and_entries_resume_after_boundary(tmp_path):
    rm, clock_box = _risk_manager(tmp_path)
    rm.check_drawdown_emergency(950.0)
    assert rm.close_all_signal() is True

    # Cross the 21:00 UTC boundary
    clock_box["now"] = datetime(2026, 7, 8, 21, 0, tzinfo=timezone.utc)
    rm.check_session_boundary(950.0)

    assert rm.close_all_signal() is False
    # New daily_start_balance is 950 post-reset, so a further evaluate at
    # the same 950 balance shows 0% daily loss and should pass drawdown.
    approval = rm.evaluate_trade(_trade_request(), current_balance=950.0)
    assert approval.approved is True


# ---------------------------------------------------------------------------
# Executor: close_all wiring + alert callback
# ---------------------------------------------------------------------------

def _open_trade(trade_id="t1", instrument="EUR_USD"):
    return OpenTrade(
        trade_id=trade_id,
        instrument=instrument,
        direction="buy",
        units=1000,
        entry_price=1.1000,
        entry_time=datetime.now(timezone.utc),
        stop_loss=1.0950,
        take_profit=1.1080,
        ml_confidence=0.6,
        risk_amount=15.0,
        sl_pips=10.0,
        tp_pips=18.0,
    )


def _mock_client(balance=950.0):
    client = MagicMock()
    client.close_trade.return_value = None
    client.get_current_price.return_value = {
        "bids": [{"price": "1.0990"}],
        "asks": [{"price": "1.0991"}],
    }
    client.get_account_balance.return_value = balance
    return client


def test_close_all_logs_each_trade_and_fires_alert_callback(tmp_path):
    rm, _ = _risk_manager(tmp_path)
    client = _mock_client()
    alerts = []
    executor = Executor(_config(), client, rm, alert_callback=lambda event, data: alerts.append((event, data)))
    executor.open_trades["t1"] = _open_trade("t1")
    executor.open_trades["t2"] = _open_trade("t2")

    results = executor.close_all(reason="daily_drawdown")

    assert len(results) == 2
    assert client.close_trade.call_count == 2
    assert executor.open_trades == {}
    assert len(alerts) == 1
    event, data = alerts[0]
    assert event == "close_all"
    assert data["reason"] == "daily_drawdown"
    assert data["closed"] == 2


def test_check_and_manage_positions_triggers_close_all_on_breach(tmp_path):
    rm, _ = _risk_manager(tmp_path)
    client = _mock_client()
    alerts = []
    executor = Executor(_config(), client, rm, alert_callback=lambda event, data: alerts.append((event, data)))
    executor.open_trades["t1"] = _open_trade("t1")

    # Balance down 5% -> breaches daily drawdown -> should close everything
    executor.check_and_manage_positions(current_balance=950.0, current_equity=950.0)

    assert executor.open_trades == {}
    assert client.close_trade.call_count == 1
    assert rm.close_all_signal() is True  # remains blocked until boundary
    assert any(event == "close_all" and data["reason"] == "daily_drawdown" for event, data in alerts)


def test_check_and_manage_positions_no_breach_leaves_trades_open(tmp_path):
    rm, _ = _risk_manager(tmp_path)
    client = _mock_client()
    executor = Executor(_config(), client, rm)
    executor.open_trades["t1"] = _open_trade("t1")

    # Balance essentially flat -> no breach
    executor.check_and_manage_positions(current_balance=995.0, current_equity=995.0)

    assert "t1" in executor.open_trades
    client.close_trade.assert_not_called()


def test_close_all_signal_also_covers_circuit_breaker_shutdown(tmp_path):
    rm, _ = _risk_manager(tmp_path)
    client = _mock_client()
    executor = Executor(_config(), client, rm)
    executor.open_trades["t1"] = _open_trade("t1")

    rm.circuit_breaker._shutdown("HARD FLOOR BREACH: test")

    executor.check_and_manage_positions(current_balance=1000.0, current_equity=1000.0)

    assert executor.open_trades == {}
    assert "circuit_breaker_shutdown" in rm.close_all_reason
