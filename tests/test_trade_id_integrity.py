"""
Task 5 — Blocker G/Q: trade-ID integrity.

Covers:
- execute_signal() treats a missing/zero MT5 ticket in the order response
  as a FAILED order: logs an ERROR, fires the alert_callback, and returns
  None. No synthetic/timestamp-based trade ID is ever fabricated, and no
  position is tracked in open_trades.
- A normal fill with a real ticket still opens and tracks the trade as
  before (regression guard).
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.config import Config
from src.risk.manager import RiskManager
from src.risk.ratchet_floor import RatchetFloor
from src.execution.executor import Executor


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


def _risk_manager(tmp_path):
    now = datetime(2026, 7, 8, 10, 0, tzinfo=timezone.utc)
    rm = RiskManager(_config(), clock=lambda: now, ratchet_floor=_floor(tmp_path))
    rm.initialize(1000.0)
    return rm


def _mock_client(order_response, balance=1000.0):
    client = MagicMock()
    client.get_current_price.return_value = {
        "bids": [{"price": "1.0990"}],
        "asks": [{"price": "1.0991"}],
        "tradeable": True,
    }
    client.get_account_balance.return_value = balance
    client.place_market_order.return_value = order_response
    return client


def test_execute_signal_missing_ticket_is_failed_order_no_synthetic_id(tmp_path):
    rm = _risk_manager(tmp_path)
    # Order "fills" but the broker response carries no real ticket at all.
    order_response = {
        "orderFillTransaction": {
            "price": "1.0991",
            "tradeOpened": {"tradeID": ""},
            "id": "",
        }
    }
    client = _mock_client(order_response)
    alerts = []
    executor = Executor(
        _config(), client, rm,
        alert_callback=lambda event, data: alerts.append((event, data)),
    )

    result = executor.execute_signal(
        instrument="EUR_USD",
        direction="buy",
        ml_confidence=0.6,
        atr_value=0.0010,
        atr_ratio=1.0,
    )

    assert result is None
    assert executor.open_trades == {}
    assert executor.trade_history == []
    assert len(alerts) == 1
    event, data = alerts[0]
    assert event == "order_failed"
    assert data["reason"] == "missing_ticket"
    assert data["instrument"] == "EUR_USD"


def test_execute_signal_zero_ticket_is_failed_order(tmp_path):
    rm = _risk_manager(tmp_path)
    # Some brokers/mocks might return a literal "0" ticket for a rejected order.
    order_response = {
        "orderFillTransaction": {
            "price": "1.0991",
            "tradeOpened": {"tradeID": "0"},
            "id": "0",
        }
    }
    client = _mock_client(order_response)
    executor = Executor(_config(), client, rm)

    result = executor.execute_signal(
        instrument="EUR_USD",
        direction="buy",
        ml_confidence=0.6,
        atr_value=0.0010,
        atr_ratio=1.0,
    )

    assert result is None
    assert executor.open_trades == {}


def test_execute_signal_with_real_ticket_opens_and_tracks_trade(tmp_path):
    rm = _risk_manager(tmp_path)
    order_response = {
        "orderFillTransaction": {
            "price": "1.0991",
            "tradeOpened": {"tradeID": "123456789"},
            "id": "987654321",
        }
    }
    client = _mock_client(order_response)
    executor = Executor(_config(), client, rm)

    result = executor.execute_signal(
        instrument="EUR_USD",
        direction="buy",
        ml_confidence=0.6,
        atr_value=0.0010,
        atr_ratio=1.0,
    )

    assert result is not None
    assert result.trade_id == "123456789"
    assert "123456789" in executor.open_trades
