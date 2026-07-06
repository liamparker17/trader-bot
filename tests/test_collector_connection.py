"""
Tests for DataCollector connectivity hardening (Blockers B1/B2/B3 + I):
- disconnect detected -> broker_down True, entries paused, mt5.disconnected alert
- reconnect detected -> symbol resync, broker_down False, mt5.reconnected alert
- idempotent: repeated disconnected/connected checks don't re-fire alerts
"""
from unittest.mock import MagicMock

from src.data.collector import DataCollector


def _make_collector():
    config = MagicMock()
    config.get_enabled_instruments.return_value = ["EUR_USD", "XAU_USD"]

    client = MagicMock()
    telegram = MagicMock()

    collector = DataCollector(config, client, telegram=telegram)
    return collector, client, telegram


def test_check_connection_stays_up_when_connected():
    collector, client, telegram = _make_collector()
    client.is_broker_connected.return_value = True

    result = collector.check_connection()

    assert result is True
    assert collector.broker_down is False
    telegram.mt5_disconnected.assert_not_called()
    telegram.mt5_reconnected.assert_not_called()


def test_check_connection_detects_disconnect_and_pauses_entries():
    collector, client, telegram = _make_collector()
    client.is_broker_connected.return_value = False

    result = collector.check_connection()

    assert result is False
    assert collector.broker_down is True
    telegram.mt5_disconnected.assert_called_once()


def test_check_connection_disconnect_alert_fires_once_not_repeatedly():
    collector, client, telegram = _make_collector()
    client.is_broker_connected.return_value = False

    collector.check_connection()
    collector.check_connection()
    collector.check_connection()

    assert collector.broker_down is True
    telegram.mt5_disconnected.assert_called_once()


def test_check_connection_recovery_resyncs_symbols_and_resumes():
    collector, client, telegram = _make_collector()

    # Start disconnected
    client.is_broker_connected.return_value = False
    collector.check_connection()
    assert collector.broker_down is True

    # Recover
    client.is_broker_connected.return_value = True
    client.reset_mock()
    client.is_broker_connected.return_value = True

    result = collector.check_connection()

    assert result is True
    assert collector.broker_down is False
    telegram.mt5_reconnected.assert_called_once()
    # Symbol suffix re-detected for every enabled instrument
    client._to_mt5_symbol.assert_any_call("EUR_USD")
    client._to_mt5_symbol.assert_any_call("XAU_USD")


def test_check_connection_never_raises_on_client_exception():
    collector, client, telegram = _make_collector()
    client.is_broker_connected.side_effect = RuntimeError("boom")

    result = collector.check_connection()

    assert result is False
    assert collector.broker_down is True
    telegram.mt5_disconnected.assert_called_once()


def test_check_connection_never_raises_when_telegram_fails():
    collector, client, telegram = _make_collector()
    client.is_broker_connected.return_value = False
    telegram.mt5_disconnected.side_effect = RuntimeError("telegram down")

    # Should not raise even though the alert call itself errors
    result = collector.check_connection()

    assert result is False
    assert collector.broker_down is True


def test_broker_down_defaults_false_before_any_check():
    collector, client, telegram = _make_collector()
    assert collector.broker_down is False
