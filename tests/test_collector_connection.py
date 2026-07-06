"""
Tests for DataCollector connectivity hardening (Blockers B1/B2/B3 + I):
- disconnect detected -> broker_down True, entries paused, mt5.disconnected alert
- reconnect detected -> symbol resync, broker_down False, mt5.reconnected alert
- idempotent: repeated disconnected/connected checks don't re-fire alerts
- flap debounce (Task 3 review fix #2): a state flip requires
  `_flap_debounce_count` (2) CONSECUTIVE agreeing checks, so a single
  flaky probe result never flips `broker_down` or fires an alert.
"""
import time
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

    # First failed check: debounced, no flip/alert yet.
    result1 = collector.check_connection()
    assert result1 is False
    assert collector.broker_down is False
    telegram.mt5_disconnected.assert_not_called()

    # Second CONSECUTIVE failed check: flips + alerts.
    result2 = collector.check_connection()
    assert result2 is False
    assert collector.broker_down is True
    telegram.mt5_disconnected.assert_called_once()


def test_check_connection_disconnect_alert_fires_once_not_repeatedly():
    collector, client, telegram = _make_collector()
    client.is_broker_connected.return_value = False

    collector.check_connection()
    collector.check_connection()
    collector.check_connection()
    collector.check_connection()

    assert collector.broker_down is True
    telegram.mt5_disconnected.assert_called_once()


def test_check_connection_single_flaky_failure_does_not_flip_or_alert():
    """Finding #2: one bad check followed by a good one must not flap
    broker_down or fire any alert — the failure streak resets."""
    collector, client, telegram = _make_collector()

    client.is_broker_connected.return_value = False
    collector.check_connection()
    assert collector.broker_down is False
    telegram.mt5_disconnected.assert_not_called()

    # Recovers before the debounce threshold is reached.
    client.is_broker_connected.return_value = True
    result = collector.check_connection()

    assert result is True
    assert collector.broker_down is False
    telegram.mt5_disconnected.assert_not_called()
    telegram.mt5_reconnected.assert_not_called()  # was never actually down


def test_check_connection_recovery_resyncs_symbols_and_resumes():
    collector, client, telegram = _make_collector()

    # Start disconnected (2 consecutive failures to cross debounce)
    client.is_broker_connected.return_value = False
    collector.check_connection()
    collector.check_connection()
    assert collector.broker_down is True

    # Recover — needs 2 consecutive successful checks too.
    client.reset_mock()
    client.is_broker_connected.return_value = True

    result1 = collector.check_connection()
    assert result1 is True
    assert collector.broker_down is True  # not yet — only 1 success so far
    telegram.mt5_reconnected.assert_not_called()

    result2 = collector.check_connection()

    assert result2 is True
    assert collector.broker_down is False
    telegram.mt5_reconnected.assert_called_once()
    # Symbol suffix re-detected for every enabled instrument
    client._to_mt5_symbol.assert_any_call("EUR_USD")
    client._to_mt5_symbol.assert_any_call("XAU_USD")


def test_check_connection_never_raises_on_client_exception():
    collector, client, telegram = _make_collector()
    client.is_broker_connected.side_effect = RuntimeError("boom")

    collector.check_connection()
    result = collector.check_connection()

    assert result is False
    assert collector.broker_down is True
    telegram.mt5_disconnected.assert_called_once()


def test_check_connection_never_raises_when_telegram_fails():
    collector, client, telegram = _make_collector()
    client.is_broker_connected.return_value = False
    telegram.mt5_disconnected.side_effect = RuntimeError("telegram down")

    # Should not raise even though the alert call itself errors
    collector.check_connection()
    result = collector.check_connection()

    assert result is False
    assert collector.broker_down is True


def test_broker_down_defaults_false_before_any_check():
    collector, client, telegram = _make_collector()
    assert collector.broker_down is False


# ----------------------------------------------------------------------
# start_streaming() / stop_streaming() thread lifecycle (finding #5)
# ----------------------------------------------------------------------

def test_start_stop_streaming_lifecycle():
    collector, client, telegram = _make_collector()

    def fake_stream(instruments):
        # Mocked stream: no real backoff/network, just a fast tick loop
        # that keeps running until the test tells it to stop.
        while True:
            yield {"type": "HEARTBEAT"}
            time.sleep(0.01)

    client.stream_prices.side_effect = fake_stream

    collector.start_streaming()
    try:
        assert collector._streaming is True
        assert collector._stream_thread is not None
        assert collector._stream_thread.daemon is True
        assert collector._stream_thread.is_alive()
        assert collector._health_thread is not None
        assert collector._health_thread.daemon is True
        assert collector._health_thread.is_alive()

        first_thread = collector._stream_thread
        first_health_thread = collector._health_thread

        # Double-start is guarded: no new threads spawned while running.
        collector.start_streaming()
        assert collector._stream_thread is first_thread
        assert collector._health_thread is first_health_thread

        collector.stop_streaming()

        # stop_streaming() must request prompt cancellation of the client
        # stream (see MT5Client.cancel_stream()), not just flip the flag.
        client.cancel_stream.assert_called_once()
        assert collector._streaming is False

        # Threads should exit promptly since the mocked stream has no real
        # backoff to wait out.
        first_thread.join(timeout=2.0)
        assert not first_thread.is_alive()
        first_health_thread.join(timeout=2.0)
        assert not first_health_thread.is_alive()
    finally:
        # Safety net in case an assertion above fails before stop_streaming().
        collector._streaming = False
