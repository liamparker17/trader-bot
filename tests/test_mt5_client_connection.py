"""
Tests for MT5Client connectivity hardening:
- is_broker_connected() health probe
- stream_prices() exponential backoff with jitter on repeated errors,
  reset on success, capped at 60s.
- backoff/poll sleeps are interruptible via cancel_stream() (Task 3 review
  fix #1), and the probe is consulted as diagnosis only when the tick loop
  itself errored, not as an independent gate (Task 3 review fix #3).

The real `MetaTrader5` module is importable in this environment (Windows
package installed) but we never want tests to touch a real terminal, so
every mt5.* call is mocked via monkeypatch on `src.data.mt5_client.mt5`.
"""
from unittest.mock import MagicMock

import pytest

from src.data import mt5_client as mt5_client_module
from src.data.mt5_client import MT5Client


class _StopTest(Exception):
    """Sentinel used to break out of the infinite stream_prices() loop in tests."""


def _make_client(monkeypatch):
    config = MagicMock()
    config.get.side_effect = lambda key, default=None: default
    config.mt5_login = None
    config.mt5_password = None
    config.mt5_server = None
    client = MT5Client(config)
    return client


# ----------------------------------------------------------------------
# is_broker_connected()
# ----------------------------------------------------------------------

def test_is_broker_connected_true_when_terminal_and_account_ok(monkeypatch):
    client = _make_client(monkeypatch)
    mock_mt5 = MagicMock()
    mock_mt5.terminal_info.return_value = MagicMock(connected=True)
    mock_mt5.account_info.return_value = MagicMock()
    monkeypatch.setattr(mt5_client_module, "mt5", mock_mt5)

    assert client.is_broker_connected() is True


def test_is_broker_connected_false_when_terminal_none(monkeypatch):
    client = _make_client(monkeypatch)
    mock_mt5 = MagicMock()
    mock_mt5.terminal_info.return_value = None
    monkeypatch.setattr(mt5_client_module, "mt5", mock_mt5)

    assert client.is_broker_connected() is False


def test_is_broker_connected_false_when_terminal_not_connected(monkeypatch):
    client = _make_client(monkeypatch)
    mock_mt5 = MagicMock()
    mock_mt5.terminal_info.return_value = MagicMock(connected=False)
    monkeypatch.setattr(mt5_client_module, "mt5", mock_mt5)

    assert client.is_broker_connected() is False


def test_is_broker_connected_false_when_account_info_none(monkeypatch):
    client = _make_client(monkeypatch)
    mock_mt5 = MagicMock()
    mock_mt5.terminal_info.return_value = MagicMock(connected=True)
    mock_mt5.account_info.return_value = None
    monkeypatch.setattr(mt5_client_module, "mt5", mock_mt5)

    assert client.is_broker_connected() is False


def test_is_broker_connected_false_on_exception_never_raises(monkeypatch):
    client = _make_client(monkeypatch)
    mock_mt5 = MagicMock()
    mock_mt5.terminal_info.side_effect = RuntimeError("boom")
    monkeypatch.setattr(mt5_client_module, "mt5", mock_mt5)

    assert client.is_broker_connected() is False


# ----------------------------------------------------------------------
# stream_prices() backoff
# ----------------------------------------------------------------------

def _setup_stream_mocks(monkeypatch, client, connected: bool):
    mock_mt5 = MagicMock()
    mock_mt5.symbol_info.return_value = MagicMock()
    mock_mt5.symbol_select.return_value = True
    if connected:
        mock_mt5.symbol_info_tick.return_value = None  # healthy: no ticks, no error
    else:
        # Disconnected is modeled as the tick fetch itself failing — this is
        # what actually drives `loop_had_error` (see fix #3: the probe below
        # is consulted only as diagnosis once an error has already occurred,
        # never as an independent gate).
        mock_mt5.symbol_info_tick.side_effect = RuntimeError("tick fetch failed")
    mock_mt5.terminal_info.return_value = MagicMock(connected=connected)
    mock_mt5.account_info.return_value = MagicMock() if connected else None
    monkeypatch.setattr(mt5_client_module, "mt5", mock_mt5)
    # Skip ensure_connected()'s real reconnect path
    monkeypatch.setattr(client, "ensure_connected", lambda: None)
    return mock_mt5


def test_stream_prices_backs_off_exponentially_on_disconnect(monkeypatch):
    client = _make_client(monkeypatch)
    _setup_stream_mocks(monkeypatch, client, connected=False)

    sleep_calls = []

    def fake_interruptible_sleep(seconds):
        sleep_calls.append(seconds)
        if len(sleep_calls) >= 4:
            raise _StopTest()

    monkeypatch.setattr(client, "_interruptible_sleep", fake_interruptible_sleep)
    monkeypatch.setattr(mt5_client_module.random, "uniform", lambda a, b: 0.0)

    gen = client.stream_prices(["EUR_USD"])
    with pytest.raises(_StopTest):
        next(gen)

    # Backoff should double each time: 1, 2, 4, 8 (jitter forced to 0)
    assert sleep_calls == [1.0, 2.0, 4.0, 8.0]


def test_stream_prices_backoff_capped_at_60s(monkeypatch):
    client = _make_client(monkeypatch)
    _setup_stream_mocks(monkeypatch, client, connected=False)

    sleep_calls = []

    def fake_interruptible_sleep(seconds):
        sleep_calls.append(seconds)
        if len(sleep_calls) >= 10:
            raise _StopTest()

    monkeypatch.setattr(client, "_interruptible_sleep", fake_interruptible_sleep)
    monkeypatch.setattr(mt5_client_module.random, "uniform", lambda a, b: 0.0)

    gen = client.stream_prices(["EUR_USD"])
    with pytest.raises(_StopTest):
        next(gen)

    assert max(sleep_calls) <= 60.0
    assert sleep_calls[-1] == 60.0  # after enough doublings, it's pinned at the cap


def test_stream_prices_resets_backoff_on_success(monkeypatch):
    client = _make_client(monkeypatch)
    mock_mt5 = _setup_stream_mocks(monkeypatch, client, connected=False)

    sleep_calls = []
    call_count = {"n": 0}

    def fake_interruptible_sleep(seconds):
        sleep_calls.append(seconds)
        call_count["n"] += 1
        if call_count["n"] == 3:
            # Recover on the 3rd iteration: tick fetch stops raising and the
            # broker probe comes back healthy too.
            mock_mt5.symbol_info_tick.side_effect = None
            mock_mt5.symbol_info_tick.return_value = None
            mock_mt5.terminal_info.return_value = MagicMock(connected=True)
            mock_mt5.account_info.return_value = MagicMock()
        if call_count["n"] >= 5:
            raise _StopTest()

    monkeypatch.setattr(client, "_interruptible_sleep", fake_interruptible_sleep)
    monkeypatch.setattr(mt5_client_module.random, "uniform", lambda a, b: 0.0)

    gen = client.stream_prices(["EUR_USD"])
    with pytest.raises(_StopTest):
        next(gen)

    # First two iterations: backing off (1.0, 2.0).
    # Then connection restored -> normal 0.1s poll interval, no more growth.
    assert sleep_calls[0] == 1.0
    assert sleep_calls[1] == 2.0
    assert sleep_calls[3] == 0.1
    assert sleep_calls[4] == 0.1


def test_stream_prices_ignores_probe_blip_when_ticks_healthy(monkeypatch):
    """Finding #3: a transient is_broker_connected() blip must not gate/
    trigger backoff on its own while ticks are flowing without error."""
    client = _make_client(monkeypatch)
    mock_mt5 = _setup_stream_mocks(monkeypatch, client, connected=True)
    # Probe itself blips false even though tick fetch never raises.
    mock_mt5.terminal_info.return_value = MagicMock(connected=False)

    sleep_calls = []

    def fake_interruptible_sleep(seconds):
        sleep_calls.append(seconds)
        if len(sleep_calls) >= 3:
            raise _StopTest()

    monkeypatch.setattr(client, "_interruptible_sleep", fake_interruptible_sleep)

    gen = client.stream_prices(["EUR_USD"])
    with pytest.raises(_StopTest):
        next(gen)

    # No backoff growth — stays at the normal 0.1s poll interval throughout.
    assert sleep_calls == [0.1, 0.1, 0.1]


# ----------------------------------------------------------------------
# Interruptible backoff / cancel_stream()
# ----------------------------------------------------------------------

def test_interruptible_sleep_stops_promptly_on_cancel(monkeypatch):
    client = _make_client(monkeypatch)
    sleep_calls = []

    def fake_sleep(seconds):
        sleep_calls.append(seconds)
        if len(sleep_calls) == 2:
            client.cancel_stream()

    monkeypatch.setattr(mt5_client_module.time, "sleep", fake_sleep)

    # Simulate the worst case: a full max_backoff (60s) sleep request.
    client._interruptible_sleep(60.0)

    # Cancellation should cut it short after ~2 chunks instead of running
    # the full 60s / 600-chunk sleep.
    assert len(sleep_calls) == 2
    assert sum(sleep_calls) < 1.0


def test_stream_prices_backoff_is_interruptible_via_cancel_stream(monkeypatch):
    """During a backoff sleep, cancel_stream() should let the generator
    exit within about one chunk (<1s simulated), not the full backoff."""
    client = _make_client(monkeypatch)
    _setup_stream_mocks(monkeypatch, client, connected=False)

    sleep_calls = []

    def fake_sleep(seconds):
        sleep_calls.append(seconds)
        # Cancel after a couple of chunks — well before even the first
        # (1s / 10-chunk) backoff period would complete.
        if len(sleep_calls) == 2:
            client.cancel_stream()

    monkeypatch.setattr(mt5_client_module.time, "sleep", fake_sleep)
    monkeypatch.setattr(mt5_client_module.random, "uniform", lambda a, b: 0.0)

    gen = client.stream_prices(["EUR_USD"])
    # Generator should raise StopIteration (exit cleanly) instead of
    # blocking through the full backoff sleep.
    with pytest.raises(StopIteration):
        next(gen)

    assert len(sleep_calls) == 2
    assert sum(sleep_calls) < 1.0
