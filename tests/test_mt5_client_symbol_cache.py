"""
Tests for MT5Client symbol-suffix caching + re-detection (Task 6 fix L):
- _to_mt5_symbol() caches the detected suffix per instrument so repeated
  calls don't re-probe the broker.
- invalidate_symbol_cache() forces the next _to_mt5_symbol() call to
  re-detect (used on every connect()/reconnect and by DataCollector's
  reconnect resync).
"""
from unittest.mock import MagicMock

from src.data import mt5_client as mt5_client_module
from src.data.mt5_client import MT5Client


def _make_client():
    config = MagicMock()
    config.get.side_effect = lambda key, default=None: default
    config.mt5_login = None
    config.mt5_password = None
    config.mt5_server = None
    return MT5Client(config)


def test_to_mt5_symbol_caches_after_first_detection(monkeypatch):
    client = _make_client()
    mock_mt5 = MagicMock()
    mock_mt5.symbol_info.return_value = MagicMock()  # "EURUSD" found on first try
    monkeypatch.setattr(mt5_client_module, "mt5", mock_mt5)

    first = client._to_mt5_symbol("EUR_USD")
    second = client._to_mt5_symbol("EUR_USD")

    assert first == second == "EURUSD"
    # Only probed the broker once -- second call served from cache.
    assert mock_mt5.symbol_info.call_count == 1


def test_to_mt5_symbol_caches_suffix_variant(monkeypatch):
    client = _make_client()
    mock_mt5 = MagicMock()

    def symbol_info(sym):
        return MagicMock() if sym == "EURUSDm" else None

    mock_mt5.symbol_info.side_effect = symbol_info
    monkeypatch.setattr(mt5_client_module, "mt5", mock_mt5)

    first = client._to_mt5_symbol("EUR_USD")
    assert first == "EURUSDm"

    call_count_after_first = mock_mt5.symbol_info.call_count
    second = client._to_mt5_symbol("EUR_USD")

    assert second == "EURUSDm"
    # No additional symbol_info probes on the cached call.
    assert mock_mt5.symbol_info.call_count == call_count_after_first


def test_invalidate_symbol_cache_forces_redetection(monkeypatch):
    client = _make_client()
    mock_mt5 = MagicMock()
    mock_mt5.symbol_info.return_value = MagicMock()
    monkeypatch.setattr(mt5_client_module, "mt5", mock_mt5)

    client._to_mt5_symbol("EUR_USD")
    assert mock_mt5.symbol_info.call_count == 1

    client.invalidate_symbol_cache()
    client._to_mt5_symbol("EUR_USD")

    # Cache was cleared -> broker probed again.
    assert mock_mt5.symbol_info.call_count == 2


def test_invalidate_symbol_cache_picks_up_changed_suffix(monkeypatch):
    """Simulates a broker where the suffix changes across a reconnect
    (e.g. terminal re-selected a different Market Watch symbol)."""
    client = _make_client()
    mock_mt5 = MagicMock()

    # Before reconnect: only "EURUSD" (no suffix) resolves.
    mock_mt5.symbol_info.side_effect = lambda sym: MagicMock() if sym == "EURUSD" else None
    monkeypatch.setattr(mt5_client_module, "mt5", mock_mt5)

    before = client._to_mt5_symbol("EUR_USD")
    assert before == "EURUSD"

    # After "reconnect": broker now requires the "m" suffix.
    mock_mt5.symbol_info.side_effect = lambda sym: MagicMock() if sym == "EURUSDm" else None
    client.invalidate_symbol_cache()

    after = client._to_mt5_symbol("EUR_USD")
    assert after == "EURUSDm"


def test_connect_invalidates_symbol_cache(monkeypatch):
    client = _make_client()
    mock_mt5 = MagicMock()
    mock_mt5.symbol_info.return_value = MagicMock()
    mock_mt5.initialize.return_value = True
    mock_mt5.account_info.return_value = MagicMock(server="Demo", login=1, balance=500, currency="USD")
    monkeypatch.setattr(mt5_client_module, "mt5", mock_mt5)

    client._to_mt5_symbol("EUR_USD")
    assert client._symbol_cache == {"EUR_USD": "EURUSD"}

    client.connect()

    assert client._symbol_cache == {}


def test_collector_resync_invalidates_cache_before_redetecting():
    from src.data.collector import DataCollector

    config = MagicMock()
    config.get_enabled_instruments.return_value = ["EUR_USD", "XAU_USD"]
    client = MagicMock()
    collector = DataCollector(config, client, telegram=MagicMock())

    collector._resync_symbols()

    client.invalidate_symbol_cache.assert_called_once()
    client._to_mt5_symbol.assert_any_call("EUR_USD")
    client._to_mt5_symbol.assert_any_call("XAU_USD")
