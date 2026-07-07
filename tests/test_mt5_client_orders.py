"""
Tests for MT5Client order-placement hardening (Task 6 fixes N/P/O):
- Order deviation scales with current spread instead of a hard-coded 20.
- Transient retcodes (REQUOTE/PRICE_OFF/PRICE_CHANGED) retry exactly once
  with a fresh price; all other retcodes hard-fail with no retry.
- Fill validation: if the DONE result is missing/invalid price or volume,
  positions are polled immediately to repair the returned fill data.

The real `MetaTrader5` module may be importable in this environment but we
never touch a real terminal -- every mt5.* call is mocked via monkeypatch
on `src.data.mt5_client.mt5`.
"""
from unittest.mock import MagicMock

import pytest

from src.data import mt5_client as mt5_client_module
from src.data.mt5_client import MT5Client, MT5Error


def _make_client():
    config = MagicMock()
    config.get.side_effect = lambda key, default=None: default
    config.mt5_login = None
    config.mt5_password = None
    config.mt5_server = None
    return MT5Client(config)


def _install_mock_mt5(monkeypatch, symbol="EURUSD"):
    """Install a mocked mt5 module wired up for a successful order flow."""
    mock_mt5 = MagicMock()
    mock_mt5.TRADE_ACTION_DEAL = "DEAL"
    mock_mt5.ORDER_TYPE_BUY = "BUY"
    mock_mt5.ORDER_TYPE_SELL = "SELL"
    mock_mt5.ORDER_TIME_GTC = "GTC"
    mock_mt5.ORDER_FILLING_IOC = "IOC"
    mock_mt5.TRADE_RETCODE_DONE = 10009

    symbol_info = MagicMock(
        point=0.00001, digits=5, trade_contract_size=100000,
        volume_step=0.01, volume_min=0.01, volume_max=100.0,
    )
    mock_mt5.symbol_info.return_value = symbol_info
    mock_mt5.symbol_select.return_value = True

    monkeypatch.setattr(mt5_client_module, "mt5", mock_mt5)
    return mock_mt5


# ----------------------------------------------------------------------
# (N) Deviation scales with spread
# ----------------------------------------------------------------------

def test_deviation_floor_is_20_for_tight_spread(monkeypatch):
    client = _make_client()
    mock_mt5 = _install_mock_mt5(monkeypatch)
    # 1-point spread: 1 * 1.5 = 1.5 -> ceil 2, floored to 20
    tick = MagicMock(bid=1.10000, ask=1.10001)
    assert client._compute_deviation("EURUSD", tick) == 20


def test_deviation_scales_above_floor_for_wide_spread(monkeypatch):
    client = _make_client()
    _install_mock_mt5(monkeypatch)
    # spread = 0.00030 / point(0.00001) = 30 points; 30 * 1.5 = 45
    tick = MagicMock(bid=1.10000, ask=1.10030)
    assert client._compute_deviation("EURUSD", tick) == 45


def test_deviation_ceils_fractional_result(monkeypatch):
    client = _make_client()
    _install_mock_mt5(monkeypatch)
    # spread = 0.00021 / 0.00001 = 21 points; 21 * 1.5 = 31.5 -> ceil 32
    tick = MagicMock(bid=1.10000, ask=1.10021)
    assert client._compute_deviation("EURUSD", tick) == 32


def test_place_market_order_uses_computed_deviation(monkeypatch):
    client = _make_client()
    mock_mt5 = _install_mock_mt5(monkeypatch)
    tick = MagicMock(bid=1.10000, ask=1.10030, time=1735689600)
    mock_mt5.symbol_info_tick.return_value = tick

    result = MagicMock(retcode=10009, price=1.10030, volume=0.10, deal=1, order=2, comment="ok")
    mock_mt5.order_send.return_value = result

    client.place_market_order("EUR_USD", 10000)

    sent_request = mock_mt5.order_send.call_args[0][0]
    assert sent_request["deviation"] == 45


# ----------------------------------------------------------------------
# (P) Retcode retry: exactly one retry on transient retcodes
# ----------------------------------------------------------------------

@pytest.mark.parametrize("retryable_retcode", [10004, 10021, 10020])
def test_transient_retcode_retries_once_then_succeeds(monkeypatch, retryable_retcode):
    client = _make_client()
    mock_mt5 = _install_mock_mt5(monkeypatch)
    tick = MagicMock(bid=1.10000, ask=1.10001, time=1735689600)
    mock_mt5.symbol_info_tick.return_value = tick

    first_result = MagicMock(retcode=retryable_retcode, comment="requote")
    second_result = MagicMock(retcode=10009, price=1.10002, volume=0.10, deal=1, order=2, comment="ok")
    mock_mt5.order_send.side_effect = [first_result, second_result]

    response = client.place_market_order("EUR_USD", 10000)

    assert mock_mt5.order_send.call_count == 2
    assert response["orderFillTransaction"]["price"] == "1.10002"


def test_transient_retcode_retry_fails_hard_after_second_attempt(monkeypatch):
    client = _make_client()
    mock_mt5 = _install_mock_mt5(monkeypatch)
    tick = MagicMock(bid=1.10000, ask=1.10001, time=1735689600)
    mock_mt5.symbol_info_tick.return_value = tick

    first_result = MagicMock(retcode=10004, comment="requote")
    second_result = MagicMock(retcode=10004, comment="requote again")
    mock_mt5.order_send.side_effect = [first_result, second_result]

    with pytest.raises(MT5Error):
        client.place_market_order("EUR_USD", 10000)

    # Exactly one retry -- never a third attempt.
    assert mock_mt5.order_send.call_count == 2


def test_non_transient_retcode_hard_fails_with_no_retry(monkeypatch):
    client = _make_client()
    mock_mt5 = _install_mock_mt5(monkeypatch)
    tick = MagicMock(bid=1.10000, ask=1.10001, time=1735689600)
    mock_mt5.symbol_info_tick.return_value = tick

    result = MagicMock(retcode=10013, comment="invalid request")  # not in retry set
    mock_mt5.order_send.return_value = result

    with pytest.raises(MT5Error):
        client.place_market_order("EUR_USD", 10000)

    assert mock_mt5.order_send.call_count == 1


# ----------------------------------------------------------------------
# (O) Fill validation
# ----------------------------------------------------------------------

def test_fill_with_valid_price_and_volume_passes_through(monkeypatch):
    client = _make_client()
    mock_mt5 = _install_mock_mt5(monkeypatch)
    tick = MagicMock(bid=1.10000, ask=1.10001, time=1735689600)
    mock_mt5.symbol_info_tick.return_value = tick

    result = MagicMock(retcode=10009, price=1.10001, volume=0.10, deal=1, order=2, comment="ok")
    mock_mt5.order_send.return_value = result

    response = client.place_market_order("EUR_USD", 10000)

    assert response["orderFillTransaction"]["price"] == "1.10001"
    assert response["orderFillTransaction"]["units"] == "0.1"
    # No repair needed -> no position poll.
    mock_mt5.positions_get.assert_not_called()


def test_fill_missing_price_polls_positions_for_repair(monkeypatch):
    client = _make_client()
    mock_mt5 = _install_mock_mt5(monkeypatch)
    tick = MagicMock(bid=1.10000, ask=1.10001, time=1735689600)
    mock_mt5.symbol_info_tick.return_value = tick

    result = MagicMock(retcode=10009, price=0.0, volume=0.10, deal=1, order=2, comment="ok")
    mock_mt5.order_send.return_value = result

    position = MagicMock(price_open=1.10001, volume=0.10)
    mock_mt5.positions_get.return_value = [position]

    response = client.place_market_order("EUR_USD", 10000)

    mock_mt5.positions_get.assert_called_once_with(ticket=2)
    assert response["orderFillTransaction"]["price"] == "1.10001"


def test_fill_missing_volume_polls_positions_for_repair(monkeypatch):
    client = _make_client()
    mock_mt5 = _install_mock_mt5(monkeypatch)
    tick = MagicMock(bid=1.10000, ask=1.10001, time=1735689600)
    mock_mt5.symbol_info_tick.return_value = tick

    result = MagicMock(retcode=10009, price=1.10001, volume=None, deal=1, order=2, comment="ok")
    mock_mt5.order_send.return_value = result

    position = MagicMock(price_open=1.10001, volume=0.10)
    mock_mt5.positions_get.return_value = [position]

    response = client.place_market_order("EUR_USD", 10000)

    mock_mt5.positions_get.assert_called_once_with(ticket=2)
    assert response["orderFillTransaction"]["units"] == "0.1"


def test_fill_repair_falls_back_when_position_not_found(monkeypatch):
    """
    When both the order result AND the position poll fail to yield a
    usable price/volume, the fill must fall back to the already-known
    pre-order request price (the tick ask/bid used to build the order),
    never to 0.0/negative -- a 0.0 entry_price would corrupt downstream
    trailing-SL and P&L math on a live account (Task 6 review finding).
    """
    client = _make_client()
    mock_mt5 = _install_mock_mt5(monkeypatch)
    tick = MagicMock(bid=1.10000, ask=1.10001, time=1735689600)
    mock_mt5.symbol_info_tick.return_value = tick

    result = MagicMock(retcode=10009, price=0.0, volume=0.0, deal=1, order=2, comment="ok")
    mock_mt5.order_send.return_value = result
    mock_mt5.positions_get.return_value = []

    # Must not raise -- falls back to the pre-order request price (BUY -> ask)
    # and the expected order volume, flagged as estimated.
    response = client.place_market_order("EUR_USD", 10000)
    fill = response["orderFillTransaction"]
    assert fill["price"] == "1.10001"
    assert fill["units"] == "0.1"
    assert fill["fill_price_estimated"] is True


@pytest.mark.parametrize(
    "result_kwargs",
    [
        dict(price=0.0, volume=0.0),
        dict(price=None, volume=None),
        dict(price=-1.0, volume=0.10),
        dict(price=1.10001, volume=-0.10),
    ],
)
def test_fill_repair_never_returns_nonpositive_price_or_volume(monkeypatch, result_kwargs):
    """No unrepairable-fill code path may surface price<=0 or volume<=0."""
    client = _make_client()
    mock_mt5 = _install_mock_mt5(monkeypatch)
    tick = MagicMock(bid=1.10000, ask=1.10001, time=1735689600)
    mock_mt5.symbol_info_tick.return_value = tick

    result = MagicMock(retcode=10009, deal=1, order=2, comment="ok", **result_kwargs)
    mock_mt5.order_send.return_value = result
    mock_mt5.positions_get.return_value = []

    response = client.place_market_order("EUR_USD", 10000)
    fill = response["orderFillTransaction"]
    assert float(fill["price"]) > 0
    assert float(fill["units"]) > 0


def test_fill_repair_raises_when_request_price_unavailable(monkeypatch):
    """If even the pre-order request price is invalid, hard-fail loudly."""
    client = _make_client()
    mock_mt5 = _install_mock_mt5(monkeypatch)
    tick = MagicMock(bid=0.0, ask=0.0, time=1735689600)
    mock_mt5.symbol_info_tick.return_value = tick

    result = MagicMock(retcode=10009, price=0.0, volume=0.0, deal=1, order=2, comment="ok")
    mock_mt5.order_send.return_value = result
    mock_mt5.positions_get.return_value = []

    with pytest.raises(MT5Error):
        client.place_market_order("EUR_USD", 10000)


# ----------------------------------------------------------------------
# close_trade uses spread-aware deviation (not hard-coded 20)
# ----------------------------------------------------------------------

def test_close_trade_uses_computed_deviation(monkeypatch):
    client = _make_client()
    mock_mt5 = _install_mock_mt5(monkeypatch)

    position = MagicMock(symbol="EURUSD", type=mock_mt5.ORDER_TYPE_BUY, volume=0.10, ticket=2)
    mock_mt5.positions_get.return_value = [position]

    # spread = 0.00030 / point(0.00001) = 30 points; 30 * 1.5 = 45
    tick = MagicMock(bid=1.10000, ask=1.10030, time=1735689600)
    mock_mt5.symbol_info_tick.return_value = tick

    result = MagicMock(retcode=10009, price=1.10000, comment="ok")
    mock_mt5.order_send.return_value = result

    client.close_trade("2")

    sent_request = mock_mt5.order_send.call_args[0][0]
    assert sent_request["deviation"] == 45


# ----------------------------------------------------------------------
# Final-review blocker: _units_to_lots must round DOWN and reject
# below-minimum sizes instead of inflating them to the broker minimum
# (which breached the 5x leverage cap on Standard accounts at R1000).
# ----------------------------------------------------------------------

def test_units_to_lots_rounds_down_to_lot_step(monkeypatch):
    _install_mock_mt5(monkeypatch)
    client = _make_client()
    # 1,500 units = 0.015 lots -> floor to 0.01, never half-round to 0.02.
    assert client._units_to_lots("EURUSD", 1500) == pytest.approx(0.01)


def test_units_to_lots_rejects_below_broker_minimum(monkeypatch):
    _install_mock_mt5(monkeypatch)
    client = _make_client()
    # 250 units = 0.0025 lots < volume_min 0.01: reject, do NOT clamp up
    # to 1,000 units (~18x leverage at R1000).
    with pytest.raises(MT5Error, match="below broker minimum"):
        client._units_to_lots("EURUSD", 250)


def test_units_to_lots_exact_minimum_passes(monkeypatch):
    _install_mock_mt5(monkeypatch)
    client = _make_client()
    assert client._units_to_lots("EURUSD", 1000) == pytest.approx(0.01)


def test_units_to_lots_caps_at_volume_max(monkeypatch):
    mock = _install_mock_mt5(monkeypatch)
    mock.symbol_info.return_value.volume_max = 2.0
    client = _make_client()
    assert client._units_to_lots("EURUSD", 500000) == pytest.approx(2.0)
