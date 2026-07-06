"""
Task 8 — main.py wiring: broker_down gating, balance-refresh failure
escalation, and pause-on-exception for the session-boundary/drawdown
check.

Follows the `_bare_bot()` pattern from test_daily_summary_scheduler.py:
construct TraderBot via __new__() (skipping __init__, which touches real
config/env/MT5) and wire only the attributes each method under test needs.
"""
from unittest.mock import MagicMock

from src.main import TraderBot


def _bare_bot():
    bot = TraderBot.__new__(TraderBot)
    bot.telegram = MagicMock()
    bot.journal = MagicMock()
    bot.performance = MagicMock()
    bot.performance.get_summary.return_value = {
        "total_trades": 0, "wins": 0, "losses": 0,
        "win_rate": 0.0, "total_pnl": 0.0, "max_drawdown_pct": 0.0,
    }
    bot.daily_summary_scheduler = MagicMock()
    bot.daily_summary_scheduler.due.return_value = None
    bot.client = MagicMock()
    bot.collector = MagicMock()
    bot.collector.broker_down = False
    bot.risk_manager = MagicMock()
    bot.risk_manager.check_drawdown_emergency.return_value = False
    bot.executor = MagicMock()
    bot.running = True
    bot._balance_refresh_failures = 0
    return bot


# ---------------------------------------------------------------------------
# _refresh_balance_cache
# ---------------------------------------------------------------------------

def test_refresh_balance_cache_success_resets_failure_counter():
    bot = _bare_bot()
    bot._balance_refresh_failures = 2
    bot.client.get_account_summary.return_value = {"balance": 100.0, "equity": 95.0}

    balance, equity = bot._refresh_balance_cache(None, None)

    assert (balance, equity) == (100.0, 95.0)
    assert bot._balance_refresh_failures == 0


def test_refresh_balance_cache_keeps_prior_values_on_failure():
    bot = _bare_bot()
    bot.client.get_account_summary.side_effect = RuntimeError("MT5 disconnected")

    balance, equity = bot._refresh_balance_cache(50.0, 48.0)

    assert (balance, equity) == (50.0, 48.0)
    assert bot._balance_refresh_failures == 1


def test_refresh_balance_cache_escalates_to_warning_after_3_consecutive_failures(caplog):
    bot = _bare_bot()
    bot.client.get_account_summary.side_effect = RuntimeError("MT5 disconnected")

    import logging
    with caplog.at_level(logging.DEBUG, logger="traderbot"):
        bot._refresh_balance_cache(None, None)
        bot._refresh_balance_cache(None, None)
        assert bot._balance_refresh_failures == 2
        assert not any(r.levelno == logging.WARNING for r in caplog.records)

        caplog.clear()
        bot._refresh_balance_cache(None, None)
        assert bot._balance_refresh_failures == 3
        assert any(r.levelno == logging.WARNING for r in caplog.records)


# ---------------------------------------------------------------------------
# _check_session_and_drawdown — isolated try/except + manual pause
# ---------------------------------------------------------------------------

def test_check_session_and_drawdown_happy_path_calls_risk_manager():
    bot = _bare_bot()

    bot._check_session_and_drawdown(1000.0, 990.0)

    bot.risk_manager.check_session_boundary.assert_called_once_with(1000.0)
    bot.risk_manager.check_drawdown_emergency.assert_called_once_with(1000.0, 990.0)
    bot.risk_manager.set_manual_pause.assert_not_called()
    bot.executor.close_all.assert_not_called()


def test_check_session_and_drawdown_closes_all_on_daily_breach():
    bot = _bare_bot()
    bot.risk_manager.check_drawdown_emergency.return_value = True
    bot.risk_manager.drawdown.get_daily_drawdown_pct.return_value = 0.05

    bot._check_session_and_drawdown(1000.0, 900.0)

    bot.executor.close_all.assert_called_once_with("daily_drawdown")
    bot.telegram.daily_stop.assert_called_once_with(1000.0, 0.05)


def test_check_session_and_drawdown_exception_engages_manual_pause_and_alerts():
    bot = _bare_bot()
    bot.risk_manager.check_session_boundary.side_effect = RuntimeError("boom")

    bot._check_session_and_drawdown(1000.0, 990.0)  # must not raise

    bot.risk_manager.set_manual_pause.assert_called_once()
    reason = bot.risk_manager.set_manual_pause.call_args.args[0]
    assert "boom" in reason
    bot.telegram._send.assert_called_once()


def test_check_session_and_drawdown_exception_survives_telegram_failure():
    bot = _bare_bot()
    bot.risk_manager.check_session_boundary.side_effect = RuntimeError("boom")
    bot.telegram._send.side_effect = RuntimeError("network down")

    bot._check_session_and_drawdown(1000.0, 990.0)  # must not raise

    bot.risk_manager.set_manual_pause.assert_called_once()


def test_check_session_and_drawdown_calls_daily_summary_on_happy_path():
    bot = _bare_bot()

    bot._check_session_and_drawdown(1000.0, 990.0)

    bot.daily_summary_scheduler.due.assert_called_once()


# ---------------------------------------------------------------------------
# _on_candle_complete — broker_down gating
# ---------------------------------------------------------------------------

def test_on_candle_complete_skips_evaluation_when_broker_down():
    bot = _bare_bot()
    bot.collector.broker_down = True
    bot._evaluate_trade_signal = MagicMock()

    bot._on_candle_complete("EUR_USD", "M1", candle=MagicMock())

    bot._evaluate_trade_signal.assert_not_called()


def test_on_candle_complete_evaluates_when_broker_up():
    bot = _bare_bot()
    bot.collector.broker_down = False
    bot._evaluate_trade_signal = MagicMock()

    bot._on_candle_complete("EUR_USD", "M1", candle=MagicMock())

    bot._evaluate_trade_signal.assert_called_once_with("EUR_USD")


def test_on_candle_complete_ignores_non_m1_timeframe():
    bot = _bare_bot()
    bot._evaluate_trade_signal = MagicMock()

    bot._on_candle_complete("EUR_USD", "M15", candle=MagicMock())

    bot._evaluate_trade_signal.assert_not_called()
