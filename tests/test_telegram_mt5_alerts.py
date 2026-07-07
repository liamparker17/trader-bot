"""
Tests for TelegramBot MT5 connectivity alert methods
(mt5_connected / mt5_disconnected / mt5_reconnected).

Alerts must be best-effort: they must never raise into the caller,
even if the underlying send mechanism blows up.
"""
from unittest.mock import MagicMock

from src.monitoring.telegram_bot import TelegramBot


def _make_bot():
    config = MagicMock()
    config.get.side_effect = lambda key, default=None: default
    config.telegram_bot_token = "token"
    config.telegram_chat_id = "chat"
    bot = TelegramBot(config)
    return bot


def test_mt5_disconnected_sends_message():
    bot = _make_bot()
    bot._send = MagicMock()

    bot.mt5_disconnected(reason="terminal closed")

    bot._send.assert_called_once()
    text = bot._send.call_args[0][0]
    assert "Disconnected" in text
    assert "terminal closed" in text


def test_mt5_reconnected_sends_message():
    bot = _make_bot()
    bot._send = MagicMock()

    bot.mt5_reconnected()

    bot._send.assert_called_once()
    text = bot._send.call_args[0][0]
    assert "Reconnected" in text


def test_mt5_connected_sends_message_with_environment():
    bot = _make_bot()
    bot._send = MagicMock()

    bot.mt5_connected(environment="demo")

    bot._send.assert_called_once()
    text = bot._send.call_args[0][0]
    assert "Connected" in text
    assert "demo" in text


def test_mt5_disconnected_never_raises_when_send_fails():
    bot = _make_bot()
    bot._send = MagicMock(side_effect=RuntimeError("network down"))

    bot.mt5_disconnected()  # must not raise


def test_mt5_reconnected_never_raises_when_send_fails():
    bot = _make_bot()
    bot._send = MagicMock(side_effect=RuntimeError("network down"))

    bot.mt5_reconnected()  # must not raise


def test_mt5_connected_never_raises_when_send_fails():
    bot = _make_bot()
    bot._send = MagicMock(side_effect=RuntimeError("network down"))

    bot.mt5_connected()  # must not raise


# ---------------------------------------------------------------------------
# bot_error (Task 5, item S: main-loop catch-all alert)
# ---------------------------------------------------------------------------

def test_bot_error_sends_message_with_type_and_message():
    bot = _make_bot()
    bot._send = MagicMock()

    bot.bot_error("ValueError", "boom")

    bot._send.assert_called_once()
    text = bot._send.call_args[0][0]
    assert "ValueError" in text
    assert "boom" in text


def test_bot_error_never_raises_when_send_fails():
    bot = _make_bot()
    bot._send = MagicMock(side_effect=RuntimeError("network down"))

    bot.bot_error("ValueError", "boom")  # must not raise


def test_bot_error_respects_disabled_toggle():
    config = MagicMock()
    config.get.side_effect = lambda key, default=None: (
        False if key == "telegram.alert_on_bot_error" else default
    )
    config.telegram_bot_token = "token"
    config.telegram_chat_id = "chat"
    bot = TelegramBot(config)
    bot._send = MagicMock()

    bot.bot_error("ValueError", "boom")

    bot._send.assert_not_called()
