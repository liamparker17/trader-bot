"""
Telegram Alert Bot — Sends real-time notifications via Telegram.

Alert types:
- Trade opened/closed
- Daily drawdown limit hit
- Emergency shutdown
- ML model retrained
- Balance milestones
- API errors
- Daily summary report
"""

import logging
import threading
from datetime import datetime, timezone
from typing import Optional

import requests

from src.config import Config

logger = logging.getLogger("traderbot.telegram")


class TelegramBot:
    """
    Sends alerts via Telegram Bot API.

    Uses simple HTTP requests (not the full python-telegram-bot library)
    for lightweight, non-blocking alerts. Messages are sent in a background
    thread to avoid blocking the trading loop.
    """

    def __init__(self, config: Config):
        self.config = config
        self.enabled = config.get("telegram.enabled", True)
        self.token = config.telegram_bot_token
        self.chat_id = config.telegram_chat_id

        # Alert toggles from config
        self.alert_trade_open = config.get("telegram.alert_on_trade_open", True)
        self.alert_trade_close = config.get("telegram.alert_on_trade_close", True)
        self.alert_daily_stop = config.get("telegram.alert_on_daily_stop", True)
        self.alert_emergency = config.get("telegram.alert_on_emergency_stop", True)
        self.alert_retrain = config.get("telegram.alert_on_ml_retrain", True)
        self.alert_api_error = config.get("telegram.alert_on_api_error", True)
        self.alert_milestone = config.get("telegram.alert_on_milestone", True)
        self.alert_bot_error = config.get("telegram.alert_on_bot_error", True)

        if self.enabled and (not self.token or not self.chat_id):
            logger.warning("Telegram enabled but token/chat_id not set. Alerts disabled.")
            self.enabled = False

    def _send(self, text: str, parse_mode: str = "HTML"):
        """Send a message via Telegram in a background thread."""
        if not self.enabled:
            return

        def _do_send():
            try:
                url = f"https://api.telegram.org/bot{self.token}/sendMessage"
                payload = {
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                }
                response = requests.post(url, json=payload, timeout=10)
                if response.status_code != 200:
                    logger.warning(f"Telegram send failed: {response.status_code} {response.text}")
            except Exception as e:
                logger.warning(f"Telegram error: {e}")

        thread = threading.Thread(target=_do_send, daemon=True)
        thread.start()

    # ------------------------------------------------------------------
    # Trade alerts
    # ------------------------------------------------------------------

    def trade_opened(
        self,
        instrument: str,
        direction: str,
        units: int,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        confidence: float,
        risk_amount: float,
    ):
        """Alert: new trade opened."""
        if not self.alert_trade_open:
            return

        emoji = "\U0001f7e2" if direction == "buy" else "\U0001f534"  # green/red circle
        arrow = "\u2191" if direction == "buy" else "\u2193"

        text = (
            f"{emoji} <b>{direction.upper()}</b> {instrument}\n"
            f"{arrow} Entry: {entry_price:.5f}\n"
            f"\U0001f6d1 SL: {stop_loss:.5f}\n"
            f"\U0001f3af TP: {take_profit:.5f}\n"
            f"\U0001f4ca Confidence: {confidence:.0%}\n"
            f"\U0001f4b0 Risk: R{risk_amount:.2f} | Units: {abs(units)}"
        )
        self._send(text)

    def trade_closed(
        self,
        instrument: str,
        direction: str,
        pnl_pips: float,
        pnl_zar: float,
        exit_reason: str,
        balance: float,
    ):
        """Alert: trade closed."""
        if not self.alert_trade_close:
            return

        if pnl_zar > 0:
            emoji = "\u2705"  # check mark
            sign = "+"
        else:
            emoji = "\U0001f534"  # red circle
            sign = ""

        text = (
            f"{emoji} <b>CLOSED</b> {instrument} {direction.upper()}\n"
            f"PnL: {sign}{pnl_pips:.1f} pips ({sign}R{pnl_zar:.2f})\n"
            f"Reason: {exit_reason}\n"
            f"\U0001f4b0 Balance: R{balance:.2f}"
        )
        self._send(text)

    # ------------------------------------------------------------------
    # Risk alerts
    # ------------------------------------------------------------------

    def daily_stop(self, balance: float, drawdown_pct: float):
        """Alert: daily drawdown limit hit."""
        if not self.alert_daily_stop:
            return

        text = (
            f"\u26d4 <b>DAILY STOP</b>\n"
            f"Drawdown: {drawdown_pct:.1%}\n"
            f"Trading paused until tomorrow.\n"
            f"\U0001f4b0 Balance: R{balance:.2f}"
        )
        self._send(text)

    def emergency_stop(self, balance: float, reason: str):
        """Alert: emergency shutdown triggered."""
        if not self.alert_emergency:
            return

        text = (
            f"\U0001f6a8 <b>EMERGENCY SHUTDOWN</b>\n"
            f"Reason: {reason}\n"
            f"\U0001f4b0 Balance: R{balance:.2f}\n"
            f"\n<i>Manual intervention required to resume.</i>"
        )
        self._send(text)

    def consecutive_losses(self, count: int, action: str):
        """Alert: consecutive loss threshold hit."""
        text = (
            f"\u26a0\ufe0f <b>{count} Consecutive Losses</b>\n"
            f"Action: {action}"
        )
        self._send(text)

    # ------------------------------------------------------------------
    # ML alerts
    # ------------------------------------------------------------------

    def ml_retrained(
        self,
        old_version: str,
        new_version: str,
        old_accuracy: float,
        new_accuracy: float,
    ):
        """Alert: ML model retrained."""
        if not self.alert_retrain:
            return

        direction = "\u2191" if new_accuracy > old_accuracy else "\u2193"

        text = (
            f"\U0001f504 <b>Model Retrained</b>\n"
            f"Version: {old_version} \u2192 {new_version}\n"
            f"Accuracy: {old_accuracy:.1%} {direction} {new_accuracy:.1%}"
        )
        self._send(text)

    # ------------------------------------------------------------------
    # Growth alerts
    # ------------------------------------------------------------------

    def milestone_reached(self, milestone: float, balance: float, message: str):
        """Alert: balance milestone reached."""
        if not self.alert_milestone:
            return

        text = f"\U0001f3c6 <b>MILESTONE</b>\n{message}"
        self._send(text)

    # ------------------------------------------------------------------
    # System alerts
    # ------------------------------------------------------------------

    def api_error(self, error_message: str):
        """Alert: API connection issue."""
        if not self.alert_api_error:
            return

        text = (
            f"\u26a0\ufe0f <b>API Error</b>\n"
            f"{error_message}"
        )
        self._send(text)

    def bot_error(self, exc_type: str, message: str):
        """
        Alert: unhandled exception caught by the main-loop catch-all.
        Best-effort \u2014 must never raise into the caller (the whole point of
        this alert is to survive errors, so a broken send path can't be
        allowed to take the loop down too).
        """
        if not self.alert_bot_error:
            return
        try:
            text = (
                f"\U0001f6a8 <b>Bot Error</b>\n"
                f"Type: {exc_type}\n"
                f"{message}"
            )
            self._send(text)
        except Exception as e:
            logger.warning(f"bot_error alert failed: {e}")

    def bot_started(self, balance: float, environment: str):
        """Alert: bot started."""
        text = (
            f"\U0001f916 <b>TraderBot Started</b>\n"
            f"Environment: {environment}\n"
            f"\U0001f4b0 Balance: R{balance:.2f}"
        )
        self._send(text)

    def bot_stopped(self, balance: float, reason: str = "manual"):
        """Alert: bot stopped."""
        text = (
            f"\U0001f6d1 <b>TraderBot Stopped</b>\n"
            f"Reason: {reason}\n"
            f"\U0001f4b0 Balance: R{balance:.2f}"
        )
        self._send(text)

    # ------------------------------------------------------------------
    # MT5 connectivity alerts
    # ------------------------------------------------------------------

    def mt5_connected(self, environment: str = ""):
        """Alert: MT5 broker connection established. Best-effort, never raises."""
        try:
            text = "\U0001f7e2 <b>MT5 Connected</b>"
            if environment:
                text += f"\nEnvironment: {environment}"
            self._send(text)
        except Exception as e:
            logger.warning(f"mt5_connected alert failed: {e}")

    def mt5_disconnected(self, reason: str = "connection lost"):
        """Alert: MT5 broker connection lost, new entries paused. Best-effort, never raises."""
        try:
            text = (
                f"\U0001f6ab <b>MT5 Disconnected</b>\n"
                f"Reason: {reason}\n"
                f"New entries paused until reconnect."
            )
            self._send(text)
        except Exception as e:
            logger.warning(f"mt5_disconnected alert failed: {e}")

    def mt5_reconnected(self):
        """Alert: MT5 broker connection restored, trading resumed. Best-effort, never raises."""
        try:
            text = (
                "✅ <b>MT5 Reconnected</b>\n"
                f"Trading resumed."
            )
            self._send(text)
        except Exception as e:
            logger.warning(f"mt5_reconnected alert failed: {e}")

    # ------------------------------------------------------------------
    # Claude manager
    # ------------------------------------------------------------------

    def manager_cycle(self, summary: str):
        """
        Alert: one Claude-manager cycle finished (or was skipped/failed).
        `summary` is a pre-formatted one-line-ish string, e.g.
        "[MANAGER] cycle: 2 applied, 1 clamped — rationale...". Best-effort,
        never raises.
        """
        try:
            self._send(f"\U0001f9e0 {summary}")
        except Exception as e:
            logger.warning(f"manager_cycle alert failed: {e}")

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    def daily_summary(
        self,
        date: str,
        trades: int,
        wins: int,
        losses: int,
        pnl: float,
        balance: float,
        win_rate: float,
        max_drawdown: float,
        manager_cycles: Optional[int] = None,
        manager_adjustments: Optional[int] = None,
        api_cost_today: Optional[float] = None,
        net_after_cost_today: Optional[float] = None,
        net_after_cost_total: Optional[float] = None,
        verdict_line: Optional[str] = None,
    ):
        """
        Send daily summary report. The manager_* / verdict parameters are
        optional (Task 14 self-funding scorecard): when provided, a Manager
        section is appended — cycles run today, adjustments applied, API
        cost today, net-after-cost P&L today + cumulative, and (from day 8
        of the API budget window onward) the SELF-FUNDING / NOT JUSTIFIED
        verdict line.
        """
        emoji = "\u2705" if pnl >= 0 else "\U0001f534"

        text = (
            f"\U0001f4ca <b>Daily Summary — {date}</b>\n\n"
            f"Trades: {trades} ({wins}W / {losses}L)\n"
            f"Win Rate: {win_rate:.0%}\n"
            f"{emoji} PnL: R{pnl:+.2f}\n"
            f"Max Drawdown: {max_drawdown:.1%}\n"
            f"\U0001f4b0 Balance: R{balance:.2f}"
        )

        if manager_cycles is not None:
            text += (
                f"\n\n\U0001f9e0 <b>Manager</b>\n"
                f"Manager cycles: {manager_cycles} "
                f"({manager_adjustments or 0} adjustments applied)\n"
                f"API cost today: R{(api_cost_today or 0.0):.2f}"
            )
            if net_after_cost_today is not None:
                text += f"\nNet after cost today: R{net_after_cost_today:+.2f}"
            if net_after_cost_total is not None:
                text += f"\nNet after cost total: R{net_after_cost_total:+.2f}"
            if verdict_line:
                text += f"\nVerdict: {verdict_line}"

        self._send(text)
