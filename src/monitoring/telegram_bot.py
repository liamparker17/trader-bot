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
    # Reports
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # AI Analyst alerts
    # ------------------------------------------------------------------

    def claude_recommendation(
        self,
        trade_id: str,
        instrument: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        confidence: float,
        reasoning: str,
        ttl_seconds: int,
    ):
        """Alert: Claude recommends a trade, awaiting approval."""
        emoji = "\U0001f7e2" if direction == "buy" else "\U0001f534"
        arrow = "\u2191" if direction == "buy" else "\u2193"

        text = (
            f"\U0001f9e0 <b>CLAUDE RECOMMENDS</b>\n\n"
            f"{emoji} <b>{direction.upper()}</b> {instrument}\n"
            f"{arrow} Entry: {entry_price:.5f}\n"
            f"\U0001f6d1 SL: {stop_loss:.5f}\n"
            f"\U0001f3af TP: {take_profit:.5f}\n"
            f"\U0001f4ca Confidence: {confidence:.0%}\n\n"
            f"<i>{reasoning}</i>\n\n"
            f"\u23f0 Expires in {ttl_seconds // 60} min\n\n"
            f"Reply:\n"
            f"<code>/approve {trade_id}</code>\n"
            f"<code>/reject {trade_id}</code>"
        )
        self._send(text)

    def claude_shadow_result(
        self,
        instrument: str,
        direction: str,
        pnl_pips: float,
        confidence: float,
        exit_reason: str,
    ):
        """Alert: shadow trade result (paper P&L)."""
        if pnl_pips > 0:
            emoji = "\u2705"
            label = "WIN"
        else:
            emoji = "\u274c"
            label = "LOSS"

        text = (
            f"\U0001f9e0 <b>SHADOW {label}</b> {instrument} {direction.upper()}\n"
            f"{emoji} {pnl_pips:+.1f} pips (paper)\n"
            f"Confidence: {confidence:.0%} | Exit: {exit_reason}"
        )
        self._send(text)

    # ------------------------------------------------------------------
    # Command polling (for /approve, /reject, /shadow, /pending)
    # ------------------------------------------------------------------

    def start_command_listener(self, approval_queue, shadow_trader=None):
        """
        Start a background thread that polls for Telegram commands.

        Supports:
            /approve <id>  - Approve a pending Claude recommendation
            /reject <id>   - Reject a pending recommendation
            /pending       - List pending recommendations
            /shadow        - Show shadow trader performance
            /ai            - Show AI analyst stats
        """
        if not self.enabled:
            return

        self._approval_queue = approval_queue
        self._shadow_trader = shadow_trader
        self._last_update_id = 0

        thread = threading.Thread(
            target=self._command_poll_loop,
            daemon=True,
            name="telegram_commands",
        )
        thread.start()
        logger.info("Telegram command listener started.")

    def _command_poll_loop(self):
        """Poll Telegram for commands every 2 seconds."""
        import time
        while True:
            try:
                self._poll_updates()
            except Exception as e:
                logger.debug(f"Telegram poll error: {e}")
            time.sleep(2)

    def _poll_updates(self):
        """Fetch and process new Telegram messages."""
        url = f"https://api.telegram.org/bot{self.token}/getUpdates"
        params = {"offset": self._last_update_id + 1, "timeout": 1}

        try:
            resp = requests.get(url, params=params, timeout=5)
            if resp.status_code != 200:
                return
            data = resp.json()
            if not data.get("ok"):
                return
        except Exception:
            return

        for update in data.get("result", []):
            self._last_update_id = update["update_id"]
            message = update.get("message", {})
            text = message.get("text", "").strip()
            chat_id = message.get("chat", {}).get("id")

            # Only respond to our configured chat
            if str(chat_id) != str(self.chat_id):
                continue

            if text.startswith("/approve "):
                trade_id = text.split(" ", 1)[1].strip()
                self._handle_approve(trade_id)
            elif text.startswith("/reject "):
                trade_id = text.split(" ", 1)[1].strip()
                self._handle_reject(trade_id)
            elif text == "/pending":
                self._handle_pending()
            elif text == "/shadow":
                self._handle_shadow()

    def _handle_approve(self, trade_id: str):
        """Process /approve command."""
        if not hasattr(self, "_approval_queue") or not self._approval_queue:
            self._send("Approval queue not available.")
            return

        trade = self._approval_queue.approve(trade_id)
        if trade:
            self._send(
                f"\u2705 <b>APPROVED:</b> {trade.instrument} {trade.direction.upper()}\n"
                f"Will execute on next candle if price is still valid."
            )
        else:
            self._send(f"\u274c Trade <code>{trade_id}</code> not found or expired.")

    def _handle_reject(self, trade_id: str):
        """Process /reject command."""
        if not hasattr(self, "_approval_queue") or not self._approval_queue:
            self._send("Approval queue not available.")
            return

        if self._approval_queue.reject(trade_id):
            self._send(f"\U0001f6ab <b>REJECTED:</b> <code>{trade_id}</code>")
        else:
            self._send(f"\u274c Trade <code>{trade_id}</code> not found.")

    def _handle_pending(self):
        """Process /pending command."""
        if not hasattr(self, "_approval_queue") or not self._approval_queue:
            self._send("Approval queue not available.")
            return

        pending = self._approval_queue.get_pending_summary()
        if not pending:
            self._send("No pending recommendations.")
            return

        lines = ["\U0001f9e0 <b>Pending Recommendations:</b>\n"]
        for p in pending:
            remaining = int(p["time_remaining"])
            lines.append(
                f"\u2022 <code>{p['id']}</code>\n"
                f"  {p['instrument']} {p['direction'].upper()} "
                f"@ {p['entry_price']:.5f}\n"
                f"  Conf: {p['confidence']:.0%} | "
                f"Expires: {remaining // 60}m {remaining % 60}s"
            )
        self._send("\n".join(lines))

    def _handle_shadow(self):
        """Process /shadow command."""
        if not hasattr(self, "_shadow_trader") or not self._shadow_trader:
            self._send("Shadow trader not available.")
            return

        perf = self._shadow_trader.get_performance()
        text = (
            f"\U0001f9e0 <b>Claude Shadow Trader</b>\n\n"
            f"Total Trades: {perf['total_trades']}\n"
            f"Win Rate: {perf['win_rate']:.0%} "
            f"({perf['wins']}W / {perf['losses']}L)\n"
            f"Total PnL: {perf['total_pnl_pips']:+.1f} pips\n"
            f"Avg PnL: {perf['avg_pnl_pips']:+.1f} pips\n"
            f"Best: {perf['best_trade_pips']:+.1f} pips\n"
            f"Worst: {perf['worst_trade_pips']:+.1f} pips\n"
            f"Open Paper Trades: {perf['open_trades']}"
        )

        # By instrument breakdown
        if perf["by_instrument"]:
            text += "\n\n<b>By Instrument:</b>"
            for inst, stats in perf["by_instrument"].items():
                text += (
                    f"\n{inst}: {stats['trades']} trades, "
                    f"{stats['win_rate']:.0%} WR, "
                    f"{stats['pnl_pips']:+.1f} pips"
                )

        self._send(text)

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
    ):
        """Send daily summary report."""
        emoji = "\u2705" if pnl >= 0 else "\U0001f534"

        text = (
            f"\U0001f4ca <b>Daily Summary — {date}</b>\n\n"
            f"Trades: {trades} ({wins}W / {losses}L)\n"
            f"Win Rate: {win_rate:.0%}\n"
            f"{emoji} PnL: R{pnl:+.2f}\n"
            f"Max Drawdown: {max_drawdown:.1%}\n"
            f"\U0001f4b0 Balance: R{balance:.2f}"
        )
        self._send(text)
