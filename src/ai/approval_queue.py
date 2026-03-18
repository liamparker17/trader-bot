"""
Approval Queue — Manual approval of Claude's trade recommendations.

When Claude recommends a trade (via shadow trader), it gets added to
an approval queue. The user can approve/reject via Telegram commands
or the Streamlit dashboard. The bot NEVER blocks waiting for approval.

Flow:
    Claude recommends -> Queue -> Telegram notification
    User replies /approve <id> or /reject <id>
    If approved within TTL -> Bot executes the trade
    If TTL expires -> Trade is discarded (no action)
"""

import json
import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass

from src.config import Config, PROJECT_ROOT

logger = logging.getLogger("traderbot.ai.approval")


@dataclass
class PendingTrade:
    """A Claude recommendation awaiting user approval."""
    id: str
    instrument: str
    direction: str
    entry_price: float  # Price at time of recommendation
    stop_loss: float
    take_profit: float
    confidence: float
    reasoning: str
    created_at: datetime
    ttl_seconds: int  # How long the recommendation is valid
    status: str = "pending"  # pending | approved | rejected | expired


class ApprovalQueue:
    """
    Non-blocking approval queue for Claude's trade recommendations.

    The bot continues operating normally. Approved trades are picked up
    on the next candle cycle and executed if still valid.
    """

    def __init__(self, config: Config):
        self.config = config
        self._pending: dict[str, PendingTrade] = {}
        self._approved: dict[str, PendingTrade] = {}
        self._lock = threading.Lock()
        self._counter = 0

        # TTL: how long a recommendation stays valid (default 5 minutes)
        self._default_ttl = config.get("ai_analyst.approval_ttl_seconds", 300)

        # Max price slippage allowed when executing an approved trade
        # (price may have moved since recommendation)
        self._max_slippage_pips = config.get("ai_analyst.approval_max_slippage_pips", 5)

        # SQLite for persistence
        db_path = config.get("monitoring.trade_journal_db", "data/trade_logs/trades.db")
        self.db_path = PROJECT_ROOT / db_path
        self._init_db()
        self._load_pending()

    def _init_db(self):
        """Create the approval queue table."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS approval_queue (
                    id TEXT PRIMARY KEY,
                    instrument TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    confidence REAL NOT NULL,
                    reasoning TEXT,
                    created_at TEXT NOT NULL,
                    ttl_seconds INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    decided_at TEXT,
                    executed INTEGER DEFAULT 0
                )
            """)

    def _load_pending(self):
        """Restore pending items from DB on startup."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM approval_queue WHERE status = 'pending'"
            ).fetchall()
            if not rows:
                return
            cols = [d[0] for d in conn.execute(
                "SELECT * FROM approval_queue LIMIT 0"
            ).description]

        for row in rows:
            r = dict(zip(cols, row))
            trade = PendingTrade(
                id=r["id"],
                instrument=r["instrument"],
                direction=r["direction"],
                entry_price=r["entry_price"],
                stop_loss=r["stop_loss"],
                take_profit=r["take_profit"],
                confidence=r["confidence"],
                reasoning=r["reasoning"] or "",
                created_at=datetime.fromisoformat(r["created_at"]),
                ttl_seconds=r["ttl_seconds"],
                status="pending",
            )
            self._pending[trade.id] = trade

        if self._pending:
            logger.info(f"Approval queue: restored {len(self._pending)} pending recommendations.")

    def add(
        self,
        instrument: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        confidence: float,
        reasoning: str,
        ttl_seconds: int | None = None,
    ) -> PendingTrade:
        """
        Add a new Claude recommendation to the queue.

        Returns the PendingTrade (caller should send Telegram notification).
        """
        with self._lock:
            self._counter += 1
            trade_id = f"rec_{instrument}_{self._counter}"

            trade = PendingTrade(
                id=trade_id,
                instrument=instrument,
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                reasoning=reasoning,
                created_at=datetime.now(timezone.utc),
                ttl_seconds=ttl_seconds or self._default_ttl,
            )

            self._pending[trade_id] = trade
            self._save(trade)

        logger.info(
            f"Queued for approval: {trade_id} | {instrument} {direction.upper()} "
            f"@ {entry_price:.5f} | Conf: {confidence:.0%}"
        )
        return trade

    def approve(self, trade_id: str) -> PendingTrade | None:
        """
        User approves a recommendation. Returns the trade if still valid.
        Called from Telegram handler or dashboard.
        """
        with self._lock:
            trade = self._pending.pop(trade_id, None)
            if trade is None:
                logger.warning(f"Approval: {trade_id} not found in pending queue.")
                return None

            # Check if expired
            age = (datetime.now(timezone.utc) - trade.created_at).total_seconds()
            if age > trade.ttl_seconds:
                trade.status = "expired"
                self._save(trade)
                logger.info(f"Approval: {trade_id} expired ({age:.0f}s > {trade.ttl_seconds}s TTL)")
                return None

            trade.status = "approved"
            self._approved[trade_id] = trade
            self._save(trade)

        logger.info(f"APPROVED: {trade_id} | {trade.instrument} {trade.direction.upper()}")
        return trade

    def reject(self, trade_id: str) -> bool:
        """User rejects a recommendation."""
        with self._lock:
            trade = self._pending.pop(trade_id, None)
            if trade is None:
                return False

            trade.status = "rejected"
            self._save(trade)

        logger.info(f"REJECTED: {trade_id}")
        return True

    def get_approved_trades(self) -> list[PendingTrade]:
        """
        Get all approved trades that haven't been executed yet.
        Called by the main loop to pick up user-approved trades.
        Non-blocking — returns empty list if nothing approved.
        """
        with self._lock:
            ready = []
            expired = []
            for trade_id, trade in self._approved.items():
                age = (datetime.now(timezone.utc) - trade.created_at).total_seconds()
                if age > trade.ttl_seconds:
                    trade.status = "expired"
                    self._save(trade)
                    expired.append(trade_id)
                else:
                    ready.append(trade)

            for tid in expired:
                del self._approved[tid]

            return ready

    def mark_executed(self, trade_id: str):
        """Mark an approved trade as executed."""
        with self._lock:
            trade = self._approved.pop(trade_id, None)
            if trade:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        "UPDATE approval_queue SET executed = 1, decided_at = ? WHERE id = ?",
                        (datetime.now(timezone.utc).isoformat(), trade_id),
                    )

    def expire_old(self):
        """Expire any pending items past their TTL."""
        now = datetime.now(timezone.utc)
        with self._lock:
            expired = []
            for trade_id, trade in self._pending.items():
                age = (now - trade.created_at).total_seconds()
                if age > trade.ttl_seconds:
                    trade.status = "expired"
                    self._save(trade)
                    expired.append(trade_id)

            for tid in expired:
                del self._pending[tid]

            if expired:
                logger.info(f"Expired {len(expired)} pending recommendations.")

    def get_pending_summary(self) -> list[dict]:
        """Get summary of all pending recommendations (for Telegram/dashboard)."""
        now = datetime.now(timezone.utc)
        return [
            {
                "id": t.id,
                "instrument": t.instrument,
                "direction": t.direction,
                "entry_price": t.entry_price,
                "stop_loss": t.stop_loss,
                "take_profit": t.take_profit,
                "confidence": t.confidence,
                "reasoning": t.reasoning[:100],
                "age_seconds": (now - t.created_at).total_seconds(),
                "ttl_seconds": t.ttl_seconds,
                "time_remaining": max(0, t.ttl_seconds - (now - t.created_at).total_seconds()),
            }
            for t in self._pending.values()
        ]

    def get_history(self, limit: int = 50) -> list[dict]:
        """Get recent approval queue history from DB."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM approval_queue ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            cols = [d[0] for d in conn.execute(
                "SELECT * FROM approval_queue LIMIT 0"
            ).description]
        return [dict(zip(cols, r)) for r in rows]

    def _save(self, trade: PendingTrade):
        """Persist to SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO approval_queue
                (id, instrument, direction, entry_price, stop_loss, take_profit,
                 confidence, reasoning, created_at, ttl_seconds, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.id, trade.instrument, trade.direction,
                trade.entry_price, trade.stop_loss, trade.take_profit,
                trade.confidence, trade.reasoning,
                trade.created_at.isoformat(),
                trade.ttl_seconds, trade.status,
            ))
