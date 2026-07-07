"""
Manager runner — the main loop: wait for the scheduler's next trigger,
build a briefing, call the model, validate/clamp, enqueue surviving
proposals onto the control queue, log the cycle, and alert Telegram.

Runs as `python -m src.manager` (see `src/manager/__main__.py`), as a
separate OS process from the trading bot (`python -m src.main`). Because
of that, this module never talks to MT5 directly for balance/equity —
it reuses the bot's existing cross-process status channel: a
`status_snapshot` control-queue round trip (same mechanism the `tb`
CLI's `status` command uses, see `cli/tb.py::cmd_status`). If the bot
isn't running (no result within the timeout), it falls back to the last
known balance from the trade journal, then the account's configured
starting balance.

All exceptions are caught inside the loop body — a failed cycle logs an
`error` outcome and Telegram alert, but the loop itself never dies.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from src.config import PROJECT_ROOT, load_config
from src.control.effective_config import EffectiveConfig
from src.control.queue import CMD_ID_RE, INBOX_DIR, OUTBOX_DIR
from src.manager import briefing as briefing_module
from src.manager import policy
from src.manager.client import ManagerAPIUnavailable, ManagerClient
from src.manager.scheduler import ManagerScheduler
from src.monitoring.telegram_bot import TelegramBot
from src.monitoring.trade_journal import TradeJournal
from src.risk.ratchet_floor import RatchetFloor

logger = logging.getLogger("traderbot.manager.runner")

DEFAULT_STATUS_TIMEOUT_SECS = 3.0
DEFAULT_POLL_INTERVAL_SECS = 30.0


def _default_balance_equity_fn(
    journal: TradeJournal,
    config,
    inbox_dir: Path,
    outbox_dir: Path,
    timeout: float = DEFAULT_STATUS_TIMEOUT_SECS,
) -> tuple[float, float]:
    """
    Cross-process balance/equity read. Tries a `status_snapshot`
    control-queue round trip first (works only while the bot process is
    running and draining its inbox); degrades to the trade journal's last
    known balance, then the configured starting balance.
    """
    try:
        from cli.tb import enqueue_command  # local import: cli/ isn't a package dependency of src/

        result = enqueue_command(
            "status_snapshot", {}, reason="", requested_by="manager",
            inbox_dir=inbox_dir, outbox_dir=outbox_dir, timeout=timeout,
        )
        if result is not None and result.get("outcome") == "applied" and isinstance(result.get("detail"), dict):
            detail = result["detail"]
            balance = detail.get("balance")
            if balance is not None:
                equity = detail.get("equity")
                return float(balance), float(equity if equity is not None else balance)
    except Exception as e:
        logger.warning(f"manager runner: status_snapshot round trip failed: {e}")

    # Fall back to the journal's last known balance_after.
    try:
        import sqlite3
        with sqlite3.connect(journal.db_path) as conn:
            row = conn.execute(
                "SELECT balance_after FROM trades WHERE balance_after IS NOT NULL "
                "ORDER BY entry_time DESC LIMIT 1"
            ).fetchone()
        if row and row[0] is not None:
            balance = float(row[0])
            return balance, balance
    except Exception as e:
        logger.warning(f"manager runner: journal-derived balance fallback failed: {e}")

    starting_balance = float(config.get("account.starting_balance_zar", 1000.0))
    return starting_balance, starting_balance


def _enqueue_tune(
    inbox_dir: Path,
    key: str,
    value: float,
    reason: str,
) -> None:
    """
    Drop a `tune` command onto the control-queue inbox (tmp-write +
    os.replace, same writer contract as `cli/tb.py::enqueue_command`).
    Fire-and-forget: the manager doesn't block waiting for the bot to
    process it (the bot may not even be running right now).
    """
    inbox_dir.mkdir(parents=True, exist_ok=True)
    cmd_id = uuid.uuid4().hex
    if not CMD_ID_RE.match(cmd_id):  # pragma: no cover — defensive; uuid4 hex always matches
        raise ValueError(f"generated an invalid command id: {cmd_id!r}")

    payload = {
        "id": cmd_id,
        "verb": "tune",
        "args": {"key": key, "value": value},
        "reason": reason,
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "requested_by": "manager",
    }
    tmp_path = inbox_dir / f"{cmd_id}.cmd.json.tmp"
    final_path = inbox_dir / f"{cmd_id}.cmd.json"
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, final_path)


class ManagerRunner:
    def __init__(
        self,
        config=None,
        journal: Optional[TradeJournal] = None,
        client=None,
        scheduler=None,
        telegram=None,
        inbox_dir: Path = INBOX_DIR,
        outbox_dir: Path = OUTBOX_DIR,
        effective_config: Optional[EffectiveConfig] = None,
        ratchet_floor: Optional[RatchetFloor] = None,
        balance_equity_fn: Optional[Callable[[], tuple[float, float]]] = None,
    ):
        self.config = config or load_config()
        self.journal = journal or TradeJournal(self.config)
        self.client = client if client is not None else ManagerClient(self.config)
        self.scheduler = scheduler or ManagerScheduler(self.config, self.journal, telegram)
        self.telegram = telegram if telegram is not None else TelegramBot(self.config)
        self.inbox_dir = inbox_dir
        self.outbox_dir = outbox_dir

        # Injected for tests; None means "compute fresh each cycle" in
        # production (tunes overlay / balance can change cycle to cycle).
        self._injected_effective_config = effective_config
        self._injected_ratchet_floor = ratchet_floor

        if balance_equity_fn is not None:
            self._balance_equity_fn = balance_equity_fn
        else:
            self._balance_equity_fn = lambda: _default_balance_equity_fn(
                self.journal, self.config, self.inbox_dir, self.outbox_dir,
            )

    # ------------------------------------------------------------------
    # One cycle
    # ------------------------------------------------------------------

    def run_once(self) -> None:
        """Check the scheduler once; run (or skip) one manager cycle. Never raises."""
        try:
            fire, trigger, reason = self.scheduler.should_fire()
        except Exception:
            logger.error("manager runner: scheduler.should_fire() failed", exc_info=True)
            return

        if not fire:
            return

        now = self.scheduler.now_fn()

        try:
            ok, budget_reason = self.scheduler.check_budget(now)
        except Exception:
            logger.error("manager runner: scheduler.check_budget() failed", exc_info=True)
            ok, budget_reason = True, None

        try:
            if not ok:
                self._log_and_alert(
                    trigger=trigger, outcome="budget_exhausted",
                    rationale=f"skipped: {budget_reason}", now=now,
                )
                self.scheduler.warn_budget_exhausted_once(now)
                return

            self._run_cycle(trigger, now)
        except Exception as e:
            logger.error(f"manager runner: cycle failed unexpectedly: {e}", exc_info=True)
            try:
                self._log_and_alert(
                    trigger=trigger, outcome="error", rationale=str(e), now=now,
                )
            except Exception:  # pragma: no cover — logging/alerting must not crash the loop
                logger.error("manager runner: failed to log/alert the cycle error", exc_info=True)
        finally:
            try:
                self.scheduler.record_cycle(now)
            except Exception:  # pragma: no cover — defensive
                logger.error("manager runner: scheduler.record_cycle() failed", exc_info=True)

    def _run_cycle(self, trigger: str, now: datetime) -> None:
        effective_config = self._injected_effective_config or EffectiveConfig.load()
        ratchet_floor = self._injected_ratchet_floor or RatchetFloor(
            min_floor_zar=self.config.get("risk.min_floor_zar", 600),
            max_total_drawdown_pct=self.config.get("risk.max_total_drawdown_pct", 0.35),
        )
        balance, equity = self._balance_equity_fn()

        briefing = briefing_module.build(self.journal, effective_config, ratchet_floor, balance, equity)

        try:
            proposals, rationale, usage = self.client.call(briefing)
        except ManagerAPIUnavailable as e:
            # A billed-but-unusable response (e.g. no tool_use block)
            # carries its usage on the exception — record the cost so the
            # budget governor still sees it.
            self._log_and_alert(
                trigger=trigger, outcome="api_unavailable", briefing=briefing,
                rationale=f"API unavailable: {e}", now=now,
                usage=getattr(e, "usage", None),
            )
            return

        # From here on the API cost has been incurred. Any failure below
        # must still write a manager_log row carrying that cost — the only
        # accepted loss window is a process crash before the log write.
        try:
            milestones = effective_config.get("growth.milestones", [1500, 2000, 3000, 4500, 6000])
            ceiling = policy.risk_ceiling_now(balance, milestones)
            applied, rejected = policy.validate_and_clamp(proposals, effective_config, ceiling)

            reason_by_key = {p.get("key"): p.get("reason", "") for p in proposals if isinstance(p, dict)}
            for entry in applied:
                _enqueue_tune(
                    self.inbox_dir,
                    key=entry["key"],
                    value=entry["value"],
                    reason=reason_by_key.get(entry["key"], entry.get("reason", "")),
                )

            outcome = "applied" if applied else "no_op"
            model_name = getattr(self.client, "model", self.config.get("manager.model", ""))
            self.journal.log_manager_cycle(
                trigger=trigger,
                briefing=briefing,
                model=model_name,
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                cost_zar=usage.get("cost_zar"),
                rationale=rationale,
                proposals=proposals,
                applied=applied,
                rejected=rejected,
                outcome=outcome,
                ts_utc=now.isoformat(),
            )
        except Exception as e:
            logger.error(
                f"manager runner: post-API failure (cost_zar={usage.get('cost_zar')}): {e}",
                exc_info=True,
            )
            self._log_and_alert(
                trigger=trigger, outcome="error", briefing=briefing,
                rationale=f"post-API failure: {e}", now=now, usage=usage,
            )
            return

        clamped_count = sum(1 for a in applied if a.get("clamped"))
        summary = (
            f"[MANAGER] cycle ({trigger}): {len(applied)} applied "
            f"({clamped_count} clamped), {len(rejected)} rejected — {rationale}"
        )
        self._safe_telegram(summary)

    def _log_and_alert(
        self,
        trigger: str,
        outcome: str,
        rationale: str,
        now: datetime,
        briefing: Optional[dict] = None,
        usage: Optional[dict] = None,
    ) -> None:
        usage = usage or {}
        try:
            self.journal.log_manager_cycle(
                trigger=trigger, briefing=briefing, rationale=rationale,
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                cost_zar=usage.get("cost_zar"),
                outcome=outcome, ts_utc=now.isoformat(),
            )
        except Exception:
            # Last-resort: the cost figure must at least land in the logs
            # so budget accounting can be reconciled by hand.
            logger.critical(
                f"manager runner: FAILED to write manager_log row "
                f"(outcome={outcome}, cost_zar={usage.get('cost_zar')}) — "
                f"API cost may be missing from budget accounting",
                exc_info=True,
            )
        self._safe_telegram(f"[MANAGER] cycle ({trigger}): {outcome} — {rationale}")

    def _safe_telegram(self, summary: str) -> None:
        if self.telegram is None:
            return
        try:
            self.telegram.manager_cycle(summary)
        except Exception:  # pragma: no cover — defensive
            logger.warning("manager runner: telegram alert failed", exc_info=True)

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------

    def run_forever(
        self,
        sleep_fn: Callable[[float], None] = time.sleep,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECS,
    ) -> None:
        """
        `python -m src.manager` entry point. Polls the scheduler every
        `poll_interval_seconds`, running (or skipping) a cycle each time.
        The loop body (`run_once`) never raises, so only a deliberate
        outer interrupt (Ctrl-C, or a test's escape hatch) ever stops it.
        """
        while True:
            self.run_once()
            sleep_fn(poll_interval_seconds)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    runner = ManagerRunner()
    runner.run_forever()


if __name__ == "__main__":
    main()
