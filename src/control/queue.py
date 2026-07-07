"""
Control queue — file-based command channel between the `tb` CLI (Task 11+)
and the running bot.

Writer side (the `tb` CLI, not implemented here): drops
`control/inbox/<id>.cmd.json.tmp`, then atomic-renames (os.replace) to
`control/inbox/<id>.cmd.json`. Command shape:

    {id, verb, args, reason, requested_at, requested_by}

Bot side (this module): `poll_once()` is called once per main-loop
iteration. It reads the inbox oldest-first, executes each command, writes
a result to `control/outbox/<id>.result.json`, then deletes the inbox
file. Because `Path.glob("*.cmd.json")` never matches a `*.cmd.json.tmp`
file, a command mid-write (still under its `.tmp` name) is never picked
up — the CLI's rename is what makes it visible.

Verbs:
    pause / resume    — reason mandatory (>=10 chars). Sets/clears a
                        manual-pause flag on RiskManager that blocks new
                        trade entries (independent of the daily-drawdown
                        block and circuit breaker).
    tune              — one dotted key + value, reason mandatory. Subject
                        to a whitelist + bounds check, the safety-floor
                        lock, the ml low<=high cross-check, and a 24h
                        rate limit for non-manager callers.
    revert            — re-applies the before-value of the last *applied*
                        tune from control_log.
    status_snapshot   — read-only; writes a best-effort status snapshot
                        to the outbox. No control_log row, no Telegram
                        message (per spec: Telegram is only for WRITE
                        commands).
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

from src.config import PROJECT_ROOT
from src.control.effective_config import EffectiveConfig, SETTINGS_PATH, _load_yaml

logger = logging.getLogger("traderbot.control.queue")

INBOX_DIR = PROJECT_ROOT / "control" / "inbox"
OUTBOX_DIR = PROJECT_ROOT / "control" / "outbox"
DEADLETTER_DIRNAME = "deadletter"

# Sanitization for command ids taken from inbox JSON. Used to build the
# outbox result path (`control/outbox/<id>.result.json`) and the
# control_log `cmd_id` column — an unsanitized id is a path-traversal
# vector (e.g. "../../etc/passwd") and a SQL-adjacent injection surface.
CMD_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Manual tune rate limit: non-manager callers get 1 applied tune per
# rolling window. `requested_by == "manager"` is exempt.
MANUAL_TUNE_RATE_LIMIT_HOURS = 24
MIN_REASON_LEN = 10

VALID_VERBS = {"pause", "resume", "tune", "revert", "status_snapshot"}

# Whitelist + bounds for the `tune` verb. `weight.<INSTRUMENT>` is handled
# separately (bounds are fixed, but the instrument must exist in
# config/instruments.yaml).
TUNE_BOUNDS = {
    "risk.risk_per_trade_pct": (0.5, 2.5),
    "ml.confidence_threshold_high": (0.50, 0.75),
    "ml.confidence_threshold_low": (0.45, 0.65),
}
WEIGHT_BOUNDS = (0.0, 1.5)


class ControlQueue:
    """
    File-based command queue processor. See module docstring for the
    inbox/outbox protocol.
    """

    def __init__(
        self,
        config,
        journal,
        telegram=None,
        risk_manager=None,
        effective_config: Optional[EffectiveConfig] = None,
        collector=None,
        executor=None,
        client=None,
        inbox_dir: Optional[Path] = None,
        outbox_dir: Optional[Path] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ):
        self.config = config
        self.journal = journal
        self.telegram = telegram
        self.risk_manager = risk_manager
        self.effective_config = effective_config or EffectiveConfig.load()
        self.collector = collector
        self.executor = executor
        self.client = client
        self.clock = clock or (lambda: datetime.now(timezone.utc))

        self.inbox_dir = inbox_dir or INBOX_DIR
        self.outbox_dir = outbox_dir or OUTBOX_DIR
        self.deadletter_dir = self.inbox_dir.parent / DEADLETTER_DIRNAME
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        self.outbox_dir.mkdir(parents=True, exist_ok=True)
        self.deadletter_dir.mkdir(parents=True, exist_ok=True)

        # Fallback manual-pause state used only when no risk_manager is
        # wired in (e.g. isolated unit tests of the queue itself).
        self._standalone_pause_reason: str = ""

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def poll_once(self) -> int:
        """
        Process every command currently sitting in the inbox, oldest
        first. Returns the number of commands processed. Safe to call
        every main-loop iteration — an empty inbox is a no-op.
        """
        commands = self._read_inbox_sorted()
        for path, cmd in commands:
            self._process(path, cmd)
        return len(commands)

    def _read_inbox_sorted(self) -> list[tuple[Path, dict]]:
        # Path.glob("*.cmd.json") never matches "*.cmd.json.tmp" — a
        # command file mid-write under its .tmp name is invisible here
        # until the writer's os.replace() makes it a real .cmd.json.
        entries = []
        for path in self.inbox_dir.glob("*.cmd.json"):
            try:
                cmd = json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                # Poison pill: a corrupt/truncated inbox file would warn
                # (and get skipped) on *every* poll forever if left in
                # place. Move it out of the inbox so it's picked up
                # exactly once, log a single WARNING, and best-effort
                # notify the CLI via the outbox using whatever id can be
                # salvaged from the filename.
                self._deadletter(path, e)
                continue
            entries.append((path, cmd))

        entries.sort(key=lambda pair: (pair[1].get("requested_at") or "", pair[0].name))
        return entries

    def _deadletter(self, path: Path, error: Exception):
        logger.warning(f"Control queue: unreadable inbox file {path.name}, moving to deadletter: {error}")

        salvaged_id = path.name.split(".cmd.json")[0]
        if CMD_ID_RE.match(salvaged_id):
            try:
                self._write_outbox(salvaged_id, {
                    "id": salvaged_id,
                    "outcome": "error",
                    "detail": f"corrupt inbox command file: {error}",
                    "applied_at": self.clock().isoformat(),
                })
            except Exception as e:
                logger.error(f"Control queue: failed to write best-effort outbox result for corrupt file {path.name}: {e}")

        dead_path = self.deadletter_dir / f"{path.name}.dead"
        try:
            os.replace(path, dead_path)
        except OSError as e:
            logger.error(f"Control queue: failed to move corrupt inbox file {path.name} to deadletter: {e}")

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _process(self, path: Path, cmd: dict):
        raw_id = cmd.get("id")
        fallback_id = path.name.split(".cmd.json")[0]
        candidate_id = raw_id if raw_id is not None else fallback_id
        verb = cmd.get("verb")
        args = cmd.get("args") or {}
        reason = cmd.get("reason") or ""
        requested_by = cmd.get("requested_by") or "unknown"

        # cmd_id sanitization: it's used verbatim to build the outbox
        # path (control/outbox/<id>.result.json) and the control_log
        # cmd_id column — an unsanitized id (e.g. "../../etc/passwd") is
        # a path-traversal vector. Reject loudly under a generated safe
        # id instead of trusting attacker/writer-supplied input.
        if not isinstance(candidate_id, str) or not CMD_ID_RE.match(candidate_id):
            self._reject_invalid_id(path, candidate_id, verb, args, reason, requested_by)
            return

        cmd_id = candidate_id

        # Crash-replay idempotency guard: if this cmd_id already reached
        # a terminal outcome in control_log, this inbox file is a replay
        # (e.g. re-dropped after a crash between the outbox write and
        # the inbox unlink) — skip re-executing the verb entirely.
        replay = self._replay_outcome(cmd_id)
        if replay is not None:
            outcome, detail = replay
            logger.warning(
                f"Control queue: replay detected for cmd_id={cmd_id!r} (already {outcome}); "
                f"skipping re-execution"
            )
            self._finalize(path, cmd_id, outcome, detail)
            return

        log_id = None
        before_json = None
        after_json = None

        try:
            if verb not in VALID_VERBS:
                outcome, detail = "rejected", f"Unknown verb: {verb!r}"
            elif verb == "status_snapshot":
                # Read-only: no control_log row, no Telegram message.
                outcome, detail = "applied", self._build_status_snapshot()
            else:
                log_id = self.journal.log_control_command(
                    verb=verb, args=args, reason=reason, requested_by=requested_by,
                    ts_utc=self.clock().isoformat(), cmd_id=cmd_id,
                )
                self._notify(f"[control] received: {verb} args={args} reason={reason!r}")

                if verb == "pause":
                    outcome, detail = self._handle_pause(reason)
                elif verb == "resume":
                    outcome, detail = self._handle_resume(reason)
                elif verb == "tune":
                    outcome, detail, before_json, after_json = self._handle_tune(
                        args, reason, requested_by
                    )
                elif verb == "revert":
                    outcome, detail, after_json = self._handle_revert(reason)
                else:  # pragma: no cover — VALID_VERBS keeps this unreachable
                    outcome, detail = "error", f"Unhandled verb: {verb}"
        except Exception as e:
            logger.error(f"Control queue: error processing {verb} ({cmd_id}): {e}", exc_info=True)
            outcome, detail = "error", str(e)

        if verb in VALID_VERBS and verb != "status_snapshot" and log_id is not None:
            try:
                self.journal.update_control_outcome(
                    log_id, outcome, before_config_json=before_json, after_config_json=after_json,
                )
            except Exception as e:
                logger.error(
                    f"Control queue: failed to update control_log outcome for {cmd_id} ({verb}): {e}",
                    exc_info=True,
                )
            self._notify(f"[control] {outcome}: {verb} ({cmd_id}) — {detail}")

        self._finalize(path, cmd_id, outcome, detail)

    def _finalize(self, path: Path, cmd_id: str, outcome: str, detail):
        """
        Post-dispatch: write the outbox result, then delete the inbox
        file last. If the outbox write fails, the inbox file is
        deliberately left in place so the next poll retries it — and,
        for a command that already reached control_log, the replay
        guard in `_process` will skip re-executing the verb and simply
        retry the outbox write, rather than double-applying it.
        """
        try:
            self._write_outbox(cmd_id, {
                "id": cmd_id,
                "outcome": outcome,
                "detail": detail,
                "applied_at": self.clock().isoformat(),
            })
        except Exception as e:
            logger.error(f"Control queue: failed to write outbox result for {cmd_id}: {e}", exc_info=True)
            return

        try:
            path.unlink()
        except OSError as e:
            logger.error(f"Control queue: failed to remove processed inbox file {path.name}: {e}", exc_info=True)

    def _replay_outcome(self, cmd_id: str) -> Optional[tuple[str, str]]:
        """
        Returns (outcome, detail) if `cmd_id` already has a terminal
        (applied/rejected) row in control_log, else None. A pending row
        (still mid-processing, shouldn't normally happen across polls
        since processing is synchronous) is not treated as a replay.
        """
        if self.journal is None:
            return None
        try:
            row = self.journal.get_control_log_by_cmd_id(cmd_id)
        except Exception as e:
            logger.warning(
                f"Control queue: replay lookup failed for {cmd_id}, proceeding with normal processing: {e}"
            )
            return None
        if row is None:
            return None
        outcome = row.get("outcome")
        if outcome not in ("applied", "rejected"):
            return None
        return outcome, f"replay: command already processed (outcome={outcome})"

    def _reject_invalid_id(self, path: Path, candidate_id, verb, args: dict, reason: str, requested_by: str):
        safe_id = f"invalid-{uuid.uuid4().hex[:12]}"
        logger.error(
            f"Control queue: rejected inbox file {path.name} — invalid command id {candidate_id!r}; "
            f"using generated id {safe_id} for the outbox/audit trail"
        )

        if self.journal is not None:
            try:
                log_id = self.journal.log_control_command(
                    verb=verb or "unknown", args=args, reason=reason, requested_by=requested_by,
                    ts_utc=self.clock().isoformat(), cmd_id=safe_id,
                )
                self.journal.update_control_outcome(log_id, "rejected")
            except Exception as e:
                logger.error(
                    f"Control queue: failed to log rejected invalid-id command from {path.name}: {e}",
                    exc_info=True,
                )

        self._finalize(path, safe_id, "rejected", f"invalid command id: {candidate_id!r}")

    def _notify(self, text: str):
        """Best-effort Telegram notification — never let it break command processing."""
        if self.telegram is None:
            return
        try:
            self.telegram._send(text)
        except Exception as e:
            logger.debug(f"Control queue: Telegram notify failed: {e}")

    def _write_outbox(self, cmd_id: str, result: dict):
        final_path = self.outbox_dir / f"{cmd_id}.result.json"
        tmp_path = self.outbox_dir / f"{cmd_id}.result.json.tmp"
        tmp_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        os.replace(tmp_path, final_path)

    # ------------------------------------------------------------------
    # pause / resume
    # ------------------------------------------------------------------

    def _handle_pause(self, reason: str) -> tuple[str, str]:
        if len(reason.strip()) < MIN_REASON_LEN:
            return "rejected", f"reason must be at least {MIN_REASON_LEN} characters"
        if self.risk_manager is not None:
            self.risk_manager.set_manual_pause(reason)
        else:
            self._standalone_pause_reason = reason
        return "applied", f"paused: {reason}"

    def _handle_resume(self, reason: str) -> tuple[str, str]:
        if len(reason.strip()) < MIN_REASON_LEN:
            return "rejected", f"reason must be at least {MIN_REASON_LEN} characters"
        if self.risk_manager is not None:
            self.risk_manager.clear_manual_pause()
        else:
            self._standalone_pause_reason = ""
        return "applied", f"resumed: {reason}"

    @property
    def manual_paused(self) -> bool:
        """Best-effort read of the manual-pause flag (delegates to RiskManager when wired)."""
        if self.risk_manager is not None:
            return bool(getattr(self.risk_manager, "manual_paused", False))
        return bool(self._standalone_pause_reason)

    # ------------------------------------------------------------------
    # tune
    # ------------------------------------------------------------------

    def _instrument_names(self) -> set[str]:
        try:
            instruments = getattr(self.config, "instruments", {}) or {}
            return set(instruments.get("instruments", {}).keys())
        except Exception:
            return set()

    def _bounds_for_key(self, key: str) -> Optional[tuple[float, float]]:
        if key in TUNE_BOUNDS:
            return TUNE_BOUNDS[key]
        if key.startswith("weight."):
            instrument = key.split(".", 1)[1]
            if instrument in self._instrument_names():
                return WEIGHT_BOUNDS
        return None

    def _manual_tune_used_in_window(self) -> bool:
        """
        True if a non-manager `tune` has been applied within the rolling
        rate-limit window (queried from control_log via the journal).
        """
        if self.journal is None:
            return False
        try:
            df = self.journal.get_control_log(verb="tune", outcome="applied")
        except Exception as e:
            logger.warning(f"Control queue: rate-limit lookup failed, allowing tune: {e}")
            return False
        if df is None or df.empty:
            return False

        df = df[df["requested_by"] != "manager"]
        if df.empty:
            return False

        cutoff = (self.clock() - timedelta(hours=MANUAL_TUNE_RATE_LIMIT_HOURS)).isoformat()
        recent = df[df["ts_utc"] >= cutoff]
        return not recent.empty

    def _handle_tune(
        self, args: dict, reason: str, requested_by: str
    ) -> tuple[str, str, Optional[str], Optional[str]]:
        if len(reason.strip()) < MIN_REASON_LEN:
            return "rejected", f"reason must be at least {MIN_REASON_LEN} characters", None, None

        key = args.get("key")
        raw_value = args.get("value")
        if not key or raw_value is None:
            return "rejected", "tune requires args.key and args.value", None, None

        bounds = self._bounds_for_key(key)
        if bounds is None:
            return "rejected", f"key not whitelisted for tuning: {key}", None, None

        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return "rejected", f"value must be numeric: {raw_value!r}", None, None

        lo, hi = bounds
        if not (lo <= value <= hi):
            return "rejected", f"value {value} out of bounds [{lo}, {hi}] for {key}", None, None

        if self.effective_config.is_safety_locked(key):
            return "rejected", f"key is safety-locked: {key}", None, None

        # ml low<=high cross-check: compare against the OTHER key's
        # current effective value when only one side is being tuned.
        if key == "ml.confidence_threshold_low":
            current_high = self.effective_config.get("ml.confidence_threshold_high")
            if current_high is not None and value > current_high:
                return (
                    "rejected",
                    f"low ({value}) must be <= high ({current_high})",
                    None, None,
                )
        elif key == "ml.confidence_threshold_high":
            current_low = self.effective_config.get("ml.confidence_threshold_low")
            if current_low is not None and current_low > value:
                return (
                    "rejected",
                    f"low ({current_low}) must be <= high ({value})",
                    None, None,
                )

        if requested_by != "manager" and self._manual_tune_used_in_window():
            return (
                "rejected",
                f"rate limit: only 1 manual tune per {MANUAL_TUNE_RATE_LIMIT_HOURS}h",
                None, None,
            )

        before_value = self.effective_config.get(key)
        before_json = json.dumps({key: before_value})

        self.effective_config.apply_tune(key, value)

        after_json = json.dumps({key: value})
        return "applied", f"tuned {key}: {before_value} -> {value}", before_json, after_json

    # ------------------------------------------------------------------
    # revert
    # ------------------------------------------------------------------

    def _handle_revert(self, reason: str) -> tuple[str, str, Optional[str]]:
        if len(reason.strip()) < MIN_REASON_LEN:
            return "rejected", f"reason must be at least {MIN_REASON_LEN} characters", None

        if self.journal is None:
            return "rejected", "no journal wired in — cannot look up prior tune", None

        try:
            df = self.journal.get_control_log(verb="tune", outcome="applied")
        except Exception as e:
            return "error", f"failed to query control log: {e}", None

        if df is None or df.empty:
            return "rejected", "no prior applied tune to revert", None

        if "id" in df.columns:
            df = df.sort_values("id", ascending=False)
        last_row = df.iloc[0]

        before_raw = last_row.get("before_config_json")
        if not before_raw:
            return "rejected", "prior tune has no recorded before-value", None

        try:
            before = json.loads(before_raw)
        except Exception as e:
            return "error", f"corrupt before_config_json on prior tune: {e}", None

        if not before:
            return "rejected", "prior tune has no recorded before-value", None

        for key, value in before.items():
            self.effective_config.apply_tune(key, value)

        after_json = json.dumps(before)
        return "applied", f"reverted to {before}", after_json

    # ------------------------------------------------------------------
    # status_snapshot
    # ------------------------------------------------------------------

    def _build_status_snapshot(self) -> dict:
        """
        Best-effort status snapshot. Any field we can't determine from the
        modules wired in is left as None (documented — never invented).
        """
        snapshot: dict = {
            "bot_up": True,
            "broker_connected": None,
            "open_positions": None,
            "todays_pnl": None,
            "drawdown_vs_cap": None,
            "entries_blocked": None,
            "manual_paused": None,
            "manual_pause_reason": None,
            "balance": None,
            "equity": None,
            "floor": None,
            "effective_config_delta": {},
        }

        try:
            if self.collector is not None:
                snapshot["broker_connected"] = not bool(getattr(self.collector, "broker_down", False))
        except Exception as e:
            logger.debug(f"status_snapshot: broker_connected unavailable: {e}")

        try:
            if self.risk_manager is not None:
                snapshot["entries_blocked"] = self.risk_manager.entries_blocked
                snapshot["manual_paused"] = self.risk_manager.manual_paused
                snapshot["manual_pause_reason"] = self.risk_manager.manual_pause_reason
                snapshot["open_positions"] = self.risk_manager.open_position_count
        except Exception as e:
            logger.debug(f"status_snapshot: risk_manager fields unavailable: {e}")

        try:
            if self.risk_manager is not None:
                floor = getattr(self.risk_manager, "ratchet_floor", None)
                if floor is not None and hasattr(floor, "current_floor"):
                    # current_floor is a @property — do not call it.
                    snapshot["floor"] = float(floor.current_floor)
        except Exception as e:
            logger.debug(f"status_snapshot: floor unavailable: {e}")

        client = self.client or getattr(self.executor, "client", None)
        try:
            if client is not None:
                summary = client.get_account_summary()
                snapshot["balance"] = summary.get("balance")
                snapshot["equity"] = summary.get("equity")
        except Exception as e:
            logger.debug(f"status_snapshot: balance/equity unavailable: {e}")

        try:
            if self.risk_manager is not None and snapshot["balance"] is not None:
                drawdown_tracker = getattr(self.risk_manager, "drawdown", None)
                if drawdown_tracker is not None:
                    daily_dd_pct = drawdown_tracker.get_daily_drawdown_pct(snapshot["balance"])
                    cap_pct = getattr(drawdown_tracker, "daily_limit", None)
                    if cap_pct:
                        snapshot["drawdown_vs_cap"] = round(daily_dd_pct / cap_pct, 4)
        except Exception as e:
            logger.debug(f"status_snapshot: drawdown_vs_cap unavailable: {e}")

        try:
            if self.journal is not None:
                today = self.clock().date().isoformat()
                trades = self.journal.get_trades(since=today, limit=1000)
                if trades is not None and not trades.empty and "net_pnl_zar" in trades.columns:
                    snapshot["todays_pnl"] = float(trades["net_pnl_zar"].fillna(0).sum())
        except Exception as e:
            logger.debug(f"status_snapshot: todays_pnl unavailable: {e}")

        try:
            baseline = _load_yaml(SETTINGS_PATH)
            delta = {}
            keys = list(TUNE_BOUNDS.keys())
            for instrument in self._instrument_names():
                keys.append(f"weight.{instrument}")
            for key in keys:
                node: object = baseline
                for part in key.split("."):
                    if isinstance(node, dict) and part in node:
                        node = node[part]
                    else:
                        node = None
                        break
                effective = self.effective_config.get(key)
                if node != effective:
                    delta[key] = {"baseline": node, "effective": effective}
            snapshot["effective_config_delta"] = delta
        except Exception as e:
            logger.debug(f"status_snapshot: effective_config_delta unavailable: {e}")

        return snapshot
