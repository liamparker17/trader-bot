"""
Manager scheduler — decides WHEN a manager cycle should fire.

Two trigger sources:
  - Timer: a fixed cadence (`manager.cycle_minutes`), only within trading
    session hours (`trading.trading_sessions`), degrading (stretching)
    the interval as the daily API budget gets tight.
  - Event: files dropped under `control/manager_events/*.json` by
    risk-side code (`src.manager.events.drop_manager_event`) on
    drawdown/consecutive-losses/circuit-breaker triggers. Consumed
    (read + deleted) oldest-first.

Both are subject to `min_gap_minutes` (no cycle sooner than this after
the previous one) and the API budget governor (skip entirely if today's
spend or the trailing N-day cumulative spend is at/over its cap).

No network, no Anthropic SDK. All wall-clock reads go through an
injectable `now_fn` so tests never depend on real time.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

from src.config import PROJECT_ROOT

logger = logging.getLogger("traderbot.manager.scheduler")

MANAGER_EVENTS_DIR = PROJECT_ROOT / "control" / "manager_events"

# Fallback estimate (ZAR) for one manager cycle's API cost, used only to
# decide whether to stretch the timer interval as the daily budget gets
# tight. The real cost of each cycle is whatever `client.call()` actually
# reports, logged via `log_manager_cycle`.
DEFAULT_ESTIMATED_COST_PER_CYCLE_ZAR = 3.0


def _session_boundary(now: datetime, reset_hour: int) -> datetime:
    """Most recent daily session-boundary timestamp at/before `now`."""
    candidate = now.replace(hour=reset_hour, minute=0, second=0, microsecond=0)
    if candidate > now:
        candidate -= timedelta(days=1)
    return candidate


class ManagerScheduler:
    def __init__(
        self,
        config,
        journal,
        telegram=None,
        now_fn: Optional[Callable[[], datetime]] = None,
        events_dir: Path = MANAGER_EVENTS_DIR,
        estimated_cost_per_cycle_zar: float = DEFAULT_ESTIMATED_COST_PER_CYCLE_ZAR,
    ):
        self.config = config
        self.journal = journal
        self.telegram = telegram
        self.now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self.events_dir = events_dir
        self.estimated_cost_per_cycle_zar = estimated_cost_per_cycle_zar

        self.enabled = bool(config.get("manager.enabled", True))
        self.cycle_minutes = config.get("manager.cycle_minutes", 60)
        self.min_gap_minutes = config.get("manager.min_gap_minutes", 20)
        self.max_cycles_per_day = config.get("manager.max_cycles_per_day", 20)
        # Same fallback chain as src/main.py and src/risk/*: the newer
        # trading.session_reset_hour_utc key is authoritative; the older
        # risk.session_boundary_hour_utc kept only as a fallback.
        self.session_boundary_hour = config.get(
            "trading.session_reset_hour_utc",
            config.get("risk.session_boundary_hour_utc", 21),
        )
        self.api_budget_zar_total = config.get("manager.api_budget_zar_total", 500)
        self.api_budget_days = config.get("manager.api_budget_days", 10)
        self.api_budget_zar_per_day = config.get("manager.api_budget_zar_per_day", 50)

        # Seed min-gap from the journal so a process restart can't fire a
        # cycle immediately after a recent one recorded by the previous run.
        self._last_cycle_at: Optional[datetime] = self._last_cycle_from_journal()
        self._next_timer_at: Optional[datetime] = None
        self._budget_warned_date = None
        # Events seen but not yet acted on (deferred by min-gap). Kept in
        # memory so a suppressed event fires once the gap expires instead
        # of being silently deleted with its file.
        self._pending_events: list = []

    def _last_cycle_from_journal(self) -> Optional[datetime]:
        try:
            df = self.journal.get_manager_log(limit=1)
            if df.empty:
                return None
            return datetime.fromisoformat(str(df.iloc[0]["ts_utc"]))
        except Exception as e:  # pragma: no cover — defensive
            logger.warning(f"manager scheduler: could not seed min-gap from manager_log: {e}")
            return None

    # ------------------------------------------------------------------
    # Event consumption
    # ------------------------------------------------------------------

    def poll_events(self) -> list:
        """Read + delete all pending event files, oldest-first."""
        if not self.events_dir.exists():
            return []
        events = []
        for path in sorted(self.events_dir.glob("*.json")):
            try:
                events.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception as e:
                logger.warning(f"manager scheduler: skipping unreadable event {path}: {e}")
            try:
                path.unlink()
            except OSError as e:  # pragma: no cover — defensive
                logger.warning(f"manager scheduler: could not delete event {path}: {e}")
        return events

    # ------------------------------------------------------------------
    # Session-hours / timer
    # ------------------------------------------------------------------

    def in_session_hours(self, now: datetime) -> bool:
        sessions = self.config.get("trading.trading_sessions", {}) or {}
        hour = now.hour
        for spec in sessions.values():
            start = spec.get("start_hour")
            end = spec.get("end_hour")
            if start is None or end is None:
                continue
            if start <= hour < end:
                return True
        return False

    # ------------------------------------------------------------------
    # Daily cap
    # ------------------------------------------------------------------

    def cycles_today(self, now: datetime) -> int:
        day_start = _session_boundary(now, self.session_boundary_hour)
        df = self.journal.get_manager_log(limit=None)
        if df.empty:
            return 0
        return int((df["ts_utc"] >= day_start.isoformat()).sum())

    def check_daily_cap(self, now: datetime) -> tuple[bool, Optional[str]]:
        if self.cycles_today(now) >= self.max_cycles_per_day:
            return False, "daily_cap_exceeded"
        return True, None

    # ------------------------------------------------------------------
    # Budget governor
    # ------------------------------------------------------------------

    def spend_today(self, now: datetime) -> float:
        day_start = _session_boundary(now, self.session_boundary_hour)
        return float(self.journal.manager_cost_since(day_start))

    def spend_over_budget_window(self, now: datetime) -> float:
        window_start = now - timedelta(days=self.api_budget_days)
        return float(self.journal.manager_cost_since(window_start))

    def check_budget(self, now: datetime) -> tuple[bool, Optional[str]]:
        """
        Returns (ok, reason). ok=False means the cycle must be skipped
        with outcome=budget_exhausted; reason is 'daily_budget_exhausted'
        or 'total_budget_exhausted'.
        """
        if self.spend_today(now) >= self.api_budget_zar_per_day:
            return False, "daily_budget_exhausted"
        if self.spend_over_budget_window(now) >= self.api_budget_zar_total:
            return False, "total_budget_exhausted"
        return True, None

    def warn_budget_exhausted_once(self, now: datetime) -> None:
        """Send a Telegram budget-exhausted warning at most once per day."""
        today = now.date()
        if self._budget_warned_date == today:
            return
        self._budget_warned_date = today
        if self.telegram is not None:
            try:
                self.telegram.manager_cycle(
                    "[MANAGER] API budget exhausted for today — cycles paused until the next window."
                )
            except Exception as e:  # pragma: no cover — defensive
                logger.warning(f"manager scheduler: budget warning telegram failed: {e}")

    def next_interval_minutes(self, now: datetime) -> float:
        """
        Normal cadence is `cycle_minutes`. When the remaining daily budget
        is less than (estimated cost per cycle * remaining cycles for the
        rest of the day at normal cadence), stretch the interval to spread
        the remaining budget across the rest of the day instead.
        """
        day_start = _session_boundary(now, self.session_boundary_hour)
        day_end = day_start + timedelta(hours=24)
        minutes_left_in_day = max(0.0, (day_end - now).total_seconds() / 60.0)

        if self.estimated_cost_per_cycle_zar <= 0 or minutes_left_in_day <= 0:
            return self.cycle_minutes

        remaining_cycles_normal_cadence = max(1, int(minutes_left_in_day // self.cycle_minutes))
        remaining_budget = max(0.0, self.api_budget_zar_per_day - self.spend_today(now))

        needed_for_normal_cadence = self.estimated_cost_per_cycle_zar * remaining_cycles_normal_cadence
        if remaining_budget >= needed_for_normal_cadence:
            return self.cycle_minutes

        affordable_cycles = int(remaining_budget // self.estimated_cost_per_cycle_zar)
        if affordable_cycles < 1:
            # Can't afford even one more full cycle today — push past the
            # remainder of the day rather than firing again immediately.
            return minutes_left_in_day + self.cycle_minutes
        stretched = minutes_left_in_day / affordable_cycles
        return max(self.cycle_minutes, stretched)

    # ------------------------------------------------------------------
    # Main decision
    # ------------------------------------------------------------------

    def should_fire(self) -> tuple[bool, Optional[str], str]:
        """
        Decide whether a cycle should fire right now.

        Returns (fire, trigger, reason):
          - fire=True, trigger='<event type>' or 'timer', reason describing why.
          - fire=False, trigger=None, reason one of
            'disabled' | 'min_gap' | 'waiting' | 'daily_cap_exceeded'.

        Event handling policy:
          - disabled: event files are left on disk untouched.
          - min-gap: events are consumed from disk but DEFERRED in memory —
            they fire as soon as the gap expires (no signal loss).
          - daily cap: deferred/pending events are DROPPED intentionally,
            with an explicit log line (the cap is a hard daily stop; a
            stale event firing after the 21:00 UTC boundary would be
            acting on yesterday's signal).

        Does NOT consult the budget governor — callers must call
        `check_budget()` separately once `fire` is True, since a
        budget-exhausted skip still needs to be logged/warned (unlike a
        plain "not time yet" no-op).
        """
        if not self.enabled:
            return False, None, "disabled"

        now = self.now_fn()
        self._pending_events.extend(self.poll_events())

        if self._last_cycle_at is not None:
            gap_minutes = (now - self._last_cycle_at).total_seconds() / 60.0
            if gap_minutes < self.min_gap_minutes:
                return False, None, "min_gap"

        ok, cap_reason = self.check_daily_cap(now)
        if not ok:
            if self._pending_events:
                logger.info(
                    "manager scheduler: dropping %d pending event(s) at daily cap: %s",
                    len(self._pending_events),
                    [e.get("type", "event") for e in self._pending_events],
                )
                self._pending_events = []
            return False, None, cap_reason

        if self._pending_events:
            events, self._pending_events = self._pending_events, []
            return True, events[0].get("type", "event"), "event"

        if self._next_timer_at is None:
            self._next_timer_at = now

        if now >= self._next_timer_at and self.in_session_hours(now):
            return True, "timer", "timer"

        return False, None, "waiting"

    def record_cycle(self, now: Optional[datetime] = None) -> None:
        """Call once a cycle has actually run (or been budget-skipped) so
        min_gap and the next timer interval are computed from this point."""
        now = now or self.now_fn()
        self._last_cycle_at = now
        self._next_timer_at = now + timedelta(minutes=self.next_interval_minutes(now))
