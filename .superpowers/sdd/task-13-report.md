# Task 13 — Manager client + scheduler + runner — Report

## Summary

Implemented the Claude-manager's API client, scheduler (including the API
budget governor), main runner loop, event-drop wiring at the three risk
trigger points, and the Telegram alert method. All work builds on the
merged Tasks 1-12 foundation (`policy.py`, `briefing.py`, `manager_log`
in `trade_journal.py`, the control queue).

## Files created

- `config/settings.yaml` — added the `manager:` block (exact match to spec).
- `src/manager/prompts/v001.md` — system prompt (role, four levers +
  bounds, `risk_ceiling_now` semantics, briefing schema, decision
  principles).
- `src/manager/prompts/champion.txt` — contains `v001.md`.
- `src/manager/client.py` — `ManagerClient`, `ManagerAPIUnavailable`,
  `cost_zar()`, `load_champion_prompt()`, `PROPOSE_ADJUSTMENTS_TOOL`.
- `src/manager/events.py` — `drop_manager_event()` atomic event-drop helper.
- `src/manager/scheduler.py` — `ManagerScheduler` (timer + event triggers,
  min-gap, daily cap, budget governor).
- `src/manager/runner.py` — `ManagerRunner` (the main loop), balance/equity
  cross-process read, control-queue tune enqueue.
- `src/manager/__main__.py` — `python -m src.manager` entry point.
- `tests/test_manager_client.py` (10 tests)
- `tests/test_manager_scheduler.py` (10 tests)
- `tests/test_manager_runner.py` (7 tests)

## Files edited

- `src/risk/drawdown_tracker.py` — event drop when intraday drawdown
  crosses `manager.event_triggers.drawdown_pct` (2.0%), gated to once per
  session day.
- `src/risk/circuit_breaker.py` — event drop when consecutive losses
  reach `manager.event_triggers.consecutive_losses` (3), and on every
  `_pause()` / `_shutdown()` call (i.e. any circuit-breaker trip).
- `src/monitoring/telegram_bot.py` — added `manager_cycle(summary)` alert
  method, following the existing `_send()` pattern.
- `tests/test_cli_tb.py` — fixed a pre-existing (pre-Task-13) regression:
  `test_manager_command_reads_existing_manager_log` manually created a
  `manager_log` table, which now already exists (Task 12's
  `TradeJournal` schema init creates it). Updated the test to insert via
  `journal.log_manager_cycle()` and assert on the real `outcome` column
  instead of a stub `note` column. This was failing at the merge base
  (276 passed / 1 failed just after `git merge d0c85bd`), not something
  I introduced — fixed it since it's directly in the manager_log
  integration surface this task builds on.
- `requirements.txt` — already had `anthropic>=0.40.0` from a prior
  commit; no change needed.
- `.gitignore` — already ignores `control/manager_events/`; no change needed.

`pip install anthropic` was run in this environment (installed
anthropic 0.116.0) so tests can import the SDK.

## How the budget governor works

Implemented in `ManagerScheduler` per the plan's amendment section,
exactly:

- `check_budget(now)` — before every cycle: sums `manager_log.cost_zar`
  since the current session-day boundary (21:00 UTC) via
  `journal.manager_cost_since(day_start)`; if that's `>= manager.api_budget_zar_per_day`
  (default R50), returns `(False, "daily_budget_exhausted")`. Otherwise
  sums cost over the trailing `manager.api_budget_days` (default 10) window;
  if `>= manager.api_budget_zar_total` (default R500), returns
  `(False, "total_budget_exhausted")`.
- When budget-exhausted, the runner logs `outcome="budget_exhausted"` via
  `log_manager_cycle` and calls `scheduler.warn_budget_exhausted_once(now)`,
  which sends exactly one Telegram warning per calendar day (tracked via
  `self._budget_warned_date`), never duplicating within the same day.
- `next_interval_minutes(now)` — the timer-cadence stretch: computes
  minutes left until the next 21:00 UTC boundary, how many cycles the
  normal `cycle_minutes` cadence would fire in that remaining time, and
  the remaining daily budget. If `remaining_budget < estimated_cost_per_cycle * remaining_cycles_normal_cadence`,
  it spreads the remaining budget evenly across the rest of the day
  (`minutes_left_in_day / affordable_cycles`), floored at `cycle_minutes`
  (never fires *faster* than normal cadence). If the remaining budget
  can't afford even one more full estimated cycle today, it pushes the
  next timer fire past the rest of today rather than firing immediately
  again.
- `estimated_cost_per_cycle_zar` is a constructor parameter (default
  R3.00) — a rough estimate used only for spacing decisions; the actual
  cost of each cycle (from `client.call()`'s real token usage) is what
  gets logged and what `check_budget` actually gates on.

## Test results

`python -m pytest tests/ -x -q` → **304 passed, 0 failed**
(277 at merge base + 1 pre-existing regression fixed + 27 new: 10 client,
10 scheduler, 7 runner).

## Deviations / judgment calls

1. **Balance/equity source**: the brief suggested `account_state.json`
   as a possible mechanism but flagged it as speculative ("if you can't
   find an existing writer... use whatever mechanism already exists").
   I found the actual existing cross-process mechanism: the
   `status_snapshot` control-queue round trip (`cli/tb.py::cmd_status`
   already does exactly this — enqueue a `status_snapshot` command,
   poll the outbox, degrade to journal-derived on timeout). I reused
   that pattern in `runner._default_balance_equity_fn()` rather than
   inventing a new `account_state.json` writer. Note:
   `src/risk/ratchet_floor.py` already uses a file literally named
   `data/account_state.json`, but that file only stores the ratchet
   floor's high-water mark, not live balance/equity — using it for
   balance/equity would have been a real bug, not just a stylistic
   choice.
2. **`api_unavailable` vs `no_op` on API-down**: the brief's "Task
   summary" component list (step 5) and the "Testing constraints"
   section disagree slightly — the component spec lists
   `api_unavailable` as its own outcome value distinct from `no_op`,
   while the testing-constraints bullet says "API down ... → outcome
   no_op + telegram alert sent". I followed the more specific/detailed
   component spec (`outcome="api_unavailable"`) since it's the
   authoritative enum definition; `no_op` is reserved for "the model
   responded but proposed nothing." This is testable either way — flag
   this in case the more literal reading (`no_op`) was actually intended.
3. **Tune enqueue is fire-and-forget**: `_enqueue_tune()` writes directly
   to the inbox (tmp-write + `os.replace`, the same writer contract as
   `cli/tb.py::enqueue_command`) but does **not** poll the outbox for a
   result, since the manager runs as an independent process and
   shouldn't block its loop waiting for the bot to be up and draining
   its inbox. The bot's `ControlQueue.poll_once()` will pick it up
   whenever it's next running.
4. **`estimated_cost_per_cycle_zar`** is not in the settings.yaml spec
   block (which the brief pins down exactly) — it's a constructor-level
   default (R3.00) used only for the interval-stretch heuristic, not a
   new config key, so it doesn't touch the required settings.yaml shape.
5. **Circuit-breaker trip event**: wired into `_pause()` and
   `_shutdown()` directly (both call sites cover all four trip reasons:
   consecutive-loss pause, win-rate pause, API-error pause, hard-floor
   shutdown) rather than adding a call at each of the four call sites
   individually — this is more surgical (one place per method) and
   still fires on "any circuit breaker trip" as specified.

## Concerns / TODOs

- `ManagerRunner`'s default constructor path (no injected client/
  scheduler/telegram) will construct a real `anthropic.Anthropic()`
  client, a real `TelegramBot`, and call `EffectiveConfig.load()` /
  `RatchetFloor()` against the real project paths — this is correct for
  production (`python -m src.manager`) but means `ManagerRunner()` with
  no args requires `ANTHROPIC_API_KEY` to be set; this was not
  exercised end-to-end against a live key (out of scope — no network
  calls in this repo's test suite by design).
- The daily-cap / budget-governor cycle counting reads the *entire*
  `manager_log` table via `get_manager_log(limit=None)` each check; fine
  at current expected volumes (≤20 rows/day) but would want a date-range
  query if the table grows very large over months of operation.
- Task 14 (day-10 verdict / `justification_report()`) is explicitly out
  of scope for this task and was not touched.

## Commits

See `git log` on this worktree branch for the exact hashes/messages
(all prefixed `feat(manager): `).
