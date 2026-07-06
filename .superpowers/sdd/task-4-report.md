# Task 4 Report — Blockers D/E/F: daily-drawdown emergency close-all + 21:00 UTC session resets

Worktree: `C:\Users\liamp\Desktop\Portfolio\TraderBot\.claude\worktrees\agent-a778bc0c65b4b60ec`
Branch: `worktree-agent-a778bc0c65b4b60ec`
Commit: `ca26c57`

## Preliminary fix: stale worktree base

The worktree had branched from `e5f680e`, which predates Task 1
(`8fcf89e feat(risk): re-base capital to R1000 and add ratcheting hard floor`).
`src/risk/ratchet_floor.py` and the `RatchetFloor` injection into
`CircuitBreaker`/`DrawdownTracker` were missing. Per the controller's
mid-task instruction, fast-forward merged: `git merge --ff-only` up to
`8fcf89e` (clean fast-forward, no conflicts — the worktree branch had zero
unique commits of its own). Confirmed `RatchetFloor` present afterward and
proceeded with Task 4 on top of it. Did **not** pull in Task 2's
single-instance-lock commit (`7aca15a`) — unrelated to this task's files
and not referenced by the brief's symbol map.

## Implementation

### `src/risk/drawdown_tracker.py`
- New module-level `session_boundary(now, reset_hour, weekday=None)` helper:
  returns the most recent daily boundary (`reset_hour`:00 UTC) or, when
  `weekday` is given, the most recent weekly boundary anchored to that
  weekday (Friday=4) at `reset_hour`:00 UTC.
- `DrawdownTracker.__init__` gained `clock` (injectable, defaults to real
  UTC) and reads `session_reset_hour` from
  `trading.session_reset_hour_utc`, falling back to the existing
  `risk.session_boundary_hour_utc` (=21) key already in
  `config/settings.yaml` (see Deviation note below).
- `initialize()`/`update()` now compute daily/weekly boundaries via
  `session_boundary()` instead of comparing calendar dates, so resets fire
  exactly at 21:00 UTC (daily) / Friday 21:00 UTC (weekly) rather than
  midnight.
- `_handle_new_day`/`_handle_new_week` now store the boundary timestamp
  itself as `current_date`/`current_week_start`.

### `src/risk/circuit_breaker.py`
- New `reset_consecutive_losses()`: resets only `consecutive_losses`.
  Deliberately does **not** touch `is_paused`/`is_shutdown`/
  `recent_outcomes` (win-rate rolling window) — those aren't session-scoped
  and `reset()` (full reset, for manual review/retrain) is left as-is.

### `src/risk/manager.py`
- `RiskManager.__init__` gained `clock` and `ratchet_floor` (both optional,
  keyword, default-preserving) so tests can inject a frozen/advancing clock
  and a tmp-path-backed `RatchetFloor` instead of touching
  `data/account_state.json`.
- `check_session_boundary(current_balance=None)`: detects the 21:00 UTC
  crossing; on crossing resets `trades_today`, calls
  `circuit_breaker.reset_consecutive_losses()`, rolls the drawdown
  tracker's baseline forward via `drawdown.update(current_balance)` when a
  balance is supplied (both share the same clock so they agree on which
  boundary just passed), and lifts `_blocked_until_boundary`.
- `check_drawdown_emergency(current_balance, current_equity=None)`: calls
  `check_session_boundary()` then `drawdown.check()`; on a fresh daily
  drawdown violation, sets `_blocked_until_boundary=True` +
  `_block_reason`, logs CRITICAL, and returns `True` **once** (returns
  `False` on subsequent calls while still blocked, so callers don't
  re-trigger close_all every tick).
- `evaluate_trade()`: new Check 0 rejects trades while
  `_blocked_until_boundary`; Check 3 (drawdown) also flags a fresh breach
  inline so a breach detected via a trade evaluation (not just the
  periodic check) still triggers the block.
- `close_all_signal()` now returns `circuit_breaker.is_shutdown OR
  _blocked_until_boundary`; new `close_all_reason` property distinguishes
  `"daily_drawdown"` from `"circuit_breaker_shutdown: <reason>"`.

### `src/execution/executor.py`
- Reused the existing `close_all(reason)` per the controller's deviation
  note (did not add a parallel `close_all_positions`). It now logs each
  individual close (instrument/direction/id/PnL) in addition to the
  existing summary log, and fires an injected `alert_callback(event,
  data)` with `{reason, requested, closed, results}` if one was wired at
  construction.
- `Executor.__init__` gained optional `alert_callback` — **Telegram is not
  touched**; per the task's constraint I did not edit `telegram_bot.py`.
  Wiring `alert_callback=telegram_bot.send_alert`-style from `main.py` is
  left to whichever task owns `main.py`.
- `check_and_manage_positions()` gained optional `current_balance`/
  `current_equity` kwargs (default `None`, so existing call sites — none
  currently exist per the callers check below — are unaffected either
  way): when a balance is supplied, it runs
  `risk_manager.check_drawdown_emergency()` and calls `close_all(reason=
  "daily_drawdown")` on a fresh breach; it also unconditionally checks
  `risk_manager.close_all_signal()` (covers circuit-breaker shutdown too)
  and calls `close_all()` with `risk_manager.close_all_reason`.

## Deviations / assumptions (flagging, not blocking)

1. **`close_all` vs `close_all_positions`** — per controller's explicit
   deviation note, reused the existing `Executor.close_all(reason)`.
2. **Config key name** — the brief says read
   `trading.session_reset_hour_utc`, but that key does not exist anywhere
   in `config/settings.yaml`; the actual key is
   `risk.session_boundary_hour_utc: 21` (comment: "daily resets fire at
   this UTC hour (NYSE close)"). Implemented a fallback chain:
   `config.get("trading.session_reset_hour_utc", config.get(
   "risk.session_boundary_hour_utc", 21))` — this satisfies the brief's
   literal instruction while working correctly against the real config.
   Flagging this in case the intent was actually to *add* the
   `trading.*` key to `settings.yaml` (out of scope for this task's file
   list, and `config/settings.yaml` wasn't listed as touchable).
3. **Wiring gap in `main.py`** — `check_and_manage_positions()` is not
   currently invoked anywhere in `src/main.py` (confirmed via a read-only
   Explore sub-agent), so the periodic drawdown-emergency check has no
   live caller yet in production. This is pre-existing (not something I
   introduced) and `main.py` is explicitly out of scope for this task;
   whichever task next touches `main.py`'s trading loop needs to call
   `executor.check_and_manage_positions(current_prices=..., current_balance=...,
   current_equity=...)` periodically, and should wire `alert_callback=` at
   `Executor` construction time to the Telegram bot's send method.
4. Sub-agent confirmed all other call sites (`main.py`'s
   `RiskManager(self.config)`, `Executor(self.config, self.client,
   self.risk_manager)`, `close_all_signal()`, `close_all("emergency_shutdown")`
   at lines ~218/220/903) remain compatible — all new constructor/method
   params are optional and keyword- or trailing-positional-safe.

## TDD evidence

Tests and implementation were developed together in each file iteration
(read existing code → write test asserting the new boundary/breach
behavior → adjust implementation until green), then the full suite was
run repeatedly. Representative RED→GREEN cycle actually hit during
iteration:

- RED: `test_block_lifts_and_entries_resume_after_boundary` failed —
  `check_session_boundary()` reset the circuit breaker/block flag but
  never rolled `DrawdownTracker.daily_start_balance` forward, so
  `evaluate_trade()` still saw the old 5%-down baseline after crossing.
- GREEN: added `current_balance` param to `check_session_boundary()`,
  calling `self.drawdown.update(current_balance)` on crossing (shared
  clock lets DrawdownTracker detect the identical boundary and rebase).
- RED: `MagicMock` balance from `client.get_account_balance()` blew up the
  `current_balance > self.high_water_mark` comparison in
  `DrawdownTracker.update()` — test mock wasn't configured with a numeric
  return value.
- GREEN: set `client.get_account_balance.return_value = balance` in the
  test fixture.

Final: `python -m pytest tests/ -x -q` → **39 passed** (17 pre-existing +
22 new across `tests/test_session_boundary.py` and
`tests/test_emergency_close.py`).

## Self-review

- Scope: only touched the 4 named files + 2 new test files (verified via
  `git diff --stat` before commit). Did not touch `mt5_client.py`,
  `collector.py`, `telegram_bot.py`, `main.py`, or `settings.yaml`.
- All new/changed public methods have docstrings explaining the
  session-boundary semantics and why they differ from `reset()`.
- Timestamps: all boundary math uses tz-aware UTC datetimes throughout;
  tests inject a mutable clock closure (`clock_box["now"]`), never sleep.
- Idempotency: `check_drawdown_emergency()` returns `True` only once per
  breach; `close_all()` is safe to call repeatedly (no-ops once
  `open_trades` is empty), so the double-trigger in
  `check_and_manage_positions()` (once from the emergency check, once from
  the generic `close_all_signal()` check) never double-closes.
- Risk: `check_and_manage_positions()`'s current-balance param is optional
  specifically so the (currently nonexistent) production call site can be
  wired incrementally without a hard coupling; flagged as a concern above
  rather than silently assumed resolved.

## Concerns for follow-up

- `main.py` needs to actually call `check_and_manage_positions(...,
  current_balance=..., current_equity=...)` periodically and wire
  `alert_callback` for this to take effect live — currently only the
  per-trade `evaluate_trade()` path and any direct test/manual call to
  `check_drawdown_emergency()` exercise this logic.
- Confirm whether `settings.yaml` should gain a `trading.session_reset_hour_utc`
  key to match the brief literally, or whether the existing
  `risk.session_boundary_hour_utc` is the intended long-term home (fallback
  chain currently handles either).

## Fix Round 1

Merged Task 2's single-instance-lock commit first: `git merge 4792d37
--no-edit` — clean, no conflicts (`main.py` gained the lock init/release
block, `src/utils/instance_lock.py`, `tests/test_instance_lock.py`).
Merge commit: `1903878`.

### Finding 1 — `close_all_signal()` conflated permanent kill with the resumable daily-drawdown pause
`src/risk/manager.py`: reverted `close_all_signal()` to
`return self.circuit_breaker.is_shutdown` only. Added a new
`entries_blocked` property (`return self._blocked_until_boundary`) as the
correct way for callers to check the resumable daily block.
`close_all_reason` no longer falls back to `"daily_drawdown"` — it only
ever describes a circuit-breaker shutdown now (empty string otherwise).

### Finding 2 — breach detection/lifting never ran from main.py
`src/main.py` `run()`: added, once per ~1s loop iteration:
- A cached balance/equity refresh via `client.get_account_summary()`,
  throttled to every 5s (`last_balance_check`) so the emergency checks
  don't hammer `account_info()` every iteration.
- `risk_manager.check_session_boundary(cached_balance)` — lifts the block
  and resets the consecutive-loss counter at the 21:00 UTC boundary,
  independent of any trade signal.
- `risk_manager.check_drawdown_emergency(cached_balance, cached_equity)` —
  on a fresh breach: `executor.close_all("daily_drawdown")`,
  `logger.critical(...)`, `telegram.daily_stop(balance, drawdown_pct)`
  (via `risk_manager.drawdown.get_daily_drawdown_pct(balance)`), and the
  bot is deliberately **not** stopped (`self.running` untouched).
- Kept the pre-existing permanent-kill branch (`close_all_signal()` →
  `close_all("emergency_shutdown")` → `telegram.emergency_stop(...)` →
  `self.running = False; break`) unchanged in behavior, now correctly
  gated on circuit-breaker shutdown only (Finding 1).

### Finding 3 — `Executor` constructed without `alert_callback`
`src/main.py`: `Executor(...)` now passes
`alert_callback=self._on_executor_alert`. New `_on_executor_alert(event,
data)` method sends a best-effort Telegram message + records a
`close_all` journal event for the `"close_all"` event, including any
`failures` list in the message. Wrapped in try/except so a failed alert
never propagates into the executor.

### Finding 4 — emergency close failures were silent
`src/execution/executor.py`:
- `close_trade()` no longer swallows `MT5Error` — it now propagates so
  `close_all()` can attribute the failure to the specific trade.
- `close_all()` now catches per-trade `MT5Error`, collects
  `{"trade_id", "error"}` into a `failures` list, includes it in the
  alert payload (`data["failures"]`) and in a summary log line, and logs
  at `logger.critical` (vs `logger.error`) both per-failure and in the
  summary when `reason` indicates an emergency close (`"daily_drawdown"`,
  `"emergency_shutdown"`, or any reason containing `"shutdown"` — this
  also covers `close_all_reason`'s `"circuit_breaker_shutdown: ..."`
  format used by the permanent-kill path).

### Finding 5 — tests
- Updated `tests/test_emergency_close.py` and
  `tests/test_session_boundary.py`: every assertion that previously
  expected `close_all_signal()` to be `True` during a daily-drawdown block
  now asserts `entries_blocked is True` and `close_all_signal() is False`
  instead (conflated-behavior tests fixed).
- Added `test_boundary_crossing_lifts_block_without_any_trade_signal`:
  simulates three loop iterations (mirroring exactly what `main.py`'s
  `run()` now calls — `check_session_boundary` + `check_drawdown_emergency`
  + the `close_all_signal` branch) via an injected clock, with no trade
  signal ever evaluated; confirms the block lifts purely from the
  boundary crossing.
- Added `test_permanent_kill_path_unchanged_by_loop_wiring`: confirms the
  circuit-breaker shutdown path still reports `stopped is True` and closes
  all positions through the same simulated-loop helper.
- Added `test_close_all_reports_per_trade_failure_detail`: mocks
  `client.close_trade` to raise `MT5Error`, asserts the failed trade stays
  in `open_trades`, the alert payload carries `failures` with trade id +
  error text, and a `CRITICAL`-level log record was emitted.

### Test run
```
python -m pytest tests/ -x -q
```
```
.................................................                        [100%]
49 passed in 0.75s
```
Full suite (Task 1/2/4 tests all included via the merge) is green.

### Deliberately unchanged
- `Executor.check_and_manage_positions()` still exists and is still not
  called anywhere in `main.py` — the brief asked for
  `check_session_boundary`/`check_drawdown_emergency` to be called
  directly from the loop (which duplicates, rather than reuses,
  `check_and_manage_positions`'s breach-handling branch), so that's what
  was wired. `check_and_manage_positions` remains available for trailing-
  stop management wiring, which is out of scope for this fix round.
- Did not touch `config/settings.yaml`'s `trading.session_reset_hour_utc`
  vs `risk.session_boundary_hour_utc` question raised in the original
  report — no new information changes that call.
