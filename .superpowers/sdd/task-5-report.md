# Task 5 Report — Blockers G/Q + H, plus item S (main-loop catch-all)

## Scope
- `src/execution/executor.py`: trade-ID integrity (Blocker G/Q)
- `src/main.py`: daily-summary scheduler (Blocker H) + main-loop catch-all (item S)
- `src/monitoring/telegram_bot.py`: `bot_error()` alert method
- Tests: `tests/test_trade_id_integrity.py`, `tests/test_daily_summary_scheduler.py`,
  additions to `tests/test_telegram_mt5_alerts.py`

Merged `cd17344` (Tasks 1-4) into this worktree branch via fast-forward before starting.

## 1. Trade-ID integrity (Blocker G/Q)

**File:** `src/execution/executor.py`, `execute_signal()` (~L197-232 post-change).

Removed the timestamp-fallback synthetic trade ID
(`trade_id = str(fill.get("id", f"local_{int(time.time())}"))`). Confirmed via
`src/data/mt5_client.py::place_market_order` (read-only, not modified — owned by another
agent) that the OANDA-compatible response always carries `tradeOpened.tradeID =
str(result.order)`, i.e. the real MT5 ticket. Empty string or the literal `"0"` now means
the broker did not actually confirm a position.

New behavior on missing/zero ticket:
- `logger.error(...)` describing the failed order and full response payload
- `risk_manager.record_api_error()` (treats it as an API-level failure, consistent with
  other order-placement error paths in the same function)
- fires `alert_callback("order_failed", {...})` if wired (best-effort, wrapped in
  try/except so a broken callback can't blow up the executor)
- returns `None` — same contract as every other rejection path in `execute_signal()`

Since `main.py::_evaluate_trade_signal()` and `_execute_approved_trades()` only call
`journal.record_trade(...)` `if trade:` (non-None), returning `None` here means **no
journal row is ever written** for a failed order — satisfies "no synthetic journal row."
No position is added to `self.open_trades` either.

Wired `main.py::_on_executor_alert()` to handle the new `"order_failed"` event: sends a
Telegram alert and records a `"order_failed"` journal *event* (distinct from a trade row —
this is the existing `record_event()` system-events table, same one used for
`close_all`/`milestone`/`retrain_trigger`).

**Tests** (`tests/test_trade_id_integrity.py`, 3 tests):
- missing ticket (`""`) → `execute_signal()` returns `None`, `open_trades` stays empty,
  alert_callback fires once with `event == "order_failed"`
- literal `"0"` ticket → same failure path
- regression guard: a real ticket (`"123456789"`) still opens and tracks the trade

## 2. Daily summary scheduler (Blocker H)

Added `DailySummaryScheduler` (module-level class in `src/main.py`, above `TraderBot`).
Reuses the Task-4 `session_boundary()` helper from `src/risk/drawdown_tracker.py` and the
same config fallback chain: `trading.session_reset_hour_utc` →
`risk.session_boundary_hour_utc` → `21`.

- `due()` returns the boundary date string (`"YYYY-MM-DD"`) if a summary hasn't fired for
  the current boundary yet, else `None`. Checks an in-memory `_last_fired_date` first
  (cheap, covers the common single-process case), then falls back to querying
  `journal.get_events(event_type="daily_summary_sent")` for a prior row matching the
  boundary date — this is the "journal event row" guard that survives a process restart.
- `mark_fired(boundary_date)` sets the in-memory guard and calls
  `journal.record_event("daily_summary_sent", boundary_date, {...})`.

`TraderBot`:
- `__init__`: `self.daily_summary_scheduler = None`
- `setup()`: `self.daily_summary_scheduler = DailySummaryScheduler(self.config, self.journal)`
  (constructed right after `self.telegram`/`self.journal`/`self.performance`)
- `run()`: inside the existing `if cached_balance is not None:` gate (same cadence as the
  session-boundary/drawdown-emergency check), calls
  `self._maybe_send_daily_summary(cached_balance)`
- `_maybe_send_daily_summary(balance)`: asks the scheduler if due; if so, pulls
  `performance.get_summary()` and calls `telegram.daily_summary(...)` (existing method,
  previously dead code — this is the first caller), then `scheduler.mark_fired(...)`.
  Wrapped in try/except — never raises into the main loop.

**Tests** (`tests/test_daily_summary_scheduler.py`, 12 tests): boundary-crossing behavior
of `DailySummaryScheduler.due()`/`mark_fired()` with an injected clock (no sleeps),
double-fire guard across a simulated process restart (fresh scheduler instance sharing
the same on-disk journal db), config fallback chain resolution, and
`TraderBot._maybe_send_daily_summary()` wiring (fires exactly once when due, no-ops when
not due, never raises on Telegram failure).

## 3. Main-loop catch-all (item S, controller-assigned)

`run()`'s single `try/except` around the *entire* `while self.running:` loop previously
meant one unhandled exception in an iteration would break out of the loop and go straight
to `shutdown()` — effectively killing the bot on any bug. Restructured so each iteration
has its own inner `try/except`, delegating to a new `_handle_loop_exception(exc)`:

- Logs `logger.error(..., exc_info=True)` with the full traceback (not `critical`, since
  the loop is continuing, not dying)
- Rate-limits the Telegram alert to once per 5 minutes **per exception type**
  (`type(exc).__name__` as the key, `self._loop_error_last_alert: dict[str, float]`,
  `self._loop_error_cooldown_seconds = 300`)
- Calls `telegram.bot_error(exc_type, str(exc))` (new method, see below) and
  `journal.record_event("bot_error", ...)`, both wrapped so a failing alert can't
  re-raise
- The outer `while self.running:` loop is untouched by the exception — it continues to
  the next iteration

The pre-existing outer `try/except` (now effectively only catching truly catastrophic
failures outside the per-iteration guard, e.g. errors in the `while` condition itself)
and its `logger.critical(...)` + `shutdown()` in `finally` are unchanged.

Added `TelegramBot.bot_error(exc_type, message)` in `src/monitoring/telegram_bot.py`:
best-effort (try/except, never raises), gated by a new
`telegram.alert_on_bot_error` config toggle (default `True`, following the existing
`alert_on_*` pattern).

**Tests**: `_handle_loop_exception` behavior covered in
`tests/test_daily_summary_scheduler.py` (alerts once, rate-limits same exception type,
treats different exception types independently, never raises when the alert itself
fails). `bot_error()` itself covered by 3 new tests appended to
`tests/test_telegram_mt5_alerts.py` (sends message with type+message, never raises on
send failure, respects the disabled toggle).

## Verification
`python -m pytest tests/ -x -q` → **93 passed** (75 pre-existing + 18 new: 3 trade-ID +
12 daily-summary/loop-error + 3 bot_error).

## Self-review / concerns
- `DailySummaryScheduler` and the loop catch-all are new, not explicitly requested as
  separate classes in the brief, but this keeps `TraderBot` testable without constructing
  the full heavy `__init__` (MT5 connection, config loading, etc.) — tests instead
  construct `TraderBot.__new__(TraderBot)` and wire only the few attributes each method
  touches, or test `DailySummaryScheduler` directly in isolation.
- `alert_callback` payload for `"order_failed"` includes the raw `response` dict — this
  is JSON-serialized by `journal.record_event()`; confirmed it's a plain dict of strings
  (`{"orderFillTransaction": {"price": ..., "tradeOpened": {"tradeID": ...}, "id": ...}}`),
  so `json.dumps` succeeds.
- Did not touch `mt5_client.py`, `trade_journal.py` schema, `evaluator.py`,
  `candle_builder.py`, `historical_loader.py` — out of scope, owned by parallel agents.
- `config/settings.yaml` already had an unused `telegram.daily_summary_hour_utc` key;
  left it untouched per brief's explicit instruction to reuse the
  `trading.session_reset_hour_utc` → `risk.session_boundary_hour_utc` → `21` chain
  instead, for consistency with Task 4's session-boundary logic.
