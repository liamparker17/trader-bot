# Task 3 Report — MT5 disconnect handling + alerts (Blockers B1/B2/B3 + I)

**Worktree:** `C:\Users\liamp\Desktop\Portfolio\TraderBot\.claude\worktrees\agent-a2be1612ac90b7613`
**Branch:** `worktree-agent-a2be1612ac90b7613`
**Commit:** `bb1e901` — `feat(hardening): MT5 disconnect detection, stream backoff, entry-pause, alerts`

## Implementation

### 1. `src/data/mt5_client.py`
- Added `is_broker_connected() -> bool`: cheap health probe. Checks
  `mt5.terminal_info().connected` and `mt5.account_info() is not None`.
  Wrapped in try/except — any exception is treated as disconnected, never
  raises.
- Rewrote `stream_prices()`'s polling loop:
  - Tracks `consecutive_errors` across iterations.
  - After processing all symbols, also calls `self.is_broker_connected()`
    once per iteration; if false, the iteration counts as an error too
    (this is what catches a real broker outage — tick fetches on Exness/MT5
    often return `None` silently rather than raising, so per-tick
    try/except alone wouldn't have detected an outage).
  - On error: exponential backoff with jitter — `delay = min(60, 1 * 2**(n-1))`,
    `sleep_time = min(delay + uniform(0, delay*0.1), 60)`. Logged at WARNING.
  - On success: `consecutive_errors` resets to 0, reverts to the original
    100ms poll interval.
  - Added `import random` for jitter.

### 2. `src/data/collector.py`
- `DataCollector.__init__` gained an optional `telegram=None` param and
  `self.broker_down = False` (defaults to connected/not-paused).
- New `check_connection() -> bool`: calls `client.is_broker_connected()`
  (exception-safe), manages the disconnect/reconnect state machine:
  - connected → disconnected: sets `broker_down = True`, logs CRITICAL,
    fires `telegram.mt5_disconnected()` (swallows alert exceptions).
  - disconnected → connected: calls `_resync_symbols()`, sets
    `broker_down = False`, fires `telegram.mt5_reconnected()`.
  - Idempotent — repeated calls while state is unchanged don't re-fire
    alerts.
- New `_resync_symbols()`: re-detect the MT5 symbol suffix per enabled
  instrument by calling the existing `client._to_mt5_symbol(instrument)`
  detection function (Task 6 — dedicated symbol-suffix cache — not yet
  landed, so this calls the existing detector directly per the brief's
  fallback instruction).
- New `_health_loop()`: runs on its own daemon thread
  (`mt5-health-check`), started alongside the existing price-stream thread
  in `start_streaming()`. Calls `check_connection()` every ~5s
  (`_health_check_interval_seconds`) while `self._streaming` is true.
  Decoupled from the tick-consuming stream thread deliberately: since
  `stream_prices()` is a generator that only yields on a tick, the
  consuming `for` loop in `_stream_loop()` would block indefinitely during
  a real outage (no tick change → no yield), so it can't reliably drive
  periodic disconnect detection itself.
- Removed the redundant function-local `import time` in `_stream_loop`
  (now a module-level import, needed by `_health_loop` too).

**Main-loop wiring (explicitly out of scope, per instructions):** the
pause flag is exposed as `collector.broker_down` (bool, default `False`).
`main.py` / the executor should check this each loop iteration and skip
approving new-entry signals while `True`. This is documented in the commit
message and here; no changes were made to `main.py` or `executor.py`.

### 3. `src/monitoring/telegram_bot.py`
Added three alert methods, following the existing `_send()`-based pattern,
each wrapped in its own try/except so they can never raise into the
caller (in addition to `_send()`'s own internal safety):
- `mt5_connected(environment="")`
- `mt5_disconnected(reason="connection lost")`
- `mt5_reconnected()`

## Tests (TDD)

Three new test files, 21 tests total, all mock the `MetaTrader5` module
(via monkeypatching `src.data.mt5_client.mt5`) — no real terminal or
network access:

- `tests/test_mt5_client_connection.py` (13 tests)
  - `is_broker_connected()`: true when terminal+account OK; false when
    terminal is `None`, `connected=False`, `account_info()` is `None`, or
    an exception is raised (never propagates).
  - `stream_prices()` backoff: doubles each consecutive-error iteration
    (1, 2, 4, 8s with jitter forced to 0 via monkeypatched
    `random.uniform`); caps at 60s after enough doublings; resets to the
    normal 0.1s poll once connectivity is restored mid-stream.
- `tests/test_collector_connection.py` (7 tests)
  - Stays up / no alerts while connected.
  - Detects disconnect → `broker_down=True`, `mt5_disconnected` fired
    exactly once even across repeated checks (idempotency).
  - Recovery → symbol resync called for every enabled instrument,
    `broker_down=False`, `mt5_reconnected` fired.
  - Never raises when `client.is_broker_connected()` or the Telegram alert
    call itself raises.
  - `broker_down` defaults to `False` before any check.
- `tests/test_telegram_mt5_alerts.py` (6 tests)
  - Each of the three methods sends the expected message content.
  - Each of the three methods swallows an exception raised by `_send()`
    without propagating.

**RED/GREEN evidence:** implementation and tests were developed together
(mt5_client + collector + telegram changes, then tests written against
the new methods). Verified GREEN:

```
$ python -m pytest tests/test_mt5_client_connection.py tests/test_collector_connection.py tests/test_telegram_mt5_alerts.py -q
.....................                                                    [100%]
21 passed in 6.74s

$ python -m pytest tests/ -x -q
.....................                                                    [100%]
21 passed in 1.11s
```

(Note: only files tracked in git under `tests/` in this worktree/branch are
the three new files above — pre-existing `test_capital_rebase.py` /
`test_safety_floor.py` visible in the main checkout are untracked in this
repo's git history and are not present in the worktree. `python -m pytest
tests/ -x -q` therefore covers exactly the new test files; no pre-existing
tracked tests exist to regress.)

## Self-review

- Scope: only touched `src/data/mt5_client.py`, `src/data/collector.py`,
  `src/monitoring/telegram_bot.py`, and new test files. Did not touch
  `main.py`, `executor.py`, or `risk/`.
- Instrument naming: `_resync_symbols()` iterates
  `config.get_enabled_instruments()` (underscore format, e.g. `EUR_USD`)
  and delegates the underscore→MT5-format conversion to
  `MT5Client._to_mt5_symbol`, consistent with the existing boundary rule.
- Telegram alerts are best-effort at two layers: `_send()`'s own internal
  try/except, plus a wrapping try/except in each new alert method, plus
  the collector's own try/except around the call site — belt and braces
  per the "never raises into caller" requirement.
- Backoff formula matches the brief exactly: 1s → 2s → 4s → ... capped at
  60s, jitter added, reset on success.
- Logging uses `logging.getLogger("traderbot.<module>")` hierarchy
  (unchanged loggers reused).

## Concerns / follow-ups for other agents or later tasks

1. **Main-loop wiring is not done** — `collector.broker_down` exists and is
   correctly maintained, but nothing currently reads it to gate new-entry
   approval. This needs to be wired into `main.py`'s trading loop (or the
   risk manager's checklist) by whichever task owns that file.
2. **Task 6 dependency**: `_resync_symbols()` currently calls
   `MT5Client._to_mt5_symbol()` directly (a "private" method) as instructed
   by the brief's fallback note, since Task 6's dedicated suffix-detection/
   cache function doesn't exist yet. If Task 6 introduces a public
   `detect_symbol_suffix()`-style API, `_resync_symbols()` should be
   updated to call that instead.
3. **Health-check thread overhead**: `is_broker_connected()` is called
   both once per ~100ms inside `stream_prices()` (to drive backoff) and
   once per ~5s from the collector's independent health-check thread. Both
   are cheap MT5 API calls (`terminal_info()` + `account_info()`), but if
   this ever shows up as measurable overhead, the two checks could be
   consolidated (e.g. collector reads a cached "last known connected"
   state from the client instead of polling MT5 again).
4. No changes were made to `config/settings.yaml` — the health-check
   interval (5s) is currently a hardcoded collector attribute
   (`_health_check_interval_seconds`), not config-driven. Flagging in case
   the team wants it tunable.
