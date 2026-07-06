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

## Fix Round 1

Fixes for the 5 review findings, applied on top of commit `bb1e901`.

### Finding 1 — interruptible backoff (Important)
- `MT5Client.__init__`: added `self._stream_cancelled = False`.
- New `MT5Client.cancel_stream()`: sets `self._stream_cancelled = True`. Safe
  to call anytime, including when no stream is running.
- New `MT5Client._interruptible_sleep(total_seconds, chunk=0.1)`: sleeps in
  `chunk`-sized steps (mirrors `DataCollector._health_loop`'s existing
  0.1s-chunked pattern), rechecking `self._stream_cancelled` between steps
  so a multi-second backoff sleep can be cut short almost immediately.
- `stream_prices()`: loop condition changed from `while True` to
  `while not self._stream_cancelled`; both the backoff sleep and the normal
  0.1s poll sleep now go through `_interruptible_sleep()` instead of raw
  `time.sleep()`. `self._stream_cancelled` is reset to `False` at the start
  of every `stream_prices()` call.
- `DataCollector.stop_streaming()`: now calls `self.client.cancel_stream()`
  in addition to flipping `self._streaming = False`, so an in-progress
  backoff sleep is interrupted right away instead of being waited out.
- Tests added: `test_interruptible_sleep_stops_promptly_on_cancel` (direct,
  no real sleeping — fake `time.sleep` cancels after 2 chunks, asserts only
  2 chunks ran and `sum(sleep_calls) < 1.0` even though a full 60s sleep was
  requested) and `test_stream_prices_backoff_is_interruptible_via_cancel_stream`
  (same assertion driven through the actual generator, expects `StopIteration`
  once cancelled).
- Existing backoff tests (`test_stream_prices_backs_off_exponentially_on_disconnect`,
  `..._backoff_capped_at_60s`, `..._resets_backoff_on_success`) were updated to
  monkeypatch the new `client._interruptible_sleep` seam instead of raw
  `time.sleep`, so they keep asserting the exact backoff progression
  (1.0, 2.0, 4.0, 8.0, ... capped at 60.0) without being coupled to the 0.1s
  chunk size.

### Finding 2 — flap debounce (Medium)
- Mechanism chosen: **require 2 consecutive agreeing health checks** before
  flipping `broker_down` in either direction (not the 60s-min-interval
  alternative) — simpler, deterministic, and easy to unit test without fake
  clocks.
- `DataCollector.__init__`: added `self._flap_debounce_count = 2`,
  `self._consecutive_failures = 0`, `self._consecutive_successes = 0`.
- `check_connection()`: increments/resets the two counters every call, and
  only flips `broker_down` (± firing the Telegram alert / resync) once the
  relevant counter reaches `_flap_debounce_count`. Documented in the
  docstring.
- Tests updated: `test_check_connection_detects_disconnect_and_pauses_entries`
  and `test_check_connection_never_raises_on_client_exception` /
  `..._never_raises_when_telegram_fails` now call `check_connection()` twice
  and assert no flip/alert after the first (flaky) call.
  `test_check_connection_recovery_resyncs_symbols_and_resumes` now drives 2
  failures to go down and 2 successes to come back up, asserting the
  alert/resync only fires on the 2nd success.
  New test `test_check_connection_single_flaky_failure_does_not_flip_or_alert`
  covers a single bad check followed by a good one never flipping
  `broker_down` or firing any alert.

### Finding 3 — probe-on-error only (Low)
- `stream_prices()`: `self.is_broker_connected()` is no longer called
  unconditionally every iteration. It's now called only inside the
  `if loop_had_error:` branch, purely as a diagnostic log line — it can no
  longer independently set `loop_had_error` when the tick loop itself
  didn't error.
- Test mocks reworked: `_setup_stream_mocks()` now simulates a real
  disconnect as `symbol_info_tick` raising an exception (the actual signal
  that drives `loop_had_error`) rather than relying on the probe returning
  `False` while ticks silently return `None`.
- New test `test_stream_prices_ignores_probe_blip_when_ticks_healthy`:
  ticks never error but `is_broker_connected()` reports `False` throughout —
  asserts the poll stays at the normal 0.1s interval with no backoff growth.

### Finding 4 — single-writer docstring (Low)
- `DataCollector.check_connection()` docstring now has an explicit
  "IMPORTANT — single-writer" paragraph stating it mutates `broker_down` and
  the new debounce counters without a lock and must only ever be called
  from the health-check thread (`_health_loop`).

### Finding 5 — thread lifecycle test (Gap)
- New test `test_start_stop_streaming_lifecycle` in
  `tests/test_collector_connection.py`: mocks `client.stream_prices` with a
  fast fake generator (no real network/backoff), starts streaming, asserts
  both threads are alive/daemon, asserts a second `start_streaming()` call
  is a no-op (guarded, same thread objects), then calls `stop_streaming()`
  and asserts `client.cancel_stream()` was called once and both threads
  join within 2s (real-time join, since the mocked stream has no backoff to
  wait out).

### Test run
```
python -m pytest tests/test_mt5_client_connection.py tests/test_collector_connection.py tests/test_telegram_mt5_alerts.py -q
```
→ `26 passed in 1.11s`

Full suite (`python -m pytest -q` from the worktree root — this worktree's
`tests/` directory only contains the Task 3 files above, so it's the same
26 tests): `26 passed in 0.80s`.

### Deliberately not changed
- Did not implement the "min 60s between alerts" debounce alternative —
  picked consecutive-check debounce instead (see Finding 2 above).
- Did not touch `main.py` wiring, Task 6's symbol-suffix TODO, or the
  health-check-interval config-driven suggestion — all out of scope for
  this review round.
