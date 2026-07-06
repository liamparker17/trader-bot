# Task 6 Report — MT5 client robustness (Fixes L/N/P/O/M)

## Summary
Implemented all five fixes from the brief plus the controller's scope
adjustment for fill validation. Merged Tasks 1–4 (`cd17344`) into the
worktree first via fast-forward (no conflicts).

## Changes

### (L) Symbol-suffix re-detection on reconnect
`src/data/mt5_client.py`:
- Added `MT5Client._symbol_cache: dict[str, str]`, populated lazily by
  `_to_mt5_symbol()`.
- `_to_mt5_symbol()` now returns the cached mapping instead of re-probing
  the broker on every call.
- New `invalidate_symbol_cache()` clears the cache; called automatically
  at the top of `connect()` (covers automatic reconnects via
  `ensure_connected()`), and callable directly for an explicit resync.
- `src/data/collector.py`: `_resync_symbols()` now calls
  `self.client.invalidate_symbol_cache()` before its existing
  per-instrument `_to_mt5_symbol()` loop, so the DataCollector's
  reconnect path forces a real re-probe instead of returning a cached
  (potentially stale) suffix. This was the only collector edit made, as
  instructed.

### (N) Deviation scales with spread
- New `MT5Client._compute_deviation(symbol, tick)`:
  `max(20, ceil(current_spread_points * 1.5))`, using
  `symbol_info(symbol).point` to convert price spread to points. Falls
  back to 20 if point/tick info is unavailable.
- Used in `place_market_order()` for both the initial attempt and the
  retry attempt (fresh spread on retry). `close_trade()` was left
  untouched (hard-coded 20) — out of scope per the brief, which only
  calls out "order deviation" and whose tests target order placement.

### (P) Retcode retry
- Module-level `RETRYABLE_RETCODES = {10004, 10021, 10020}` (REQUOTE,
  PRICE_OFF, PRICE_CHANGED).
- `place_market_order()`: if the first `order_send()` result's retcode is
  in that set, fetches a fresh tick, recomputes price + deviation, and
  retries exactly once. Any retcode outside the set, or a second failure
  after the retry, raises `MT5Error` as before (no further retries).

### (O) Fill validation (scope-adjusted into mt5_client.py)
- New `MT5Client._validate_fill(symbol, expected_volume, result)`: after
  a `TRADE_RETCODE_DONE` result, checks `result.price`/`result.volume`
  are present and `> 0`. If either is missing/invalid, immediately polls
  `mt5.positions_get(ticket=result.order)` (not waiting for the 60s
  reconcile loop) and repairs the fill from the position's
  `price_open`/`volume`. If the position can't be found either, falls
  back to price `0.0`/expected volume rather than raising, so the trade
  (which likely did fill) isn't lost from the return path.
- `place_market_order()`'s return dict gained an additive `"units"` key
  (the validated fill volume) inside `orderFillTransaction`; existing
  keys (`price`, `tradeOpened.tradeID`, `id`) are unchanged in shape, so
  `executor.py` (not touched, per instructions) continues to work
  unmodified — confirmed by reading `executor.py`'s consumption of the
  response (`fill.get("price")`, `fill.get("tradeOpened", {}).get("tradeID")`,
  `fill.get("id")`).

### (M) Timestamp hygiene
- New `src/utils/timeutil.py::to_utc(dt_or_epoch)`: accepts int/float
  epoch seconds or an already timezone-aware `datetime`
  (`pandas.Timestamp` included, since it subclasses `datetime`); raises
  `ValueError` (and logs) on naive datetimes or unsupported types.
  Explicitly rejects `bool` (since `bool` is an `int` subclass in
  Python).
- Applied at every MT5 ingestion point in the files I own:
  - `src/data/mt5_client.py`: `get_open_positions()`, `get_candles()`,
    `get_candles_batch()`, `get_current_price()`, `stream_prices()`
    (all previously did `datetime.fromtimestamp(x, tz=timezone.utc)`),
    and `_parse_time()` now routes a passed-in `datetime` through
    `to_utc()` instead of returning it unchecked.
  - `src/data/candle_builder.py`: `on_tick()` (parsed ISO tick time) and
    `load_historical()` (DataFrame index timestamps).
  - `src/data/historical_loader.py`: `_candles_to_dataframe()` (raw MT5
    candle `time` string → `pd.Timestamp`).
  - Verified all call sites feeding these paths already produce
    tz-aware UTC datetimes (historical_loader always emits UTC-aware
    parquet caches; DataCollector always sources `load_historical()`
    input from `fetch_and_cache()`), so no naive-datetime regressions
    were introduced by tightening this.

## Tests
New files:
- `tests/test_timeutil.py` (9 tests) — epoch conversion, aware
  passthrough/conversion, naive rejection, `pandas.Timestamp` handling,
  bool/unsupported-type rejection.
- `tests/test_mt5_client_orders.py` (13 tests) — deviation floor/scaling/
  ceiling, deviation used in the real request, retryable retcodes
  (parametrized 10004/10021/10020) retry-once-then-succeed, retry-once-
  then-hard-fail (exactly 2 `order_send` calls, never 3), non-transient
  retcode hard-fails with zero retries, fill validation pass-through /
  price-repair / volume-repair / fallback-when-position-not-found.
- `tests/test_mt5_client_symbol_cache.py` (6 tests) — caching after first
  detection, caching a suffix variant, `invalidate_symbol_cache()`
  forcing re-detection, re-detection picking up a *changed* suffix
  across a simulated reconnect, `connect()` invalidating the cache, and
  `DataCollector._resync_symbols()` invalidating the cache before its
  existing `_to_mt5_symbol` loop.

All mock `MetaTrader5` entirely via `monkeypatch.setattr(mt5_client_module, "mt5", ...)` —
no real terminal/network touched.

`python -m pytest tests/ -x -q` → **103 passed** (75 pre-existing + 28
new), run repeatedly while iterating and once more before commit.

## Self-review notes
- Confirmed `executor.py` and `main.py` were not touched (grep + diff
  stat show only `src/data/mt5_client.py`, `src/data/candle_builder.py`,
  `src/data/historical_loader.py`, `src/data/collector.py`, new
  `src/utils/timeutil.py`, and new test files).
- The one collector edit is minimal (one `invalidate_symbol_cache()`
  call + docstring update) and doesn't change any other collector
  behavior; the existing collector reconnect test
  (`tests/test_collector_connection.py::test_check_connection_recovery_resyncs_symbols_and_resumes`)
  still passes unmodified because `_to_mt5_symbol` is still called with
  the same plain positional args it asserts on.
- Considered also hardening `close_trade()`/`modify_trade()` with the
  same deviation/retry logic for consistency, but left them untouched —
  the brief's explicit test scenarios and scope adjustment center on
  order placement, and touching more call sites than required risks
  scope creep per the global "fix only what was reported" rule.
- `_validate_fill()`'s "position not found" fallback returns price
  `0.0` rather than raising, on the reasoning that the order likely did
  fill (retcode was DONE) and losing the trade/ticket info from the
  return path would be worse for downstream risk tracking than a
  best-effort record with a placeholder price; this is a judgment call
  worth a second look if reviewed.

## Fix Round 1

Review flagged the `0.0` fallback above as a live-account defect and two
smaller findings. All three addressed in `src/data/mt5_client.py` plus
`tests/test_mt5_client_orders.py`.

### 1. (Blocking) `_validate_fill` unrepairable path returned `price=0.0`
- `_validate_fill()` now takes a new required `request_price` parameter
  — the pre-order tick price (`tick.ask`/`tick.bid`) already computed in
  `place_market_order()` before `order_send()`. Return type changed from
  `tuple[float, float]` to `tuple[float, float, bool]`, the third value
  being `fill_price_estimated`.
- When both the order result and the `positions_get(ticket=...)` poll
  fail to yield a usable price/volume, the method now falls back to
  `request_price` (never `0.0`/negative) and the caller's
  `expected_volume`, logs at WARNING with the ticket + reason, and
  returns `fill_price_estimated=True`.
- If `request_price` itself is missing/`<= 0` (or the expected volume is
  `<= 0`), `_validate_fill` now raises `MT5Error` instead of returning
  garbage — a loud hard-fail rather than a silently corrupted entry
  price.
- `place_market_order()` passes `request_price=price` (the same local
  `price` used to build the order request) into `_validate_fill`, and
  the returned `fill_price_estimated` flag is added to the response as
  `orderFillTransaction["fill_price_estimated"]`. `executor.py` was not
  touched — its `fill.get("price", entry_price)` consumer is unchanged
  and now always receives a `> 0` price.
- Bug found and fixed while wiring this up: on the transient-retcode
  retry path, `request["price"]` was refreshed with a new tick but the
  local `price` variable used later for `request_price` was not —
  fixed by reassigning `price` alongside `request["price"]` on retry, so
  the fallback always reflects the actual last price sent to
  `order_send()`.

### 2. (Minor) `close_trade()` hard-coded `deviation=20`
- `close_trade()` now computes `tick = mt5.symbol_info_tick(symbol)` (as
  before) and calls `self._compute_deviation(symbol, tick)` to build the
  `deviation` field, matching `place_market_order()`'s spread-aware
  behavior instead of a fixed `20`.

### 3. Test corrections/additions in `tests/test_mt5_client_orders.py`
- Rewrote `test_fill_repair_falls_back_when_position_not_found`: no
  longer asserts `price == "0.0"` (the defect). Now asserts the fill
  price equals the pre-order request price (`"1.10001"`, the BUY-side
  tick ask), the volume falls back to the expected order volume
  (`"0.1"`), and `orderFillTransaction["fill_price_estimated"] is True`.
- Added `test_fill_repair_never_returns_nonpositive_price_or_volume`,
  parametrized over four unrepairable scenarios (`price=volume=0.0`,
  `price=volume=None`, negative price, negative volume) — all assert
  the final response's price and units are `> 0`.
- Added `test_fill_repair_raises_when_request_price_unavailable`: tick
  bid/ask both `0.0` and result/position poll both fail to repair ->
  `place_market_order()` raises `MT5Error` instead of returning a
  garbage fill.
- Added `test_close_trade_uses_computed_deviation`: mocks a wide spread
  (30 points) and asserts the `order_send` request sent by
  `close_trade()` carries `deviation == 45` (matching
  `_compute_deviation`'s formula), not the old hard-coded `20`.

### Test runs
- `python -m pytest tests/test_mt5_client_orders.py tests/test_mt5_client_symbol_cache.py tests/test_timeutil.py -q`
  → **34 passed**.
- `python -m pytest tests/ -x -q` → **109 passed** (103 prior + 6 new:
  4 parametrized non-positive cases + 1 request-price-unavailable case +
  1 close_trade deviation case; the rewritten fallback test replaces an
  existing test rather than adding one).

### Deliberately unchanged
- `executor.py` — off-limits per brief; its `.get("price", entry_price)`
  consumer needed no change since it now always receives a sane,
  positive price.
- `modify_trade()` — not mentioned in the findings; still untouched,
  consistent with the original scope decision.
