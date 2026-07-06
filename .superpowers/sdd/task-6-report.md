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
