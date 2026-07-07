# Task 7 Report — Fixes R + K (journal fees, evaluator checkpoint persistence)

## Scope
Per controller adjustment: only (R) journal fee/swap columns + net P&L math, and (K)
evaluator checkpoint persistence. (S) exception alerting was reassigned to Task 5 and
explicitly skipped. Did not touch main.py, telegram_bot.py, executor.py, or mt5_client.py.

## (R) `src/monitoring/trade_journal.py`
- `trades` table gains `commission REAL DEFAULT 0`, `swap REAL DEFAULT 0`,
  `net_pnl_zar REAL` columns.
- New `_migrate_fee_columns(conn)`: guards each `ALTER TABLE ... ADD COLUMN` behind a
  `PRAGMA table_info(trades)` check, so it's a no-op on already-migrated or
  freshly-created DBs, and safely backfills legacy DBs that predate Task 7.
  Called from `_init_db()` on every journal open.
- `record_trade(...)` gained `commission: float = 0.0, swap: float = 0.0` kwargs
  (defaulted so existing call sites in main.py, which this task must not touch,
  keep working unchanged). It now stores commission/swap and computes
  `net_pnl_zar` via the new `TradeJournal.compute_net_pnl` staticmethod when
  `pnl_zar` is not None (still-open trades get `net_pnl_zar = NULL`).
- **Sign convention (documented in code + brief):** MT5 deal objects report
  `commission` and `swap` as already-signed values — in virtually all cases both
  are costs and therefore negative. Net P&L = `gross_pnl + commission + swap`,
  which naturally reduces the gross result for the common case and would only
  increase it for the rare positive-swap (carry) case.

## Fix Round 1

**Review finding:** `evaluator_state` only persisted the scalar retrain counters
(`trades_since_retrain`, `last_retrain_time`, `model_version`). The `self.trades`
deque (maxlen=5000) that powers the win-rate degradation branch of
`should_retrain()` was never persisted — after a restart it started empty, so the
win-rate trigger was silently disabled until `win_rate_lookback` (default 100) new
trades had accumulated in memory again. Brief item (K) requires retrain triggers
to survive process death; this branch didn't.

**Change summary (`src/ml/evaluator.py` only, + tests):**
- New `evaluator_trades` table (`CREATE TABLE IF NOT EXISTS`, idempotent, created
  in `_init_state_table()` alongside `evaluator_state`): columns `id` (autoincrement),
  `prediction`, `predicted_action`, `actual_outcome`, `pnl`, `timestamp` — the exact
  fields `TradeRecord` needs to be reconstructed.
- `record_trade()` now calls `self.save_state(new_trade=trade)` instead of
  `save_state()`.
- `save_state(new_trade=None)`: within the **same** `sqlite3.connect()`
  block/transaction as the existing `evaluator_state` upsert, if `new_trade` is
  given it inserts the row into `evaluator_trades`, then prunes to
  `max(win_rate_lookback, 500)` most-recent rows (`DELETE ... WHERE id NOT IN
  (SELECT id ... ORDER BY id DESC LIMIT ?)`). `should_retrain()`'s win-rate branch
  only ever reads the last `win_rate_lookback` trades, so there's no need to
  persist the full 5000-deep in-memory deque — this avoids unbounded DB growth
  while keeping identical trigger behavior. `mark_retrained()`'s existing
  `save_state()` call (no `new_trade`) is unaffected.
- `load_state()`: in addition to the existing scalar-counter load, now also
  `SELECT`s all rows from `evaluator_trades` ordered `id ASC` (oldest → newest)
  and rebuilds `self.trades` as a `deque(..., maxlen=self.trades.maxlen)` from
  them, so `should_retrain()` sees the identical window pre/post restart.
- Both reads (`evaluator_state` row + `evaluator_trades` rows) happen inside one
  `sqlite3.connect()` block in `load_state()`, matching the "one connection per
  call" constraint.
- Deliberately unchanged: `_init_state_table()`'s idempotency guarantee, the
  `evaluator_state` schema/upsert, `main.py`'s existing `self.evaluator.load_state()`
  call site (untouched, out of scope), and the 5000-deep in-memory deque size
  (only the *persisted* window is capped, not the live `self.trades`).

**Tests added** (`tests/test_evaluator_persistence.py`):
- `test_win_rate_trigger_survives_restart`: seeds `win_rate_lookback` (100)
  losing trades on one `Evaluator` (confirms `should_retrain()` trips with a
  "Win rate trigger" reason), then constructs a **fresh** `Evaluator` against the
  same tmp-path SQLite DB. Confirms the trigger is *not* active before
  `load_state()` (nothing in memory yet) and *is* active immediately after
  `load_state()` — proving the window round-trips and the branch behaves
  identically across a simulated restart. Real sqlite, tz-aware UTC via
  `TradeRecord`'s default timestamp, no mocks.
- `test_win_rate_trigger_control_fresh_db_does_not_trip`: control case — a brand
  new DB with nothing persisted must not spuriously trip the trigger after
  `load_state()`.

**Test command + output:**
```
python -m pytest tests/test_evaluator_persistence.py tests/test_journal_fees.py -q
...............
15 passed in 1.12s

python -m pytest tests/ -x -q
........................................................................ [ 80%]
..................                                                       [100%]
90 passed in 1.64s
```
(88 pre-existing + 2 new = 90, all green.)

## (K) `src/ml/evaluator.py`
- Evaluator's `save_state()`/`load_state()` used to persist to a private
  `data/trade_logs/evaluator_state.json` file. Replaced with a dedicated
  `evaluator_state` SQLite table in the **same DB file** as the trade journal
  (`monitoring.trade_journal_db` config key), so state truly lives alongside
  the journal per the brief ("persist evaluator state via the journal... or a
  dedicated table").
- `_init_state_table()`: idempotent `CREATE TABLE IF NOT EXISTS evaluator_state`
  with a single-row (`id=1`, `CHECK (id = 1)`) schema — called from `__init__`.
- `save_state()`: upsert (`INSERT ... ON CONFLICT(id) DO UPDATE`) — always
  exactly one row, no duplicate/growing rows across repeated saves.
- `load_state()`: reads the row if present; no-op (leaves defaults) if the DB
  has no persisted state yet (first run).
- Persistence is now automatic on **every update**, not just explicit calls:
  `record_trade()` and `mark_retrained()` both call `self.save_state()`
  internally, on top of main.py's existing explicit `load_state()` (post-init)
  and `save_state()` (periodic) calls — so retrain-trigger counters survive a
  crash between explicit checkpoints, not just a clean restart.
- `Evaluator.__init__(config)` signature unchanged (still just `config`), and
  `load_state()`/`save_state()` remain zero-arg — main.py's existing call sites
  (`self.evaluator.load_state()` at init, `self.evaluator.save_state()` later)
  needed no changes.
- Removed now-dead `json` import (was only used by the old JSON-file
  save/load path).

## Tests (TDD — written first, confirmed red, then green)
- `tests/test_journal_fees.py` (7 tests): fresh-DB columns present; migration
  idempotent on re-open of an already-migrated DB; migration correctly
  backfills a legacy DB missing the columns; `record_trade` defaults
  commission/swap to 0 and computes net P&L; net P&L math with negative
  fees; net P&L stays NULL for a still-open trade; `compute_net_pnl`
  staticmethod math (including a losing-trade case).
- `tests/test_evaluator_persistence.py` (6 tests): `evaluator_state` table
  created idempotently across repeated constructions; full save/load
  round-trip simulating a restart; load on an empty/fresh DB is a no-op;
  `record_trade` persists state automatically (visible to a fresh instance
  without an explicit `save_state()`); `mark_retrained` persists
  automatically; `save_state()` upserts (one row) rather than duplicating.

## Verification
- `python -m pytest tests/test_journal_fees.py tests/test_evaluator_persistence.py -q`
  → 13 passed.
- `python -m pytest tests/ -x -q` → 88 passed (75 pre-existing + 13 new), all green.

## Self-review
- Scope held: only `trade_journal.py`, `evaluator.py`, and two new test files
  touched. No changes to main.py/telegram_bot.py/executor.py/mt5_client.py.
- Did not implement (S) exception alerting — explicitly out of scope per
  controller reassignment to Task 5.
- Left the stale, untracked `data/trade_logs/evaluator_state.json` file
  (leftover from a prior run under the old JSON-based path, or another
  agent's session) alone — it's untracked, unused by the new code, and
  deleting files outside this task's scope wasn't requested.
- Did not surface `net_pnl_zar` in the Streamlit dashboard or
  `performance.py` — out of scope for this task (brief only covers journal
  schema/math + evaluator persistence); flagging as a natural Task 8+ follow-up.

## Concerns / follow-ups for controller
- `evaluator.py`'s `Path` import was already unused before this task (dead
  import, pre-existing) — left untouched per "fix only what was reported."
- Whoever wires real MT5 deal `commission`/`swap` into `journal.record_trade(...)`
  calls in main.py (not this task, per scope) should double check the sign
  convention documented above against their actual `mt5.history_deals_get`
  reads before trusting `net_pnl_zar` values in production.
