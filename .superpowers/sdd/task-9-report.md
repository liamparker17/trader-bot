# Task 9 — `tb` CLI — Implementation Report

## Summary
Implemented `cli/tb.py` (+ `cli/__init__.py`) as a pure-consumer CLI for TraderBot's
control plane (Task 8's `ControlQueue`), trade journal, effective-config overlay,
ML model store, and log file. No `src/` files were modified.

## Merge
Merged `a1959ec` (Tasks 1-8, control plane) into the worktree branch via
`git merge a1959ec --no-edit` — fast-forward, no conflicts.

## Files
- `cli/__init__.py` — package marker.
- `cli/tb.py` — argparse CLI, `python -m cli.tb <cmd>`.
- `tests/test_cli_tb.py` — 25 tests.

## Commands implemented

### Read
- `status` — enqueues `status_snapshot`, polls outbox up to 5s (0.1s interval,
  both overridable for tests). On a result, returns the snapshot dict +
  `bot_running: true` + `generated_at`. On timeout (bot not running / not
  draining the inbox), degrades to a journal-derived status:
  `{"bot_running": false, "generated_at", "todays_pnl", "open_positions"}`,
  where `open_positions` is trades with `exit_price IS NULL` and `todays_pnl`
  sums today's `net_pnl_zar` (falls back to `pnl_zar`).
- `trades [--days N]` — `journal.get_trades(since=cutoff)`, JSON-safe via
  `df.to_json(orient="records")` round-trip (handles NaN/NaT -> null cleanly).
- `perf [--days N]` — computed directly (not via `PerformanceTracker`, which has
  no date-filter support) from filtered completed trades: total/wins/losses,
  win_rate, profit_factor (None if no losses), total_pnl, gross_profit/loss.
- `positions` — journal-derived open positions (no live MT5 access from the CLI).
- `config` — `EffectiveConfig.load()` vs baseline `settings.yaml`, delta over
  `TUNE_BOUNDS` keys + `weight.<instrument>` for each configured instrument,
  plus the safety-locked key list.
- `logs [--tail N] [--level L]` — tails `data/logs/traderbot.log`
  (`| LEVEL |` substring filter matching main.py's log format); empty list if
  the file doesn't exist yet.
- `model` — reads `model_store/latest_version.txt` + `model_<version>_meta.json`;
  additionally reads `evaluator_state` (id=1 row) guarded by a table-existence
  check (works whether or not the bot has run yet).
- `manager [--days N] [--verdict]` — `--verdict` returns the required stub
  `{"verdict": "PENDING", "reason": "manager not yet active"}`. Otherwise reads
  `manager_log` guarded by a table-existence check — returns `{"count": 0,
  "entries": []}` until Task 12 creates that table.

### Write
`pause`/`resume --reason "..."`, `tune key=value --reason "..."`, `revert
--reason "..."` — each validates `--reason` client-side (>=10 chars, matching
`MIN_REASON_LEN` imported from `src.control.queue`) *before* touching the
filesystem, then enqueues `{id, verb, args, reason, requested_at,
requested_by}` via tmp-write + `os.replace` into `control/inbox/`, and polls
`control/outbox/<id>.result.json` (5s/0.1s, overridable). A bot-side rejection
(e.g. out-of-bounds tune) still prints the JSON result but exits 1 (only
`outcome == "applied"` exits 0). A missing responder (bot not running) times
out and reports `{"error": "timed out ... is the bot running?"}` + exit 1.

Note: `--reason` is deliberately **not** `argparse required=True` — a missing
reason needs to surface as our own `{"error": ...}` JSON on stdout, not
argparse's stderr usage text + `exit(2)`.

## Design decisions / deviations
- `perf` reimplements the summary computation rather than reusing
  `PerformanceTracker.get_summary()`, because that method has no `--days`
  filtering (it always uses `get_all_trades_df()`). No `src/` changes needed
  since the computation is simple and journal-only.
- `manager` and `model`'s `evaluator_state` read use raw `sqlite3` against
  `journal.db_path` (an existing public attribute) with a
  `sqlite_master` table-existence guard, rather than adding new `TradeJournal`
  methods — kept the "CLI is a pure consumer, don't touch src/" constraint
  literally. No missing read API was needed.

## Verification
- `python -m pytest tests/test_cli_tb.py -x -q` → 25 passed.
- `python -m pytest tests/ -x -q` → **209 passed** (184 pre-existing + 25 new),
  fully green.
- Manual smoke test against the real repo (`python -m cli.tb status/manager
  /config/pause --reason short/tune bad_format ...`) confirmed correct
  degraded-status output, safety-locked key listing, and JSON error + exit 1
  paths with no bot running.

## Self-review
- All commands print exactly one JSON document to stdout; all failure paths
  (validation, timeout, unexpected exception) are caught at the `main()`
  boundary and reported as `{"error": ...}` — no bare traceback can leak to
  stdout.
- Timestamps are UTC tz-aware ISO (`datetime.now(timezone.utc).isoformat()`).
- No `sleep()` beyond the 0.1s poll interval in production paths; tests use a
  shorter injectable `timeout` for the one true-timeout test (0.3s) to keep
  the suite fast without waiting the full 5s.
- Command ids are `uuid.uuid4().hex` (regex-validated against `CMD_ID_RE`
  before use) — matches the writer contract in `src/control/queue.py`.
