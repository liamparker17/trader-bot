# Task 8 Report — File-based control queue (`tb` CLI ↔ bot)

Status: **DONE**. Full test suite green: 176 passed (142 pre-existing + 34 new).
Commits: `6855e1d`, `2f1dea0`, `9b225c0` on top of `5f2aee5` (no new branch).

Note: `.superpowers/sdd/task-8-brief.md` referenced in the task instructions
did not exist in this worktree (only task-3..7 reports were present). Worked
from the verbatim brief embedded in the orchestrator's task message instead.

## What was implemented

### 1. `src/control/queue.py` (new) — `ControlQueue`
- Polls `control/inbox/*.cmd.json` (relative to `PROJECT_ROOT`) once per
  `poll_once()` call, oldest-first (sorted by `requested_at` then filename).
  `Path.glob("*.cmd.json")` never matches a `*.cmd.json.tmp` file, so a
  command still mid-write under its `.tmp` name is invisible until the
  writer's `os.replace()` makes it a real `.cmd.json` — this is the
  atomicity guarantee, verified by `test_stray_tmp_file_never_picked_up`.
- Verbs: `pause`, `resume`, `tune`, `revert`, `status_snapshot`.
  - `pause`/`resume`: reason ≥10 chars mandatory; delegates to
    `RiskManager.set_manual_pause()`/`clear_manual_pause()` (new methods,
    see below). Falls back to an internal flag if no `risk_manager` is wired
    (standalone queue testing).
  - `tune`: whitelist + bounds exactly per brief
    (`risk.risk_per_trade_pct` [0.5,2.5], `ml.confidence_threshold_high`
    [0.50,0.75], `ml.confidence_threshold_low` [0.45,0.65],
    `weight.<INSTRUMENT>` [0.0,1.5] validated against
    `config.instruments["instruments"]` keys). Rejects on: unknown/
    non-whitelisted key, out-of-bounds value, `EffectiveConfig.is_safety_locked()`,
    or a low>high cross-check (compares the tuned key's value against the
    OTHER threshold key's *current effective* value when only one side is
    being tuned). Rate limit: 1 applied manual tune (`requested_by != "manager"`)
    per rolling 24h, queried via `TradeJournal.get_control_log(verb="tune",
    outcome="applied")` and filtered/compared client-side; `requested_by ==
    "manager"` is exempt.
  - `revert`: finds the most recent `control_log` row with
    `verb='tune', outcome='applied'`, re-applies its `before_config_json`
    values via `EffectiveConfig.apply_tune()`.
  - `status_snapshot`: read-only — **no** `control_log` row and **no**
    Telegram message (per brief: Telegram is only for write verbs). Best-effort
    snapshot with documented `None` defaults for anything not derivable from
    the wired-in modules (see `_build_status_snapshot()` docstring/comments).
- Telegram: best-effort (`try/except`, swallowed) on every write verb — one
  message on receipt, one on outcome.
- Timestamps: tz-aware UTC throughout, via an injectable `clock` callable
  (same pattern as `DailySummaryScheduler`/`RiskManager`).

### 2. `src/monitoring/trade_journal.py` — `control_log` table
Added the table (`id, ts_utc, verb, args_json, reason, requested_by,
before_config_json, after_config_json, outcome`) to `_init_db()`, plus three
helper methods following the `record_event`/`get_events` conventions:
`log_control_command()` (returns row id, accepts an optional `ts_utc` so
`ControlQueue`'s injectable clock — not wall-clock `datetime.now()` — drives
the timestamp used for rate-limit comparisons in tests),
`update_control_outcome()` (pending → applied/rejected/error, `COALESCE`s
before/after JSON so a pause/resume update that doesn't pass them doesn't
null out anything), and `get_control_log()` (filtered query, most recent first).

### 3. `src/control/effective_config.py` — `apply_tune()`
Read-modify-write against `TUNES_PATH` (already `data/control/effective_config.json`
— reused as-is, not diverged): reads existing overlay JSON, deep-merges in
the new dotted key (via new `_expand_dotted()` helper), atomic
write-tmp-then-`os.replace()`, and also updates the instance's in-memory
`_data` so a caller holding a reference sees the change immediately. A
fresh `EffectiveConfig.load()` (simulating a restart or a separate `tb`
process) picks it up from disk. `apply_tune()` does **no** validation
itself — `ControlQueue` is the sole enforcement point for whitelist/bounds/
safety-lock/rate-limit, keeping single responsibility.

### 4. `src/main.py` wiring
- `setup()`: `self.effective_config = EffectiveConfig.load()`; wired into
  `self.risk_manager.sizer.effective_config` and `self.predictor.effective_config`;
  `self.control_queue = ControlQueue(...)` constructed after `self.executor`
  (collector already exists by that point in `setup()`'s existing ordering).
- `run()`'s loop: `self.control_queue.poll_once()` called near the top,
  inside its own `try/except` (logs + continues, never touches the generic
  catch-all).
- `_on_candle_complete()`: returns early (debug log) when
  `self.collector.broker_down`, before `_evaluate_trade_signal()` runs.
- Extracted two loop-body blocks into testable methods:
  - `_refresh_balance_cache(cached_balance, cached_equity)` — same 5s-cached
    refresh as before; now tracks `self._balance_refresh_failures`, escalating
    from `logger.debug` to `logger.warning` after 3 consecutive failures,
    reset to 0 on success. Returns `(balance, equity)`, unchanged on failure
    so callers keep the last-known-good values.
  - `_check_session_and_drawdown(cached_balance, cached_equity)` — the
    existing `check_session_boundary`/`check_drawdown_emergency` logic,
    called from `run()` only when `cached_balance is not None and not
    self.collector.broker_down` (stale-data guard). Wrapped in its OWN
    `try/except`, distinct from `_handle_loop_exception`'s generic
    catch-all: on exception, calls `self.risk_manager.set_manual_pause(...)`
    and sends a dedicated Telegram alert (both best-effort/non-raising).
- `RiskManager` (`src/risk/manager.py`): added `_manual_pause_reason` state,
  `manual_paused` / `manual_pause_reason` properties, `set_manual_pause()`/
  `clear_manual_pause()`, and a new `evaluate_trade()` gate ("Check -1",
  before the existing session-boundary check) that rejects with `"Manually
  paused: <reason>"` while active. This is independent of `entries_blocked`
  (daily-drawdown, auto-lifted at session boundary) and of the circuit
  breaker's shutdown/`force_resume` — it persists until explicitly cleared.
- Read-through at the whitelisted tune sites:
  - `src/risk/position_sizer.py`: `PositionSizer.effective_config` (default
    `None`) read at the top of `calculate()` for the *global*
    `risk.risk_per_trade_pct` fallback, before any per-instrument override
    in `instruments.yaml` (which still wins, unchanged behavior).
  - `src/ml/predictor.py`: `Predictor.effective_config` (default `None`)
    read at the top of the threshold comparisons inside `get_signal()` for
    `ml.confidence_threshold_high`/`low`.
  - Both default to `None` and fall back to the config-cached value from
    `__init__`, so every pre-existing caller/test is unaffected until
    `main.py`'s `setup()` wires the real `EffectiveConfig` in.
  - `weight.<INSTRUMENT>` has **no runtime consumer anywhere in the
    codebase** (confirmed via grep) — the whitelist/bounds validation in
    `ControlQueue` is implemented and tested, but there is nothing yet that
    *reads* `weight.<INSTRUMENT>` to affect trading. Documented as a
    forward-looking hook (see Concerns below).

## Files touched

- `src/control/queue.py` (new, ~330 lines)
- `src/control/effective_config.py` (+`_expand_dotted`, `apply_tune`, `import os`)
- `src/monitoring/trade_journal.py` (`control_log` table in `_init_db` ~L100-113;
  `log_control_command`/`update_control_outcome`/`get_control_log` ~L275-337)
- `src/risk/manager.py` (`_manual_pause_reason` init ~L143-150; Check -1 in
  `evaluate_trade()` ~L242-246; `manual_paused`/`manual_pause_reason`/
  `set_manual_pause`/`clear_manual_pause` ~L403-424)
- `src/risk/position_sizer.py` (`effective_config` attr in `__init__`;
  `global_risk_pct` read-through before `inst_risk_pct` calc)
- `src/ml/predictor.py` (`effective_config` attr in `__init__`; threshold
  read-through at top of `get_signal()`)
- `src/main.py` (imports; `__init__` new attrs `effective_config`,
  `control_queue`, `_balance_refresh_failures`; `setup()` wiring; `run()`
  loop changes; `_on_candle_complete()` broker_down gate; new
  `_refresh_balance_cache()`/`_check_session_and_drawdown()` methods)
- Tests (new): `tests/test_control_queue.py` (20 tests),
  `tests/test_effective_config_tune.py` (3 tests),
  `tests/test_main_control_wiring.py` (11 tests)

## Test summary

`python -m pytest tests/ -x -q` → **176 passed** (142 pre-existing + 34 new),
0 failures, run in ~3.7s. Ran the full suite after each of the 3 commits to
confirm green before committing.

New-test coverage maps directly to the brief's list: full round-trip per
verb, out-of-bounds rejection, safety-locked rejection, manual-vs-manager
rate limit (both sides: 2nd manual within 24h rejected, allowed after 24h,
manager exempt), revert restores prior value (and rejects with nothing to
revert), atomicity (`.tmp` file ignored), control_log lifecycle
(pending→applied/rejected/error), plus main.py's broker_down gate,
balance-refresh escalation, and pause-on-exception.

## Design decisions / deviations from the brief (with rationale)

1. **Inbox/outbox path**: used `PROJECT_ROOT / "control" / "inbox"` and
   `.../"outbox"` (top-level `control/`, sibling to `src/`), distinct from
   `data/control/effective_config.json` (the pre-existing `TUNES_PATH`).
   The brief's own wording is inconsistent (`control/inbox/...` vs
   `control/effective_config.json`) — I kept `TUNES_PATH` exactly as it
   already existed (per the brief's explicit instruction to reuse it, not
   diverge) and used a fresh `control/` directory for the command channel,
   since inbox/outbox is a different concern (transient command files, not
   persisted config state) from the overlay JSON.
2. **`before_config_json` recorded at outcome-update time, not at receipt**:
   `log_control_command()` writes the `pending` row before validation runs,
   so the pre-tune value isn't known yet; `_handle_tune()` computes it and
   passes it to `update_control_outcome(before_config_json=..., after_config_json=...)`
   in the same call as the outcome. Simpler than a second UPDATE and avoids
   a window where `before_config_json` is transiently wrong.
3. **`ts_utc` is now an optional param on `log_control_command()`**, sourced
   from `ControlQueue`'s injectable `clock` rather than always
   `datetime.now(timezone.utc)`. Needed this to make the 24h rate-limit
   window testable without real sleeps — otherwise the rate-limit check
   (which uses the same injected clock for its cutoff) would compare
   against wall-clock-stamped rows and never line up in tests.
4. **Extracted `_refresh_balance_cache()` and `_check_session_and_drawdown()`
   as methods** rather than leaving the logic inline in `run()`'s `while`
   loop. Not explicitly requested, but required for TDD coverage of item
   (e)/(f) in the brief — inline code inside a `while self.running:` loop
   with real `time.sleep(1)` isn't unit-testable. Followed the existing
   `_bare_bot()` / `TraderBot.__new__(TraderBot)` pattern from
   `test_daily_summary_scheduler.py`.
5. **`weight.<INSTRUMENT>` has no runtime consumer** (see above) — the
   validation/whitelist machinery is fully implemented and tested in
   `ControlQueue`, but tuning it currently has no observable effect on
   trading since nothing in `position_sizer.py`/`main.py` reads a
   per-instrument weight today. Left as-is per the brief's own instruction
   ("don't invent data you can't get") — flagging for whoever adds
   instrument-weight-based sizing/allocation later.
6. **`manual_paused` on `RiskManager` is separate from `entries_blocked`**:
   `entries_blocked` is daily-drawdown-specific and auto-lifts at the next
   session boundary (existing behavior, untouched); `manual_paused` is a
   distinct switch that persists until an explicit `resume` command (or
   code fix), matching the brief's "Sets/clears a manual-pause flag" wording
   for the queue's pause/resume verbs. `evaluate_trade()` checks both
   (manual pause first, "Check -1"), so either one blocks new entries.
7. **Did not touch circuit_breaker.py / drawdown_tracker.py hardcoded
   safety-floor fallbacks (600/0.35)** — brief explicitly said leave these.
   Confirmed no edits made to those files.

## Concerns / TODOs for later tasks

- **Task 11 (`src/ai/` removal)**: `src/main.py` still imports and wires
  `AIAnalyst`/`ShadowTrader`/`ApprovalQueue` from `src/ai/` — untouched by
  this task, as instructed (only borrowed *ideas* from
  `approval_queue.py`'s sqlite/dataclass patterns, never imported it). When
  Task 11 deletes `src/ai/`, those three import lines and the
  `self.analyst`/`self.approval_queue`/`self.shadow` wiring in `setup()`/`run()`
  will need removal — unrelated to this task's new code, which doesn't
  reference any `src/ai/` symbol.
- **`tb` CLI (writer side) not implemented here** — out of scope per the
  brief ("Writer... not your concern here"). It needs to: generate a unique
  `id`, write `<id>.cmd.json.tmp` then `os.replace()` to `<id>.cmd.json` in
  `control/inbox/`, and poll `control/outbox/<id>.result.json` for the
  result. The exact JSON shape it must produce is documented in
  `src/control/queue.py`'s module docstring.
- **`weight.<INSTRUMENT>` runtime wiring** — flagged above; whoever adds
  instrument-level position-size weighting should read it through
  `EffectiveConfig` the same way `risk.risk_per_trade_pct` is now read in
  `position_sizer.py`.
- **`status_snapshot`'s `todays_pnl`** sums `net_pnl_zar` over
  `journal.get_trades(since=<today's date string>)` — this is a best-effort
  approximation (string-prefix comparison against `entry_time`, matching
  the existing `get_trades(since=...)` query semantics elsewhere in the
  codebase); it is not equity-curve-accurate intraday P&L, just realized
  trade P&L for the day so far.
- **Control queue directories (`control/inbox/`, `control/outbox/`) are
  created on `ControlQueue.__init__`** (`mkdir(parents=True, exist_ok=True)`)
  but are not currently in `.gitignore`. Given the recent `chore: gitignore
  .worktrees/ and data/control runtime files` commit pattern, the `tb` CLI
  task (or a follow-up) should add `control/inbox/` and `control/outbox/`
  to `.gitignore` alongside the existing `data/control/` entry, so runtime
  command/result files don't get committed.

## Fix Round 1

Status: **DONE**. Full suite green: 184 passed (176 baseline + 8 new).

### Finding 1 — Crash-replay idempotency
- `src/monitoring/trade_journal.py`: added an idempotent migration
  `_migrate_control_log_columns()` (PRAGMA table_info guard, same pattern as
  `_migrate_fee_columns`) that adds `control_log.cmd_id TEXT`. `log_control_command()`
  now accepts and persists `cmd_id`. Added `get_control_log_by_cmd_id(cmd_id)` —
  returns the most recent row for that id, or `None`.
- `src/control/queue.py`: `_process()` now calls `_replay_outcome(cmd_id)` before
  any dispatch. If `cmd_id` already has a terminal `control_log` row
  (`outcome in (applied, rejected)`), the verb is **not** re-executed — the
  outbox result is written straight from the logged outcome (`detail` explains
  it's a replay) and the inbox file is deleted. No new `control_log` row, no
  second call into `_handle_tune`/`_handle_pause`/etc., so no double quota
  consumption of the manual-tune rate limit or double-apply of a tune/pause.
- Post-dispatch steps (`update_control_outcome`, outbox write, inbox unlink)
  are now wrapped individually: a failure updating `control_log` logs ERROR
  but still proceeds to write the outbox result (best-effort delivery to the
  CLI); a failure writing the outbox result logs ERROR and returns *without*
  unlinking the inbox file — leaving it in place so the next poll retries, at
  which point the replay guard (control_log already terminal) skips
  re-execution and just retries the outbox write. Unlink is always the last
  step, matching the "consistent state" requirement.
- Tests: `test_replay_after_crash_does_not_reexecute`,
  `test_replay_of_rejected_command_stays_rejected` — process, delete the
  outbox result, re-drop the identical inbox file, assert exactly one
  `control_log` row for the command and no re-application.

### Finding 2 — cmd_id sanitization
- `src/control/queue.py`: added `CMD_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")`.
  `_process()` validates the id (from JSON `id`, falling back to the filename
  stem) before it's used anywhere. An invalid id is rejected via the new
  `_reject_invalid_id()`: logs ERROR, best-effort logs a `control_log` row
  with `outcome='rejected'` under a freshly generated safe id
  (`invalid-<uuid4 hex[:12]>`), writes the outbox result under that safe id
  (never the attacker-controlled string), and deletes the inbox file.
- Tests: `test_invalid_cmd_id_path_traversal_rejected` (`id: "../../etc/passwd"`
  — asserts nothing escapes `outbox_dir`, outbox result id is not the raw
  string, control_log row is rejected) and
  `test_invalid_cmd_id_missing_falls_back_to_safe_id` (malformed filename stem
  and no `id` field at all).

### Finding 3 — Poison-pill corrupt JSON
- `src/control/queue.py`: `_read_inbox_sorted()` now routes a JSON-decode
  failure to `_deadletter()` instead of `continue`-ing forever. `_deadletter()`
  logs one WARNING, attempts to salvage an id from the filename (validated
  against `CMD_ID_RE`) to write a best-effort `outcome='error'` outbox result,
  then `os.replace()`s the file into `control/deadletter/<name>.dead` (dir
  created in `__init__`, alongside inbox/outbox). Once moved, the file no
  longer matches the inbox glob, so it is never retried/re-warned.
- Tests: `test_corrupt_inbox_json_moved_to_deadletter_not_retried` (moved once,
  second `poll_once()` is a no-op, best-effort outbox failure result present)
  and `test_corrupt_inbox_json_only_warns_once` (3 polls of the same corrupt
  file -> exactly 1 "deadletter" WARNING via caplog).

### Finding 4 — Dead `drawdown_vs_cap` status field
- `src/control/queue.py`: `_build_status_snapshot()` now populates
  `drawdown_vs_cap` from `risk_manager.drawdown.get_daily_drawdown_pct(balance)`
  divided by `risk_manager.drawdown.daily_limit` (both already exposed by
  `DrawdownTracker`), only when `balance` was resolved from the client; stays
  `None` otherwise (documented "never invented" pattern already used
  elsewhere in this method).
- Tests: extended with `test_status_snapshot_drawdown_vs_cap_populated`
  (0.02 daily dd / 0.04 cap -> 0.5) and
  `test_status_snapshot_drawdown_vs_cap_null_when_balance_unknown` (no client
  wired -> balance and drawdown_vs_cap both `None`). Added `FakeDrawdownTracker`
  and `FakeClient` test doubles.

### Test commands + output
```
python -m pytest tests/test_control_queue.py tests/test_main_control_wiring.py -q
.......................................                                  [100%]
39 passed in 2.78s

python -m pytest tests/ -x -q
........................................................................ [ 39%]
........................................................................ [ 78%]
........................................                                 [100%]
184 passed in 4.17s
```

### Deliberately unchanged
- No changes outside `src/control/queue.py`, `src/monitoring/trade_journal.py`,
  and `tests/test_control_queue.py` (constraint: nothing else touched).
- `control_log` gained only `cmd_id`; no `detail`/message column was added —
  replay outbox `detail` is synthesized (`"replay: command already processed
  (outcome=...)"`) rather than replaying the original human-readable detail
  text, since that text was never persisted and the brief didn't ask for a
  schema change beyond `cmd_id`.
