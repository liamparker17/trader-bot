# Task 12 Report — Claude-manager core: policy + briefing + manager_log

## Status: DONE

## Merge
`git merge 680e891 --no-edit` applied cleanly at start (fast-forward-ish merge bringing in
`src/control/`, `src/risk/ratchet_floor.py`, `src/utils/`, and ~20 test files from prior tasks).
Baseline after merge: 190 tests passing.

## What was built

### 1. `src/monitoring/trade_journal.py` (commit `0e9d88b`)
- New `manager_log` table, columns exactly as specified: `id, ts_utc, trigger, briefing_json,
  model, input_tokens, output_tokens, cost_zar, rationale, proposals_json, applied_json,
  rejected_json, outcome`.
- `_migrate_manager_log_columns()` — no-op placeholder today, added purely to match the
  established `_migrate_control_log_columns` / `_migrate_fee_columns` idempotent-migration
  convention for future column additions.
- `log_manager_cycle(trigger, briefing=None, model="", input_tokens=None, output_tokens=None,
  cost_zar=None, rationale="", proposals=None, applied=None, rejected=None, outcome="",
  ts_utc=None) -> int` — inserts one row, JSON-serializing dict/list args, returns row id.
- `get_manager_log(days=None, limit=None) -> pd.DataFrame` — most-recent-first, optional
  rolling-window (`days`) and row-count (`limit`) filters.
- `manager_cost_since(ts) -> float` — sums `cost_zar` (NULL treated as 0) for rows at/after
  `ts` (accepts `datetime` or ISO string).

### 2. `src/manager/policy.py` (commit `c0201ac`)
- `LEVERS = dict(TUNE_BOUNDS)` — imports `TUNE_BOUNDS` / `WEIGHT_BOUNDS` from
  `src.control.queue` rather than redefining bounds (single source of truth, per brief).
- `risk_ceiling_now(balance, milestones) -> float` — growth-stage ladder: below the first
  milestone → 1.5; then 1.8 / 2.0 / 2.2 / 2.5 at each subsequent milestone; hard-capped at 2.5
  always (`RISK_CEILING_HARD_CAP`).
- `growth_stage(balance, milestones) -> int` — count of milestones reached/passed.
- `validate_and_clamp(proposals, effective_config, risk_ceiling_now) -> (applied, rejected)`:
  - Caps at 3 proposals/cycle; 4th+ rejected outright with `rejection_reason:
    "cycle_limit_exceeded"`.
  - Unknown key / unknown `weight.<INSTRUMENT>` instrument (checked against
    `config/instruments.yaml`, path overridable via `policy.INSTRUMENTS_PATH` for tests) →
    rejected (`"unknown_key"` / `"bad_instrument"`).
  - Non-numeric value → rejected (`"non_numeric"`).
  - Safety-locked key (`effective_config.is_safety_locked`) → rejected (`"safety_locked"`).
  - In-bounds values pass through unclamped (`clamped: False`); out-of-bounds values clamp to
    the nearest bound (`clamped: True`).
  - `risk.risk_per_trade_pct`'s effective upper bound is `min(static_bound_max, risk_ceiling_now)`.
  - `ml.confidence_threshold_low <= ml.confidence_threshold_high` enforced pair-wise against
    the *resulting* config (proposed+clamped value where proposed, else the current
    `effective_config` value for the untouched side); a violation rejects both proposed sides
    (`"threshold_invariant"`) rather than silently reordering or applying one.

### 3. `src/manager/briefing.py` (commit `7c8475e`)
- `build(journal, effective_config, ratchet_floor, balance, equity, extra=None) -> dict`.
- Every data source wrapped in a `_safe()` helper that degrades to null/empty/default and logs
  a warning instead of raising — evaluator table missing, model_store missing, journal empty,
  etc. all produce a valid (if sparse) briefing.
- Fields: `generated_at_utc, balance, equity, floor, headroom_to_floor, today_pnl_zar,
  drawdown_vs_cap {today_drawdown_pct, cap_pct}, instruments {<INSTRUMENT>: {trades, win_rate,
  profit_factor, net_pnl_zar, current_weight}}, open_positions, config_delta (tunes overlay
  JSON), model_version ("unknown" if no `model_store/latest_version.txt`), recent_accuracy
  (None if no `evaluator_trades` table/rows), growth_stage, risk_ceiling_now, milestones
  ([{milestone, reached}]), last_manager_actions (up to 5, from `manager_log`).
- Size guard: `_enforce_size_cap()` truncates `open_positions` / `last_manager_actions`
  (newest-first lists — truncation drops from the tail, i.e. keeps the newest entries) in a
  loop until the JSON-serialized briefing is ≤ `MAX_BRIEFING_CHARS` (16000, ≈4k tokens), then
  hard-asserts the cap is met.
- `extra` dict, when provided, is merged in as `briefing["extra"]`; omitted key entirely when
  not passed.

## Tests
- `tests/test_manager_log.py` — 7 tests (table columns, idempotent migration, CRUD, days/limit
  filters, cost summation with NULL handling, datetime/string `ts` acceptance).
- `tests/test_manager_policy.py` — 34 tests (LEVERS/WEIGHT_BOUNDS identity with queue.py,
  ceiling ladder below/at/between/above every milestone incl. hard cap, growth_stage per
  milestone, clamp matrix below/at/above bounds for every lever incl. risk-ceiling override,
  unknown key / bad instrument / safety-locked / non-numeric rejection, 4th-proposal rejection,
  threshold-pair invariant both-proposed and one-side-vs-current-effective-value cases).
- `tests/test_manager_briefing.py` — 15 tests (schema keys, degraded-empty-journal case, model
  version present/absent, recent_accuracy absent, growth_stage/risk_ceiling_now consistency,
  milestones reached flags, per-instrument weight default, floor/headroom, last-5-actions
  behavior + 5-item cap, extra merge/omission, size-cap assertion, truncation-keeps-newest with
  400 synthetic open positions).

`python -m pytest tests/ -x -q` → **246 passed** (190 baseline + 56 new), no regressions.

## Self-review
- No Anthropic SDK / network calls anywhere in `src/manager/` — pure functions + sqlite, as
  required (Task 13 owns the client).
- `LEVERS`/`WEIGHT_BOUNDS` are imported, not redefined, from `src.control.queue` — verified via
  `test_levers_reuses_tune_bounds` / `test_weight_bounds_identity` so the two gates can't drift.
- All timestamps tz-aware UTC (`datetime.now(timezone.utc)`), matching the codebase convention.
- Ran the full suite (not just the new files) before every commit; three separate commits
  (journal table, policy, briefing) per the "may split" guidance.
- Fixed a pandas `FutureWarning` (`fillna` on object-dtype column) in `briefing.py` by casting
  to float first — avoids relying on soon-deprecated implicit downcasting.

## Concerns / things Task 13 (or a reviewer) should know
- `validate_and_clamp`'s threshold-pair rejection, when both `threshold_low` and
  `threshold_high` are proposed together and violate the invariant, rejects **both** rather
  than trying to salvage one — this seemed like the safer behavior given no tie-breaking rule
  was specified, but a reviewer may want the manager to instead retry with adjusted values in
  a later cycle rather than losing both proposals.
- `briefing.py`'s "config delta from baseline" is implemented as the raw tunes-overlay JSON
  (`data/control/effective_config.json`), not a computed diff against `settings.yaml` baseline
  values — since the overlay *is* the delta by construction (only tuned keys are ever written
  there), this should be equivalent, but flagging the interpretation in case Task 13 expected a
  different shape (e.g. `{key: {before, after}}`).
- Per-instrument stats and open-positions queries scan the full `trades` table with a large
  `limit`/no limit; fine at current bot scale (single demo account, low trade volume) but could
  be optimized with narrower SQL (`WHERE exit_price IS NULL` already pushed down for open
  positions; 7-day stats rely on `journal.get_trades()`'s existing `since` filter).

## Worktree
`C:\Users\liamp\Desktop\Portfolio\TraderBot\.claude\worktrees\agent-af47d0f89884220a9`
Branch: `worktree-agent-af47d0f89884220a9`
