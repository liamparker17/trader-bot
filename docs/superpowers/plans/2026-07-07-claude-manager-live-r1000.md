# Implementation Plan: Claude Portfolio Manager + Live R1000

**Spec:** `docs/superpowers/specs/2026-07-07-claude-manager-live-r1000-design.md` (read for rationale; this plan is self-contained for execution)
**Branch:** `feat/claude-manager-live-r1000`
**Baseline already present:** `src/control/effective_config.py` (EffectiveConfig: `load()`, `get(dotted_key)`, `is_safety_locked(dotted_key)`, `safety_keys`), `config/safety_floor.yaml`, ZAR config keys, `tests/test_safety_floor.py`.

## Global Constraints (bind every task)

- **Currency is ZAR everywhere.** Keys use `_zar` suffix where absolute amounts appear.
- **Instrument names**: `EUR_USD` style internally; MT5 format only inside `mt5_client.py`.
- **All timestamps tz-aware UTC.**
- **Logging**: `logging.getLogger("traderbot.<module>")`.
- **Config access** via dot-notation `config.get("a.b.c")`; runtime tunes overlay via EffectiveConfig; `safety_floor.yaml` always wins.
- **The trading session day boundary is 21:00 UTC** (`trading.session_reset_hour_utc: 21` exists in settings.yaml).
- **Claude manager may only write through the control queue** (same path as `tb tune`). No privileged interface.
- **Manager levers + hard bounds** (exact values):
  - `weight.<INSTRUMENT>` ∈ [0.0, 1.5] (default 1.0; INSTRUMENT must exist in instruments.yaml)
  - `risk.risk_per_trade_pct` ∈ [0.5, 2.5]
  - `ml.confidence_threshold_high` ∈ [0.50, 0.75]
  - `ml.confidence_threshold_low` ∈ [0.45, 0.65], and must be ≤ threshold_high
- **Ratcheting hard floor** (safety_floor.yaml): `floor_zar = max(min_floor_zar, high_water_mark * (1 - max_total_drawdown_pct))` with `min_floor_zar: 600`, `max_total_drawdown_pct: 0.35`. Floor never decreases. High-water mark persists across restarts (SQLite `events`/state table or JSON state file `data/account_state.json`).
- **Milestones**: `[1500, 2000, 3000, 4500, 6000]`, `target_balance: 6000`, `starting_balance: 1000`.
- **Per-cycle manager limits**: ≤3 adjustments/cycle, ≥20 min between cycles, ≤20 cycles/day.
- **Tests**: pytest, files under `tests/`, follow `tests/test_safety_floor.py` style. Every task adds/updates tests and runs its own test files plus `python -m pytest tests/ -x -q`.
- **No network in unit tests** (mock Anthropic/MT5/Telegram).
- **Windows environment** — no POSIX-only APIs (e.g., use `msvcrt`/atomic file create for locks, not fcntl).

## Task List

### Task 1 — Capital re-base for R1000 + ratcheting floor config
**Files:** `config/settings.yaml`, `config/safety_floor.yaml`, `src/control/effective_config.py` (only if key-shape change needed), `tests/test_safety_floor.py`, new `tests/test_capital_rebase.py`
1. settings.yaml: `account.starting_balance: 1000`, `account.currency: "ZAR"`, `growth.milestones: [1500, 2000, 3000, 4500, 6000]`, `growth.target_balance: 6000`. Keep `risk.risk_per_trade_pct` at its current value.
2. safety_floor.yaml: replace `hard_floor_zar: 9000` with:
   ```yaml
   risk:
     min_floor_zar: 600
     max_total_drawdown_pct: 0.35
   ```
   (keep daily_drawdown_stop_pct, max_leverage_effective, circuit_breaker block).
3. New module `src/risk/ratchet_floor.py`: class `RatchetFloor(min_floor_zar, max_total_drawdown_pct, state_path="data/account_state.json")` with `update(balance) -> float` (updates high-water mark, persists atomically, returns current floor), `current_floor` property, `is_breached(equity) -> bool`. Floor monotonically non-decreasing. State file JSON: `{"high_water_mark": float, "updated_utc": iso}`. Handles missing/corrupt state file by seeding HWM = max(balance, starting HWM 1000).
4. Grep all readers of `hard_floor_zar`/`hard_floor` and update them to use RatchetFloor (circuit_breaker.py currently reads a hard floor — wire it there; keep its kill-switch semantics: breach → emergency stop).
5. Tests: floor math (HWM 1000 → floor 650; HWM 2000 → 1300; never below 600; never decreases when balance falls), persistence round-trip, corrupt-file recovery, breach detection.

### Task 2 — Blocker A: single-instance lock
**Files:** `src/main.py`, new `src/utils/instance_lock.py` (or inline in main), `tests/test_instance_lock.py`
1. Acquire an exclusive lock at startup (`data/traderbot.lock`) using atomic O_CREAT|O_EXCL create with PID inside + stale-lock detection (if PID not running, reclaim; use `psutil` if already a dep, else ctypes/OpenProcess or tasklist check). Windows-safe.
2. On failure: log CRITICAL, exit code 1, no MT5 connection attempted.
3. Release on clean shutdown (existing signal handlers).
4. Tests: acquire, second-acquire fails, stale lock reclaimed, release allows re-acquire.

### Task 3 — Blockers B1/B2/B3 + I: MT5 disconnect handling + alerts
**Files:** `src/data/mt5_client.py`, `src/data/collector.py`, `src/monitoring/telegram_bot.py`, tests
1. `MT5Client.is_broker_connected() -> bool`: cheap health probe (terminal_info().connected AND account_info() is not None), called once per main-loop iteration.
2. `stream_prices()` loop: on repeated errors, exponential backoff with jitter (1s → 2s → 4s… cap 60s), reset on success. No unbounded 100ms spam.
3. Collector: on detected disconnect → set `self.broker_down = True`, pause new-entry signals (main loop checks this), Telegram alert `mt5.disconnected`; on recovery → re-detect symbol suffix (see Task 6 note — if Task 6 not yet done, call existing detection function), Telegram `mt5.reconnected`, resume.
4. Telegram: add alert methods for connect/disconnect/reconnect; best-effort (never raises into caller).
5. Tests: mock mt5 module; simulate disconnect → entries paused + alert fired; reconnect → resumed + alert.

### Task 4 — Blockers D/E/F: drawdown emergency close + 21:00 UTC session resets
**Files:** `src/risk/drawdown_tracker.py`, `src/risk/manager.py`, `src/risk/circuit_breaker.py`, `src/execution/executor.py`, tests
1. Daily drawdown breach (4%): risk manager triggers `executor.close_all_positions(reason="daily_drawdown")` (new method: closes every open position at market via MT5Client, logs each, Telegram alert), then blocks new entries until next session boundary.
2. Daily reset boundary: 21:00 UTC (read `trading.session_reset_hour_utc`), not midnight. Applies to daily drawdown counters AND consecutive-loss counter in circuit_breaker.
3. Weekly drawdown reset: week boundary = Friday 21:00 UTC.
4. Tests: breach → close_all called + entries blocked; counters reset exactly at 21:00 UTC crossing (freeze time via injected clock or monkeypatched now()); consecutive losses reset at boundary.

### Task 5 — Blockers G/Q + H: trade-ID integrity + daily summary scheduler
**Files:** `src/execution/executor.py`, `src/main.py`, `src/monitoring/telegram_bot.py`, tests
1. Remove timestamp fallback for trade IDs: if MT5 order result lacks a ticket, treat as failed order → log ERROR, Telegram alert, do NOT record a synthetic trade. Position/journal records must key on real MT5 ticket.
2. Daily summary: main loop fires `telegram.daily_summary()` once per day at 21:00 UTC (guard against double-fire via last-fired date persisted in memory + journal event row).
3. Tests: missing ticket → no journal row + alert; summary fires exactly once per boundary crossing.

### Task 6 — Fixes L/N/P/O/M: MT5 client robustness
**Files:** `src/data/mt5_client.py`, `src/execution/executor.py`, `src/data/historical_loader.py`, `src/data/candle_builder.py`, tests
1. (L) Symbol-suffix detection: re-run on every reconnect, not just startup.
2. (N) Order deviation: `max(20, ceil(current_spread_points * 1.5))` instead of hard-coded 20.
3. (P) Retcode handling: retry once (fresh price) on transient retcodes REQUOTE(10004)/PRICE_OFF(10021)/PRICE_CHANGED(10020); all others still hard-fail.
4. (O) Fill validation: after order_send DONE, validate returned volume/price present and sane; if fields missing, poll positions for the ticket immediately rather than waiting for the 60s reconcile.
5. (M) Timestamp hygiene: single helper `to_utc(dt_or_epoch)` used at every MT5 ingestion point; reject/log naive datetimes.
6. Tests: mocked mt5 — suffix re-detection on reconnect; deviation scales with spread; requote retries once then succeeds/fails; naive timestamp raises/converts.

### Task 7 — Fixes R/S/K: journal fees, exception alerting, ML checkpoints
**Files:** `src/monitoring/trade_journal.py`, `src/main.py`, `src/ml/evaluator.py`, `src/monitoring/telegram_bot.py`, tests
1. (R) `trades` table: add `commission REAL DEFAULT 0`, `swap REAL DEFAULT 0` columns (ALTER TABLE migration on open if missing); populate from MT5 deal info on close; net P&L includes them.
2. (S) Main-loop catch-all: unhandled exception in an iteration → log traceback + Telegram `bot.error` alert (rate-limited to 1/5min per exception type), continue loop (don't die silently).
3. (K) Evaluator: persist live-accuracy counters to journal (events table or dedicated table) every update so retrain triggers survive restarts.
4. Tests: migration adds columns idempotently; net P&L math; exception alert rate-limiting; evaluator state round-trip.

### Task 8 — Control queue + control_log + bot integration
**Files:** new `src/control/queue.py` (repurpose logic from `src/ai/approval_queue.py` if useful, else fresh), `src/monitoring/trade_journal.py` (control_log table), `src/main.py`, tests
1. File-based queue: writer drops `control/inbox/<id>.cmd.json.tmp` then atomic-renames to `.cmd.json`. Command: `{id, verb, args, reason, requested_at, requested_by}`.
2. Bot polls inbox once per main-loop iteration: process oldest; execute; write `control/outbox/<id>.result.json` `{id, outcome, detail, applied_at}`; delete inbox file.
3. Verbs: `pause`, `resume`, `tune`, `revert`, `status_snapshot` (writes full status JSON to outbox — this is how tb reads live state).
4. `tune`: one dotted key + value. Reject if: key not in whitelist (Global Constraints), value out of bounds, key safety-locked, or `threshold_low > threshold_high` would result. Apply to EffectiveConfig overlay + persist `control/effective_config.json`.
5. **Rate limit:** manual tunes (`requested_by` != "manager") limited 1 per rolling 24h; manager tunes limited by manager-side caps instead.
6. `control_log` table in journal.db: `id, ts_utc, verb, args_json, reason, requested_by, before_config_json, after_config_json, outcome`. Row written pending → updated applied/rejected/error. `revert` re-applies `before_config_json` of last applied tune.
7. Telegram on every write command (request + outcome), best-effort.
8. Tests: full round-trip on tmp dirs; out-of-bounds rejected; safety-locked rejected; rate limit; revert; atomicity (no partial reads of .tmp).

### Task 9 — `tb` CLI
**Files:** new `cli/__init__.py`, `cli/tb.py`, tests
1. `python -m cli.tb <cmd>` prints JSON to stdout, exit 0/1.
2. Read cmds: `status` (via status_snapshot round-trip w/ 5s timeout; if bot not running, degrade to journal-derived status + `"bot_running": false`), `trades [--days N]`, `perf [--days N]`, `positions`, `config`, `logs [--tail N] [--level L]`, `model`, `manager [--days N]` (reads manager_log; empty list fine before Task 12).
3. Write cmds: `pause/resume --reason "..."` (reason ≥10 chars), `tune key=value --reason "..."`, `revert` — all enqueue via Task 8 queue and wait ≤5s for outbox result.
4. Tests: each command against a seeded journal + fake outbox responder thread.

### Task 10 — Per-instrument weight in position sizing
**Files:** `src/risk/position_sizer.py`, `src/risk/manager.py`, tests
1. Read `weight.<INSTRUMENT>` from EffectiveConfig (default 1.0). `risk_amount *= weight` before size calc; then all existing clamps (leverage cap, min/max size, round DOWN) still apply.
2. `weight == 0.0` → risk manager rejects entry for that instrument ("muted by weight").
3. Tests: weight scales size; 0 mutes; 1.5 still respects leverage cap; unknown instrument key rejected at tune time (Task 8 whitelist validates instrument exists).

### Task 11 — AI scaffolding cleanup
**Files:** `src/main.py`, delete `src/ai/*`, move `TRADING_BRAIN.md` → `docs/personas/trading-brain.md`, `CLAUDE.md`
1. Unwire analyst from main.py (currently wired ~lines 373-383): remove import + call sites; trading decisions flow purely ML+risk.
2. Delete `src/ai/analyst.py`, `shadow_trader.py`, `prompts.py`, `approval_queue.py`, `__init__.py` (after Task 8 has harvested anything useful).
3. `git mv TRADING_BRAIN.md docs/personas/trading-brain.md` (if file exists at root).
4. Update CLAUDE.md architecture tree: remove src/ai, add src/control, cli/, src/manager (placeholder note).
5. Tests: full suite green; grep confirms no `src.ai` imports remain.

### Task 12 — Manager core: policy + briefing + manager_log
**Files:** new `src/manager/__init__.py`, `src/manager/policy.py`, `src/manager/briefing.py`, journal manager_log table, tests
1. `policy.py`: `LEVERS` dict (exact bounds from Global Constraints). `validate_and_clamp(proposals, effective_config, risk_ceiling_now) -> (applied, rejected)`; numeric out-of-bounds → clamp to nearest bound (record `clamped: true`); unknown key / bad instrument / safety-locked → reject with reason; enforce ≤3 per cycle (keep first 3, reject rest); enforce threshold_low ≤ threshold_high pair-wise against resulting config; `risk_per_trade_pct` additionally capped at `risk_ceiling_now`.
2. `risk_ceiling_now(balance, milestones) -> float`: growth-stage ladder — below 1500: 1.5; ≥1500: 1.8; ≥2000: 2.0; ≥3000: 2.2; ≥4500: 2.5. (Hard cap 2.5 regardless.)
3. `briefing.py`: `build(journal, effective_config, ratchet_floor, balance) -> dict` — compact JSON: balance, equity, floor, headroom to floor, today P&L, drawdown vs 4% cap, per-instrument last-7d stats (trades, win rate, PF, net_pnl_zar, current weight), open positions summary, current config delta from baseline, model version + recent accuracy, `growth_stage`, `risk_ceiling_now`, milestones state, last 5 manager actions. Must not exceed ~4k tokens (truncate trade lists).
4. `manager_log` table: `id, ts_utc, trigger, briefing_json, model, input_tokens, output_tokens, cost_zar, rationale, proposals_json, applied_json, rejected_json, outcome`.
5. Tests: clamp matrix (each lever: below/at/above bounds), 4th proposal rejected, threshold pair invariant, ceiling ladder, briefing schema + token budget (rough char cap), manager_log CRUD.

### Task 13 — Manager client + scheduler + runner
**Files:** new `src/manager/client.py`, `src/manager/scheduler.py`, `src/manager/runner.py`, `requirements.txt` (+anthropic), settings.yaml `manager:` block, tests
1. settings.yaml new block:
   ```yaml
   manager:
     enabled: true
     model: "claude-opus-4-8"
     cycle_minutes: 60
     min_gap_minutes: 20
     max_cycles_per_day: 20
     max_adjustments_per_cycle: 3
     event_triggers: { drawdown_pct: 2.0, consecutive_losses: 3, circuit_breaker: true }
     usd_zar_rate: 18.0        # for cost accounting only
   ```
2. `client.py`: Anthropic SDK; system prompt (write it: role = bounded portfolio risk manager for a scalping bot; explain levers, bounds, briefing schema; instruct conservatism, prefer no-op when uncertain; forbid exceeding bounds); one user message = briefing JSON; `tools=[propose_adjustments]` with `tool_choice={"type":"tool"}` forcing the call; schema: `{adjustments: [{key, value, reason}], rationale}`; parse tool_use block; return `(proposals, rationale, usage)`. Retries: 3 with exp backoff on APIError/timeout; on give-up raise `ManagerAPIUnavailable`.
3. Cost: `cost_zar = (input_tokens * in_price + output_tokens * out_price) * usd_zar_rate` — put Opus 4.8 per-token USD prices in one constants dict with a comment to verify against current pricing.
4. `scheduler.py`: computes next timer fire (only within enabled session hours from settings), listens for event flags (file `control/manager_events/*.json` dropped by the bot on drawdown>2%/3 losses/breaker trip — add those drops in risk manager/circuit breaker with a tiny helper), enforces min-gap + daily cap (count from manager_log, day = 21:00 UTC boundary).
5. `runner.py`: `python -m src.manager` loop: wait for next trigger → build briefing → client call → policy validate → enqueue surviving tunes on control queue (requested_by="manager") → write manager_log row (outcome: applied/no_op/api_unavailable/error) → Telegram summary (`[MANAGER] cycle: 2 applied, 1 clamped — rationale…`). All exceptions caught → log + Telegram, never crash loop.
6. Event-drop helper wired: bot writes event files at the three trigger points.
7. Tests: mocked Anthropic client (no network): full cycle happy path; API-down → no_op + alert; clamped proposal flows to control queue correctly; daily cap stops cycle 21; min-gap suppresses rapid events; cost math.

### Task 14 — Self-funding scorecard
**Files:** `src/monitoring/performance.py`, `src/monitoring/telegram_bot.py`, `cli/tb.py` (manager cmd already reads manager_log), tests
1. `performance.py`: `net_pnl_after_api(days) = realized_net_pnl - sum(manager_log.cost_zar)` over window; expose in perf summary dict.
2. Daily summary Telegram includes: cycles run today, adjustments applied, API cost ZAR today, net-after-cost P&L today + cumulative.
3. Tests: math with seeded journal + manager_log.

### Task 15 — Backtest integration for the managed system
**Files:** `backtest/simulator.py`, `backtest/runner.py`, new `backtest/manager_sim.py`, tests
1. `--manager` mode for the simulator: every 60 simulated minutes within session hours, build a briefing from **simulator state** (same schema as live briefing — reuse `briefing.build` with an adapter over sim equity/trades/weights) and get proposals, apply through the SAME `policy.validate_and_clamp`, mutate sim's effective params (risk %, thresholds, weights).
2. Two manager backends: `--manager=heuristic` (deterministic, no API: e.g., reduce weight 0.25 on instrument with PF<0.8 over trailing 20 trades, raise 0.25 if PF>1.5, nudge risk% toward ceiling after milestone, mute instrument after 5 straight losses — document exactly) and `--manager=claude` (real API via manager client; requires ANTHROPIC_API_KEY; logs cost; ~10 calls per simulated day).
3. Simulator gains: per-instrument weights affecting position size, ratcheting floor kill-switch, 21:00 UTC daily resets, R1000 starting balance from config, milestone/risk-ceiling ladder.
4. Multi-instrument: simulator runs all instruments with data available in `data/historical/` for the chosen window.
5. Report additions: manager decision log (per cycle: proposals/applied/clamped), API cost total, net-after-cost equity curve, comparison table baseline (no manager) vs managed.
6. Tests: heuristic manager unit tests; sim applies weight changes to subsequent sizing; floor kill-switch stops sim; no-API needed for tests.

### Task 16 — Fetch fresh data + run last-week backtest + report
**Files:** none new (runs pipeline), report at `docs/reports/2026-07-07-lastweek-managed-backtest.md`
1. Attempt `python -m src.main --fetch-data` (needs local MT5 terminal login; .env creds). If it fails, use the freshest available cached window and SAY SO in the report.
2. Train/refresh model if the pipeline requires it for the window (walk-forward: train strictly before test window).
3. Run: baseline backtest (no manager) AND `--manager=heuristic` AND (if ANTHROPIC_API_KEY present in env) `--manager=claude`, over the last 7 available days, all instruments with data, starting balance R1000.
4. Report: window used, trades, win rate, PF, net P&L ZAR, max DD, manager decisions timeline, API cost, net-after-cost, baseline-vs-managed comparison, honest caveats (spread/slippage assumptions).

### Task 17 — Runbook + docs (fix J)
**Files:** `docs/runbooks/vps-provisioning.md`, `docs/runbooks/live-cutover-r1000.md`, `CLAUDE.md`
1. VPS runbook: Hetzner Windows provisioning, MT5 install + auto-login, NSSM install for BOTH services (traderbot + manager) with `RestartDelay=10000`, `AppRotateFiles=1`, SSH key + local `tb` alias.
2. Live cutover checklist: the 5-item hardening gate from the spec, .env swap to live creds, ANTHROPIC_API_KEY, verify safety floor state file seeded, first-hour supervision checklist.
3. CLAUDE.md updated: new architecture tree (control/, cli/, manager/), R1000 capital ladder, manager cadence + levers, run commands (`python -m src.manager`, `python -m cli.tb status`).

## Execution notes

- Order: 1 → 2..7 (hardening; 2-7 independent of each other after 1) → 8 → 9,10 (need 8) → 11 (needs 8) → 12 → 13 (needs 12) → 14 (needs 13's manager_log) → 15 (needs 12) → 16 (needs 15) → 17.
- `pip install anthropic` + add to requirements in Task 13.
- Each task: TDD, commit(s) on `feat/claude-manager-live-r1000`, run full suite.

### Task 18 — Prompt lab: Ralph loop for manager-prompt optimization
**Files:** new `backtest/prompt_lab.py`, `src/manager/prompts/` (versioned prompt files, `champion.md` symlink-equivalent via `champion.txt` containing filename), `src/manager/client.py` (load system prompt from champion file), tests
1. Prompt variants live as files `src/manager/prompts/v001.md`, `v002.md`… Client loads the one named in `champion.txt` (fallback: highest version).
2. `python -m backtest.prompt_lab --variants v001,v002 --window <days>`: for each variant, run the managed backtest (`--manager=claude`) over the same window/seed, score = net-after-cost P&L with a max-drawdown penalty (`score = net_pnl_zar - 2 * max_dd_zar`), write results to `backtest/prompt_lab_results.jsonl` (variant, window, trades, pnl, dd, api_cost, score).
3. Champion selection: best score wins; update `champion.txt`. Never auto-delete losing variants (audit trail).
4. Ralph-loop mode `--auto N`: N iterations of [run champion + challenger → if challenger wins, promote → generate next challenger by mutating champion (the orchestrating Claude Code session writes the mutation based on the decision-log analysis; the lab pauses and prints what happened for the orchestrator to act)]. The lab is the harness; the mutation intelligence is the orchestrator session.
5. Requires ANTHROPIC_API_KEY; if absent, lab exits with a clear message. Heuristic backend usable for harness smoke tests (no API).
6. Tests: scoring math, champion promotion, results file append, no-key graceful exit (all with heuristic/mocked backend).

## Amendment (2026-07-07): API budget governor

Binds Tasks 13 & 14:
- settings.yaml `manager:` block gains: `api_budget_zar_total: 500`, `api_budget_days: 10`, `api_budget_zar_per_day: 50`.
- Scheduler enforcement (Task 13): before each cycle, sum `manager_log.cost_zar`. If today's spend ≥ per-day cap OR cumulative spend ≥ total budget → skip cycle, log `outcome=budget_exhausted`, Telegram-warn once per day. Timer cycles degrade gracefully (fewer, evenly spaced) as the daily cap approaches: if remaining daily budget < estimated cycle cost × remaining cycles, stretch the interval.
- Day-10 verdict (Task 14): scorecard gains `justification_report()` — cumulative net P&L, cumulative API cost, net-after-cost, verdict line (`SELF-FUNDING` if net-after-cost > 0 and P&L uplift vs heuristic baseline > API cost, else `NOT JUSTIFIED`). Emitted in daily summary from day 8 onward and via `tb manager --verdict`.
