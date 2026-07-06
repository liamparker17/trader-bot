# Claude Portfolio Manager + Live R1000 Cutover

**Date:** 2026-07-07
**Status:** Design approved (verbal); pending written-spec review
**Supersedes:** the "No Claude API integration" Non-Goal in `2026-05-02-autonomous-vps-and-cli-control-design.md`. That decision was budget-driven and has been reversed. All *other* parts of the 2026-05-02 spec (hardening audit, `tb` control surface, safety-floor model, control queue) remain in force and are dependencies of this one.
**Brainstorm with:** Liam Parker

> **Currency:** all monetary values are **South African Rand (ZAR)**. Exness account base currency is ZAR. MT5 reports balance/P&L in account currency, used verbatim.

## Goal

Turn TraderBot into a self-directing, background-running trading system supervised by a Claude "portfolio manager" that makes ~10 API calls per trading day to tune the bot within hard safety bounds — and take it **live on a real-money Exness account starting at R1000**, after a full hardening pass.

The question we want answered: **starting from R1000, does the Claude-managed bot grow itself, and does the manager earn more than it costs to run?** If it climbs, the risk/floor ladder ratchets up with it.

## What changed from the 2026-05-02 design

| 2026-05-02 decision | Now |
|---|---|
| No Claude API in the loop (budget) | **Claude API IS the manager**, scheduled, in the loop |
| Claude = manual human-triggered inspector | Claude = **autonomous bounded tuner**, ~10 cycles/day |
| Month-long demo before live | **Live R1000 immediately, after full hardening** (no multi-day demo) |
| `src/ai/analyst.py` to be deleted | Manager is a **new, cleaner** component; `src/ai/*` still removed/repurposed |
| Capital ladder for ~R500 start | **Re-based around R1000**, with a ratcheting floor |

## Non-Goals

- Claude does **not** approve, open, close, or size individual trades. The ML model + risk manager own the trade lifecycle.
- Claude cannot touch any `safety_floor.yaml` value. Hard floors always win.
- No new instruments, indicators, or model-architecture work. Levers operate on existing config.
- No web UI rework (Streamlit dashboard stays for ad-hoc inspection; a manager panel is a nice-to-have, deferred).

## Architecture

```
┌────────────────────── Hetzner Windows VPS ───────────────────────┐
│  traderbot service (NSSM)  — PURE trading loop, unchanged authority│
│    • ML signals + risk manager decide every trade                  │
│    • polls control/inbox/*.cmd.json each loop (≤2s latency)         │
│    • applies effective_config overlay; safety_floor.yaml wins       │
│    • writes journal.db, emits Telegram                              │
│           ▲ reads (JSON via tb)        ▲ writes (tune cmds)         │
│           │                            │                           │
│  manager process (NEW: src/manager/)  — separate process/scheduler │
│    trigger (timer ~60min in-session, or event) →                   │
│      1. build briefing  (calls tb status/perf/trades/model/config) │
│      2. Claude API call (Opus 4.8, forced tool-use)                 │
│      3. validate proposals vs whitelist + hard bounds (clamp/reject)│
│      4. enqueue survivors as `tb tune` cmds → control queue         │
│      5. record api cost + decision to manager_log                   │
└────────────────────────────────────────────────────────────────────┘
        Developer (local Claude Code) still uses tb for manual ops.
```

**Design principle — least privilege by construction:** the manager has **no privileged interface**. It can only do what the `tb` CLI can do. Reads via `tb status/perf/trades/model/config`; writes via `tb tune`. Every write flows through the *same* control queue, `control_log` audit, safety-floor clamp, and Telegram broadcast that a human `tb tune` uses. If the manager process dies or the API is unreachable, the trading loop keeps running on its last effective config — trading is never blocked on Claude.

### Components

**`src/manager/` (new)**
- `briefing.py` — assembles a compact JSON portfolio briefing from the `tb` read commands: balance, today's P&L, drawdown vs cap, per-instrument performance (win rate, PF, net P&L, current weight), open positions, recent trades, current effective config delta, model version + calibration drift, and the current milestone/floor state.
- `client.py` — Claude API wrapper (Anthropic SDK, model `claude-opus-4-8`). Sends the system prompt + briefing; forces a `propose_adjustments` tool call. Retries with backoff on transient API errors; on hard failure, logs and **no-ops the cycle** (bot unaffected).
- `policy.py` — the manager's authority: the lever whitelist + hard bounds (mirrors `tb tune`), the per-cycle adjustment cap (≤3), the reasoning contract. Validates + clamps every proposal; rejects out-of-band loudly.
- `scheduler.py` — fires cycles on the timer and on subscribed events; enforces cadence and a minimum inter-cycle gap.
- `runner.py` — the process entry point (`python -m src.manager`); wires the above and writes `manager_log`.

**Forced structured output.** The Claude call uses tool-use. The model must respond by calling:
```
propose_adjustments(adjustments: [{ key: string, value: number, reason: string }],
                    rationale: string)
```
No free-text is parsed for actions. `rationale` is logged for the human. Each `adjustment.key` must be in the whitelist; each `value` is clamped to bounds; anything else is dropped with a logged reason.

**New lever: per-instrument weight.** A continuous risk multiplier per instrument, `weight.<INSTRUMENT>` ∈ [0.0, 1.5]:
- Stored in the effective-config overlay (default 1.0 for every enabled instrument).
- Consumed in `risk/position_sizer.py`: `risk_amount *= weight[instrument]` (then the existing floors/rounding/leverage cap still apply — a weight can only *reduce* below or *modestly raise* the base risk, never breach `risk_per_trade_pct` bounds or the leverage cap).
- `weight = 0.0` ⇒ instrument effectively muted (no new entries) without touching `instruments.yaml`.
- Added to the `tb tune` whitelist and to the manager's lever set.

**`manager_log` (new SQLite table in `journal.db`)**
Columns: `id, ts_utc, trigger (timer|drawdown|losses|breaker), briefing_json, model, input_tokens, output_tokens, cost_zar, rationale, proposals_json, applied_json, rejected_json, outcome`. One row per cycle. Powers the self-funding scorecard and a `tb manager` read command.

## Manager authority (levers + bounds)

| Lever | Hard min | Hard max | Effect |
|---|---|---|---|
| `weight.<INSTRUMENT>` **(new)** | 0.0 | 1.5 | Per-instrument risk multiplier; 0.0 mutes it |
| `risk.risk_per_trade_pct` | 0.5 | 2.5 | Global per-trade risk % |
| `ml.threshold_high` | 0.50 | 0.75 | "Trade freely" ML cutoff |
| `ml.threshold_low` | 0.45 | 0.65 | "Trade with confirmation" cutoff; must be ≤ threshold_high |

**Un-overridable (in `safety_floor.yaml`, never a lever, Claude cannot target):**
- Daily drawdown stop (4%), reset at 21:00 UTC.
- **Ratcheting hard floor** (see Capital section) — replaces the fixed R9000.
- Max effective leverage (5×).
- All circuit breakers (consecutive losses, spread blowout, API error threshold, min win rate).

**Per-cycle limits:** ≤3 adjustments per cycle; a minimum 20-minute gap between cycles even under event storms; each applied tune is audited + Telegram-broadcast exactly like a manual tune.

## Cadence + triggers

- **Timer:** one cycle every ~60 min while any configured trading session is active ⇒ ~10 cycles/day. No cycles outside session hours (nothing to manage).
- **Events (off-schedule wake-ups):** intraday drawdown crosses 2%; 3 consecutive losses; any circuit-breaker trip. Event cycles respect the 20-min minimum gap.
- **Cost predictability:** timer cycles are bounded; event cycles are rate-limited. Expected ≤ ~15 calls/day worst case.

## Capital re-basing around R1000 (with ratcheting floor)

Current config is built for a ~R500 start (milestones `[750,1000,2000,3000,6000]`, fixed `hard_floor: 9000`). Re-base:

- `account.starting_balance: 1000`, `account.currency: ZAR`.
- **Ratcheting hard floor** (replaces fixed R9000, lives in `safety_floor.yaml`):
  `floor = max(R600, high_water_mark × (1 − max_total_drawdown_pct))` with `max_total_drawdown_pct = 0.35` and an absolute never-below of **R600** (protects the initial stake). As the high-water mark climbs, the floor climbs with it and never falls — locking in gains. The manager cannot touch either the R600 minimum or the 35% band.
- **Re-based milestones:** `[1500, 2000, 3000, 4500, 6000]`, `target_balance: 6000`. Milestone alerts + reinvest phases keyed to these.
- **Threshold ratcheting on growth:** at each milestone crossed *upward and held*, the risk band's usable ceiling steps up (e.g. base `risk_per_trade_pct` allowance nudges toward the 2.5 cap) and the manager is informed of the new headroom in its briefing. Ratchets are **one-way within a session-day** and reset evaluation at the 21:00 UTC boundary. Concretely: the briefing exposes `growth_stage` and `risk_ceiling_now`, and `policy.py` uses `risk_ceiling_now` (≤ hard 2.5) as the effective max for that cycle. This is the "if it builds itself to that level it can up the threshold" behaviour — automatic, bounded, and reversible if the balance falls back.

## Self-funding scorecard

- Every cycle logs `input_tokens`, `output_tokens`, and computed `cost_zar` (Opus 4.8 pricing × live USD/ZAR from config, no network call in the hot path).
- `monitoring/performance.py` gains a **net-of-manager-cost** line: `net_pnl_after_api = realized_pnl − Σ cost_zar`.
- `tb manager [--days N]` and the daily Telegram summary report: cycles run, adjustments applied, total API cost, and net-after-cost P&L. This directly answers "is the agent worth its keep?"

## Pre-live hardening gate (blocking)

Live R1000 cutover is **gated** on all of the following passing. Same code runs demo/live — cutover is one credential/flag change after the gate.

1. **All 2026-05-02 blockers A–I** fixed (single-instance lock; MT5 disconnect detection + trade-pause + backoff; hard-floor unit fix — now the ratcheting floor; drawdown emergency close-all; 21:00 UTC session reset for drawdown + consecutive losses; MT5-ticket-required trade IDs; daily-summary scheduler; MT5 connect/disconnect/reconnect alerts).
2. **All 2026-05-02 fixes J–S** (NSSM watchdog config; ML checkpoint persistence; suffix re-detection on reconnect; timestamp tz validation; spread-aware order deviation; fill-transaction validation + transient-requote retry; journal fee/swap columns; general-exception Telegram alerting).
3. **Fresh audit** of the two net-new surfaces — the manager process and the per-instrument weight path — plus a re-check that the R1000 re-base didn't leave any code reading the old R500/R9000 assumptions.
4. **Control-plane completion** — the `tb` read+write surface, `control_log`, safety-floor loader, and control queue from the 2026-05-02 plan (partially built on `feat/autonomous-vps-cli`) finished, since the manager depends on them.
5. **Dry-run**: one full manager cycle against the demo account in a single supervised session — verify briefing assembly, a real Claude call, proposal clamping (deliberately provoke an out-of-bounds proposal and confirm it's rejected + logged), and that an applied tune shows up in `control_log` + Telegram. Force a service restart mid-run and confirm clean recovery.

Only after 1–5 pass does the bot point at the live R1000 account.

## Data flow (manager cycle)

```
trigger → briefing.build() ──JSON──▶ client.call(system_prompt, briefing)
   ▲                                        │ tool-use
   │                                        ▼
manager_log ◀── runner records ──  propose_adjustments([{key,value,reason}], rationale)
                                            │
                              policy.validate_and_clamp()
                                            │ survivors
                                            ▼
                        control queue (tb tune) → bot applies → control_log + Telegram
```

## Error handling

- **API down / timeout / malformed tool call:** retry w/ backoff (cap ~3); on give-up, log a no-op cycle to `manager_log` (`outcome=api_unavailable`) and Telegram-warn. Trading continues on last config.
- **Out-of-bounds proposal:** clamped to bound if numeric-in-family, else dropped; logged in `rejected_json`; not silently swallowed.
- **Manager process crash:** NSSM (or a separate scheduled task) restarts it; trading loop is independent and unaffected.
- **Conflicting/rapid tunes:** 20-min min gap + ≤3/cycle + the existing control-queue serialization prevent thrash.
- **Cost runaway:** hard daily cap on manager cycles (e.g. 20); on cap, manager idles until next 21:00 UTC boundary and Telegram-warns.

## Testing strategy

- **Unit:** `policy.validate_and_clamp` (in-bounds pass-through, out-of-bounds clamp, unknown-key drop, threshold_low ≤ threshold_high invariant, per-cycle cap); ratcheting-floor math; per-instrument weight applied in position sizer then re-clamped by leverage cap.
- **Contract:** `briefing.build()` produces a schema-valid briefing from a seeded journal; `client` maps a mocked tool-use response to proposals.
- **Integration (demo):** the item-5 dry-run above, scripted where possible.
- **Backtest unaffected:** manager is a live-only component; the backtest path does not invoke it. Confirm backtests still run green after the R1000 re-base.

## Open questions (non-blocking)

- Streamlit "Manager" panel (cycles, costs, net-after-cost) — **deferred**.
- Should event-triggered cycles use a cheaper/faster model than timer cycles? — default **no** (Opus 4.8 for all; ~10/day cost is negligible); revisit if cost data says otherwise.
- Weekly manager retro (a longer-horizon cycle that reviews the week) — **deferred**, easy to add later.

## Risks

- **Manager reasoning error** → mitigated by the hard-bounds clamp, ≤3/cycle, full audit + Telegram visibility, and least-privilege (can't exceed `tb tune`).
- **Live R1000 first-contact** → mitigated by the hardening gate + the supervised dry-run; but real money from day one is the user's explicit choice.
- **MT5 quirks on VPS** (auto-login failure after Windows update) → "no candles in N min" alert; RDP in.
- **Ratcheting floor too tight on a tiny account** → R600 absolute minimum protects the stake; 35% band chosen to allow scalping variance without a nuisance kill. Tunable in `safety_floor.yaml` (by the human, not the manager).
