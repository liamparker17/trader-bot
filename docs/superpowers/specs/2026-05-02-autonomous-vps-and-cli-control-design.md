# Autonomous VPS Deployment + Claude-Code CLI Control

**Date:** 2026-05-02
**Status:** Approved (design); audit findings TBD
**Author brainstorm with:** Liam Parker

## Goal

Take TraderBot from "backtest-only on developer's PC" to "running unattended on a Hetzner Windows VPS for one month against an Exness MT5 demo account, controllable on demand from the developer's local Claude Code session via a `tb` CLI."

The bot trades fully autonomously on its existing ML signals and risk manager. Claude Code is **not** in the trade-approval loop. It is invoked by the developer when they want to inspect, pause, or tune the bot — never on a schedule, never via the Anthropic API.

Success at the end of the month-long run = a confident go/no-go on live capital, backed by a real journal of what happened, what the bot did, and what manual interventions (if any) were needed.

## Non-Goals

- No Claude API integration on the VPS. No background `analyst.py`, no scheduled prompts, no `approval_queue.py`.
- No real-time per-trade LLM approval. The ML model + risk manager decide trades.
- No web UI changes (Streamlit dashboard stays as-is for ad-hoc local inspection).
- No new instruments, indicators, or model architecture work — this is a deployment + ops + control-plane project.
- No live-money rollout. That's a separate decision after the month-long demo run.

## Architecture

```
┌─────────────────────┐                    ┌──────────────────────────────────┐
│  Developer machine  │                    │  Hetzner Windows VPS  (~€10/mo)  │
│                     │                    │  ┌────────────────────────────┐  │
│  ┌──────────────┐   │   SSH + JSON       │  │ MT5 Terminal (auto-login)   │  │
│  │ Claude Code  │───┼───────────────────▶│  └────────────────────────────┘  │
│  │   + tb CLI   │   │   tb status        │  ┌────────────────────────────┐  │
│  └──────────────┘   │   tb pause ...     │  │ traderbot.exe (NSSM service)│  │
│                     │   tb tune ...      │  │  - main loop                │  │
│  ┌──────────────┐   │                    │  │  - polls control queue      │  │
│  │   Telegram   │◀──┼────────────────────┼──│  - writes journal.db        │  │
│  │  (heartbeat) │   │                    │  │  - emits Telegram alerts    │  │
│  └──────────────┘   │                    │  └────────────────────────────┘  │
└─────────────────────┘                    └──────────────────────────────────┘
```

### Components

**VPS (Hetzner CX22 or CPX21 Windows, EU region)**
- MT5 terminal launches on boot, auto-logs into the Exness demo account (saved credentials).
- Bot runs as a Windows service via [NSSM](https://nssm.cc/), which auto-restarts the process on crash with exponential backoff.
- Single-instance lock file prevents two bot copies from racing the same account.
- All logs go to `logs/traderbot.log` with daily rotation (kept 30 days).

**`tb` CLI (new module: `cli/tb.py`)**
- A standalone Python entry point installed inside the bot's venv on the VPS.
- Read commands query the SQLite trade journal and the in-memory state of the running bot (via the control queue protocol below).
- Write commands enqueue a command file the bot picks up on its next loop iteration (≤2s latency).
- Locally, the developer adds an alias: `tb = ssh user@vps "cd C:/traderbot && venv\Scripts\python -m cli.tb $@"` — Claude Code (running locally) calls `tb` via its Bash tool.

**Control queue (new module: `src/control/queue.py`)**
- Polled by the bot once per main-loop iteration.
- File-based: `control/inbox/*.cmd.json` (atomic rename from `.tmp`). Bot processes the oldest, writes a result to `control/outbox/<id>.result.json`, then deletes the inbox file.
- Each command carries: `id`, `verb`, `args`, `reason`, `requested_at`, `requested_by`.
- Writes mutate effective config in memory and persist to `control/effective_config.json` so they survive bot restarts. Tunes do NOT mutate `config/settings.yaml` — the YAML remains the baseline.

**Audit log (new SQLite table in existing `journal.db`)**
- Table `control_log` columns: `id`, `ts_utc`, `verb`, `args_json`, `reason`, `requested_by`, `before_config_json`, `after_config_json`, `outcome`.
- Every write command writes a row before and after execution.
- `tb revert` consumes from this table.

**Telegram alerts (extends existing `monitoring/telegram_bot.py`)**
- New alert types: `control.command_executed`, `control.tune_applied`, `control.bot_paused`, `control.bot_resumed`.
- Existing alerts (trades, circuit breakers, daily summary) stay as-is.
- Telegram is best-effort — bot keeps trading if Telegram is down.

## CLI Surface

### Read commands (no audit log, no Telegram, no rate limit)

| Command | Output |
|---|---|
| `tb status` | JSON: bot up/down, MT5 connection state, open positions count, today's P&L, drawdown vs daily cap, last trade time, current model version, current effective config delta from baseline |
| `tb trades [--days N]` | JSON list of recent trades (entry, exit, P&L, instrument, ML prob, exit reason) |
| `tb perf [--days N]` | JSON: win rate, profit factor, Sharpe, max drawdown, per-instrument breakdown, hourly breakdown |
| `tb positions` | JSON list of open positions with unrealized P&L and SL/TP distance in pips |
| `tb config` | Current effective config (baseline + tunes overlaid) |
| `tb logs [--tail N] [--level L]` | Tail of `logs/traderbot.log` filtered by level |
| `tb model` | Model version, last 50 predictions vs realized outcomes, calibration drift indicator |

### Write commands (audit-logged, Telegram-alerted)

| Command | Effect | Constraints |
|---|---|---|
| `tb pause --reason "..."` | Bot stops opening new trades. Open positions retain their server-side SL/TP. | Reason mandatory, ≥10 chars |
| `tb resume --reason "..."` | Bot resumes opening new trades. | Reason mandatory |
| `tb tune <key>=<value> --reason "..."` | Tunes one whitelisted param within hard bounds. | One key per call. Max 1 tune per rolling 24h. Reason mandatory. Out-of-bound rejects loudly. |
| `tb revert` | Reverts the last `tune`. | No-op if last command wasn't a `tune`. |

### Excluded by design (too dangerous)
- No `close-position` — risk manager owns position lifecycle.
- No `force-trade` — no LLM-originated entries.
- No raw config edit — only whitelisted tune keys.
- No model retrain trigger from CLI (separate `python -m src.main --train` workflow stays).

## Authority Model + Guardrails

### Whitelisted tunable params

| Key | Hard min | Hard max | Notes |
|---|---|---|---|
| `risk.risk_per_trade_pct` | 0.5 | 2.5 | Prevents over-leveraging or tuning to zero |
| `ml.threshold_high` | 0.50 | 0.75 | Keeps the ML signal in a sane band |
| `ml.threshold_low` | 0.45 | 0.65 | Same; must be ≤ `threshold_high` |
| `risk.max_consecutive_losses` | 2 | 8 | Pause aggressiveness |
| `instruments.<name>.enabled` | bool | bool | Can disable an instrument; cannot enable an instrument not present in `config/instruments.yaml` |

### Un-overridable hard floors (locked in `config/safety_floor.yaml`, never tunable)

- `risk.daily_drawdown_stop_pct = 4` — hard stop the day at -4%.
- `risk.hard_floor_balance = 9000` — kill switch if equity drops here.
- `risk.max_leverage_effective = 5`.
- `circuit_breaker.api_error_threshold = 10`.
- `circuit_breaker.spread_blowout_pause_minutes = 30`.

`safety_floor.yaml` is loaded **after** `settings.yaml` and the effective-config overlay; its values always win. The CLI rejects any tune that targets a safety-floor key.

### Rate limiting

- Max 1 `tune` command per rolling 24 hours, enforced by the bot reading `control_log` before applying.
- `pause` and `resume` are not rate-limited (need to be reactive in emergencies).

### Logging

Every write command, before applying:
1. Writes `control_log` row with `outcome = pending`.
2. Sends Telegram message: `[CONTROL] {verb} requested by {requested_by} — {reason}`.
3. Applies the change.
4. Updates `control_log` row with `outcome = applied | rejected | error` and `after_config_json`.
5. Sends Telegram confirmation: `[CONTROL] {verb} {outcome}: {summary}`.

## Bot Hardening Audit (audit pass needed before implementation)

Before any of this can run unattended for a month, the existing code must be verified across the following axes. Each item is verified during the audit phase (Phase 1 below); findings are turned into a punch list with severity (block / fix / nice-to-have) and fed into the implementation plan.

| # | Concern | What to verify |
|---|---|---|
| 1 | Crash recovery | Bot restarts cleanly under NSSM. On restart, reconciles open positions from MT5 → local state. Does not double-process the in-progress candle. |
| 2 | MT5 disconnect | If MT5 drops the broker connection, bot backs off, retries, and does not attempt to trade in the meantime. |
| 3 | Position reconciliation | The 60s reconcile loop handles broker-side closes (SL/TP hit while bot was offline) by updating the journal and emitting Telegram. |
| 4 | Server-side SL/TP | Every entry path enforces server-side SL/TP. No code path opens a naked position. |
| 5 | News + spread spikes | Spread filter in `config/instruments.yaml` actually blocks entries when spread exceeds threshold (NFP, FOMC). |
| 6 | Weekend rollover | Bot stops Friday close, resumes Sunday open, does not act on weekend gap candles. |
| 7 | Telegram failures | Telegram outage does not block trading or wedge the main loop. |
| 8 | Single-instance lock | Two copies of the bot cannot run against the same account. |
| 9 | Time sync | All timestamps are tz-aware UTC end-to-end (config, journal, logs, MT5 calls). |
| 10 | `src/ai/` fate | Confirm `analyst.py`, `approval_queue.py`, `shadow_trader.py`, `prompts.py`, `TRADING_BRAIN.md` are dead weight under the new design and can be deleted. |

## Implementation Phases

1. **Audit pass** (no code changes) — verify the 10 hardening items via parallel read-only agents. Output: severity-ranked punch list. **This phase happens immediately after the spec is approved.**
2. **Hardening fixes** — fix every audit item rated `block` or `fix`. `nice-to-have` items deferred or split into a follow-up.
3. **Control plane** — build `cli/tb.py`, `src/control/queue.py`, `control_log` table, the safety-floor config loader, and the Telegram hooks for write commands.
4. **Dead code removal** — delete `src/ai/*` and `TRADING_BRAIN.md` if the audit confirms they're unused. Update CLAUDE.md.
5. **VPS provisioning runbook** — short Markdown doc with exact steps to provision Hetzner, install MT5, set auto-login, install bot as NSSM service, set up SSH key for `tb` from local machine.
6. **48-hour shakedown on demo** — run on the VPS for 2 days. Exercise every CLI command. Force a restart and verify clean recovery. Only after this passes does the month-long test begin.

## Open Questions (none blocking)

- Do we want a `tb` self-test command that exercises a no-op control round-trip on demand (useful as a healthcheck)? — **deferred, can add later**
- Do we want to mirror the journal SQLite back to the developer's local machine periodically (rclone)? — **deferred, can be done by hand via `scp` on demand**

## Risks

- **MT5 quirks on a fresh Windows VPS** — auto-login sometimes silently fails after Windows updates. Mitigation: monitor for "no candles received in N minutes" and Telegram alert. If it fires, RDP in.
- **Loadshedding-equivalent at Hetzner is rare but nonzero** — Hetzner SLAs are good but not perfect. Acceptable for a demo; revisit before live.
- **Claude Code reasoning errors during a `tune`** — the developer is the human-in-the-loop. The CLI will refuse out-of-band tunes; rationale is logged and Telegram-broadcast so the developer sees every change.
- **Audit could surface ship-blocking issues that delay the project significantly** — that's the point of doing the audit. Better to find them now than during a live run.
