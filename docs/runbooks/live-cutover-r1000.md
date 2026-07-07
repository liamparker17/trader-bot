# Live Cutover Checklist — R1000 Real-Money Exness Account

**Scope:** the one-time cutover from demo to a real-money Exness account starting at R1000, per
`docs/superpowers/specs/2026-07-07-claude-manager-live-r1000-design.md`. Same code runs demo and live —
cutover is a credential/flag change, gated on the hardening checklist below passing in full first.

**Do not skip the gate.** The system is going live with real money on day one, with no month-long demo
soak — the hardening gate is the only safety net standing in for that soak period.

## Pre-live hardening gate (blocking — all 5 must pass)

1. **All 2026-05-02 blockers A–I fixed**: single-instance lock (`src/utils/instance_lock.py`); MT5
   disconnect detection + trade-pause + exponential backoff; hard-floor unit fix (now the ratcheting
   floor, `src/risk/ratchet_floor.py`); drawdown emergency close-all
   (`executor.close_all(reason="daily_drawdown")`); 21:00 UTC session reset for daily
   drawdown + consecutive losses; MT5-ticket-required trade IDs (no timestamp-fallback synthetic
   trades); daily-summary scheduler; MT5 connect/disconnect/reconnect Telegram alerts.
2. **All 2026-05-02 fixes J–S**: NSSM watchdog config (`RestartDelay=10000`, `AppRotateFiles=1` — see
   `docs/runbooks/vps-provisioning.md`); ML checkpoint persistence across restarts; symbol-suffix
   re-detection on every reconnect (not just startup); timestamp tz validation (`to_utc()` helper);
   spread-aware order deviation (`max(20, ceil(spread_points * 1.5))`); fill-transaction validation +
   transient-requote retry (REQUOTE/PRICE_OFF/PRICE_CHANGED); journal fee/swap columns
   (`commission`, `swap`) feeding net P&L; general-exception Telegram alerting from the main loop.
3. **Fresh audit of the two net-new surfaces**: the manager process (`src/manager/*`) and the
   per-instrument `weight.<INSTRUMENT>` path (`src/risk/position_sizer.py` reading
   `weight.<INSTRUMENT>` from `EffectiveConfig`, defaulting to 1.0, muting at 0.0, still respecting the
   leverage cap at 1.5) — plus a re-check that nothing still reads the old R500/R9000 assumptions
   (grep for `9000`, `hard_floor`, and the old `[750, 1000, 2000, 3000, 6000]` milestone list across
   `src/`, `config/`, and `docs/`).
4. **Control-plane completion**: `cli/tb.py` full read+write surface, `control_log` audit table,
   `config/safety_floor.yaml` loader, and `src/control/queue.py` (inbox/outbox command queue) all
   finished and exercised — the manager depends on every one of these.
5. **Dry-run**: one full manager cycle against the **demo** account in a single supervised session.
   Verify, in order:
   - Briefing assembly succeeds (`src/manager/briefing.py` produces a schema-valid JSON snapshot).
   - A real Claude API call completes and returns a forced `propose_adjustments` tool call.
   - Proposal clamping works: deliberately provoke an out-of-bounds proposal (e.g. via a manual `tb
     tune` staged just before the cycle, or a mocked briefing that would tempt the model past a bound)
     and confirm it is rejected/clamped and logged in `manager_log.rejected_json` — not silently
     dropped.
   - An applied tune shows up in both `control_log` (via the standard `tb tune`-equivalent write path)
     and a Telegram message.
   - Force a service restart (`nssm restart traderbot` and separately `nssm restart manager`) mid-run
     and confirm clean recovery: trading loop reconciles positions from MT5, manager resumes on its
     next scheduled cycle, no double-application of a tune.

Only after all 5 pass does the bot point at the live R1000 account.

## `.env` swap to live credentials

1. **Stop both services first** (`nssm stop traderbot`, `nssm stop manager`) — never edit `.env` while
   the trading loop is polling MT5.
2. Back up the demo `.env` (e.g. `.env.demo.bak`) so you can revert quickly if the cutover needs to be
   aborted.
3. Update in `.env`:
   - `MT5_LOGIN` → the live account number (Exness live account, R1000 initial deposit).
   - `MT5_PASSWORD` → the live account's investor or trading password (trading password — the bot needs
     to place orders).
   - `MT5_SERVER` → the **live** Exness server name (this differs from the demo server, e.g.
     `Exness-MT5Real7` vs `Exness-MT5Trial7` — confirm the exact string from the Exness Personal Area,
     do not guess).
   - `ANTHROPIC_API_KEY` → confirm present and valid (required for the manager service; without it the
     manager no-ops every cycle per its error-handling contract — trading is unaffected, but the
     manager provides no value while this is missing).
   - `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` → unchanged, but re-verify the chat ID is the one you
     want live alerts sent to (not a demo test chat).
4. In the MT5 terminal on the VPS, log out of the demo account and log into the live account
   interactively once, confirming auto-login / "save account info" is re-enabled for the live login
   (same setting as `docs/runbooks/vps-provisioning.md` step 3.2).
5. Double-check `config/instruments.yaml` — confirm only the intended instruments are `enabled: true`
   for the live cutover (do not assume the demo enablement set is what you want live).

## Safety-floor state seeding

The ratcheting floor's high-water mark must start correctly for a **fresh live account**, not carry over
demo-account numbers.

1. Before starting the `traderbot` service against the live account, delete or archive any existing
   `data/account_state.json` from demo runs. `RatchetFloor._load_high_water_mark()` seeds
   `STARTING_HIGH_WATER_MARK_ZAR` (1000.0) whenever the state file is missing or corrupt, so a clean
   deletion is sufficient — do not hand-edit the file unless you've confirmed the exact JSON shape
   `{"high_water_mark": <float>, "updated_utc": <iso8601>}`.
2. Start `traderbot` once, confirm it connects to the live account and reports the actual live balance
   (should be ~R1000, the deposited amount). The first `RatchetFloor.update(balance)` call raises the
   HWM above the 1000.0 seed only if the live balance is actually higher than it; if the live deposit is
   exactly R1000 the HWM stays at 1000.0, which is the expected starting point.
3. Verify the seeded state file: `data/account_state.json` should now show
   `high_water_mark` ≈ live balance, and `RatchetFloor.current_floor` should compute to
   `max(600, high_water_mark * 0.65)` — at HWM 1000 this is R650 (`max(600, 650)`). Confirm via
   `tb status` or `tb config` that the effective floor matches this expectation before trading begins.
4. Confirm `config/safety_floor.yaml` still has `min_floor_zar: 600` and
   `max_total_drawdown_pct: 0.35` unchanged from the demo config — these are never touched by cutover.

## Telegram verification

1. Restart `traderbot` and `manager` and confirm both send a startup/connect alert to the correct live
   Telegram chat.
2. Manually trigger one alert of each class you rely on for first-hour supervision, if feasible without
   real risk (e.g. an MT5 connect/disconnect alert can be provoked by briefly pausing MT5 network
   access on a demo run beforehand — do this rehearsal on demo, not live).
3. Confirm the daily summary message format includes the manager scorecard fields once Task 14 lands:
   cycles run, adjustments applied, API cost (ZAR), net-after-cost P&L — verify after Task 13/14 land.
4. Sanity-check message latency (should arrive within a few seconds of the triggering event) — Telegram
   sends are best-effort/non-blocking, so a slow or dropped message should never be interpreted as "the
   bot is stuck," but it should also not be a persistent problem.

## First-hour supervision checklist

Stay actively watching (RDP or `tb` polling every few minutes) for the first hour of live trading:

- [ ] `tb status` shows `bot_running: true`, correct live balance, and the expected effective config
      (baseline settings.yaml + any already-applied tunes, no stale demo-era `EffectiveConfig` overlay).
- [ ] First live trade (if one fires) has a real MT5 ticket in `trades` (no synthetic/timestamp-fallback
      ID) and server-side SL/TP attached (per "ALL positions have server-side SL/TP" design rule).
- [ ] Spread on the traded instrument(s) is within the configured live spread limits
      (`config/instruments.yaml` per-instrument spread caps) — live spreads can differ meaningfully from
      demo.
- [ ] Position size for the first trade matches the expected 1.5%-of-R1000 risk calculation (accounting
      for any active `weight.<INSTRUMENT>` multiplier) — hand-verify the math once against
      `src/risk/position_sizer.py`'s formula.
- [ ] Drawdown tracker and ratchet floor are both updating on each balance change — `tb status` /
      `data/account_state.json` should reflect a HWM that only moves up.
- [ ] If a manager cycle fires during the hour: confirm it appears in `manager_log`
      (`tb manager --days 1`), the API cost looks sane (not wildly over the ~R50/day budget cap), and
      any applied tune shows in `control_log` + Telegram — verify after Task 13 lands.
- [ ] No repeated-restart pattern in `Get-Service traderbot, manager` or the service log files (a
      crash-loop in the first hour is the highest-signal early-warning sign of a live-only issue that
      demo testing didn't surface — e.g. a live-server symbol-suffix difference).
- [ ] Circuit breakers are visibly armed (check `tb status` for consecutive-loss counters at 0, no
      active spread-blowout pause) — confirms the breaker state didn't inherit anything stale from demo.

If anything on this list looks wrong, `tb pause --reason "..."` immediately (this only blocks new
entries; open positions keep their server-side SL/TP) and investigate before resuming.
