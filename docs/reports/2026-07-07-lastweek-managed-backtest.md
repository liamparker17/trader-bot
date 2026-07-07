# Last-Week Managed Backtest — Baseline vs Heuristic Manager (Task 16)

**Date of run:** 2026-07-07
**Command:** `python -m backtest.runner --manager heuristic --window-days 7`
**Starting balance:** R1000 (from `account.starting_balance_zar`)

## Data window — STALE DATA CAVEAT (read first)

A fresh fetch was attempted (`python -m src.main --fetch-data`) and **failed**:
MT5 terminal `Authorization failed` ×3 — the local terminal is not
running/logged in. Per the plan, the run uses the **freshest cached data**
instead:

- **Window used:** 2026-02-27 10:40 UTC → 2026-03-06 10:40 UTC (last 7
  available days; the cache ends **four months before this report's date**)
- Instruments: EUR_USD, GBP_USD, USD_JPY, XAU_USD (all four with cached
  M1+M15 data)
- Walk-forward hygiene: the 7-day window sits inside the last-30% test
  split, so the loaded model's training data (first 70%, ending ~mid-Feb)
  is never replayed. Model loaded: **v1.40** (latest saved; no retrain was
  required for this window).

Re-run this on fresh data once the MT5 terminal is logged in — treat these
numbers as a pipeline validation, not a current-market estimate.

## Results

| Metric | Baseline (no manager) | Managed (heuristic) |
|---|---|---|
| Trades | 21 | 21 |
| Win rate | 52.4% | 52.4% |
| Profit factor | 1.59 | 1.59 |
| Total P&L (ZAR) | +10.48 | +10.48 |
| Return % | +1.0% | +1.0% |
| Max drawdown % | 1.1% | 1.1% |
| Final balance | R1010.48 | R1010.48 |
| Killed by ratchet floor | No | No |
| Manager cycles | 0 | 66 |
| API cost (ZAR) | 0.00 | 0.00 |
| **Net P&L after cost** | **+10.48** | **+10.48** |

### Claude-manager run

**Skipped — `ANTHROPIC_API_KEY` is not present in the environment/.env on
this machine.** The pipeline supports it (`--manager claude`); run it on the
VPS (or after adding the key) for the three-way comparison the self-funding
verdict needs.

## Manager decision timeline

All 66 heuristic cycles were `no_op` ("no rule fired; holding current
parameters"), at the correct cadence: hourly within 07–20 UTC session hours,
starting 2026-02-27 10:40 and ending 2026-03-06 10:40. Risk ceiling stayed
1.50 (balance never crossed the first R1500 milestone).

Why no rule fired, and why that is the *correct* outcome on this window:

- The PF-based weight rules arm only once an instrument has ≥20 closed
  trades in its trailing window; with 21 trades total across four
  instruments in 7 days, no instrument got close.
- No instrument hit 5 consecutive losses (mute rule).
- No milestone was crossed (risk-nudge rule), so risk stayed at baseline.

So baseline and managed runs are identical — the manager plumbing ran at
full cadence (66 briefings → proposals → `policy.validate_and_clamp` →
param overlay) and correctly chose to do nothing. This is a validation
that the managed pipeline is wired and non-destructive, not evidence of
manager alpha.

Contrast: on the longer ~3-month window (`--manager heuristic` without
`--window-days`), the heuristic actively degrades performance (weight rules
walked USD_JPY 1.5→0.0 on a stale trailing-20 PF; baseline +R175 vs managed
+R17) — a known heuristic weakness recorded in progress.md for triage, not
a pipeline defect.

## Honest caveats

1. **Stale data** — window ends 2026-03-06, four months old (see above).
2. **7 days is far too small a sample** (21 trades) to distinguish manager
   policies; all manager rules stayed dormant. The self-funding verdict
   (Task 14) requires the R500/10-day live window plus a heuristic baseline
   over the same period.
3. **Spread/slippage are simulated** by the backtest engine's fixed
   assumptions per instrument; live Exness spreads (especially XAU_USD
   outside 13–17 UTC) will differ.
4. **No Claude run** (no API key here) — the three-way comparison
   (baseline / heuristic / claude) is outstanding; heuristic-vs-baseline is
   the only pair this report can show.
5. Model v1.40's logged training accuracy (0.958) looks optimistic relative
   to live expectations — walk-forward test-window metrics (this report's
   table) are the numbers that matter.
6. The heuristic manager's known stale-PF weakness (item above) means the
   heuristic baseline for the Day-10 verdict should be re-scored once the
   rule gets a "requires new trades between reductions" guard.

## Reproduction

```bash
python -m src.main --fetch-data                                  # needs MT5 terminal logged in
python -m backtest.runner --manager heuristic --window-days 7    # baseline vs heuristic
python -m backtest.runner --manager claude --window-days 7       # needs ANTHROPIC_API_KEY
```
