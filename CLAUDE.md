# TraderBot — Claude Code Context

## What This Is
Forex + Gold (XAU/USD) scalping bot. Starts R1000 ZAR with a ratcheting safety floor, targets R6,000+, supervised by a Claude "portfolio manager" process that tunes bounded risk levers ~10x/day. Exness/MT5 demo first, then live.

## Quick Reference

### Run Commands
```bash
python -m src.main              # Live/demo trading — ML + risk manager own every trade decision, unchanged
python -m src.main --backtest   # Backtest
python -m src.main --fetch-data # Fetch MT5 historical data
python -m src.main --dashboard  # Streamlit UI (port 8501)
python -m backtest.runner       # Full train + backtest pipeline
python -m src.manager           # Claude portfolio-manager process (separate service; verify after Task 13 lands)
python -m cli.tb <command>      # tb CLI — reads: status/trades/perf/positions/config/logs/model/manager;
                                 #   writes: pause/resume/tune/revert (all via the control queue)
```

### Config
- `config/settings.yaml` — All trading params (risk, indicators, ML, sessions, growth/milestones; gains a `manager:` block — verify after Task 13 lands)
- `config/instruments.yaml` — Per-instrument config (pip location, spread limits, MT5 symbols)
- `config/safety_floor.yaml` — Hard floors the control queue/manager can never tune: `daily_drawdown_stop_pct: 4.0`, `min_floor_zar: 600`, `max_total_drawdown_pct: 0.35`, `max_leverage_effective: 5.0`, circuit-breaker thresholds
- `.env` — Secrets (MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, ANTHROPIC_API_KEY)

### Architecture (9 modules, 55+ files, ~10500 LOC)

```
src/
├── config.py                    # YAML + .env loader, Config object with dot-notation access
├── main.py                      # Orchestrator: wires all modules, trading loop, signal handlers
├── control/
│   ├── queue.py                 # File-based command queue (control/inbox → outbox). VALID_VERBS =
│   │                             #   {pause, resume, tune, revert, status_snapshot}. TUNE_BOUNDS +
│   │                             #   WEIGHT_BOUNDS whitelist; control_log audit row per command;
│   │                             #   manual tunes rate-limited to 1/24h (manager tunes exempt)
│   └── effective_config.py      # Runtime tune overlay on top of settings.yaml; safety_floor.yaml always wins
├── utils/
│   ├── instance_lock.py         # Single-instance startup lock (data/traderbot.lock)
│   └── timeutil.py              # to_utc() — single timestamp-hygiene helper for all MT5 ingestion
├── data/
│   ├── mt5_client.py            # MetaTrader 5 Python API (orders, candles, pricing, polling stream)
│   ├── oanda_client.py          # [LEGACY] OANDA v20 client — kept for reference only
│   ├── candle_builder.py        # Ticks → M1/M15 OHLC candles with rolling buffers
│   ├── historical_loader.py     # Bulk fetch via MT5 copy_rates, Parquet cache, incremental updates
│   └── collector.py             # Orchestrator: threaded streaming, warmup, error tracking
├── indicators/
│   ├── registry.py              # BaseIndicator ABC + IndicatorRegistry plugin system
│   ├── trend.py                 # EMA(8,21), MACD(12,26,9) — registered on import
│   ├── momentum.py              # RSI(14), MomentumQuality — registered on import
│   ├── volatility.py            # ATR(14), BollingerBands(20,2), VolatilityRegime — registered on import
│   ├── statistical.py           # StatisticalFeatures (z-score, percentile, autocorrelation)
│   ├── price_action.py          # PriceAction (pin bars, engulfing, inside bars, consecutive)
│   ├── session.py               # SessionFeatures (London/NY distance, overlap, cyclical hour)
│   └── engine.py                # Runs all 10 indicators, builds 47-feature ML vector
├── ml/
│   ├── feature_builder.py       # Historical candles → labeled (X, y) dataset, EXCLUDE_FROM_MODEL set
│   ├── trainer.py               # XGBoost training, walk-forward validation, model versioning
│   ├── predictor.py             # Load model, predict probability, generate trade/skip signal
│   ├── evaluator.py             # Live accuracy tracking, retrain triggers
│   └── model_store/             # Saved .joblib models + _meta.json
├── risk/
│   ├── position_sizer.py        # risk_amount / (SL_pips * pip_value), volatility/loss adjustments
│   ├── drawdown_tracker.py      # Daily/weekly/total drawdown, high-water mark
│   ├── circuit_breaker.py       # Consecutive losses, win rate, spread, API errors, hard floor
│   └── manager.py               # Central gate: 8-point checklist → TradeApproval
├── execution/
│   └── executor.py              # Signal → risk check → MT5 order → fill confirm → trailing SL
├── growth/
│   ├── reinvestment.py          # Growth phase (100% reinvest) vs harvest phase (50/50)
│   ├── milestone_tracker.py     # R750/1000/2000/3000/6000 milestone alerts
│   └── scaling.py               # Instrument/position scaling recommendations by balance
├── monitoring/
│   ├── trade_journal.py         # SQLite trade log (trades, daily_summary, events, control_log, manager_log tables)
│   ├── performance.py           # Equity curve, drawdown, Sharpe, instrument/hourly breakdown
│   ├── telegram_bot.py          # Non-blocking alerts for all events
│   └── dashboard/app.py         # Streamlit: Overview, Trade History, Performance, ML Status
└── manager/                     # Claude portfolio-manager (separate process, python -m src.manager)
    ├── policy.py                 # LEVERS (mirrors control/queue.py TUNE_BOUNDS/WEIGHT_BOUNDS),
    │                              #   risk_ceiling_now() growth-stage ladder, validate_and_clamp()
    ├── briefing.py                # Assembles compact JSON cycle snapshot (balance, floor headroom,
    │                              #   per-instrument stats, model drift, growth_stage) for the API call
    ├── client.py                  # Anthropic SDK wrapper, forced tool-use propose_adjustments — verify after Task 13 lands
    ├── scheduler.py               # Cadence, min-gap, daily-cycle cap, API budget governor — verify after Task 13 lands
    ├── runner.py                  # python -m src.manager entry point — verify after Task 13 lands
    └── prompts/                  # Versioned system-prompt files + champion.txt — verify after Task 13/18 land

backtest/
├── runner.py                    # End-to-end: generate data → train → validate → backtest → report
└── simulator.py                 # Full backtest engine with spread/slippage, risk rules, equity tracking

cli/
└── tb.py                        # `tb` CLI: JSON in/out, reads via status_snapshot + journal, writes via control queue
```

### Key Trading Parameters
- **Capital**: starts R1000 ZAR (`account.starting_balance_zar`), target R6,000 (`growth.target_balance`)
- **Milestones**: `growth.milestones: [1500, 2000, 3000, 4500, 6000]` — re-based from the old R500/[750,1000,2000,3000,6000] ladder
- **Ratcheting hard floor** (`config/safety_floor.yaml`, replaces the old fixed R9000): `floor_zar = max(min_floor_zar, high_water_mark × (1 − max_total_drawdown_pct))`, with `min_floor_zar: 600` and `max_total_drawdown_pct: 0.35`. Floor is monotonically non-decreasing; HWM persists in `data/account_state.json` (`src/risk/ratchet_floor.py`, class `RatchetFloor`)
- **Risk**: 1.5% per trade baseline (`risk.risk_per_trade_pct`), 4% daily drawdown stop (`daily_drawdown_stop_pct`)
- **SL/TP**: 1.5×ATR stop-loss, 1.8×ATR take-profit (1.2:1 R:R)
- **ML threshold**: ≥0.55 trade freely, ≥0.50 only with indicator confirmation (`ml.confidence_threshold_high` / `_low`)
- **Instruments**: EUR_USD, GBP_USD, USD_JPY, XAU_USD (enabled in instruments.yaml)
- **Timeframes**: M15 trend direction, M1 entry timing
- **Circuit breakers**: 3 losses → halve size, 5 losses → 30min pause, <45% win rate → pause+retrain
- **Leverage cap**: 5× effective max (`max_leverage_effective`)
- **Sessions**: Forex 07-20 UTC, Gold 13-17 UTC; session/day boundary for drawdown + consecutive-loss counters is 21:00 UTC

### Claude Portfolio Manager
A bounded, autonomous tuner running as its own process/service — least-privilege by construction: it can only enqueue the same `tb tune` commands a human can, through the same `src/control/queue.py` whitelist, clamp, audit (`control_log`), and Telegram broadcast. It never opens/closes/sizes a trade directly.
- **Levers + hard bounds** (`src/manager/policy.py` `LEVERS`, mirrored in `src/control/queue.py` `TUNE_BOUNDS`/`WEIGHT_BOUNDS`):
  - `weight.<INSTRUMENT>` ∈ [0.0, 1.5] (default 1.0; 0.0 mutes the instrument)
  - `risk.risk_per_trade_pct` ∈ [0.5, 2.5], additionally capped per cycle at `risk_ceiling_now(balance, milestones)` — a growth-stage ladder (1.5 below the first milestone, stepping up to the hard 2.5 cap as milestones are crossed)
  - `ml.confidence_threshold_high` ∈ [0.50, 0.75]
  - `ml.confidence_threshold_low` ∈ [0.45, 0.65], must stay ≤ `threshold_high`
  - Un-overridable, ever: everything in `config/safety_floor.yaml` (daily drawdown stop, ratcheting floor, leverage cap, circuit breakers)
- **Cadence**: ~1 cycle/60min while a trading session is active (~10/day) plus event-triggered cycles (intraday drawdown ≥2%, 3 consecutive losses, any circuit-breaker trip), rate-limited to a 20-minute minimum gap and ≤3 adjustments/cycle — verify exact `manager:` settings.yaml block (`cycle_minutes`, `min_gap_minutes`, `max_cycles_per_day`, `event_triggers`) after Task 13 lands
- **API budget governor**: `api_budget_zar_total: 500` over `api_budget_days: 10` (`api_budget_zar_per_day: 50`); scheduler skips a cycle and logs `outcome=budget_exhausted` once caps are hit — verify after Task 13 lands
- **Audit**: every cycle writes a `manager_log` row (trigger, briefing, model, token/cost usage, rationale, applied/rejected proposals, outcome); `python -m cli.tb manager [--days N]` reads it back; net-of-manager-cost P&L surfaces in `monitoring/performance.py` and the daily Telegram summary

### ML Feature Architecture (47 features for model, ~57 computed total)

**10 Registered Indicator Classes:**
1. EMA (4): ema_fast/slow/crossover/distance
2. RSI (4): rsi_value/overbought/oversold/divergence
3. ATR (2): atr_value/ratio
4. MACD (4): macd_value/signal/histogram/crossover
5. BollingerBands (6): bb_upper/lower/middle/position/width/squeeze
6. MomentumQuality (4): macd_hist_accel/rsi_roc/momentum_consistency/rsi_distance_50
7. VolatilityRegime (4): atr_expansion_ratio/bb_squeeze_duration/volatility_regime/range_to_atr
8. StatisticalFeatures (4): price_zscore/price_percentile/autocorrelation_1/close_to_high_ratio
9. PriceAction (4): pin_bar_score/engulfing_score/inside_bar_tightness/consecutive_direction
10. SessionFeatures (5): minutes_since_london/minutes_since_ny/session_overlap/hour_sin/hour_cos

**Excluded from model (EXCLUDE_FROM_MODEL in feature_builder.py):**
Raw price-scale features: ema_fast, ema_slow, bb_upper, bb_lower, bb_middle, atr_value, macd_value, macd_signal
Circular feature: trend_15min (used as direction gate, not model input)
Replaced feature: hour_of_day (replaced by cyclical hour_sin/hour_cos)

**Additional computed features:**
- Price action: candle_body_ratio, upper_wick_ratio, lower_wick_ratio, close_vs_open
- Context: day_of_week, spread_current
- Interactions: momentum_x_vol_regime, trend_x_session
- Lagged (1-bar & 3-bar): macd_hist_accel, rsi_roc, momentum_consistency

### Data Flow
```
MT5 Tick Poll → CandleBuilder → IndicatorEngine → FeatureBuilder → Predictor
    → Signal (trade/skip) → RiskManager (approve/reject) → Executor → MT5 Order
    → TradeJournal + Telegram + Evaluator feedback loop
```

### Important Design Rules
- ALL positions have server-side SL/TP (survives bot crash)
- Position size always rounds DOWN
- Walk-forward validation only (never test on training data)
- Reconcile local vs broker state every 60 seconds
- Indicators register via registry pattern (import triggers registration)
- Registry key for BollingerBands is "bollingerbands" (class name lowered)
- `_indicators_agree()` uses INDEPENDENT signals (RSI, BB position, divergence) — NOT the same EMA/MACD signals used in trend_15min
- MT5Client returns OANDA-compatible dict formats (same keys) for backward compat
- MT5 uses lots not units — conversion handled inside mt5_client.py
- MT5 symbols have broker suffix auto-detected (EURUSD, EURUSDm, etc.)

### Broker: Exness via MetaTrader 5
- **Platform**: MT5 (MetaTrader5 Python package, Windows only)
- **Broker**: Exness (FSCA regulated, accepts South African clients)
- **Min deposit**: R18 (~$1 USD)
- **Account type**: Standard or Standard Cent (demo first)
- **MT5 streaming**: Polling-based (100ms interval), not true WebSocket
- **Symbol format**: "EURUSD" (no underscore), with possible broker suffix

### Build Status
- Core 7 trading modules complete and integrated; broker migration OANDA → MT5/Exness complete
- ML model v2.0 architecture: 47 features, 10 indicator classes, no circular dependencies
- Backtest pipeline runs end-to-end on real EUR/USD data (~89k M1 candles)
- Model trained on real data — feature importance well-distributed (no single feature >5%)
- 2026-05-02 hardening pass (blockers A-I, fixes J-S) complete: single-instance lock, MT5 disconnect
  detection + backoff + alerts, drawdown emergency close-all, 21:00 UTC session resets, real-ticket-only
  trade IDs, daily-summary scheduler, symbol-suffix re-detection, spread-aware order deviation,
  transient-requote retry, journal fee/swap columns, general-exception Telegram alerting
- Control plane complete: `src/control/queue.py` (whitelist + bounds + control_log audit),
  `src/control/effective_config.py` overlay, `cli/tb.py` read/write CLI, per-instrument
  `weight.<INSTRUMENT>` risk multiplier wired into `src/risk/position_sizer.py`
- `src/ai/*` (old manual-inspector scaffolding) removed; superseded by `src/manager/` (see below)
- Capital re-based R500 → **R1000** with the ratcheting floor (`src/risk/ratchet_floor.py`) replacing
  the fixed R9000 hard floor; milestones re-based to `[1500, 2000, 3000, 4500, 6000]`
- `src/manager/policy.py` (LEVERS, `risk_ceiling_now`, `validate_and_clamp`) and `src/manager/briefing.py`
  (cycle snapshot builder) complete; `client.py`/`scheduler.py`/`runner.py` + the `manager:` settings.yaml
  block are Task 13, in progress — verify presence/exact keys after it lands
- Live R1000 cutover is gated on the pre-live hardening checklist — see
  `docs/runbooks/live-cutover-r1000.md`; VPS provisioning — see `docs/runbooks/vps-provisioning.md`

### ML Model v2 — Results (ALL CHECKS PASSED)
**Backtest results (EUR_USD, ~27k M1 test candles, model v1.11):**
- **Win Rate: 50.4%** | Profit Factor: 1.27 | Sharpe: 1.83
- 119 trades, +R354 (+3.2%), Max Drawdown 2.3%
- Feature importance: well-distributed, top feature 3.08% (vs 70% before fix)

**Key fixes that made this work:**
1. Removed circular feature (trend_15min excluded from model via EXCLUDE_FROM_MODEL)
2. Fixed `_indicators_agree()` to use independent signals (RSI/BB/divergence, not EMA/MACD)
3. Removed raw price-scale features (ema_fast, bb_upper, etc.)
4. Added 21 new features across 5 new indicator classes + interactions + lags
5. Excluded neutral-trend candles from training (was poisoning labels as false negatives)
6. Balanced scale_pos_weight=1.0 (dataset is ~50/50 after excluding neutrals)
7. Removed auto scale_pos_weight adjustment in trainer
8. ML thresholds lowered: 0.55 high / 0.50 low (from 0.65/0.55)
9. R:R set to 1.2:1 (SL=1.5×ATR, TP=1.8×ATR) — breakeven at 45.5% win rate

**Next steps for further improvement:**
- Train on multiple instruments (combined dataset)
- Test on different time periods for robustness
- Consider adding volume profile or order flow features
- Try different timeframe combinations (M5 entry instead of M1)
- Exness demo account live testing

### Tech Stack
Python 3.11 | MetaTrader5 | XGBoost | pandas | requests | SQLite | Streamlit | Telegram Bot API
Parquet (pyarrow) | pyyaml | python-dotenv

### Coding Conventions
- Logging: `logging.getLogger("traderbot.<module>")` hierarchy
- Config access: `config.get("risk.risk_per_trade_pct")` dot-notation
- Instrument names: underscore format (EUR_USD) in our code, MT5 format (EURUSD) at API boundary
- All timestamps: UTC timezone-aware
- Error handling: MT5Error for API issues, retry with exponential backoff
