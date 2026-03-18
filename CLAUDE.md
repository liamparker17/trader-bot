# TraderBot — Claude Code Context

## What This Is
Forex + Gold (XAU/USD) scalping bot. Starts R500 ZAR, targets R6,000+. Exness/MT5 demo first, then live.

## Quick Reference

### Run Commands
```bash
python -m src.main              # Live/demo trading
python -m src.main --backtest   # Backtest
python -m src.main --fetch-data # Fetch MT5 historical data
python -m src.main --dashboard  # Streamlit UI (port 8501)
python -m backtest.runner       # Full train + backtest pipeline
```

### Config
- `config/settings.yaml` — All trading params (risk, indicators, ML, sessions)
- `config/instruments.yaml` — Per-instrument config (pip location, spread limits, MT5 symbols)
- `.env` — Secrets (MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

### Architecture (7 modules, 48 files, ~8500 LOC)

```
src/
├── config.py                    # YAML + .env loader, Config object with dot-notation access
├── main.py                      # Orchestrator: wires all modules, trading loop, signal handlers
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
└── monitoring/
    ├── trade_journal.py         # SQLite trade log (trades, daily_summary, events tables)
    ├── performance.py           # Equity curve, drawdown, Sharpe, instrument/hourly breakdown
    ├── telegram_bot.py          # Non-blocking alerts for all events
    └── dashboard/app.py         # Streamlit: Overview, Trade History, Performance, ML Status

backtest/
├── runner.py                    # End-to-end: generate data → train → validate → backtest → report
└── simulator.py                 # Full backtest engine with spread/slippage, risk rules, equity tracking
```

### Key Trading Parameters
- **Risk**: 1.5% per trade, 4% daily drawdown stop, R9000 hard floor
- **SL/TP**: 1.5×ATR stop-loss, 1.8×ATR take-profit (1.2:1 R:R)
- **ML threshold**: ≥0.55 trade freely, ≥0.50 only with indicator confirmation
- **Instruments**: EUR_USD, GBP_USD, USD_JPY, XAU_USD (enabled in instruments.yaml)
- **Timeframes**: M15 trend direction, M1 entry timing
- **Circuit breakers**: 3 losses → halve size, 5 losses → 30min pause, <45% win rate → pause+retrain
- **Leverage cap**: 5× effective max
- **Sessions**: Forex 07-20 UTC, Gold 13-17 UTC

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
- All 7 modules complete and integrated
- Broker migration: OANDA → MT5/Exness (complete)
- ML model v2.0 architecture: 47 features, 10 indicator classes, no circular dependencies
- Backtest pipeline runs end-to-end on real EUR/USD data (~89k M1 candles)
- Model trained on real data — feature importance well-distributed (no single feature >5%)

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
