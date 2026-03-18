# TraderBot — Full Architecture Plan

## Configuration Summary

| Parameter | Value |
|---|---|
| Starting Capital | R500 ZAR (~$27 USD) |
| Target | R6,000+ |
| Instruments | EUR/USD, GBP/USD, USD/JPY, XAU/USD |
| Broker | OANDA (demo → live) |
| API | REST (orders) + WebSocket (prices) |
| Style | Scalping, 1–15 min holds |
| Trade Frequency | 20–60/day (adaptive) |
| Risk Per Trade | 1.5% of balance |
| Daily Drawdown Limit | 4% |
| Hard Floor | R350 (emergency shutdown) |
| Leverage Cap | 3–5x effective |
| ML Model | Random Forest / XGBoost |
| Timeframes | 15min (trend) + 1min (entry) |
| Indicators | EMA, RSI, ATR, MACD, Bollinger Bands |
| Dashboard | Streamlit web UI |
| Alerts | Telegram bot |
| Reinvestment | 100% until R6k, then 50/50 |

---

# Step 2 — Architecture & Module Skeletons

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    TRADERBOT CORE                        │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │  Data     │→│ Indicator │→│   ML     │              │
│  │ Collector │  │ Calculator│  │ Predictor│              │
│  └──────────┘  └──────────┘  └────┬─────┘              │
│                                    │                     │
│                              ┌─────▼─────┐              │
│                              │  Signal    │              │
│                              │  Combiner  │              │
│                              └─────┬─────┘              │
│                                    │                     │
│                    ┌───────────────▼───────────────┐     │
│                    │     Risk Manager (gate)       │     │
│                    └───────────────┬───────────────┘     │
│                                    │                     │
│                              ┌─────▼─────┐              │
│                              │  Trade     │              │
│                              │ Executor   │              │
│                              └─────┬─────┘              │
│                                    │                     │
│  ┌──────────┐  ┌──────────┐  ┌────▼─────┐              │
│  │ Growth & │←│ Logger & │←│ Position │              │
│  │Reinvest  │  │ Monitor  │  │ Manager  │              │
│  └──────────┘  └──────────┘  └──────────┘              │
│                                                         │
│  ┌──────────────────────────────────────────────┐       │
│  │  Streamlit Dashboard  +  Telegram Alerts     │       │
│  └──────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
```

## Data Flow

```
Live Price Feed (WebSocket)
        │
        ▼
  Candle Builder (1min bars)
        │
        ├──▶ Indicator Engine (EMA, RSI, ATR, MACD, BB)
        │         │
        │         ▼
        │    Feature Vector [15+ features]
        │         │
        │         ▼
        │    ML Model → Probability Score (0.0 – 1.0)
        │         │
        │         ▼
        │    Signal Combiner → BUY / SELL / SKIP
        │         │
        │         ▼
        │    Risk Gate → Position size, SL, TP
        │         │
        │         ▼
        │    Trade Executor → OANDA REST API
        │         │
        │         ▼
        │    Position Manager → Track open trades
        │         │
        │         ▼
        └──▶ Logger → DB + Streamlit + Telegram
```

---

## Module 1: Data Collection

### Purpose
Fetches historical candle data for backtesting and receives live price ticks
for real-time trading.

### Skeleton

```
data/
├── collector.py          # Main data fetching class
├── oanda_client.py       # OANDA API wrapper (REST + streaming)
├── candle_builder.py     # Converts ticks → 1min & 15min candles
├── historical_loader.py  # Fetches bulk historical data for backtesting
└── cache.py              # Local SQLite/Parquet cache for downloaded data
```

### Key Design Decisions

- **Historical data**: Fetch 6–12 months of 1-minute candles per instrument
  from OANDA's REST API. Store locally in Parquet files (fast, compact).
- **Live data**: WebSocket streaming for tick-level prices. Build 1-min candles
  in memory. Every 15 completed 1-min candles, also emit a 15-min candle.
- **Preprocessing**: Handle gaps (weekends, holidays), normalize timestamps to
  UTC, detect and discard bad ticks (spikes > 5 ATR from prior close).
- **Rate limits**: OANDA allows ~120 requests/sec on REST. Historical fetches
  will use pagination with backoff. Live stream has no rate limit.

### Error Handling
- WebSocket disconnect → auto-reconnect with exponential backoff (1s, 2s, 4s…
  max 60s). Log every disconnect. If 5 failures in 10 minutes → pause trading,
  send Telegram alert.
- REST API errors → retry 3x with backoff, then log and skip.
- Data quality → reject candles with zero volume or impossible OHLC
  (high < low, etc.).

---

## Module 2: Indicator Calculation

### Purpose
Computes technical indicators from candle data. These are the features the ML
model uses and also provide standalone trading signals.

### Skeleton

```
indicators/
├── engine.py             # Orchestrates all indicator calculations
├── trend.py              # EMA, MACD
├── momentum.py           # RSI
├── volatility.py         # ATR, Bollinger Bands
└── registry.py           # Register/deregister indicators dynamically
```

### Default Indicators

| Indicator | What It Does (Plain Language) | Trading Use |
|---|---|---|
| **EMA (8, 21)** | Exponential Moving Average — smooths price to show trend direction. 8-period is fast (reacts quickly), 21 is slow. When fast crosses above slow = uptrend starting. | Trend direction. Only take BUY signals when EMA-8 > EMA-21 on 15min. |
| **RSI (14)** | Relative Strength Index — measures if price moved too far too fast. Scale 0–100. Above 70 = "overbought" (may reverse down). Below 30 = "oversold" (may bounce up). | Filter out exhausted moves. Don't buy if RSI > 75. Don't sell if RSI < 25. |
| **ATR (14)** | Average True Range — measures how much price typically moves per candle. Higher ATR = more volatile market. | Set stop-loss distance. SL = 1.5 × ATR. Also used to skip trading if volatility is abnormally low (no profit opportunity) or high (too dangerous). |
| **MACD (12, 26, 9)** | Moving Average Convergence Divergence — tracks momentum by comparing two EMAs and their difference. Signal line crossovers indicate momentum shifts. | Confirmation. Only enter trade if MACD agrees with EMA trend direction. |
| **Bollinger Bands (20, 2)** | A channel around price based on standard deviations. Price near upper band = stretched high. Price near lower band = stretched low. Width shows volatility. | Mean-reversion context. If price touches upper band + RSI > 70, that's a strong reversal signal. Also, tight bands ("squeeze") can precede breakouts. |

### Multi-Timeframe Logic

```
15-minute candle indicators → Determine TREND BIAS (bullish / bearish / neutral)
1-minute candle indicators  → Determine ENTRY TIMING (when to pull the trigger)

Rule: Only take 1-min entries that AGREE with 15-min trend.
```

### Modularity
- Each indicator class implements a common interface: `calculate(candles) → Series`
- Registry pattern: add/remove indicators via config file without code changes.
- New indicators (e.g., VWAP, Stochastic) can be dropped in as plugins.

---

## Module 3: ML / Prediction

### Purpose
Takes indicator values as input features, outputs a probability score (0.0–1.0)
representing confidence that a trade in the given direction will be profitable.

### Skeleton

```
ml/
├── feature_builder.py    # Constructs feature vectors from indicators
├── trainer.py            # Trains/retrains models
├── predictor.py          # Loads model, runs inference
├── evaluator.py          # Tracks model performance metrics
├── model_store/          # Saved model files (versioned)
└── config.py             # Hyperparameters, thresholds
```

### ML Approach: Gradient Boosted Trees (XGBoost)

**Why XGBoost over Random Forest:**
- Better at capturing non-linear relationships
- Handles imbalanced data (more losing trades than winning)
- Fast inference (< 1ms per prediction)
- Built-in feature importance (shows which indicators matter most)
- Works well on tabular data with 15–30 features

**Why NOT deep learning yet:**
- Small dataset initially (6–12 months of 1-min data)
- Overfitting risk is high with neural networks on limited data
- XGBoost is more interpretable — critical when debugging with real money
- No GPU needed

### Input Features (per candle)

```
TREND FEATURES:
  - ema_8_value              (raw EMA-8)
  - ema_21_value             (raw EMA-21)
  - ema_crossover            (1 if EMA-8 > EMA-21, else 0)
  - ema_distance             (EMA-8 minus EMA-21, normalized)
  - macd_value               (MACD line value)
  - macd_signal              (signal line value)
  - macd_histogram           (MACD minus signal)
  - trend_15min              (1=bullish, 0=neutral, -1=bearish)

MOMENTUM FEATURES:
  - rsi_value                (RSI 0–100)
  - rsi_divergence           (price making higher high but RSI making lower high)

VOLATILITY FEATURES:
  - atr_value                (current ATR)
  - atr_ratio                (current ATR / 50-period average ATR)
  - bb_position              (where price is within Bollinger Band, 0.0–1.0)
  - bb_width                 (band width, normalized)
  - bb_squeeze               (1 if width < 20th percentile, else 0)

PRICE ACTION:
  - candle_body_ratio        (body size / total range)
  - upper_wick_ratio         (upper wick / total range)
  - lower_wick_ratio         (lower wick / total range)
  - close_vs_open            (1 if bullish candle, else 0)

CONTEXT:
  - hour_of_day              (0–23, captures session effects)
  - day_of_week              (0–4, Mon–Fri)
  - spread_current           (current bid-ask spread)
```

**Total: ~22 features**

### Labels (What We're Predicting)

For each historical candle, look ahead and determine:
- Did price hit a 1.5:1 reward-to-risk target BEFORE hitting the stop-loss?
- Label = 1 (profitable) or 0 (loss/breakeven)

This creates a binary classification problem.

### Probability Threshold

```
ML outputs: P(profitable) = 0.0 to 1.0

Trading rules:
  P >= 0.65  →  TAKE TRADE (high confidence)
  P >= 0.55  →  TAKE TRADE only if ALL indicators agree
  P < 0.55   →  SKIP (not enough edge)
```

### Adaptive Retraining Loop

```
INITIAL TRAINING:
  1. Train on 6–12 months of historical data
  2. Validate on most recent 2 months (walk-forward)
  3. Deploy model v1

ONGOING RETRAINING:
  Every 500 completed trades OR every 2 weeks (whichever first):
    1. Add new trade outcomes to training data
    2. Retrain model
    3. Compare new model vs. current on holdout set
    4. If new model is better → deploy it
    5. If new model is worse → keep current, log warning
    6. If win rate drops below 52% for 200+ trades → FORCE retrain
       and reduce position sizes by 50% until performance recovers

DECISION CHECKPOINT:
  - Every retrain generates a report (feature importance, accuracy,
    precision, recall, profit factor)
  - During live phase: human must approve model swap
  - During demo phase: auto-swap allowed
```

---

## Module 4: Trade Execution

### Purpose
Translates trade signals into actual orders on OANDA. Handles the messy
reality of slippage, latency, and partial fills.

### Skeleton

```
execution/
├── executor.py           # Main order placement logic
├── order_manager.py      # Track pending/open/closed orders
├── position_tracker.py   # Real-time position state
└── reconciler.py         # Verify local state matches broker state
```

### Order Placement Logic

```
Signal received: BUY EUR/USD, confidence 0.72

STEP 1 — PRE-TRADE CHECKS (Decision Checkpoint):
  ✓ Risk Manager approves position size
  ✓ No existing position in same instrument
  ✓ Daily trade count < max (60)
  ✓ Daily drawdown < 4%
  ✓ Balance > R350 (hard floor)
  ✓ Spread is within acceptable range (< 2x typical)
  ✓ Current time is within trading session hours

STEP 2 — CALCULATE ORDER PARAMETERS:
  - Direction: BUY
  - Size: calculated by Risk Manager (units, not lots)
  - Stop-Loss: entry - (1.5 × ATR)
  - Take-Profit: entry + (2.25 × ATR)  [1.5:1 R:R]
  - Order type: MARKET (for scalping speed)

STEP 3 — EXECUTE:
  - Send market order via OANDA REST API
  - Record timestamp of submission

STEP 4 — CONFIRM:
  - Verify fill price within 2 pips of expected
  - If slippage > 2 pips → log warning, adjust SL/TP accordingly
  - If order rejected → log error, skip trade
  - If partial fill → accept what we got, adjust SL/TP for actual size

STEP 5 — MONITOR:
  - Track position via streaming prices
  - Trailing stop option: move SL to breakeven after 1:1 R:R achieved
```

### Error Handling

| Scenario | Response |
|---|---|
| API timeout | Retry once after 1s. If fails again, skip trade, log. |
| Order rejected (insufficient margin) | Log, reduce position size by 50%, retry once. |
| Partial fill | Accept partial, set SL/TP for filled amount. |
| Slippage > 3 pips | Close immediately at market, log as "slippage exit". |
| Price moved away during processing | Requote — if new price still viable, execute. Otherwise skip. |
| Duplicate order detection | Check for existing position before placing. Never double up. |

### Reconciliation
Every 60 seconds, compare local position state with OANDA account state.
If mismatch detected → log critical error, use OANDA's state as truth, alert
via Telegram.

---

## Module 5: Risk Management

### Purpose
The gatekeeper. Every trade request passes through risk management before
execution. This module protects the account from ruin.

### Skeleton

```
risk/
├── manager.py            # Central risk gate
├── position_sizer.py     # Calculate trade size
├── drawdown_tracker.py   # Track daily/weekly/total drawdown
├── circuit_breaker.py    # Emergency stop conditions
└── volatility_adjuster.py # Modify risk based on market conditions
```

### Position Sizing Rules

```python
# FORMULA (conceptual, not code):
#
# risk_amount = account_balance × 0.015           (1.5% risk)
# stop_loss_pips = ATR × 1.5                      (volatility-based)
# pip_value = (depends on instrument and account currency)
# position_size = risk_amount / (stop_loss_pips × pip_value)
#
# EXAMPLE on R500 account, EUR/USD:
#   risk_amount = 500 × 0.015 = R7.50
#   ATR = 8 pips → SL = 12 pips
#   pip_value for micro lot ≈ R1.80
#   position_size = 7.50 / (12 × 1.80) = 0.35 micro lots
#   → Round down to 0.3 micro lots (ALWAYS round down, never up)
```

### Stop-Loss & Take-Profit Logic

```
STOP-LOSS:
  - Base: 1.5 × ATR from entry
  - Minimum: 5 pips (to avoid getting stopped by noise)
  - Maximum: 20 pips (to cap risk on high-volatility instruments)
  - For XAU/USD: scale these by gold's pip value

TAKE-PROFIT:
  - Base: 2.25 × ATR from entry (1.5:1 reward-to-risk)
  - Alternative: trail stop to breakeven after 1:1 achieved,
    then let winner run with trailing stop at 1 × ATR

TRAILING STOP (optional, after 1:1 R:R reached):
  - Move SL to breakeven (entry price)
  - Then trail by 1 × ATR as price moves favorably
  - This lets winning trades capture extra profit
```

### Daily Drawdown Limit

```
At start of each trading day (00:00 UTC):
  - Record starting_balance
  - max_allowed_loss = starting_balance × 0.04  (4%)

During trading:
  - If (starting_balance - current_equity) >= max_allowed_loss:
    → Close ALL open positions immediately
    → Stop all new trades for the day
    → Send Telegram alert: "Daily drawdown limit hit. Trading paused."
    → Log everything

Next day: reset and resume.
```

### Volatility-Adjusted Risk

```
normal_atr = 50-period average ATR
current_atr = latest ATR value
volatility_ratio = current_atr / normal_atr

IF volatility_ratio > 2.0:
  → Reduce position size by 50% (market is twice as wild as normal)
  → Widen SL to 2.0 × ATR
  → Log: "High volatility mode active"

IF volatility_ratio < 0.3:
  → Skip trading (market is dead, spreads will eat profits)
  → Log: "Low volatility — no trades"

IF volatility_ratio between 0.3 and 2.0:
  → Normal trading
```

### Circuit Breaker (Emergency Stops)

```
CONDITION                              → ACTION
─────────────────────────────────────────────────────
Balance < R350                         → FULL SHUTDOWN. Close all. Alert.
3 consecutive losses                   → Reduce size 50% for next 3 trades
5 consecutive losses                   → Pause 30 minutes. Alert.
Daily drawdown >= 4%                   → Stop for the day.
Weekly drawdown >= 8%                  → Stop for the week. Force ML retrain.
Win rate < 45% over last 100 trades    → Pause. Force ML retrain. Alert.
Spread > 3× normal                     → Skip trades on that instrument.
API errors > 5 in 10 minutes           → Pause all trading. Alert.
```

---

## Module 6: Logging, Monitoring & UI

### Purpose
Track everything. Provide real-time visibility. Generate data for ML feedback.

### Skeleton

```
monitoring/
├── logger.py             # Structured logging (JSON format)
├── trade_journal.py      # Detailed record of every trade
├── performance.py        # Calculate metrics (win rate, PnL, Sharpe, etc.)
├── telegram_bot.py       # Send alerts via Telegram
└── dashboard/
    ├── app.py            # Streamlit main app
    ├── pages/
    │   ├── overview.py   # Account balance, daily PnL, open positions
    │   ├── trades.py     # Trade history table with filters
    │   ├── performance.py # Charts: equity curve, drawdown, win rate
    │   └── ml_status.py  # Model version, accuracy, feature importance
    └── components/
        ├── charts.py     # Plotly chart helpers
        └── controls.py   # Pause/resume/close-all buttons
```

### What Gets Logged (Every Trade)

```
{
  "trade_id": "2026-03-05-001",
  "timestamp_open": "2026-03-05T10:23:45Z",
  "timestamp_close": "2026-03-05T10:31:12Z",
  "instrument": "EUR/USD",
  "direction": "BUY",
  "entry_price": 1.08432,
  "exit_price": 1.08461,
  "stop_loss": 1.08312,
  "take_profit": 1.08612,
  "position_size": 300,
  "pnl_pips": 2.9,
  "pnl_zar": 5.22,
  "ml_confidence": 0.72,
  "indicators_at_entry": { "ema_cross": true, "rsi": 42, "atr": 0.00080 },
  "exit_reason": "take_profit",
  "slippage_pips": 0.3,
  "spread_at_entry": 1.2,
  "balance_after": 505.22,
  "15min_trend": "bullish",
  "model_version": "v1.2"
}
```

### Telegram Alerts

```
ALERT TRIGGERS:
  - Trade opened  → "🟢 BUY EUR/USD @ 1.08432 | SL: 1.08312 | TP: 1.08612 | Conf: 72%"
  - Trade closed  → "✅ EUR/USD +2.9 pips (+R5.22) | Balance: R505.22"
  - Loss           → "🔴 EUR/USD -8.1 pips (-R7.50) | Balance: R492.50"
  - Daily stop     → "⛔ Daily drawdown limit (4%) hit. Trading paused until tomorrow."
  - Emergency stop → "🚨 EMERGENCY: Balance below R350. All trading halted."
  - ML retrain     → "🔄 Model retrained: v1.2 → v1.3 | Accuracy: 61% → 63%"
  - API error      → "⚠️ OANDA connection lost. Attempting reconnect..."
```

### Streamlit Dashboard Pages

```
PAGE 1 — OVERVIEW:
  - Current balance (large number)
  - Today's PnL (+ or -)
  - Open positions (live updating)
  - Win rate (today / all-time)
  - Equity curve chart
  - PAUSE / RESUME / CLOSE ALL buttons

PAGE 2 — TRADE HISTORY:
  - Sortable/filterable table of all trades
  - Filter by: instrument, direction, win/loss, date range
  - Click trade to see full details + chart

PAGE 3 — PERFORMANCE ANALYTICS:
  - Equity curve over time
  - Drawdown chart
  - Win rate by instrument
  - Win rate by hour of day
  - Average R:R achieved
  - Profit factor (gross profit / gross loss)
  - Sharpe ratio (risk-adjusted return)

PAGE 4 — ML STATUS:
  - Current model version
  - Feature importance bar chart
  - Accuracy / precision / recall over time
  - Last retrain date
  - Next retrain trigger
```

---

## Module 7: Growth & Reinvestment

### Purpose
Manages how profits are reinvested to compound account growth.

### Skeleton

```
growth/
├── reinvestment.py       # Reinvestment logic and rules
├── milestone_tracker.py  # Track balance milestones
└── scaling.py            # Adjust trade frequency and size as balance grows
```

### Reinvestment Rules

```
PHASE 1: GROWTH (R500 → R6,000)
  - 100% of profits reinvested
  - Position size recalculated every trade based on current balance
  - As balance grows, absolute risk per trade grows (1.5% of larger number)
  - Trade frequency can increase as margin allows

PHASE 2: HARVEST (R6,000+)
  - 50% of monthly profits withdrawn
  - 50% reinvested for continued growth
  - If drawdown occurs, temporarily switch to 100% reinvest until recovered

MILESTONES (each triggers a log entry + Telegram alert):
  R750   → "50% growth achieved"
  R1,000 → "Account doubled"
  R2,000 → "4× growth"
  R3,000 → "Halfway to target"
  R6,000 → "TARGET REACHED — switching to 50/50 harvest mode"
```

### Adaptive Trade Sizing as Balance Grows

```
Balance R500:   risk = R7.50/trade,   ~0.3 micro lots
Balance R1,000: risk = R15/trade,     ~0.7 micro lots
Balance R2,000: risk = R30/trade,     ~1.4 micro lots
Balance R5,000: risk = R75/trade,     ~3.5 micro lots

The bot automatically scales up because position size is always
calculated as percentage of CURRENT balance.

SAFETY: If balance drops below a milestone, position size automatically
shrinks. This is the power of percentage-based sizing.
```

---

# Step 3 — Top 5 Risks & Mitigation

## Risk 1: Account Wipeout from Overleveraging

**What could happen:** With R500, even a small sequence of losses at high
leverage can destroy the account. 5 losses at 2% risk each = 10% gone. At
high leverage, slippage can make actual losses larger than intended.

**Mitigation:**
- 1.5% risk cap per trade (not 2%)
- R350 hard floor (lose 30% max before full shutdown)
- Effective leverage capped at 3–5× (not broker's offered 100:1)
- Volatility adjustment reduces size in wild markets
- Consecutive loss circuit breaker (3 losses → halve size)

**Checkpoint:** After every 50 trades, review if actual risk per trade
matched intended risk. If slippage consistently adds > 0.5% extra risk,
tighten position sizing formula.

---

## Risk 2: ML Model Overfitting / Degradation

**What could happen:** Model performs great on historical data but fails on
live data (overfitting). Or market regime changes and model becomes stale.

**Hidden edge case:** Model trained on trending market, deployed into ranging
market. Confidence scores look good but trades keep losing.

**Mitigation:**
- Walk-forward validation (train on old data, test on recent)
- Never deploy a model without out-of-sample testing
- Track live accuracy vs. backtest accuracy — if gap > 10%, flag it
- Forced retrain every 2 weeks or 500 trades
- Win rate circuit breaker: < 45% over 100 trades → pause + retrain
- Feature importance monitoring: if rankings shift dramatically, alert

**Checkpoint:** Every model version generates a comparison report.
Human review required before deploying to live account.

---

## Risk 3: Broker/API Failures

**What could happen:** OANDA API goes down during open positions. Orders
fail. WebSocket disconnects. Prices stale.

**Hidden edge case:** API returns success but order never actually filled
("phantom fill"). Bot thinks it has a position, but it doesn't.

**Mitigation:**
- Reconciliation check every 60 seconds (local state vs broker state)
- Heartbeat monitoring on WebSocket (if no data for 5 seconds, reconnect)
- All open positions have server-side SL/TP (set at broker level, not just
  locally tracked). This way, even if bot crashes, SL protects the trade.
- If API down > 2 minutes → close all positions when connection returns
- Graceful degradation: if streaming fails, fall back to REST polling

**Checkpoint:** Weekly review of API error logs. If errors trending up,
investigate or consider backup broker.

---

## Risk 4: Spread Widening / Low Liquidity Kills

**What could happen:** During news events or low-liquidity hours (e.g.,
Asian session for EUR/USD), spreads widen to 5–10 pips. With a 12-pip SL,
a 5-pip spread means you start the trade already 40% toward your stop.

**Hidden edge case:** Spread is normal at entry, but widens during the trade
and eats into profit or triggers early stop.

**Mitigation:**
- Check spread BEFORE every trade. If spread > 2× average, skip.
- Avoid trading 30 minutes before and after major news events
  (integrate economic calendar API or maintain a manual schedule)
- Avoid low-liquidity periods (21:00–01:00 UTC for FX majors)
- For gold (XAU/USD), only trade during London+NY overlap (13:00–17:00 UTC)
- Track "spread cost as % of profit" metric. If > 30%, reduce frequency.

**Checkpoint:** Monthly analysis of trades lost primarily due to spread.
If pattern emerges, add time-of-day filters.

---

## Risk 5: Psychological Risk — Interfering with the Bot

**What could happen:** During a losing streak, user manually overrides the
bot, closes trades early, or changes parameters emotionally. This destroys
the statistical edge the system relies on.

**Hidden edge case:** User increases risk per trade after a winning streak
("it's working, let's push it"). This violates the compounding model and
can lead to catastrophic loss.

**Mitigation:**
- Parameter changes require cooldown period (can't change more than 1×/week
  during live trading)
- All manual overrides are logged with reason
- Dashboard shows "if you had let the bot run" comparison
- Weekly automated report comparing bot decisions vs. manual interventions
- During demo phase: absolutely no manual interference — pure data collection

**Checkpoint:** Monthly review of manual interventions and their impact.
Use this data to decide if override rules should be tightened.

---

# Step 4 — Iterative Testing & Adaptive Learning Plan

## Phase 1: Historical Backtesting (Weeks 1–3)

```
GOAL: Validate strategy logic and ML model on past data.

STEPS:
1. Fetch 12 months of 1-min data for all 4 instruments
2. Calculate all indicators
3. Build labeled dataset (profitable trade = 1, losing = 0)
4. Train XGBoost model with walk-forward cross-validation
   - Train: months 1–8
   - Validate: months 9–10
   - Test: months 11–12
5. Run full backtest simulation:
   - Simulate every trade with realistic spread, slippage (1 pip added)
   - Track: win rate, profit factor, max drawdown, Sharpe ratio
6. PASS CRITERIA:
   ✓ Win rate > 55%
   ✓ Profit factor > 1.3
   ✓ Max drawdown < 15%
   ✓ Sharpe ratio > 1.0
   ✓ Minimum 500 trades in test period

DECISION CHECKPOINT:
  If criteria NOT met → adjust indicators, features, thresholds. Repeat.
  If criteria met → proceed to Phase 2.
  DO NOT proceed to demo without passing backtest.
```

## Phase 2: Demo Account Simulation (Weeks 4–7)

```
GOAL: Validate in live market conditions without real money.

STEPS:
1. Deploy bot on OANDA demo account (R500 virtual balance)
2. Run 24/5 for minimum 3 weeks
3. Track all metrics in real-time via Streamlit + Telegram
4. Compare demo performance to backtest expectations
5. Identify gaps:
   - Execution latency impact
   - Spread impact vs. backtest assumptions
   - Slippage reality vs. backtest model
6. PASS CRITERIA:
   ✓ Win rate within 5% of backtest
   ✓ No critical bugs or crashes
   ✓ Risk management rules working correctly
   ✓ Reconciliation shows no state mismatches
   ✓ ML confidence correlates with actual outcomes
   ✓ Minimum 300 live demo trades

DECISION CHECKPOINT:
  If performance significantly worse than backtest → investigate.
  Common causes: overfitting, unrealistic backtest assumptions.
  Fix and re-run demo. Do NOT proceed to live without passing.
```

## Phase 3: Adaptive ML Retraining (Ongoing from Week 4)

```
GOAL: Keep model current as market conditions evolve.

TRIGGER CONDITIONS (any one):
  - 500 new trades completed since last retrain
  - 2 weeks elapsed since last retrain
  - Win rate dropped below 50% over last 100 trades
  - Market regime change detected (ATR shifts > 50% from baseline)

RETRAINING PROCESS:
  1. Combine historical data + all new trade data
  2. Retrain with same hyperparameters
  3. Also run hyperparameter search (if scheduled — monthly)
  4. Evaluate new model on holdout set
  5. Compare: new model accuracy vs. current model accuracy
  6. If new model better by > 1% accuracy → swap in
  7. If new model worse → keep current, log, investigate

HUMAN REVIEW REQUIRED (live phase only):
  - Any model swap
  - Any change to feature set
  - Any change to confidence thresholds
```

## Phase 4: Stress Testing (Week 5–6, parallel with demo)

```
GOAL: Verify bot handles extreme conditions gracefully.

TEST SCENARIOS:
1. Flash crash simulation
   - Inject 50-pip instant move in historical data
   - Verify SL triggers correctly
   - Verify no cascade of bad orders

2. API failure simulation
   - Kill WebSocket mid-trade
   - Verify reconnect logic
   - Verify server-side SL protects position
   - Verify reconciliation detects issues

3. Spread spike simulation
   - Inject 10× normal spread
   - Verify bot skips trades
   - Verify existing positions not adversely affected

4. Consecutive loss simulation
   - Force 10 consecutive losing trades
   - Verify circuit breakers activate correctly
   - Verify position sizing reduces
   - Verify Telegram alerts fire

5. Balance near hard floor
   - Set balance to R360
   - Verify position sizing is minimal
   - Verify R350 shutdown triggers correctly

PASS CRITERIA: All 5 scenarios handled without unintended behavior.
```

## Phase 5: Progressive Live Deployment (Week 8+)

```
GOAL: Transition to real money with minimal risk.

STAGE 1 — MICRO LIVE (Weeks 8–10):
  - Fund OANDA live account with R500
  - Trade at 50% of normal position size (effectively 0.75% risk)
  - Run for 2 weeks minimum
  - Compare to demo performance running in parallel
  - PASS: Performance within 10% of demo

STAGE 2 — NORMAL LIVE (Weeks 11+):
  - Scale to full 1.5% risk per trade
  - Continue monitoring closely for 2 more weeks
  - Weekly performance reviews

STAGE 3 — AUTONOMOUS (Week 13+):
  - Bot runs with minimal daily oversight
  - Weekly Telegram summary reports
  - Monthly deep-dive review of all metrics
  - Continue ML retraining loop

DECISION CHECKPOINTS:
  - After Stage 1: Human decision to scale up or extend micro phase
  - After every losing week: Review and decide if parameter adjustment needed
  - At R1,000 milestone: Review everything. Consider adding instruments.
  - At R3,000 milestone: Consider adding RL module (Phase 3 ML upgrade)
```

---

# Step 5 — Explanations & Documentation

## Plain-Language Module Explanations

### Data Collection Module
Think of this as the bot's "eyes and ears." It watches the market constantly,
recording every price movement. It stores old price data (like a history book)
and listens to live prices (like a news feed). Without clean, reliable data,
every other module fails.

### Indicator Module
Indicators are like lenses for reading the market. Raw price data is noisy —
indicators smooth it out and highlight patterns. EMA shows "which direction is
the market trending?" RSI shows "has the price moved too far, too fast?"
ATR shows "how wild is the market right now?" Each gives a different view,
and together they form a complete picture.

### ML Module
The ML model is the bot's "brain." It looks at all the indicator values and
asks: "Based on what I've seen before, when the market looked like THIS, did
the trade usually win?" It outputs a confidence percentage. If confidence is
high enough, the bot takes the trade. It keeps learning from new trades to
get smarter over time.

### Trade Execution Module
This is the bot's "hands." Once the brain decides to trade, this module
actually places the order with the broker. It handles all the messy details:
what if the price moved while we were deciding? What if the broker rejected
our order? What if we only got part of what we asked for? It makes sure
every order is placed correctly or not at all.

### Risk Management Module
This is the bot's "survival instinct." It answers one question: "How much
can I afford to lose on this trade without endangering the account?" It
calculates position sizes, sets stop-losses, and has emergency shutoffs.
Even if the brain and hands make mistakes, risk management prevents disaster.
Think of it as a circuit breaker in your house — it cuts power before
anything catches fire.

### Monitoring Module
This is the bot's "diary" and "alarm system." It records every single
thing that happens (for later review and learning). It also watches for
danger signs and immediately alerts you via Telegram. The Streamlit
dashboard lets you check on the bot anytime, like a security camera feed.

### Growth Module
This is the bot's "financial advisor." It decides how profits are used.
In the growth phase, every cent goes back into trading to compound faster.
Once the target is reached, it splits profits between withdrawal and
reinvestment. It automatically scales trade sizes up as the account grows
and scales them down during drawdowns.

## Key Assumptions

```
1. OANDA demo data is representative of live execution
   RISK: Demo fills are always perfect. Live may have worse slippage.
   MITIGATION: Add 1 pip slippage to all backtest/demo calculations.

2. Historical patterns have some predictive power for the future
   RISK: Markets can change regime (trending → ranging → chaotic).
   MITIGATION: ML retraining loop + regime detection via ATR ratio.

3. 60% win rate with 1.5:1 R:R is achievable
   RISK: This is optimistic for a fully automated system.
   MITIGATION: Bot is profitable even at 52% win rate with 1.5:1 R:R.
   Break-even is ~40% with 1.5:1 R:R. We have margin.

4. R500 is enough to trade meaningfully
   RISK: Some brokers don't allow positions this small.
   MITIGATION: OANDA allows nano lots (1 unit). Confirmed viable.

5. Local Windows machine has sufficient uptime
   RISK: PC restarts, power outages, internet drops.
   MITIGATION: All positions have server-side SL/TP. Move to VPS
   once strategy is proven. Bot auto-recovers on restart.
```

## Potential Failure Points

```
FAILURE POINT              IMPACT        DETECTION              RESPONSE
────────────────────────── ───────────── ────────────────────── ──────────────────
Internet disconnect        Medium        WebSocket heartbeat    Server-side SL
                                                                protects positions

PC crash/restart           Medium        Process monitor        Auto-restart script
                                                                + reconciliation

OANDA API change           High          Version check on       Pin API version,
                                         startup                alert on mismatch

Market flash crash         High          ATR spike detection    Widen SL, reduce
                                                                size, or pause

Model degradation          High          Win rate tracking      Auto-retrain +
                                                                size reduction

Broker insolvency          Critical      N/A (external)         Use regulated broker
                                                                (OANDA is FCA/CFTC)

Data corruption            Medium        Checksum validation    Rebuild from broker
                                                                API, log corruption
```

---

# Unsafe Assumption Flags

```
⚠️  FLAG 1: "12× return in months" is aggressive
    REALITY: Professional hedge funds target 15–25% annually.
    12× in 4–8 months would be extraordinary. Possible with
    compounding and high frequency, but expect it to take
    6–12+ months with drawdowns along the way.

⚠️  FLAG 2: "60% win rate" is not guaranteed
    REALITY: Many successful systems run at 45–55% win rate
    but profit via favorable R:R ratio. Design for 50% win rate,
    celebrate if you get 60%.

⚠️  FLAG 3: Scalping on small accounts has high relative spread cost
    REALITY: A 1.2-pip spread on a 12-pip target = 10% cost per trade.
    Over 1000 trades, this is significant. Monitor spread-to-target
    ratio obsessively.

⚠️  FLAG 4: Running 24/5 on a Windows desktop is unreliable
    REALITY: Windows Update restarts, sleep mode, internet drops.
    Plan for VPS migration once strategy is validated.
    All positions must have broker-side stop-losses as insurance.
```

---

# Project File Structure

```
TraderBot/
├── ARCHITECTURE.md          ← This file
├── config/
│   ├── settings.yaml        # All configurable parameters
│   ├── instruments.yaml     # Instrument-specific settings
│   └── secrets.env          # API keys (gitignored)
├── src/
│   ├── __init__.py
│   ├── main.py              # Entry point, orchestrator
│   ├── data/
│   │   ├── collector.py
│   │   ├── oanda_client.py
│   │   ├── candle_builder.py
│   │   ├── historical_loader.py
│   │   └── cache.py
│   ├── indicators/
│   │   ├── engine.py
│   │   ├── trend.py
│   │   ├── momentum.py
│   │   ├── volatility.py
│   │   └── registry.py
│   ├── ml/
│   │   ├── feature_builder.py
│   │   ├── trainer.py
│   │   ├── predictor.py
│   │   ├── evaluator.py
│   │   └── model_store/
│   ├── execution/
│   │   ├── executor.py
│   │   ├── order_manager.py
│   │   ├── position_tracker.py
│   │   └── reconciler.py
│   ├── risk/
│   │   ├── manager.py
│   │   ├── position_sizer.py
│   │   ├── drawdown_tracker.py
│   │   ├── circuit_breaker.py
│   │   └── volatility_adjuster.py
│   ├── growth/
│   │   ├── reinvestment.py
│   │   ├── milestone_tracker.py
│   │   └── scaling.py
│   └── monitoring/
│       ├── logger.py
│       ├── trade_journal.py
│       ├── performance.py
│       ├── telegram_bot.py
│       └── dashboard/
│           ├── app.py
│           └── pages/
├── tests/
│   ├── test_indicators.py
│   ├── test_risk.py
│   ├── test_execution.py
│   ├── test_ml.py
│   └── test_stress.py
├── backtest/
│   ├── runner.py
│   ├── simulator.py
│   └── reports/
├── data/
│   ├── historical/          # Parquet files
│   └── trade_logs/          # Trade journal DB
├── requirements.txt
├── .env.example
└── .gitignore
```

---

# Recommended Implementation Order

```
WEEK 1:  Project setup + Data Collection Module + Historical Loader
WEEK 2:  Indicator Module + Feature Builder
WEEK 3:  ML Training + Backtesting Framework
WEEK 4:  Risk Management + Trade Execution (paper trading)
WEEK 5:  OANDA Demo Integration + Reconciliation
WEEK 6:  Monitoring + Telegram + Streamlit Dashboard
WEEK 7:  Stress Testing + Bug Fixes
WEEK 8:  Demo validation period begins (3 weeks)
WEEK 11: Live micro-trading begins (if demo passes)
```
