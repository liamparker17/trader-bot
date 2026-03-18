"""
Prompt builders for the AI Analyst.

Each function constructs a structured prompt for a specific analyst task,
injecting live market data into the template.
"""

from pathlib import Path

BRAIN_PATH = Path(__file__).resolve().parent.parent.parent / "TRADING_BRAIN.md"

_brain_cache: str | None = None


def get_system_prompt() -> str:
    """Load the TRADING_BRAIN.md system prompt (cached after first read)."""
    global _brain_cache
    if _brain_cache is None:
        _brain_cache = BRAIN_PATH.read_text(encoding="utf-8")
    return _brain_cache


def build_trade_review_prompt(
    instrument: str,
    direction: str,
    ml_confidence: float,
    features: dict,
    open_positions: list[dict],
    daily_pnl: float,
    balance: float,
    atr_value: float,
    spread: float,
    recent_trades: list[dict] | None = None,
) -> str:
    """Build prompt for reviewing a proposed trade."""
    positions_str = "None" if not open_positions else "\n".join(
        f"  - {p.get('instrument', '?')} {p.get('direction', '?')} "
        f"(entry: {p.get('entry_price', '?')}, pnl: {p.get('unrealized_pnl', '?')})"
        for p in open_positions
    )

    recent_str = "None"
    if recent_trades:
        recent_str = "\n".join(
            f"  - {t.get('instrument', '?')} {t.get('direction', '?')} "
            f"{'WIN' if t.get('pnl', 0) > 0 else 'LOSS'} {t.get('pnl', 0):.2f}"
            for t in recent_trades[-5:]
        )

    # Select key features to send (avoid flooding tokens)
    key_features = {
        k: round(v, 4) if isinstance(v, float) else v
        for k, v in features.items()
        if k in {
            "trend_15min", "rsi_value", "macd_histogram", "macd_crossover",
            "bb_position", "bb_squeeze", "atr_ratio", "ema_distance",
            "momentum_consistency", "volatility_regime", "rsi_divergence",
            "price_zscore", "pin_bar_score", "engulfing_score",
            "session_overlap", "ema_crossover",
        }
    }

    return f"""## Trade Review Request

**Proposed Trade:**
- Instrument: {instrument}
- Direction: {direction.upper()}
- ML Confidence: {ml_confidence:.1%}
- ATR: {atr_value:.5f}
- Current Spread: {spread:.1f} pips

**Key Indicators:**
{_dict_to_yaml(key_features)}

**Account State:**
- Balance: ${balance:.2f}
- Daily PnL: ${daily_pnl:+.2f} ({daily_pnl / balance * 100:+.1f}%)
- Open Positions:
{positions_str}

**Last 5 Trades:**
{recent_str}

Review this trade and respond with your JSON decision."""


def build_session_briefing_prompt(
    session: str,
    instruments: list[str],
    market_data: dict[str, dict],
    balance: float,
    daily_pnl: float,
) -> str:
    """Build prompt for a pre-session market briefing."""
    data_lines = []
    for inst in instruments:
        d = market_data.get(inst, {})
        data_lines.append(
            f"**{inst}:**\n"
            f"  - Last close: {d.get('close', 'N/A')}\n"
            f"  - ATR(14): {d.get('atr', 'N/A')}\n"
            f"  - RSI(14): {d.get('rsi', 'N/A')}\n"
            f"  - EMA trend: {d.get('ema_trend', 'N/A')}\n"
            f"  - BB position: {d.get('bb_position', 'N/A')}\n"
            f"  - Volatility regime: {d.get('volatility_regime', 'N/A')}"
        )

    return f"""## Pre-Session Briefing Request

**Session:** {session}
**Account Balance:** ${balance:.2f}
**Daily PnL so far:** ${daily_pnl:+.2f}

**Market Data:**
{chr(10).join(data_lines)}

Provide your session briefing JSON with regime analysis, directional bias per instrument, key levels, and any risk events."""


def build_regime_check_prompt(
    instrument: str,
    features: dict,
    recent_candles_summary: str,
) -> str:
    """Build prompt for a quick regime classification."""
    key_features = {
        k: round(v, 4) if isinstance(v, float) else v
        for k, v in features.items()
        if k in {
            "atr_ratio", "bb_width", "bb_squeeze", "volatility_regime",
            "trend_15min", "momentum_consistency", "price_zscore",
            "ema_distance", "autocorrelation_1",
        }
    }

    return f"""## Regime Check: {instrument}

**Indicators:**
{_dict_to_yaml(key_features)}

**Recent Price Action (last 30 M1 candles):**
{recent_candles_summary}

Classify the current market regime and recommend action. Respond with JSON only."""


def build_session_review_prompt(
    session: str,
    trades: list[dict],
    daily_pnl: float,
    balance: float,
    win_rate: float,
) -> str:
    """Build prompt for post-session review."""
    trades_str = "No trades taken." if not trades else "\n".join(
        f"  {i+1}. {t.get('instrument', '?')} {t.get('direction', '?')} | "
        f"{'WIN' if t.get('pnl', 0) > 0 else 'LOSS'} ${t.get('pnl', 0):+.2f} | "
        f"ML: {t.get('ml_confidence', 0):.0%} | "
        f"Reason: {t.get('exit_reason', 'N/A')}"
        for i, t in enumerate(trades)
    )

    return f"""## Post-Session Review

**Session:** {session}
**Trades Taken:**
{trades_str}

**Results:**
- Daily PnL: ${daily_pnl:+.2f}
- Balance: ${balance:.2f}
- Session Win Rate: {win_rate:.0%}
- Total Trades: {len(trades)}

Analyze the session: What patterns do you see? Any recurring mistakes? Suggestions for next session? Respond in JSON with keys: "assessment", "patterns", "mistakes", "suggestions"."""


def build_retrospective_prompt(
    instrument: str,
    strategy: str,
    day_summary: str,
    pip_size: float,
    sl_atr_mult: float,
    tp_atr_mult: float,
) -> str:
    """Build prompt for end-of-day retrospective analysis."""
    return f"""## End-of-Day Retrospective: {instrument}

You are reviewing today's complete price action for {instrument}. Your task: identify every trade you would have taken today with EXACT entry prices and times.

**Instrument Strategy Context:**
- Strategy: {strategy}
- Pip size: {pip_size}
- Typical SL: {sl_atr_mult} x ATR | TP: {tp_atr_mult} x ATR

**Today's Price Data (M15 bars with indicators):**
```
{day_summary}
```

Based on this data, identify the trades you would have taken today. For each trade, provide the EXACT entry bar time and price level from the data above.

Respond with JSON:
```json
{{
  "trades": [
    {{
      "entry_time": "2024-01-15 08:30:00",
      "direction": "buy" or "sell",
      "entry_price": 1.08432,
      "stop_loss": 1.08312,
      "take_profit": 1.08612,
      "confidence": 0.7,
      "reasoning": "Pullback to EMA in uptrend, RSI bouncing from 40, BB at 0.3"
    }}
  ],
  "market_assessment": "Brief overall assessment of today's price action",
  "missed_opportunities": "Any setups you see in hindsight that were borderline"
}}
```

Rules:
- Only include trades with CLEAR setups. Quality over quantity.
- Entry prices must match actual candle data (use close prices from the bars above).
- SL/TP must be at structural levels using ATR from the data.
- Minimum 1.5:1 reward:risk ratio.
- If no good setups existed today, return an empty trades array — that's a valid answer.
- Maximum 5 trades per instrument per day (you're a scalper, not a machine gun)."""


def build_shadow_trade_prompt(
    instrument: str,
    features: dict,
    current_price: float,
    atr_value: float,
    spread: float,
    balance: float,
    m1_summary: str = "",
) -> str:
    """Build prompt for Claude to independently recommend a trade."""
    key_features = {
        k: round(v, 4) if isinstance(v, float) else v
        for k, v in features.items()
        if k in {
            "trend_15min", "rsi_value", "macd_histogram", "macd_crossover",
            "bb_position", "bb_width", "bb_squeeze", "atr_ratio", "ema_distance",
            "ema_crossover", "momentum_consistency", "volatility_regime",
            "rsi_divergence", "price_zscore", "pin_bar_score", "engulfing_score",
            "session_overlap", "consecutive_direction", "inside_bar_tightness",
            "price_percentile", "autocorrelation_1", "close_to_high_ratio",
        }
    }

    return f"""## Independent Trade Recommendation: {instrument}

You are scanning {instrument} for a trade opportunity. This is YOUR call — not reviewing someone else's signal. You decide whether to trade or pass.

**Current Price:** {current_price:.5f}
**ATR(14):** {atr_value:.5f}
**Spread:** {spread:.1f} pips
**Account Balance:** ${balance:.2f}

**Indicators:**
{_dict_to_yaml(key_features)}

**Recent M1 Price Action:**
{m1_summary if m1_summary else "Not available"}

Based on this data, do you see a high-probability trade setup right now?

Respond with JSON:
- If you want to trade:
```json
{{
  "action": "trade",
  "direction": "buy" or "sell",
  "stop_loss": <price level>,
  "take_profit": <price level>,
  "confidence": 0.0 to 1.0,
  "reasoning": "Why this trade (2-3 sentences)"
}}
```
- If no setup:
```json
{{
  "action": "pass",
  "reasoning": "Why not (1 sentence)"
}}
```

Rules:
- Only trade if you see a CLEAR setup. When in doubt, pass.
- SL must be at a structural level (swing high/low, beyond ATR), not arbitrary.
- Minimum 1.5:1 reward:risk ratio.
- Consider the spread cost — it eats into your edge.
- Use ATR to size your SL/TP. Typical SL = 1.5 * ATR, TP = 2.5 * ATR."""


def _dict_to_yaml(d: dict) -> str:
    """Simple dict to YAML-ish string for prompt injection."""
    return "\n".join(f"  {k}: {v}" for k, v in d.items())
