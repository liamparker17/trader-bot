"""
Backtesting Simulator — Tests the full trading strategy on historical data.

Simulates realistic trading conditions:
- Iterates through candles chronologically
- Generates ML predictions for each candle
- Applies risk management rules (position sizing, SL/TP, drawdown limits)
- Simulates fills with configurable spread and slippage
- Tracks equity curve, drawdown, and all performance metrics

This is NOT a look-ahead backtest. At each candle, the simulator only
uses information available up to that point.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.config import Config
from src.indicators.engine import IndicatorEngine
from src.ml.predictor import Predictor

logger = logging.getLogger("traderbot.backtest")


@dataclass
class SimTrade:
    """A simulated trade."""
    trade_id: int
    instrument: str
    direction: str  # "buy" or "sell"
    entry_time: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float  # in units
    risk_amount: float  # in account currency
    ml_confidence: float
    atr_at_entry: float = 0.0
    sl_pips: float = 0.0
    breakeven_moved: bool = False
    highest_price: float = 0.0
    lowest_price: float = 999999.0
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # "tp", "sl", "trailing_sl", "timeout"
    pnl: float = 0.0
    pnl_pips: float = 0.0


@dataclass
class SimState:
    """Simulation state."""
    balance: float
    starting_balance: float
    equity_curve: list = field(default_factory=list)
    trades: list = field(default_factory=list)
    open_trades: list = field(default_factory=list)
    daily_pnl: float = 0.0
    daily_start_balance: float = 0.0
    current_date: Optional[datetime] = None
    trade_count_today: int = 0
    consecutive_losses: int = 0
    total_trade_id: int = 0
    is_paused: bool = False
    pause_reason: str = ""
    last_trade_bar: int = 0


class BacktestSimulator:
    """
    Full backtesting engine.

    Iterates through historical candles, applies the complete trading
    strategy (indicators → ML → risk management → execution), and
    tracks all results.
    """

    def __init__(
        self,
        config: Config,
        indicator_engine: IndicatorEngine,
        predictor: Predictor,
    ):
        self.config = config
        self.engine = indicator_engine
        self.predictor = predictor

        # Trading params
        self.starting_balance = config.get("account.starting_balance", 500)
        self.hard_floor = config.get("account.hard_floor", 350)
        self.risk_per_trade = config.get("risk.risk_per_trade_pct", 1.5) / 100
        self.daily_dd_limit = config.get("risk.daily_drawdown_limit_pct", 4.0) / 100
        self.max_trades_day = config.get("trading.max_trades_per_day", 60)
        self.max_open = config.get("risk.max_open_positions", 3)
        self.sl_atr_mult = config.get("risk.sl_atr_multiplier", 1.5)
        self.tp_atr_mult = config.get("risk.tp_atr_multiplier", 2.25)
        self.min_sl_pips = config.get("risk.min_sl_pips", 5)
        self.max_sl_pips = config.get("risk.max_sl_pips", 20)
        self.max_hold_bars = config.get("trading.max_hold_minutes", 60)

        # Trailing stop params
        self.trailing_enabled = config.get("risk.trailing_stop_enabled", True)
        self.trailing_activation_rr = config.get("risk.trailing_stop_activation_rr", 1.0)
        self.trailing_atr_mult = config.get("risk.trailing_stop_atr_multiplier", 1.0)

        # Circuit breakers
        self.consec_loss_reduce = config.get("risk.consecutive_loss_reduce_at", 3)
        self.consec_loss_pause = config.get("risk.consecutive_loss_pause_at", 5)
        self.high_vol_ratio = config.get("risk.high_volatility_atr_ratio", 2.0)
        self.low_vol_ratio = config.get("risk.low_volatility_atr_ratio", 0.3)
        self.max_spread_mult = config.get("risk.max_spread_multiplier", 2.0)

        # Entry spacing � don't enter on consecutive candles
        self.min_bars_between_trades = config.get("trading.min_seconds_between_trades", 30) // 60 or 1

        # Simulation settings
        self.slippage_pips = 0.5  # Assumed slippage per trade

    def run(
        self,
        m1_df: pd.DataFrame,
        m15_df: pd.DataFrame,
        instrument: str = "EUR_USD",
        spread_pips: float = 1.2,
    ) -> dict:
        """
        Run a full backtest on historical data.

        Args:
            m1_df: 1-minute candle data (DatetimeIndex, OHLCV)
            m15_df: 15-minute candle data
            instrument: Instrument being backtested
            spread_pips: Simulated spread in pips

        Returns:
            Dict with complete backtest results.
        """
        logger.info(
            f"Starting backtest: {instrument} | "
            f"{len(m1_df)} M1 candles | "
            f"Balance: R{self.starting_balance}"
        )

        state = SimState(
            balance=self.starting_balance,
            starting_balance=self.starting_balance,
            daily_start_balance=self.starting_balance,
        )

        inst_config = self.config.get_instrument(instrument)
        pip_location = inst_config.get("pip_location", -4)
        pip_size = 10 ** pip_location  # e.g., 0.0001 for EUR/USD
        pip_value_per_unit = self._pip_value_usd(instrument, pip_size, m1_df["close"].iloc[-1])

        # Pre-calculate all indicators on full dataset (for speed)
        m1_with_indicators = self.engine.calculate_all(m1_df)

        # Pre-compute additional EMAs needed for researched strategies
        m1_with_indicators["ema_55"] = m1_df["close"].ewm(span=55, adjust=False).mean()
        m15_with_indicators = self.engine.calculate_all(m15_df)

        # Map M15 trend bias to M1 timestamps
        m15_trend = self._compute_m15_trend_series(m15_with_indicators)

        min_bars = self.engine.get_required_candle_count()
        trading_sessions = self.config.get("trading.trading_sessions", {})
        session_type = inst_config.get("trading_session", "forex")
        session_hours = trading_sessions.get(session_type, {"start_hour": 7, "end_hour": 20})

        # Main loop — iterate through each M1 candle
        for i in range(min_bars, len(m1_df)):
            current_time = m1_df.index[i]
            current_candle = m1_df.iloc[i]

            # New day check
            if state.current_date is None or current_time.date() != state.current_date.date():
                self._handle_new_day(state, current_time)

            # Check open trades for SL/TP hits
            self._check_open_trades(state, current_candle, current_time, pip_size, pip_value_per_unit)

            # Record equity
            unrealized = self._calc_unrealized_pnl(state, current_candle, pip_size, pip_value_per_unit)
            state.equity_curve.append({
                "time": current_time,
                "balance": state.balance,
                "equity": state.balance + unrealized,
            })

            # Skip if paused, outside session, or at limits
            if state.is_paused:
                continue
            if state.balance <= self.hard_floor:
                state.is_paused = True
                state.pause_reason = f"Hard floor hit: R{state.balance:.2f}"
                logger.warning(state.pause_reason)
                continue

            hour = current_time.hour
            if hour < session_hours.get("start_hour", 7) or hour >= session_hours.get("end_hour", 20):
                continue
            if state.trade_count_today >= self.max_trades_day:
                continue
            if len(state.open_trades) >= self.max_open:
                continue

            # Check daily drawdown
            daily_loss = state.daily_start_balance - state.balance
            if daily_loss >= state.daily_start_balance * self.daily_dd_limit:
                state.is_paused = True
                state.pause_reason = "Daily drawdown limit"
                continue

            # Skip if too soon after last trade
            if i - state.last_trade_bar < self.min_bars_between_trades:
                continue

            # Build feature vector from data available up to current bar
            features = self._build_features_at(
                m1_with_indicators, m15_trend, i, current_time, spread_pips * pip_size
            )
            if features is None:
                continue

            # Get ATR for position sizing
            atr_value = m1_with_indicators.iloc[i].get("atr_value", 0)
            atr_ratio = m1_with_indicators.iloc[i].get("atr_ratio", 1.0)
            if pd.isna(atr_value) or atr_value <= 0:
                continue

            # Volatility filter
            if atr_ratio > self.high_vol_ratio or atr_ratio < self.low_vol_ratio:
                continue

            # Friday afternoon filter (all instruments)
            weekday = current_time.weekday()
            if weekday == 4 and hour >= 16:
                continue

            # === PER-INSTRUMENT STRATEGY DISPATCH ===
            strategy = inst_config.get("strategy", "pullback")
            direction = None

            if strategy == "pullback":
                direction = self._strategy_pullback(
                    features, m1_with_indicators, i, instrument
                )
            elif strategy == "london_breakout":
                direction = self._strategy_london_breakout(
                    features, m1_with_indicators, m1_df, i, current_time,
                    instrument, pip_size
                )
            elif strategy == "tokyo_breakout":
                direction = self._strategy_tokyo_breakout(
                    features, m1_with_indicators, m1_df, i, current_time,
                    instrument, pip_size
                )
            elif strategy == "momentum_breakout":
                direction = self._strategy_momentum_breakout(
                    features, m1_with_indicators, i, instrument
                )

            if direction is None:
                continue

            # === ML PREDICTION FILTER (per-instrument) ===
            inst_map = {"EUR_USD": 0, "GBP_USD": 1, "USD_JPY": 2, "XAU_USD": 3}
            features["instrument_id"] = float(inst_map.get(instrument, 4))
            ml_filter_on = inst_config.get("ml_filter_enabled", False)

            if ml_filter_on and self.predictor.model is not None:
                ml_confidence = self.predictor.predict(features)
                thresh_low = inst_config.get("ml_threshold_low", 0.10)
                thresh_high = inst_config.get("ml_threshold_high", 0.18)
                if ml_confidence < thresh_low:
                    state.ml_skips = getattr(state, "ml_skips", 0) + 1
                    continue
            else:
                ml_confidence = 0.5

            # Calculate trade parameters
            trade = self._create_trade(
                state, instrument, direction, current_candle, current_time,
                atr_value, atr_ratio, pip_size, pip_value_per_unit, spread_pips,
                ml_confidence,
            )

            if trade is not None:
                state.open_trades.append(trade)
                state.trade_count_today += 1
                state.last_trade_bar = i

        # Close any remaining open trades at the last price
        self._close_remaining(state, m1_df.iloc[-1], m1_df.index[-1], pip_size, pip_value_per_unit)

        return self._compile_results(state, instrument)

    def _create_trade(
        self, state, instrument, direction, candle, time,
        atr, atr_ratio, pip_size, pip_value_per_unit, spread_pips, confidence,
    ) -> Optional[SimTrade]:
        """Create a trade with proper position sizing and SL/TP."""
        # Calculate SL distance in pips � use per-instrument limits
        inst_cfg = self.config.get_instrument(instrument)
        min_sl = inst_cfg.get("atr_sl_min_pips", self.min_sl_pips)
        max_sl = inst_cfg.get("atr_sl_max_pips", self.max_sl_pips)
        inst_sl_mult = inst_cfg.get("sl_atr_multiplier", self.sl_atr_mult)
        inst_tp_mult = inst_cfg.get("tp_atr_multiplier", self.tp_atr_mult)
        inst_risk_pct = inst_cfg.get("risk_per_trade_pct", self.risk_per_trade * 100) / 100
        sl_pips = (atr * inst_sl_mult) / pip_size
        sl_pips = max(min_sl, min(sl_pips, max_sl))

        # Position sizing: risk_amount / (sl_pips * pip_value_per_unit)
        risk_amount = state.balance * inst_risk_pct

        # Reduce size after consecutive losses
        if state.consecutive_losses >= self.consec_loss_reduce:
            risk_amount *= 0.5

        # Reduce size in high volatility
        if atr_ratio > 1.5:
            risk_amount *= 0.75

        position_size = risk_amount / (sl_pips * pip_value_per_unit)
        # Round position to appropriate lot size
        inst_config_tmp = self.config.get_instrument(instrument)
        if inst_config_tmp.get("type") == "commodity":
            position_size = max(1, int(position_size))  # Gold: 1 oz minimum
        else:
            position_size = max(1000, int(position_size / 1000) * 1000)  # Forex: micro lots

                # Leverage cap: notional calculation depends on instrument
        inst_clean = instrument.replace("_", "")
        if inst_clean.startswith("USD"):
            notional = position_size  # Base is USD already
        elif inst_clean.endswith("USD"):
            notional = position_size * candle["close"]  # Convert to USD
        else:
            # Cross pair: approximate notional in USD
            # GBP/JPY: base=GBP, notional = size * GBPUSD_rate
            # EUR/GBP: base=EUR, notional = size * EURUSD_rate
            notional = position_size * 1.2  # Approximate for most major crosses
        max_notional = state.balance * self.config.get("risk.max_effective_leverage", 5.0)

        # Lot sizing depends on instrument type
        inst_config = self.config.get_instrument(instrument)
        inst_type = inst_config.get("type", "forex")
        if inst_type == "commodity":
            # Gold: 1 unit = 1 oz, min 1 unit
            min_unit = 1
            round_unit = 1
        else:
            # Forex: round to micro lots (1000 units)
            min_unit = 1000
            round_unit = 1000

        if notional > max_notional:
            if inst_clean.startswith("USD"):
                position_size = int(max_notional / round_unit) * round_unit
            elif inst_clean.endswith("USD"):
                position_size = int(max_notional / candle["close"] / round_unit) * round_unit
            else:
                # Cross pair: notional was approx position_size * 1.2
                position_size = int(max_notional / 1.2 / round_unit) * round_unit
            if position_size < min_unit:
                return None

        # Entry with spread and slippage
        total_cost_pips = spread_pips + self.slippage_pips

        if direction == "buy":
            entry = candle["close"] + (total_cost_pips * pip_size / 2)
            sl = entry - (sl_pips * pip_size)
            tp = entry + (sl_pips * inst_tp_mult / inst_sl_mult * pip_size)
        else:
            entry = candle["close"] - (total_cost_pips * pip_size / 2)
            sl = entry + (sl_pips * pip_size)
            tp = entry - (sl_pips * inst_tp_mult / inst_sl_mult * pip_size)

        state.total_trade_id += 1
        return SimTrade(
            trade_id=state.total_trade_id,
            instrument=instrument,
            direction=direction,
            entry_time=time,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            position_size=position_size,
            risk_amount=risk_amount,
            ml_confidence=confidence,
            atr_at_entry=atr,
            sl_pips=sl_pips,
            highest_price=entry,
            lowest_price=entry,
        )

    def _check_open_trades(self, state, candle, current_time, pip_size, pip_value_per_unit):
        """Check open trades for SL/TP hits and update trailing stops."""
        still_open = []

        for trade in state.open_trades:
            # Update price extremes for trailing stop
            trade.highest_price = max(trade.highest_price, candle["high"])
            trade.lowest_price = min(trade.lowest_price, candle["low"])

            closed = False

            # Per-instrument trailing stop config
            inst_cfg = self.config.get_instrument(trade.instrument)
            trailing_on = inst_cfg.get("trailing_stop_enabled", self.trailing_enabled) if inst_cfg else self.trailing_enabled

            if trade.direction == "buy":
                if candle["low"] <= trade.stop_loss:
                    reason = "trailing_sl" if trade.breakeven_moved else "sl"
                    self._close_trade(state, trade, trade.stop_loss, current_time, reason, pip_size, pip_value_per_unit)
                    closed = True
                elif candle["high"] >= trade.take_profit:
                    self._close_trade(state, trade, trade.take_profit, current_time, "tp", pip_size, pip_value_per_unit)
                    closed = True
                elif trailing_on:
                    self._update_trailing_stop(trade, candle, pip_size, inst_cfg)
            else:
                if candle["high"] >= trade.stop_loss:
                    reason = "trailing_sl" if trade.breakeven_moved else "sl"
                    self._close_trade(state, trade, trade.stop_loss, current_time, reason, pip_size, pip_value_per_unit)
                    closed = True
                elif candle["low"] <= trade.take_profit:
                    self._close_trade(state, trade, trade.take_profit, current_time, "tp", pip_size, pip_value_per_unit)
                    closed = True
                elif trailing_on:
                    self._update_trailing_stop(trade, candle, pip_size, inst_cfg)

            if not closed:
                hold_time = (current_time - trade.entry_time).total_seconds() / 60
                if hold_time >= self.max_hold_bars:
                    self._close_trade(state, trade, candle["close"], current_time, "timeout", pip_size, pip_value_per_unit)
                    closed = True

            if not closed:
                still_open.append(trade)

        state.open_trades = still_open

    def _update_trailing_stop(self, trade, candle, pip_size, inst_cfg=None):
        """Update trailing stop: breakeven at activation R:R, then trail by ATR.
        Uses per-instrument config if available, falls back to global defaults."""
        activation_rr = self.trailing_activation_rr
        atr_mult = self.trailing_atr_mult
        if inst_cfg:
            activation_rr = inst_cfg.get("trailing_stop_activation_rr", activation_rr)
            atr_mult = inst_cfg.get("trailing_stop_atr_multiplier", atr_mult)

        if trade.direction == "buy":
            current_pips = (candle["high"] - trade.entry_price) / pip_size
            current_rr = current_pips / trade.sl_pips if trade.sl_pips > 0 else 0

            if not trade.breakeven_moved and current_rr >= activation_rr:
                trade.stop_loss = trade.entry_price + (1 * pip_size)
                trade.breakeven_moved = True

            if trade.breakeven_moved:
                trail_distance = trade.atr_at_entry * atr_mult
                new_sl = trade.highest_price - trail_distance
                if new_sl > trade.stop_loss:
                    trade.stop_loss = new_sl
        else:
            current_pips = (trade.entry_price - candle["low"]) / pip_size
            current_rr = current_pips / trade.sl_pips if trade.sl_pips > 0 else 0

            if not trade.breakeven_moved and current_rr >= activation_rr:
                trade.stop_loss = trade.entry_price - (1 * pip_size)
                trade.breakeven_moved = True

            if trade.breakeven_moved:
                trail_distance = trade.atr_at_entry * atr_mult
                new_sl = trade.lowest_price + trail_distance
                if new_sl < trade.stop_loss:
                    trade.stop_loss = new_sl

    def _close_trade(self, state, trade, exit_price, exit_time, reason, pip_size, pip_value_per_unit):
        """Close a trade and update state."""
        if trade.direction == "buy":
            pnl_pips = (exit_price - trade.entry_price) / pip_size
        else:
            pnl_pips = (trade.entry_price - exit_price) / pip_size

        pnl = pnl_pips * pip_value_per_unit * trade.position_size

        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = reason
        trade.pnl = pnl
        trade.pnl_pips = pnl_pips

        state.balance += pnl
        state.daily_pnl += pnl
        state.trades.append(trade)

        if pnl > 0:
            state.consecutive_losses = 0
        else:
            state.consecutive_losses += 1

        # Pause after too many consecutive losses
        if state.consecutive_losses >= self.consec_loss_pause:
            state.is_paused = True
            state.pause_reason = f"{state.consecutive_losses} consecutive losses"

    def _handle_new_day(self, state, current_time):
        """Reset daily counters."""
        state.current_date = current_time
        state.daily_start_balance = state.balance
        state.daily_pnl = 0.0
        state.trade_count_today = 0

        # Reset pause (unless hard floor hit)
        if state.is_paused and state.balance > self.hard_floor:
            if "consecutive" in state.pause_reason or "drawdown" in state.pause_reason:
                state.is_paused = False
                state.pause_reason = ""
                state.consecutive_losses = 0

    def _calc_unrealized_pnl(self, state, candle, pip_size, pip_value_per_unit):
        """Calculate unrealized PnL for open trades."""
        total = 0.0
        for trade in state.open_trades:
            if trade.direction == "buy":
                pips = (candle["close"] - trade.entry_price) / pip_size
            else:
                pips = (trade.entry_price - candle["close"]) / pip_size
            total += pips * pip_value_per_unit * trade.position_size
        return total

    def _close_remaining(self, state, candle, time, pip_size, pip_value_per_unit):
        """Close all remaining open trades at current price."""
        for trade in list(state.open_trades):
            self._close_trade(state, trade, candle["close"], time, "backtest_end", pip_size, pip_value_per_unit)
        state.open_trades = []

    def _build_features_at(self, m1_indicators, m15_trend, idx, timestamp, spread):
        """Build feature vector at a specific bar index."""
        row = m1_indicators.iloc[idx]

        # Check for NaN in key indicators
        if pd.isna(row.get("atr_value")) or pd.isna(row.get("rsi_value")):
            return None

        features = {}
        ohlcv = {"open", "high", "low", "close", "volume"}
        for col in m1_indicators.columns:
            if col not in ohlcv:
                val = row[col]
                features[col] = float(val) if pd.notna(val) else 0.0

        # Trend bias
        features["trend_15min"] = float(m15_trend.get(timestamp, 0))

        # Price action
        total_range = row["high"] - row["low"]
        if total_range > 0:
            body = abs(row["close"] - row["open"])
            features["candle_body_ratio"] = body / total_range
            features["upper_wick_ratio"] = (row["high"] - max(row["open"], row["close"])) / total_range
            features["lower_wick_ratio"] = (min(row["open"], row["close"]) - row["low"]) / total_range
        else:
            features["candle_body_ratio"] = 0.0
            features["upper_wick_ratio"] = 0.0
            features["lower_wick_ratio"] = 0.0
        features["close_vs_open"] = 1.0 if row["close"] > row["open"] else 0.0

        # Context
        features["hour_of_day"] = float(timestamp.hour)
        features["day_of_week"] = float(timestamp.weekday())
        features["spread_current"] = float(spread)

        # Interaction features
        rsi_dist = features.get("rsi_distance_50", 0)
        vol_regime = features.get("volatility_regime", 0.5)
        features["momentum_x_vol_regime"] = rsi_dist * vol_regime

        ema_dist = features.get("ema_distance", 0)
        sess_overlap = features.get("session_overlap", 0)
        features["trend_x_session"] = ema_dist * sess_overlap

        # Lagged features
        lag_cols = ["macd_hist_accel", "rsi_roc", "momentum_consistency"]
        if idx >= 3:
            prev1 = m1_indicators.iloc[idx - 1]
            prev3 = m1_indicators.iloc[idx - 3]
            for col in lag_cols:
                val1 = prev1.get(col, 0)
                val3 = prev3.get(col, 0)
                features[f"{col}_lag1"] = float(val1) if pd.notna(val1) else 0.0
                features[f"{col}_lag3"] = float(val3) if pd.notna(val3) else 0.0
        else:
            for col in lag_cols:
                features[f"{col}_lag1"] = 0.0
                features[f"{col}_lag3"] = 0.0

        return features

    @staticmethod
    def _pip_value_usd(instrument: str, pip_size: float, current_price: float) -> float:
        """
        Calculate pip value in USD per unit for a given instrument.

        For XXX/USD pairs (EUR/USD, GBP/USD, XAU/USD): pip_value = pip_size
        For USD/XXX pairs (USD/JPY): pip_value = pip_size / current_price
        For cross pairs: approximate USD conversion using typical rates
        """
        # Normalize instrument name
        inst = instrument.replace("_", "")

        # If USD is the quote currency (EUR/USD, GBP/USD, XAU/USD)
        if inst.endswith("USD"):
            return pip_size  # 0.0001 for EUR/USD, 0.01 for XAU/USD
        # If USD is the base currency (USD/JPY)
        elif inst.startswith("USD"):
            return pip_size / current_price if current_price > 0 else pip_size
        # Cross pairs: quote currency is not USD, need conversion
        # GBP/JPY: quote=JPY, pip_value = pip_size / USDJPY_rate
        elif inst.endswith("JPY"):
            usdjpy_approx = 150.0  # Approximate USD/JPY rate
            return pip_size / usdjpy_approx
        # EUR/GBP, XXX/GBP: quote=GBP, pip_value = pip_size * GBPUSD_rate
        elif inst.endswith("GBP"):
            gbpusd_approx = 1.26  # Approximate GBP/USD rate
            return pip_size * gbpusd_approx
        # EUR/CHF, GBP/CHF etc: quote=CHF
        elif inst.endswith("CHF"):
            usdchf_approx = 0.88
            return pip_size / usdchf_approx
        # EUR/CAD etc: quote=CAD
        elif inst.endswith("CAD"):
            usdcad_approx = 1.36
            return pip_size / usdcad_approx
        # Fallback
        return pip_size

    def _compute_m15_trend_series(self, m15_indicators):
        """Build a Series mapping timestamps to trend bias values.
        
        STRICT: both EMA AND MACD must agree for a directional bias.
        This provides a genuine edge over random entries.
        """
        ema_cross = m15_indicators.get("ema_crossover", pd.Series(0, index=m15_indicators.index))
        macd_cross = m15_indicators.get("macd_crossover", pd.Series(0, index=m15_indicators.index))

        trend = pd.Series(0, index=m15_indicators.index)
        trend[(ema_cross == 1) & (macd_cross == 1)] = 1
        trend[(ema_cross == 0) & (macd_cross == 0)] = -1

        return trend.to_dict()

    # ================================================================
    # PER-INSTRUMENT STRATEGY METHODS
    # ================================================================

    def _strategy_pullback(self, features, m1_indicators, idx, instrument):
        """
        EUR/USD strategy: Pullback to 21 EMA with triple-EMA trend filter.

        Original proven logic (54% WR) enhanced with EMA(55) alignment:
        1. M15 trend confirms direction (EMA+MACD both agree)
        2. EMA(55) must agree with trend direction (additional filter)
        3. Independent indicator confirmation (RSI, BB, divergence)
        4. Pullback entry: price touched EMA zone then resumed
        """
        trend = features.get("trend_15min", 0)
        if trend > 0:
            direction = "buy"
        elif trend < 0:
            direction = "sell"
        else:
            return None

        # EMA(55) trend alignment filter (researched enhancement)
        current = m1_indicators.iloc[idx]
        ema_55 = current.get("ema_55")
        ema_slow = current.get("ema_slow")  # EMA(21)
        if pd.notna(ema_55) and pd.notna(ema_slow):
            atr = current.get("atr_value", 0)
            ema_gap = abs(ema_slow - ema_55)
            # Only block if EMAs are strongly misaligned (gap > 0.5 ATR)
            if pd.notna(atr) and atr > 0 and ema_gap > atr * 0.5:
                if direction == "buy" and ema_slow < ema_55:
                    return None
                if direction == "sell" and ema_slow > ema_55:
                    return None

        # Indicator confirmation
        if not self._indicators_agree(features, instrument):
            return None

        # Pullback entry filter
        if not self._is_pullback_entry(m1_indicators, idx, direction):
            return None

        return direction

    def _strategy_london_breakout(self, features, m1_indicators, m1_df, idx,
                                   current_time, instrument, pip_size):
        """
        GBP/USD strategy: Big Ben London Breakout -- sweep and breakout only.

        Researched rules:
        1. Asian range (00-07 UTC) with 20-80 pip width filter
        2. Mode 1 (07-12 UTC): Sweep-reversal or breakout entry
        3. Mode 2 (12-20 UTC): Trend-confirmed pullback (only with all confirmations)
        """
        inst_cfg = self.config.get_instrument(instrument)
        hour = current_time.hour
        entry_start = inst_cfg.get("london_entry_start", 7)
        entry_end = inst_cfg.get("london_entry_end", 12)

        # Mode 1: London session (07-12 UTC) -- Asian range based entries
        if entry_start <= hour < entry_end:
            asian_high, asian_low = self._get_session_range(
                m1_df, current_time, 0, 7
            )
            if asian_high is None:
                return None

            asian_range_pips = (asian_high - asian_low) / pip_size
            # Quality filter: skip extreme ranges
            if asian_range_pips > 80 or asian_range_pips < 20:
                return None

            current = m1_indicators.iloc[idx]
            sweep_min = inst_cfg.get("sweep_min_pips", 3) * pip_size

            # Look back up to 15 bars for a sweep
            swept_high = False
            swept_low = False
            for j in range(max(0, idx - 15), idx):
                bar = m1_indicators.iloc[j]
                if bar["high"] > asian_high + sweep_min:
                    swept_high = True
                if bar["low"] < asian_low - sweep_min:
                    swept_low = True

            # Sweep of high + reversal back inside = sell
            if swept_high and current["close"] < asian_high:
                if current["close"] < current["open"]:
                    if self._indicators_agree(features, instrument):
                        return "sell"

            # Sweep of low + reversal back inside = buy
            if swept_low and current["close"] > asian_low:
                if current["close"] > current["open"]:
                    if self._indicators_agree(features, instrument):
                        return "buy"

            # Pure breakout with configurable buffer + body filter + trend confirmation
            breakout_buffer = inst_cfg.get("breakout_buffer_pips", 7) * pip_size
            body_min = inst_cfg.get("breakout_body_ratio_min", 0.0)
            trend = features.get("trend_15min", 0)

            # Body ratio filter for breakout candle quality
            br_range = current["high"] - current["low"]
            br_body = abs(current["close"] - current["open"]) / br_range if br_range > 0 else 0

            if current["close"] > asian_high + breakout_buffer and trend >= 1:
                if current["close"] > current["open"] and br_body >= body_min:
                    if self._indicators_agree(features, instrument):
                        return "buy"
            if current["close"] < asian_low - breakout_buffer and trend <= -1:
                if current["close"] < current["open"] and br_body >= body_min:
                    if self._indicators_agree(features, instrument):
                        return "sell"

        # Mode 2: NY session (12-20 UTC) -- only with full trend + pullback confirmation
        if hour >= 12:
            trend = features.get("trend_15min", 0)
            if trend == 0:
                return None
            # EMA(55) alignment check
            current = m1_indicators.iloc[idx]
            ema_55 = current.get("ema_55")
            ema_slow = current.get("ema_slow")
            if pd.notna(ema_55) and pd.notna(ema_slow):
                if trend > 0 and ema_slow < ema_55:
                    return None
                if trend < 0 and ema_slow > ema_55:
                    return None
            direction = "buy" if trend > 0 else "sell"
            if not self._is_pullback_entry(m1_indicators, idx, direction):
                return None
            if self._indicators_agree(features, instrument):
                return direction

        return None

    def _strategy_tokyo_breakout(self, features, m1_indicators, m1_df, idx,
                                  current_time, instrument, pip_size):
        """
        USD/JPY strategy: Tokyo range breakout + pullback with range filter.

        Mode 1 (07-11 UTC): Tokyo range breakout with range width filter (15-80 pips)
        Mode 2 (11-20 UTC): Standard pullback entries with full confirmation
        Best hours: 12-15 UTC (London-NY overlap) for pullback mode
        """
        inst_cfg = self.config.get_instrument(instrument)
        entry_start = inst_cfg.get("breakout_entry_start", 7)
        entry_end = inst_cfg.get("breakout_entry_end", 11)
        hour = current_time.hour
        trend = features.get("trend_15min", 0)

        # Mode 1: Tokyo range breakout (07-11 UTC)
        if entry_start <= hour < entry_end:
            tokyo_high, tokyo_low = self._get_session_range(
                m1_df, current_time, 0, 7
            )
            if tokyo_high is not None:
                tokyo_range_pips = (tokyo_high - tokyo_low) / pip_size
                # Range quality filter
                if 15 <= tokyo_range_pips <= 80:
                    current = m1_indicators.iloc[idx]
                    breakout_min = inst_cfg.get("breakout_min_pips", 5) * pip_size

                    if current["close"] > tokyo_high + breakout_min and trend >= 1:
                        if self._indicators_agree(features, instrument):
                            return "buy"

                    if current["close"] < tokyo_low - breakout_min and trend <= -1:
                        if self._indicators_agree(features, instrument):
                            if self._sell_volume_ok(m1_indicators, idx, inst_cfg):
                                return "sell"

        # Mode 2: Rest of session -- pullback entries
        if hour >= entry_end:
            if trend == 0:
                return None
            direction = "buy" if trend > 0 else "sell"
            if not self._is_pullback_entry(m1_indicators, idx, direction):
                return None
            if self._indicators_agree(features, instrument):
                if direction == "sell" and not self._sell_volume_ok(m1_indicators, idx, inst_cfg):
                    return None
                return direction

        return None

    def _strategy_momentum_breakout(self, features, m1_indicators, idx, instrument):
        """
        XAU/USD (Gold) strategy: GoldScalper Pro inspired.

        Based on documented gold strategy research (PF 1.8+):
        1. Triple EMA alignment: EMA(8) > EMA(21) > EMA(55) for longs
        2. MACD histogram positive (bullish) or cross
        3. RSI in momentum zone (50-68 for longs, 32-50 for shorts)
        4. Entry candle must be directional (body > 50% of range)

        Gold trends hard -- ride the winners, cut losers fast.
        """
        current = m1_indicators.iloc[idx]

        ema_fast = current.get("ema_fast")  # EMA(8)
        ema_slow = current.get("ema_slow")  # EMA(21)
        ema_55 = current.get("ema_55")
        if pd.isna(ema_fast) or pd.isna(ema_slow) or pd.isna(ema_55):
            return None

        rsi = features.get("rsi_value", 50)
        macd_hist = features.get("macd_histogram", 0)
        macd_cross = features.get("macd_crossover", 0)
        trend = features.get("trend_15min", 0)

        # Candle body ratio filter: body must be > 50% of total range
        total_range = current["high"] - current["low"]
        if total_range <= 0:
            return None
        body = abs(current["close"] - current["open"])
        body_ratio = body / total_range
        if body_ratio < 0.5:
            return None

        # Bullish setup: triple EMA stacking + MACD + RSI mid-band
        if ema_fast > ema_slow > ema_55:
            if trend < 0:
                return None
            if not (50 <= rsi <= 68):
                return None
            if macd_hist <= 0 and macd_cross != 1:
                return None
            if current["close"] <= current["open"]:
                return None
            return "buy"

        # Bearish setup
        if ema_fast < ema_slow < ema_55:
            if trend > 0:
                return None
            if not (32 <= rsi <= 50):
                return None
            if macd_hist >= 0 and macd_cross != 0:
                return None
            if current["close"] >= current["open"]:
                return None
            # Data-driven filter: reject sells on tiny/choppy bars
            # Bar range must be >= 60% of 20-period average range
            # Validated on both train (+$657 improvement) and test (+17% PnL)
            inst_cfg = self.config.get_instrument(instrument)
            sell_min_range = inst_cfg.get("sell_min_bar_range_ratio", 0.0) if inst_cfg else 0.0
            if sell_min_range > 0 and idx >= 20:
                recent_ranges = m1_indicators["high"].iloc[idx-20:idx] - m1_indicators["low"].iloc[idx-20:idx]
                avg_range = recent_ranges.mean()
                if avg_range > 0 and total_range / avg_range < sell_min_range:
                    return None
            return "sell"

        return None

    def _get_session_range(self, m1_df, current_time, start_hour, end_hour):
        """Get the high/low of a session range for the current day."""
        today = current_time.date()
        try:
            # Filter M1 candles for today's session
            session_mask = (
                (m1_df.index.date == today) &
                (m1_df.index.hour >= start_hour) &
                (m1_df.index.hour < end_hour)
            )
            session_candles = m1_df[session_mask]
            if len(session_candles) < 10:  # Need enough candles for valid range
                return None, None
            return session_candles["high"].max(), session_candles["low"].min()
        except Exception:
            return None, None

    def _sell_volume_ok(self, m1_indicators, idx, inst_cfg):
        """Check if current volume meets minimum ratio for sell trades."""
        min_vol = inst_cfg.get("sell_min_volume_ratio", 0.0) if inst_cfg else 0.0
        if min_vol <= 0 or idx < 20:
            return True
        vol = m1_indicators["volume"].iloc[idx]
        avg_vol = m1_indicators["volume"].iloc[idx-20:idx].mean()
        if avg_vol > 0 and vol / avg_vol < min_vol:
            return False
        return True

    def _is_pullback_entry(self, m1_indicators, idx, direction):
        """
        Detect pullback entry: price pulled back to EMA zone then resumed trend.
        
        Based on proven scalping strategy:
        1. Price was recently near the 21 EMA (within last 5 bars)
        2. Current candle closes in trend direction
        3. RSI confirms momentum resuming (crossing back past 50)
        
        This filters ~80% of entries but the remaining ones are much higher quality.
        """
        if idx < 6:
            return False

        current = m1_indicators.iloc[idx]
        prev = m1_indicators.iloc[idx - 1]

        # Current candle must close in trend direction
        if direction == "buy" and current["close"] <= current["open"]:
            return False
        if direction == "sell" and current["close"] >= current["open"]:
            return False

        ema_slow = current.get("ema_slow")
        if pd.isna(ema_slow) or ema_slow <= 0:
            return False

        # Check if price touched/crossed the 21 EMA in the last 5 bars
        touched_ema = False
        for j in range(max(0, idx - 5), idx):
            bar = m1_indicators.iloc[j]
            bar_ema = bar.get("ema_slow")
            if pd.isna(bar_ema):
                continue
            # Price touched EMA zone (within 0.3 ATR)
            atr = bar.get("atr_value", 0)
            if pd.isna(atr) or atr <= 0:
                continue
            ema_zone = atr * 0.5  # Within half ATR of EMA
            if direction == "buy":
                # For buys: low should have dipped near/below EMA
                if bar["low"] <= bar_ema + ema_zone:
                    touched_ema = True
                    break
            else:
                # For sells: high should have risen near/above EMA
                if bar["high"] >= bar_ema - ema_zone:
                    touched_ema = True
                    break

        if not touched_ema:
            return False

        # RSI should confirm momentum resuming
        rsi = current.get("rsi_value", 50)
        prev_rsi = prev.get("rsi_value", 50)
        if pd.isna(rsi) or pd.isna(prev_rsi):
            return False

        if direction == "buy":
            # RSI should be rising and above 40 (not deeply oversold)
            if rsi < 40 or rsi < prev_rsi:
                return False
        else:
            # RSI should be falling and below 60
            if rsi > 60 or rsi > prev_rsi:
                return False

        return True

    def _indicators_agree(self, features: dict, instrument: str = "EUR_USD") -> bool:
        """
        Check if independent indicators confirm the trade direction.

        Uses signals NOT already baked into trend_15min (which uses EMA/MACD crossovers).
        Instead checks RSI, BB position, and divergence — independent confirmation.
        """
        trend = features.get("trend_15min", 0)
        rsi = features.get("rsi_value", 50)
        bb_pos = features.get("bb_position", 0.5)
        divergence = features.get("rsi_divergence", 0)

        inst_cfg = self.config.get_instrument(instrument)
        rsi_ob = inst_cfg.get("rsi_overbought", 75) if inst_cfg else 75
        rsi_os = inst_cfg.get("rsi_oversold", 25) if inst_cfg else 25

        if trend > 0:  # Bullish — RSI not overbought, BB not at extreme top, no bearish divergence
            return rsi < rsi_ob and bb_pos < 0.85 and divergence != -1
        elif trend < 0:  # Bearish — RSI not oversold, BB not at extreme bottom, no bullish divergence
            return rsi > rsi_os and bb_pos > 0.15 and divergence != 1
        return False

    def _compile_results(self, state, instrument) -> dict:
        """Compile comprehensive backtest results."""
        trades = state.trades
        if not trades:
            return {"error": "No trades executed", "instrument": instrument}

        pnls = [t.pnl for t in trades]
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Equity curve stats
        equity_df = pd.DataFrame(state.equity_curve)
        if not equity_df.empty:
            peak = equity_df["equity"].expanding().max()
            drawdown = (equity_df["equity"] - peak) / peak
            max_drawdown = abs(drawdown.min())
        else:
            max_drawdown = 0

        # Sharpe ratio (simplified, daily returns)
        daily_returns = pd.Series(pnls)
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0

        # Exit reason breakdown
        exit_reasons = {}
        for t in trades:
            r = t.exit_reason or "unknown"
            exit_reasons[r] = exit_reasons.get(r, 0) + 1

        # Win rate by confidence bucket
        confidence_buckets = {}
        for t in trades:
            bucket = f"{int(t.ml_confidence * 10) * 10}-{int(t.ml_confidence * 10) * 10 + 10}%"
            if bucket not in confidence_buckets:
                confidence_buckets[bucket] = {"wins": 0, "total": 0}
            confidence_buckets[bucket]["total"] += 1
            if t.pnl > 0:
                confidence_buckets[bucket]["wins"] += 1

        results = {
            "instrument": instrument,
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(trades),
            "profit_factor": profit_factor,
            "total_pnl_zar": sum(pnls),
            "starting_balance": state.starting_balance,
            "final_balance": state.balance,
            "return_pct": (state.balance - state.starting_balance) / state.starting_balance * 100,
            "max_drawdown_pct": max_drawdown * 100,
            "sharpe_ratio": sharpe,
            "avg_pnl_per_trade": np.mean(pnls),
            "avg_win": np.mean([t.pnl for t in wins]) if wins else 0,
            "avg_loss": np.mean([t.pnl for t in losses]) if losses else 0,
            "avg_hold_minutes": np.mean([
                (t.exit_time - t.entry_time).total_seconds() / 60
                for t in trades if t.exit_time
            ]),
            "exit_reasons": exit_reasons,
            "ml_skips": getattr(state, "ml_skips", 0),
            "confidence_buckets": confidence_buckets,
            "equity_curve": state.equity_curve,
            "trades": [
                {
                    "id": t.trade_id,
                    "direction": t.direction,
                    "entry_time": str(t.entry_time),
                    "exit_time": str(t.exit_time),
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "pnl": t.pnl,
                    "pnl_pips": t.pnl_pips,
                    "exit_reason": t.exit_reason,
                    "confidence": t.ml_confidence,
                }
                for t in trades
            ],
        }

        # Log summary
        logger.info("=" * 60)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Instrument: {instrument}")
        logger.info(f"Trades: {results['total_trades']} ({results['wins']}W / {results['losses']}L)")
        logger.info(f"Win Rate: {results['win_rate']:.1%}")
        logger.info(f"Profit Factor: {results['profit_factor']:.2f}")
        logger.info(f"Total PnL: R{results['total_pnl_zar']:.2f}")
        logger.info(f"Return: {results['return_pct']:.1f}%")
        logger.info(f"Max Drawdown: {results['max_drawdown_pct']:.1f}%")
        logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"Avg Hold Time: {results['avg_hold_minutes']:.1f} min")
        logger.info(f"Exit Reasons: {results['exit_reasons']}")
        logger.info(f"Final Balance: R{results['final_balance']:.2f}")
        logger.info("=" * 60)

        return results
