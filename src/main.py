"""
TraderBot — Main Entry Point

Usage:
    python -m src.main              # Run the bot (live/demo trading)
    python -m src.main --backtest   # Run backtesting
    python -m src.main --fetch-data # Fetch historical data only
    python -m src.main --dashboard  # Launch Streamlit dashboard
"""

import argparse
import signal
import sys
import time
import logging
import logging.handlers
import threading
from pathlib import Path

from src.config import load_config
from src.data.mt5_client import MT5Client
from src.data.collector import DataCollector
from src.indicators.engine import IndicatorEngine
from src.ml.predictor import Predictor
from src.ml.evaluator import Evaluator
from src.risk.manager import RiskManager, TradeRequest
from src.execution.executor import Executor
from src.growth.reinvestment import GrowthManager
from src.growth.milestone_tracker import MilestoneTracker
from src.monitoring.trade_journal import TradeJournal
from src.monitoring.performance import PerformanceTracker
from src.monitoring.telegram_bot import TelegramBot
from src.ai.analyst import AIAnalyst
from src.ai.shadow_trader import ShadowTrader
from src.ai.approval_queue import ApprovalQueue

logger = logging.getLogger("traderbot")


class TraderBot:
    """Main orchestrator that ties all modules together."""

    def __init__(self):
        self.config = load_config()
        self.running = False

        # All modules
        self.client = None
        self.collector = None
        self.engine = None
        self.predictor = None
        self.evaluator = None
        self.risk_manager = None
        self.executor = None
        self.growth = None
        self.milestones = None
        self.journal = None
        self.performance = None
        self.telegram = None
        self.analyst = None
        self.shadow = None
        self.approval_queue = None

    def setup(self):
        """Initialize all modules."""
        logger.info("Initializing TraderBot...")
        logger.info(f"Environment: {self.config.broker_environment}")
        logger.info(f"Instruments: {self.config.get_enabled_instruments()}")

        # Validate credentials
        if not self.config.mt5_login:
            logger.error("MT5_LOGIN not set. Copy .env.example to .env and fill in your credentials.")
            sys.exit(1)
        if not self.config.mt5_password:
            logger.error("MT5_PASSWORD not set.")
            sys.exit(1)

        # Initialize MT5 client and connect
        self.client = MT5Client(self.config)
        if not self.client.connect():
            logger.error("Failed to connect to MT5 terminal. Is MetaTrader 5 running?")
            sys.exit(1)
        self.engine = IndicatorEngine(self.config)
        self.predictor = Predictor(self.config)
        self.evaluator = Evaluator(self.config)
        self.risk_manager = RiskManager(self.config)
        self.journal = TradeJournal(self.config)
        self.performance = PerformanceTracker(self.journal)
        self.telegram = TelegramBot(self.config)

        # AI Analyst (optional Claude layer)
        self.analyst = AIAnalyst(self.config)
        self.approval_queue = ApprovalQueue(self.config)
        self.shadow = ShadowTrader(self.config, self.analyst)

        # Data collector with candle callback
        self.collector = DataCollector(
            self.config, self.client,
            on_candle_complete=self._on_candle_complete,
        )

        # Growth tracking
        self.milestones = MilestoneTracker(
            self.config,
            on_milestone=self._on_milestone,
        )
        self.growth = GrowthManager(self.config)

        # Executor
        self.executor = Executor(self.config, self.client, self.risk_manager)

        # Load ML model
        if not self.predictor.load_model():
            logger.warning("No ML model found. Run backtest.runner first to train a model.")

        # Initialize with account balance
        try:
            balance = self.client.get_account_balance()
            self.risk_manager.initialize(balance)
            self.growth.initialize(balance)
            self.milestones.check(balance)
            self.evaluator.load_state()
            logger.info(f"Account balance: ${balance:.2f}")
        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            logger.info("Continuing with configured starting balance...")
            starting = self.config.get("account.starting_balance", 500)
            self.risk_manager.initialize(starting)
            self.growth.initialize(starting)

        logger.info("TraderBot initialized successfully.")

    def run(self):
        """Main trading loop."""
        self.running = True
        logger.info("TraderBot starting...")

        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        # Send startup notification
        try:
            balance = self.client.get_account_balance()
            self.telegram.bot_started(balance, self.config.broker_environment)
        except Exception:
            pass

        # Warm up candle builder with recent history
        logger.info("Warming up with historical data...")
        self.collector.warm_up_candle_builder(candle_count=200)

        # Start price streaming
        self.collector.start_streaming()

        # Start reconciliation thread
        recon_thread = threading.Thread(
            target=self._reconciliation_loop,
            daemon=True,
            name="reconciliation",
        )
        recon_thread.start()

        # Start Telegram command listener for /approve, /reject, /pending, /shadow
        if self.telegram and self.approval_queue:
            self.telegram.start_command_listener(
                approval_queue=self.approval_queue,
                shadow_trader=self.shadow,
            )

        # AI Analyst: pre-session briefing
        if self.analyst.is_active:
            try:
                instruments = self.config.get_enabled_instruments()
                balance = self.client.get_account_balance()
                briefing = self.analyst.session_briefing(
                    session="startup",
                    instruments=instruments,
                    market_data={},  # Will be populated once candles are available
                    balance=balance,
                )
                if briefing:
                    logger.info(f"AI Session Briefing: regime={briefing.get('regime', 'unknown')}")
                    for inst, bias in briefing.get("bias", {}).items():
                        logger.info(f"  {inst}: {bias}")
                    self.telegram._send(
                        f"<b>AI Briefing:</b> {briefing.get('regime', '?')} regime\n"
                        + "\n".join(f"  {k}: {v}" for k, v in briefing.get("bias", {}).items())
                    )
            except Exception as e:
                logger.debug(f"AI briefing skipped: {e}")

        last_status_log = 0
        try:
            while self.running:
                # Main loop runs at ~1 second intervals
                # Actual trading decisions happen in _on_candle_complete callback
                time.sleep(1)

                # Hourly performance status log
                now = time.time()
                if now - last_status_log >= 3600:
                    last_status_log = now
                    try:
                        bal = self.client.get_account_balance()
                        summary = self.performance.get_summary()
                        logger.info(
                            f"[STATUS] Balance: ${bal:.2f} | "
                            f"Trades: {summary.get('total_trades', 0)} | "
                            f"WR: {summary.get('win_rate', 0):.1%} | "
                            f"PnL: ${summary.get('total_pnl', 0):.2f} | "
                            f"PF: {summary.get('profit_factor', 0):.2f}"
                        )
                    except Exception:
                        pass

                # Check for emergency shutdown
                if self.risk_manager.close_all_signal():
                    logger.critical("Emergency shutdown signal — closing all positions")
                    results = self.executor.close_all("emergency_shutdown")
                    try:
                        balance = self.client.get_account_balance()
                        self.telegram.emergency_stop(balance, "Hard floor breach")
                    except Exception:
                        pass
                    self.running = False
                    break

        except Exception as e:
            logger.critical(f"Unhandled exception in main loop: {e}", exc_info=True)
        finally:
            self.shutdown()

    def _on_candle_complete(self, instrument: str, timeframe: str, candle):
        """
        Callback fired when a new candle completes.
        This is where trading decisions are made.
        """
        # Only trade on M1 candle completions
        if timeframe != "M1":
            return

        if not self.running:
            return

        try:
            logger.info(f"M1 candle complete: {instrument}")
            self._evaluate_trade_signal(instrument)
        except Exception as e:
            logger.error(f"Error evaluating signal for {instrument}: {e}", exc_info=True)

    def _evaluate_trade_signal(self, instrument: str):
        """
        Evaluate whether to trade on a new M1 candle.

        Uses per-instrument strategy dispatch:
        - EUR_USD: Pullback to 21 EMA with triple-EMA(55) trend filter
        - GBP_USD: Big Ben London Breakout (sweep/breakout + NY pullback)
        - USD_JPY: Tokyo box breakout + pullback
        - XAU_USD: GoldScalper Pro (triple EMA + MACD + RSI momentum)
        """
        # Get candle data
        m1_df = self.collector.get_candles_df(instrument, "M1", count=200)
        m15_df = self.collector.get_candles_df(instrument, "M15", count=50)

        if m1_df.empty or len(m1_df) < 60:
            return

        # --- Session and time filters ---
        from datetime import datetime, timezone
        now_utc = datetime.now(timezone.utc)
        inst_config = self.config.get_instrument(instrument)
        session_type = inst_config.get("trading_session", "forex") if inst_config else "forex"
        trading_sessions = self.config.get("trading.trading_sessions", {})
        session_hours = trading_sessions.get(session_type, {"start_hour": 7, "end_hour": 20})
        start_h = session_hours.get("start_hour", 7)
        end_h = session_hours.get("end_hour", 20)
        hour = now_utc.hour

        # Session boundary filter
        if hour < start_h or hour >= end_h:
            return
        # Avoid first/last hour of forex session (choppy, wide spreads)
        # Gold session is only 4 hours, don't skip any
        if session_type != "gold" and (hour == start_h or hour == end_h - 1):
            return
        # Avoid Friday afternoon (weekend gap risk)
        if now_utc.weekday() == 4 and hour >= 16:
            return

        # Get current spread
        try:
            spread = self.client.get_spread(instrument)
        except Exception:
            return

        # Build feature vector
        features = self.engine.build_feature_vector(m1_df, m15_df, spread)
        if features is None:
            return

        # Compute indicators with extras (EMA 55) for strategy logic
        m1_with_ind = self.engine.calculate_all_with_extras(m1_df)
        current = m1_with_ind.iloc[-1]

        # Per-instrument pip size
        pip_location = inst_config.get("pip_location", -4) if inst_config else -4
        pip_size = 10 ** pip_location

        # === Execute any user-approved Claude recommendations ===
        current_price = float(m1_df.iloc[-1]["close"])
        if self.approval_queue:
            self._execute_approved_trades(instrument, current_price, features)

        # === PER-INSTRUMENT STRATEGY DISPATCH ===
        strategy = inst_config.get("strategy", "pullback") if inst_config else "pullback"
        direction = None

        if strategy == "pullback":
            direction = self._strategy_pullback(features, m1_with_ind, instrument)
        elif strategy == "london_breakout":
            direction = self._strategy_london_breakout(
                features, m1_with_ind, m1_df, now_utc, instrument, pip_size, inst_config
            )
        elif strategy == "tokyo_breakout":
            direction = self._strategy_tokyo_breakout(
                features, m1_with_ind, m1_df, now_utc, instrument, pip_size, inst_config
            )
        elif strategy == "momentum_breakout":
            direction = self._strategy_momentum_breakout(features, m1_with_ind)

        if direction is None:
            return

        logger.info(f"SIGNAL: {instrument} {direction} via {strategy}")

        # Per-instrument ML filtering (matching backtest behavior)
        ml_filter_on = inst_config.get("ml_filter_enabled", False) if inst_config else False
        ml_confidence = 0.5

        if ml_filter_on and self.predictor.model is not None:
            features["instrument_id"] = float({"EUR_USD": 0, "GBP_USD": 1, "USD_JPY": 2, "XAU_USD": 3}.get(instrument, 4))
            ml_confidence = self.predictor.predict(features)
            thresh_low = inst_config.get("ml_threshold_low", 0.10)
            thresh_high = inst_config.get("ml_threshold_high", 0.18)
            if ml_confidence < thresh_low:
                logger.info(f"ML skip {instrument}: {ml_confidence:.4f} < {thresh_low}")
                return
            logger.info(f"ML pass {instrument}: {ml_confidence:.4f} (thresh={thresh_low}/{thresh_high})")
        elif self.predictor.model is None:
            # Rules-only: strategies already confirmed entry
            pass
        else:
            # ML disabled for this instrument, use rules-only
            pass

        atr_value = features.get("atr_value", 0)
        atr_ratio = features.get("atr_ratio", 1.0)
        if atr_value <= 0:
            return

        # --- AI Analyst review (optional) ---
        if self.analyst.is_active:
            try:
                balance = self.client.get_account_balance()
                daily_pnl = self.performance.get_summary().get("total_pnl", 0.0)
                open_positions = [
                    {"instrument": t.instrument, "direction": t.direction,
                     "entry_price": t.entry_price, "unrealized_pnl": t.current_pnl}
                    for t in self.executor.open_trades.values()
                ] if self.executor.open_trades else []

                review = self.analyst.review_trade(
                    instrument=instrument,
                    direction=direction,
                    ml_confidence=ml_confidence,
                    features=features,
                    open_positions=open_positions,
                    daily_pnl=daily_pnl,
                    balance=balance,
                    atr_value=atr_value,
                    spread=spread,
                )

                if review["source"] == "ai_analyst":
                    logger.info(
                        f"AI Review: {'APPROVED' if review['approved'] else 'REJECTED'} "
                        f"({review['confidence']:.0%}) — {review['reasoning']}"
                    )
                    if review.get("warnings"):
                        for w in review["warnings"]:
                            logger.warning(f"AI Warning: {w}")

                    require_approval = self.config.get("ai_analyst.require_approval", False)
                    if not review["approved"] and require_approval:
                        logger.info(f"Trade BLOCKED by AI Analyst: {instrument} {direction}")
                        return

                    # Apply size modification if suggested
                    size_mult = review.get("modifications", {}).get("reduce_size")
                    if size_mult and isinstance(size_mult, (int, float)) and 0 < size_mult < 1:
                        atr_ratio = atr_ratio * size_mult
                        logger.info(f"AI Analyst reduced position size by {size_mult:.0%}")

            except Exception as e:
                logger.debug(f"AI review skipped: {e}")

        # Execute through the executor (which goes through risk manager)
        trade = self.executor.execute_signal(
            instrument=instrument,
            direction=direction,
            ml_confidence=ml_confidence,
            atr_value=atr_value,
            atr_ratio=atr_ratio,
        )

        if trade:
            trend = features.get("trend_15min", 0)
            # Log to journal
            self.journal.record_trade(
                trade_id=trade.trade_id,
                instrument=trade.instrument,
                direction=trade.direction,
                units=trade.units,
                entry_price=trade.entry_price,
                entry_time=trade.entry_time,
                stop_loss=trade.stop_loss,
                take_profit=trade.take_profit,
                ml_confidence=trade.ml_confidence,
                model_version=self.predictor.version or "",
                trend_15min=int(trend),
            )

            # Telegram alert
            self.telegram.trade_opened(
                instrument=trade.instrument,
                direction=trade.direction,
                units=trade.units,
                entry_price=trade.entry_price,
                stop_loss=trade.stop_loss,
                take_profit=trade.take_profit,
                confidence=trade.ml_confidence,
                risk_amount=trade.risk_amount,
            )

    # ================================================================
    # PER-INSTRUMENT STRATEGY METHODS
    # ================================================================

    def _strategy_pullback(self, features, m1_with_ind, instrument="EUR_USD"):
        """
        EUR/USD: Pullback to 21 EMA with triple-EMA(55) trend filter.
        """
        import pandas as pd

        trend = features.get("trend_15min", 0)
        if trend > 0:
            direction = "buy"
        elif trend < 0:
            direction = "sell"
        else:
            return None

        # EMA(55) alignment filter
        current = m1_with_ind.iloc[-1]
        ema_55 = current.get("ema_55")
        ema_slow = current.get("ema_slow")
        atr = current.get("atr_value", 0)
        if pd.notna(ema_55) and pd.notna(ema_slow) and pd.notna(atr) and atr > 0:
            ema_gap = abs(ema_slow - ema_55)
            if ema_gap > atr * 0.5:
                if direction == "buy" and ema_slow < ema_55:
                    return None
                if direction == "sell" and ema_slow > ema_55:
                    return None

        if not self._check_indicators_agree(features, instrument):
            return None

        # Pullback entry
        m1_df_subset = m1_with_ind[["open", "high", "low", "close", "volume"]].copy()
        if not self.engine.is_pullback_entry(m1_df_subset, direction):
            return None

        return direction

    def _strategy_london_breakout(self, features, m1_with_ind, m1_df, current_time,
                                   instrument, pip_size, inst_config):
        """
        GBP/USD: Big Ben London Breakout -- sweep/breakout + NY pullback.
        """
        import pandas as pd

        hour = current_time.hour
        entry_start = inst_config.get("london_entry_start", 7)
        entry_end = inst_config.get("london_entry_end", 12)

        # Mode 1: London session (07-12 UTC)
        if entry_start <= hour < entry_end:
            asian_high, asian_low = self.engine.get_session_range(
                m1_df, current_time, 0, 7
            )
            if asian_high is None:
                return None

            asian_range_pips = (asian_high - asian_low) / pip_size
            if asian_range_pips > 80 or asian_range_pips < 20:
                return None

            current = m1_with_ind.iloc[-1]
            sweep_min = inst_config.get("sweep_min_pips", 3) * pip_size
            trend = features.get("trend_15min", 0)

            # Look back up to 15 bars for a sweep
            swept_high = False
            swept_low = False
            lookback = min(15, len(m1_with_ind) - 1)
            for j in range(-lookback - 1, -1):
                bar = m1_with_ind.iloc[j]
                if bar["high"] > asian_high + sweep_min:
                    swept_high = True
                if bar["low"] < asian_low - sweep_min:
                    swept_low = True

            # Sweep of high + reversal = sell
            if swept_high and current["close"] < asian_high:
                if current["close"] < current["open"]:
                    if self._check_indicators_agree(features, instrument):
                        return "sell"

            # Sweep of low + reversal = buy
            if swept_low and current["close"] > asian_low:
                if current["close"] > current["open"]:
                    if self._check_indicators_agree(features, instrument):
                        return "buy"

            # Pure breakout with configurable buffer + body filter + trend confirmation
            breakout_buffer = inst_config.get("breakout_buffer_pips", 7) * pip_size
            body_min = inst_config.get("breakout_body_ratio_min", 0.0)

            # Body ratio filter for breakout candle quality
            br_range = current["high"] - current["low"]
            br_body = abs(current["close"] - current["open"]) / br_range if br_range > 0 else 0

            if current["close"] > asian_high + breakout_buffer and trend >= 1:
                if current["close"] > current["open"] and br_body >= body_min:
                    if self._check_indicators_agree(features, instrument):
                        return "buy"
            if current["close"] < asian_low - breakout_buffer and trend <= -1:
                if current["close"] < current["open"] and br_body >= body_min:
                    if self._check_indicators_agree(features, instrument):
                        return "sell"

        # Mode 2: NY session pullback (12-20 UTC)
        if hour >= 12:
            trend = features.get("trend_15min", 0)
            if trend == 0:
                return None
            current = m1_with_ind.iloc[-1]
            ema_55 = current.get("ema_55")
            ema_slow = current.get("ema_slow")
            if pd.notna(ema_55) and pd.notna(ema_slow):
                if trend > 0 and ema_slow < ema_55:
                    return None
                if trend < 0 and ema_slow > ema_55:
                    return None
            direction = "buy" if trend > 0 else "sell"
            m1_df_subset = m1_with_ind[["open", "high", "low", "close", "volume"]].copy()
            if not self.engine.is_pullback_entry(m1_df_subset, direction):
                return None
            if self._check_indicators_agree(features, instrument):
                return direction

        return None

    def _strategy_tokyo_breakout(self, features, m1_with_ind, m1_df, current_time,
                                  instrument, pip_size, inst_config):
        """
        USD/JPY: Tokyo box breakout + pullback entries.
        """
        hour = current_time.hour
        trend = features.get("trend_15min", 0)
        entry_start = inst_config.get("breakout_entry_start", 7)
        entry_end = inst_config.get("breakout_entry_end", 11)

        # Mode 1: Tokyo range breakout (07-11 UTC)
        if entry_start <= hour < entry_end:
            tokyo_high, tokyo_low = self.engine.get_session_range(
                m1_df, current_time, 0, 7
            )
            if tokyo_high is not None:
                tokyo_range_pips = (tokyo_high - tokyo_low) / pip_size
                if 15 <= tokyo_range_pips <= 80:
                    current = m1_with_ind.iloc[-1]
                    breakout_min = inst_config.get("breakout_min_pips", 5) * pip_size

                    if current["close"] > tokyo_high + breakout_min and trend >= 1:
                        if self._check_indicators_agree(features, instrument):
                            return "buy"

                    if current["close"] < tokyo_low - breakout_min and trend <= -1:
                        if self._check_indicators_agree(features, instrument):
                            if self._sell_volume_ok(m1_with_ind, inst_config):
                                return "sell"

        # Mode 2: Rest of session -- pullback entries
        if hour >= entry_end:
            if trend == 0:
                return None
            direction = "buy" if trend > 0 else "sell"
            m1_df_subset = m1_with_ind[["open", "high", "low", "close", "volume"]].copy()
            if not self.engine.is_pullback_entry(m1_df_subset, direction):
                return None
            if self._check_indicators_agree(features, instrument):
                if direction == "sell" and not self._sell_volume_ok(m1_with_ind, inst_config):
                    return None
                return direction

        return None

    def _strategy_momentum_breakout(self, features, m1_with_ind):
        """
        XAU/USD: GoldScalper Pro -- triple EMA + MACD + RSI momentum.
        """
        import pandas as pd

        current = m1_with_ind.iloc[-1]
        ema_fast = current.get("ema_fast")
        ema_slow = current.get("ema_slow")
        ema_55 = current.get("ema_55")
        if pd.isna(ema_fast) or pd.isna(ema_slow) or pd.isna(ema_55):
            return None

        rsi = features.get("rsi_value", 50)
        macd_hist = features.get("macd_histogram", 0)
        macd_cross = features.get("macd_crossover", 0)
        trend = features.get("trend_15min", 0)

        # Body ratio filter
        total_range = current["high"] - current["low"]
        if total_range <= 0:
            return None
        body = abs(current["close"] - current["open"])
        if body / total_range < 0.5:
            return None

        # Bullish
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

        # Bearish
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
            inst_cfg = self.config.get_instrument("XAU_USD")
            sell_min_range = inst_cfg.get("sell_min_bar_range_ratio", 0.0) if inst_cfg else 0.0
            if sell_min_range > 0 and len(m1_with_ind) >= 20:
                recent = m1_with_ind.iloc[-20:]
                avg_range = (recent["high"] - recent["low"]).mean()
                if avg_range > 0 and total_range / avg_range < sell_min_range:
                    return None
            return "sell"

        return None

    def _sell_volume_ok(self, m1_with_ind, inst_config):
        """Check if current volume meets minimum ratio for sell trades."""
        min_vol = inst_config.get("sell_min_volume_ratio", 0.0)
        if min_vol <= 0 or len(m1_with_ind) < 20:
            return True
        vol = m1_with_ind["volume"].iloc[-1]
        avg_vol = m1_with_ind["volume"].iloc[-20:].mean()
        if avg_vol > 0 and vol / avg_vol < min_vol:
            return False
        return True

    def _check_indicators_agree(self, features: dict, instrument: str = "EUR_USD") -> bool:
        """
        Check if independent indicators confirm the trade direction.

        Uses per-instrument RSI thresholds. Checks RSI, BB position, and
        divergence — independent of the EMA/MACD crossovers used in trend_15min.
        """
        trend = features.get("trend_15min", 0)
        rsi = features.get("rsi_value", 50)
        bb_pos = features.get("bb_position", 0.5)
        divergence = features.get("rsi_divergence", 0)

        inst_config = self.config.get_instrument(instrument)
        rsi_ob = inst_config.get("rsi_overbought", 75) if inst_config else 75
        rsi_os = inst_config.get("rsi_oversold", 25) if inst_config else 25

        if trend > 0:  # Bullish
            return rsi < rsi_ob and bb_pos < 0.85 and divergence != -1
        elif trend < 0:  # Bearish
            return rsi > rsi_os and bb_pos > 0.15 and divergence != 1
        return False

    def _send_shadow_summary(self, result: dict):
        """Send shadow retrospective results via Telegram."""
        shadow_pnl = result.get("shadow_pnl_pips", 0)
        bot_pnl = result.get("bot_pnl_pips", 0)
        shadow_emoji = "\u2705" if shadow_pnl > 0 else "\U0001f534"
        bot_emoji = "\u2705" if bot_pnl > 0 else "\U0001f534"

        text = (
            f"\U0001f9e0 <b>Claude Shadow Report — {result['date']}</b>\n\n"
            f"<b>Claude's Trades (paper):</b>\n"
            f"  Trades: {result['shadow_total']} "
            f"({result['shadow_wins']}W / {result['shadow_losses']}L)\n"
            f"  {shadow_emoji} PnL: {shadow_pnl:+.1f} pips\n\n"
            f"<b>Bot's Actual Trades:</b>\n"
            f"  Trades: {result['bot_trades']}\n"
            f"  {bot_emoji} PnL: {bot_pnl:+.1f} pips\n\n"
        )

        diff = shadow_pnl - bot_pnl
        if diff > 0:
            text += f"\U0001f4a1 Claude was ahead by {diff:+.1f} pips today."
        elif diff < 0:
            text += f"\U0001f916 Bot was ahead by {abs(diff):.1f} pips today."
        else:
            text += "Dead even today."

        # Show individual trades
        trades = result.get("shadow_trades", [])
        if trades:
            text += "\n\n<b>Claude's Trades:</b>"
            for t in trades[:5]:
                pnl = t.get("pnl_pips", 0)
                emoji = "\u2705" if pnl > 0 else "\u274c"
                text += (
                    f"\n{emoji} {t['instrument']} {t['direction'].upper()} "
                    f"{pnl:+.1f} pips ({t.get('exit_reason', '?')})"
                )

        self.telegram._send(text)

    def _execute_approved_trades(self, instrument: str, current_price: float, features: dict):
        """Pick up user-approved Claude recommendations and execute them."""
        try:
            approved = self.approval_queue.get_approved_trades()
            for trade in approved:
                if trade.instrument != instrument:
                    continue

                # Check price hasn't moved too far from recommendation
                inst_config = self.config.get_instrument(instrument)
                pip_loc = inst_config.get("pip_location", -4) if inst_config else -4
                pip_size = 10 ** pip_loc
                max_slip = self.config.get("ai_analyst.approval_max_slippage_pips", 5)
                slippage_pips = abs(current_price - trade.entry_price) / pip_size

                if slippage_pips > max_slip:
                    logger.info(
                        f"Approved trade {trade.id} skipped: price moved "
                        f"{slippage_pips:.1f} pips (max {max_slip})"
                    )
                    self.approval_queue.mark_executed(trade.id)
                    continue

                atr_value = features.get("atr_value", 0)
                atr_ratio = features.get("atr_ratio", 1.0)
                if atr_value <= 0:
                    continue

                logger.info(
                    f"Executing approved Claude trade: {trade.id} | "
                    f"{instrument} {trade.direction.upper()} | "
                    f"Conf: {trade.confidence:.0%}"
                )

                result = self.executor.execute_signal(
                    instrument=instrument,
                    direction=trade.direction,
                    ml_confidence=trade.confidence,
                    atr_value=atr_value,
                    atr_ratio=atr_ratio,
                )

                self.approval_queue.mark_executed(trade.id)

                if result:
                    self.journal.record_trade(
                        trade_id=result.trade_id,
                        instrument=result.instrument,
                        direction=result.direction,
                        units=result.units,
                        entry_price=result.entry_price,
                        entry_time=result.entry_time,
                        stop_loss=result.stop_loss,
                        take_profit=result.take_profit,
                        ml_confidence=trade.confidence,
                        model_version=f"claude_{trade.id}",
                        trend_15min=int(features.get("trend_15min", 0)),
                    )
                    self.telegram.trade_opened(
                        instrument=result.instrument,
                        direction=result.direction,
                        units=result.units,
                        entry_price=result.entry_price,
                        stop_loss=result.stop_loss,
                        take_profit=result.take_profit,
                        confidence=trade.confidence,
                        risk_amount=result.risk_amount,
                    )

        except Exception as e:
            logger.debug(f"Approved trade execution error: {e}")

    def _reconciliation_loop(self):
        """Background thread: sync with broker and check positions."""
        interval = self.config.get("monitoring.reconciliation_interval_seconds", 60)
        while self.running:
            try:
                self.executor.sync_with_broker()

                # Check for closed trades and update tracking
                balance = self.client.get_account_balance()
                self.milestones.check(balance)
                self.growth.update(balance)

                # End-of-day shadow retrospective
                from datetime import datetime as dt_cls, timezone as tz_cls
                hour_utc = dt_cls.now(tz_cls.utc).hour
                if self.shadow and self.shadow.should_run(hour_utc):
                    try:
                        logger.info("Running end-of-day shadow retrospective...")
                        result = self.shadow.run_retrospective(
                            collector=self.collector,
                            engine=self.engine,
                            journal=self.journal,
                            mt5_client=self.client,
                        )
                        if result and self.telegram:
                            self._send_shadow_summary(result)
                    except Exception as e:
                        logger.error(f"Shadow retrospective failed: {e}", exc_info=True)

                # Expire pending approvals
                if self.approval_queue:
                    self.approval_queue.expire_old()

                # Check ML retrain triggers
                should_retrain, reason = self.evaluator.should_retrain()
                if should_retrain:
                    logger.info(f"ML retrain suggested: {reason}")
                    self.journal.record_event("retrain_trigger", reason)

            except Exception as e:
                logger.error(f"Reconciliation error: {e}")

            time.sleep(interval)

    def _on_milestone(self, milestone: float, balance: float, message: str):
        """Callback when a milestone is reached."""
        self.telegram.milestone_reached(milestone, balance, message)
        self.journal.record_event("milestone", message, {"milestone": milestone, "balance": balance})

    def fetch_historical_data(self):
        """Fetch and cache historical data for all instruments."""
        if not self.client:
            self.client = MT5Client(self.config)
            self.client.connect()
        from src.data.historical_loader import HistoricalLoader
        loader = HistoricalLoader(self.config, self.client)
        loader.fetch_all_instruments("M1")
        loader.fetch_all_instruments("M15")
        logger.info("Historical data fetch complete.")

    def run_backtest(self):
        """Run backtesting."""
        from backtest.runner import run_backtest
        run_backtest()

    def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down TraderBot...")
        self.running = False

        # Stop streaming
        if self.collector:
            self.collector.stop_streaming()

        # Close all open positions
        if self.executor and self.executor.open_trades:
            logger.info("Closing open positions...")
            self.executor.close_all("shutdown")

        # Save evaluator state
        if self.evaluator:
            self.evaluator.save_state()

        # Send shutdown notification
        if self.telegram:
            try:
                balance = self.client.get_account_balance() if self.client else 0
                self.telegram.bot_stopped(balance, "shutdown")
            except Exception:
                pass

        # Close API sessions
        if self.client:
            self.client.close()

        logger.info("TraderBot shutdown complete.")

    def _handle_shutdown(self, signum, frame):
        """Handle SIGINT/SIGTERM."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False


def main():
    # Log to both console and file for persistent review
    log_dir = Path(__file__).resolve().parent.parent / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "traderbot.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10_000_000, backupCount=5, encoding="utf-8"
            ),
        ],
    )

    parser = argparse.ArgumentParser(description="TraderBot — Forex & Gold Scalping Bot")
    parser.add_argument("--backtest", action="store_true", help="Run backtesting mode")
    parser.add_argument("--fetch-data", action="store_true", help="Fetch historical data only")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard")
    args = parser.parse_args()

    if args.dashboard:
        import subprocess
        dashboard_path = "src/monitoring/dashboard/app.py"
        port = 8501
        logger.info(f"Launching dashboard on port {port}...")
        subprocess.run(["streamlit", "run", dashboard_path, "--server.port", str(port)])
        return

    bot = TraderBot()
    bot.setup()

    if args.fetch_data:
        bot.fetch_historical_data()
    elif args.backtest:
        bot.run_backtest()
    else:
        bot.run()


if __name__ == "__main__":
    main()
