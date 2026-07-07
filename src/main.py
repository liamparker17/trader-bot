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
from datetime import datetime, timezone
from typing import Callable, Optional
from pathlib import Path

from src.config import load_config
from src.utils.instance_lock import InstanceLock
from src.data.mt5_client import MT5Client
from src.data.collector import DataCollector
from src.indicators.engine import IndicatorEngine
from src.ml.predictor import Predictor
from src.ml.evaluator import Evaluator
from src.risk.manager import RiskManager, TradeRequest
from src.risk.drawdown_tracker import session_boundary
from src.execution.executor import Executor
from src.growth.reinvestment import GrowthManager
from src.growth.milestone_tracker import MilestoneTracker
from src.monitoring.trade_journal import TradeJournal
from src.monitoring.performance import PerformanceTracker
from src.monitoring.telegram_bot import TelegramBot
from src.control.effective_config import EffectiveConfig
from src.control.queue import ControlQueue

logger = logging.getLogger("traderbot")


class DailySummaryScheduler:
    """
    Decides when the once-daily Telegram summary is due, at the same
    trading session boundary (default 21:00 UTC) used elsewhere for
    daily/weekly drawdown resets (see `session_boundary()` in
    drawdown_tracker.py).

    Config fallback chain (matches Task 4's session-boundary pattern):
        trading.session_reset_hour_utc -> risk.session_boundary_hour_utc -> 21

    Guards against double-firing two ways:
    - An in-memory `last fired` boundary date, for the common case of a
      single long-running process.
    - A persisted journal event row (event_type="daily_summary_sent"),
      so a process restart shortly after the boundary doesn't re-send.
    """

    EVENT_TYPE = "daily_summary_sent"

    def __init__(self, config, journal, clock: Optional[Callable[[], datetime]] = None):
        self.config = config
        self.journal = journal
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.reset_hour = config.get(
            "trading.session_reset_hour_utc",
            config.get("risk.session_boundary_hour_utc", 21),
        )
        self._last_fired_date: Optional[str] = None

    def due(self) -> Optional[str]:
        """
        Return the boundary date string ("YYYY-MM-DD") if a summary is
        due and hasn't already been sent for that boundary, else None.
        """
        now = self.clock()
        boundary = session_boundary(now, self.reset_hour)
        boundary_date = boundary.date().isoformat()

        if self._last_fired_date == boundary_date:
            return None

        # Survive process restarts: check the journal for a prior
        # daily_summary_sent event recorded for this boundary date.
        try:
            events = self.journal.get_events(event_type=self.EVENT_TYPE, limit=10)
            if events is not None and not events.empty and "message" in events.columns:
                if boundary_date in set(events["message"]):
                    self._last_fired_date = boundary_date
                    return None
        except Exception as e:
            logger.debug(f"DailySummaryScheduler: journal lookback failed: {e}")

        return boundary_date

    def mark_fired(self, boundary_date: str):
        """Record that the summary for `boundary_date` has been sent."""
        self._last_fired_date = boundary_date
        try:
            self.journal.record_event(
                self.EVENT_TYPE, boundary_date, {"boundary_date": boundary_date}
            )
        except Exception as e:
            logger.warning(f"Failed to record {self.EVENT_TYPE} event: {e}")


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
        self.instance_lock = InstanceLock()
        self.daily_summary_scheduler = None
        self.effective_config = None
        self.control_queue = None

        # Main-loop catch-all (item S): rate-limit Telegram alerts to at
        # most one per exception type every N seconds, so a repeatedly
        # failing iteration doesn't spam Telegram.
        self._loop_error_last_alert: dict = {}
        self._loop_error_cooldown_seconds = 300

        # Task 8: consecutive balance/equity refresh failures — escalated
        # from debug to warning after 3 in a row, reset to 0 on success.
        self._balance_refresh_failures = 0

    def setup(self):
        """Initialize all modules."""
        logger.info("Initializing TraderBot...")
        logger.info(f"Environment: {self.config.broker_environment}")
        logger.info(f"Instruments: {self.config.get_enabled_instruments()}")

        # Single-instance lock — must succeed before any MT5 connection
        # is attempted, so two bot processes can never trade the same
        # account concurrently.
        if not self.instance_lock.acquire():
            logger.critical(
                "Could not acquire single-instance lock. "
                "Another TraderBot process appears to be running. Exiting."
            )
            sys.exit(1)

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
        self.daily_summary_scheduler = DailySummaryScheduler(self.config, self.journal)

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

        # Executor — wire a best-effort Telegram alert callback so
        # close_all()'s per-batch alert (including any close failures)
        # actually reaches the operator in production.
        self.executor = Executor(
            self.config, self.client, self.risk_manager,
            alert_callback=self._on_executor_alert,
        )

        # EffectiveConfig overlay (Task 8): settings.yaml <+ tunes overlay
        # <+ safety_floor.yaml (last wins). Read through it at the
        # whitelisted use sites so a `tb tune` command takes effect on the
        # next read, no restart required.
        self.effective_config = EffectiveConfig.load()
        self.risk_manager.sizer.effective_config = self.effective_config
        self.predictor.effective_config = self.effective_config

        # Control queue (Task 8): polls control/inbox/*.cmd.json once per
        # main-loop iteration for pause/resume/tune/revert/status_snapshot
        # commands from the `tb` CLI (Task 11+).
        self.control_queue = ControlQueue(
            config=self.config,
            journal=self.journal,
            telegram=self.telegram,
            risk_manager=self.risk_manager,
            effective_config=self.effective_config,
            collector=self.collector,
            executor=self.executor,
            client=self.client,
        )

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
            starting = self.config.get("account.starting_balance_zar", 500)
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

        last_status_log = 0
        last_balance_check = 0.0
        cached_balance = None
        cached_equity = None
        try:
            while self.running:
                # Catch-all for a single iteration: an unhandled exception
                # here must never kill the loop silently. Log the full
                # traceback, fire a rate-limited Telegram alert, and let
                # the loop continue on the next iteration.
                try:
                    # Main loop runs at ~1 second intervals
                    # Actual trading decisions happen in _on_candle_complete callback
                    time.sleep(1)

                    # Control queue (Task 8): process any pending
                    # pause/resume/tune/revert/status_snapshot commands.
                    # Wrapped separately so a queue-processing bug can
                    # never take down the trading loop.
                    if self.control_queue is not None:
                        try:
                            self.control_queue.poll_once()
                        except Exception as e:
                            logger.error(f"Control queue poll failed: {e}", exc_info=True)

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

                    # Refresh cached balance/equity every 5s (not every ~1s
                    # iteration) so the emergency checks below don't hammer
                    # account_info on the broker.
                    if now - last_balance_check >= 5:
                        last_balance_check = now
                        cached_balance, cached_equity = self._refresh_balance_cache(
                            cached_balance, cached_equity
                        )

                    # Session boundary (21:00 UTC) reset + daily-drawdown
                    # emergency check, skipped while broker_down (stale
                    # cached_balance/cached_equity). See
                    # _check_session_and_drawdown() docstring for details,
                    # including its own isolated exception handling.
                    if cached_balance is not None and not self.collector.broker_down:
                        self._check_session_and_drawdown(cached_balance, cached_equity)

                    # Check for PERMANENT emergency shutdown (hard floor breach /
                    # circuit-breaker shutdown) — distinct from the resumable
                    # daily-drawdown block above. Stops the bot entirely.
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
                    self._handle_loop_exception(e)

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

        # Task 8: skip new-entry evaluation entirely while the broker
        # connection is down — the health thread hasn't confirmed a fresh
        # price/account feed, so any signal here would be evaluated
        # against stale data. Debug-level only: this can fire every
        # candle during an outage and must not spam the log.
        if self.collector.broker_down:
            logger.debug(f"Skipping signal evaluation for {instrument}: broker_down")
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

    def _on_executor_alert(self, event: str, data: dict):
        """
        Best-effort Telegram bridge for Executor alerts (close_all's batch
        summary, and failed orders with no real MT5 ticket). Never allowed
        to raise into the executor — a failed alert must not block
        position management.
        """
        try:
            if event == "close_all":
                reason = data.get("reason", "unknown")
                closed = data.get("closed", 0)
                requested = data.get("requested", 0)
                msg = f"close_all ({reason}): {closed}/{requested} positions closed"
                failures = data.get("failures") or []
                if failures:
                    msg += f" | {len(failures)} FAILED: {failures}"
                self.telegram._send(f"<b>Position close-all:</b> {msg}")
                self.journal.record_event("close_all", msg, data)
            elif event == "order_failed":
                instrument = data.get("instrument", "?")
                direction = data.get("direction", "?")
                reason = data.get("reason", "unknown")
                msg = f"Order failed for {instrument} {direction}: {reason} (no MT5 ticket returned)"
                self.telegram._send(f"⚠️ <b>Order Failed</b>\n{msg}")
                self.journal.record_event("order_failed", msg, data)
        except Exception as e:
            logger.error(f"_on_executor_alert failed: {e}")

    def _refresh_balance_cache(self, cached_balance, cached_equity):
        """
        Refresh the cached balance/equity from the broker. On failure,
        keeps the previous cached values (so callers keep working off the
        last-known-good numbers) and escalates the log level from debug
        to warning once 3 consecutive refreshes have failed — a single
        blip shouldn't page anyone, but a sustained outage should be
        visible. The counter resets to 0 on the next success.

        Returns (balance, equity) — unchanged from the inputs on failure.
        """
        try:
            summary = self.client.get_account_summary()
            self._balance_refresh_failures = 0
            return summary["balance"], summary["equity"]
        except Exception as e:
            self._balance_refresh_failures += 1
            if self._balance_refresh_failures >= 3:
                logger.warning(
                    f"Balance/equity refresh failed "
                    f"{self._balance_refresh_failures}x in a row: {e}"
                )
            else:
                logger.debug(f"Balance/equity refresh failed: {e}")
            return cached_balance, cached_equity

    def _check_session_and_drawdown(self, cached_balance: float, cached_equity: float):
        """
        Session boundary (21:00 UTC) reset + daily-drawdown emergency
        check — runs every iteration (via the main loop), independent of
        whether a trade signal fired. Lifts the resumable daily block and
        resets the consecutive-loss counter at the boundary; on a fresh
        daily-drawdown breach, closes all positions but keeps the bot
        RUNNING with entries blocked until the next boundary (see
        RiskManager.entries_blocked). Also fires the once-per-day Telegram
        summary at the same boundary.

        Callers should skip calling this entirely while broker_down is
        True (cached_balance/cached_equity would be stale) — see run().

        This method has its OWN try/except, distinct from the generic
        per-iteration catch-all (_handle_loop_exception): an exception
        here specifically engages RiskManager's manual pause (blocking
        new entries until a `resume` command or code fix + restart) and
        sends a dedicated Telegram alert, rather than just being logged
        and rate-limited like any other loop exception.
        """
        try:
            self.risk_manager.check_session_boundary(cached_balance)
            if self.risk_manager.check_drawdown_emergency(cached_balance, cached_equity):
                logger.critical(
                    "DAILY DRAWDOWN BREACH — closing all positions; "
                    "bot stays running, entries blocked until next "
                    "session boundary"
                )
                self.executor.close_all("daily_drawdown")
                try:
                    drawdown_pct = self.risk_manager.drawdown.get_daily_drawdown_pct(
                        cached_balance
                    )
                    self.telegram.daily_stop(cached_balance, drawdown_pct)
                except Exception:
                    pass

            self._maybe_send_daily_summary(cached_balance)
        except Exception as e:
            logger.error(
                f"Session boundary / drawdown emergency check failed — "
                f"pausing new entries: {e}",
                exc_info=True,
            )
            self.risk_manager.set_manual_pause(
                f"session boundary/drawdown check exception: {e}"
            )
            try:
                self.telegram._send(
                    f"<b>Session/drawdown check failed</b> — new entries paused: {e}"
                )
            except Exception:
                pass

    def _maybe_send_daily_summary(self, balance: float):
        """
        Fire the once-per-day Telegram summary at the 21:00 UTC session
        boundary (or whatever `trading.session_reset_hour_utc` /
        `risk.session_boundary_hour_utc` resolves to). Best-effort — must
        never raise into the main loop.
        """
        if not self.daily_summary_scheduler:
            return
        try:
            boundary_date = self.daily_summary_scheduler.due()
            if boundary_date is None:
                return

            summary = self.performance.get_summary()
            self.telegram.daily_summary(
                date=boundary_date,
                trades=summary.get("total_trades", 0),
                wins=summary.get("wins", 0),
                losses=summary.get("losses", 0),
                pnl=summary.get("total_pnl", 0.0),
                balance=balance,
                win_rate=summary.get("win_rate", 0.0),
                max_drawdown=summary.get("max_drawdown_pct", 0.0),
            )
            self.daily_summary_scheduler.mark_fired(boundary_date)
        except Exception as e:
            logger.error(f"Daily summary failed: {e}")

    def _handle_loop_exception(self, exc: Exception):
        """
        Catch-all for an unhandled exception inside a single main-loop
        iteration (item S). Logs the full traceback, sends a Telegram
        alert rate-limited to once per 5 minutes per exception type, and
        lets the loop continue — a single bad iteration must never kill
        the bot silently.
        """
        exc_type = type(exc).__name__
        logger.error(
            f"Unhandled exception in main loop iteration ({exc_type}): {exc}",
            exc_info=True,
        )

        now = time.time()
        last_alert = self._loop_error_last_alert.get(exc_type, 0.0)
        if now - last_alert >= self._loop_error_cooldown_seconds:
            self._loop_error_last_alert[exc_type] = now
            try:
                self.telegram.bot_error(exc_type, str(exc))
                self.journal.record_event(
                    "bot_error", str(exc), {"exception_type": exc_type}
                )
            except Exception as e:
                logger.error(f"_handle_loop_exception alert failed: {e}")

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

        # Release single-instance lock last, once everything else is torn down
        if self.instance_lock:
            self.instance_lock.release()

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
