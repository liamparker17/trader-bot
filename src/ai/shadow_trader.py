"""
Shadow Trader — End-of-day retrospective analysis by Claude.

Instead of calling Claude on every candle (expensive), this runs ONCE
at the end of each trading session. Claude receives the full day's
price data and identifies trades it would have taken, with exact
entry/exit prices. We then verify those against actual candle data
to compute real paper P&L.

Cost: ~1-2 API calls per day (~$0.05-0.15/day) vs hundreds in real-time.
"""

import json
import logging
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from src.config import Config, PROJECT_ROOT

logger = logging.getLogger("traderbot.ai.shadow")


class ShadowTrader:
    """
    End-of-day retrospective shadow trading.

    At session end, sends the day's candle data to Claude and asks:
    "What trades would you have taken today?"

    Claude returns specific entries with timestamps, SL, TP.
    We verify each against actual price data to get real paper P&L.
    Results are logged to SQLite for long-term tracking.
    """

    def __init__(self, config: Config, analyst):
        self.config = config
        self.analyst = analyst
        self._last_run_date: str | None = None

        # SQLite
        db_path = config.get("monitoring.trade_journal_db", "data/trade_logs/trades.db")
        self.db_path = PROJECT_ROOT / db_path
        self._init_db()

    def _init_db(self):
        """Create shadow trades table."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS shadow_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    instrument TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_time TEXT NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    exit_price REAL,
                    exit_time TEXT,
                    pnl_pips REAL,
                    exit_reason TEXT,
                    confidence REAL,
                    reasoning TEXT,
                    verified INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS shadow_daily_summary (
                    date TEXT PRIMARY KEY,
                    total_trades INTEGER,
                    wins INTEGER,
                    losses INTEGER,
                    total_pnl_pips REAL,
                    win_rate REAL,
                    instruments_traded TEXT,
                    claude_commentary TEXT,
                    bot_trades INTEGER,
                    bot_pnl_pips REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
        logger.info("Shadow trader DB initialized.")

    def should_run(self, hour_utc: int) -> bool:
        """Check if it's time for end-of-day analysis."""
        run_hour = self.config.get("ai_analyst.shadow_review_hour_utc", 21)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if hour_utc == run_hour and self._last_run_date != today:
            return True
        return False

    def run_retrospective(
        self,
        collector,
        engine,
        journal,
        mt5_client,
    ):
        """
        Run the end-of-day retrospective analysis.

        Args:
            collector: DataCollector (for candle data)
            engine: IndicatorEngine (to compute indicators on historical data)
            journal: TradeJournal (to compare with bot's actual trades)
            mt5_client: MT5Client (to fetch today's candles if buffer insufficient)
        """
        if not self.analyst.is_active:
            logger.info("Shadow retrospective skipped: AI Analyst not active.")
            return

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._last_run_date = today

        instruments = self.config.get_enabled_instruments()
        all_shadow_trades = []

        for instrument in instruments:
            try:
                trades = self._analyze_instrument(
                    instrument, collector, engine, mt5_client
                )
                all_shadow_trades.extend(trades)
            except Exception as e:
                logger.error(f"Shadow analysis failed for {instrument}: {e}")

        # Get bot's actual trades for comparison
        bot_trades_df = journal.get_trades(since=today, limit=200)
        bot_trade_count = len(bot_trades_df) if not bot_trades_df.empty else 0
        bot_pnl = bot_trades_df["pnl_pips"].sum() if not bot_trades_df.empty and "pnl_pips" in bot_trades_df else 0

        # Save summary
        wins = sum(1 for t in all_shadow_trades if t.get("pnl_pips", 0) > 0)
        losses = sum(1 for t in all_shadow_trades if t.get("pnl_pips", 0) <= 0 and t.get("exit_price"))
        total_pnl = sum(t.get("pnl_pips", 0) for t in all_shadow_trades)
        instruments_traded = list(set(t["instrument"] for t in all_shadow_trades))

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO shadow_daily_summary
                (date, total_trades, wins, losses, total_pnl_pips, win_rate,
                 instruments_traded, bot_trades, bot_pnl_pips)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                today, len(all_shadow_trades), wins, losses, total_pnl,
                wins / len(all_shadow_trades) if all_shadow_trades else 0,
                json.dumps(instruments_traded),
                bot_trade_count, float(bot_pnl) if bot_pnl else 0,
            ))

        logger.info(
            f"Shadow retrospective complete: {len(all_shadow_trades)} trades, "
            f"{wins}W/{losses}L, {total_pnl:+.1f} pips | "
            f"Bot: {bot_trade_count} trades, {float(bot_pnl):+.1f} pips"
        )

        return {
            "date": today,
            "shadow_trades": all_shadow_trades,
            "shadow_total": len(all_shadow_trades),
            "shadow_wins": wins,
            "shadow_losses": losses,
            "shadow_pnl_pips": total_pnl,
            "bot_trades": bot_trade_count,
            "bot_pnl_pips": float(bot_pnl) if bot_pnl else 0,
        }

    def _analyze_instrument(self, instrument, collector, engine, mt5_client):
        """Ask Claude what trades it would have taken today for one instrument."""
        # Get today's M1 candles (up to ~780 for 13 hours)
        m1_df = collector.get_candles_df(instrument, "M1", count=800)
        if m1_df.empty or len(m1_df) < 60:
            # Try fetching from MT5 directly
            m1_df = mt5_client.get_candles(instrument, "M1", count=800)
            if m1_df is None or m1_df.empty:
                return []

        # Compute indicators on the full day
        m1_with_ind = engine.calculate_all(m1_df)

        # Compress into a token-efficient summary:
        # Sample every 15 minutes (every 15th M1 candle) for the overview,
        # plus hourly indicator snapshots
        summary = self._build_day_summary(instrument, m1_with_ind)

        # Get instrument config for context
        inst_config = self.config.get_instrument(instrument)
        pip_loc = inst_config.get("pip_location", -4) if inst_config else -4
        pip_size = 10 ** pip_loc
        strategy = inst_config.get("strategy", "unknown") if inst_config else "unknown"
        sl_mult = inst_config.get("sl_atr_multiplier", 1.5) if inst_config else 1.5
        tp_mult = inst_config.get("tp_atr_multiplier", 2.5) if inst_config else 2.5

        from src.ai.prompts import build_retrospective_prompt

        prompt = build_retrospective_prompt(
            instrument=instrument,
            strategy=strategy,
            day_summary=summary,
            pip_size=pip_size,
            sl_atr_mult=sl_mult,
            tp_atr_mult=tp_mult,
        )

        response = self.analyst._call_claude(prompt, max_tokens=1500)
        if response is None:
            return []

        parsed = self.analyst._parse_json_response(response)
        if parsed is None:
            logger.warning(f"Shadow: failed to parse retrospective for {instrument}")
            return []

        # Verify each proposed trade against actual candle data
        proposed_trades = parsed.get("trades", [])
        verified_trades = []

        for trade in proposed_trades:
            verified = self._verify_trade(trade, instrument, m1_df, pip_size)
            if verified:
                self._save_trade(verified)
                verified_trades.append(verified)

        logger.info(
            f"Shadow {instrument}: Claude proposed {len(proposed_trades)} trades, "
            f"{len(verified_trades)} verified"
        )

        return verified_trades

    def _build_day_summary(self, instrument: str, m1_with_ind: pd.DataFrame) -> str:
        """
        Compress a day's M1 candles into a token-efficient summary for Claude.

        Strategy: Send M15 resampled OHLC + key indicators at each bar.
        ~50 bars for a full day = ~2000 tokens (vs ~15000 for raw M1).
        """
        lines = []

        # Resample to M15 for overview
        df = m1_with_ind.copy()
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            df = df.set_index("time")

        # Manual M15 resampling: take every 15th row as a snapshot
        step = 15
        for i in range(0, len(df), step):
            chunk = df.iloc[i:i + step]
            if chunk.empty:
                continue

            bar_time = chunk.index[0] if isinstance(chunk.index[0], (datetime, pd.Timestamp)) else i
            o = chunk.iloc[0]["open"]
            h = chunk["high"].max()
            l = chunk["low"].min()
            c = chunk.iloc[-1]["close"]

            # Key indicators from the last candle of the chunk
            last = chunk.iloc[-1]
            rsi = last.get("rsi_value", "")
            ema_d = last.get("ema_distance", "")
            bb_pos = last.get("bb_position", "")
            atr = last.get("atr_value", "")
            macd_h = last.get("macd_histogram", "")

            rsi_str = f"{rsi:.1f}" if isinstance(rsi, float) else ""
            bb_str = f"{bb_pos:.2f}" if isinstance(bb_pos, float) else ""
            atr_str = f"{atr:.5f}" if isinstance(atr, float) else ""

            lines.append(
                f"{bar_time} | O:{o:.5f} H:{h:.5f} L:{l:.5f} C:{c:.5f} | "
                f"RSI:{rsi_str} BB:{bb_str} ATR:{atr_str}"
            )

        return "\n".join(lines)

    def _verify_trade(
        self, trade: dict, instrument: str, m1_df: pd.DataFrame, pip_size: float
    ) -> dict | None:
        """
        Verify a Claude-proposed trade against actual candle data.

        Checks if the entry price was actually reachable, and simulates
        whether SL or TP would have been hit first.
        """
        try:
            direction = trade.get("direction", "").lower()
            entry_price = float(trade.get("entry_price", 0))
            sl = float(trade.get("stop_loss", 0))
            tp = float(trade.get("take_profit", 0))
            entry_time_str = trade.get("entry_time", "")
            confidence = float(trade.get("confidence", 0.5))
            reasoning = trade.get("reasoning", "")
        except (ValueError, TypeError):
            return None

        if direction not in ("buy", "sell") or entry_price <= 0:
            return None
        if not sl or not tp:
            return None

        # Find the candle at or after the proposed entry time
        entry_idx = None
        if entry_time_str and "time" in m1_df.columns:
            try:
                for i, row in m1_df.iterrows():
                    row_time = str(row.get("time", ""))
                    if row_time >= entry_time_str:
                        entry_idx = i
                        break
            except Exception:
                pass

        if entry_idx is None:
            # Can't verify without time, accept on faith but mark unverified
            return {
                "instrument": instrument,
                "direction": direction,
                "entry_price": entry_price,
                "entry_time": entry_time_str,
                "stop_loss": sl,
                "take_profit": tp,
                "exit_price": None,
                "exit_time": None,
                "pnl_pips": 0,
                "exit_reason": "unverified",
                "confidence": confidence,
                "reasoning": reasoning,
            }

        # Simulate forward from entry candle
        subsequent = m1_df.loc[entry_idx:]
        exit_price = None
        exit_time = None
        exit_reason = None

        for _, candle in subsequent.iterrows():
            if direction == "buy":
                if candle["low"] <= sl:
                    exit_price = sl
                    exit_reason = "sl_hit"
                elif candle["high"] >= tp:
                    exit_price = tp
                    exit_reason = "tp_hit"
            else:  # sell
                if candle["high"] >= sl:
                    exit_price = sl
                    exit_reason = "sl_hit"
                elif candle["low"] <= tp:
                    exit_price = tp
                    exit_reason = "tp_hit"

            if exit_price:
                exit_time = str(candle.get("time", ""))
                break

        # If neither hit by end of day, mark as open (use last close)
        if exit_price is None:
            last_candle = m1_df.iloc[-1]
            exit_price = float(last_candle["close"])
            exit_time = str(last_candle.get("time", ""))
            exit_reason = "session_end"

        # Calculate pips
        if direction == "buy":
            pnl_pips = (exit_price - entry_price) / pip_size
        else:
            pnl_pips = (entry_price - exit_price) / pip_size

        return {
            "instrument": instrument,
            "direction": direction,
            "entry_price": entry_price,
            "entry_time": entry_time_str,
            "stop_loss": sl,
            "take_profit": tp,
            "exit_price": exit_price,
            "exit_time": exit_time,
            "pnl_pips": round(pnl_pips, 1),
            "exit_reason": exit_reason,
            "confidence": confidence,
            "reasoning": reasoning,
        }

    def _save_trade(self, trade: dict):
        """Persist a verified shadow trade."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO shadow_trades
                (date, instrument, direction, entry_price, entry_time,
                 stop_loss, take_profit, exit_price, exit_time,
                 pnl_pips, exit_reason, confidence, reasoning, verified)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                today, trade["instrument"], trade["direction"],
                trade["entry_price"], trade["entry_time"],
                trade["stop_loss"], trade["take_profit"],
                trade["exit_price"], trade["exit_time"],
                trade["pnl_pips"], trade["exit_reason"],
                trade.get("confidence", 0), trade.get("reasoning", ""),
                1 if trade["exit_reason"] != "unverified" else 0,
            ))

    def get_performance(self, days: int = 30) -> dict:
        """Get shadow trader performance over recent days."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

        with sqlite3.connect(self.db_path) as conn:
            trades = conn.execute(
                "SELECT * FROM shadow_trades WHERE date >= ? AND verified = 1",
                (cutoff,),
            ).fetchall()
            cols = [d[0] for d in conn.execute(
                "SELECT * FROM shadow_trades LIMIT 0"
            ).description]

            summaries = conn.execute(
                "SELECT * FROM shadow_daily_summary WHERE date >= ? ORDER BY date DESC",
                (cutoff,),
            ).fetchall()
            sum_cols = [d[0] for d in conn.execute(
                "SELECT * FROM shadow_daily_summary LIMIT 0"
            ).description]

        if not trades:
            return {
                "total_trades": 0, "wins": 0, "losses": 0,
                "win_rate": 0, "total_pnl_pips": 0,
                "avg_pnl_pips": 0, "by_instrument": {},
                "daily_summaries": [], "vs_bot": {},
            }

        trade_dicts = [dict(zip(cols, t)) for t in trades]
        pnls = [t["pnl_pips"] or 0 for t in trade_dicts]
        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p <= 0)

        # By instrument
        by_inst = {}
        for t in trade_dicts:
            inst = t["instrument"]
            if inst not in by_inst:
                by_inst[inst] = {"trades": 0, "wins": 0, "pnl_pips": 0}
            by_inst[inst]["trades"] += 1
            pnl = t["pnl_pips"] or 0
            by_inst[inst]["pnl_pips"] += pnl
            if pnl > 0:
                by_inst[inst]["wins"] += 1
        for inst in by_inst:
            n = by_inst[inst]["trades"]
            by_inst[inst]["win_rate"] = by_inst[inst]["wins"] / n if n else 0

        # Vs bot comparison
        sum_dicts = [dict(zip(sum_cols, s)) for s in summaries]
        shadow_total_pnl = sum(s.get("total_pnl_pips", 0) or 0 for s in sum_dicts)
        bot_total_pnl = sum(s.get("bot_pnl_pips", 0) or 0 for s in sum_dicts)

        return {
            "total_trades": len(trade_dicts),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / len(trade_dicts) if trade_dicts else 0,
            "total_pnl_pips": sum(pnls),
            "avg_pnl_pips": sum(pnls) / len(pnls) if pnls else 0,
            "best_trade_pips": max(pnls) if pnls else 0,
            "worst_trade_pips": min(pnls) if pnls else 0,
            "by_instrument": by_inst,
            "daily_summaries": sum_dicts[-7:],  # Last 7 days
            "vs_bot": {
                "shadow_pnl_pips": shadow_total_pnl,
                "bot_pnl_pips": bot_total_pnl,
                "shadow_ahead_by": shadow_total_pnl - bot_total_pnl,
            },
        }
