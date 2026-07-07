"""
Manager briefing — compact JSON snapshot of account/trading state fed to
the Claude-manager each cycle. Task 13 owns the actual model call; this
module only assembles the input.

Pure function reading from TradeJournal (sqlite), EffectiveConfig, and
RatchetFloor. Every individual data source degrades to null/defaults on
error — `build()` itself must never raise just because one source
(evaluator table, model_store meta, control-log history, ...) is missing
or stale.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional

import src.control.effective_config as ec_module
from src.config import PROJECT_ROOT
from src.control.effective_config import EffectiveConfig
from src.manager import policy

logger = logging.getLogger("traderbot.manager.briefing")

MODEL_STORE_PATH = PROJECT_ROOT / "src" / "ml" / "model_store"

# Serialized briefing must fit comfortably inside a manager prompt
# (~4k tokens, rough 4-chars-per-token heuristic).
MAX_BRIEFING_CHARS = 16000

DEFAULT_MILESTONES = [1500, 2000, 3000, 4500, 6000]
DEFAULT_DAILY_DRAWDOWN_LIMIT_PCT = 4.0
DEFAULT_INSTRUMENT_WEIGHT = 1.0
RECENT_ACCURACY_LOOKBACK = 20
LAST_MANAGER_ACTIONS_LIMIT = 5
LOOKBACK_DAYS = 7

# Fields that hold newest-first lists and may be truncated (dropping the
# oldest entries) to keep the briefing under MAX_BRIEFING_CHARS.
TRUNCATABLE_LIST_FIELDS = ("open_positions", "last_manager_actions")


def _safe(fn, default=None):
    """Run fn(), degrading to `default` (with a warning) on any exception."""
    try:
        return fn()
    except Exception as e:
        logger.warning(f"Briefing: data source degraded to default ({e})")
        return default


def _model_version() -> str:
    def _read():
        latest_path = MODEL_STORE_PATH / "latest_version.txt"
        if latest_path.exists():
            version = latest_path.read_text(encoding="utf-8").strip()
            if version:
                return version
        return "unknown"

    return _safe(_read, "unknown")


def _instrument_names() -> list:
    return _safe(lambda: sorted(policy._valid_instruments()), [])


def _instrument_stats(journal, instrument: str, since_iso: str) -> dict:
    empty = {"trades": 0, "win_rate": None, "profit_factor": None, "net_pnl_zar": 0.0}

    def _compute():
        df = journal.get_trades(instrument=instrument, since=since_iso, limit=100000)
        if "exit_price" in df.columns:
            df = df[df["exit_price"].notna()]
        trades = len(df)
        if trades == 0:
            return dict(empty)

        pnl_col = "net_pnl_zar" if "net_pnl_zar" in df.columns else "pnl_zar"
        pnls = df[pnl_col].astype(float).fillna(0.0)

        wins = int((pnls > 0).sum())
        win_rate = wins / trades if trades else None
        gross_profit = float(pnls[pnls > 0].sum())
        gross_loss = float(-pnls[pnls < 0].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None

        return {
            "trades": int(trades),
            "win_rate": round(float(win_rate), 4) if win_rate is not None else None,
            "profit_factor": round(float(profit_factor), 4) if profit_factor is not None else None,
            "net_pnl_zar": round(float(pnls.sum()), 2),
        }

    return _safe(_compute, dict(empty))


def _open_positions(journal) -> list:
    def _compute():
        with sqlite3.connect(journal.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                """
                SELECT trade_id, instrument, direction, units, entry_price,
                       entry_time, stop_loss, take_profit
                FROM trades
                WHERE exit_price IS NULL
                ORDER BY entry_time DESC
                """
            )
            return [dict(row) for row in cur.fetchall()]

    return _safe(_compute, [])


def _today_pnl(journal) -> Optional[float]:
    def _compute():
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        ).isoformat()
        df = journal.get_trades(since=today_start, limit=100000)
        if df.empty:
            return 0.0
        pnl_col = "net_pnl_zar" if "net_pnl_zar" in df.columns else "pnl_zar"
        return round(float(df[pnl_col].astype(float).fillna(0.0).sum()), 2)

    return _safe(_compute, None)


def _drawdown_vs_cap(today_pnl: Optional[float], balance: Optional[float], cap_pct: float) -> dict:
    if today_pnl is None or not balance:
        return {"today_drawdown_pct": None, "cap_pct": cap_pct}
    dd_pct = max(0.0, -today_pnl / balance * 100.0)
    return {"today_drawdown_pct": round(dd_pct, 3), "cap_pct": cap_pct}


def _config_delta() -> dict:
    def _compute():
        path = ec_module.TUNES_PATH
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {}

    return _safe(_compute, {})


def _recent_accuracy(journal, last_n: int = RECENT_ACCURACY_LOOKBACK) -> Optional[float]:
    def _compute():
        with sqlite3.connect(journal.db_path) as conn:
            cur = conn.execute(
                "SELECT prediction, actual_outcome FROM evaluator_trades ORDER BY id DESC LIMIT ?",
                (last_n,),
            )
            rows = cur.fetchall()
        if not rows:
            return None
        correct = sum(
            1 for prediction, outcome in rows
            if (prediction >= 0.5 and outcome == 1) or (prediction < 0.5 and outcome == 0)
        )
        return round(correct / len(rows), 4)

    return _safe(_compute, None)


def _milestones_state(balance: float, milestones: list) -> list:
    return [{"milestone": m, "reached": balance >= m} for m in milestones]


def _last_manager_actions(journal, limit: int = LAST_MANAGER_ACTIONS_LIMIT) -> list:
    def _compute():
        df = journal.get_manager_log(limit=limit)
        if df.empty:
            return []
        cols = [c for c in ("ts_utc", "trigger", "outcome", "rationale") if c in df.columns]
        return df[cols].to_dict(orient="records")

    return _safe(_compute, [])


def _serialized_len(briefing: dict) -> int:
    return len(json.dumps(briefing, default=str))


def _enforce_size_cap(briefing: dict) -> None:
    """
    Truncate newest-first list fields (dropping the oldest entries) until
    the serialized briefing fits MAX_BRIEFING_CHARS, then hard-assert.
    """
    while _serialized_len(briefing) > MAX_BRIEFING_CHARS:
        shrank = False
        for key in TRUNCATABLE_LIST_FIELDS:
            lst = briefing.get(key)
            if isinstance(lst, list) and lst:
                lst.pop()  # newest-first list -> drop the oldest (tail)
                shrank = True
                if _serialized_len(briefing) <= MAX_BRIEFING_CHARS:
                    return
        if not shrank:
            break

    assert _serialized_len(briefing) <= MAX_BRIEFING_CHARS, (
        f"briefing exceeds {MAX_BRIEFING_CHARS} chars even after truncation "
        f"({_serialized_len(briefing)} chars)"
    )


def build(
    journal,
    effective_config: EffectiveConfig,
    ratchet_floor,
    balance: float,
    equity: float,
    extra: Optional[dict] = None,
) -> dict:
    """
    Assemble the compact JSON briefing handed to the Claude-manager.

    Every field degrades independently (never raises for a missing/stale
    source); the only hard failure mode is the final size-cap assertion,
    which truncates trade-list fields (keeping the newest entries) before
    asserting.
    """
    milestones = _safe(
        lambda: effective_config.get("growth.milestones", DEFAULT_MILESTONES) or DEFAULT_MILESTONES,
        DEFAULT_MILESTONES,
    )
    daily_dd_cap = _safe(
        lambda: effective_config.get(
            "risk.daily_drawdown_limit_pct", DEFAULT_DAILY_DRAWDOWN_LIMIT_PCT
        ) or DEFAULT_DAILY_DRAWDOWN_LIMIT_PCT,
        DEFAULT_DAILY_DRAWDOWN_LIMIT_PCT,
    )

    floor = _safe(lambda: ratchet_floor.current_floor, None)
    headroom_to_floor = (equity - floor) if (floor is not None and equity is not None) else None

    today_pnl = _today_pnl(journal)
    drawdown_vs_cap = _drawdown_vs_cap(today_pnl, balance, daily_dd_cap)

    since_7d = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).isoformat()
    per_instrument = {}
    for instrument in _instrument_names():
        stats = _instrument_stats(journal, instrument, since_7d)
        stats["current_weight"] = _safe(
            lambda i=instrument: effective_config.get(f"weight.{i}", DEFAULT_INSTRUMENT_WEIGHT),
            DEFAULT_INSTRUMENT_WEIGHT,
        )
        per_instrument[instrument] = stats

    briefing = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "balance": balance,
        "equity": equity,
        "floor": floor,
        "headroom_to_floor": headroom_to_floor,
        "today_pnl_zar": today_pnl,
        "drawdown_vs_cap": drawdown_vs_cap,
        "instruments": per_instrument,
        "open_positions": _open_positions(journal),
        "config_delta": _config_delta(),
        "model_version": _model_version(),
        "recent_accuracy": _recent_accuracy(journal),
        "growth_stage": policy.growth_stage(balance, milestones),
        "risk_ceiling_now": policy.risk_ceiling_now(balance, milestones),
        "milestones": _milestones_state(balance, milestones),
        "last_manager_actions": _last_manager_actions(journal),
    }
    if extra:
        briefing["extra"] = extra

    _enforce_size_cap(briefing)
    return briefing
