"""
Manager-in-the-loop backtesting (Task 15).

Adds a Claude-portfolio-manager simulation layer on top of
`backtest.simulator.BacktestSimulator`:

- `PortfolioSimulator` runs ALL instruments over a shared account on one
  merged timeline (shared balance/equity, shared open-position and daily
  limits), with:
    * per-instrument weight multipliers affecting position size
      (mirrors src/risk/position_sizer.py Step 3.5 semantics);
    * a ratcheting floor kill-switch using src.risk.ratchet_floor.RatchetFloor
      (floor = max(min_floor_zar, HWM * (1 - max_total_drawdown_pct)),
      monotonic; the sim STOPS when equity <= floor);
    * 21:00 UTC daily resets (drawdown + consecutive-loss counters);
    * starting balance from config `account.starting_balance_zar` (R1000);
    * milestone/risk-ceiling ladder via src.manager.policy.risk_ceiling_now.

- Every `manager.cycle_minutes` (default 60) simulated minutes within
  session hours, a briefing is built from SIMULATOR state via
  `src.manager.briefing.build` (reused through `SimJournalAdapter`, see
  below), handed to the manager backend, and the returned proposals are
  passed through THE SAME `src.manager.policy.validate_and_clamp` gate
  used by the live manager before mutating the sim's effective params.
  No proposal ever bypasses validate_and_clamp.

Manager backends
----------------
`HeuristicManager` — deterministic, NO API. Exact rules, evaluated per
cycle from the briefing (per-instrument stats are carried in
briefing["extra"]["sim"]["per_instrument"]):

  R1 MUTE:        if an instrument has >= 5 consecutive losses (counted
                  per instrument, reset on any win on that instrument and
                  at the 21:00 UTC daily reset) and its current weight
                  > 0.0 -> propose `weight.<INSTRUMENT> = 0.0`.
  R2 RISK NUDGE:  if `growth_stage` increased since the previous cycle
                  (a milestone was crossed) -> propose
                  `risk.risk_per_trade_pct = risk_ceiling_now` (the
                  growth-stage ceiling from policy.risk_ceiling_now).
                  No nudge on the very first cycle (no previous stage).
  R3 WEIGHT DOWN: if profit factor over the trailing 20 closed trades on
                  an instrument is < 0.8 (only evaluated once >= 20
                  closed trades exist) and weight > 0.0 -> propose
                  `weight.<INSTRUMENT> = current_weight - 0.25`
                  (not below 0.0). Skipped if R1 fired for it.
  R4 WEIGHT UP:   if trailing-20 profit factor is > 1.5 (>= 20 closed
                  trades) and current weight < 1.5 -> propose
                  `weight.<INSTRUMENT> = current_weight + 0.25`. The raw
                  proposal may exceed the 1.5 weight bound —
                  validate_and_clamp clamps it (deliberate: exercises the
                  shared clamp path).

  Priority order when assembling the cycle's proposals:
  R1 mutes first, then R2 risk nudge, then R3 weight downs, then R4
  weight ups; truncated to policy.MAX_PROPOSALS_PER_CYCLE (3).

`ClaudeManager` — real Anthropic API via src.manager.client.ManagerClient
(requires ANTHROPIC_API_KEY; ~10-13 calls per simulated day at the
default 60-minute cadence). Logs and accumulates cost per cycle. NEVER
used in unit tests — tests use HeuristicManager or a stub.

Documented deviations from live semantics
-----------------------------------------
1. Ratio-scaled levers. Live `instruments.yaml` gives EVERY enabled
   instrument its own `risk_per_trade_pct` (2.5/2.0/2.5/1.5) and its own
   ML thresholds on a different calibration scale (~0.10/0.18) than the
   global levers (0.50-0.75). Applying a tuned global value verbatim
   would either be a no-op (risk: per-instrument overrides win in
   src/risk/position_sizer.py) or mute all trading (ML: 0.50+ vs ~0.15
   confidence scale). The sim therefore applies tuned levers as RATIO
   SCALERS on the per-instrument values:
       effective = per_instrument_value * (tuned_value / settings_default)
   with the effective per-instrument risk additionally clamped to the
   lever's own [0.5, 2.5]% bounds. Direction and relative magnitude of
   every manager decision are preserved; absolute values are re-anchored
   to each instrument's calibration.
2. Briefing time re-anchoring. `briefing.build` computes lookback windows
   from wall-clock `datetime.now(timezone.utc)`; a backtest lives in the
   past. `SimJournalAdapter.get_trades` converts each `since` bound into
   a duration (now - since) and re-anchors it at the current SIM time, so
   "last 7 days" means the sim's last 7 days. Briefing fields backed by
   direct sqlite access (open_positions, recent_accuracy) degrade to
   their documented defaults via briefing's `_safe`; the sim supplies its
   own open-position count and per-instrument trailing stats in
   briefing["extra"]["sim"] instead.
3. RatchetFloor state is persisted to a throwaway temp file (fresh per
   run, pre-seeded with the sim's starting balance as HWM) — never to
   the live `data/account_state.json` — and updated on balance change
   rather than every candle to avoid per-candle disk writes.
"""

from __future__ import annotations

import json
import logging
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

import src.control.effective_config as ec_module
from backtest.simulator import BacktestSimulator, SimState
from src.control.effective_config import (
    EffectiveConfig,
    _deep_merge,
    _expand_dotted,
    _load_yaml,
)
from src.manager import briefing as briefing_mod
from src.manager import policy
from src.risk.ratchet_floor import RatchetFloor

logger = logging.getLogger("traderbot.backtest.manager_sim")

# 21:00 UTC session/day boundary (matches live drawdown/consecutive-loss
# counter resets — see src/risk and tests/test_session_boundary.py).
DAY_BOUNDARY_HOUR_UTC = 21

# Heuristic rule constants (documented in the module docstring).
TRAILING_WINDOW = 20
PF_REDUCE_BELOW = 0.8
PF_RAISE_ABOVE = 1.5
WEIGHT_STEP = 0.25
MUTE_AFTER_CONSECUTIVE_LOSSES = 5

# Weight bounds from the shared control-queue whitelist (via policy).
_WEIGHT_LO, _WEIGHT_HI = policy.WEIGHT_BOUNDS


def trading_day(ts) -> "datetime.date":
    """
    Trading-day label for a timestamp, with the day boundary at 21:00 UTC.
    21:00 belongs to the NEXT trading day (matches the live 21:00 UTC
    session-reset convention). Naive timestamps are treated as UTC.
    """
    return (ts + timedelta(hours=24 - DAY_BOUNDARY_HOUR_UTC)).date()


class SimEffectiveConfig(EffectiveConfig):
    """
    In-memory EffectiveConfig for backtests: settings.yaml + safety_floor
    (floor wins), NO tunes-overlay file read and NO disk writes ever.
    validate_and_clamp consults .get()/.is_safety_locked() on this.
    """

    @classmethod
    def from_files(cls) -> "SimEffectiveConfig":
        # Resolved via the module so tests can monkeypatch the paths the
        # same way they do for EffectiveConfig.load().
        settings = _load_yaml(ec_module.SETTINGS_PATH)
        floor = _load_yaml(ec_module.SAFETY_FLOOR_PATH)
        return cls(_deep_merge(settings, floor), cls._flat_keys(floor))

    def apply_in_memory(self, dotted_key: str, value) -> None:
        """Apply a validated tune to the in-memory view only (no disk)."""
        self._data = _deep_merge(self._data, _expand_dotted(dotted_key, value))


class SimParamOverlay:
    """
    The sim's mutable effective params — mutated ONLY with values that
    already passed policy.validate_and_clamp (see PortfolioSimulator).
    Plugged into BacktestSimulator.param_overlay for sizing.
    """

    def __init__(self, base_risk_pct: float, base_threshold_high: float, base_threshold_low: float):
        self.base_risk_pct = float(base_risk_pct)
        self.base_threshold_high = float(base_threshold_high)
        self.base_threshold_low = float(base_threshold_low)
        self.risk_pct: Optional[float] = None
        self.threshold_high: Optional[float] = None
        self.threshold_low: Optional[float] = None
        self.weights: dict = {}

    def weight(self, instrument: str) -> float:
        return self.weights.get(instrument, 1.0)

    def effective_risk_pct(self, inst_risk_frac: float) -> float:
        """
        Ratio-scale a per-instrument risk fraction by tuned/base global
        risk %, clamped to the lever's own [0.5, 2.5]% bounds (as
        fractions). Identity when no risk tune has been applied.
        """
        if self.risk_pct is None:
            return inst_risk_frac
        lo, hi = policy.LEVERS[policy.RISK_KEY]
        scaled = inst_risk_frac * (self.risk_pct / self.base_risk_pct)
        return max(lo / 100.0, min(scaled, hi / 100.0))

    def scaled_ml_thresholds(self, inst_low: float, inst_high: float) -> tuple:
        """Ratio-scale per-instrument ML thresholds by tuned/base globals."""
        low, high = inst_low, inst_high
        if self.threshold_low is not None and self.base_threshold_low > 0:
            low = inst_low * (self.threshold_low / self.base_threshold_low)
        if self.threshold_high is not None and self.base_threshold_high > 0:
            high = inst_high * (self.threshold_high / self.base_threshold_high)
        return low, high


class SimJournalAdapter:
    """
    Duck-types just enough of monitoring.trade_journal.TradeJournal for
    src.manager.briefing.build to run off SIMULATOR state (see module
    docstring, deviation 2, for the `since` re-anchoring).
    """

    # briefing._open_positions/_recent_accuracy sqlite-connect to this and
    # find no tables -> they _safe-degrade to their documented defaults.
    db_path = ":memory:"

    def __init__(self):
        self.closed_trades: list = []   # SimTrade objects, append-only
        self.decisions: list = []       # manager decision-log dicts
        self.sim_now = None             # current sim timestamp (set per cycle)

    @staticmethod
    def _naive_utc(ts) -> datetime:
        """Coerce a datetime/Timestamp to naive UTC for arithmetic."""
        ts = pd.Timestamp(ts)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return ts.to_pydatetime()

    def get_trades(self, instrument: Optional[str] = None, since: Optional[str] = None,
                   limit: int = 100000) -> pd.DataFrame:
        rows = [
            {
                "instrument": t.instrument,
                "entry_time": t.entry_time,
                "exit_price": t.exit_price,
                "net_pnl_zar": t.pnl,
            }
            for t in self.closed_trades
        ]
        df = pd.DataFrame(rows, columns=["instrument", "entry_time", "exit_price", "net_pnl_zar"])
        if instrument is not None:
            df = df[df["instrument"] == instrument]
        if since is not None and self.sim_now is not None and len(df):
            # Re-anchor the wall-clock window at the sim clock.
            lookback = datetime.now(timezone.utc) - datetime.fromisoformat(since)
            cutoff = self._naive_utc(self.sim_now) - lookback
            entry = df["entry_time"].map(self._naive_utc)
            df = df[entry >= cutoff]
        return df.tail(limit)

    def get_manager_log(self, limit: int = 5) -> pd.DataFrame:
        rows = [
            {
                "ts_utc": d.get("ts_utc"),
                "trigger": d.get("trigger"),
                "outcome": d.get("outcome"),
                "rationale": d.get("rationale"),
            }
            for d in reversed(self.decisions[-limit:])
        ]
        return pd.DataFrame(rows, columns=["ts_utc", "trigger", "outcome", "rationale"])


class HeuristicManager:
    """Deterministic manager backend — exact rules in the module docstring."""

    name = "heuristic"

    def __init__(self):
        self._prev_growth_stage: Optional[int] = None

    def propose(self, briefing: dict) -> tuple:
        per_inst = ((briefing.get("extra") or {}).get("sim") or {}).get("per_instrument", {})

        mutes, downs, ups = [], [], []
        for inst in sorted(per_inst):
            stats = per_inst[inst] or {}
            weight = float(stats.get("weight", 1.0))

            # R1 MUTE
            consec = int(stats.get("consecutive_losses", 0))
            if consec >= MUTE_AFTER_CONSECUTIVE_LOSSES and weight > 0.0:
                mutes.append({
                    "key": f"weight.{inst}",
                    "value": 0.0,
                    "reason": f"{consec} consecutive losses on {inst}; muting",
                })
                continue

            pf = stats.get("trailing_pf")
            n = int(stats.get("trailing_trades", 0))
            if pf is None or n < TRAILING_WINDOW:
                continue

            # R3 WEIGHT DOWN
            if pf < PF_REDUCE_BELOW and weight > 0.0:
                downs.append({
                    "key": f"weight.{inst}",
                    "value": max(0.0, round(weight - WEIGHT_STEP, 4)),
                    "reason": f"trailing-{TRAILING_WINDOW} PF {pf:.2f} < {PF_REDUCE_BELOW}; reducing weight",
                })
            # R4 WEIGHT UP (raw value may exceed the bound; clamp handles it)
            elif pf > PF_RAISE_ABOVE and weight < _WEIGHT_HI:
                ups.append({
                    "key": f"weight.{inst}",
                    "value": round(weight + WEIGHT_STEP, 4),
                    "reason": f"trailing-{TRAILING_WINDOW} PF {pf:.2f} > {PF_RAISE_ABOVE}; raising weight",
                })

        # R2 RISK NUDGE
        risk_props = []
        stage = int(briefing.get("growth_stage", 0) or 0)
        if self._prev_growth_stage is not None and stage > self._prev_growth_stage:
            ceiling = briefing.get("risk_ceiling_now")
            if ceiling is not None:
                risk_props.append({
                    "key": policy.RISK_KEY,
                    "value": float(ceiling),
                    "reason": f"milestone crossed (growth stage {self._prev_growth_stage} -> {stage}); "
                              f"nudging risk toward ceiling {ceiling}",
                })
        self._prev_growth_stage = stage

        proposals = (mutes + risk_props + downs + ups)[:policy.MAX_PROPOSALS_PER_CYCLE]
        rationale = (
            "; ".join(p["reason"] for p in proposals)
            if proposals else "no rule fired; holding current parameters"
        )
        usage = {"input_tokens": 0, "output_tokens": 0, "cost_zar": 0.0}
        return proposals, rationale, usage


class ClaudeManager:
    """
    Real-API manager backend via src.manager.client.ManagerClient.
    Requires ANTHROPIC_API_KEY (unless an injected fake `client` is
    passed). NEVER instantiated without a fake client in unit tests.
    """

    name = "claude"

    def __init__(self, config, client=None):
        # Local import: keeps this module importable without the anthropic
        # package for heuristic-only use.
        from src.manager.client import ManagerClient
        self._client = ManagerClient(config, client=client)
        self.total_cost_zar = 0.0
        self.cycles = 0

    def propose(self, briefing: dict) -> tuple:
        from src.manager.client import ManagerAPIUnavailable
        self.cycles += 1
        try:
            proposals, rationale, usage = self._client.call(briefing)
        except ManagerAPIUnavailable as e:
            usage = getattr(e, "usage", None) or {"input_tokens": 0, "output_tokens": 0, "cost_zar": 0.0}
            self.total_cost_zar += usage.get("cost_zar", 0.0)
            logger.warning("Claude manager cycle %d: API unavailable (%s)", self.cycles, e)
            return [], f"api_unavailable: {e}", usage

        self.total_cost_zar += usage.get("cost_zar", 0.0)
        logger.info(
            "Claude manager cycle %d: cost R%.3f (cumulative R%.2f), %d proposal(s)",
            self.cycles, usage.get("cost_zar", 0.0), self.total_cost_zar, len(proposals),
        )
        return proposals, rationale, usage


class PortfolioSimulator:
    """
    Multi-instrument, shared-account backtest with an optional manager in
    the loop. `manager=None` is the baseline (identical engine, no
    manager cycles) for the baseline-vs-managed comparison.
    """

    def __init__(self, config, indicator_engine, predictor, manager=None,
                 floor_state_path=None, cycle_minutes: Optional[int] = None):
        self.config = config
        self.engine = indicator_engine
        self.predictor = predictor
        self.manager = manager

        self.sim = BacktestSimulator(config, indicator_engine, predictor)
        self.starting_balance = config.get("account.starting_balance_zar", 1000)
        self.milestones = config.get("growth.milestones", briefing_mod.DEFAULT_MILESTONES)
        self.cycle_minutes = int(cycle_minutes or config.get("manager.cycle_minutes", 60))

        self.sim_config = SimEffectiveConfig.from_files()
        self.overlay = SimParamOverlay(
            base_risk_pct=config.get("risk.risk_per_trade_pct", 1.5),
            base_threshold_high=config.get("ml.confidence_threshold_high", 0.55),
            base_threshold_low=config.get("ml.confidence_threshold_low", 0.50),
        )
        self.sim.param_overlay = self.overlay

        # Ratchet floor: throwaway state file, HWM pre-seeded to the sim's
        # starting balance (deviation 3 in the module docstring).
        min_floor = self.sim_config.get("risk.min_floor_zar", 600)
        max_total_dd = self.sim_config.get("risk.max_total_drawdown_pct", 0.35)
        if floor_state_path is None:
            floor_state_path = Path(tempfile.mkdtemp(prefix="tb_sim_floor_")) / "account_state.json"
        floor_state_path = Path(floor_state_path)
        floor_state_path.parent.mkdir(parents=True, exist_ok=True)
        floor_state_path.write_text(
            json.dumps({"high_water_mark": float(self.starting_balance)}), encoding="utf-8"
        )
        self.floor = RatchetFloor(min_floor, max_total_dd, state_path=floor_state_path)

        # Daily drawdown stop comes from the (untunable) safety floor.
        self.daily_dd_limit = self.sim_config.get("risk.daily_drawdown_stop_pct", 4.0) / 100.0

        self.journal = SimJournalAdapter()
        self.decisions: list = []
        self.api_cost_total = 0.0
        self._last_cycle_ts = None
        self._mgr_session = (7, 20)  # recomputed in run() from instrument sessions

        self._instruments: list = []
        self._closed_by_inst: dict = {}
        self._consec_by_inst: dict = {}

    # ---------------- manager cycle plumbing ----------------

    @staticmethod
    def _trading_day(ts):
        return trading_day(ts)

    def _in_manager_session(self, ts) -> bool:
        start, end = self._mgr_session
        return start <= ts.hour < end

    def _manager_due(self, ts) -> bool:
        """Cycle cadence: every cycle_minutes, only within session hours."""
        if self.manager is None or not self._in_manager_session(ts):
            return False
        if self._last_cycle_ts is None:
            return True
        return (ts - self._last_cycle_ts) >= timedelta(minutes=self.cycle_minutes)

    def _per_instrument_briefing_stats(self) -> dict:
        out = {}
        for inst in self._instruments:
            closed = self._closed_by_inst.get(inst, [])
            tail = closed[-TRAILING_WINDOW:]
            pf = None
            if len(tail) >= TRAILING_WINDOW:
                gross_profit = sum(t.pnl for t in tail if t.pnl > 0)
                gross_loss = -sum(t.pnl for t in tail if t.pnl < 0)
                pf = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
            out[inst] = {
                "trailing_pf": pf,
                "trailing_trades": len(tail),
                "consecutive_losses": self._consec_by_inst.get(inst, 0),
                "weight": self.overlay.weight(inst),
            }
        return out

    def _apply_tune(self, key: str, value) -> None:
        """Apply ONE already-validated/clamped tune to the sim params."""
        value = float(value)
        if key.startswith("weight."):
            self.overlay.weights[key.split(".", 1)[1]] = value
        elif key == policy.RISK_KEY:
            self.overlay.risk_pct = value
        elif key == policy.THRESHOLD_HIGH_KEY:
            self.overlay.threshold_high = value
        elif key == policy.THRESHOLD_LOW_KEY:
            self.overlay.threshold_low = value
        else:  # pragma: no cover — validate_and_clamp only passes known keys
            logger.warning("Applied tune for unhandled key %s (no sim effect)", key)
        self.sim_config.apply_in_memory(key, value)

    def _run_manager_cycle(self, ts, state: SimState, equity: float) -> dict:
        """
        One manager cycle: briefing (from sim state, via briefing.build)
        -> manager.propose -> policy.validate_and_clamp -> apply survivors.
        """
        self.journal.sim_now = ts
        extra = {
            "sim": {
                "time_utc": str(ts),
                "per_instrument": self._per_instrument_briefing_stats(),
                "open_positions": len(state.open_trades),
            }
        }
        briefing = briefing_mod.build(
            self.journal, self.sim_config, self.floor,
            balance=state.balance, equity=equity, extra=extra,
        )
        proposals, rationale, usage = self.manager.propose(briefing)

        ceiling = policy.risk_ceiling_now(state.balance, self.milestones)
        applied, rejected = policy.validate_and_clamp(proposals, self.sim_config, ceiling)
        for entry in applied:
            self._apply_tune(entry["key"], entry["value"])

        cost = float(usage.get("cost_zar", 0.0))
        self.api_cost_total += cost
        decision = {
            "ts_utc": str(ts),
            "trigger": "timer",
            "proposals": proposals,
            "applied": applied,
            "rejected": rejected,
            "rationale": rationale,
            "cost_zar": cost,
            "risk_ceiling": ceiling,
            "outcome": "applied" if applied else "no_op",
        }
        self.decisions.append(decision)
        self.journal.decisions.append(decision)
        self._last_cycle_ts = ts
        return decision

    # ---------------- account/day plumbing ----------------

    def _daily_reset(self, state: SimState, ts) -> None:
        """21:00 UTC trading-day reset: drawdown + consecutive-loss counters."""
        state.current_date = ts
        state.daily_start_balance = state.balance
        state.daily_pnl = 0.0
        state.trade_count_today = 0
        state.consecutive_losses = 0
        for inst in self._consec_by_inst:
            self._consec_by_inst[inst] = 0
        if state.is_paused and (
            "consecutive" in state.pause_reason or "drawdown" in state.pause_reason.lower()
        ):
            state.is_paused = False
            state.pause_reason = ""

    def _record_closes(self, state: SimState, n_before: int) -> None:
        """Track per-instrument closes for trailing PF / consec-loss stats."""
        for t in state.trades[n_before:]:
            self._closed_by_inst.setdefault(t.instrument, []).append(t)
            self.journal.closed_trades.append(t)
            if t.pnl > 0:
                self._consec_by_inst[t.instrument] = 0
            else:
                self._consec_by_inst[t.instrument] = self._consec_by_inst.get(t.instrument, 0) + 1

    def _unrealized(self, state: SimState, contexts: dict) -> float:
        total = 0.0
        for t in state.open_trades:
            ctx = contexts[t.instrument]
            price = ctx["last_close"]
            if price is None:
                continue
            if t.direction == "buy":
                pips = (price - t.entry_price) / ctx["pip_size"]
            else:
                pips = (t.entry_price - price) / ctx["pip_size"]
            total += pips * ctx["pip_value"] * t.position_size
        return total

    def _force_close_all(self, state: SimState, contexts: dict, ts, reason: str) -> None:
        n_before = len(state.trades)
        for t in list(state.open_trades):
            ctx = contexts[t.instrument]
            price = ctx["last_close"] if ctx["last_close"] is not None else t.entry_price
            self.sim._close_trade(state, t, price, ts, reason, ctx["pip_size"], ctx["pip_value"])
        state.open_trades = []
        self._record_closes(state, n_before)

    # ---------------- main loop ----------------

    def run(self, data: dict) -> dict:
        """
        Run the portfolio backtest.

        Args:
            data: {instrument: (m1_df, m15_df)} — all instruments trade a
                  single shared account on one merged timeline.
        Returns:
            Report dict (see _compile) incl. manager decision log, API
            cost, and a net-after-cost equity curve.
        """
        if not data:
            return {"error": "No instrument data supplied"}

        self._instruments = sorted(data.keys())
        self._closed_by_inst = {i: [] for i in self._instruments}
        self._consec_by_inst = {i: 0 for i in self._instruments}

        contexts = {}
        sessions = self.config.get("trading.trading_sessions", {})
        for inst in self._instruments:
            m1_df, m15_df = data[inst]
            m1i = self.engine.calculate_all(m1_df)
            m1i["ema_55"] = m1_df["close"].ewm(span=55, adjust=False).mean()
            m15i = self.engine.calculate_all(m15_df)
            icfg = self.config.get_instrument(inst) or {}
            pip_size = 10 ** icfg.get("pip_location", -4)
            contexts[inst] = {
                "m1": m1_df,
                "m1i": m1i,
                "trend": self.sim._compute_m15_trend_series(m15i),
                "icfg": icfg,
                "pip_size": pip_size,
                "pip_value": self.sim._pip_value_usd(inst, pip_size, m1_df["close"].iloc[-1]),
                "session": sessions.get(icfg.get("trading_session", "forex"),
                                        {"start_hour": 7, "end_hour": 20}),
                "spread": icfg.get("typical_spread_pips", 1.2),
                "pos": {ts: i for i, ts in enumerate(m1_df.index)},
                "last_close": None,
                "last_trade_bar": -(10 ** 9),
            }

        self._mgr_session = (
            min(c["session"].get("start_hour", 7) for c in contexts.values()),
            max(c["session"].get("end_hour", 20) for c in contexts.values()),
        )

        min_bars = self.engine.get_required_candle_count()
        timeline = sorted(set().union(*[set(c["pos"]) for c in contexts.values()]))

        state = SimState(
            balance=self.starting_balance,
            starting_balance=self.starting_balance,
            daily_start_balance=self.starting_balance,
        )

        killed = False
        kill_info = None
        current_tday = None
        inst_map = {"EUR_USD": 0, "GBP_USD": 1, "USD_JPY": 2, "XAU_USD": 3}

        for ts in timeline:
            tday = self._trading_day(ts)
            if tday != current_tday:
                current_tday = tday
                self._daily_reset(state, ts)

            # 1) closes / trailing stops, per instrument with a candle at ts
            for inst in self._instruments:
                ctx = contexts[inst]
                idx = ctx["pos"].get(ts)
                if idx is None:
                    continue
                candle = ctx["m1"].iloc[idx]
                ctx["last_close"] = candle["close"]
                n_before = len(state.trades)
                mine = [t for t in state.open_trades if t.instrument == inst]
                others = [t for t in state.open_trades if t.instrument != inst]
                state.open_trades = mine
                self.sim._check_open_trades(state, candle, ts, ctx["pip_size"], ctx["pip_value"])
                state.open_trades.extend(others)
                self._record_closes(state, n_before)

            # 2) equity point (net-of-API-cost curve alongside)
            equity = state.balance + self._unrealized(state, contexts)
            state.equity_curve.append({
                "time": ts,
                "balance": state.balance,
                "equity": equity,
                "equity_net": equity - self.api_cost_total,
            })

            # 3) ratcheting floor kill-switch (floor only ever rises)
            if state.balance > self.floor.high_water_mark:
                self.floor.update(state.balance)
            if self.floor.is_breached(equity):
                self._force_close_all(state, contexts, ts, "floor_kill")
                killed = True
                kill_info = {
                    "time": str(ts),
                    "equity": equity,
                    "floor": self.floor.current_floor,
                    "high_water_mark": self.floor.high_water_mark,
                }
                logger.warning(
                    "RATCHET FLOOR KILL: equity R%.2f <= floor R%.2f at %s — sim stopped",
                    equity, self.floor.current_floor, ts,
                )
                break

            # 4) manager cycle (60 sim-minutes, session hours only)
            if self._manager_due(ts):
                self._run_manager_cycle(ts, state, equity)

            # 5) entries
            if state.is_paused:
                continue
            daily_loss = state.daily_start_balance - state.balance
            if daily_loss >= state.daily_start_balance * self.daily_dd_limit:
                state.is_paused = True
                state.pause_reason = "Daily drawdown limit"
                continue
            if state.trade_count_today >= self.sim.max_trades_day:
                continue

            for inst in self._instruments:
                if len(state.open_trades) >= self.sim.max_open:
                    break
                ctx = contexts[inst]
                idx = ctx["pos"].get(ts)
                if idx is None or idx < min_bars:
                    continue
                if self.overlay.weight(inst) <= 0.0:
                    continue  # muted instrument
                hour = ts.hour
                sess = ctx["session"]
                if hour < sess.get("start_hour", 7) or hour >= sess.get("end_hour", 20):
                    continue
                if idx - ctx["last_trade_bar"] < self.sim.min_bars_between_trades:
                    continue

                candle = ctx["m1"].iloc[idx]
                features = self.sim._build_features_at(
                    ctx["m1i"], ctx["trend"], idx, ts, ctx["spread"] * ctx["pip_size"]
                )
                if features is None:
                    continue

                row = ctx["m1i"].iloc[idx]
                atr_value = row.get("atr_value", 0)
                atr_ratio = row.get("atr_ratio", 1.0)
                if pd.isna(atr_value) or atr_value <= 0:
                    continue
                if atr_ratio > self.sim.high_vol_ratio or atr_ratio < self.sim.low_vol_ratio:
                    continue
                if ts.weekday() == 4 and hour >= 16:
                    continue

                strategy = ctx["icfg"].get("strategy", "pullback")
                if strategy == "pullback":
                    direction = self.sim._strategy_pullback(features, ctx["m1i"], idx, inst)
                elif strategy == "london_breakout":
                    direction = self.sim._strategy_london_breakout(
                        features, ctx["m1i"], ctx["m1"], idx, ts, inst, ctx["pip_size"])
                elif strategy == "tokyo_breakout":
                    direction = self.sim._strategy_tokyo_breakout(
                        features, ctx["m1i"], ctx["m1"], idx, ts, inst, ctx["pip_size"])
                elif strategy == "momentum_breakout":
                    direction = self.sim._strategy_momentum_breakout(features, ctx["m1i"], idx, inst)
                else:
                    direction = None
                if direction is None:
                    continue

                features["instrument_id"] = float(inst_map.get(inst, 4))
                if ctx["icfg"].get("ml_filter_enabled", False) and self.predictor.model is not None:
                    ml_confidence = self.predictor.predict(features)
                    thresh_low = ctx["icfg"].get("ml_threshold_low", 0.10)
                    thresh_high = ctx["icfg"].get("ml_threshold_high", 0.18)
                    thresh_low, thresh_high = self.overlay.scaled_ml_thresholds(thresh_low, thresh_high)
                    if ml_confidence < thresh_low:
                        state.ml_skips = getattr(state, "ml_skips", 0) + 1
                        continue
                else:
                    ml_confidence = 0.5

                trade = self.sim._create_trade(
                    state, inst, direction, candle, ts, atr_value, atr_ratio,
                    ctx["pip_size"], ctx["pip_value"], ctx["spread"], ml_confidence,
                )
                if trade is not None:
                    state.open_trades.append(trade)
                    state.trade_count_today += 1
                    ctx["last_trade_bar"] = idx

        if not killed and timeline:
            self._force_close_all(state, contexts, timeline[-1], "backtest_end")

        return self._compile(state, killed, kill_info)

    # ---------------- reporting ----------------

    def _compile(self, state: SimState, killed: bool, kill_info: Optional[dict]) -> dict:
        trades = state.trades
        pnls = [t.pnl for t in trades]
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        equity_df = pd.DataFrame(state.equity_curve)
        if not equity_df.empty:
            peak = equity_df["equity"].expanding().max()
            drawdown = (equity_df["equity"] - peak) / peak
            max_drawdown = abs(drawdown.min())
            max_drawdown_zar = float((peak - equity_df["equity"]).max())
        else:
            max_drawdown = 0.0
            max_drawdown_zar = 0.0

        per_instrument = {}
        for inst in self._instruments:
            inst_trades = self._closed_by_inst.get(inst, [])
            inst_wins = [t for t in inst_trades if t.pnl > 0]
            per_instrument[inst] = {
                "trades": len(inst_trades),
                "wins": len(inst_wins),
                "pnl_zar": round(sum(t.pnl for t in inst_trades), 2),
                "final_weight": self.overlay.weight(inst),
            }

        total_pnl = sum(pnls)
        return {
            "mode": "managed" if self.manager is not None else "baseline",
            "manager_backend": getattr(self.manager, "name", None),
            "instruments": list(self._instruments),
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": (len(wins) / len(trades)) if trades else 0.0,
            "profit_factor": profit_factor if trades else 0.0,
            "total_pnl_zar": total_pnl,
            "starting_balance": state.starting_balance,
            "final_balance": state.balance,
            "return_pct": (state.balance - state.starting_balance) / state.starting_balance * 100,
            "max_drawdown_pct": max_drawdown * 100,
            "max_drawdown_zar": round(max_drawdown_zar, 2),
            "killed_by_floor": killed,
            "kill_info": kill_info,
            "ml_skips": getattr(state, "ml_skips", 0),
            "per_instrument": per_instrument,
            "manager_decisions": list(self.decisions),
            "manager_cycles": len(self.decisions),
            "api_cost_zar": round(self.api_cost_total, 2),
            "net_pnl_after_cost_zar": round(total_pnl - self.api_cost_total, 2),
            "equity_curve": state.equity_curve,
            "final_params": {
                "risk_pct": self.overlay.risk_pct,
                "threshold_high": self.overlay.threshold_high,
                "threshold_low": self.overlay.threshold_low,
                "weights": dict(self.overlay.weights),
            },
            "trades": [
                {
                    "id": t.trade_id,
                    "instrument": t.instrument,
                    "direction": t.direction,
                    "entry_time": str(t.entry_time),
                    "exit_time": str(t.exit_time),
                    "pnl": t.pnl,
                    "exit_reason": t.exit_reason,
                }
                for t in trades
            ],
        }


def run_managed_backtest_for_prompt(
    backend: str,
    prompt_path,
    window_days: int,
    client=None,
    data: Optional[dict] = None,
) -> dict:
    """
    Adapter for backtest/prompt_lab.py (Task 18): run ONE managed backtest
    with the given system-prompt file over the last `window_days` days of
    the cached test window.

    Returns {trades, net_pnl_zar (NET of API cost), max_dd_zar,
    api_cost_zar, report} — the keys prompt_lab.score_result expects.

    Args:
        backend:     "claude" (prompt_path becomes the manager's system
                     prompt; needs ANTHROPIC_API_KEY unless `client` is
                     injected) or "heuristic" (prompt ignored — harness
                     smoke only).
        prompt_path: path to the prompt variant file.
        window_days: trailing window (days) of the test data to use;
                     0/None = the full test window.
        client:      optional injected Anthropic-compatible client (tests).
        data:        optional {instrument: (m1_df, m15_df)} override; when
                     None the standard cached test window is loaded.
    """
    from src.config import load_config
    from src.indicators.engine import IndicatorEngine
    from src.ml.predictor import Predictor

    config = load_config()
    engine = IndicatorEngine(config)
    predictor = Predictor(config)
    if not predictor.load_model():
        predictor.model = None
        logger.warning("prompt-lab backtest: no saved model — rules-only")

    if data is None:
        from backtest.runner import _managed_test_window  # lazy: avoid import cycle
        data = _managed_test_window(config, logger)
    if not data:
        raise RuntimeError("No cached backtest data under data/historical/")

    if window_days:
        trimmed = {}
        for inst, (m1_df, m15_df) in data.items():
            cutoff = m1_df.index[-1] - pd.Timedelta(days=window_days)
            m1_w = m1_df[m1_df.index > cutoff]
            m15_w = m15_df[m15_df.index > cutoff]
            if len(m1_w):
                trimmed[inst] = (m1_w, m15_w)
        data = trimmed

    if backend == "claude":
        manager = ClaudeManager(config, client=client)
        # Override the champion prompt with this specific variant.
        manager._client.system_prompt = Path(prompt_path).read_text(encoding="utf-8")
    else:
        manager = HeuristicManager()

    report = PortfolioSimulator(config, engine, predictor, manager=manager).run(data)
    return {
        "trades": report.get("total_trades", 0),
        "net_pnl_zar": report.get("net_pnl_after_cost_zar", 0.0),
        "max_dd_zar": report.get("max_drawdown_zar", 0.0),
        "api_cost_zar": report.get("api_cost_zar", 0.0),
        "report": report,
    }


def comparison_table(baseline: dict, managed: dict) -> str:
    """Baseline (no manager) vs managed comparison, as printable text."""
    rows = [
        ("Trades", "total_trades", "{:d}"),
        ("Win rate", "win_rate", "{:.1%}"),
        ("Profit factor", "profit_factor", "{:.2f}"),
        ("Total PnL (ZAR)", "total_pnl_zar", "{:.2f}"),
        ("Return %", "return_pct", "{:.1f}"),
        ("Max drawdown %", "max_drawdown_pct", "{:.1f}"),
        ("Final balance", "final_balance", "{:.2f}"),
        ("Killed by floor", "killed_by_floor", "{}"),
        ("Manager cycles", "manager_cycles", "{:d}"),
        ("API cost (ZAR)", "api_cost_zar", "{:.2f}"),
        ("Net PnL after cost", "net_pnl_after_cost_zar", "{:.2f}"),
    ]
    lines = [
        f"{'Metric':<22} {'Baseline':>14} {'Managed':>14}",
        "-" * 52,
    ]
    for label, key, fmt in rows:
        def _fmt(report):
            value = report.get(key)
            try:
                return fmt.format(value)
            except (TypeError, ValueError):
                return str(value)
        lines.append(f"{label:<22} {_fmt(baseline):>14} {_fmt(managed):>14}")
    return "\n".join(lines)


def format_decision_log(managed: dict) -> str:
    """Manager decision log (per cycle: proposals/applied/clamped) as text."""
    lines = ["MANAGER DECISION LOG"]
    for d in managed.get("manager_decisions", []):
        lines.append(
            f"[{d['ts_utc']}] {d['outcome']} (cost R{d['cost_zar']:.3f}, "
            f"risk ceiling {d['risk_ceiling']:.2f}) — {d['rationale']}"
        )
        for a in d.get("applied", []):
            clamp = " [CLAMPED]" if a.get("clamped") else ""
            lines.append(f"    applied  {a['key']} = {a['value']}{clamp} ({a['reason']})")
        for r in d.get("rejected", []):
            lines.append(f"    rejected {r['key']} = {r['value']} ({r['rejection_reason']})")
    if len(lines) == 1:
        lines.append("  (no manager cycles)")
    return "\n".join(lines)
