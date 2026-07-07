"""
Task 15 — backtest/manager_sim.py: manager-in-the-loop backtesting.

Covers (no network, no API key, no data files):
- HeuristicManager rules (mute, PF weight down/up, milestone risk nudge,
  <20-trade guard, 3-proposal cap, no repeat mute).
- Weight changes affect subsequent position sizing through the simulator
  overlay hook (and weight 0.0 rejects the trade).
- Tuned risk % ratio-scales sizing.
- Manager proposals flow through policy.validate_and_clamp (out-of-bounds
  values clamped, bad keys rejected, risk capped at risk_ceiling_now).
- Ratchet-floor kill-switch stops the sim.
- Manager cycle cadence respects session hours + 60-min spacing.
- 21:00 UTC trading-day boundary and daily resets.
- End-to-end smoke run on synthetic data (heuristic backend, no API).
"""

import numpy as np
import pandas as pd
import pytest

import backtest.manager_sim as ms
from backtest.manager_sim import (
    HeuristicManager,
    PortfolioSimulator,
    SimParamOverlay,
    comparison_table,
    trading_day,
)
from backtest.simulator import BacktestSimulator, SimState
from src.config import load_config
from src.indicators.engine import IndicatorEngine
from src.ml.predictor import Predictor
from src.risk.ratchet_floor import RatchetFloor


# ---------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def engine(config):
    return IndicatorEngine(config)


@pytest.fixture(scope="module")
def predictor(config):
    # No model on purpose: rules-only mode, no ML gate.
    return Predictor(config)


def _portfolio(config, engine, predictor, tmp_path, manager=None, **kw):
    return PortfolioSimulator(
        config, engine, predictor, manager=manager,
        floor_state_path=tmp_path / "floor_state.json", **kw,
    )


def _brief(per_inst=None, stage=0, ceiling=1.5):
    return {
        "growth_stage": stage,
        "risk_ceiling_now": ceiling,
        "extra": {"sim": {"per_instrument": per_inst or {}}},
    }


def _inst_stats(pf=None, trades=0, consec=0, weight=1.0):
    return {
        "trailing_pf": pf,
        "trailing_trades": trades,
        "consecutive_losses": consec,
        "weight": weight,
    }


def _make_synth_data(days=2, start="2026-06-01", seed=7, base=1.10):
    """Synthetic M1/M15 random-walk OHLCV covering full UTC days."""
    idx = pd.date_range(start, periods=days * 24 * 60, freq="1min")
    rng = np.random.default_rng(seed)
    close = base + np.cumsum(rng.normal(0, 0.00015, len(idx)))
    open_ = np.concatenate([[base], close[:-1]])
    spread = np.abs(rng.normal(0, 0.0001, len(idx)))
    m1 = pd.DataFrame({
        "open": open_,
        "high": np.maximum(open_, close) + spread,
        "low": np.minimum(open_, close) - spread,
        "close": close,
        "volume": rng.integers(1, 100, len(idx)).astype(float),
    }, index=idx)
    m15 = m1.resample("15min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    }).dropna()
    return m1, m15


# ---------------------------------------------------------------- heuristic rules

class TestHeuristicManager:
    def test_mute_after_five_consecutive_losses(self):
        hm = HeuristicManager()
        proposals, _, usage = hm.propose(_brief({"EUR_USD": _inst_stats(consec=5, weight=1.0)}))
        assert proposals == [{
            "key": "weight.EUR_USD", "value": 0.0,
            "reason": proposals[0]["reason"],
        }]
        assert "consecutive losses" in proposals[0]["reason"]
        assert usage["cost_zar"] == 0.0

    def test_already_muted_instrument_not_re_muted(self):
        hm = HeuristicManager()
        proposals, _, _ = hm.propose(_brief({"EUR_USD": _inst_stats(consec=7, weight=0.0)}))
        assert proposals == []

    def test_weight_reduced_when_pf_below_0_8_over_20_trades(self):
        hm = HeuristicManager()
        proposals, _, _ = hm.propose(_brief({"EUR_USD": _inst_stats(pf=0.5, trades=20, weight=1.0)}))
        assert proposals[0]["key"] == "weight.EUR_USD"
        assert proposals[0]["value"] == pytest.approx(0.75)

    def test_weight_reduction_floors_at_zero(self):
        hm = HeuristicManager()
        proposals, _, _ = hm.propose(_brief({"EUR_USD": _inst_stats(pf=0.2, trades=25, weight=0.1)}))
        assert proposals[0]["value"] == 0.0

    def test_weight_raised_when_pf_above_1_5(self):
        hm = HeuristicManager()
        proposals, _, _ = hm.propose(_brief({"XAU_USD": _inst_stats(pf=2.0, trades=20, weight=1.0)}))
        assert proposals[0]["key"] == "weight.XAU_USD"
        assert proposals[0]["value"] == pytest.approx(1.25)

    def test_weight_raise_can_propose_above_bound_for_clamping(self):
        # At weight 1.4 the raw proposal is 1.65 (> 1.5 bound) — the shared
        # validate_and_clamp gate is responsible for clamping it.
        hm = HeuristicManager()
        proposals, _, _ = hm.propose(_brief({"XAU_USD": _inst_stats(pf=3.0, trades=20, weight=1.4)}))
        assert proposals[0]["value"] == pytest.approx(1.65)

    def test_no_weight_raise_at_cap(self):
        hm = HeuristicManager()
        proposals, _, _ = hm.propose(_brief({"XAU_USD": _inst_stats(pf=3.0, trades=20, weight=1.5)}))
        assert proposals == []

    def test_no_weight_change_under_20_trades(self):
        hm = HeuristicManager()
        proposals, _, _ = hm.propose(_brief({"EUR_USD": _inst_stats(pf=0.1, trades=19, weight=1.0)}))
        assert proposals == []

    def test_pf_exactly_at_boundaries_no_proposal(self):
        hm = HeuristicManager()
        per = {
            "EUR_USD": _inst_stats(pf=0.8, trades=20, weight=1.0),
            "GBP_USD": _inst_stats(pf=1.5, trades=20, weight=1.0),
        }
        proposals, _, _ = hm.propose(_brief(per))
        assert proposals == []

    def test_risk_nudge_after_milestone_crossed(self):
        hm = HeuristicManager()
        first, _, _ = hm.propose(_brief(stage=0, ceiling=1.5))
        assert first == []  # no previous stage -> no nudge
        second, _, _ = hm.propose(_brief(stage=1, ceiling=1.8))
        assert second == [{
            "key": "risk.risk_per_trade_pct", "value": 1.8,
            "reason": second[0]["reason"],
        }]
        # No repeat nudge while the stage holds.
        third, _, _ = hm.propose(_brief(stage=1, ceiling=1.8))
        assert third == []

    def test_no_nudge_when_stage_drops(self):
        hm = HeuristicManager()
        hm.propose(_brief(stage=2, ceiling=2.0))
        proposals, _, _ = hm.propose(_brief(stage=1, ceiling=1.8))
        assert proposals == []

    def test_at_most_three_proposals_mutes_first(self):
        hm = HeuristicManager()
        hm.propose(_brief(stage=0, ceiling=1.5))
        per = {
            "A_USD": _inst_stats(consec=5, weight=1.0),
            "B_USD": _inst_stats(consec=6, weight=1.0),
            "C_USD": _inst_stats(pf=0.4, trades=20, weight=1.0),
            "D_USD": _inst_stats(pf=2.4, trades=20, weight=1.0),
        }
        proposals, _, _ = hm.propose(_brief(per, stage=1, ceiling=1.8))
        assert len(proposals) == 3
        keys = [p["key"] for p in proposals]
        # priority: mutes, then risk nudge, then weight changes
        assert keys[:2] == ["weight.A_USD", "weight.B_USD"]
        assert keys[2] == "risk.risk_per_trade_pct"


# ---------------------------------------------------------------- overlay sizing

class TestOverlaySizing:
    ENTRY_KW = dict(
        atr=0.0100,  # huge ATR -> sl_pips deterministic at the per-inst max
        atr_ratio=1.0,
        pip_size=1e-4,
        pip_value_per_unit=1e-4,
        spread_pips=1.0,
        confidence=0.5,
    )

    def _trade(self, config, engine, predictor, weight, balance=1_000_000.0, risk_pct=None):
        sim = BacktestSimulator(config, engine, predictor)
        overlay = SimParamOverlay(1.5, 0.55, 0.50)
        overlay.weights = {"EUR_USD": weight}
        overlay.risk_pct = risk_pct
        sim.param_overlay = overlay
        state = SimState(balance=balance, starting_balance=balance)
        candle = pd.Series({"open": 1.1, "high": 1.101, "low": 1.099, "close": 1.1, "volume": 10.0})
        return sim._create_trade(
            state, "EUR_USD", "buy", candle, pd.Timestamp("2026-06-01 10:00"),
            **self.ENTRY_KW,
        )

    def test_weight_scales_position_size(self, config, engine, predictor):
        t_02 = self._trade(config, engine, predictor, weight=0.2)
        t_01 = self._trade(config, engine, predictor, weight=0.1)
        assert t_02 is not None and t_01 is not None
        assert t_02.position_size == pytest.approx(2 * t_01.position_size)

    def test_weight_zero_rejects_trade(self, config, engine, predictor):
        assert self._trade(config, engine, predictor, weight=0.0) is None

    def test_no_overlay_matches_weight_one(self, config, engine, predictor):
        sim = BacktestSimulator(config, engine, predictor)  # no overlay at all
        state = SimState(balance=1_000_000.0, starting_balance=1_000_000.0)
        candle = pd.Series({"open": 1.1, "high": 1.101, "low": 1.099, "close": 1.1, "volume": 10.0})
        t_plain = sim._create_trade(
            state, "EUR_USD", "buy", candle, pd.Timestamp("2026-06-01 10:00"),
            **self.ENTRY_KW,
        )
        t_w1 = self._trade(config, engine, predictor, weight=1.0)
        assert t_plain.position_size == t_w1.position_size
        assert t_plain.risk_amount == pytest.approx(t_w1.risk_amount)

    def test_tuned_risk_ratio_scales_sizing(self, config, engine, predictor):
        # Tuned 0.75 vs base 1.5 -> ratio 0.5 -> half the risk amount.
        # Small weight keeps the leverage clamp non-binding.
        t_base = self._trade(config, engine, predictor, weight=0.1, risk_pct=None)
        t_half = self._trade(config, engine, predictor, weight=0.1, risk_pct=0.75)
        assert t_half.risk_amount == pytest.approx(t_base.risk_amount * 0.5)

    def test_risk_scaling_clamped_to_lever_bounds(self):
        overlay = SimParamOverlay(1.5, 0.55, 0.50)
        overlay.risk_pct = 2.5  # ratio 1.667 on a 2.5% instrument -> 4.17% raw
        assert overlay.effective_risk_pct(0.025) == pytest.approx(0.025)  # capped at 2.5%
        overlay.risk_pct = 0.5  # ratio 0.333 on a 1.0% instrument -> 0.33% raw
        assert overlay.effective_risk_pct(0.010) == pytest.approx(0.005)  # floored at 0.5%

    def test_ml_threshold_ratio_scaling(self):
        overlay = SimParamOverlay(1.5, 0.55, 0.50)
        low, high = overlay.scaled_ml_thresholds(0.10, 0.18)
        assert (low, high) == (0.10, 0.18)  # identity before any tune
        overlay.threshold_low = 0.60   # 1.2x base 0.50
        overlay.threshold_high = 0.66  # 1.2x base 0.55
        low, high = overlay.scaled_ml_thresholds(0.10, 0.18)
        assert low == pytest.approx(0.12)
        assert high == pytest.approx(0.216)


# ---------------------------------------------------------------- validate/clamp path

class _StubManager:
    name = "stub"

    def __init__(self, proposals):
        self._proposals = proposals

    def propose(self, briefing):
        return list(self._proposals), "stub rationale", {
            "input_tokens": 0, "output_tokens": 0, "cost_zar": 0.0,
        }


class TestValidateAndClampPath:
    def test_out_of_bounds_proposals_get_clamped(self, config, engine, predictor, tmp_path):
        stub = _StubManager([
            {"key": "weight.EUR_USD", "value": 9.0, "reason": "way too big"},
            {"key": "risk.risk_per_trade_pct", "value": 99.0, "reason": "absurd"},
        ])
        ps = _portfolio(config, engine, predictor, tmp_path, manager=stub)
        state = SimState(balance=1000.0, starting_balance=1000.0)
        decision = ps._run_manager_cycle(pd.Timestamp("2026-06-01 10:00"), state, equity=1000.0)

        applied = {a["key"]: a for a in decision["applied"]}
        # weight clamped to the 1.5 bound
        assert applied["weight.EUR_USD"]["value"] == 1.5
        assert applied["weight.EUR_USD"]["clamped"] is True
        # risk clamped to risk_ceiling_now (1.5 at R1000, below first milestone)
        assert applied["risk.risk_per_trade_pct"]["value"] == 1.5
        assert applied["risk.risk_per_trade_pct"]["clamped"] is True
        # ... and the sim's effective params were mutated with CLAMPED values
        assert ps.overlay.weight("EUR_USD") == 1.5
        assert ps.overlay.risk_pct == 1.5

    def test_bad_key_rejected_and_not_applied(self, config, engine, predictor, tmp_path):
        stub = _StubManager([
            {"key": "risk.min_floor_zar", "value": 1.0, "reason": "nope"},
            {"key": "weight.NOT_AN_INSTRUMENT", "value": 1.0, "reason": "nope"},
        ])
        ps = _portfolio(config, engine, predictor, tmp_path, manager=stub)
        state = SimState(balance=1000.0, starting_balance=1000.0)
        decision = ps._run_manager_cycle(pd.Timestamp("2026-06-01 10:00"), state, equity=1000.0)
        assert decision["applied"] == []
        assert decision["outcome"] == "no_op"
        assert {r["rejection_reason"] for r in decision["rejected"]} == {
            "unknown_key", "bad_instrument",
        }
        assert ps.overlay.weights == {}
        assert ps.overlay.risk_pct is None

    def test_heuristic_raise_gets_clamped_through_policy(self, config, engine, predictor, tmp_path):
        hm = HeuristicManager()
        ps = _portfolio(config, engine, predictor, tmp_path, manager=hm)
        # Rig sim state so trailing-20 PF > 1.5 at weight 1.4 -> raw 1.65.
        ps._instruments = ["EUR_USD"]
        ps._closed_by_inst = {"EUR_USD": []}
        ps._consec_by_inst = {"EUR_USD": 0}
        ps.overlay.weights["EUR_USD"] = 1.4

        class _T:
            def __init__(self, pnl):
                self.pnl = pnl
        ps._closed_by_inst["EUR_USD"] = [_T(10.0)] * 19 + [_T(-1.0)]

        state = SimState(balance=1000.0, starting_balance=1000.0)
        decision = ps._run_manager_cycle(pd.Timestamp("2026-06-01 10:00"), state, equity=1000.0)
        applied = {a["key"]: a for a in decision["applied"]}
        assert applied["weight.EUR_USD"]["original_value"] == pytest.approx(1.65)
        assert applied["weight.EUR_USD"]["value"] == 1.5
        assert applied["weight.EUR_USD"]["clamped"] is True
        assert ps.overlay.weight("EUR_USD") == 1.5

    def test_decision_logged_with_cost(self, config, engine, predictor, tmp_path):
        stub = _StubManager([])
        ps = _portfolio(config, engine, predictor, tmp_path, manager=stub)
        state = SimState(balance=1000.0, starting_balance=1000.0)
        ps._run_manager_cycle(pd.Timestamp("2026-06-01 10:00"), state, equity=1000.0)
        assert len(ps.decisions) == 1
        d = ps.decisions[0]
        assert d["outcome"] == "no_op"
        assert d["cost_zar"] == 0.0
        assert ps.api_cost_total == 0.0
        # decision also lands in the journal adapter for the next briefing
        log = ps.journal.get_manager_log()
        assert len(log) == 1


# ---------------------------------------------------------------- cadence

class TestManagerCadence:
    def test_no_cycle_outside_session_hours(self, config, engine, predictor, tmp_path):
        ps = _portfolio(config, engine, predictor, tmp_path, manager=HeuristicManager())
        ps._mgr_session = (7, 20)
        assert ps._manager_due(pd.Timestamp("2026-06-01 05:00")) is False
        assert ps._manager_due(pd.Timestamp("2026-06-01 20:00")) is False
        assert ps._manager_due(pd.Timestamp("2026-06-01 07:00")) is True

    def test_sixty_minute_spacing(self, config, engine, predictor, tmp_path):
        ps = _portfolio(config, engine, predictor, tmp_path, manager=HeuristicManager())
        ps._mgr_session = (7, 20)
        ps._last_cycle_ts = pd.Timestamp("2026-06-01 10:00")
        assert ps._manager_due(pd.Timestamp("2026-06-01 10:59")) is False
        assert ps._manager_due(pd.Timestamp("2026-06-01 11:00")) is True

    def test_no_cycles_without_manager(self, config, engine, predictor, tmp_path):
        ps = _portfolio(config, engine, predictor, tmp_path, manager=None)
        assert ps._manager_due(pd.Timestamp("2026-06-01 10:00")) is False


# ---------------------------------------------------------------- 21:00 UTC resets

class TestDayBoundary:
    def test_2100_utc_starts_next_trading_day(self):
        assert trading_day(pd.Timestamp("2026-06-01 20:59")) != \
            trading_day(pd.Timestamp("2026-06-01 21:00"))
        assert trading_day(pd.Timestamp("2026-06-01 21:00")) == \
            trading_day(pd.Timestamp("2026-06-02 20:59"))

    def test_daily_reset_clears_counters_and_unpauses(self, config, engine, predictor, tmp_path):
        ps = _portfolio(config, engine, predictor, tmp_path)
        ps._consec_by_inst = {"EUR_USD": 5, "XAU_USD": 2}
        state = SimState(balance=950.0, starting_balance=1000.0,
                         daily_start_balance=1000.0)
        state.consecutive_losses = 4
        state.trade_count_today = 12
        state.daily_pnl = -50.0
        state.is_paused = True
        state.pause_reason = "5 consecutive losses"

        ps._daily_reset(state, pd.Timestamp("2026-06-01 21:00"))

        assert state.daily_start_balance == 950.0
        assert state.daily_pnl == 0.0
        assert state.trade_count_today == 0
        assert state.consecutive_losses == 0
        assert ps._consec_by_inst == {"EUR_USD": 0, "XAU_USD": 0}
        assert state.is_paused is False
        assert state.pause_reason == ""

    def test_daily_reset_keeps_unrelated_pause(self, config, engine, predictor, tmp_path):
        ps = _portfolio(config, engine, predictor, tmp_path)
        state = SimState(balance=1000.0, starting_balance=1000.0)
        state.is_paused = True
        state.pause_reason = "Hard floor hit: R600.00"
        ps._daily_reset(state, pd.Timestamp("2026-06-01 21:00"))
        assert state.is_paused is True


# ---------------------------------------------------------------- floor kill-switch

class TestFloorKillSwitch:
    def test_floor_kill_stops_the_sim(self, config, engine, predictor, tmp_path):
        ps = _portfolio(config, engine, predictor, tmp_path, manager=HeuristicManager())
        # Force a floor above the starting balance: first equity point breaches.
        ps.floor = RatchetFloor(2000.0, 0.35, state_path=tmp_path / "kill_floor.json")
        m1, m15 = _make_synth_data(days=1)
        report = ps.run({"EUR_USD": (m1, m15)})
        assert report["killed_by_floor"] is True
        assert report["kill_info"]["floor"] == 2000.0
        # killed on the very first bar: one equity point, no manager cycles after
        assert len(report["equity_curve"]) == 1
        assert report["manager_decisions"] == []

    def test_floor_semantics_match_ratchet_floor(self, config, engine, predictor, tmp_path):
        ps = _portfolio(config, engine, predictor, tmp_path)
        # Seeded HWM = starting balance (R1000): floor = max(600, 1000*0.65) = 650
        assert ps.floor.current_floor == pytest.approx(650.0)
        assert ps.floor.is_breached(650.0) is True
        assert ps.floor.is_breached(650.01) is False
        # monotonic: HWM never lowers the floor
        ps.floor.update(2000.0)
        assert ps.floor.current_floor == pytest.approx(1300.0)
        ps.floor.update(1000.0)
        assert ps.floor.current_floor == pytest.approx(1300.0)


# ---------------------------------------------------------------- end-to-end smoke

class TestEndToEndSmoke:
    def test_managed_run_completes_with_cycles_and_zero_cost(
        self, config, engine, predictor, tmp_path
    ):
        m1, m15 = _make_synth_data(days=2)
        ps = _portfolio(config, engine, predictor, tmp_path, manager=HeuristicManager())
        report = ps.run({"EUR_USD": (m1, m15)})

        assert report["mode"] == "managed"
        assert report["manager_backend"] == "heuristic"
        assert report["api_cost_zar"] == 0.0
        assert report["starting_balance"] == config.get("account.starting_balance_zar", 1000)
        assert report["manager_cycles"] > 0
        assert len(report["equity_curve"]) > 0
        # net-after-cost curve present and equal to equity at zero cost
        pt = report["equity_curve"][-1]
        assert pt["equity_net"] == pytest.approx(pt["equity"])
        assert report["net_pnl_after_cost_zar"] == pytest.approx(report["total_pnl_zar"], abs=0.01)
        # every cycle happened within manager session hours, >= 60 min apart
        cycle_times = [pd.Timestamp(d["ts_utc"]) for d in report["manager_decisions"]]
        assert all(7 <= t.hour < 20 for t in cycle_times)
        gaps = np.diff([t.value for t in cycle_times])
        assert (gaps >= 60 * 60 * 1e9).all()

    def test_baseline_run_has_no_manager_activity(self, config, engine, predictor, tmp_path):
        m1, m15 = _make_synth_data(days=1)
        ps = _portfolio(config, engine, predictor, tmp_path, manager=None)
        report = ps.run({"EUR_USD": (m1, m15)})
        assert report["mode"] == "baseline"
        assert report["manager_decisions"] == []
        assert report["api_cost_zar"] == 0.0

    def test_multi_instrument_shared_account(self, config, engine, predictor, tmp_path):
        m1_eur, m15_eur = _make_synth_data(days=1, seed=7, base=1.10)
        m1_gold, m15_gold = _make_synth_data(days=1, seed=11, base=2400.0)
        ps = _portfolio(config, engine, predictor, tmp_path, manager=HeuristicManager())
        report = ps.run({"EUR_USD": (m1_eur, m15_eur), "XAU_USD": (m1_gold, m15_gold)})
        assert report["instruments"] == ["EUR_USD", "XAU_USD"]
        assert set(report["per_instrument"]) == {"EUR_USD", "XAU_USD"}
        # one shared account: single equity curve, one manager decision stream
        assert len(report["equity_curve"]) > 0
        assert report["manager_cycles"] > 0

    def test_comparison_table_renders(self, config, engine, predictor, tmp_path):
        m1, m15 = _make_synth_data(days=1)
        base = _portfolio(config, engine, predictor, tmp_path).run({"EUR_USD": (m1, m15)})
        managed = _portfolio(
            config, engine, predictor, tmp_path, manager=HeuristicManager()
        ).run({"EUR_USD": (m1, m15)})
        table = comparison_table(base, managed)
        assert "Baseline" in table and "Managed" in table
        assert "API cost (ZAR)" in table
        assert "Net PnL after cost" in table


# ---------------------------------------------------------------- ClaudeManager (mocked)

class _FakeToolBlock:
    type = "tool_use"

    def __init__(self, input_dict):
        self.input = input_dict


class _FakeUsage:
    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeResponse:
    def __init__(self, adjustments, rationale, input_tokens=1000, output_tokens=100):
        self.content = [_FakeToolBlock({"adjustments": adjustments, "rationale": rationale})]
        self.usage = _FakeUsage(input_tokens, output_tokens)


class _FakeMessages:
    def __init__(self, response):
        self._response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class _FakeAnthropicClient:
    def __init__(self, response):
        self.messages = _FakeMessages(response)


class TestClaudeManagerMocked:
    """ClaudeManager with an injected fake client — never touches the network."""

    def test_proposals_flow_through_clamp_and_cost_is_logged(
        self, config, engine, predictor, tmp_path
    ):
        from backtest.manager_sim import ClaudeManager

        response = _FakeResponse(
            adjustments=[{"key": "weight.EUR_USD", "value": 2.0, "reason": "fake"}],
            rationale="fake rationale",
            input_tokens=2000,
            output_tokens=200,
        )
        manager = ClaudeManager(config, client=_FakeAnthropicClient(response))
        ps = _portfolio(config, engine, predictor, tmp_path, manager=manager)
        state = SimState(balance=1000.0, starting_balance=1000.0)
        decision = ps._run_manager_cycle(pd.Timestamp("2026-06-01 10:00"), state, equity=1000.0)

        # out-of-bounds Claude proposal clamped by the shared policy gate
        assert decision["applied"][0]["value"] == 1.5
        assert decision["applied"][0]["clamped"] is True
        assert ps.overlay.weight("EUR_USD") == 1.5
        # cost accounted per cycle and accumulated
        assert decision["cost_zar"] > 0.0
        assert ps.api_cost_total == pytest.approx(decision["cost_zar"])
        assert manager.total_cost_zar == pytest.approx(decision["cost_zar"])
        # the briefing actually went to the (fake) API
        assert len(manager._client.client.messages.calls) == 1


# ---------------------------------------------------------------- runner CLI

class TestRunnerArgs:
    def test_default_is_baseline_pipeline(self):
        from backtest.runner import _parse_args
        assert _parse_args([]).manager is None

    def test_manager_backends(self):
        from backtest.runner import _parse_args
        assert _parse_args(["--manager", "heuristic"]).manager == "heuristic"
        assert _parse_args(["--manager", "claude"]).manager == "claude"

    def test_unknown_backend_rejected(self):
        from backtest.runner import _parse_args
        with pytest.raises(SystemExit):
            _parse_args(["--manager", "random"])
