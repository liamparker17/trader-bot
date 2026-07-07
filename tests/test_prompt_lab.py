"""
Task 18 — backtest/prompt_lab.py: Ralph-loop harness for manager-prompt
optimization.

Covers (all with fake/heuristic backends — no network, no API key):
- Scoring math: score = net_pnl_zar - 2 * max_dd_zar.
- Champion promotion: best score wins, champion.txt updated, losing
  variants never deleted.
- Results file: one JSONL row appended per variant run.
- No ANTHROPIC_API_KEY -> clear message, graceful exit (no crash).
"""
import json
import os
from pathlib import Path

import pytest

from backtest.prompt_lab import (
    PromptLab,
    score_result,
)


def _fake_run(results_by_variant):
    """Return an injectable run_backtest_fn serving canned results."""
    calls = []

    def run(variant_name: str, prompt_path: Path, window_days: int):
        calls.append((variant_name, window_days))
        return dict(results_by_variant[variant_name])

    run.calls = calls
    return run


@pytest.fixture
def prompts_dir(tmp_path):
    d = tmp_path / "prompts"
    d.mkdir()
    (d / "v001.md").write_text("PROMPT ONE", encoding="utf-8")
    (d / "v002.md").write_text("PROMPT TWO", encoding="utf-8")
    (d / "champion.txt").write_text("v001.md", encoding="utf-8")
    return d


@pytest.fixture
def results_path(tmp_path):
    return tmp_path / "prompt_lab_results.jsonl"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def test_score_is_net_pnl_minus_twice_max_dd():
    assert score_result({"net_pnl_zar": 100.0, "max_dd_zar": 20.0}) == pytest.approx(60.0)


def test_score_negative_pnl():
    assert score_result({"net_pnl_zar": -50.0, "max_dd_zar": 30.0}) == pytest.approx(-110.0)


# ---------------------------------------------------------------------------
# Variant runs + results file
# ---------------------------------------------------------------------------

def test_run_variants_appends_jsonl_rows(prompts_dir, results_path):
    run_fn = _fake_run({
        "v001": {"trades": 10, "net_pnl_zar": 100.0, "max_dd_zar": 20.0, "api_cost_zar": 5.0},
        "v002": {"trades": 12, "net_pnl_zar": 80.0, "max_dd_zar": 5.0, "api_cost_zar": 4.0},
    })
    lab = PromptLab(prompts_dir=prompts_dir, results_path=results_path,
                    run_backtest_fn=run_fn)
    outcome = lab.run(["v001", "v002"], window_days=7)

    lines = [json.loads(l) for l in results_path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert lines[0]["variant"] == "v001"
    assert lines[0]["window_days"] == 7
    assert lines[0]["score"] == pytest.approx(60.0)
    assert lines[1]["variant"] == "v002"
    assert lines[1]["score"] == pytest.approx(70.0)
    assert outcome["winner"] == "v002"


def test_results_file_appends_across_runs(prompts_dir, results_path):
    run_fn = _fake_run({
        "v001": {"trades": 1, "net_pnl_zar": 10.0, "max_dd_zar": 1.0, "api_cost_zar": 1.0},
    })
    lab = PromptLab(prompts_dir=prompts_dir, results_path=results_path,
                    run_backtest_fn=run_fn)
    lab.run(["v001"], window_days=7)
    lab.run(["v001"], window_days=7)
    lines = results_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_unknown_variant_raises(prompts_dir, results_path):
    lab = PromptLab(prompts_dir=prompts_dir, results_path=results_path,
                    run_backtest_fn=_fake_run({}))
    with pytest.raises(FileNotFoundError):
        lab.run(["v999"], window_days=7)


# ---------------------------------------------------------------------------
# Champion promotion
# ---------------------------------------------------------------------------

def test_winner_promoted_to_champion(prompts_dir, results_path):
    run_fn = _fake_run({
        "v001": {"trades": 10, "net_pnl_zar": 50.0, "max_dd_zar": 10.0, "api_cost_zar": 5.0},
        "v002": {"trades": 10, "net_pnl_zar": 120.0, "max_dd_zar": 10.0, "api_cost_zar": 5.0},
    })
    lab = PromptLab(prompts_dir=prompts_dir, results_path=results_path,
                    run_backtest_fn=run_fn)
    outcome = lab.run(["v001", "v002"], window_days=7, promote=True)

    assert outcome["winner"] == "v002"
    assert (prompts_dir / "champion.txt").read_text(encoding="utf-8").strip() == "v002.md"
    # Losing variant file is never deleted (audit trail).
    assert (prompts_dir / "v001.md").exists()


def test_no_promotion_without_flag(prompts_dir, results_path):
    run_fn = _fake_run({
        "v001": {"trades": 10, "net_pnl_zar": 50.0, "max_dd_zar": 10.0, "api_cost_zar": 5.0},
        "v002": {"trades": 10, "net_pnl_zar": 120.0, "max_dd_zar": 10.0, "api_cost_zar": 5.0},
    })
    lab = PromptLab(prompts_dir=prompts_dir, results_path=results_path,
                    run_backtest_fn=run_fn)
    lab.run(["v001", "v002"], window_days=7, promote=False)
    assert (prompts_dir / "champion.txt").read_text(encoding="utf-8").strip() == "v001.md"


def test_champion_kept_on_tie_or_loss(prompts_dir, results_path):
    # Challenger scores equal -> incumbent champion retained.
    run_fn = _fake_run({
        "v001": {"trades": 10, "net_pnl_zar": 100.0, "max_dd_zar": 10.0, "api_cost_zar": 5.0},
        "v002": {"trades": 10, "net_pnl_zar": 100.0, "max_dd_zar": 10.0, "api_cost_zar": 5.0},
    })
    lab = PromptLab(prompts_dir=prompts_dir, results_path=results_path,
                    run_backtest_fn=run_fn)
    outcome = lab.run(["v001", "v002"], window_days=7, promote=True)
    assert outcome["winner"] == "v001"
    assert (prompts_dir / "champion.txt").read_text(encoding="utf-8").strip() == "v001.md"


# ---------------------------------------------------------------------------
# No-key graceful exit
# ---------------------------------------------------------------------------

def test_main_without_api_key_exits_gracefully(prompts_dir, results_path, monkeypatch, capsys):
    from backtest.prompt_lab import main

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    rc = main([
        "--variants", "v001,v002", "--window", "7",
        "--prompts-dir", str(prompts_dir),
        "--results", str(results_path),
    ])
    captured = capsys.readouterr()
    assert rc == 1
    assert "ANTHROPIC_API_KEY" in captured.out + captured.err
    assert not results_path.exists()


def test_main_smoke_backend_needs_no_key(prompts_dir, results_path, monkeypatch, capsys):
    """--backend=smoke runs the harness end-to-end with canned results —
    used for harness smoke tests without any API key."""
    from backtest.prompt_lab import main

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    rc = main([
        "--variants", "v001,v002", "--window", "7",
        "--prompts-dir", str(prompts_dir),
        "--results", str(results_path),
        "--backend", "smoke",
    ])
    assert rc == 0
    lines = results_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
