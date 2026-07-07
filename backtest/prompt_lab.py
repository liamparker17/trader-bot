"""
Prompt lab — Ralph-loop harness for manager-prompt optimization (Task 18).

Prompt variants live as files `src/manager/prompts/vNNN.md`; `champion.txt`
holds the filename of the current champion (see src/manager/prompts/README.md).

    python -m backtest.prompt_lab --variants v001,v002 --window 7

For each variant, run the managed backtest over the same window, score it,
and append one JSONL row to `backtest/prompt_lab_results.jsonl`:

    {variant, window_days, trades, net_pnl_zar, max_dd_zar, api_cost_zar,
     score, ts_utc}

Scoring: `score = net_pnl_zar - 2 * max_dd_zar` (net-after-cost P&L with a
max-drawdown penalty). Champion selection (`--promote`): strictly-best score
wins and `champion.txt` is updated; ties keep the incumbent champion; losing
variant files are NEVER deleted (audit trail).

Backends:
- `claude` (default): real managed backtest with the Claude manager
  (`--manager=claude`, backtest/manager_sim.py). Requires ANTHROPIC_API_KEY;
  without it the lab exits with a clear message and code 1.
- `heuristic`: managed backtest with the deterministic no-API heuristic
  manager. Note the heuristic ignores the prompt entirely — this backend
  exists to smoke-test the full backtest plumbing, not to compare prompts.
- `smoke`: no backtest at all — deterministic canned results derived from
  the variant name. Exercises the harness (scoring, results file, champion
  promotion) with zero dependencies; used by unit tests and CI.

Ralph-loop mode `--auto N`: N iterations of [run champion + challenger →
promote if the challenger wins → PAUSE]. The lab is only the harness — the
mutation intelligence is the orchestrating Claude Code session, which reads
the printed decision-log analysis, writes the next challenger `vNNN.md`,
and re-invokes the lab.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from src.config import PROJECT_ROOT

logger = logging.getLogger("traderbot.prompt_lab")

PROMPTS_DIR = PROJECT_ROOT / "src" / "manager" / "prompts"
RESULTS_PATH = PROJECT_ROOT / "backtest" / "prompt_lab_results.jsonl"

DRAWDOWN_PENALTY = 2.0


def score_result(result: dict) -> float:
    """score = net_pnl_zar - 2 * max_dd_zar"""
    return float(result["net_pnl_zar"]) - DRAWDOWN_PENALTY * float(result["max_dd_zar"])


def _smoke_run(variant_name: str, prompt_path: Path, window_days: int) -> dict:
    """Deterministic canned result derived from the variant name — harness
    smoke tests only, no backtest, no API."""
    seed = sum(prompt_path.read_bytes()) % 97
    return {
        "trades": 10 + seed % 5,
        "net_pnl_zar": float(50 + seed),
        "max_dd_zar": float(5 + seed % 7),
        "api_cost_zar": 0.0,
    }


def _managed_backtest_run(backend: str) -> Callable[[str, Path, int], dict]:
    """
    Real managed-backtest runner (Task 15's backtest/manager_sim.py),
    resolved lazily so the harness itself has no import-time dependency
    on the simulator stack.
    """
    def run(variant_name: str, prompt_path: Path, window_days: int) -> dict:
        from backtest import manager_sim  # lazy: Task 15 module

        return manager_sim.run_managed_backtest_for_prompt(
            backend=backend,
            prompt_path=prompt_path,
            window_days=window_days,
        )

    return run


class PromptLab:
    def __init__(
        self,
        prompts_dir: Path = PROMPTS_DIR,
        results_path: Path = RESULTS_PATH,
        run_backtest_fn: Optional[Callable[[str, Path, int], dict]] = None,
    ):
        self.prompts_dir = prompts_dir
        self.results_path = results_path
        self.run_backtest_fn = run_backtest_fn or _managed_backtest_run("claude")

    # ------------------------------------------------------------------

    def _variant_path(self, variant: str) -> Path:
        name = variant if variant.endswith(".md") else f"{variant}.md"
        path = self.prompts_dir / name
        if not path.exists():
            raise FileNotFoundError(f"prompt variant not found: {path}")
        return path

    def _current_champion(self) -> Optional[str]:
        champion_file = self.prompts_dir / "champion.txt"
        if not champion_file.exists():
            return None
        name = champion_file.read_text(encoding="utf-8").strip()
        return name.removesuffix(".md") if name else None

    def _append_result(self, row: dict) -> None:
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with self.results_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    def _promote(self, variant: str) -> None:
        """Point champion.txt at `variant`. Never deletes losing variants."""
        (self.prompts_dir / "champion.txt").write_text(
            f"{variant}.md", encoding="utf-8")
        logger.info(f"prompt lab: promoted {variant} to champion")

    # ------------------------------------------------------------------

    def run(self, variants: list, window_days: int, promote: bool = False) -> dict:
        """
        Run each variant over the same window, append a JSONL row per
        variant, and return {results, winner, promoted}.

        Winner = highest score; on a tie the incumbent champion (or the
        earlier-listed variant if no incumbent is involved) is kept —
        promotion requires a STRICT improvement.
        """
        results = {}
        for variant in variants:
            path = self._variant_path(variant)
            result = self.run_backtest_fn(variant, path, window_days)
            row = {
                "variant": variant,
                "window_days": window_days,
                "trades": result.get("trades"),
                "net_pnl_zar": result.get("net_pnl_zar"),
                "max_dd_zar": result.get("max_dd_zar"),
                "api_cost_zar": result.get("api_cost_zar"),
                "score": score_result(result),
                "ts_utc": datetime.now(timezone.utc).isoformat(),
            }
            self._append_result(row)
            results[variant] = row
            logger.info(
                f"prompt lab: {variant} scored {row['score']:.2f} "
                f"(pnl R{row['net_pnl_zar']:.2f}, dd R{row['max_dd_zar']:.2f})"
            )

        incumbent = self._current_champion()
        winner = None
        best_score = float("-inf")
        # Evaluate the incumbent first so any challenger must STRICTLY beat it.
        ordered = sorted(results, key=lambda v: v != incumbent)
        for variant in ordered:
            if results[variant]["score"] > best_score:
                best_score = results[variant]["score"]
                winner = variant

        promoted = False
        if promote and winner is not None and winner != incumbent:
            self._promote(winner)
            promoted = True

        return {"results": results, "winner": winner, "promoted": promoted}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m backtest.prompt_lab",
        description="Manager-prompt optimization harness (Task 18)",
    )
    parser.add_argument("--variants", required=True,
                        help="Comma-separated variant names, e.g. v001,v002")
    parser.add_argument("--window", type=int, default=7,
                        help="Backtest window in days (default 7)")
    parser.add_argument("--backend", choices=["claude", "heuristic", "smoke"],
                        default="claude")
    parser.add_argument("--promote", action="store_true",
                        help="Update champion.txt if a challenger strictly wins")
    parser.add_argument("--auto", type=int, default=None, metavar="N",
                        help="Ralph-loop mode: champion-vs-challenger, promote on "
                             "win, then pause for the orchestrator to mutate")
    parser.add_argument("--prompts-dir", type=Path, default=PROMPTS_DIR)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    return parser


def main(argv: Optional[list] = None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args(argv)

    if args.backend == "claude" and not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "prompt lab: ANTHROPIC_API_KEY is not set — the claude backend "
            "needs it. Set the key, or use --backend heuristic|smoke for a "
            "no-API harness run.",
            file=sys.stderr,
        )
        return 1

    if args.backend == "smoke":
        run_fn = _smoke_run
    else:
        run_fn = _managed_backtest_run(args.backend)

    lab = PromptLab(prompts_dir=args.prompts_dir, results_path=args.results,
                    run_backtest_fn=run_fn)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    if args.auto is not None:
        # Ralph loop: each iteration runs champion + latest challenger and
        # pauses. The orchestrating session writes the next mutation.
        incumbent = lab._current_champion()
        if incumbent is None:
            print("prompt lab: no champion.txt — cannot run --auto", file=sys.stderr)
            return 1
        challengers = [v for v in variants if v != incumbent]
        if not challengers:
            print("prompt lab: --auto needs at least one non-champion variant",
                  file=sys.stderr)
            return 1
        outcome = lab.run([incumbent, challengers[-1]], window_days=args.window,
                          promote=True)
        print(json.dumps({
            "mode": "auto",
            "iteration_complete": True,
            "outcome": outcome,
            "next_step": (
                "Orchestrator: analyze the decision log in the results file, "
                "write the next challenger vNNN.md mutation, and re-invoke "
                "with --auto."
            ),
        }, indent=2))
        return 0

    outcome = lab.run(variants, window_days=args.window, promote=args.promote)
    print(json.dumps(outcome, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
