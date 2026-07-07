"""
tb — command-line control interface for TraderBot (Task 9).

Usage:
    python -m cli.tb <command> [options]

Every command prints exactly one JSON document to stdout and exits 0 on
success, 1 on failure (including business-rejections from the bot side
and CLI-level validation errors, which are printed as {"error": ...}).

Read commands (status, trades, perf, positions, config, logs, model,
manager) are pure consumers of the trade journal / effective-config /
model-store / log files on disk — they never modify src/ state.

Write commands (pause, resume, tune, revert) drop a command file into
`control/inbox/` and poll `control/outbox/` for the matching result,
mirroring the file-based protocol implemented by src/control/queue.py
(Task 8). If the bot process isn't running (nothing ever drains the
inbox), the poll times out and the CLI reports that explicitly rather
than hanging forever.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from src.config import Config, PROJECT_ROOT, load_config
from src.control.effective_config import EffectiveConfig, SETTINGS_PATH, _load_yaml
from src.control.queue import CMD_ID_RE, INBOX_DIR, MIN_REASON_LEN, OUTBOX_DIR, TUNE_BOUNDS
from src.monitoring.trade_journal import TradeJournal

DEFAULT_POLL_TIMEOUT_SECS = 5.0
DEFAULT_POLL_INTERVAL_SECS = 0.1

LOG_PATH = PROJECT_ROOT / "data" / "logs" / "traderbot.log"
MODEL_STORE = PROJECT_ROOT / "src" / "ml" / "model_store"


class TbError(Exception):
    """User-facing CLI error — caught in main() and reported as {"error": ...}, exit 1."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cutoff_iso(days: Optional[int]) -> Optional[str]:
    if days is None:
        return None
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def _df_records(df) -> list:
    """
    Convert a pandas DataFrame to a list of JSON-safe records (NaN/NaT ->
    null) without hand-rolling NaN cleanup — round-tripping through
    `to_json` lets pandas do that conversion correctly.
    """
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records", date_format="iso"))


def _table_exists(db_path: Path, table_name: str) -> bool:
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        return cur.fetchone() is not None


# ---------------------------------------------------------------------
# Control-queue I/O (write-command round trip)
# ---------------------------------------------------------------------

def enqueue_command(
    verb: str,
    args: dict,
    reason: str,
    requested_by: str,
    inbox_dir: Path,
    outbox_dir: Path,
    timeout: float = DEFAULT_POLL_TIMEOUT_SECS,
    poll_interval: float = DEFAULT_POLL_INTERVAL_SECS,
) -> Optional[dict]:
    """
    Drop a command file into `inbox_dir` (tmp-write + os.replace, per the
    ControlQueue writer contract) and poll `outbox_dir` for the matching
    result. Returns the parsed result dict, or None if no result appeared
    within `timeout` seconds (bot not running / not draining the inbox).
    """
    cmd_id = uuid.uuid4().hex
    if not CMD_ID_RE.match(cmd_id):  # pragma: no cover — defensive; uuid4 hex always matches
        raise TbError(f"generated an invalid command id: {cmd_id!r}")

    inbox_dir.mkdir(parents=True, exist_ok=True)
    outbox_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "id": cmd_id,
        "verb": verb,
        "args": args,
        "reason": reason,
        "requested_at": _now_iso(),
        "requested_by": requested_by,
    }

    tmp_path = inbox_dir / f"{cmd_id}.cmd.json.tmp"
    final_path = inbox_dir / f"{cmd_id}.cmd.json"
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, final_path)

    result_path = outbox_dir / f"{cmd_id}.result.json"
    deadline = time.monotonic() + timeout
    while True:
        if result_path.exists():
            try:
                return json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass  # result file mid-write; keep polling
        if time.monotonic() >= deadline:
            return None
        time.sleep(poll_interval)


def _validate_reason(reason: Optional[str]) -> str:
    if not reason or len(reason.strip()) < MIN_REASON_LEN:
        raise TbError(f"--reason is required and must be at least {MIN_REASON_LEN} characters")
    return reason


def _parse_tune_assignment(assignment: str) -> tuple[str, str]:
    if "=" not in assignment:
        raise TbError("tune argument must be in the form key=value")
    key, _, value = assignment.partition("=")
    key, value = key.strip(), value.strip()
    if not key or not value:
        raise TbError("tune argument must be in the form key=value")
    return key, value


def _enqueue_and_report(
    verb: str,
    args: dict,
    reason: str,
    inbox_dir: Path,
    outbox_dir: Path,
    timeout: float,
    poll_interval: float,
) -> dict:
    result = enqueue_command(
        verb, args, reason=reason, requested_by="cli",
        inbox_dir=inbox_dir, outbox_dir=outbox_dir,
        timeout=timeout, poll_interval=poll_interval,
    )
    if result is None:
        raise TbError(
            f"timed out after {timeout}s waiting for the bot to process '{verb}' "
            "— is the bot running?"
        )
    return result


# ---------------------------------------------------------------------
# Read commands
# ---------------------------------------------------------------------

def cmd_status(
    journal: TradeJournal,
    inbox_dir: Path,
    outbox_dir: Path,
    timeout: float,
    poll_interval: float,
) -> dict:
    result = enqueue_command(
        "status_snapshot", {}, reason="", requested_by="cli",
        inbox_dir=inbox_dir, outbox_dir=outbox_dir,
        timeout=timeout, poll_interval=poll_interval,
    )
    if result is not None and result.get("outcome") == "applied" and isinstance(result.get("detail"), dict):
        snapshot = dict(result["detail"])
        snapshot["bot_running"] = True
        snapshot["generated_at"] = _now_iso()
        return snapshot

    return _journal_derived_status(journal)


def _journal_derived_status(journal: TradeJournal) -> dict:
    today = datetime.now(timezone.utc).date().isoformat()
    todays_pnl = None
    open_positions = None

    try:
        todays = journal.get_trades(since=today, limit=10000)
        if todays is not None and not todays.empty:
            pnl_col = "net_pnl_zar" if "net_pnl_zar" in todays.columns else "pnl_zar"
            if pnl_col in todays.columns:
                todays_pnl = float(todays[pnl_col].fillna(0).sum())
    except Exception:
        todays_pnl = None

    try:
        all_trades = journal.get_trades(limit=100000)
        if all_trades is not None and not all_trades.empty and "exit_price" in all_trades.columns:
            open_positions = int(all_trades["exit_price"].isna().sum())
    except Exception:
        open_positions = None

    return {
        "bot_running": False,
        "generated_at": _now_iso(),
        "todays_pnl": todays_pnl,
        "open_positions": open_positions,
    }


def cmd_trades(journal: TradeJournal, days: Optional[int]) -> dict:
    df = journal.get_trades(since=_cutoff_iso(days), limit=10000)
    records = _df_records(df)
    return {"count": len(records), "trades": records}


def cmd_perf(journal: TradeJournal, days: Optional[int]) -> dict:
    df = journal.get_trades(since=_cutoff_iso(days), limit=100000)
    if df is None or df.empty:
        completed = df
    else:
        completed = df[df["exit_price"].notna()]

    if completed is None or completed.empty:
        return {
            "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
            "profit_factor": None, "total_pnl": 0.0,
        }

    pnl_col = "net_pnl_zar" if "net_pnl_zar" in completed.columns and completed["net_pnl_zar"].notna().any() else "pnl_zar"
    pnls = completed[pnl_col].fillna(0)
    wins = completed[pnls > 0]
    losses = completed[pnls <= 0]
    gross_profit = float(wins[pnl_col].fillna(0).sum()) if not wins.empty else 0.0
    gross_loss = float(-losses[pnl_col].fillna(0).sum()) if not losses.empty else 0.0

    return {
        "total_trades": int(len(completed)),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate": len(wins) / len(completed) if len(completed) else 0.0,
        "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else None,
        "total_pnl": float(pnls.sum()),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
    }


def cmd_positions(journal: TradeJournal) -> dict:
    df = journal.get_trades(limit=100000)
    if df is not None and not df.empty and "exit_price" in df.columns:
        open_df = df[df["exit_price"].isna()]
    else:
        open_df = df
    records = _df_records(open_df)
    return {"count": len(records), "positions": records}


def cmd_config(config: Config) -> dict:
    eff = EffectiveConfig.load()
    baseline = _load_yaml(SETTINGS_PATH)

    keys = list(TUNE_BOUNDS.keys())
    try:
        instruments = config.instruments.get("instruments", {}) or {}
        keys.extend(f"weight.{name}" for name in instruments.keys())
    except Exception:
        pass

    delta = {}
    for key in keys:
        node = baseline
        for part in key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                node = None
                break
        effective = eff.get(key)
        if node != effective:
            delta[key] = {"baseline": node, "effective": effective}

    return {
        "settings_path": str(SETTINGS_PATH),
        "effective_config_delta": delta,
        "safety_locked_keys": sorted(eff.safety_keys()),
    }


def cmd_logs(tail: int, level: Optional[str]) -> dict:
    if not LOG_PATH.exists():
        return {"log_path": str(LOG_PATH), "lines": []}

    lines = LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
    if level:
        needle = f"| {level.upper()} |"
        lines = [line for line in lines if needle in line]
    if tail:
        lines = lines[-tail:]
    return {"log_path": str(LOG_PATH), "lines": lines}


def cmd_model(journal: TradeJournal) -> dict:
    latest_path = MODEL_STORE / "latest_version.txt"
    if not latest_path.exists():
        return {"version": None, "metadata": None, "detail": "no trained model found"}

    version = latest_path.read_text(encoding="utf-8").strip()
    meta_path = MODEL_STORE / f"model_{version}_meta.json"
    metadata = None
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            metadata = {"error": f"failed to read model metadata: {e}"}

    result: dict = {"version": version, "metadata": metadata}

    if _table_exists(journal.db_path, "evaluator_state"):
        with sqlite3.connect(journal.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM evaluator_state WHERE id = 1").fetchone()
            result["evaluator_state"] = dict(row) if row is not None else None
    else:
        result["evaluator_state"] = None

    return result


def cmd_manager(
    journal: TradeJournal,
    days: Optional[int],
    verdict: bool,
    baseline_pnl: Optional[float] = None,
) -> dict:
    if verdict:
        # Task 14 self-funding scorecard. `baseline_pnl` is the same-window
        # P&L of the no-API heuristic manager (from the managed backtest,
        # Task 15/16) — without it the verdict is conservative.
        from src.monitoring.performance import PerformanceTracker

        if not _table_exists(journal.db_path, "manager_log"):
            return {"verdict": "PENDING", "reason": "manager not yet active"}
        return PerformanceTracker(journal).justification_report(
            heuristic_baseline_pnl_zar=baseline_pnl,
        )

    if not _table_exists(journal.db_path, "manager_log"):
        # Task 12 creates manager_log — until then, an empty result is
        # correct (there is nothing to report), not an error.
        return {"count": 0, "entries": []}

    query = "SELECT * FROM manager_log"
    params: list = []
    cutoff = _cutoff_iso(days)
    if cutoff is not None:
        query += " WHERE ts_utc >= ?"
        params.append(cutoff)
    query += " ORDER BY id DESC"

    with sqlite3.connect(journal.db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()
        entries = [dict(row) for row in rows]

    return {"count": len(entries), "entries": entries}


# ---------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tb", description="TraderBot control CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Bot status (live round-trip, degrades to journal-derived)")

    p_trades = sub.add_parser("trades", help="Recent trades")
    p_trades.add_argument("--days", type=int, default=None)

    p_perf = sub.add_parser("perf", help="Performance summary")
    p_perf.add_argument("--days", type=int, default=None)

    sub.add_parser("positions", help="Currently open positions (journal-derived)")
    sub.add_parser("config", help="Baseline settings vs effective-config overlay delta")

    p_logs = sub.add_parser("logs", help="Tail the bot's log file")
    p_logs.add_argument("--tail", type=int, default=100)
    p_logs.add_argument("--level", type=str, default=None)

    sub.add_parser("model", help="Current ML model version/metadata + evaluator state")

    p_manager = sub.add_parser("manager", help="Claude-manager audit log")
    p_manager.add_argument("--days", type=int, default=None)
    p_manager.add_argument("--verdict", action="store_true")
    p_manager.add_argument("--baseline-pnl", type=float, default=None,
                           help="Heuristic-manager same-window P&L (ZAR) for the uplift check")

    # `--reason` is intentionally NOT argparse `required=True`: a missing
    # --reason should surface as our own {"error": ...} JSON on stdout via
    # `_validate_reason`, not argparse's stderr usage message + exit(2).
    p_pause = sub.add_parser("pause", help="Manually pause new trade entries")
    p_pause.add_argument("--reason", type=str, default=None)

    p_resume = sub.add_parser("resume", help="Resume trade entries")
    p_resume.add_argument("--reason", type=str, default=None)

    p_tune = sub.add_parser("tune", help="Tune a whitelisted config key: key=value")
    p_tune.add_argument("assignment", type=str)
    p_tune.add_argument("--reason", type=str, default=None)

    p_revert = sub.add_parser("revert", help="Revert the last applied tune")
    p_revert.add_argument("--reason", type=str, default=None)

    return parser


def main(
    argv: Optional[list] = None,
    config: Optional[Config] = None,
    journal: Optional[TradeJournal] = None,
    inbox_dir: Optional[Path] = None,
    outbox_dir: Optional[Path] = None,
    timeout: float = DEFAULT_POLL_TIMEOUT_SECS,
    poll_interval: float = DEFAULT_POLL_INTERVAL_SECS,
) -> int:
    """
    Entry point. `config`/`journal`/`inbox_dir`/`outbox_dir`/`timeout`/
    `poll_interval` are injectable for tests; real CLI usage (`python -m
    cli.tb ...`) leaves them at their production defaults.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    inbox_dir = inbox_dir if inbox_dir is not None else INBOX_DIR
    outbox_dir = outbox_dir if outbox_dir is not None else OUTBOX_DIR

    payload: dict
    exit_code = 0

    try:
        cfg = config or load_config()
        jrn = journal or TradeJournal(cfg)

        if args.command == "status":
            payload = cmd_status(jrn, inbox_dir, outbox_dir, timeout, poll_interval)
        elif args.command == "trades":
            payload = cmd_trades(jrn, args.days)
        elif args.command == "perf":
            payload = cmd_perf(jrn, args.days)
        elif args.command == "positions":
            payload = cmd_positions(jrn)
        elif args.command == "config":
            payload = cmd_config(cfg)
        elif args.command == "logs":
            payload = cmd_logs(args.tail, args.level)
        elif args.command == "model":
            payload = cmd_model(jrn)
        elif args.command == "manager":
            payload = cmd_manager(jrn, args.days, args.verdict, args.baseline_pnl)
        elif args.command == "pause":
            reason = _validate_reason(args.reason)
            payload = _enqueue_and_report("pause", {}, reason, inbox_dir, outbox_dir, timeout, poll_interval)
            exit_code = 0 if payload.get("outcome") == "applied" else 1
        elif args.command == "resume":
            reason = _validate_reason(args.reason)
            payload = _enqueue_and_report("resume", {}, reason, inbox_dir, outbox_dir, timeout, poll_interval)
            exit_code = 0 if payload.get("outcome") == "applied" else 1
        elif args.command == "tune":
            key, value = _parse_tune_assignment(args.assignment)
            reason = _validate_reason(args.reason)
            payload = _enqueue_and_report(
                "tune", {"key": key, "value": value}, reason, inbox_dir, outbox_dir, timeout, poll_interval,
            )
            exit_code = 0 if payload.get("outcome") == "applied" else 1
        elif args.command == "revert":
            reason = _validate_reason(args.reason)
            payload = _enqueue_and_report("revert", {}, reason, inbox_dir, outbox_dir, timeout, poll_interval)
            exit_code = 0 if payload.get("outcome") == "applied" else 1
        else:  # pragma: no cover — argparse `required=True` on subparsers keeps this unreachable
            raise TbError(f"unknown command: {args.command}")
    except TbError as e:
        print(json.dumps({"error": str(e)}))
        return 1
    except Exception as e:  # noqa: BLE001 — top-level CLI boundary: never leak a traceback to stdout
        print(json.dumps({"error": f"unexpected error: {e}"}))
        return 1

    print(json.dumps(payload, indent=2, default=str))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
