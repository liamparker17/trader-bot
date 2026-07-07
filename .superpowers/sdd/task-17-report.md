# Task 17 Report — VPS Runbook + Live-Cutover Checklist + CLAUDE.md Refresh

## Summary
Docs-only task. Merged `d0c85bd` (Tasks 1-12) into this worktree first (fast-forward, e5f680e → d0c85bd).
Wrote two new runbooks and refreshed CLAUDE.md, all grepped/verified against actual code in this worktree
rather than the plan/spec prose alone.

## Files changed
- `CLAUDE.md` (repo root) — refreshed. Note: this file is listed in `.gitignore` (removed from tracking
  in commit `e5f680e`, "chore: remove Claude config files from tracking") and did not exist in this fresh
  worktree checkout at all (untracked files aren't copied into new worktrees). Recreated it from the
  content in the primary checkout (`C:\Users\liamp\Desktop\Portfolio\TraderBot\CLAUDE.md`), applied the
  requested edits, and force-added (`git add -f`) since the task brief explicitly calls for it "tracked
  in this worktree." Flagging this discrepancy for visibility rather than silently overriding the
  `.gitignore` intent.
- `docs/runbooks/vps-provisioning.md` — new. Hetzner Windows Server provisioning, Windows OpenSSH setup
  + key auth, Python/venv/repo setup, MT5 install + auto-login, two NSSM services (`traderbot` running
  `python -m src.main`, `manager` running `python -m src.manager`) both with `AppRestartDelay=10000` and
  `AppRotateFiles=1`, the local `tb` SSH alias (PowerShell + bash variants), log locations table (with
  paths verified against `src/main.py` and `config/settings.yaml`, not invented), and a post-reboot
  verification checklist for both services.
- `docs/runbooks/live-cutover-r1000.md` — new. The 5-item pre-live hardening gate transcribed from the
  spec (blockers A-I, fixes J-S, fresh audit of manager + weight-path + no-stale-R500/R9000-assumptions,
  control-plane completion, supervised dry-run with deliberate out-of-bounds-proposal test), `.env` swap
  steps (live MT5 creds + `ANTHROPIC_API_KEY`, with a warning that the live MT5 server name differs from
  demo), safety-floor state seeding (delete/archive `data/account_state.json`, confirms `RatchetFloor`
  seeding behavior read directly from `src/risk/ratchet_floor.py`), Telegram verification steps, and a
  first-hour supervision checklist (8 checkable items).

## Verification against code (not just plan prose)
Used a subagent (Explore) plus direct grep/read to confirm before writing, rather than trusting the plan
document's field names:
- `cli/tb.py`: exact subcommands/flags, `MIN_REASON_LEN`-style validation, exit-code convention, 5s
  outbox timeout.
- `src/control/queue.py`: `VALID_VERBS`, `TUNE_BOUNDS` (`risk.risk_per_trade_pct` [0.5,2.5],
  `ml.confidence_threshold_high` [0.50,0.75], `ml.confidence_threshold_low` [0.45,0.65]),
  `WEIGHT_BOUNDS` (0.0,1.5), 24h manual-tune rate limit, `control_log` schema.
- `src/manager/policy.py`: `LEVERS`, `MAX_PROPOSALS_PER_CYCLE = 3`, `risk_ceiling_now` ladder
  (1.5 → 1.8 → 2.0 → 2.2 → 2.5 hard cap).
- `config/settings.yaml` / `config/safety_floor.yaml`: confirmed R1000 re-base already landed
  (`account.starting_balance_zar: 1000`, milestones `[1500,2000,3000,4500,6000]`), confirmed no
  `manager:` block exists yet in this worktree (Task 13 not landed here).
- `src/risk/ratchet_floor.py`: exact seeding behavior (`STARTING_HIGH_WATER_MARK_ZAR = 1000.0`, seeds on
  missing/corrupt state file, does NOT seed to `max(balance, 1000)` directly — corrected an initial
  drafting mistake on this point before committing).
- `src/execution/executor.py`: the actual method is `close_all(reason=...)`, not
  `close_all_positions(...)` as loosely implied by spec prose — corrected in the doc.
- `src/monitoring/trade_journal.py` / `src/main.py`: actual log/db paths (`data/logs/traderbot.log`,
  `data/trade_logs/trades.db`) rather than assumed conventional names.
- Confirmed `src/manager/client.py`, `scheduler.py`, `runner.py` do not exist yet in this worktree
  (Task 13 in progress in parallel) — every reference to them in both new docs and CLAUDE.md is
  explicitly marked "verify after Task 13 lands."

## Test suite
Ran `python -m pytest tests/ -q`: **276 passed, 1 failed**
(`tests/test_cli_tb.py::test_manager_command_reads_existing_manager_log` — `sqlite3.OperationalError:
table manager_log already exists`). This is a pre-existing schema-ordering conflict between Task 9's test
(which tries to `CREATE TABLE manager_log` itself) and Task 12's `TradeJournal` now creating that table
on init — it exists on the merged `d0c85bd` tip prior to any of this task's docs-only changes (verified:
no Python file was touched by this task; `git status --short` shows only new doc files). Left as-is per
scope (Task 17 is docs-only; this is a Task 9/12 test-ordering bug, not something this task should fix).

## Commit
`922052c` — "docs: VPS runbook, live-cutover checklist, CLAUDE.md refresh" on branch
`worktree-agent-a214956b55dab1b6b`.
