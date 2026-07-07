# Task 11 — AI scaffolding cleanup — Report

## Merge
`git merge 680e891 --no-edit` fast-forwarded cleanly (e5f680e..680e891), bringing in Tasks 1-8/10
(src/control, src/utils, ratchet_floor, new tests, etc.) before starting this task's work.

## What was done

### 1. Unwired AI analyst from `src/main.py`
- Removed imports: `from src.ai.analyst import AIAnalyst`, `ShadowTrader`, `ApprovalQueue` (was L37-39).
- Removed instance fields `self.analyst`, `self.shadow`, `self.approval_queue` from `__init__`.
- Removed `setup()` block instantiating `AIAnalyst`/`ApprovalQueue`/`ShadowTrader`.
- Removed the Telegram command-listener start call (`start_command_listener(approval_queue=..., shadow_trader=...)`).
- Removed the pre-session AI briefing block in `run()`.
- Removed the `_execute_approved_trades(...)` call site inside `_on_candle_complete` (the
  `current_price` local it depended on was otherwise unused — confirmed via grep before removing).
- Removed the entire "AI Analyst review (optional)" block between ML confidence check and
  `self.executor.execute_signal(...)` — the ML+risk approval path into `execute_signal` is
  untouched and now runs unconditionally as the single trade-execution path.
- Deleted the `_send_shadow_summary` and `_execute_approved_trades` methods in full.
- Removed the end-of-day shadow retrospective block and the "expire pending approvals" block
  from `_reconciliation_loop`; the ML-retrain-trigger check right after them is untouched.
- Verified `_execute_approved_trades`'s real-execution call (`self.executor.execute_signal`) was
  never reused elsewhere — the direct-execution path in the main `_on_candle_complete` flow (the
  one wired to ML+risk) is a separate call site and was left completely alone.

### 2. Telegram bot cleanup (`src/monitoring/telegram_bot.py`)
- Removed `claude_recommendation()`, `claude_shadow_result()`, `start_command_listener()`,
  `_command_poll_loop()`, `_poll_updates()`, `_handle_approve()`, `_handle_reject()`,
  `_handle_pending()`, `_handle_shadow()` — confirmed via grep that `/approve`, `/reject`,
  `/pending`, `/shadow` were the *only* commands the poll loop handled, so the whole listener
  shell was removed rather than partially gutted (per brief guidance).
- `threading` import retained — still used by the (unrelated) async-send helper thread.

### 3. Deleted `src/ai/*`
Removed `__init__.py`, `analyst.py`, `approval_queue.py`, `prompts.py`, `shadow_trader.py` via
`git rm -r src/ai`. Per the brief note, `approval_queue.py` had already been harvested by Task 8
(now lives as `src/control/queue.py`), so all five files were deleted outright with no further
harvesting needed.

### 4. Docs
`git mv TRADING_BRAIN.md docs/personas/trading-brain.md`.

### 5. Config
Removed the entire `ai_analyst:` section from `config/settings.yaml` (enabled, api_key_env,
model, regime_model, require_approval, min_ml_confidence_for_review,
regime_check_interval_minutes, session_briefing_enabled, session_review_enabled, shadow_enabled,
shadow_review_hour_utc, approval_ttl_seconds, approval_max_slippage_pips).

### 6. CLAUDE.md — NOT updated, with justification
`CLAUDE.md` at the repo root is gitignored/untracked (see `.gitignore`: `CLAUDE.md` and `.claude/`
entries, and commit e5f680e "chore: remove Claude config files from tracking"). It does not exist
inside this worktree's checkout at all (worktrees don't inherit untracked files), and the tool
sandbox explicitly refused an Edit targeting the shared-checkout path outside this worktree
("Edit the worktree copy of this file instead of the shared-checkout path"). Since the file isn't
part of this worktree's git state, any edit here could not be committed on this branch anyway.
Separately: I read the actual root CLAUDE.md content and it turns out it never documented
`src/ai` in its architecture tree in the first place (no `src/ai`, `AIAnalyst`, `ShadowTrader`,
`ai_analyst`, or `TRADING_BRAIN` references at all) — so there is no stale AI-analyst reference to
strip from it. It's also missing `src/control`, `src/utils`, and any `cli/`/`src/manager`
placeholder note (pre-existing gap from Tasks 8/10, not something this task introduced or made
worse). Recommend a follow-up (outside this worktree, direct edit or a small PR-less doc pass by
the user) to add those sections since I cannot commit to that file from here.

## Verification
- `python -c "import src.main"` — succeeds, no errors.
- `python -m pytest tests/ -x -q` — **190 passed**, no failures, no skips of ai-related tests
  needed (none referenced `src.ai`/AI-analyst symbols to begin with).
- `grep -rn "src\.ai\|ai_analyst\|AIAnalyst\|ShadowTrader\|shadow_trader\|approval_queue" src/ tests/ config/`
  → zero hits (grep exit code 1). `cli/` doesn't exist yet in this repo so it was omitted from the
  scoped grep (nothing to search).
- Full-repo grep (all non-`.md` files) for the same patterns → zero hits.

## Commits (on worktree branch `worktree-agent-a9ebbad9890e5a44c`)
1. `4096984` — `refactor: unwire AI analyst from main loop` (main.py + telegram_bot.py unwiring,
   plus the src/ai deletion and TRADING_BRAIN.md move that were already staged going into this
   commit).
2. `694d767` — `chore: remove ai_analyst config section` (settings.yaml).

## Concerns / follow-ups
- CLAUDE.md architecture-tree update could not be performed from within this isolated worktree
  (see above) — needs a direct edit against the shared checkout, done by the user or a
  non-worktree-isolated session.
- Trading behavior: confirmed unchanged — the ML+risk approval flow in `_on_candle_complete` and
  the direct `executor.execute_signal` call for normal signals were not touched; only the AI
  Analyst overlay (review/gate/size-reduction) and the separate Claude-approved-trade execution
  path were removed.
