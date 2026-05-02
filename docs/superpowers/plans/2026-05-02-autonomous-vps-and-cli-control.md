# Autonomous VPS Deployment + Claude-Code CLI Control — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take TraderBot from backtest-only to running unattended on a Hetzner Windows VPS for one month against an Exness MT5 demo account, controllable on demand from the developer's local Claude Code session via a `tb` CLI over SSH.

**Architecture:** Single-instance bot launched by NSSM as a Windows service; polls a file-based control queue every loop iteration; emits Telegram heartbeats; surfaces read+write CLI commands over SSH. Hardened across crash recovery, MT5 disconnect handling, drawdown enforcement, session-boundary resets, trade ID integrity, and audit-logged tunable parameters.

**Tech Stack:** Python 3.11 · MetaTrader5 · XGBoost · pandas · SQLite (existing `trades.db`) · Telegram Bot API · NSSM (Windows service wrapper) · pytest · OpenSSH (Windows built-in).

**Spec reference:** `docs/superpowers/specs/2026-05-02-autonomous-vps-and-cli-control-design.md`

**One spec deviation:** the spec said "repurpose `src/ai/approval_queue.py` as the control queue." After reading the file, that's impractical — it is SQLite-backed with a trade-recommendation schema (`instrument`, `direction`, `entry_price`, `confidence`, `reasoning`) and `approve/reject` semantics, whereas the spec design specifies a **file-based** control queue with a generic `verb/args/reason/requested_by` shape. Refactoring would be more work than rewriting. **Decision: `approval_queue.py` is deleted with the rest of `src/ai/`; `src/control/queue.py` is written fresh (~80 lines).**

---

## File Structure

### New files
```
config/
  safety_floor.yaml                  # Hard floors that override settings.yaml + tunes
src/control/
  __init__.py
  queue.py                           # File-based command queue (inbox/outbox)
  effective_config.py                # Loads settings.yaml + tunes + safety_floor (overlay model)
  audit_log.py                       # Wrapper around control_log SQLite table
  single_instance.py                 # Cross-platform single-instance lock (TCP bind)
  schedulers.py                      # Internal timers: daily summary, session-boundary resets
cli/
  __init__.py
  tb.py                              # `tb` command-line entry point
docs/personas/
  trading-brain.md                   # Moved from /TRADING_BRAIN.md
docs/runbooks/
  vps-provisioning.md                # Hetzner + MT5 + NSSM + SSH setup
  shakedown-checklist.md             # 48-hour pre-month dry run
tests/
  test_control_queue.py
  test_single_instance.py
  test_safety_floor.py
  test_session_boundary.py
  test_drawdown_breach_close.py
  test_trade_id_no_fallback.py
  test_audit_log.py
  test_effective_config.py
  test_tb_cli_smoke.py
data/
  control/
    inbox/                           # *.cmd.json (gitignored)
    outbox/                          # *.result.json (gitignored)
    effective_config.json            # Tunes overlay (gitignored)
    traderbot.lock                   # Single-instance TCP-port marker (gitignored)
```

### Modified files
```
config/settings.yaml                 # currency → ZAR, hard_floor → hard_floor_zar, risk_per_trade_pct, session-boundary key
src/main.py                          # Remove src.ai imports; add lock check; wire control queue + schedulers; remove AI briefing/review/shadow paths
src/risk/circuit_breaker.py          # Daily reset of consecutive losses; spread blowout pause; better hard_floor key
src/risk/drawdown_tracker.py         # Session-boundary reset (21:00 UTC); breach now signals close-all
src/risk/manager.py                  # Wire drawdown breach → executor.close_all
src/data/mt5_client.py               # Broker-disconnect health probe; backoff on stream; spread-aware deviation; suffix re-detection on reconnect; transient retcode retry
src/data/collector.py                # Pause trading on stream errors; alert on disconnect/reconnect
src/execution/executor.py            # Trade ID must be MT5 ticket; fail loudly on missing; better fill validation
src/monitoring/telegram_bot.py       # Add: connection_lost, connection_restored, control_event, exception_alert; remove claude_recommendation, claude_shadow_result; add command listener for control responses (optional Phase-4 read-only); daily_summary triggered by scheduler
src/monitoring/trade_journal.py      # Add control_log table; add fee/swap columns to trades table (with migration)
.gitignore                           # data/control/, data/traderbot.lock
requirements.txt                     # Add psutil (for cross-platform process check)
CLAUDE.md                            # Update file inventory; remove src/ai/; add src/control/, cli/
```

### Deleted files
```
src/ai/__init__.py
src/ai/analyst.py
src/ai/approval_queue.py
src/ai/prompts.py
src/ai/shadow_trader.py
TRADING_BRAIN.md   (moved, not deleted — see docs/personas/trading-brain.md)
```

---

# Phase 0 — Foundation (settings, safety floor, currency)

These changes touch nothing dangerous and unblock everything downstream. Do them first.

### Task 0.1: Settings — currency, hard floor, risk %

**Files:**
- Modify: `config/settings.yaml:5-19, 41-50`

- [ ] **Step 1: Patch `config/settings.yaml` account section**

Replace the `account:` block (lines 5-9) with:

```yaml
# --- Account ---
account:
  starting_balance_zar: 500          # R500 starting demo balance
  currency: "ZAR"                    # All monetary values throughout the bot are ZAR
  hard_floor_zar: 9000               # Emergency shutdown if balance drops here (ZAR)
```

- [ ] **Step 2: Patch `risk_per_trade_pct` in the risk block**

Change line 43 from `risk_per_trade_pct: 2.5` to:

```yaml
  risk_per_trade_pct: 1.5            # Matches CLAUDE.md design baseline
```

- [ ] **Step 3: Add session-boundary key to risk block**

Insert at the end of the `risk:` block (around line 68):

```yaml
  # Session boundary — daily resets fire at this UTC hour (NYSE close)
  session_boundary_hour_utc: 21
```

- [ ] **Step 4: Verify config loads**

Run: `python -c "from src.config import load_config; c = load_config(); print(c.get('account.hard_floor_zar'), c.get('risk.risk_per_trade_pct'), c.get('risk.session_boundary_hour_utc'))"`
Expected: `9000 1.5 21`

- [ ] **Step 5: Commit**

```bash
git add config/settings.yaml
git commit -m "config: switch account to ZAR; align risk %; add session boundary key"
```

---

### Task 0.2: Safety floor file (un-overridable hard limits)

**Files:**
- Create: `config/safety_floor.yaml`
- Create: `src/control/__init__.py`
- Create: `src/control/effective_config.py`
- Test: `tests/test_safety_floor.py`

- [ ] **Step 1: Write `config/safety_floor.yaml`**

```yaml
# ============================================================
# Safety Floor — values that always win over settings.yaml and
# any runtime tunes. The CLI rejects any tune that targets a
# key listed here. If you need to change one of these, you must
# do it by hand and restart the bot.
# ============================================================
risk:
  daily_drawdown_stop_pct: 4.0        # Hard stop the day at -4%
  hard_floor_zar: 9000                # Kill switch if equity drops here (ZAR)
  max_leverage_effective: 5.0         # Cap effective leverage
circuit_breaker:
  api_error_threshold: 10             # Pause if more than N API errors in 10 min
  spread_blowout_pause_minutes: 30    # Pause an instrument for N min after blowout
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_safety_floor.py`:

```python
from src.control.effective_config import EffectiveConfig

def test_safety_floor_overrides_settings():
    eff = EffectiveConfig.load()
    # Even if someone edits settings.yaml to weaken the floor, safety_floor wins
    assert eff.get("risk.hard_floor_zar") == 9000
    assert eff.get("risk.daily_drawdown_stop_pct") == 4.0

def test_is_safety_locked_rejects_floor_keys():
    eff = EffectiveConfig.load()
    assert eff.is_safety_locked("risk.hard_floor_zar") is True
    assert eff.is_safety_locked("risk.daily_drawdown_stop_pct") is True
    assert eff.is_safety_locked("risk.risk_per_trade_pct") is False  # tunable
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_safety_floor.py -v`
Expected: ImportError or ModuleNotFoundError on `src.control.effective_config`.

- [ ] **Step 4: Implement `src/control/__init__.py` (empty) and `src/control/effective_config.py`**

`src/control/__init__.py`:
```python
"""Control plane: command queue, audit log, scheduling, single-instance lock."""
```

`src/control/effective_config.py`:
```python
"""
Effective config = settings.yaml ⊕ tunes overlay ⊕ safety_floor.yaml (last wins).

The CLI tune command writes to data/control/effective_config.json. The bot
reads this overlay on every config access. safety_floor.yaml is loaded last
and always wins; the CLI refuses to write tunes that target safety-floor keys.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from src.config import PROJECT_ROOT

logger = logging.getLogger("traderbot.control.config")

SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"
SAFETY_FLOOR_PATH = PROJECT_ROOT / "config" / "safety_floor.yaml"
TUNES_PATH = PROJECT_ROOT / "data" / "control" / "effective_config.json"


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursive dict merge — overlay wins on leaf keys."""
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class EffectiveConfig:
    def __init__(self, data: dict, safety_keys: set[str]):
        self._data = data
        self._safety_keys = safety_keys

    @classmethod
    def load(cls) -> "EffectiveConfig":
        settings = _load_yaml(SETTINGS_PATH)
        floor = _load_yaml(SAFETY_FLOOR_PATH)

        tunes: dict = {}
        if TUNES_PATH.exists():
            try:
                tunes = json.loads(TUNES_PATH.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Failed to read tunes overlay {TUNES_PATH}: {e}")

        merged = _deep_merge(_deep_merge(settings, tunes), floor)
        safety_keys = cls._flat_keys(floor)
        return cls(merged, safety_keys)

    @staticmethod
    def _flat_keys(d: dict, prefix: str = "") -> set[str]:
        keys = set()
        for k, v in d.items():
            full = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                keys |= EffectiveConfig._flat_keys(v, full)
            else:
                keys.add(full)
        return keys

    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def is_safety_locked(self, dotted_key: str) -> bool:
        return dotted_key in self._safety_keys

    def safety_keys(self) -> set[str]:
        return set(self._safety_keys)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_safety_floor.py -v`
Expected: PASS (both tests).

- [ ] **Step 6: Commit**

```bash
git add config/safety_floor.yaml src/control/__init__.py src/control/effective_config.py tests/test_safety_floor.py
git commit -m "feat(control): safety floor + effective config overlay"
```

---

# Phase 1 — AI Scaffolding Cleanup (do early to de-risk everything else)

The `src/ai/*` files are wired into `main.py` in 5 places. Removing them surgically before doing the bigger hardening work means later edits to `main.py` happen in a clean file, not one full of legacy AI plumbing.

### Task 1.1: Move TRADING_BRAIN.md to docs/personas

**Files:**
- Move: `TRADING_BRAIN.md` → `docs/personas/trading-brain.md`

- [ ] **Step 1: Create directory and git-mv**

```bash
mkdir -p docs/personas
git mv TRADING_BRAIN.md docs/personas/trading-brain.md
```

- [ ] **Step 2: Verify path**

Run: `ls docs/personas/trading-brain.md`
Expected: file exists.

- [ ] **Step 3: Commit**

```bash
git commit -m "docs: move TRADING_BRAIN.md to docs/personas/trading-brain.md"
```

---

### Task 1.2: Remove `src/ai` imports and call sites from main.py

**Files:**
- Modify: `src/main.py:33-35` (imports), `60-62` (instance vars), `92-94` (init), `164-191` (telegram listener + AI briefing), `362-406` (AI review path), `841-857` (shadow + approval_queue in reconciliation loop)

- [ ] **Step 1: Remove imports at top of `src/main.py`**

Delete lines 33-35:

```python
from src.ai.analyst import AIAnalyst
from src.ai.shadow_trader import ShadowTrader
from src.ai.approval_queue import ApprovalQueue
```

- [ ] **Step 2: Remove instance vars in `__init__`**

Delete lines 60-62:

```python
        self.analyst = None
        self.shadow = None
        self.approval_queue = None
```

- [ ] **Step 3: Remove init in `setup()`**

Delete lines 91-94 (the AI Analyst comment and three `self.analyst/approval_queue/shadow = …` assignments).

- [ ] **Step 4: Remove the telegram approval-queue listener**

Delete lines 164-169 (the `if self.telegram and self.approval_queue:` block calling `start_command_listener`).

- [ ] **Step 5: Remove the AI session-briefing block**

Delete lines 171-191 (`# AI Analyst: pre-session briefing` through the `except Exception as e: logger.debug(...)`).

- [ ] **Step 6: Remove the AI review block in `_on_candle_complete`**

Delete lines 362-406 (the `# --- AI Analyst review (optional) ---` block through its trailing `except`).

- [ ] **Step 7: Remove shadow + approval_queue lines from `_reconciliation_loop`**

In the reconciliation loop (around lines 838-857), delete:

```python
                # End-of-day shadow retrospective
                from datetime import datetime as dt_cls, timezone as tz_cls
                hour_utc = dt_cls.now(tz_cls.utc).hour
                if self.shadow and self.shadow.should_run(hour_utc):
                    try:
                        logger.info("Running end-of-day shadow retrospective...")
                        result = self.shadow.run_retrospective(...)
                        if result and self.telegram:
                            self._send_shadow_summary(result)
                    except Exception as e:
                        logger.error(f"Shadow retrospective failed: {e}", exc_info=True)

                # Expire pending approvals
                if self.approval_queue:
                    self.approval_queue.expire_old()
```

- [ ] **Step 8: Search for any remaining `_send_shadow_summary` reference**

Run: `grep -n "_send_shadow_summary\|approval_queue\|self\.shadow\|self\.analyst" src/main.py`
Expected: 0 results. If any remain, delete the surrounding block.

- [ ] **Step 9: Verify the bot still imports cleanly**

Run: `python -c "from src.main import TraderBot"`
Expected: no error.

- [ ] **Step 10: Commit**

```bash
git add src/main.py
git commit -m "refactor(main): unwire src/ai analyst, shadow trader, approval queue"
```

---

### Task 1.3: Delete `src/ai/`

**Files:**
- Delete: `src/ai/__init__.py`, `src/ai/analyst.py`, `src/ai/approval_queue.py`, `src/ai/prompts.py`, `src/ai/shadow_trader.py`, `src/ai/gold_features.py` (if any AI-flavored, not the ml/gold_features.py)

- [ ] **Step 1: Sanity check — confirm no remaining importers**

Run: `grep -rn "from src.ai\|import src.ai" src/ tests/ backtest/ cli/ 2>/dev/null`
Expected: 0 results.

- [ ] **Step 2: Remove the directory**

```bash
git rm -r src/ai/
```

- [ ] **Step 3: Remove ai_analyst block from settings.yaml**

Delete lines 143-158 of `config/settings.yaml` (the entire `# --- AI Analyst (Claude) ---` section through `approval_max_slippage_pips: 5`).

- [ ] **Step 4: Verify the bot still starts (dry-run setup only, no MT5)**

Run: `python -c "from src.main import TraderBot; b = TraderBot()"` (skip `.setup()` since it needs MT5 creds — just instantiation).
Expected: no error.

- [ ] **Step 5: Commit**

```bash
git add src/ai config/settings.yaml
git commit -m "chore: remove src/ai scaffolding (analyst, shadow, approval queue, prompts)"
```

---

### Task 1.4: Update CLAUDE.md to reflect the new module layout

**Files:**
- Modify: `CLAUDE.md` — Architecture section (~line 38)

- [ ] **Step 1: Replace the architecture file tree**

In `CLAUDE.md`, the architecture tree currently lists 7 modules including `src/ai/`. Replace the `src/` tree (in the `### Architecture` section) with a tree that:
- Removes any line referencing `src/ai/`
- Adds new `src/control/` lines (`queue.py`, `effective_config.py`, `audit_log.py`, `single_instance.py`, `schedulers.py`)
- Adds a new top-level `cli/` section with `tb.py`

(The implementing agent should read the current tree and apply minimum-diff edits.)

- [ ] **Step 2: Remove the `### Build Status` reference to AI scaffolding** if any.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(CLAUDE): update module inventory after AI cleanup + control plane intro"
```

---

# Phase 2 — Hardening Blockers

Each task addresses one BLOCKER from the audit. Order is dependency-driven (foundation → I/O → risk → execution → monitoring).

### Task 2.1: Single-instance lock (Blocker A)

**Files:**
- Create: `src/control/single_instance.py`
- Test: `tests/test_single_instance.py`
- Modify: `src/main.py:64-69` (top of `setup()`)
- Modify: `requirements.txt`
- Modify: `.gitignore`

We use a TCP-port bind to a local-only port. Cross-platform, automatic cleanup on process death, no need for psutil or PID files.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_single_instance.py
import pytest
from src.control.single_instance import SingleInstanceLock, AlreadyRunning

def test_acquire_then_release_allows_second_acquire():
    lock = SingleInstanceLock(port=54017)
    lock.acquire()
    lock.release()
    lock2 = SingleInstanceLock(port=54017)
    lock2.acquire()
    lock2.release()

def test_second_acquire_while_first_held_raises():
    a = SingleInstanceLock(port=54018)
    a.acquire()
    try:
        b = SingleInstanceLock(port=54018)
        with pytest.raises(AlreadyRunning):
            b.acquire()
    finally:
        a.release()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_single_instance.py -v`
Expected: ImportError on `src.control.single_instance`.

- [ ] **Step 3: Implement `src/control/single_instance.py`**

```python
"""Single-instance lock via local TCP port bind. Cross-platform, auto-cleanup."""

from __future__ import annotations

import socket
import logging

logger = logging.getLogger("traderbot.control.lock")

DEFAULT_PORT = 54017  # arbitrary local port; not exposed externally


class AlreadyRunning(RuntimeError):
    pass


class SingleInstanceLock:
    def __init__(self, port: int = DEFAULT_PORT):
        self._port = port
        self._sock: socket.socket | None = None

    def acquire(self) -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        try:
            s.bind(("127.0.0.1", self._port))
        except OSError as e:
            s.close()
            raise AlreadyRunning(
                f"Another TraderBot instance is already running (port {self._port} in use)"
            ) from e
        s.listen(1)
        self._sock = s
        logger.info(f"Single-instance lock acquired on port {self._port}")

    def release(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_single_instance.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Wire into `setup()` in `src/main.py`**

At the top of `setup()` (right after `logger.info("Initializing TraderBot...")` on line 66), add:

```python
        from src.control.single_instance import SingleInstanceLock, AlreadyRunning
        self._instance_lock = SingleInstanceLock()
        try:
            self._instance_lock.acquire()
        except AlreadyRunning as e:
            logger.error(str(e))
            sys.exit(2)
```

And in `__init__`, add to the instance-var block:

```python
        self._instance_lock = None
```

And in `shutdown()` (around line 893, before `self.running = False`), add release:

```python
        if self._instance_lock:
            self._instance_lock.release()
```

- [ ] **Step 6: Add `data/traderbot.lock` placeholder to `.gitignore`**

Add to `.gitignore`:
```
data/control/
data/traderbot.lock
```

- [ ] **Step 7: Commit**

```bash
git add src/control/single_instance.py tests/test_single_instance.py src/main.py .gitignore
git commit -m "feat(control): single-instance lock via local TCP port bind (blocker A)"
```

---

### Task 2.2: MT5 broker-disconnect detection + backoff + Telegram alerts (Blockers B1, B2, B3, I)

**Files:**
- Modify: `src/data/mt5_client.py:128, 384-410`
- Modify: `src/data/collector.py:150-157`
- Modify: `src/monitoring/telegram_bot.py` (add `connection_lost`, `connection_restored`)

This is the largest single hardening change. It groups B1+B2+B3+I because they are one logical fix: detect broker connection loss → pause new trades → backoff retry → alert on both edges.

- [ ] **Step 1: Add Telegram connection alerts**

In `src/monitoring/telegram_bot.py`, before the `# Reports` section divider, add:

```python
    def connection_lost(self, detail: str = ""):
        text = (
            f"⚠️ <b>MT5 connection lost</b>\n"
            f"{detail}\n"
            f"Bot will pause new entries and retry with backoff."
        )
        self._send(text)

    def connection_restored(self, downtime_seconds: float):
        text = (
            f"✅ <b>MT5 connection restored</b>\n"
            f"Downtime: {downtime_seconds:.0f}s\n"
            f"Bot resuming normal operation."
        )
        self._send(text)
```

- [ ] **Step 2: Add `is_broker_connected()` health probe to `mt5_client.py`**

Read `src/data/mt5_client.py` around line 100-150 to find the existing `connect()` method, then add (near the connection methods):

```python
    def is_broker_connected(self) -> bool:
        """
        True iff (a) the MT5 terminal is initialized AND
                 (b) the terminal is currently connected to the broker.
        Returns False if either is false. Cheap to call (microseconds).
        """
        try:
            import MetaTrader5 as mt5
            ti = mt5.terminal_info()
            if ti is None:
                return False
            return bool(ti.connected)
        except Exception:
            return False
```

- [ ] **Step 3: Refactor `stream_prices()` to add backoff + connection check**

In `mt5_client.py:384-410` (the `stream_prices()` function), the inner loop must:
- Check `is_broker_connected()` periodically (every iteration is fine — the call is cheap).
- On disconnect: yield a `{"type": "disconnected"}` sentinel and pause the loop.
- On consecutive empty-tick failures: exponential backoff (start 0.1s, double up to 30s, then sustain).
- On reconnect: yield `{"type": "reconnected", "downtime_s": <seconds>}`.

The actual signature is currently a generator yielding tick dicts. Augment it to yield two new sentinel dict types (`{"type": "disconnected"}`, `{"type": "reconnected", "downtime_s": float}`) intermixed with normal ticks. The collector consumes these and reacts.

Key code shape:

```python
    def stream_prices(self, instrument: str):
        import MetaTrader5 as mt5
        symbol = self._to_mt5_symbol(instrument)
        consecutive_empty = 0
        last_state_connected = True
        disconnect_started_at: float | None = None
        backoff = 0.1  # seconds

        while True:
            if not self.is_broker_connected():
                if last_state_connected:
                    last_state_connected = False
                    disconnect_started_at = time.monotonic()
                    yield {"type": "disconnected", "instrument": instrument}
                time.sleep(min(backoff, 30.0))
                backoff = min(backoff * 2.0, 30.0)
                continue

            if not last_state_connected:
                downtime = time.monotonic() - (disconnect_started_at or time.monotonic())
                last_state_connected = True
                backoff = 0.1
                # Re-detect symbol suffix in case broker mapping changed (Fix L)
                self._suffix_cache.pop(instrument, None)
                symbol = self._to_mt5_symbol(instrument)
                yield {"type": "reconnected", "instrument": instrument, "downtime_s": downtime}

            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                consecutive_empty += 1
                if consecutive_empty > 50:
                    time.sleep(0.2)  # cool off; broker may be slow
                else:
                    time.sleep(self._tick_poll_interval)
                continue

            consecutive_empty = 0
            yield {
                "type": "tick",
                "instrument": instrument,
                "bid": tick.bid,
                "ask": tick.ask,
                "time": tick.time,
            }
            time.sleep(self._tick_poll_interval)
```

(Implementer: read the existing `stream_prices()` and adapt — the existing yield shape may already be a dict; if not, wrap.)

- [ ] **Step 4: Update `collector.py:150-157` to react to sentinels**

Find the consumer loop in `src/data/collector.py` that iterates `mt5_client.stream_prices(...)`. Modify the per-tick handler to branch on `tick["type"]`:

```python
        for tick in self.client.stream_prices(instrument):
            ttype = tick.get("type", "tick")
            if ttype == "disconnected":
                self._on_disconnect(instrument)
                continue
            if ttype == "reconnected":
                self._on_reconnect(instrument, tick.get("downtime_s", 0.0))
                continue
            # Normal tick path:
            self._handle_tick(instrument, tick)
```

Add the two helper methods on `DataCollector`:

```python
    def _on_disconnect(self, instrument: str):
        if self._disconnected_instruments is None:
            self._disconnected_instruments = set()
        was_empty = len(self._disconnected_instruments) == 0
        self._disconnected_instruments.add(instrument)
        if was_empty:
            logger.warning("MT5 broker disconnect detected — pausing new entries")
            self.trading_paused_due_to_disconnect = True
            if self.telegram:
                self.telegram.connection_lost(f"Instrument {instrument} stream lost")

    def _on_reconnect(self, instrument: str, downtime_s: float):
        if self._disconnected_instruments and instrument in self._disconnected_instruments:
            self._disconnected_instruments.remove(instrument)
        if not self._disconnected_instruments:
            self.trading_paused_due_to_disconnect = False
            logger.info(f"MT5 broker reconnected after {downtime_s:.1f}s")
            if self.telegram:
                self.telegram.connection_restored(downtime_s)
```

(Implementer: `self.telegram` may not currently be passed in — check `DataCollector.__init__` and add it as a constructor arg, then pass it from `main.py`'s setup.)

- [ ] **Step 5: Wire the trading-paused flag into the entry path**

In `src/main.py:_on_candle_complete()` (or wherever entries are decided — search for `executor.execute_signal`), add a guard at the top of the entry decision block:

```python
        if getattr(self.collector, "trading_paused_due_to_disconnect", False):
            logger.debug(f"Skipping entry on {instrument}: broker disconnected")
            return
```

- [ ] **Step 6: Manual smoke test**

Run: `python -m src.main` — start the bot, then in a separate terminal kill the MT5 terminal process. Within ~5s you should see "MT5 broker disconnect detected" in logs and a Telegram alert. Restart MT5 terminal; within ~5s you should see "MT5 broker reconnected" + Telegram alert.

If MT5 isn't installed locally, this can be deferred to the VPS shakedown (Phase 6).

- [ ] **Step 7: Commit**

```bash
git add src/data/mt5_client.py src/data/collector.py src/monitoring/telegram_bot.py src/main.py
git commit -m "feat(mt5): broker disconnect detection, backoff, telegram alerts (blockers B1-B3, I)"
```

---

### Task 2.3: Hard-floor key alignment (Blocker C, completes Foundation)

**Files:**
- Modify: `src/risk/circuit_breaker.py:45`
- Modify: `src/risk/drawdown_tracker.py:33`

Both modules already read `account.hard_floor_zar` from config (with defaults of `350` — wrong unit). After Task 0.1 the settings file uses `account.hard_floor_zar: 9000`, so the keys now match. Need to update default values from `350` → `9000` so the field is correct even if the key is missing.

- [ ] **Step 1: Update default in `circuit_breaker.py`**

Line 45, change:
```python
        self.hard_floor = config.get("account.hard_floor_zar", 350)
```
to:
```python
        self.hard_floor = config.get("account.hard_floor_zar", 9000)
```

- [ ] **Step 2: Update default in `drawdown_tracker.py`**

Line 33, change:
```python
        self.hard_floor = config.get("account.hard_floor_zar", 350)
```
to:
```python
        self.hard_floor = config.get("account.hard_floor_zar", 9000)
```

- [ ] **Step 3: Quick verification**

Run: `python -c "from src.config import load_config; from src.risk.circuit_breaker import CircuitBreaker; cb = CircuitBreaker(load_config()); print('hard_floor =', cb.hard_floor)"`
Expected: `hard_floor = 9000`

- [ ] **Step 4: Commit**

```bash
git add src/risk/circuit_breaker.py src/risk/drawdown_tracker.py
git commit -m "fix(risk): align hard_floor_zar default to R9000 (blocker C)"
```

---

### Task 2.4: Drawdown breach must close all positions (Blocker D)

**Files:**
- Modify: `src/risk/manager.py` (add `daily_drawdown_breached()` signal)
- Modify: `src/risk/drawdown_tracker.py` (expose breach state)
- Modify: `src/main.py:217-227` (the existing emergency-shutdown handler currently only handles hard floor — extend to handle DD breach)
- Test: `tests/test_drawdown_breach_close.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_drawdown_breach_close.py
from src.risk.drawdown_tracker import DrawdownTracker
from src.config import load_config

def test_breach_signals_close_all():
    cfg = load_config()
    dd = DrawdownTracker(cfg)
    dd.initialize(10000)
    # Simulate -5% intraday loss (above 4% limit)
    res = dd.check(current_balance=9500)
    assert res["allowed"] is False
    assert dd.is_breached() is True

def test_no_breach_below_limit():
    cfg = load_config()
    dd = DrawdownTracker(cfg)
    dd.initialize(10000)
    res = dd.check(current_balance=9700)  # -3%
    assert res["allowed"] is True
    assert dd.is_breached() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_drawdown_breach_close.py -v`
Expected: AttributeError on `dd.is_breached()`.

- [ ] **Step 3: Add `is_breached()` to `DrawdownTracker`**

In `src/risk/drawdown_tracker.py`, add an instance var `self._breached = False` to `__init__` and `initialize()`. In the `check()` method, set `self._breached = True` when `daily_dd >= self.daily_limit`. Add:

```python
    def is_breached(self) -> bool:
        return self._breached

    def clear_breach(self):
        """Called on session boundary reset."""
        self._breached = False
```

- [ ] **Step 4: Wire into `RiskManager`**

In `src/risk/manager.py`, find the `close_all_signal()` method (already exists, returns True for hard-floor breach). Extend it:

```python
    def close_all_signal(self) -> tuple[bool, str]:
        """Returns (should_close_all, reason)."""
        if self.circuit_breaker.is_shutdown:
            return True, f"hard floor: {self.circuit_breaker.shutdown_reason}"
        if self.drawdown_tracker.is_breached():
            return True, "daily drawdown breached"
        return False, ""
```

(If the existing signature returns `bool`, update callers in `main.py` to unpack.)

- [ ] **Step 5: Update `main.py:218-227` to handle the new return shape**

```python
                # Check for emergency shutdown
                should_close, reason = self.risk_manager.close_all_signal()
                if should_close:
                    logger.critical(f"Emergency close-all signal: {reason}")
                    results = self.executor.close_all(f"emergency:{reason}")
                    try:
                        balance = self.client.get_account_balance()
                        self.telegram.emergency_stop(balance, reason)
                    except Exception:
                        pass
                    if reason.startswith("hard floor"):
                        self.running = False
                        break
                    # Drawdown breach: close all but stay alive — entries are blocked
                    # by drawdown_tracker.is_breached() until session boundary reset.
```

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_drawdown_breach_close.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/risk/drawdown_tracker.py src/risk/manager.py src/main.py tests/test_drawdown_breach_close.py
git commit -m "feat(risk): drawdown breach now closes all positions (blocker D)"
```

---

### Task 2.5: Session-boundary reset (21:00 UTC) for daily DD + consecutive losses (Blockers E, F)

**Files:**
- Create: `src/control/schedulers.py`
- Test: `tests/test_session_boundary.py`
- Modify: `src/risk/drawdown_tracker.py:156-171` (replace UTC-midnight reset with session-boundary aware reset)
- Modify: `src/risk/circuit_breaker.py` (add `reset_session_state()`)
- Modify: `src/main.py` (start scheduler thread)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_session_boundary.py
from datetime import datetime, timezone
from src.control.schedulers import seconds_until_next_session_boundary

def test_boundary_at_21_utc():
    # 20:30 UTC -> 30 min away
    now = datetime(2026, 5, 4, 20, 30, tzinfo=timezone.utc)
    assert seconds_until_next_session_boundary(now, hour_utc=21) == 30 * 60

def test_boundary_after_21_utc():
    # 21:30 UTC -> 23h30m to next 21:00 UTC
    now = datetime(2026, 5, 4, 21, 30, tzinfo=timezone.utc)
    assert seconds_until_next_session_boundary(now, hour_utc=21) == 23 * 3600 + 30 * 60
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_session_boundary.py -v`
Expected: ImportError on `src.control.schedulers`.

- [ ] **Step 3: Implement `src/control/schedulers.py`**

```python
"""Internal scheduling: session-boundary resets, daily summary trigger."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("traderbot.control.schedulers")


def seconds_until_next_session_boundary(now: datetime, hour_utc: int) -> int:
    """Seconds from `now` to the next occurrence of `hour_utc:00 UTC`."""
    target = now.replace(hour=hour_utc, minute=0, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return int((target - now).total_seconds())


class SessionBoundaryScheduler:
    """Fires `on_boundary` callback once per day at the configured UTC hour."""

    def __init__(self, hour_utc: int, on_boundary, name: str = "session-boundary"):
        self._hour_utc = hour_utc
        self._on_boundary = on_boundary
        self._name = name
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name=self._name)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            now = datetime.now(timezone.utc)
            wait_s = seconds_until_next_session_boundary(now, self._hour_utc)
            logger.info(f"{self._name}: next fire in {wait_s}s ({wait_s/3600:.1f}h)")
            if self._stop.wait(wait_s):
                return
            try:
                self._on_boundary()
            except Exception as e:
                logger.error(f"{self._name} callback error: {e}", exc_info=True)
```

- [ ] **Step 4: Run scheduler test**

Run: `python -m pytest tests/test_session_boundary.py -v`
Expected: PASS (the two pure-function tests).

- [ ] **Step 5: Replace `_handle_new_day` triggering in `drawdown_tracker.py`**

In `src/risk/drawdown_tracker.py:71-73`, the `update()` method currently calls `_handle_new_day` when `now.date() != self.current_date.date()`. Change this to only fire externally — remove the auto-trigger from `update()`:

Delete (or comment out) lines 71-73:
```python
        # Check for new day
        if self.current_date is None or now.date() != self.current_date.date():
            self._handle_new_day(current_balance, now)
```

Add a public method:
```python
    def session_boundary_reset(self, current_balance: float):
        """Called by SessionBoundaryScheduler at 21:00 UTC."""
        self._handle_new_day(current_balance, datetime.now(timezone.utc))
        self.clear_breach()
```

- [ ] **Step 6: Add `reset_session_state()` to `circuit_breaker.py`**

In `src/risk/circuit_breaker.py`, add:

```python
    def reset_session_state(self):
        """Called at session boundary (21:00 UTC). Resets per-day counters
        but does NOT clear hard shutdown state."""
        if self.is_shutdown:
            logger.warning("Session reset skipped: circuit breaker is in SHUTDOWN state")
            return
        old = self.consecutive_losses
        self.consecutive_losses = 0
        self.api_errors.clear()
        # Note: recent_outcomes (rolling 100-trade win rate) is intentionally NOT reset
        logger.info(f"Session boundary reset: cleared {old} consecutive losses, API error log")
```

- [ ] **Step 7: Wire scheduler into `main.py`**

In `setup()` (after risk_manager init), add:

```python
        from src.control.schedulers import SessionBoundaryScheduler
        boundary_hour = self.config.get("risk.session_boundary_hour_utc", 21)
        self._session_scheduler = SessionBoundaryScheduler(
            hour_utc=boundary_hour,
            on_boundary=self._on_session_boundary,
        )
```

In `__init__`, add `self._session_scheduler = None`.

In `run()` (after streaming starts, around line 162), add:

```python
        self._session_scheduler.start()
```

In `shutdown()` (around line 894):

```python
        if self._session_scheduler:
            self._session_scheduler.stop()
```

Add the callback method on `TraderBot`:

```python
    def _on_session_boundary(self):
        try:
            balance = self.client.get_account_balance()
        except Exception:
            balance = 0.0
        logger.info("Session boundary fired (21:00 UTC). Resetting daily counters.")
        self.risk_manager.drawdown_tracker.session_boundary_reset(balance)
        self.risk_manager.circuit_breaker.reset_session_state()
```

- [ ] **Step 8: Commit**

```bash
git add src/control/schedulers.py tests/test_session_boundary.py src/risk/drawdown_tracker.py src/risk/circuit_breaker.py src/main.py
git commit -m "feat(risk): session-boundary reset at 21:00 UTC for DD + losses (blockers E, F)"
```

---

### Task 2.6: Trade ID must be MT5 ticket — fail loudly on missing (Blocker G, Fix Q)

**Files:**
- Modify: `src/execution/executor.py:186-222`
- Test: `tests/test_trade_id_no_fallback.py`

- [ ] **Step 1: Read `src/execution/executor.py` lines 170-230 (the `place_market_order` callsite)** to see the exact code structure.

Run: Read tool with offset 170, limit 70.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_trade_id_no_fallback.py
from unittest.mock import MagicMock
import pytest
from src.execution.executor import Executor
from src.config import load_config

def test_missing_trade_id_raises_and_alerts():
    cfg = load_config()
    client = MagicMock()
    risk = MagicMock()
    risk.evaluate.return_value.approved = True
    risk.evaluate.return_value.units = 100
    risk.evaluate.return_value.stop_loss_pips = 10
    risk.evaluate.return_value.take_profit_pips = 18
    # Broker returns a fill response with NO trade ID
    client.place_market_order.return_value = {"orderFillTransaction": {}}

    executor = Executor(cfg, client, risk)
    with pytest.raises(ValueError, match="trade ID"):
        executor.execute_signal(
            instrument="EUR_USD", direction="buy",
            ml_confidence=0.7, atr_value=0.0010, atr_ratio=1.0,
        )
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_trade_id_no_fallback.py -v`
Expected: PASS-fail mismatch (likely the existing code falls back to `local_<timestamp>` instead of raising).

- [ ] **Step 4: Modify executor to require MT5 ticket**

In `src/execution/executor.py:186-195`, replace the trade-ID fallback with a hard error. The current code shape (per audit):

```python
            trade_id = (
                response.get("orderFillTransaction", {}).get("tradeOpened", {}).get("tradeID")
                or response.get("orderFillTransaction", {}).get("id")
                or f"local_{int(time.time())}"   # <-- DELETE THIS
            )
```

Replace with:

```python
            fill = response.get("orderFillTransaction") or {}
            trade_id = (
                fill.get("tradeOpened", {}).get("tradeID")
                or fill.get("id")
            )
            if not trade_id or not isinstance(trade_id, str):
                msg = (
                    f"BROKER RESPONSE MISSING TRADE ID: {response!r}. "
                    "Position may be open on broker but is untracked locally. "
                    "Reconciliation will detect on next 60s sync."
                )
                logger.error(msg)
                if hasattr(self, "telegram") and self.telegram:
                    self.telegram.exception_alert("trade_id_missing", msg)
                raise ValueError(f"Missing trade ID in broker response: {response!r}")
```

(Note: `telegram.exception_alert` is added in Task 3.9. Until then, the `hasattr` guard makes this safe.)

- [ ] **Step 5: Run test to verify pass**

Run: `python -m pytest tests/test_trade_id_no_fallback.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/execution/executor.py tests/test_trade_id_no_fallback.py
git commit -m "fix(executor): require MT5 ticket as trade ID; never fall back to timestamp (blockers G, Q)"
```

---

### Task 2.7: Daily summary scheduler (Blocker H)

**Files:**
- Modify: `src/control/schedulers.py` (add `DailySummaryScheduler` class)
- Modify: `src/main.py` (wire it)
- Modify: `src/monitoring/performance.py` (expose a `daily_breakdown()` if not already present)

- [ ] **Step 1: Add `DailySummaryScheduler` to `src/control/schedulers.py`**

Append:

```python
class DailySummaryScheduler:
    """Fires `on_summary` callback once per day at the configured UTC hour."""

    def __init__(self, hour_utc: int, on_summary, name: str = "daily-summary"):
        self._hour_utc = hour_utc
        self._on_summary = on_summary
        self._name = name
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name=self._name)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            now = datetime.now(timezone.utc)
            wait_s = seconds_until_next_session_boundary(now, self._hour_utc)
            logger.info(f"{self._name}: next fire in {wait_s}s")
            if self._stop.wait(wait_s):
                return
            try:
                self._on_summary()
            except Exception as e:
                logger.error(f"{self._name} callback error: {e}", exc_info=True)
```

- [ ] **Step 2: Wire into `main.py`**

In `__init__` add `self._daily_summary_scheduler = None`.

In `setup()`:

```python
        from src.control.schedulers import DailySummaryScheduler
        summary_hour = self.config.get("telegram.daily_summary_hour_utc", 21)
        self._daily_summary_scheduler = DailySummaryScheduler(
            hour_utc=summary_hour,
            on_summary=self._send_daily_summary,
        )
```

In `run()` after `_session_scheduler.start()`:

```python
        self._daily_summary_scheduler.start()
```

In `shutdown()`:

```python
        if self._daily_summary_scheduler:
            self._daily_summary_scheduler.stop()
```

Add the method:

```python
    def _send_daily_summary(self):
        from datetime import datetime as _dt, timezone as _tz
        try:
            today = _dt.now(_tz.utc).date().isoformat()
            balance = self.client.get_account_balance()
            summary = self.performance.get_summary(days=1)
            self.telegram.daily_summary(
                date=today,
                trades=summary.get("total_trades", 0),
                wins=summary.get("wins", 0),
                losses=summary.get("losses", 0),
                pnl=summary.get("total_pnl", 0.0),
                balance=balance,
                win_rate=summary.get("win_rate", 0.0),
                max_drawdown=summary.get("max_drawdown", 0.0),
            )
            logger.info("Daily summary sent.")
        except Exception as e:
            logger.error(f"Daily summary failed: {e}", exc_info=True)
```

- [ ] **Step 3: Verify `performance.get_summary(days=1)` exists**

Run: `grep -n "def get_summary" src/monitoring/performance.py`
If the existing signature is `get_summary()` without args, add a `days: int | None = None` parameter that filters trades by `exit_time` >= now - timedelta(days=days). The implementer should read the current method (it operates over `journal` DataFrames) and add the filter.

- [ ] **Step 4: Smoke-test the wiring (no waiting 24h)**

Manually fire the callback in a Python REPL:

```python
from src.main import TraderBot
bot = TraderBot(); bot.setup()
bot._send_daily_summary()  # Should send a Telegram message
```

(Skip if you don't have Telegram configured locally.)

- [ ] **Step 5: Commit**

```bash
git add src/control/schedulers.py src/main.py src/monitoring/performance.py
git commit -m "feat(monitoring): wire DailySummaryScheduler at 21:00 UTC (blocker H)"
```

---

# Phase 3 — Hardening Fixes

### Task 3.1: NSSM watchdog runbook (Fix J)

**Files:**
- Create: `docs/runbooks/nssm-service.md`

This is documentation, not code — it goes into the VPS provisioning runbook in Phase 5 too, but is captured here as a standalone reference.

- [ ] **Step 1: Write the runbook**

```markdown
# NSSM Service Configuration

Install NSSM (https://nssm.cc) on the VPS, then:

    nssm install TraderBot "C:\traderbot\venv\Scripts\python.exe" "-m" "src.main"
    nssm set TraderBot AppDirectory "C:\traderbot"
    nssm set TraderBot AppStdout "C:\traderbot\data\logs\nssm-stdout.log"
    nssm set TraderBot AppStderr "C:\traderbot\data\logs\nssm-stderr.log"
    nssm set TraderBot AppRotateFiles 1
    nssm set TraderBot AppRotateOnline 1
    nssm set TraderBot AppRotateBytes 10485760
    nssm set TraderBot AppExit Default Restart
    nssm set TraderBot AppRestartDelay 10000
    nssm set TraderBot Start SERVICE_AUTO_START
    nssm start TraderBot

To inspect:
    nssm status TraderBot
    nssm stop TraderBot
```

- [ ] **Step 2: Commit**

```bash
git add docs/runbooks/nssm-service.md
mkdir -p docs/runbooks  # if it does not already exist after Phase 1
git commit -m "docs(runbook): NSSM service configuration with restart delay + log rotation (fix J)"
```

---

### Task 3.2: ML retraining incremental checkpointing (Fix K)

**Files:**
- Modify: `src/ml/trainer.py` (add a `save_checkpoint(step, partial_state)` call inside training loop)
- Modify: `src/ml/evaluator.py` (load checkpoint on startup if present)

- [ ] **Step 1: Read `src/ml/trainer.py` to find the train loop** — likely a `train()` method that iterates over hyperparameter / cv folds.

- [ ] **Step 2: Add a checkpoint write per CV fold**

After each fold (typically inside the `for fold in folds:` loop), append:

```python
            checkpoint_path = PROJECT_ROOT / "src" / "ml" / "model_store" / f"_checkpoint_{instrument}.json"
            checkpoint_path.write_text(json.dumps({
                "instrument": instrument,
                "fold_index": fold_idx,
                "completed_folds": fold_idx + 1,
                "started_at": started_at.isoformat(),
                "metrics_so_far": metrics_so_far,
            }))
```

- [ ] **Step 3: On train start, look for existing checkpoint and log it**

At the top of `train()`:

```python
        checkpoint_path = PROJECT_ROOT / "src" / "ml" / "model_store" / f"_checkpoint_{instrument}.json"
        if checkpoint_path.exists():
            logger.warning(
                f"Stale training checkpoint found for {instrument}: {checkpoint_path}. "
                "Previous train was interrupted. Starting fresh."
            )
            checkpoint_path.unlink()
```

- [ ] **Step 4: On successful train completion, clear the checkpoint**

At end of `train()`, after model is saved:

```python
        if checkpoint_path.exists():
            checkpoint_path.unlink()
```

- [ ] **Step 5: Commit**

```bash
git add src/ml/trainer.py
git commit -m "feat(ml): write per-fold checkpoint during training; clear on success (fix K)"
```

---

### Task 3.3: Symbol suffix re-detection on reconnect (Fix L)

**Status:** Already covered by Task 2.2 step 3 (the `self._suffix_cache.pop(instrument, None)` line on reconnect). Verify.

- [ ] **Step 1: Verify suffix cache is invalidated on reconnect**

Run: `grep -n "_suffix_cache.pop" src/data/mt5_client.py`
Expected: 1 match (inside `stream_prices()`).

- [ ] **Step 2: If not present, add it** — the implementer should look at `_to_mt5_symbol` (around line 631-662) to confirm the cache attribute name. Common names: `_symbol_cache`, `_resolved_symbols`. Adjust the pop call to match.

- [ ] **Step 3: No commit (covered by 2.2)**

---

### Task 3.4: Tick timestamp validation (Fix M)

**Files:**
- Modify: `src/data/candle_builder.py` (around line 98 per audit)

- [ ] **Step 1: Add validation around `fromisoformat`**

Find the `fromisoformat` call around line 98 of `candle_builder.py`. Wrap it:

```python
        try:
            ts = datetime.fromisoformat(tick_time_str)
        except (ValueError, TypeError) as e:
            logger.warning(f"Bad tick timestamp {tick_time_str!r}: {e}; dropping tick")
            return
        if ts.tzinfo is None:
            # Treat naive timestamps as UTC (broker server time is UTC for Exness)
            ts = ts.replace(tzinfo=timezone.utc)
```

- [ ] **Step 2: Commit**

```bash
git add src/data/candle_builder.py
git commit -m "fix(data): validate tick timestamps; coerce naive to UTC (fix M)"
```

---

### Task 3.5: Spread-aware order deviation (Fix N)

**Files:**
- Modify: `src/data/mt5_client.py:456` (the `place_market_order` deviation field)

- [ ] **Step 1: Make deviation a function of current spread**

Find the order request dict in `place_market_order`. Currently:

```python
            "deviation": 20,
```

Change to:

```python
            "deviation": max(20, int(current_spread_points * 1.5)),
```

This requires `current_spread_points` to be available in scope. The function already takes a `current_price`/`bid`/`ask` — derive spread:

```python
            tick = mt5.symbol_info_tick(symbol)
            if tick is not None:
                spread_points = abs(tick.ask - tick.bid) / mt5.symbol_info(symbol).point
            else:
                spread_points = 0
            deviation = max(20, int(spread_points * 1.5))
```

Then use `deviation` in the request.

- [ ] **Step 2: Commit**

```bash
git add src/data/mt5_client.py
git commit -m "fix(mt5): spread-aware order deviation; min 20 points (fix N)"
```

---

### Task 3.6: Fill-transaction structure validation (Fix O)

**Status:** Already partially handled by Task 2.6 (the strict trade-ID check raises on malformed fills). Add a logger.error with the raw response shape so debugging is easier.

- [ ] **Step 1: Verify the `raise ValueError` from Task 2.6 includes the response repr** — yes, it does.

- [ ] **Step 2: No additional commit (covered by 2.6)**

---

### Task 3.7: Exponential backoff for transient MT5 retcodes (Fix P)

**Files:**
- Modify: `src/data/mt5_client.py:478-482` (the order_send error path)

- [ ] **Step 1: Identify transient retcodes**

Per MT5 docs:
- `TRADE_RETCODE_REQUOTE` (10004) — broker re-quoted; retry
- `TRADE_RETCODE_PRICE_CHANGED` (10020) — retry
- `TRADE_RETCODE_PRICE_OFF` (10021) — retry
- `TRADE_RETCODE_TIMEOUT` (10008) — retry
Other retcodes are real errors and should not retry.

- [ ] **Step 2: Wrap the order_send in a retry loop with backoff**

In `place_market_order` (or wherever `mt5.order_send` is called), wrap:

```python
        import MetaTrader5 as mt5
        retryable = {
            mt5.TRADE_RETCODE_REQUOTE,
            mt5.TRADE_RETCODE_PRICE_CHANGED,
            mt5.TRADE_RETCODE_PRICE_OFF,
            mt5.TRADE_RETCODE_TIMEOUT,
        }
        backoff = 0.05
        last_response = None
        for attempt in range(3):
            response = mt5.order_send(request)
            last_response = response
            if response is None:
                time.sleep(backoff); backoff *= 2
                continue
            if response.retcode == mt5.TRADE_RETCODE_DONE:
                break
            if response.retcode not in retryable:
                break
            logger.info(f"Order retry {attempt+1}/3 (retcode {response.retcode})")
            time.sleep(backoff); backoff *= 2

        if last_response is None or last_response.retcode != mt5.TRADE_RETCODE_DONE:
            raise MT5Error(f"order_send failed: {last_response}")
```

- [ ] **Step 3: Commit**

```bash
git add src/data/mt5_client.py
git commit -m "fix(mt5): retry transient retcodes (requote, price changed, timeout) with backoff (fix P)"
```

---

### Task 3.8: Journal fee/swap columns (Fix R)

**Files:**
- Modify: `src/monitoring/trade_journal.py:41-69`

- [ ] **Step 1: Add columns to the `trades` table**

Add to the CREATE TABLE statement (line 45-69):

```sql
                    commission_zar REAL DEFAULT 0,
                    swap_zar REAL DEFAULT 0,
                    net_pnl_zar REAL,
```

- [ ] **Step 2: Add migration for existing databases**

Add to `_init_db` after the CREATE TABLE blocks:

```python
            # Migration: add fee/swap columns if missing (idempotent)
            for column, ddl in [
                ("commission_zar", "REAL DEFAULT 0"),
                ("swap_zar", "REAL DEFAULT 0"),
                ("net_pnl_zar", "REAL"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE trades ADD COLUMN {column} {ddl}")
                except sqlite3.OperationalError:
                    pass  # Column already exists
```

- [ ] **Step 3: Update `record_trade` to accept these fields**

Add `commission_zar=0.0, swap_zar=0.0, net_pnl_zar=None` parameters; include in the INSERT.

- [ ] **Step 4: Update the executor to pass them on close**

In `executor.py` where it calls `journal.record_trade`, fetch commission/swap from MT5's deal info and pass through.

- [ ] **Step 5: Commit**

```bash
git add src/monitoring/trade_journal.py src/execution/executor.py
git commit -m "feat(journal): commission, swap, net_pnl columns with idempotent migration (fix R)"
```

---

### Task 3.9: General exception alerting (Fix S)

**Files:**
- Modify: `src/monitoring/telegram_bot.py` (add `exception_alert(category, message)`)
- Modify: `src/main.py` (wire to top-level exception handler in main loop)

- [ ] **Step 1: Add `exception_alert` to `TelegramBot`**

```python
    def exception_alert(self, category: str, message: str):
        text = (
            f"⛔ <b>EXCEPTION [{category}]</b>\n"
            f"<pre>{message[:1500]}</pre>"
        )
        self._send(text)
```

- [ ] **Step 2: Wire into the main loop's outer except block**

In `src/main.py:229-231`, change:

```python
        except Exception as e:
            logger.critical(f"Unhandled exception in main loop: {e}", exc_info=True)
```

to:

```python
        except Exception as e:
            logger.critical(f"Unhandled exception in main loop: {e}", exc_info=True)
            try:
                import traceback
                self.telegram.exception_alert(
                    category="main_loop",
                    message=f"{e}\n\n{traceback.format_exc()}",
                )
            except Exception:
                pass
```

- [ ] **Step 3: Commit**

```bash
git add src/monitoring/telegram_bot.py src/main.py
git commit -m "feat(monitoring): exception_alert with category + traceback to Telegram (fix S)"
```

---

# Phase 4 — Control Plane (file-based queue, audit log, `tb` CLI)

### Task 4.1: Control queue (`src/control/queue.py`) + tests

**Files:**
- Create: `src/control/queue.py`
- Test: `tests/test_control_queue.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_control_queue.py
import json
import time
from pathlib import Path
import pytest
from src.control.queue import ControlQueue, ControlCommand

@pytest.fixture
def queue(tmp_path):
    return ControlQueue(root=tmp_path)

def test_enqueue_and_drain(queue):
    cid = queue.enqueue(ControlCommand(verb="status", args={}, reason="probe", requested_by="cli"))
    assert cid.startswith("cmd_")
    pending = queue.drain()
    assert len(pending) == 1
    assert pending[0].verb == "status"
    # Draining is idempotent — already-consumed are removed
    assert queue.drain() == []

def test_write_result_creates_outbox_file(queue, tmp_path):
    cid = queue.enqueue(ControlCommand(verb="pause", args={}, reason="news", requested_by="cli"))
    pending = queue.drain()
    queue.write_result(pending[0].id, {"outcome": "applied", "summary": "paused"})
    out = (tmp_path / "outbox" / f"{pending[0].id}.result.json").read_text()
    payload = json.loads(out)
    assert payload["outcome"] == "applied"

def test_atomic_write_via_tmp_rename(queue, tmp_path):
    # Mid-write should never expose a partial .cmd.json
    queue.enqueue(ControlCommand(verb="status", args={}, reason="probe", requested_by="cli"))
    cmd_files = list((tmp_path / "inbox").glob("*.cmd.json"))
    tmp_files = list((tmp_path / "inbox").glob("*.tmp"))
    assert len(cmd_files) == 1
    assert len(tmp_files) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_control_queue.py -v`
Expected: ImportError on `src.control.queue`.

- [ ] **Step 3: Implement `src/control/queue.py`**

```python
"""
File-based command queue used by the `tb` CLI to send commands to the
running bot. Commands are atomically written to inbox/, the bot polls
once per main-loop iteration, processes the oldest first, and writes
results to outbox/.

Atomic-write protocol:
    1. Caller writes to inbox/<id>.cmd.json.tmp
    2. Caller os.replace(.tmp, .cmd.json)  -- atomic on Windows + POSIX

Bot reads inbox/, sorts by mtime, processes oldest, deletes after writing
to outbox/<id>.result.json.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("traderbot.control.queue")


@dataclass
class ControlCommand:
    verb: str
    args: dict
    reason: str
    requested_by: str
    requested_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    id: str = field(default_factory=lambda: f"cmd_{uuid.uuid4().hex[:12]}")


class ControlQueue:
    def __init__(self, root: Path):
        self._root = Path(root)
        self._inbox = self._root / "inbox"
        self._outbox = self._root / "outbox"
        self._inbox.mkdir(parents=True, exist_ok=True)
        self._outbox.mkdir(parents=True, exist_ok=True)

    def enqueue(self, command: ControlCommand) -> str:
        path = self._inbox / f"{command.id}.cmd.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(command), ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)  # atomic
        return command.id

    def drain(self) -> list[ControlCommand]:
        """Return all pending commands sorted oldest first; remove from inbox."""
        files = sorted(self._inbox.glob("*.cmd.json"), key=lambda p: p.stat().st_mtime)
        out: list[ControlCommand] = []
        for f in files:
            try:
                payload = json.loads(f.read_text(encoding="utf-8"))
                out.append(ControlCommand(**payload))
            except Exception as e:
                logger.error(f"Bad command file {f}: {e}")
            finally:
                try:
                    f.unlink()
                except FileNotFoundError:
                    pass
        return out

    def write_result(self, command_id: str, payload: dict[str, Any]) -> None:
        path = self._outbox / f"{command_id}.result.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)

    def read_result(self, command_id: str, timeout_s: float = 5.0) -> dict[str, Any] | None:
        """Block up to timeout_s waiting for a result file."""
        path = self._outbox / f"{command_id}.result.json"
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if path.exists():
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            time.sleep(0.05)
        return None
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/test_control_queue.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/control/queue.py tests/test_control_queue.py
git commit -m "feat(control): file-based command queue with atomic writes"
```

---

### Task 4.2: control_log audit table + audit_log helper

**Files:**
- Modify: `src/monitoring/trade_journal.py:_init_db` (add control_log table)
- Create: `src/control/audit_log.py`
- Test: `tests/test_audit_log.py`

- [ ] **Step 1: Add control_log table to journal init**

In `src/monitoring/trade_journal.py:_init_db`, append a third CREATE TABLE:

```python
            conn.execute("""
                CREATE TABLE IF NOT EXISTS control_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cmd_id TEXT,
                    ts_utc TEXT NOT NULL,
                    verb TEXT NOT NULL,
                    args_json TEXT,
                    reason TEXT,
                    requested_by TEXT,
                    before_config_json TEXT,
                    after_config_json TEXT,
                    outcome TEXT NOT NULL,
                    notes TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS ix_control_log_ts ON control_log(ts_utc)")
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_audit_log.py
from pathlib import Path
import sqlite3
from src.control.audit_log import AuditLog

def test_record_and_read_back(tmp_path):
    db = tmp_path / "j.db"
    sqlite3.connect(db).execute("""
        CREATE TABLE control_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cmd_id TEXT, ts_utc TEXT, verb TEXT, args_json TEXT,
            reason TEXT, requested_by TEXT,
            before_config_json TEXT, after_config_json TEXT,
            outcome TEXT, notes TEXT
        )""")
    log = AuditLog(db)
    cid = log.record_pending(cmd_id="cmd_x", verb="pause", args={}, reason="news", requested_by="cli", before_config={"x": 1})
    log.record_outcome(cmd_id="cmd_x", outcome="applied", after_config={"x": 1, "paused": True}, notes="ok")
    rows = log.recent(limit=1)
    assert rows[0]["verb"] == "pause"
    assert rows[0]["outcome"] == "applied"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_audit_log.py -v`
Expected: ImportError.

- [ ] **Step 4: Implement `src/control/audit_log.py`**

```python
"""Audit log for control-plane commands. Backed by trades.db control_log table."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AuditLog:
    def __init__(self, db_path: Path):
        self._db_path = Path(db_path)

    def record_pending(self, *, cmd_id: str, verb: str, args: dict, reason: str,
                       requested_by: str, before_config: dict) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT INTO control_log
                   (cmd_id, ts_utc, verb, args_json, reason, requested_by, before_config_json, outcome)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')""",
                (cmd_id, datetime.now(timezone.utc).isoformat(), verb,
                 json.dumps(args), reason, requested_by, json.dumps(before_config)),
            )

    def record_outcome(self, *, cmd_id: str, outcome: str, after_config: dict | None = None,
                       notes: str | None = None) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """UPDATE control_log
                   SET outcome = ?, after_config_json = ?, notes = ?
                   WHERE cmd_id = ?""",
                (outcome, json.dumps(after_config or {}), notes or "", cmd_id),
            )

    def last_tune(self) -> dict[str, Any] | None:
        with sqlite3.connect(self._db_path) as conn:
            cur = conn.execute(
                """SELECT cmd_id, ts_utc, args_json, before_config_json, outcome
                   FROM control_log
                   WHERE verb = 'tune' AND outcome = 'applied'
                   ORDER BY id DESC LIMIT 1"""
            )
            row = cur.fetchone()
        if not row:
            return None
        return {
            "cmd_id": row[0], "ts_utc": row[1],
            "args": json.loads(row[2]), "before_config": json.loads(row[3]),
            "outcome": row[4],
        }

    def tune_count_last_24h(self) -> int:
        cutoff = (datetime.now(timezone.utc).timestamp() - 24 * 3600)
        cutoff_iso = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat()
        with sqlite3.connect(self._db_path) as conn:
            cur = conn.execute(
                "SELECT COUNT(*) FROM control_log WHERE verb='tune' AND outcome='applied' AND ts_utc > ?",
                (cutoff_iso,),
            )
            return cur.fetchone()[0]

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        with sqlite3.connect(self._db_path) as conn:
            cur = conn.execute(
                """SELECT cmd_id, ts_utc, verb, args_json, reason, requested_by,
                          before_config_json, after_config_json, outcome, notes
                   FROM control_log ORDER BY id DESC LIMIT ?""",
                (limit,),
            )
            rows = cur.fetchall()
        return [
            {
                "cmd_id": r[0], "ts_utc": r[1], "verb": r[2],
                "args": json.loads(r[3] or "{}"), "reason": r[4], "requested_by": r[5],
                "before_config": json.loads(r[6] or "{}"),
                "after_config": json.loads(r[7] or "{}"),
                "outcome": r[8], "notes": r[9],
            } for r in rows
        ]
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_audit_log.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/monitoring/trade_journal.py src/control/audit_log.py tests/test_audit_log.py
git commit -m "feat(control): control_log table + AuditLog helper"
```

---

### Task 4.3: Control queue polling in main loop + write-command handlers

**Files:**
- Modify: `src/main.py` (poll queue once per loop iteration; dispatch verbs to handlers)
- Modify: `src/monitoring/telegram_bot.py` (add `control_event(verb, outcome, summary)`)

- [ ] **Step 1: Add `control_event` to TelegramBot**

```python
    def control_event(self, verb: str, outcome: str, summary: str = "", requested_by: str = "cli"):
        emoji = "✅" if outcome == "applied" else "⛔"
        text = (
            f"{emoji} <b>[CONTROL] {verb} {outcome}</b>\n"
            f"by: {requested_by}\n"
            f"{summary}"
        )
        self._send(text)
```

- [ ] **Step 2: Wire control queue into `setup()` and `run()`**

In `__init__`:
```python
        self._control_queue = None
        self._audit_log = None
        self._tunes_paused = False
```

In `setup()` (after journal init):
```python
        from src.control.queue import ControlQueue
        from src.control.audit_log import AuditLog
        from src.control.effective_config import EffectiveConfig
        control_root = PROJECT_ROOT / "data" / "control"
        self._control_queue = ControlQueue(root=control_root)
        self._audit_log = AuditLog(self.journal.db_path)
        self._effective_config = EffectiveConfig.load()
```

(`PROJECT_ROOT` is from `src.config`.)

- [ ] **Step 3: Poll queue once per main-loop iteration**

In `run()`'s `while self.running:` block, before the hourly status log:

```python
                # Poll control queue
                if self._control_queue:
                    cmds = self._control_queue.drain()
                    for cmd in cmds:
                        self._handle_control_command(cmd)
```

- [ ] **Step 4: Implement the dispatcher**

Add `_handle_control_command` to `TraderBot`:

```python
    def _handle_control_command(self, cmd):
        """Dispatch a CLI command to the matching handler. All write commands
        are audit-logged before and after; all emit a Telegram event."""
        from src.control.queue import ControlQueue
        verb = cmd.verb
        before = self._snapshot_effective_config()
        outcome, summary = "rejected", "unknown verb"
        try:
            self._audit_log.record_pending(
                cmd_id=cmd.id, verb=verb, args=cmd.args, reason=cmd.reason,
                requested_by=cmd.requested_by, before_config=before,
            )
            self.telegram.control_event(verb, "requested",
                summary=cmd.reason, requested_by=cmd.requested_by)

            if verb == "pause":
                self._tunes_paused = True
                outcome, summary = "applied", "Bot paused — no new entries"
            elif verb == "resume":
                self._tunes_paused = False
                outcome, summary = "applied", "Bot resumed"
            elif verb == "tune":
                outcome, summary = self._apply_tune(cmd.args, cmd.reason)
            elif verb == "revert":
                outcome, summary = self._revert_last_tune()
            elif verb in ("status", "trades", "perf", "positions", "config", "logs", "model"):
                # Read-only verbs are dispatched to read handlers (see Task 4.4)
                from cli.handlers import dispatch_read
                outcome, summary_obj = dispatch_read(self, verb, cmd.args)
                self._control_queue.write_result(cmd.id, {"outcome": outcome, "data": summary_obj})
                self._audit_log.record_outcome(cmd_id=cmd.id, outcome=outcome,
                                                after_config=before, notes="read-only")
                return  # read-only path is done
            else:
                outcome, summary = "rejected", f"unknown verb: {verb}"
        except Exception as e:
            outcome, summary = "error", f"{type(e).__name__}: {e}"
            logger.exception(f"Control verb {verb} failed")

        after = self._snapshot_effective_config()
        self._audit_log.record_outcome(cmd_id=cmd.id, outcome=outcome,
                                        after_config=after, notes=summary)
        self._control_queue.write_result(cmd.id, {"outcome": outcome, "summary": summary})
        self.telegram.control_event(verb, outcome, summary=summary,
                                     requested_by=cmd.requested_by)
```

- [ ] **Step 5: Implement helpers `_snapshot_effective_config`, `_apply_tune`, `_revert_last_tune`**

```python
    def _snapshot_effective_config(self):
        return {
            "risk.risk_per_trade_pct": self.config.get("risk.risk_per_trade_pct"),
            "ml.confidence_threshold_high": self.config.get("ml.confidence_threshold_high"),
            "ml.confidence_threshold_low": self.config.get("ml.confidence_threshold_low"),
            "risk.consecutive_loss_pause_at": self.config.get("risk.consecutive_loss_pause_at"),
            "_paused": self._tunes_paused,
        }

    def _apply_tune(self, args: dict, reason: str) -> tuple[str, str]:
        TUNABLE = {
            "risk.risk_per_trade_pct": (0.5, 2.5),
            "ml.confidence_threshold_high": (0.50, 0.75),
            "ml.confidence_threshold_low": (0.45, 0.65),
            "risk.consecutive_loss_pause_at": (2, 8),
        }
        key = args.get("key"); value = args.get("value")
        if key not in TUNABLE:
            return "rejected", f"key not tunable: {key}"
        if self._effective_config.is_safety_locked(key):
            return "rejected", f"key is safety-locked: {key}"
        lo, hi = TUNABLE[key]
        try:
            value = float(value)
        except Exception:
            return "rejected", f"value not numeric: {value}"
        if not (lo <= value <= hi):
            return "rejected", f"value out of bounds: {value} not in [{lo}, {hi}]"
        if self._audit_log.tune_count_last_24h() >= 1:
            return "rejected", "rate limit: max 1 tune per 24h"
        # Apply
        self._effective_config = self._write_tune(key, value)
        self.config.set(key, value)  # may need to add a .set() method to Config
        return "applied", f"{key} = {value} (reason: {reason})"

    def _write_tune(self, key: str, value):
        from src.control.effective_config import EffectiveConfig, TUNES_PATH
        TUNES_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            tunes = json.loads(TUNES_PATH.read_text(encoding="utf-8"))
        except Exception:
            tunes = {}
        # nested set
        node = tunes
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
        TUNES_PATH.write_text(json.dumps(tunes, indent=2), encoding="utf-8")
        return EffectiveConfig.load()

    def _revert_last_tune(self) -> tuple[str, str]:
        last = self._audit_log.last_tune()
        if not last:
            return "rejected", "no tune to revert"
        key = last["args"].get("key")
        # Restore prior value from before_config snapshot
        prior_value = last["before_config"].get(key)
        if prior_value is None:
            return "rejected", "prior value unknown"
        self._effective_config = self._write_tune(key, prior_value)
        self.config.set(key, prior_value)
        return "applied", f"reverted {key} -> {prior_value}"
```

- [ ] **Step 6: Wire `_tunes_paused` into the entry guard**

In `_on_candle_complete` (right after the broker-disconnect guard from Task 2.2):

```python
        if self._tunes_paused:
            logger.debug(f"Skipping entry on {instrument}: bot paused via control plane")
            return
```

- [ ] **Step 7: Add `set` method to `Config` if missing**

Read `src/config.py` to see if `Config.set(key, value)` exists. If not, add a simple in-memory setter:

```python
    def set(self, dotted_key: str, value):
        node = self._data
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
```

- [ ] **Step 8: Commit**

```bash
git add src/main.py src/monitoring/telegram_bot.py src/config.py
git commit -m "feat(control): dispatch verbs from queue; pause/resume/tune/revert with audit log"
```

---

### Task 4.4: `tb` CLI entry point + read handlers

**Files:**
- Create: `cli/__init__.py` (empty)
- Create: `cli/tb.py`
- Create: `cli/handlers.py` (read-handler dispatch)
- Test: `tests/test_tb_cli_smoke.py`

- [ ] **Step 1: Write empty `cli/__init__.py`**

```python
"""Command-line interface for talking to a running TraderBot instance."""
```

- [ ] **Step 2: Write read handlers in `cli/handlers.py`**

```python
"""Read-only handlers for the tb CLI. Each returns (outcome, data_dict)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path


def dispatch_read(bot, verb: str, args: dict) -> tuple[str, dict]:
    handlers = {
        "status": _status,
        "trades": _trades,
        "perf": _perf,
        "positions": _positions,
        "config": _config,
        "logs": _logs,
        "model": _model,
    }
    fn = handlers.get(verb)
    if fn is None:
        return "rejected", {"error": f"unknown read verb: {verb}"}
    try:
        return "applied", fn(bot, args)
    except Exception as e:
        return "error", {"error": f"{type(e).__name__}: {e}"}


def _status(bot, args):
    try:
        balance = bot.client.get_account_balance()
        connected = bot.client.is_broker_connected()
    except Exception:
        balance, connected = 0.0, False
    return {
        "balance_zar": balance,
        "broker_connected": connected,
        "open_positions": len(bot.executor.open_trades) if bot.executor else 0,
        "paused": bot._tunes_paused,
        "model_version": bot.predictor.model_version if bot.predictor else None,
        "effective_config": bot._snapshot_effective_config(),
        "drawdown_today_pct": bot.risk_manager.drawdown_tracker.get_daily_drawdown_pct(balance) * 100,
    }


def _trades(bot, args):
    days = int(args.get("days", 7))
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    import sqlite3
    with sqlite3.connect(bot.journal.db_path) as conn:
        rows = conn.execute(
            "SELECT trade_id, instrument, direction, units, entry_price, exit_price, "
            "entry_time, exit_time, pnl_zar, exit_reason, ml_confidence "
            "FROM trades WHERE entry_time > ? ORDER BY entry_time DESC",
            (cutoff,),
        ).fetchall()
    cols = ["trade_id", "instrument", "direction", "units", "entry_price", "exit_price",
            "entry_time", "exit_time", "pnl_zar", "exit_reason", "ml_confidence"]
    return {"trades": [dict(zip(cols, r)) for r in rows]}


def _perf(bot, args):
    days = int(args.get("days", 7))
    return bot.performance.get_summary(days=days)


def _positions(bot, args):
    if not bot.executor:
        return {"positions": []}
    out = []
    for trade in bot.executor.open_trades.values():
        out.append({
            "trade_id": trade.trade_id, "instrument": trade.instrument,
            "direction": trade.direction, "units": trade.units,
            "entry_price": trade.entry_price, "stop_loss": trade.stop_loss,
            "take_profit": trade.take_profit,
            "unrealized_pnl_zar": getattr(trade, "current_pnl", 0.0),
        })
    return {"positions": out}


def _config(bot, args):
    return bot._snapshot_effective_config()


def _logs(bot, args):
    tail = int(args.get("tail", 200))
    level = args.get("level", "").upper()
    log_file = Path(bot.config._project_root) / "data" / "logs" / "traderbot.log"
    if not log_file.exists():
        return {"lines": [], "warning": f"log file missing: {log_file}"}
    with log_file.open("r", encoding="utf-8", errors="ignore") as fh:
        lines = fh.readlines()
    if level:
        lines = [ln for ln in lines if f"| {level} |" in ln]
    return {"lines": lines[-tail:]}


def _model(bot, args):
    if not bot.predictor or not bot.predictor.model:
        return {"loaded": False}
    return {
        "loaded": True,
        "version": getattr(bot.predictor, "model_version", "unknown"),
        "feature_count": getattr(bot.predictor, "feature_count", None),
    }
```

- [ ] **Step 3: Write the `tb.py` entry point**

```python
"""tb — TraderBot CLI. Sends a command into the running bot's queue and prints the result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make sure src/ is importable when run as `python -m cli.tb`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT
from src.control.queue import ControlQueue, ControlCommand


READ_VERBS = ["status", "trades", "perf", "positions", "config", "logs", "model"]
WRITE_VERBS = ["pause", "resume", "tune", "revert"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tb", description="TraderBot CLI")
    sub = p.add_subparsers(dest="verb", required=True)

    for v in READ_VERBS:
        sp = sub.add_parser(v)
        sp.add_argument("--days", type=int, default=7)
        sp.add_argument("--tail", type=int, default=200)
        sp.add_argument("--level", type=str, default="")

    pause_p = sub.add_parser("pause")
    pause_p.add_argument("--reason", required=True)

    resume_p = sub.add_parser("resume")
    resume_p.add_argument("--reason", required=True)

    tune_p = sub.add_parser("tune")
    tune_p.add_argument("kv", help="key=value")
    tune_p.add_argument("--reason", required=True)

    revert_p = sub.add_parser("revert")

    return p


def main():
    args = build_parser().parse_args()

    queue = ControlQueue(root=PROJECT_ROOT / "data" / "control")

    if args.verb == "tune":
        if "=" not in args.kv:
            print("ERROR: tune argument must be key=value", file=sys.stderr)
            sys.exit(2)
        k, v = args.kv.split("=", 1)
        cmd = ControlCommand(
            verb="tune", args={"key": k, "value": v},
            reason=args.reason, requested_by="cli",
        )
    elif args.verb in ("pause", "resume"):
        cmd = ControlCommand(verb=args.verb, args={}, reason=args.reason, requested_by="cli")
    elif args.verb == "revert":
        cmd = ControlCommand(verb="revert", args={}, reason="-", requested_by="cli")
    elif args.verb in READ_VERBS:
        ra = {}
        if args.verb in ("trades", "perf"): ra["days"] = args.days
        if args.verb == "logs":
            ra["tail"] = args.tail
            ra["level"] = args.level
        cmd = ControlCommand(verb=args.verb, args=ra, reason="-", requested_by="cli")
    else:
        print(f"unknown verb: {args.verb}", file=sys.stderr); sys.exit(2)

    cmd_id = queue.enqueue(cmd)
    result = queue.read_result(cmd_id, timeout_s=10.0)
    if result is None:
        print(json.dumps({"error": "timeout waiting for bot response (is bot running?)"}, indent=2))
        sys.exit(3)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Smoke test (no live bot needed)**

```python
# tests/test_tb_cli_smoke.py
import subprocess
import sys

def test_tb_help():
    r = subprocess.run([sys.executable, "-m", "cli.tb", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "TraderBot CLI" in r.stdout

def test_tb_unknown_verb_errors():
    r = subprocess.run([sys.executable, "-m", "cli.tb", "frobnicate"],
                       capture_output=True, text=True)
    assert r.returncode != 0
```

Run: `python -m pytest tests/test_tb_cli_smoke.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cli/__init__.py cli/tb.py cli/handlers.py tests/test_tb_cli_smoke.py
git commit -m "feat(cli): tb command-line entry point with read + write verbs"
```

---

# Phase 5 — VPS Provisioning Runbook

### Task 5.1: Hetzner + MT5 + NSSM + SSH provisioning runbook

**Files:**
- Create: `docs/runbooks/vps-provisioning.md`

- [ ] **Step 1: Write the runbook**

```markdown
# VPS Provisioning Runbook — Hetzner Windows + MT5 + TraderBot

Target: Hetzner CX22 or CPX21 Windows Server 2022 (~€10-15/mo, EU region).
Outcome: TraderBot running 24/7 as a Windows service, controllable via SSH from your local Claude Code.

## 1. Provision the VPS

1. Log into Hetzner Cloud Console.
2. New project → New server.
3. **Location:** Falkenstein or Nuremberg (EU). Closest to Exness MT5 servers.
4. **Image:** Windows Server 2022 (Standard).
5. **Type:** CPX21 (3 vCPU, 4 GB RAM) — comfortable for MT5 + bot + room.
6. **SSH keys:** add your local public key (Hetzner uses these for the OpenSSH server preinstalled in the image).
7. **Name:** `traderbot-prod`.
8. Create. Note the IPv4.

## 2. First-time login

1. RDP into the VPS using credentials emailed by Hetzner.
2. Verify OpenSSH is running:
   `Get-Service sshd`
3. Add your local public key to `C:\Users\<your-user>\.ssh\authorized_keys` if not already there.
4. From your local machine, verify SSH works:
   `ssh Administrator@<vps-ip>`

## 3. Install TraderBot dependencies

On the VPS (PowerShell as admin):
```powershell
# Python 3.11
winget install -e --id Python.Python.3.11

# MT5 terminal (Exness branded)
# Download from https://www.exness.com/trading-platforms/metatrader-5/ and run installer

# NSSM
choco install nssm -y   # If choco isn't installed: install Chocolatey first
# OR manually from https://nssm.cc/download

# Git
winget install -e --id Git.Git
```

## 4. Clone and bootstrap TraderBot

```powershell
mkdir C:\traderbot
cd C:\traderbot
git clone <repo-url> .
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 5. Configure credentials

Copy `.env.example` to `.env` and fill in:
- `MT5_LOGIN` — Exness demo account number
- `MT5_PASSWORD` — Exness demo password
- `MT5_SERVER` — e.g., "Exness-MT5Trial7"
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`

## 6. Configure MT5 auto-login

1. Launch MT5 terminal manually.
2. Tools → Options → Server tab → check "Save account information".
3. Log in to your Exness demo account.
4. Tools → Options → Charts → uncheck "Disable charts" (lower CPU).
5. Tools → Options → Expert Advisors → check "Allow algorithmic trading".
6. Quit MT5. Reopen — should auto-login.

## 7. Set MT5 to launch on boot

1. Right-click MT5 shortcut → "Pin to Start" or copy to:
   `C:\Users\<user>\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup\`
2. Restart the VPS. Verify MT5 auto-launches and auto-logs in.

## 8. Install TraderBot as NSSM service

See `docs/runbooks/nssm-service.md` for the exact `nssm` commands. Briefly:
```powershell
nssm install TraderBot "C:\traderbot\venv\Scripts\python.exe" "-m" "src.main"
nssm set TraderBot AppDirectory "C:\traderbot"
nssm set TraderBot AppStdout "C:\traderbot\data\logs\nssm-stdout.log"
nssm set TraderBot AppStderr "C:\traderbot\data\logs\nssm-stderr.log"
nssm set TraderBot AppRotateFiles 1
nssm set TraderBot AppRotateOnline 1
nssm set TraderBot AppRotateBytes 10485760
nssm set TraderBot AppExit Default Restart
nssm set TraderBot AppRestartDelay 10000
nssm set TraderBot Start SERVICE_AUTO_START
nssm start TraderBot
```

Verify: `nssm status TraderBot` → `SERVICE_RUNNING`. Telegram should ping a "TraderBot Started" message.

## 9. Set up `tb` alias on your local machine

On your local machine (Windows PowerShell):

```powershell
# Add this to your $PROFILE
function tb { ssh Administrator@<vps-ip> "cd C:\traderbot && .\venv\Scripts\python.exe -m cli.tb $args" }
```

Or add to bash on macOS/Linux:
```bash
alias tb='ssh Administrator@<vps-ip> "cd C:\\traderbot \&\& .\\venv\\Scripts\\python.exe -m cli.tb"'
```

Test: `tb status` — should print a JSON status block.

## 10. Cutover checks

- Force-restart the VPS (Hetzner console → Power off, Power on). Verify:
  - MT5 auto-launches.
  - NSSM auto-starts TraderBot.
  - You receive a "TraderBot Started" Telegram alert within 90s.
- `tb status` works from local machine.
- `tb pause --reason "post-deploy verification"` succeeds.
- Telegram receives `[CONTROL] pause applied` event.
- `tb resume --reason "test done"` re-enables.
```

- [ ] **Step 2: Commit**

```bash
git add docs/runbooks/vps-provisioning.md
git commit -m "docs(runbook): VPS provisioning end-to-end (Hetzner + MT5 + NSSM + SSH + tb alias)"
```

---

# Phase 6 — Shakedown

### Task 6.1: 48-hour shakedown checklist

**Files:**
- Create: `docs/runbooks/shakedown-checklist.md`

- [ ] **Step 1: Write the checklist**

```markdown
# 48-Hour Shakedown Checklist

Run on the VPS for 48 hours before starting the month-long unattended demo.
Goal: prove every safety mechanism actually fires.

## Day 1 — startup + read commands

- [ ] Service installed via NSSM, status = SERVICE_RUNNING.
- [ ] Telegram "TraderBot Started" received.
- [ ] `tb status` returns sensible JSON (broker_connected: true, balance > 0).
- [ ] `tb config` shows ZAR currency, hard_floor_zar = 9000, risk_per_trade_pct = 1.5.
- [ ] `tb trades --days 1` returns trades from today (or empty if no trades yet — fine).
- [ ] `tb logs --tail 50 --level ERROR` returns no errors.

## Day 1 — write commands

- [ ] `tb pause --reason "shakedown test"` succeeds. Telegram event received. Bot stops opening new trades (verify via logs over the next M1 candle).
- [ ] `tb resume --reason "shakedown test"` succeeds. Telegram event received.
- [ ] `tb tune risk.risk_per_trade_pct=1.0 --reason "shakedown test"` succeeds. Telegram event received.
- [ ] `tb status` reflects the new value.
- [ ] Second `tb tune` within 24h is REJECTED (rate limit). Telegram event shows rejection.
- [ ] `tb revert` succeeds. Telegram event received. `tb status` shows risk_per_trade_pct back to 1.5.
- [ ] Out-of-bounds `tb tune risk.risk_per_trade_pct=10.0 --reason "test"` is REJECTED.
- [ ] Safety-locked `tb tune risk.hard_floor_zar=5000 --reason "test"` is REJECTED.

## Day 1 — recovery

- [ ] Force-kill the bot process from PowerShell: `Stop-Process -Name python -Force`.
- [ ] Within 10 seconds, NSSM auto-restarts. Telegram "TraderBot Started" received.
- [ ] Reconciliation loop runs within 60s of restart. Verify open positions are picked up: `tb positions` should match what was open before kill.

## Day 1 — disconnect

- [ ] Manually kill MT5 terminal: `Stop-Process -Name terminal64 -Force`.
- [ ] Within 5 seconds: Telegram "MT5 connection lost" alert.
- [ ] `tb status` shows `broker_connected: false`.
- [ ] No new trades open during disconnect (check logs).
- [ ] Restart MT5 (or wait — MT5 may auto-reopen if pinned to startup).
- [ ] Within 10 seconds of MT5 reconnecting to broker: Telegram "MT5 connection restored" alert with downtime.

## Day 2 — daily summary + session boundary

- [ ] At 21:00 UTC: Telegram "Daily Summary" received with date, trades, wins/losses, P&L, balance.
- [ ] At 21:00 UTC: log line "Session boundary fired" appears. `tb status` shows daily_drawdown_pct reset to ~0.
- [ ] If consecutive_losses had been > 0, verify they reset to 0 at 21:00 UTC.

## Decision gate

If ALL checks pass: green light to start the month-long unattended run.
If ANY block-level check fails: fix the issue (or roll back the deploy) and re-shakedown.
```

- [ ] **Step 2: Commit**

```bash
git add docs/runbooks/shakedown-checklist.md
git commit -m "docs(runbook): 48-hour shakedown checklist before month-long demo"
```

---

# Final integration

### Task 7.1: requirements.txt + project housekeeping

- [ ] **Step 1: Add psutil to requirements (only used if needed; lock file uses TCP so optional)**

Skip if all tasks pass without psutil. The TCP lock works without it.

- [ ] **Step 2: Run the full test suite**

Run: `python -m pytest tests/ -v`
Expected: all tests pass.

- [ ] **Step 3: Final spec/plan link in README or CLAUDE.md**

Add a one-line pointer in `CLAUDE.md` under `### Build Status`:

```
- Autonomous-on-VPS + tb CLI: see docs/superpowers/specs/2026-05-02-autonomous-vps-and-cli-control-design.md
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(CLAUDE): pointer to autonomous-VPS spec"
```

---

# Self-review (writing-plans skill final step)

**Spec coverage:**
- Architecture diagram (deployment topology) → covered by VPS runbook (Phase 5) + control plane (Phase 4).
- CLI surface (read + write commands) → Tasks 4.3, 4.4.
- Authority model + tunable whitelist → Task 4.3 (`_apply_tune` TUNABLE dict).
- Un-overridable safety floor → Tasks 0.2 + 4.3 (`is_safety_locked`).
- Audit log → Task 4.2.
- Telegram hooks for control commands → Task 4.3 (`control_event`).
- All 11 BLOCKERS (A-I) → Tasks 2.1-2.7.
- All 10 FIXES (J-S) → Tasks 3.1-3.9.
- AI scaffolding cleanup → Tasks 1.1-1.4.
- VPS provisioning runbook → Task 5.1.
- 48-hour shakedown → Task 6.1.

**Placeholder scan:** No "TBD", "TODO", or "implement later" markers; every step has either exact code, exact commands, or specific file:line references.

**Type consistency:** `ControlCommand` (queue.py) used in queue + dispatcher + cli — same shape throughout. `EffectiveConfig` consistent. `AuditLog` API matches between Tasks 4.2 and 4.3.

**Spec deviation noted:** `approval_queue.py` is deleted (not repurposed). Documented in plan opening.
