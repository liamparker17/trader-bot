"""
Manager event-drop helper — small, surgical utility used by risk-side
code (drawdown tracker, circuit breaker) to signal the out-of-band
manager scheduler that something worth an early cycle just happened.

Writes `control/manager_events/<ts>_<type>.json` via the same
tmp-write-then-os.replace atomic pattern used by `src.control.queue`, so
a half-written file can never be picked up mid-write.

The scheduler (`src.manager.scheduler.ManagerScheduler`) consumes
(reads + deletes) these files oldest-first.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.config import PROJECT_ROOT

logger = logging.getLogger("traderbot.manager.events")

MANAGER_EVENTS_DIR = PROJECT_ROOT / "control" / "manager_events"


def drop_manager_event(
    event_type: str,
    detail: Optional[dict] = None,
    events_dir: Path = MANAGER_EVENTS_DIR,
) -> None:
    """
    Write one event file for the manager scheduler to pick up. Best-effort:
    never raises — a failure here must never interrupt trading logic.
    """
    try:
        events_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y%m%dT%H%M%S%f")
        name = f"{ts}_{event_type}_{uuid.uuid4().hex[:8]}"
        payload = {
            "type": event_type,
            "detail": detail or {},
            "ts_utc": now.isoformat(),
        }
        tmp_path = events_dir / f"{name}.json.tmp"
        final_path = events_dir / f"{name}.json"
        tmp_path.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp_path, final_path)
    except Exception as e:  # pragma: no cover — defensive, must never raise
        logger.warning(f"drop_manager_event({event_type!r}) failed: {e}")
