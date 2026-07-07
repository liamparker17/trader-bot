"""
Ratchet Floor — a hard equity floor that only ever rises.

Replaces the old fixed hard_floor_zar kill switch (R9000). Instead of a
static number, the floor is derived from the account's all-time high-water
mark (HWM):

    floor = max(min_floor_zar, high_water_mark * (1 - max_total_drawdown_pct))

Because the HWM is monotonically non-decreasing, the floor never falls even
when the account balance drops — profits "ratchet in" a safer floor, but
losses never lower it below its current level.

State (the HWM) is persisted to a small JSON file so it survives bot
restarts. Writes are atomic (write-tmp-then-os.replace) so a crash mid-write
never leaves a corrupt file; if the file is ever missing or unreadable, the
HWM reseeds to the starting balance rather than failing closed or open.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

logger = logging.getLogger("traderbot.risk.ratchet_floor")

# Seed value used when no state file exists yet or it can't be read.
# Matches the re-based R1000 starting balance (Task 1).
STARTING_HIGH_WATER_MARK_ZAR = 1000.0


class RatchetFloor:
    """Tracks a monotonically non-decreasing equity floor for the account."""

    def __init__(
        self,
        min_floor_zar: float,
        max_total_drawdown_pct: float,
        state_path: Union[str, Path] = "data/account_state.json",
    ):
        self.min_floor_zar = min_floor_zar
        self.max_total_drawdown_pct = max_total_drawdown_pct
        self.state_path = Path(state_path)
        self._high_water_mark: float = self._load_high_water_mark()

    def _load_high_water_mark(self) -> float:
        """Load the persisted HWM, seeding a sane default if missing/corrupt."""
        if not self.state_path.exists():
            return STARTING_HIGH_WATER_MARK_ZAR

        try:
            raw = self.state_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            hwm = float(data["high_water_mark"])
            if hwm <= 0:
                raise ValueError(f"non-positive high_water_mark: {hwm}")
            return hwm
        except Exception as e:
            logger.warning(
                f"Corrupt or unreadable account state at {self.state_path} ({e}); "
                f"reseeding high-water mark to R{STARTING_HIGH_WATER_MARK_ZAR:.2f}"
            )
            return STARTING_HIGH_WATER_MARK_ZAR

    @property
    def current_floor(self) -> float:
        """The current hard floor, in ZAR."""
        return max(
            self.min_floor_zar,
            self._high_water_mark * (1 - self.max_total_drawdown_pct),
        )

    @property
    def high_water_mark(self) -> float:
        return self._high_water_mark

    def update(self, balance: float) -> float:
        """
        Update the high-water mark with the latest balance (if it's a new
        high), persist state, and return the resulting current floor.
        """
        if balance > self._high_water_mark:
            self._high_water_mark = balance
        self._persist()
        return self.current_floor

    def is_breached(self, equity: float) -> bool:
        """True if equity is at or below the current floor."""
        return equity <= self.current_floor

    def _persist(self) -> None:
        """Atomically write state to disk (write-tmp-then-os.replace)."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        payload = {
            "high_water_mark": self._high_water_mark,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        }
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp_path, self.state_path)
