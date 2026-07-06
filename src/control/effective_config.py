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
