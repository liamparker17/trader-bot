"""
Effective config = settings.yaml ⊕ tunes overlay ⊕ safety_floor.yaml (last wins).

The CLI tune command writes to data/control/effective_config.json. The bot
reads this overlay on every config access. safety_floor.yaml is loaded last
and always wins; the CLI refuses to write tunes that target safety-floor keys.
"""

from __future__ import annotations

import json
import logging
import os
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


def _expand_dotted(dotted_key: str, value: Any) -> dict:
    """Turn "a.b.c", 1.0 into {"a": {"b": {"c": 1.0}}}."""
    parts = dotted_key.split(".")
    node: dict = {parts[-1]: value}
    for part in reversed(parts[:-1]):
        node = {part: node}
    return node


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

    def apply_tune(self, dotted_key: str, value: Any) -> None:
        """
        Apply a single dotted-key tune: update this instance's in-memory
        view AND persist to the tunes overlay JSON (TUNES_PATH) via an
        atomic read-modify-write so concurrently-tuned keys are merged,
        not clobbered.

        Callers (src/control/queue.py) are responsible for validating the
        key/value against the whitelist, bounds, and safety-floor lock
        BEFORE calling this — it performs no validation of its own.
        """
        existing: dict = {}
        if TUNES_PATH.exists():
            try:
                existing = json.loads(TUNES_PATH.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Failed to read existing tunes overlay {TUNES_PATH}: {e}")
                existing = {}

        nested = _expand_dotted(dotted_key, value)
        merged = _deep_merge(existing, nested)

        TUNES_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = TUNES_PATH.with_suffix(TUNES_PATH.suffix + ".tmp")
        tmp_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        os.replace(tmp_path, TUNES_PATH)

        # Reflect the change in this instance immediately too (in addition
        # to a fresh EffectiveConfig.load() picking it up from disk), so a
        # caller holding a reference doesn't need to re-load to see it.
        self._data = _deep_merge(self._data, nested)
