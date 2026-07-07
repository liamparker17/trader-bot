"""
Manager policy — lever whitelist, growth-stage risk ceiling, and the
validate/clamp gate that stands between a Claude-manager's proposals and
the control queue.

No Anthropic SDK, no network. Pure functions operating on
`EffectiveConfig` (for current values + safety-floor checks) and plain
dicts/lists (for proposals). The control queue (`src/control/queue.py`)
still does the actual enqueue/apply — this module only decides what's
*allowed* to be enqueued.

`LEVERS` / `WEIGHT_BOUNDS` deliberately re-export the bounds already
defined in `src.control.queue` (`TUNE_BOUNDS` / `WEIGHT_BOUNDS`) rather
than redefining them, so the two gates (manual `tb tune` and the
Claude-manager) can never drift apart.
"""

from __future__ import annotations

from typing import Any, Optional

import yaml

from src.config import PROJECT_ROOT
from src.control.effective_config import EffectiveConfig
from src.control.queue import TUNE_BOUNDS, WEIGHT_BOUNDS

# Single source of truth for lever bounds — see module docstring.
LEVERS: dict[str, tuple[float, float]] = dict(TUNE_BOUNDS)

INSTRUMENTS_PATH = PROJECT_ROOT / "config" / "instruments.yaml"

# Max number of proposals the manager may enact per cycle. Proposals
# beyond this are rejected outright (not evaluated for validity).
MAX_PROPOSALS_PER_CYCLE = 3

THRESHOLD_LOW_KEY = "ml.confidence_threshold_low"
THRESHOLD_HIGH_KEY = "ml.confidence_threshold_high"
RISK_KEY = "risk.risk_per_trade_pct"

# Hard cap on risk_per_trade_pct regardless of growth stage.
RISK_CEILING_HARD_CAP = 2.5

# Growth-stage risk ceiling ladder. Index i is the ceiling once balance
# has reached milestones[i] (milestones assumed sorted ascending, as in
# config/settings.yaml's growth.milestones: [1500, 2000, 3000, 4500, 6000]).
_CEILING_LADDER = [1.8, 2.0, 2.2, 2.5, 2.5]
_CEILING_BELOW_FIRST_MILESTONE = 1.5


def _valid_instruments() -> set[str]:
    """Instrument names from config/instruments.yaml (empty set if missing)."""
    if not INSTRUMENTS_PATH.exists():
        return set()
    try:
        data = yaml.safe_load(INSTRUMENTS_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        return set()
    return set((data.get("instruments") or {}).keys())


def risk_ceiling_now(balance: float, milestones: list) -> float:
    """
    Growth-stage risk_per_trade_pct ceiling, ladder keyed off `milestones`
    (ascending balance thresholds, e.g. growth.milestones in
    settings.yaml). Below the first milestone: 1.5. At/above each
    subsequent milestone the ceiling steps up per `_CEILING_LADDER`.
    Always hard-capped at `RISK_CEILING_HARD_CAP` (2.5).
    """
    thresholds = sorted(milestones or [])
    ceiling = _CEILING_BELOW_FIRST_MILESTONE
    for threshold, ladder_value in zip(thresholds, _CEILING_LADDER):
        if balance >= threshold:
            ceiling = ladder_value
    return min(ceiling, RISK_CEILING_HARD_CAP)


def growth_stage(balance: float, milestones: list) -> int:
    """Number of milestones reached/passed at this balance (0..len(milestones))."""
    thresholds = sorted(milestones or [])
    return sum(1 for threshold in thresholds if balance >= threshold)


def _bounds_or_error(key: str) -> tuple[Optional[tuple[float, float]], Optional[tuple[str, str]]]:
    """
    Resolve (bounds, None) for a whitelisted key, or (None, (code, message))
    for a key that can never be applied (unknown key / unknown instrument).
    """
    if key in LEVERS:
        return LEVERS[key], None
    if key.startswith("weight."):
        instrument = key.split(".", 1)[1]
        if instrument in _valid_instruments():
            return WEIGHT_BOUNDS, None
        return None, ("bad_instrument", f"unknown instrument for weight tune: {instrument}")
    return None, ("unknown_key", f"key not whitelisted for tuning: {key}")


def validate_and_clamp(
    proposals: list,
    effective_config: EffectiveConfig,
    risk_ceiling_now: float,
) -> tuple[list, list]:
    """
    Validate and clamp a manager's proposed tunes for one cycle.

    Returns (applied, rejected):
      - applied entries:  {key, value, original_value, reason, clamped}
        `value` is the (possibly clamped) proposed value; "applied" here
        means "approved for enqueue" — the control queue does the actual
        apply downstream.
      - rejected entries: {key, value, reason, rejection_reason}

    Rules:
      - At most MAX_PROPOSALS_PER_CYCLE (3) proposals are considered; the
        4th+ are rejected outright (cycle limit), regardless of validity.
      - Unknown key / unknown instrument (weight.<X>) -> rejected.
      - Non-numeric value -> rejected.
      - Safety-locked key (EffectiveConfig.is_safety_locked) -> rejected.
      - In-bounds numeric values pass through unclamped; out-of-bounds
        values are clamped to the nearest bound (`clamped: True`).
      - `risk.risk_per_trade_pct`'s effective upper bound is additionally
        capped at `risk_ceiling_now` (min of the lever's static bound and
        the growth-stage ceiling).
      - `ml.confidence_threshold_low` <= `ml.confidence_threshold_high` is
        enforced pair-wise against the RESULTING config: if only one side
        is proposed, it's checked against the other side's current
        effective value; if both are proposed, they're checked against
        each other post-clamp. A violation rejects the offending
        proposal(s) (both, if both proposed) rather than applying either.
    """
    accepted = list(proposals[:MAX_PROPOSALS_PER_CYCLE])
    overflow = list(proposals[MAX_PROPOSALS_PER_CYCLE:])

    # Dedup within the accepted slice BEFORE validation: last occurrence
    # of a given key wins (positional order of the raw slice is otherwise
    # preserved); earlier occurrences are rejected outright as superseded
    # so a key can never appear twice across (applied, rejected).
    last_index_for_key: dict[str, int] = {}
    for i, p in enumerate(accepted):
        last_index_for_key[p.get("key")] = i

    superseded: list = []
    deduped_accepted: list = []
    for i, p in enumerate(accepted):
        key = p.get("key")
        if last_index_for_key.get(key) != i:
            superseded.append(p)
        else:
            deduped_accepted.append(p)
    accepted = deduped_accepted

    # key -> (final_value, clamped, original_value)
    resolved: dict[str, tuple[float, bool, Any]] = {}
    # key -> (rejection_reason_code, message)
    reject_map: dict[str, tuple[str, str]] = {}

    for p in accepted:
        key = p.get("key")
        raw_value = p.get("value")

        bounds, error = _bounds_or_error(key)
        if error is not None:
            reject_map[key] = error
            continue

        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            reject_map[key] = ("non_numeric", f"value must be numeric: {raw_value!r}")
            continue

        if effective_config.is_safety_locked(key):
            reject_map[key] = ("safety_locked", f"key is safety-locked: {key}")
            continue

        lo, hi = bounds
        if key == RISK_KEY:
            hi = min(hi, risk_ceiling_now)

        clamped = False
        final_value = value
        if value < lo:
            final_value = lo
            clamped = True
        elif value > hi:
            final_value = hi
            clamped = True

        resolved[key] = (final_value, clamped, value)

    # Pair-wise threshold_low <= threshold_high invariant against the
    # RESULTING config (proposed+clamped value if proposed, else the
    # current effective value).
    if THRESHOLD_LOW_KEY in resolved or THRESHOLD_HIGH_KEY in resolved:
        low_val = (
            resolved[THRESHOLD_LOW_KEY][0] if THRESHOLD_LOW_KEY in resolved
            else effective_config.get(THRESHOLD_LOW_KEY)
        )
        high_val = (
            resolved[THRESHOLD_HIGH_KEY][0] if THRESHOLD_HIGH_KEY in resolved
            else effective_config.get(THRESHOLD_HIGH_KEY)
        )
        if low_val is not None and high_val is not None and low_val > high_val:
            reason = ("threshold_invariant", f"low ({low_val}) must be <= high ({high_val})")
            for key in (THRESHOLD_LOW_KEY, THRESHOLD_HIGH_KEY):
                if key in resolved:
                    reject_map[key] = reason
                    del resolved[key]

    applied = []
    rejected = []
    for p in accepted:
        key = p.get("key")
        raw_value = p.get("value")
        if key in reject_map:
            code, message = reject_map[key]
            rejected.append({
                "key": key,
                "value": raw_value,
                "reason": message,
                "rejection_reason": code,
            })
        else:
            final_value, clamped, original_value = resolved[key]
            applied.append({
                "key": key,
                "value": final_value,
                "original_value": original_value,
                "reason": "clamped to bound" if clamped else "within bounds",
                "clamped": clamped,
            })

    for p in overflow:
        rejected.append({
            "key": p.get("key"),
            "value": p.get("value"),
            "reason": f"cycle limit exceeded (max {MAX_PROPOSALS_PER_CYCLE} proposals per cycle)",
            "rejection_reason": "cycle_limit_exceeded",
        })

    for p in superseded:
        rejected.append({
            "key": p.get("key"),
            "value": p.get("value"),
            "reason": "duplicate key in same cycle; a later proposal for this key supersedes it",
            "rejection_reason": "duplicate_key_superseded",
        })

    return applied, rejected
