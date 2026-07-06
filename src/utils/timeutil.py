"""
Timestamp normalization helper for MT5 ingestion points.

Every raw timestamp coming out of MetaTrader5 (Unix epoch seconds on ticks/
rates/positions) or parsed from strings must funnel through `to_utc()` so
every ingested timestamp is timezone-aware UTC. Naive datetimes are rejected
(raises ValueError) rather than silently assumed to be UTC -- that silent
assumption is exactly the kind of bug this helper exists to catch.
"""
import logging
from datetime import datetime, timezone

logger = logging.getLogger("traderbot.timeutil")


def to_utc(dt_or_epoch) -> datetime:
    """
    Normalize a timestamp to a timezone-aware UTC datetime.

    Accepts:
    - int/float: treated as Unix epoch seconds (as returned by MT5's
      `time` fields on ticks, rates and positions).
    - datetime (including pandas.Timestamp, which subclasses datetime):
      must already be timezone-aware; naive datetimes are rejected.

    Raises:
        ValueError: if given a naive datetime or an unsupported type.
    """
    if isinstance(dt_or_epoch, bool):
        raise ValueError(f"to_utc() cannot convert bool: {dt_or_epoch!r}")

    if isinstance(dt_or_epoch, (int, float)):
        return datetime.fromtimestamp(dt_or_epoch, tz=timezone.utc)

    if isinstance(dt_or_epoch, datetime):
        if dt_or_epoch.tzinfo is None or dt_or_epoch.tzinfo.utcoffset(dt_or_epoch) is None:
            logger.error(f"to_utc() received a naive datetime: {dt_or_epoch!r}")
            raise ValueError(
                f"to_utc() requires a timezone-aware datetime, got naive: {dt_or_epoch!r}"
            )
        return dt_or_epoch.astimezone(timezone.utc)

    raise ValueError(f"to_utc() cannot convert type {type(dt_or_epoch)!r}: {dt_or_epoch!r}")
