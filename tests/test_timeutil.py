"""
Tests for src.utils.timeutil.to_utc() -- the single timestamp-normalization
helper used at every MT5 ingestion point (ticks, rates, positions, parsed
strings). Naive datetimes must be rejected, not silently assumed UTC.
"""
from datetime import datetime, timezone, timedelta

import pandas as pd
import pytest

from src.utils.timeutil import to_utc


def test_epoch_int_converts_to_aware_utc():
    result = to_utc(1735689600)
    assert result.tzinfo is not None
    assert result.tzinfo.utcoffset(result) == timedelta(0)


def test_epoch_float_converts_to_aware_utc():
    result = to_utc(1735689600.5)
    assert result.tzinfo is not None


def test_aware_utc_datetime_passthrough():
    dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
    result = to_utc(dt)
    assert result == dt
    assert result.tzinfo is not None


def test_aware_non_utc_datetime_converted_to_utc():
    tz_plus_2 = timezone(timedelta(hours=2))
    dt = datetime(2026, 1, 1, 12, 0, tzinfo=tz_plus_2)
    result = to_utc(dt)
    assert result.tzinfo == timezone.utc
    assert result.hour == 10  # 12:00+02:00 -> 10:00 UTC


def test_naive_datetime_raises_value_error():
    naive = datetime(2026, 1, 1, 12, 0)
    with pytest.raises(ValueError):
        to_utc(naive)


def test_pandas_timestamp_aware_passthrough():
    ts = pd.Timestamp("2026-01-01T00:00:00.000000000Z")
    result = to_utc(ts)
    assert result.tzinfo is not None


def test_pandas_timestamp_naive_raises():
    ts = pd.Timestamp("2026-01-01T00:00:00")
    with pytest.raises(ValueError):
        to_utc(ts)


def test_unsupported_type_raises_value_error():
    with pytest.raises(ValueError):
        to_utc("2026-01-01T00:00:00Z")


def test_bool_rejected_even_though_bool_is_int_subclass():
    with pytest.raises(ValueError):
        to_utc(True)
