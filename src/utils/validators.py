"""
Data validation utilities for the ON-CHAIN SUPER SIGNALS™ project.

This module centralizes all validation logic for:
- Supported chains (BTC/ETH)
- Date ranges (no future dates, valid ordering)
- Daily aggregate data structures prior to storage

All other modules MUST use these functions instead of implementing
their own validation logic.

Key principles:
- Data minimization: do not log or persist raw blockchain records.
- Performance: validations should be lightweight and avoid copying data.
- Clarity: error messages must be explicit and actionable.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Union

import numpy as np
import pandas as pd
import polars as pl

from src.config.settings import CHAINS
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Aggregates can be Pandas DataFrame, Polars DataFrame, dict, or other supported structures.
AggregateData = Union[pd.DataFrame, pl.DataFrame, dict, Any]


def _is_critical_column(col_name: str) -> bool:
    """
    Determine whether a column is considered critical.

    Critical columns must not contain NaN or infinite values.
    A column is critical if its name contains any of the configured
    keywords (case-insensitive).
    """
    critical_keywords = ["price", "volume", "count", "value", "total"]
    name = str(col_name).lower()
    return any(keyword in name for keyword in critical_keywords)


def validate_chain(chain: str) -> None:
    """
    Validate that a given chain identifier is supported.

    Args:
        chain: Chain symbol (e.g. "BTC", "ETH", case-insensitive).

    Raises:
        ValueError: If the chain is unsupported or invalid.
    """
    # Normalize chain input to uppercase string.
    chain_str = str(chain).strip()
    chain_upper = chain_str.upper()

    if not chain_str:
        logger.debug("Empty chain validation attempt.")
        raise ValueError(f"Unsupported chain: '{chain}'. Supported chains: {CHAINS}")

    if chain_upper not in CHAINS:
        logger.debug(
            "Invalid chain validation attempt",
            extra={"chain": chain_str, "supported_chains": CHAINS},
        )
        raise ValueError(f"Unsupported chain: '{chain}'. Supported chains: {CHAINS}")


def _ensure_date_type(value: Any, name: str) -> date:
    """
    Ensure that the provided value is a datetime.date (but not datetime).

    Args:
        value: Value to be validated.
        name: Field name for error messages ("start" or "end").

    Returns:
        A datetime.date instance.

    Raises:
        TypeError: If the value is not a date or is a datetime.
    """
    if value is None:
        logger.debug("Date validation failed: %s is None", name)
        raise TypeError(f"{name} must be a datetime.date instance, got None")

    # Reject datetime explicitly (even though it is a subclass of date).
    if isinstance(value, datetime):
        logger.debug(
            "Date validation failed: %s is datetime instead of date",
            name,
            extra={"value": repr(value)},
        )
        raise TypeError(f"{name} must be a datetime.date, not datetime.datetime")

    if not isinstance(value, date):
        logger.debug(
            "Date validation failed: %s is not a date instance",
            name,
            extra={"type": type(value).__name__},
        )
        raise TypeError(
            f"{name} must be a datetime.date instance, got {type(value).__name__}"
        )

    return value


def validate_date_range(start: date, end: date) -> None:
    """
    Validate that a date range is well-formed and does not include future dates.

    Rules:
    - Both start and end must be datetime.date (NOT datetime).
    - start <= end
    - Neither start nor end may be after today's date (UTC).

    Args:
        start: Start date of the range.
        end: End date of the range.

    Raises:
        TypeError: If start or end is not a date.
        ValueError: If the range is invalid or contains future dates.
    """
    start_date = _ensure_date_type(start, "start")
    end_date = _ensure_date_type(end, "end")

    if start_date > end_date:
        logger.debug(
            "Invalid date range (start > end)",
            extra={"start": start_date.isoformat(), "end": end_date.isoformat()},
        )
        raise ValueError(f"Start date {start_date} is after end date {end_date}")

    today_utc = datetime.utcnow().date()
    if start_date > today_utc or end_date > today_utc:
        logger.debug(
            "Date range contains future dates",
            extra={
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "today_utc": today_utc.isoformat(),
            },
        )
        raise ValueError(
            f"Date range contains future dates: {start_date} to {end_date}"
        )


def _extract_date_value(value: Any) -> date:
    """
    Normalize various date-like values to a datetime.date.

    Supports:
    - datetime.date
    - datetime.datetime (converted via .date())
    - ISO8601 strings ("YYYY-MM-DD")

    Raises:
        ValueError: If the value cannot be interpreted as a date.
    """
    if isinstance(value, date) and not isinstance(value, datetime):
        return value

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).date()
        except ValueError:
            logger.debug(
                "Failed to parse date string in aggregates",
                extra={"value": value},
            )
            raise ValueError(f"Unable to parse date value '{value}' in aggregates")

    logger.debug(
        "Unsupported date value type in aggregates",
        extra={"type": type(value).__name__, "value": repr(value)},
    )
    raise ValueError(f"Unsupported date value type: {type(value).__name__}")


def _validate_pandas_aggregates(df: pd.DataFrame, expected_date: date) -> None:
    """
    Internal helper to validate a Pandas DataFrame of daily aggregates.
    """
    if df.empty:
        logger.warning(
            "Empty aggregates detected (Pandas)",
            extra={"expected_date": expected_date.isoformat()},
        )
        raise ValueError(f"Aggregates are empty for date {expected_date}")

    if len(df) != 1:
        logger.warning(
            "Unexpected number of rows in daily aggregates (Pandas)",
            extra={"expected_date": expected_date.isoformat(), "rows": len(df)},
        )
        raise ValueError(
            f"Expected 1 row for date {expected_date}, got {len(df)} rows"
        )

    # Optional: date column check
    if "date" in df.columns:
        actual = df["date"].iloc[0]
        actual_date = _extract_date_value(actual)
        if actual_date != expected_date:
            logger.error(
                "Date mismatch in aggregates (Pandas)",
                extra={
                    "expected_date": expected_date.isoformat(),
                    "actual_date": actual_date.isoformat(),
                },
            )
            raise ValueError(
                f"Date mismatch: expected {expected_date}, got {actual_date}"
            )

    # NaN / inf checks
    nan_cols = df.columns[df.isna().any()].tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_cols = [
        col for col in numeric_cols if np.isinf(df[col].to_numpy()).any()
    ]

    invalid_critical_cols: list[str] = []
    optional_nan_cols: list[str] = []

    # Process NaN columns
    for col in nan_cols:
        if _is_critical_column(col):
            invalid_critical_cols.append(col)
        else:
            optional_nan_cols.append(col)

    # Process inf columns (always critical if numeric)
    for col in inf_cols:
        if col not in invalid_critical_cols:
            invalid_critical_cols.append(col)

    if optional_nan_cols:
        logger.warning(
            "NaN found in non-critical columns (Pandas aggregates)",
            extra={
                "expected_date": expected_date.isoformat(),
                "columns": optional_nan_cols,
            },
        )

    if invalid_critical_cols:
        logger.error(
            "Invalid values (NaN/inf) found in critical columns (Pandas aggregates)",
            extra={
                "expected_date": expected_date.isoformat(),
                "columns": invalid_critical_cols,
            },
        )
        raise ValueError(
            f"Invalid values (NaN/inf) found in columns: {invalid_critical_cols}"
        )


def _validate_polars_aggregates(df: pl.DataFrame, expected_date: date) -> None:
    """
    Internal helper to validate a Polars DataFrame of daily aggregates.
    """
    if df.height == 0:
        logger.warning(
            "Empty aggregates detected (Polars)",
            extra={"expected_date": expected_date.isoformat()},
        )
        raise ValueError(f"Aggregates are empty for date {expected_date}")

    if df.height != 1:
        logger.warning(
            "Unexpected number of rows in daily aggregates (Polars)",
            extra={"expected_date": expected_date.isoformat(), "rows": df.height},
        )
        raise ValueError(
            f"Expected 1 row for date {expected_date}, got {df.height} rows"
        )

    # Optional: date column check
    if "date" in df.columns:
        actual = df["date"][0]
        actual_date = _extract_date_value(actual)
        if actual_date != expected_date:
            logger.error(
                "Date mismatch in aggregates (Polars)",
                extra={
                    "expected_date": expected_date.isoformat(),
                    "actual_date": actual_date.isoformat(),
                },
            )
            raise ValueError(
                f"Date mismatch: expected {expected_date}, got {actual_date}"
            )

    # NaN / inf checks
    nan_cols: list[str] = []
    inf_cols: list[str] = []

    for col in df.columns:
        series = df[col]
        # Nulls count as NaN for our purposes.
        try:
            has_null = bool(series.is_null().any())
        except Exception:
            has_null = False

        # Infinite detection only sensible for float types.
        is_float = series.dtype in (pl.Float32, pl.Float64)
        if is_float:
            try:
                has_inf = bool(series.is_infinite().any())
            except Exception:
                has_inf = False
        else:
            has_inf = False

        if has_null:
            nan_cols.append(col)
        if has_inf:
            inf_cols.append(col)

    invalid_critical_cols: list[str] = []
    optional_nan_cols: list[str] = []

    for col in nan_cols:
        if _is_critical_column(col):
            invalid_critical_cols.append(col)
        else:
            optional_nan_cols.append(col)

    for col in inf_cols:
        if col not in invalid_critical_cols:
            invalid_critical_cols.append(col)

    if optional_nan_cols:
        logger.warning(
            "NaN found in non-critical columns (Polars aggregates)",
            extra={
                "expected_date": expected_date.isoformat(),
                "columns": optional_nan_cols,
            },
        )

    if invalid_critical_cols:
        logger.error(
            "Invalid values (NaN/inf) found in critical columns (Polars aggregates)",
            extra={
                "expected_date": expected_date.isoformat(),
                "columns": invalid_critical_cols,
            },
        )
        raise ValueError(
            f"Invalid values (NaN/inf) found in columns: {invalid_critical_cols}"
        )


def _validate_dict_aggregates(data: dict, expected_date: date) -> None:
    """
    Internal helper to validate a dict representing daily aggregates.

    For simplicity and given expected small size, this is converted to a
    single-row Pandas DataFrame and validated using the Pandas logic.
    """
    if not data:
        logger.warning(
            "Empty aggregates detected (dict)",
            extra={"expected_date": expected_date.isoformat()},
        )
        raise ValueError(f"Aggregates are empty for date {expected_date}")

    # Convert to single-row DataFrame; acceptable since dict aggregates are small.
    df = pd.DataFrame([data])
    _validate_pandas_aggregates(df, expected_date)


def validate_daily_aggregates(data: AggregateData, expected_date: date) -> None:
    """
    Validate that daily aggregates have a correct structure and values.

    Validation steps:
    1. Type validation (Pandas, Polars, or dict).
    2. Non-empty check.
    3. Single-row check (Pandas/Polars).
    4. Optional date column matching against expected_date.
    5. NaN/inf checks on critical vs non-critical columns.

    Args:
        data: Aggregated daily data (Pandas DataFrame, Polars DataFrame, or dict).
        expected_date: The date these aggregates are supposed to represent.

    Raises:
        ValueError: If the aggregates are invalid or unsupported.
    """
    # None is always invalid.
    if data is None:
        logger.warning(
            "Aggregates validation failed: data is None",
            extra={"expected_date": expected_date.isoformat()},
        )
        raise ValueError(f"Aggregates are empty for date {expected_date}")

    # Pandas DataFrame
    if isinstance(data, pd.DataFrame):
        _validate_pandas_aggregates(data, expected_date)
        return

    # Polars DataFrame
    if isinstance(data, pl.DataFrame):
        _validate_polars_aggregates(data, expected_date)
        return

    # dict
    if isinstance(data, dict):
        _validate_dict_aggregates(data, expected_date)
        return

    # Unsupported type
    logger.warning(
        "Unsupported aggregate data type",
        extra={
            "expected_date": expected_date.isoformat(),
            "type": type(data).__name__,
        },
    )
    raise ValueError(f"Unsupported data type: {type(data)}")
