"""
src/services/alert_service.py

Service layer for alert retrieval in ON-CHAIN SUPER SIGNALS™.

Responsibilities:
- Translate high-level alert queries (date ranges, chain, alert type, limits)
  into database calls via src.utils.db_handler.
- Enforce business rules such as:
  - No future dates
  - Maximum history window (MAX_ALERT_HISTORY_DAYS)
  - Supported chains
  - "Not yet processed" handling for today's date before the daily pipeline
    has completed.
- Provide a clean abstraction (AlertsService) for the API layer to consume.
- Perform basic data integrity checks on fetched alert records.

Data minimization:
- Read-only access to the `alerts` table via db_handler.
- No access to raw blockchain data or feature tables.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from src.api.schemas import AlertResponse
from src.config.settings import MAX_ALERT_HISTORY_DAYS
from src.utils import db_handler, validators
from src.utils.logger import get_logger

__all__ = [
    "AlertServiceError",
    "DatabaseError",
    "DataIntegrityError",
    "NotYetProcessedError",
    "AlertsService",
]


_logger = get_logger(__name__)


class AlertServiceError(Exception):
    """Base exception for all errors in AlertsService."""
    pass


class DatabaseError(AlertServiceError):
    """Raised when there is an issue with the underlying database operation."""
    pass


class DataIntegrityError(AlertServiceError):
    """
    Raised when fetched data fails basic integrity checks
    (e.g. missing mandatory fields or malformed records).
    """
    pass


class NotYetProcessedError(AlertServiceError):
    """
    Raised when requesting data for today, but the daily pipeline has not run yet.

    For ON-CHAIN SUPER SIGNALS™ the daily pipelines are expected to complete
    by 02:00 UTC. Requests for today's alerts before this time should be
    treated as "not yet processed" rather than "no data".
    """
    pass


class AlertsService:
    """
    Service class responsible for retrieving alert data from DuckDB via db_handler.

    This class encapsulates all business logic for:
    - Validating input parameters (dates, chains, history windows)
    - Handling "not yet processed" semantics for today's data
    - Converting high-level requests into concrete database queries
    - Performing basic integrity checks on the returned alert records
    """

    #: Required fields that each alert record from the database must contain.
    _REQUIRED_ALERT_FIELDS = ("id", "chain", "date", "alert_type", "message", "created_at")

    def __init__(self, db_handler_instance: Optional[Any] = None) -> None:
        """
        Initialize the AlertsService.

        Parameters
        ----------
        db_handler_instance : Optional[Any]
            Optional injected db_handler-like instance for testing.
            If None, uses the shared src.utils.db_handler module.
        """
        self._logger = _logger
        self._db_handler = db_handler_instance if db_handler_instance is not None else db_handler

    # -------------------------------------------------------------------------
    # PUBLIC METHODS
    # -------------------------------------------------------------------------

    def get_alerts_for_date(self, target_date: date, chain: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve all alerts for a specific date and optional chain.

        Parameters
        ----------
        target_date : date
            The date for which to fetch alerts. Must not be in the future
            relative to current UTC date.
        chain : Optional[str]
            Optional chain identifier (e.g. "BTC", "ETH").
            If provided, it must be a supported chain.

        Returns
        -------
        List[Dict[str, Any]]
            A list of alert records as dictionaries, each matching the alerts
            table schema:
            {
                "id": int,
                "chain": str,
                "date": date,
                "alert_type": str,
                "message": str | None,
                "created_at": datetime | None,
            }

        Raises
        ------
        ValueError
            If target_date is in the future or if chain is invalid.
        NotYetProcessedError
            If target_date is today (UTC) and the daily pipeline is not yet
            expected to have completed (before 02:00 UTC) or if no alerts exist
            in the alerts table for today after the expected pipeline time.
        DatabaseError
            If an underlying database operation fails.
        DataIntegrityError
            If the fetched data is missing mandatory fields or is malformed.
        """
        utc_today = datetime.now(timezone.utc).date()

        # Future date validation
        if target_date > utc_today:
            self._logger.debug(
                "Rejected get_alerts_for_date due to future date",
                extra={"target_date": str(target_date), "utc_today": str(utc_today)},
            )
            raise ValueError("target_date cannot be in the future")

        # "Not yet processed" handling for today's date
        if target_date == utc_today:
            now_utc = datetime.now(timezone.utc)

            # Before expected pipeline completion (02:00 UTC): always "not yet processed"
            if now_utc.hour < 2:
                self._logger.info(
                    "Alerts for today requested before pipeline completion window",
                    extra={
                        "target_date": str(target_date),
                        "now_utc": now_utc.isoformat(),
                        "cutoff_hour_utc": 2,
                    },
                )
                raise NotYetProcessedError(
                    "Alerts for today are not available yet. "
                    "Daily pipeline is expected to complete by 02:00 UTC."
                )

            # After expected pipeline completion: check alerts table directly
            try:
                alerts_exist = self._db_handler.check_if_alerts_exist(target_date)
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "Database error while checking if alerts exist for today",
                    extra={
                        "target_date": str(target_date),
                        "now_utc": now_utc.isoformat(),
                        "error_type": type(exc).__name__,
                    },
                )
                raise DatabaseError("Failed to check if alerts exist for today.") from exc

            if not alerts_exist:
                self._logger.info(
                    "No alerts found for today after pipeline window; treating as not yet processed",
                    extra={
                        "target_date": str(target_date),
                        "now_utc": now_utc.isoformat(),
                        "cutoff_hour_utc": 2,
                    },
                )
                raise NotYetProcessedError(
                    "Alerts for today are not available yet. "
                    "Daily pipeline window has passed, but no alerts exist in the alerts table."
                )

        # Chain validation (if provided)
        normalized_chain: Optional[str] = None
        if chain is not None:
            normalized_chain = chain.upper()
            validators.validate_chain(normalized_chain)

        try:
            alerts: List[Dict[str, Any]] = self._db_handler.get_alerts_for_date(
                target_date,
                normalized_chain,
            )
        except Exception as exc:  # noqa: BLE001
            # Wrap any underlying db_handler exception into a DatabaseError
            self._logger.error(
                "Database error in get_alerts_for_date",
                extra={
                    "target_date": str(target_date),
                    "chain": normalized_chain,
                    "error_type": type(exc).__name__,
                },
            )
            raise DatabaseError("Failed to fetch alerts for date from database.") from exc

        # Basic integrity checks (structure + Pydantic validation)
        self._validate_alert_records(alerts, context="get_alerts_for_date", date_context=target_date)

        return alerts

    def get_recent_alerts(
        self,
        days: int,
        chain: Optional[str] = None,
        alert_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve alerts for a recent rolling window.

        Parameters
        ----------
        days : int
            Number of days in the look-back window. Must be > 0 and
            <= MAX_ALERT_HISTORY_DAYS.
        chain : Optional[str]
            Optional chain identifier (e.g. "BTC", "ETH").
            If provided, it must be a supported chain.
        alert_type : Optional[str]
            Optional alert type filter (e.g. "whale_spike", "stress_event").
            This value is forwarded to db_handler without additional validation.
        limit : Optional[int]
            Optional maximum number of alerts to return. Forwarded directly to
            db_handler for efficient limiting in the database layer.

        Returns
        -------
        List[Dict[str, Any]]
            A list of alert records as dictionaries, matching the alerts table
            schema.

        Raises
        ------
        ValueError
            If days <= 0, days > MAX_ALERT_HISTORY_DAYS, or chain is invalid.
        DatabaseError
            If an underlying database operation fails.
        DataIntegrityError
            If the fetched data is missing mandatory fields or is malformed.
        """
        # Validate days range
        if days <= 0:
            self._logger.debug(
                "Rejected get_recent_alerts due to non-positive days",
                extra={"days": days},
            )
            raise ValueError("days must be greater than 0")

        if days > MAX_ALERT_HISTORY_DAYS:
            self._logger.debug(
                "Rejected get_recent_alerts due to days exceeding maximum history",
                extra={"days": days, "max_allowed": MAX_ALERT_HISTORY_DAYS},
            )
            raise ValueError(f"days must be less than or equal to {MAX_ALERT_HISTORY_DAYS}")

        # Chain validation (if provided)
        normalized_chain: Optional[str] = None
        if chain is not None:
            normalized_chain = chain.upper()
            validators.validate_chain(normalized_chain)

        # Compute start_date inclusive, such that days=7 covers the last 7 days
        # including today (UTC).
        utc_today = datetime.now(timezone.utc).date()
        start_date = utc_today - timedelta(days=days - 1)

        self._logger.debug(
            "Computed start_date for recent alerts",
            extra={
                "days": days,
                "start_date": str(start_date),
                "utc_today": str(utc_today),
            },
        )

        try:
            alerts: List[Dict[str, Any]] = self._db_handler.get_recent_alerts(
                start_date=start_date,
                chain=normalized_chain,
                alert_type=alert_type,
                limit=limit,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Database error in get_recent_alerts",
                extra={
                    "days": days,
                    "start_date": str(start_date),
                    "chain": normalized_chain,
                    "alert_type": alert_type,
                    "limit": limit,
                    "error_type": type(exc).__name__,
                },
            )
            raise DatabaseError("Failed to fetch recent alerts from database.") from exc

        # Basic integrity checks (structure + Pydantic validation)
        self._validate_alert_records(
            alerts,
            context="get_recent_alerts",
            date_context=None,
        )

        return alerts

    # -------------------------------------------------------------------------
    # PRIVATE HELPERS
    # -------------------------------------------------------------------------

    def _validate_alert_records(
        self,
        alerts: List[Dict[str, Any]],
        *,
        context: str,
        date_context: Optional[date],
    ) -> None:
        """
        Perform basic integrity checks on alert records returned from db_handler.

        Parameters
        ----------
        alerts : List[Dict[str, Any]]
            The list of records returned from the database.
        context : str
            Name of the calling context (e.g. "get_alerts_for_date").
        date_context : Optional[date]
            Optional date associated with the query, used for logging.

        Raises
        ------
        DataIntegrityError
            If any record is not a dict, is missing required fields, or fails
            Pydantic validation against the AlertResponse schema.
        """
        if not alerts:
            # Empty lists are perfectly valid; nothing further to validate.
            self._logger.debug(
                "No alerts returned from database",
                extra={"context": context, "date": str(date_context) if date_context else None},
            )
            return

        for index, record in enumerate(alerts):
            if not isinstance(record, dict):
                self._logger.error(
                    "Non-dict record encountered in alert results",
                    extra={"context": context, "index": index, "record_type": type(record).__name__},
                )
                raise DataIntegrityError(
                    f"Alert record at index {index} is not a dictionary (got {type(record).__name__})."
                )

            missing_fields = [field for field in self._REQUIRED_ALERT_FIELDS if field not in record]
            if missing_fields:
                self._logger.error(
                    "Alert record missing required fields",
                    extra={
                        "context": context,
                        "index": index,
                        "missing_fields": missing_fields,
                        "date": str(date_context) if date_context else None,
                    },
                )
                raise DataIntegrityError(
                    f"Alert record at index {index} is missing required fields: {', '.join(missing_fields)}"
                )

            # Strict Pydantic validation against the public AlertResponse schema
            try:
                AlertResponse.model_validate(record)
            except ValidationError as exc:
                self._logger.error(
                    "Alert record failed Pydantic validation against AlertResponse schema",
                    extra={
                        "context": context,
                        "index": index,
                        "date": str(date_context) if date_context else None,
                        "validation_errors": exc.errors(),
                    },
                )
                raise DataIntegrityError(
                    f"Alert record at index {index} failed AlertResponse validation: {exc}"
                ) from exc

        self._logger.debug(
            "Alert records passed integrity validation",
            extra={
                "context": context,
                "count": len(alerts),
                "date": str(date_context) if date_context else None,
            },
        )
