"""
src/api/routes/alerts.py

REST API endpoints for alert data in ON-CHAIN SUPER SIGNALS™.

Endpoints:
- GET /api/v1/alerts/today — Today's alerts
- GET /api/v1/alerts/date/{target_date} — Alerts for specific date
- GET /api/v1/alerts/history — Recent alerts (last N days)

Data minimization:
- Only derived alerts are returned
- No raw blockchain data is ever exposed
"""

from __future__ import annotations

from datetime import date
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status

from src.api.schemas import (
    AlertListResponse,
    AlertTypeEnum,
    ChainEnum,
    ErrorResponse,
)
from src.config.settings import MAX_ALERT_HISTORY_DAYS
from src.services.alert_service import (
    AlertsService,
    DatabaseError,
    DataIntegrityError,
    NotYetProcessedError,
)
from src.utils.logger import get_logger

__all__ = ["router"]

_logger = get_logger(__name__)


def get_alerts_service() -> AlertsService:
    """
    FastAPI dependency som instansierer AlertsService.

    Brukes for å unngå global tjenesteinstansiering og gjøre tjenestelaget
    lettere å teste og konfigurere.
    """
    return AlertsService()


router = APIRouter(
    prefix="/api/v1/alerts",
    tags=["alerts"],
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Internal server error",
        },
    },
)


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.get(
    "/today",
    response_model=AlertListResponse,
    summary="Get today's alerts",
    description="Returns all alerts generated for the current date (UTC).",
    responses={
        status.HTTP_200_OK: {"description": "Alerts found (or empty list)"},
        status.HTTP_409_CONFLICT: {
            "model": ErrorResponse,
            "description": "Data for today has not been fully processed yet",
        },
    },
)
async def get_alerts_today(
    chain: Optional[ChainEnum] = Query(
        default=None,
        description="Filter by blockchain: BTC or ETH",
    ),
    alert_type: Optional[AlertTypeEnum] = Query(
        default=None,
        description="Filter by alert type",
    ),
    alerts_service: AlertsService = Depends(get_alerts_service),
) -> AlertListResponse:
    """
    Henter alle alerts for dagens dato (UTC).

    Kan filtreres på kjede og/eller alert-type.
    Tom liste returneres hvis ingen alerts finnes.
    """
    _logger.info(
        "Fetching alerts for today",
        extra={
            "chain": chain.value if chain else None,
            "alert_type": alert_type.value if alert_type else None,
        },
    )

    try:
        response = alerts_service.get_alerts_today(
            chain=chain,
            alert_type=alert_type,
        )
    except NotYetProcessedError as exc:
        # Spesialhåndtering for "for tidlig i dag" når dagens data ikke er klare ennå.
        error = ErrorResponse(
            error="Data not yet available",
            detail=str(exc),
        )
        _logger.warning(
            "Requested alerts for today but data is not yet processed",
            extra={
                "chain": chain.value if chain else None,
                "alert_type": alert_type.value if alert_type else None,
                "error_type": type(exc).__name__,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=error.model_dump(),
        ) from exc
    except (DatabaseError, DataIntegrityError) as exc:
        error = ErrorResponse(
            error="Database error",
            detail="Failed to fetch alerts for today.",
        )
        _logger.error(
            "Database/data integrity error while fetching alerts for today",
            extra={
                "chain": chain.value if chain else None,
                "alert_type": alert_type.value if alert_type else None,
                "error_type": type(exc).__name__,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error.model_dump(),
        ) from exc

    if response.count == 0:
        _logger.warning(
            "No alerts found for today",
            extra={
                "chain": chain.value if chain else None,
                "alert_type": alert_type.value if alert_type else None,
            },
        )
    else:
        _logger.info(
            "Returning alerts for today",
            extra={
                "count": response.count,
                "chain": chain.value if chain else None,
                "alert_type": alert_type.value if alert_type else None,
            },
        )

    return response


@router.get(
    "/date/{target_date}",
    response_model=AlertListResponse,
    summary="Get alerts for a specific date",
    description="Returns all alerts generated for the given date (UTC).",
    responses={
        status.HTTP_200_OK: {"description": "Alerts found (or empty list)"},
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "Invalid date",
        },
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Data for this date is not yet available",
        },
        status.HTTP_409_CONFLICT: {
            "model": ErrorResponse,
            "description": "Data for this date has not been fully processed yet",
        },
    },
)
async def get_alerts_for_date(
    target_date: date = Path(
        ...,
        description="Target date in YYYY-MM-DD format (UTC)",
    ),
    chain: Optional[ChainEnum] = Query(
        default=None,
        description="Filter by blockchain: BTC or ETH",
    ),
    alert_type: Optional[AlertTypeEnum] = Query(
        default=None,
        description="Filter by alert type",
    ),
    alerts_service: AlertsService = Depends(get_alerts_service),
) -> AlertListResponse:
    """
    Henter alle alerts for en spesifikk dato (UTC).

    Kan filtreres på kjede og/eller alert-type.
    Tom liste returneres hvis ingen alerts finnes.
    """
    _logger.info(
        "Fetching alerts for date",
        extra={
            "date": str(target_date),
            "chain": chain.value if chain else None,
            "alert_type": alert_type.value if alert_type else None,
        },
    )

    try:
        response = alerts_service.get_alerts_for_date(
            target_date=target_date,
            chain=chain,
            alert_type=alert_type,
        )
    except NotYetProcessedError as exc:
        # Spesialhåndtering for "for tidlig i dag" når dagens data ikke er klare ennå.
        error = ErrorResponse(
            error="Data not yet available",
            detail=str(exc),
        )
        _logger.warning(
            "Requested alerts for date that is not yet processed",
            extra={
                "date": str(target_date),
                "chain": chain.value if chain else None,
                "alert_type": alert_type.value if alert_type else None,
                "error_type": type(exc).__name__,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=error.model_dump(),
        ) from exc
    except ValueError as exc:
        # Expecting ValueError from service for invalid future dates, etc.
        error = ErrorResponse(
            error="Invalid date",
            detail=str(exc),
        )
        _logger.warning(
            "Rejected invalid date for alerts",
            extra={
                "date": str(target_date),
                "chain": chain.value if chain else None,
                "alert_type": alert_type.value if alert_type else None,
                "error_type": type(exc).__name__,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error.model_dump(),
        ) from exc
    except (DatabaseError, DataIntegrityError) as exc:
        error = ErrorResponse(
            error="Database error",
            detail="Failed to fetch alerts for date.",
        )
        _logger.error(
            "Database/data integrity error while fetching alerts for date",
            extra={
                "date": str(target_date),
                "chain": chain.value if chain else None,
                "alert_type": alert_type.value if alert_type else None,
                "error_type": type(exc).__name__,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error.model_dump(),
        ) from exc

    if response.count == 0:
        _logger.warning(
            "No alerts found for date",
            extra={
                "date": str(target_date),
                "chain": chain.value if chain else None,
                "alert_type": alert_type.value if alert_type else None,
            },
        )
    else:
        _logger.info(
            "Returning alerts for date",
            extra={
                "date": str(target_date),
                "count": response.count,
                "chain": chain.value if chain else None,
                "alert_type": alert_type.value if alert_type else None,
            },
        )

    return response


@router.get(
    "/history",
    response_model=AlertListResponse,
    summary="Get recent alerts",
    description="Returns alerts generated in the last N days (UTC).",
    responses={
        status.HTTP_200_OK: {"description": "Alerts found (or empty list)"},
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "Invalid parameter",
        },
    },
)
async def get_recent_alerts(
    days: int = Query(
        default=7,
        ge=1,
        le=MAX_ALERT_HISTORY_DAYS,
        description=(
            "Number of days to look back "
            f"(1-{MAX_ALERT_HISTORY_DAYS}, configured in settings)."
        ),
    ),
    chain: Optional[ChainEnum] = Query(
        default=None,
        description="Filter by blockchain: BTC or ETH",
    ),
    alert_type: Optional[AlertTypeEnum] = Query(
        default=None,
        description="Filter by alert type",
    ),
    limit: int = Query(
        default=50,
        ge=1,
        le=1000,
        description="Maximum number of alerts to return (1-1000). Default 50.",
    ),
    alerts_service: AlertsService = Depends(get_alerts_service),
) -> AlertListResponse:
    """
    Henter alerts for de siste N dagene (UTC).

    - Validering av days-intervall håndteres i AlertsService og konfigurasjon
    - Validerer limit via Pydantic (1–1000, standard 50)
    - Filtrering på chain og alert_type håndteres i AlertsService
    """
    _logger.info(
        "Fetching recent alerts",
        extra={
            "days": days,
            "chain": chain.value if chain else None,
            "alert_type": alert_type.value if alert_type else None,
            "limit": limit,
        },
    )

    try:
        response = alerts_service.get_recent_alerts(
            days=days,
            chain=chain,
            alert_type=alert_type,
            limit=limit,
        )
    except ValueError as exc:
        # Expecting ValueError from service for invalid days range, etc.
        error = ErrorResponse(
            error="Invalid parameter",
            detail=str(exc),
        )
        _logger.warning(
            "Rejected invalid parameter for recent alerts",
            extra={
                "days": days,
                "chain": chain.value if chain else None,
                "alert_type": alert_type.value if alert_type else None,
                "limit": limit,
                "error_type": type(exc).__name__,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error.model_dump(),
        ) from exc
    except (DatabaseError, DataIntegrityError) as exc:
        error = ErrorResponse(
            error="Database error",
            detail="Failed to fetch recent alerts.",
        )
        _logger.error(
            "Database/data integrity error while fetching recent alerts",
            extra={
                "days": days,
                "chain": chain.value if chain else None,
                "alert_type": alert_type.value if alert_type else None,
                "limit": limit,
                "error_type": type(exc).__name__,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error.model_dump(),
        ) from exc

    if response.count == 0:
        _logger.warning(
            "No alerts found in recent history",
            extra={
                "days": days,
                "chain": chain.value if chain else None,
                "alert_type": alert_type.value if alert_type else None,
                "limit": limit,
            },
        )
    else:
        _logger.info(
            "Returning recent alerts",
            extra={
                "days": days,
                "count": response.count,
                "chain": chain.value if chain else None,
                "alert_type": alert_type.value if alert_type else None,
                "limit": limit,
            },
        )

    return response
