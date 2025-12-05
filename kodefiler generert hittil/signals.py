"""
src/api/routes/signals.py

REST API endpoints for signal data in ON-CHAIN SUPER SIGNALS™.

Endpoints:
- GET /api/v1/signals/latest — Latest signal for all chains
- GET /api/v1/signals/latest/{chain} — Latest signal for one chain
- GET /api/v1/signals/history/{chain} — Historical signals for one chain

All responses include mandatory LEGAL_DISCLAIMER per legal requirements.
All data is read-only from DuckDB (no writes, no ML inference).

Data minimization:
- Only derived/aggregated signals are returned
- No raw blockchain data is ever exposed
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Path, Query, status

from src.api.schemas import (
    AlertResponse,
    AlertTypeEnum,
    ChainEnum,
    DailySignalResponse,
    ErrorResponse,
    LEGAL_DISCLAIMER,
    RegimeEnum,
    RiskStateEnum,
    SignalHistoryResponse,
)
from src.utils import db_handler
from src.utils.logger import get_logger

__all__ = ["router"]

_logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/signals",
    tags=["signals"],
    responses={
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Signal not found",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Internal server error",
        },
    },
)


# =============================================================================
# PRIVATE HELPERS
# =============================================================================


def _db_alert_to_response(alert_dict: Dict[str, Any]) -> AlertResponse:
    """
    Konverterer en db_handler alert-dict til AlertResponse.

    Parametre
    ---------
    alert_dict:
        Dict fra db_handler.get_alerts_for_date().
        Forventet format:
        {
            "id": int,
            "chain": str,
            "date": date,
            "alert_type": str,
            "message": str | None,
            "created_at": datetime | None
        }

    Returnerer
    ----------
    AlertResponse
    """
    created_at = alert_dict.get("created_at") or datetime.now(timezone.utc)
    message = alert_dict.get("message") or ""

    return AlertResponse(
        id=int(alert_dict["id"]),
        chain=ChainEnum(alert_dict["chain"]),
        date=alert_dict["date"],
        alert_type=AlertTypeEnum(alert_dict["alert_type"]),
        message=message,
        created_at=created_at,
    )


def _db_signal_to_response(
    signal_dict: Dict[str, Any],
    alerts: List[Dict[str, Any]],
) -> DailySignalResponse:
    """
    Konverterer en db_handler signal-dict til DailySignalResponse.

    Parametre
    ---------
    signal_dict:
        Dict fra db_handler.get_latest_signal() eller query_signals().
        Forventet format:
        {
            "chain": str,
            "date": date,
            "risk_score": float,
            "regime": str,
            "risk_state": str,
            "whale_index": float,
            "stress_index": float,
            "momentum_score": float,
            "created_at": datetime | None
        }
    alerts:
        Liste av alert-dicts fra db_handler.get_alerts_for_date().

    Returnerer
    ----------
    DailySignalResponse
        Ferdig respons-modell med:
        - Alle felter mappet
        - alerts konvertert til List[AlertResponse]
        - disclaimer satt til LEGAL_DISCLAIMER
        - updated_at satt til created_at fra DB (eller nåværende tid som fallback)
    """
    created_at = signal_dict.get("created_at") or datetime.now(timezone.utc)

    alert_models = [_db_alert_to_response(alert) for alert in alerts]

    return DailySignalResponse(
        chain=ChainEnum(signal_dict["chain"]),
        date=signal_dict["date"],
        risk_score=float(signal_dict["risk_score"]),
        regime=RegimeEnum(signal_dict["regime"]),
        risk_state=RiskStateEnum(signal_dict["risk_state"]),
        whale_index=float(signal_dict["whale_index"]),
        stress_index=float(signal_dict["stress_index"]),
        momentum_score=float(signal_dict["momentum_score"]),
        alerts=alert_models,
        disclaimer=LEGAL_DISCLAIMER,
        updated_at=created_at,
    )


def _validate_date_range(start_date: date, end_date: date) -> None:
    """
    Validerer at datointervall er gyldig.

    Kaster
    ------
    HTTPException(400)
        Hvis start_date > end_date.
    HTTPException(400)
        Hvis noen dato er i fremtiden.
    """
    today = date.today()

    if start_date > end_date:
        error = ErrorResponse(
            error="Invalid date range",
            detail="start_date must be <= end_date",
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error.model_dump(),
        )

    if start_date > today or end_date > today:
        error = ErrorResponse(
            error="Invalid date",
            detail="Dates cannot be in the future",
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error.model_dump(),
        )


def _normalize_query_result(result: Any) -> List[Dict[str, Any]]:
    """
    Normaliserer resultatet fra db_handler.query_signals til en liste med dicts.

    Merk:
    - db_handler.query_signals returnerer per i dag List[Dict[str, Any]].
    - DataFrame-håndteringen under er defensiv koding for å være fremtidskompatibel
      dersom implementasjonen i db_handler senere endres til å returnere DataFrame.
    """
    if result is None:
        return []

    if isinstance(result, list):
        return result

    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        return to_dict("records")

    # Fallback: uventet type
    _logger.error(
        "Unexpected query_signals result type",
        extra={"type": type(result).__name__},
    )
    return []


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.get(
    "/latest/{chain}",
    response_model=DailySignalResponse,
    summary="Get latest signal for a chain",
    description="Returns the most recent daily signal for the specified blockchain.",
    responses={
        status.HTTP_200_OK: {"description": "Latest signal found"},
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "No signal exists for this chain",
        },
    },
)
async def get_latest_signal_for_chain(
    chain: ChainEnum = Path(
        ...,
        description="Blockchain: BTC or ETH",
    ),
) -> DailySignalResponse:
    """
    Henter siste tilgjengelige signal for gitt kjede.

    Inkluderer:
    - Risk score (0-100)
    - Regime (Bull/Neutral/Bear)
    - Risk state (Risk ON/Neutral/Risk OFF)
    - Whale/stress/momentum indekser
    - Dagens alerts (hvis noen)
    - Juridisk disclaimer
    """
    _logger.info(
        "Fetching latest signal",
        extra={"chain": chain.value},
    )

    try:
        signal_dict = db_handler.get_latest_signal(chain.value)
    except Exception as exc:  # noqa: BLE001
        _logger.error(
            "Database error while fetching latest signal",
            extra={"chain": chain.value, "error": str(exc)},
        )
        error = ErrorResponse(error="Database error", detail=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error.model_dump(),
        ) from exc

    if signal_dict is None:
        _logger.warning(
            "No signal found for chain",
            extra={"chain": chain.value},
        )
        error = ErrorResponse(
            error="No signal found for chain",
            detail=chain.value,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error.model_dump(),
        )

    signal_date = signal_dict["date"]

    try:
        alerts = db_handler.get_alerts_for_date(signal_date, chain.value)
    except Exception as exc:  # noqa: BLE001
        _logger.error(
            "Database error while fetching alerts for latest signal",
            extra={
                "chain": chain.value,
                "date": str(signal_date),
                "error": str(exc),
            },
        )
        error = ErrorResponse(error="Database error", detail=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error.model_dump(),
        ) from exc

    response = _db_signal_to_response(signal_dict, alerts or [])

    _logger.info(
        "Returning latest signal",
        extra={
            "chain": chain.value,
            "date": str(response.date),
        },
    )
    return response


@router.get(
    "/history/{chain}",
    response_model=SignalHistoryResponse,
    summary="Get historical signals for a chain",
    description=(
        "Returns historical daily signals for the specified blockchain "
        "in a given date range."
    ),
    responses={
        status.HTTP_200_OK: {"description": "Historical signals returned"},
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "Invalid date range",
        },
    },
)
async def get_signal_history_for_chain(
    chain: ChainEnum = Path(
        ...,
        description="Blockchain: BTC or ETH",
    ),
    start_date: date = Query(
        ...,
        description="Start date (inclusive) in YYYY-MM-DD format",
    ),
    end_date: date = Query(
        ...,
        description="End date (inclusive) in YYYY-MM-DD format",
    ),
) -> SignalHistoryResponse:
    """
    Henter historiske signaler for gitt kjede i et datointervall.

    Tom liste (ingen signaler) returneres som en gyldig 200-respons.
    """
    _logger.info(
        "Fetching signal history",
        extra={
            "chain": chain.value,
            "start_date": str(start_date),
            "end_date": str(end_date),
        },
    )

    _validate_date_range(start_date, end_date)

    try:
        raw_result = db_handler.query_signals(chain.value, start_date, end_date)
    except Exception as exc:  # noqa: BLE001
        _logger.error(
            "Database error while querying signal history",
            extra={
                "chain": chain.value,
                "start_date": str(start_date),
                "end_date": str(end_date),
                "error": str(exc),
            },
        )
        error = ErrorResponse(error="Database error", detail=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error.model_dump(),
        ) from exc

    signal_dicts = _normalize_query_result(raw_result)
    responses: List[DailySignalResponse] = []

    for signal_dict in signal_dicts:
        signal_date = signal_dict["date"]
        try:
            alerts = db_handler.get_alerts_for_date(signal_date, chain.value)
        except Exception as exc:  # noqa: BLE001
            _logger.error(
                "Database error while fetching alerts for history signal",
                extra={
                    "chain": chain.value,
                    "date": str(signal_date),
                    "error": str(exc),
                },
            )
            error = ErrorResponse(error="Database error", detail=str(exc))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error.model_dump(),
            ) from exc

        responses.append(_db_signal_to_response(signal_dict, alerts or []))

    history_response = SignalHistoryResponse(
        chain=chain,
        start_date=start_date,
        end_date=end_date,
        signals=responses,
        count=len(responses),
        disclaimer=LEGAL_DISCLAIMER,
    )

    if not responses:
        _logger.warning(
            "No signals found in history range",
            extra={
                "chain": chain.value,
                "start_date": str(start_date),
                "end_date": str(end_date),
            },
        )

    _logger.info(
        "Returning signal history",
        extra={
            "chain": chain.value,
            "count": len(responses),
            "start_date": str(start_date),
            "end_date": str(end_date),
        },
    )

    return history_response


@router.get(
    "/latest",
    response_model=List[DailySignalResponse],
    summary="Get latest signals for all chains",
    description="Returns the most recent daily signal for all supported blockchains.",
    responses={
        status.HTTP_200_OK: {"description": "Latest signals found (or empty list)"},
    },
)
async def get_latest_signals_for_all_chains() -> List[DailySignalResponse]:
    """
    Henter siste signal for ALLE kjeder.

    Kjeder uten tilgjengelige signaler blir hoppet over.
    Tom liste returneres som gyldig 200-respons dersom ingen signaler finnes.
    """
    _logger.info("Fetching latest signals for all chains")

    responses: List[DailySignalResponse] = []

    for chain in ChainEnum:
        try:
            signal_dict = db_handler.get_latest_signal(chain.value)
        except Exception as exc:  # noqa: BLE001
            _logger.error(
                "Database error while fetching latest signal for chain",
                extra={"chain": chain.value, "error": str(exc)},
            )
            error = ErrorResponse(error="Database error", detail=str(exc))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error.model_dump(),
            ) from exc

        if signal_dict is None:
            _logger.warning(
                "No latest signal found for chain when fetching all",
                extra={"chain": chain.value},
            )
            continue

        signal_date = signal_dict["date"]

        try:
            alerts = db_handler.get_alerts_for_date(signal_date, chain.value)
        except Exception as exc:  # noqa: BLE001
            _logger.error(
                "Database error while fetching alerts for latest-all signal",
                extra={
                    "chain": chain.value,
                    "date": str(signal_date),
                    "error": str(exc),
                },
            )
            error = ErrorResponse(error="Database error", detail=str(exc))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error.model_dump(),
            ) from exc

        responses.append(_db_signal_to_response(signal_dict, alerts or []))

    _logger.info(
        "Returning latest signals for all chains",
        extra={"count": len(responses)},
    )

    return responses
