# src/api/routes/chains.py

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException, Path, status

from src.api.schemas import (
    ChainEnum,
    ChainListResponse,
    ChainStatusResponse,
    ChainSummary,
    DailySignalResponse,
    ErrorResponse,
    LEGAL_DISCLAIMER,
    RegimeEnum,
)
from src.api.routes.signals import _db_signal_to_response
from src.utils import db_handler
from src.utils.logger import get_logger

__all__ = ["router"]

_logger = get_logger(__name__)

# Terskler for helsestatus (i timer)
_HEALTHY_THRESHOLD_HOURS: int = 24
_DEGRADED_THRESHOLD_HOURS: int = 48

router = APIRouter(
    prefix="/chains",
    tags=["chains"],
    responses={
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)


# ============================================================================
# PRIVATE HJELPEFUNKSJONER
# ============================================================================


def _determine_chain_status(
    latest_signal_date: Optional[date],
    has_model: bool,
) -> Literal["healthy", "degraded", "unhealthy"]:
    """
    Bestemmer helsestatus for en kjede basert på:
    - alder på siste signal
    - om modell eksisterer

    Returnerer: "healthy" | "degraded" | "unhealthy".
    """
    if latest_signal_date is None or not has_model:
        return "unhealthy"

    # Konverter dato til aware datetime i UTC
    latest_dt = datetime.combine(
        latest_signal_date,
        datetime.min.time(),
        tzinfo=timezone.utc,
    )
    now = datetime.now(timezone.utc)
    signal_age = now - latest_dt

    if signal_age <= timedelta(hours=_HEALTHY_THRESHOLD_HOURS):
        return "healthy"
    if signal_age <= timedelta(hours=_DEGRADED_THRESHOLD_HOURS):
        return "degraded"
    return "unhealthy"


def _build_chain_status_response(
    chain: ChainEnum,
) -> ChainStatusResponse:
    """
    Bygger komplett ChainStatusResponse ved å:
    1. Hente siste signal fra DB.
    2. Hente siste modellperformance fra DB.
    3. Beregne helsestatus.
    """
    # Hent siste signal (aggregerte on-chain-signaler, ingen rådata)
    latest_signal: Optional[Dict[str, Any]] = db_handler.get_latest_signal(chain.value)

    latest_signal_date: Optional[date] = None
    latest_risk_score: Optional[float] = None
    latest_regime: Optional[RegimeEnum] = None

    if latest_signal is not None:
        raw_date = latest_signal.get("date")
        if isinstance(raw_date, datetime):
            latest_signal_date = raw_date.date()
        elif isinstance(raw_date, date):
            latest_signal_date = raw_date
        elif isinstance(raw_date, str):
            try:
                latest_signal_date = datetime.fromisoformat(raw_date).date()
            except ValueError:
                _logger.warning(
                    "Could not parse latest_signal_date for chain %s from value %r",
                    chain.value,
                    raw_date,
                )

        risk_score = latest_signal.get("risk_score")
        latest_risk_score = float(risk_score) if risk_score is not None else None

        regime = latest_signal.get("regime")
        if regime is not None:
            try:
                latest_regime = RegimeEnum(regime)
            except ValueError:
                _logger.warning(
                    "Invalid regime %r for chain %s in latest_signal",
                    regime,
                    chain.value,
                )

    # Hent modell-informasjon (aggregert performance, ingen rådata)
    model_perf: Optional[Dict[str, Any]] = db_handler.get_latest_model_performance(
        chain.value
    )

    has_model = model_perf is not None
    model_version: Optional[str] = None
    last_model_trained_at: Optional[datetime] = None

    if model_perf is not None:
        version = model_perf.get("model_version")
        model_version = str(version) if version is not None else None

        trained_at = model_perf.get("trained_at")
        if isinstance(trained_at, datetime):
            last_model_trained_at = trained_at.astimezone(timezone.utc)
        elif isinstance(trained_at, str):
            try:
                last_model_trained_at = datetime.fromisoformat(trained_at).replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                _logger.warning(
                    "Could not parse trained_at for chain %s from value %r",
                    chain.value,
                    trained_at,
                )

    status_value = _determine_chain_status(
        latest_signal_date=latest_signal_date,
        has_model=has_model,
    )

    return ChainStatusResponse(
        chain=chain,
        latest_signal_date=latest_signal_date,
        latest_risk_score=latest_risk_score,
        latest_regime=latest_regime,
        model_version=model_version,
        last_model_trained_at=last_model_trained_at,
        status=status_value,
    )


# ============================================================================
# ENDPOINTS
# ============================================================================


@router.get(
    "/{chain}/signals",
    response_model=DailySignalResponse,
    summary="Get latest signal for chain",
)
async def get_chain_signals(
    chain: ChainEnum = Path(..., description="Blockchain: BTC or ETH"),
) -> DailySignalResponse:
    """
    Henter siste signal for spesifisert kjede.

    Returnerer en DailySignalResponse med aggregerte on-chain signaler.
    """
    try:
        latest_signal: Optional[Dict[str, Any]] = db_handler.get_latest_signal(
            chain.value
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        _logger.error(
            "Failed to fetch latest signal for chain %s: %s",
            chain.value,
            exc,
        )
        error = ErrorResponse(
            error="FETCH_LATEST_SIGNAL_FAILED",
            detail=f"Failed to fetch latest signal for chain {chain.value}",
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error.model_dump(),
        ) from exc

    if latest_signal is None:
        error = ErrorResponse(
            error="SIGNAL_NOT_FOUND",
            detail=f"No signal found for chain {chain.value}",
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error.model_dump(),
        )

    # Hent alerts for samme dato/kjede (aggregerte alerts, ingen rådata)
    raw_date = latest_signal.get("date")
    signal_date: Optional[date] = None
    if isinstance(raw_date, datetime):
        signal_date = raw_date.date()
    elif isinstance(raw_date, date):
        signal_date = raw_date
    elif isinstance(raw_date, str):
        try:
            signal_date = datetime.fromisoformat(raw_date).date()
        except ValueError:
            _logger.warning(
                "Could not parse signal_date for alerts lookup from value %r",
                raw_date,
            )

    alerts: List[Dict[str, Any]] = []
    if signal_date is not None:
        try:
            # Hvis get_alerts_for_date skulle returnere None, sørger vi for tom liste
            alerts = db_handler.get_alerts_for_date(signal_date, chain.value) or []
        except Exception as exc:  # pragma: no cover - defensive logging
            _logger.error(
                "Failed to fetch alerts for chain %s and date %s: %s",
                chain.value,
                signal_date,
                exc,
            )
            # Ved feil i alerts fortsetter vi uten alerts, for å ikke blokkere signaler
            alerts = []

    try:
        # Deleger til felles konverteringsfunksjon for å unngå duplisering
        return _db_signal_to_response(latest_signal, alerts)
    except ValueError as exc:
        _logger.error(
            "Invalid signal record for chain %s: %s",
            chain.value,
            exc,
        )
        error = ErrorResponse(
            error="INVALID_SIGNAL_RECORD",
            detail="Invalid signal record in database",
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error.model_dump(),
        ) from exc


@router.get(
    "/{chain}/status",
    response_model=ChainStatusResponse,
    summary="Get status for chain",
)
async def get_chain_status(
    chain: ChainEnum = Path(..., description="Blockchain: BTC or ETH"),
) -> ChainStatusResponse:
    """
    Henter helsestatus og metadata for kjede.

    Status baseres på:
    - alder på siste signal
    - om en modellversjon er tilgjengelig
    """
    try:
        return _build_chain_status_response(chain)
    except Exception as exc:  # pragma: no cover - defensive logging
        _logger.error(
            "Failed to build chain status for %s: %s",
            chain.value,
            exc,
        )
        error = ErrorResponse(
            error="CHAIN_STATUS_FAILED",
            detail=f"Failed to compute status for chain {chain.value}",
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error.model_dump(),
        ) from exc


@router.get(
    "",
    response_model=ChainListResponse,
    summary="List all supported chains",
)
async def list_chains() -> ChainListResponse:
    """
    Lister alle støttede kjeder med status.

    Bruker ChainStatusResponse per kjede og projiserer til en kort ChainSummary.
    """
    summaries: List[ChainSummary] = []

    for chain in ChainEnum:
        try:
            status_response = _build_chain_status_response(chain)
        except Exception as exc:  # pragma: no cover - defensive logging
            _logger.error(
                "Failed to build chain status while listing chains for %s: %s",
                chain.value,
                exc,
            )
            # Ved feil for én kjede merker vi den som unhealthy uten å stoppe hele lista.
            summaries.append(
                ChainSummary(
                    chain=chain,
                    latest_signal_date=None,
                    status="unhealthy",
                )
            )
            continue

        summaries.append(
            ChainSummary(
                chain=chain,
                latest_signal_date=status_response.latest_signal_date,
                status=status_response.status,
            )
        )

    return ChainListResponse(
        chains=summaries,
        count=len(summaries),
    )
