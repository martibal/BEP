from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, List, Optional, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


__all__ = [
    "API_VERSION",
    "LEGAL_DISCLAIMER",
    "ChainEnum",
    "RegimeEnum",
    "RiskStateEnum",
    "AlertTypeEnum",
    "AlertResponse",
    "DailySignalResponse",
    "SignalHistoryResponse",
    "AlertListResponse",
    "ChainStatusResponse",
    "ChainSummary",
    "ChainListResponse",
    "HealthCheckResponse",
    "SignalHistoryParams",
    "AlertsQueryParams",
    "ErrorResponse",
    "ValidationErrorResponse",
]


API_VERSION: str = "1.0.0"

LEGAL_DISCLAIMER: str = (
    "ON-CHAIN SUPER SIGNALS™ leverer kun automatiserte datasignaler. "
    "Dette er ikke finansielle råd, ikke anbefalinger og ikke oppfordringer "
    "til kjøp eller salg. Produktet distribuerer kun avledede analyser. "
    "Rå blockchain-data distribueres ikke."
)


# =============================================================================
# ENUMS
# =============================================================================


class ChainEnum(str, Enum):
    """
    Støttede blockchain-kjeder.
    """

    BTC = "BTC"
    ETH = "ETH"


class RegimeEnum(str, Enum):
    """
    Market regime klassifisering.
    """

    BULL = "Bull"
    NEUTRAL = "Neutral"
    BEAR = "Bear"


class RiskStateEnum(str, Enum):
    """
    Risk state klassifisering.
    """

    RISK_ON = "Risk ON"
    NEUTRAL = "Neutral"
    RISK_OFF = "Risk OFF"


class AlertTypeEnum(str, Enum):
    """
    Alert-typer som kan genereres.
    """

    WHALE_SPIKE = "whale_spike"
    STRESS_EVENT = "stress_event"
    MOMENTUM_COLLAPSE = "momentum_collapse"
    REGIME_SHIFT = "regime_shift"


# =============================================================================
# RESPONSE MODELS
# =============================================================================


class AlertResponse(BaseModel):
    """
    Én alert-hendelse.
    Matcher DuckDB alerts-tabellen.
    """

    id: int
    chain: ChainEnum
    date: date
    alert_type: AlertTypeEnum
    message: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DailySignalResponse(BaseModel):
    """
    Komplett daglig signal for én kjede.
    Matcher DuckDB daily_signals-tabellen pluss metadata.
    Inkluderer PÅKREVD juridisk disclaimer.
    """

    chain: ChainEnum
    date: date
    risk_score: float = Field(ge=0, le=100)
    regime: RegimeEnum
    risk_state: RiskStateEnum
    whale_index: float = Field(ge=0, le=100)
    stress_index: float = Field(ge=0, le=100)
    momentum_score: float = Field(ge=0, le=100)
    alerts: List[AlertResponse] = Field(default_factory=list)
    disclaimer: str = Field(default=LEGAL_DISCLAIMER)
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class SignalHistoryResponse(BaseModel):
    """
    Historiske signaler for én kjede.
    Brukes av GET /api/v1/signals/history endpoint.
    """

    chain: ChainEnum
    start_date: date
    end_date: date
    signals: List[DailySignalResponse]
    count: int
    disclaimer: str = Field(default=LEGAL_DISCLAIMER)


class AlertListResponse(BaseModel):
    """
    Liste med alerts for en gitt dato/periode.
    Brukes av GET /api/v1/alerts endpoints.
    """

    chain: Optional[ChainEnum] = None
    date: Optional[date] = None
    alerts: List[AlertResponse]
    count: int


class ChainStatusResponse(BaseModel):
    """
    Status for én kjede.
    Brukes av GET /api/v1/chains/{chain} endpoint.
    """

    chain: ChainEnum
    latest_signal_date: Optional[date] = None
    latest_risk_score: Optional[float] = None
    latest_regime: Optional[RegimeEnum] = None
    model_version: Optional[str] = None
    last_model_trained_at: Optional[datetime] = None
    status: Literal["healthy", "degraded", "unhealthy"]


class ChainSummary(BaseModel):
    """
    Kort oppsummering av én kjede for listevisning.

    Brukes av GET /api/v1/chains endpoint for å gi en
    kompakt oversikt over alle støttede kjeder.
    """

    chain: ChainEnum
    latest_signal_date: Optional[date] = None
    status: Literal["healthy", "degraded", "unhealthy"]


class ChainListResponse(BaseModel):
    """
    Liste over alle støttede kjeder med status.

    Brukes av GET /api/v1/chains endpoint.
    """

    chains: List[ChainSummary]
    count: int


class HealthCheckResponse(BaseModel):
    """
    API health check response.
    Brukes av GET /health endpoint.
    """

    status: Literal["healthy", "degraded", "unhealthy"]
    timestamp: datetime
    database_connected: bool
    latest_signal_age_hours: Optional[float] = None
    version: str


# =============================================================================
# REQUEST / QUERY MODELS
# =============================================================================


class SignalHistoryParams(BaseModel):
    """
    Query-parametre for historiske signaler.
    """

    start_date: date
    end_date: date

    @field_validator("start_date")
    @classmethod
    def validate_start_not_in_future(cls, v: date) -> date:
        today = date.today()
        if v > today:
            raise ValueError("start_date cannot be in the future.")
        return v

    @field_validator("end_date")
    @classmethod
    def validate_end_date(cls, v: date, info: ValidationInfo) -> date:
        start = info.data.get("start_date")
        if start and v < start:
            raise ValueError("end_date cannot be before start_date.")
        today = date.today()
        if v > today:
            raise ValueError("end_date cannot be in the future.")
        return v


class AlertsQueryParams(BaseModel):
    """
    Query-parametre for alerts.
    """

    chain: Optional[ChainEnum] = None
    date: Optional[date] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    alert_type: Optional[AlertTypeEnum] = None
    limit: int = Field(default=100, ge=1, le=1000)

    @field_validator("end_date")
    @classmethod
    def validate_end_date(
        cls,
        v: Optional[date],
        info: ValidationInfo,
    ) -> Optional[date]:
        if v is None:
            return v

        start = info.data.get("start_date")
        if start and v < start:
            raise ValueError("end_date cannot be before start_date.")

        today = date.today()
        if v > today:
            raise ValueError("end_date cannot be in the future.")
        return v

    @field_validator("start_date")
    @classmethod
    def validate_start_not_in_future(cls, v: Optional[date]) -> Optional[date]:
        if v is None:
            return v
        today = date.today()
        if v > today:
            raise ValueError("start_date cannot be in the future.")
        return v

    @field_validator("date")
    @classmethod
    def validate_date_not_in_future(cls, v: Optional[date]) -> Optional[date]:
        if v is None:
            return v
        today = date.today()
        if v > today:
            raise ValueError("date cannot be in the future.")
        return v


# =============================================================================
# ERROR MODELS
# =============================================================================


class ErrorResponse(BaseModel):
    """
    Standard feilrespons for alle API-feil.
    """

    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class ValidationErrorResponse(BaseModel):
    """
    Valideringsfeil-respons for Pydantic-feil.
    """

    error: str = "Validation Error"
    details: List[dict[str, Any]]
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
