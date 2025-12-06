from __future__ import annotations

"""
FastAPI application entry point for ON-CHAIN SUPER SIGNALS™.

This module is responsible for:
- Initializing the FastAPI application.
- Configuring logging, CORS, and metadata.
- Including all API routers under a versioned prefix.
- Defining root and health endpoints.
- Installing global exception handlers.
"""

from datetime import date, datetime, timezone
from typing import Any, Dict, Literal, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.schemas import (
    API_VERSION,
    LEGAL_DISCLAIMER,
    HealthCheckResponse,
    ValidationErrorResponse,
)
from src.api.routes import alerts, chains, signals
from src.utils.logger import get_logger, setup_logging

logger = get_logger("api.main")


def configure_routers(app: FastAPI) -> None:
    """
    Include all versioned API routers under the /api/v1 prefix.

    Routers:
    - signals.router → /api/v1/signals
    - alerts.router  → /api/v1/alerts
    - chains.router  → /api/v1/chains
    """
    app.include_router(signals.router, prefix="/api/v1", tags=["signals"])
    app.include_router(alerts.router, prefix="/api/v1", tags=["alerts"])
    app.include_router(chains.router, prefix="/api/v1", tags=["chains"])


def configure_cors(app: FastAPI) -> None:
    """
    Configure CORS middleware.

    For now, allow all origins, methods, and headers to keep local development
    and dashboard integration frictionless. Production environments should
    constrain allowed origins via environment/config, but this is managed
    outside this file.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def configure_exception_handlers(app: FastAPI) -> None:
    """
    Register global exception handlers for common FastAPI error types.

    - RequestValidationError → ValidationErrorResponse with HTTP 422
    - HTTPException → passthrough with logging
    """

    @app.exception_handler(RequestValidationError)
    async def request_validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        logger.warning(
            "Request validation error on path '%s': %s",
            request.url.path,
            exc.errors(),
        )
        response_model = ValidationErrorResponse(detail=exc.errors())
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=response_model.model_dump(),
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request,
        exc: HTTPException,
    ) -> JSONResponse:
        logger.error(
            "HTTPException raised on path '%s': status=%s, detail=%r",
            request.url.path,
            exc.status_code,
            exc.detail,
        )
        content: Dict[str, Any] = {"detail": exc.detail}
        return JSONResponse(status_code=exc.status_code, content=content)


def get_application() -> FastAPI:
    """
    Application factory for the FastAPI app.

    Responsibilities
    ----------------
    - Initialize logging via setup_logging().
    - Create the FastAPI application instance with correct metadata.
    - Configure CORS.
    - Include routers.
    - Register global exception handlers.

    Returns
    -------
    FastAPI
        A fully configured FastAPI application instance.
    """
    # Ensure centralized logging is configured before anything else.
    setup_logging()

    description = (
        "ON-CHAIN SUPER SIGNALS™ API\n\n"
        f"{LEGAL_DISCLAIMER}"
    )

    app = FastAPI(
        title="ON-CHAIN SUPER SIGNALS API",
        version=API_VERSION,
        description=description,
    )

    configure_cors(app)
    configure_routers(app)
    configure_exception_handlers(app)

    @app.get("/", summary="API root", tags=["meta"])
    async def root() -> Dict[str, str]:
        """
        Simple root endpoint for quick availability checks.

        Returns a lightweight JSON message with the current API version.
        """
        return {
            "message": "Welcome to ON-CHAIN SUPER SIGNALS API",
            "version": API_VERSION,
        }

    @app.get(
        "/health",
        response_model=HealthCheckResponse,
        summary="Health check",
        tags=["meta"],
    )
    async def health() -> HealthCheckResponse:
        """
        Health check endpoint with database connectivity verification.

        Returns HTTP 200 OK with HealthCheckResponse containing:
        - status: "healthy", "degraded", or "unhealthy"
        - timestamp: current UTC time
        - database_connected: whether DuckDB is accessible
        - latest_signal_age_hours: hours since last signal (if available)
        - version: API version string

        Status logic:
        - "healthy": DB connected and signal < 24 hours old
        - "degraded": DB connected but signal 24–48 hours old, or DB OK but no signals yet
        - "unhealthy": DB not connected or signal > 48 hours old
        """
        from src.utils import db_handler

        now_utc = datetime.now(timezone.utc)

        # Default values
        database_connected = False
        latest_signal_age_hours: Optional[float] = None
        status_value: Literal["healthy", "degraded", "unhealthy"] = "unhealthy"

        try:
            # Attempt to get latest signal to verify DB connectivity
            latest_signal = db_handler.get_latest_signal("BTC")
            database_connected = True

            if latest_signal is not None:
                signal_date = latest_signal.get("date")
                signal_dt: Optional[datetime] = None

                if isinstance(signal_date, datetime):
                    signal_dt = signal_date
                elif isinstance(signal_date, date):
                    signal_dt = datetime.combine(
                        signal_date,
                        datetime.min.time(),
                        tzinfo=timezone.utc,
                    )
                elif isinstance(signal_date, str):
                    try:
                        parsed = datetime.fromisoformat(signal_date)
                        if parsed.tzinfo is None:
                            parsed = parsed.replace(tzinfo=timezone.utc)
                        signal_dt = parsed
                    except ValueError:
                        logger.warning(
                            "Health check: could not parse signal_date %r", signal_date
                        )

                if signal_dt is not None:
                    if signal_dt.tzinfo is None:
                        signal_dt = signal_dt.replace(tzinfo=timezone.utc)
                    age = now_utc - signal_dt
                    latest_signal_age_hours = age.total_seconds() / 3600.0

                    # Determine status based on signal age
                    if latest_signal_age_hours <= 24:
                        status_value = "healthy"
                    elif latest_signal_age_hours <= 48:
                        status_value = "degraded"
                    else:
                        status_value = "unhealthy"
                else:
                    # DB connected but we couldn't parse signal date
                    status_value = "degraded"
            else:
                # DB connected but no signals yet
                status_value = "degraded"

        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Health check: database connection failed: %s", exc)
            database_connected = False
            status_value = "unhealthy"

        return HealthCheckResponse(
            status=status_value,
            timestamp=now_utc,
            database_connected=database_connected,
            latest_signal_age_hours=latest_signal_age_hours,
            version=API_VERSION,
        )

    return app


app = get_application()
