from __future__ import annotations

"""
FastAPI application entry point for ON-CHAIN SUPER SIGNALS™.

This module is responsible for:
- Initializing the FastAPI application.
- Configuring logging, CORS, and metadata.
- Including all API routers under a versioned prefix.
- Defining root and health endpoints.
- Installing global exception handlers.

Specification reference:
- API main specification (Claude prompt 30). :contentReference[oaicite:0]{index=0}
- Endelig kravspesifikasjon (Styringsdokument BEP). :contentReference[oaicite:1]{index=1}
- Arkitektur-dokument (prosjektstruktur og ansvar). :contentReference[oaicite:2]{index=2}
"""

from datetime import datetime, timezone
from typing import Any, Dict

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
from src.api.routes import alerts, signals
from src.utils.logger import get_logger, setup_logging

logger = get_logger("api.main")


def configure_routers(app: FastAPI) -> None:
    """
    Include all versioned API routers under the /api/v1 prefix.

    Routers:
    - signals.router → /api/v1/signals
    - alerts.router  → /api/v1/alerts
    """
    app.include_router(signals.router, prefix="/api/v1", tags=["signals"])
    app.include_router(alerts.router, prefix="/api/v1", tags=["alerts"])


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
        Lightweight health check endpoint.

        Returns HTTP 200 OK with HealthCheckResponse containing:
        - status: always "OK"
        - version: API version string
        - timestamp: current UTC time
        """
        now_utc = datetime.now(timezone.utc)
        return HealthCheckResponse(
            status="OK",
            version=API_VERSION,
            timestamp=now_utc,
        )

    return app


app = get_application()
