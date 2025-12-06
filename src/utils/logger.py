"""
Centralized logging setup for the ON-CHAIN SUPER SIGNALS™ project.

This module configures:
- A global rotating file logger (logs/app.log by default)
- UTC timestamps for all log entries
- A simple API for obtaining loggers across the codebase

Usage (in any module):

    from src.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Processing started", extra={"chain": "BTC"})

All other modules MUST obtain loggers via get_logger() and MUST NOT
reconfigure logging on their own.
"""

from __future__ import annotations

import logging
import logging.handlers
import time
from pathlib import Path
from typing import Optional

from src.config.settings import (
    LOG_FILE,
    LOG_LEVEL,
    LOG_MAX_BYTES,
    LOG_BACKUP_COUNT,
    LOG_FORMAT,
    LOGS_DIR,
)

# Flag to ensure logging is configured only once per process.
_logging_configured: bool = False

# Default max bytes fallback (10 MB) if configuration is invalid (<= 0).
_DEFAULT_MAX_BYTES: int = 10 * 1024 * 1024


def setup_logging() -> None:
    """
    Configure the root logger with a rotating file handler.

    This function is idempotent: calling it multiple times in the same
    process will have no additional effect after the first successful call.

    Behavior:
        - Ensures LOGS_DIR exists.
        - Creates a RotatingFileHandler pointing to LOG_FILE.
        - Uses LOG_MAX_BYTES and LOG_BACKUP_COUNT from settings.
        - Falls back to _DEFAULT_MAX_BYTES if LOG_MAX_BYTES is invalid (<= 0).
        - Sets UTC timestamps for all log entries.
        - Sets the log level based on LOG_LEVEL, falling back to INFO on invalid values.

    Raises:
        PermissionError: If the process cannot write to LOGS_DIR.
        OSError: For other filesystem-related errors when creating directories or handlers.
    """
    global _logging_configured

    if _logging_configured:
        return

    # Ensure logs directory exists.
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise PermissionError(f"Cannot write to logs directory: {LOGS_DIR}") from exc
    except OSError:
        # Re-raise other OS-level errors; caller should treat this as fatal.
        raise

    root_logger = logging.getLogger()

    # Determine log level, with graceful fallback.
    level_name = (LOG_LEVEL or "").upper()
    invalid_level = False
    if level_name and hasattr(logging, level_name):
        level_value = getattr(logging, level_name)
        if isinstance(level_value, int):
            log_level = level_value
        else:
            log_level = logging.INFO
            invalid_level = True
    else:
        log_level = logging.INFO
        invalid_level = bool(level_name)  # True if a non-empty but invalid value was provided.

    # Determine maxBytes for rotation.
    effective_max_bytes = LOG_MAX_BYTES
    invalid_max_bytes = False
    try:
        if effective_max_bytes is None or int(effective_max_bytes) <= 0:
            effective_max_bytes = _DEFAULT_MAX_BYTES
            invalid_max_bytes = True
        else:
            effective_max_bytes = int(effective_max_bytes)
    except (TypeError, ValueError):
        effective_max_bytes = _DEFAULT_MAX_BYTES
        invalid_max_bytes = True

    # Configure the rotating file handler.
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(LOG_FILE),
            maxBytes=effective_max_bytes,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
    except PermissionError as exc:
        raise PermissionError(f"Cannot open log file for writing: {LOG_FILE}") from exc
    except OSError:
        # Re-raise any other OS-related errors.
        raise

    formatter = logging.Formatter(
        fmt=LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Ensure timestamps are in UTC.
    formatter.converter = time.gmtime

    file_handler.setFormatter(formatter)

    # Avoid adding duplicate handlers if setup_logging is called more than once
    # in abnormal circumstances.
    if not any(isinstance(h, logging.handlers.RotatingFileHandler) and getattr(h, "baseFilename", None) == file_handler.baseFilename  # type: ignore[attr-defined]
               for h in root_logger.handlers):
        root_logger.addHandler(file_handler)

    root_logger.setLevel(log_level)

    _logging_configured = True

    # Log configuration anomalies now that logging is fully set up.
    logger = root_logger.getChild("onchain_super_signals.logger")
    if invalid_level:
        logger.warning(
            "Invalid LOG_LEVEL '%s' in settings; falling back to INFO.",
            LOG_LEVEL,
        )
    if invalid_max_bytes:
        logger.warning(
            "Invalid LOG_MAX_BYTES '%s' in settings; falling back to default %d bytes.",
            LOG_MAX_BYTES,
            _DEFAULT_MAX_BYTES,
        )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a pre-configured logger with the given name.

    This function guarantees that the global logging configuration has been
    initialized before returning the logger.

    Args:
        name: The logger name, typically __name__ from the calling module.
              If None, the root logger is returned.

    Returns:
        logging.Logger: A configured logger instance.

    Raises:
        PermissionError: If logging cannot be initialized due to filesystem permissions.
        OSError: For other filesystem-related errors during initialization.
    """
    setup_logging()
    if name is None:
        return logging.getLogger()
    return logging.getLogger(str(name))


def add_console_handler(level: str = "DEBUG") -> None:
    """
    Add a console (stdout) handler to the root logger for debugging.

    This is primarily intended for development and manual debugging. In
    production, file logging is typically sufficient.

    The function is safe to call multiple times; it will not add duplicate
    console handlers.

    Args:
        level: The log level for console output (e.g. "DEBUG", "INFO").

    Raises:
        PermissionError: If logging cannot be initialized.
        OSError: If logging initialization fails due to filesystem issues.
    """
    setup_logging()
    root_logger = logging.getLogger()

    # Determine console log level with graceful fallback.
    level_name = (level or "").upper()
    if level_name and hasattr(logging, level_name):
        level_value = getattr(logging, level_name)
        console_level = level_value if isinstance(level_value, int) else logging.DEBUG
    else:
        console_level = logging.DEBUG

    # Check if a StreamHandler already exists to avoid duplicates.
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            # Assume an existing console handler is sufficient.
            return

    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt=LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    formatter.converter = time.gmtime
    console_handler.setFormatter(formatter)
    console_handler.setLevel(console_level)

    root_logger.addHandler(console_handler)
