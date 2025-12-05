"""
Global configuration for the ON-CHAIN SUPER SIGNALS™ project.

This module defines:
- Project metadata
- Filesystem paths
- AWS S3 configuration
- Supported blockchain chains
- Logging configuration
- Pipeline configuration
- Helper utilities for S3 prefixes and directory creation

All other modules must import configuration values from here instead of
hardcoding paths or environment-specific settings.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List

# ============================================================================
# 1. PROJECT METADATA
# ============================================================================

PROJECT_NAME: str = "ON-CHAIN SUPER SIGNALS"
VERSION: str = "1.0.0"

# ============================================================================
# 2. BASE PATHS
# ============================================================================

# Determine a sensible default base directory based on the platform,
# but allow overriding via the BEP_BASE_DIR environment variable.
if sys.platform == "win32":
    _DEFAULT_BASE_DIR = "D:/BEP"
else:
    # On non-Windows systems, default to a "BEP" folder in the user's home.
    _DEFAULT_BASE_DIR = str(Path.home() / "BEP")

BASE_DIR: Path = Path(os.getenv("BEP_BASE_DIR", _DEFAULT_BASE_DIR))

# Core project directories (no I/O on import, directories may not exist yet).
DATA_DIR: Path = BASE_DIR / "data"
CONFIG_DIR: Path = BASE_DIR / "config"
SRC_DIR: Path = BASE_DIR / "src"
LOGS_DIR: Path = BASE_DIR / "logs"
TMP_DIR: Path = DATA_DIR / "tmp"

# ============================================================================
# 3. DATABASE PATHS
# ============================================================================

DB_DIR: Path = DATA_DIR / "db"
DB_PATH: Path = DB_DIR / "signals.duckdb"

# ============================================================================
# 4. MODEL & NORMALIZATION PATHS
# ============================================================================

MODEL_DIR: Path = DATA_DIR / "models"
NORMALIZATION_DIR: Path = DATA_DIR / "normalization"

# ============================================================================
# 5. AWS S3 CONFIGURATION
# ============================================================================

# Primary public blockchain dataset (MIT-0 license, commercial use allowed)
AWS_BUCKET: str = "aws-public-blockchain"
AWS_REGION: str = os.getenv("BEP_AWS_REGION", "us-east-1")

# Prefixes for chain-specific data within the public bucket.
AWS_S3_PREFIX_BTC: str = "v1.0/btc"
AWS_S3_PREFIX_ETH: str = "v1.0/eth"

# Public bucket: no signed requests / no credentials required.
AWS_NO_SIGN_REQUEST: bool = True

# ============================================================================
# 6. BLOCKCHAIN CHAINS
# ============================================================================

# Supported chains for ON-CHAIN SUPER SIGNALS™.
CHAINS: List[str] = ["BTC", "ETH"]

# Mapping from chain symbol to an external identifier used for price/metadata
# lookups in compatible data sources. The actual data source must respect
# licensing and allow commercial use of derived data.
CHAIN_TICKERS: Dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
}

# ============================================================================
# 7. LOGGING CONFIGURATION
# ============================================================================

# Allow log level override via environment variable (e.g. DEBUG/INFO/WARNING).
LOG_LEVEL: str = os.getenv("BEP_LOG_LEVEL", "INFO")

LOG_FILE: Path = LOGS_DIR / "app.log"
LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT: int = 5

LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ============================================================================
# 8. FEATURE & MODEL CONFIGURATION PATHS
# ============================================================================

FEATURES_CONFIG_PATH: Path = CONFIG_DIR / "features.yaml"
MODEL_CONFIG_PATH: Path = CONFIG_DIR / "model_config.yaml"

# ============================================================================
# 9. PIPELINE & LOCKFILE CONFIGURATION
# ============================================================================

# All timestamps and scheduling are in UTC.
PIPELINE_TIMEZONE: str = "UTC"

# Lockfile directory: use platform-appropriate temp directory.
if sys.platform == "win32":
    # On Windows, prefer the system TEMP directory, fallback to a sensible default.
    _default_temp = os.getenv("TEMP", "C:/Windows/Temp")
    LOCKFILE_DIR: Path = Path(_default_temp)
else:
    # On Unix-like systems, use /tmp if it exists; otherwise fall back to TMP_DIR.
    _tmp_path = Path("/tmp")
    LOCKFILE_DIR = _tmp_path if _tmp_path.exists() else TMP_DIR

# S3 fetch behavior: timeout and retry policy.
S3_FETCH_TIMEOUT_SEC: int = 30
S3_FETCH_RETRIES: int = 3

# ============================================================================
# 10. HELPER FUNCTIONS
# ============================================================================


def get_chain_s3_prefix(chain: str) -> str:
    """
    Return the S3 prefix for the given blockchain chain.

    Args:
        chain: Chain symbol, e.g. "BTC" or "ETH".

    Returns:
        The S3 prefix string for the specified chain, such as "v1.0/btc".

    Raises:
        ValueError: If the chain is not supported.
    """
    chain_upper = chain.upper()
    if chain_upper == "BTC":
        return AWS_S3_PREFIX_BTC
    if chain_upper == "ETH":
        return AWS_S3_PREFIX_ETH
    raise ValueError(f"Unknown chain: {chain}")


def ensure_directories_exist() -> None:
    """
    Create all required project directories if they do not already exist.

    This function performs filesystem I/O and MUST NOT be called at import time.
    It should be invoked explicitly by setup scripts (e.g. scripts/setup_db.py)
    or deployment tooling before running pipelines.

    Raises:
        PermissionError: If the process lacks permissions to create directories.
        OSError: For other OS-level errors related to directory creation.
    """
    directories = [
        DATA_DIR,
        DB_DIR,
        MODEL_DIR,
        NORMALIZATION_DIR,
        TMP_DIR,
        LOGS_DIR,
        CONFIG_DIR,
        SRC_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
