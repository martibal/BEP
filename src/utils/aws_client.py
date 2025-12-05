"""
AWS S3 client utilities for the ON-CHAIN SUPER SIGNALS™ project.

This module provides a thin, well-defined integration layer against the
AWS Public Blockchain Data S3 bucket. It is the ONLY allowed source of
raw blockchain data.

Key responsibilities:
- Initialize an unsigned boto3 S3 client against the public bucket
- Fetch daily blocks data for a given chain and date
- Fetch daily transactions data for a given chain and date (with optional limit)
- Check if data exists for a given chain and date without loading it
- Apply robust retry logic with exponential backoff for transient errors

Data minimization:
- All data remains in memory; no raw blockchain data is written to disk.
- Callers are responsible for aggregating and discarding raw data as early as possible.
"""

from __future__ import annotations

import time
from datetime import date
from typing import Any, Optional

import boto3
import pandas as pd
import pyarrow.parquet as pq
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError, ReadTimeoutError

from config.settings import (
    AWS_BUCKET,
    AWS_REGION,
    AWS_NO_SIGN_REQUEST,
    S3_FETCH_TIMEOUT_SEC,
    S3_FETCH_RETRIES,
    get_chain_s3_prefix,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _retry_on_failure(
    func,
    *args: Any,
    max_retries: int = S3_FETCH_RETRIES,
    **kwargs: Any,
) -> Any:
    """
    Retry function on transient S3 errors with exponential backoff.

    Transient errors:
        - EndpointConnectionError
        - ReadTimeoutError
        - Selected ClientError codes (e.g. 503, SlowDown)

    Non-transient errors:
        - 403 / AccessDenied -> PermissionError
        - Corrupt data / other ClientErrors -> raised immediately
        - 404 / NoSuchKey -> treated as "no data", returns None

    Args:
        func: Callable to execute.
        *args: Positional arguments for func.
        max_retries: Maximum number of attempts (default: S3_FETCH_RETRIES).
        **kwargs: Keyword arguments for func.

    Returns:
        Result of func(*args, **kwargs), or None for 404/NoSuchKey.

    Raises:
        PermissionError: On 403/AccessDenied.
        Exception: On unrecoverable errors after retries are exhausted.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except (EndpointConnectionError, ReadTimeoutError) as exc:
            if attempt == max_retries:
                logger.error(
                    "S3 fetch failed after max retries (network/timeout)",
                    extra={
                        "attempts": max_retries,
                        "error": repr(exc),
                    },
                    exc_info=True,
                )
                raise

            backoff = 2 ** (attempt - 1)  # 1s, 2s, 4s, ...
            logger.warning(
                "S3 fetch failed, retrying",
                extra={
                    "attempt": attempt,
                    "backoff_sec": backoff,
                    "error_type": type(exc).__name__,
                },
            )
            time.sleep(backoff)
        except ClientError as exc:
            error = exc.response.get("Error", {}) if hasattr(exc, "response") else {}
            error_code = str(error.get("Code", "Unknown"))

            # 404: No data (not an error; treated as "no data available")
            if error_code in ("404", "NoSuchKey", "NotFound"):
                logger.debug(
                    "S3 object not found",
                    extra={"error_code": error_code},
                )
                return None

            # 403: Permission denied (fatal)
            if error_code in ("403", "AccessDenied"):
                logger.error(
                    "S3 access denied",
                    extra={"error_code": error_code},
                    exc_info=True,
                )
                raise PermissionError("Access denied to S3 bucket") from exc

            # Rate limiting / throttling / transient server errors: retry
            if error_code in ("503", "SlowDown", "RequestTimeout", "Throttling"):
                if attempt == max_retries:
                    logger.error(
                        "S3 fetch failed after max retries (server/ratelimit)",
                        extra={
                            "attempts": max_retries,
                            "error_code": error_code,
                        },
                        exc_info=True,
                    )
                    raise

                backoff = 2 ** (attempt - 1)
                logger.warning(
                    "S3 server/ratelimit error, retrying",
                    extra={
                        "attempt": attempt,
                        "backoff_sec": backoff,
                        "error_code": error_code,
                    },
                )
                time.sleep(backoff)
                continue

            # Other ClientErrors: don't retry, raise immediately
            logger.error(
                "S3 client error (non-retriable)",
                extra={"error_code": error_code},
                exc_info=True,
            )
            raise


class S3Client:
    """
    Low-level AWS S3 client for accessing the AWS Public Blockchain Data bucket.

    The client:
    - Uses unsigned requests (no credentials) against a public bucket.
    - Applies explicit networking timeouts and disables internal retries
      (we manage retries via _retry_on_failure).
    """

    def __init__(self) -> None:
        """
        Initialize the boto3 S3 client with correct configuration.

        Configuration:
            - signature_version = UNSIGNED (for public bucket, if enabled)
            - region_name = AWS_REGION
            - connect_timeout = S3_FETCH_TIMEOUT_SEC
            - read_timeout = S3_FETCH_TIMEOUT_SEC
            - retries = {'max_attempts': 1} (we handle retries ourselves)
        """
        if AWS_NO_SIGN_REQUEST:
            boto_config = Config(
                signature_version=UNSIGNED,
                region_name=AWS_REGION,
                connect_timeout=S3_FETCH_TIMEOUT_SEC,
                read_timeout=S3_FETCH_TIMEOUT_SEC,
                retries={"max_attempts": 1},
            )
        else:
            boto_config = Config(
                region_name=AWS_REGION,
                connect_timeout=S3_FETCH_TIMEOUT_SEC,
                read_timeout=S3_FETCH_TIMEOUT_SEC,
                retries={"max_attempts": 1},
            )

        self._s3_client = boto3.client("s3", config=boto_config)
        self._bucket = AWS_BUCKET

        logger.info(
            "S3 client initialized",
            extra={"bucket": AWS_BUCKET, "region": AWS_REGION},
        )

    # --------------------------------------------------------------------- #
    # Public methods
    # --------------------------------------------------------------------- #

    def fetch_daily_blocks(self, chain: str, target_date: date) -> Optional[pd.DataFrame]:
        """
        Fetch daily blocks data for a given chain and date.

        S3 path structure (AWS Public Blockchain Data):
            s3://aws-public-blockchain/v1.0/btc/blocks/date=YYYY-MM-DD/
            s3://aws-public-blockchain/v1.0/eth/blocks/date=YYYY-MM-DD/

        Args:
            chain: "BTC" or "ETH".
            target_date: Date to fetch blocks for.

        Returns:
            pandas.DataFrame with blocks data, or None if no data exists.

        Raises:
            PermissionError: If access to S3 is denied (403).
            Exception: For other non-retriable or unrecoverable errors.
        """
        prefix = get_chain_s3_prefix(chain)
        date_str = target_date.isoformat()
        s3_prefix = f"{prefix}/blocks/date={date_str}/"

        def _list_objects() -> dict:
            return self._s3_client.list_objects_v2(
                Bucket=self._bucket,
                Prefix=s3_prefix,
                MaxKeys=100,
            )

        response = _retry_on_failure(_list_objects)

        # response == None indicates a 404 / "no such key".
        if response is None:
            logger.warning(
                "No blocks found for chain/date (404)",
                extra={"chain": chain, "date": date_str},
            )
            return None

        contents = response.get("Contents")
        if not contents:
            logger.warning(
                "No blocks found for chain/date",
                extra={"chain": chain, "date": date_str},
            )
            return None

        parquet_keys = [obj["Key"] for obj in contents if obj.get("Key", "").endswith(".parquet")]
        if not parquet_keys:
            logger.warning(
                "No parquet block files found for chain/date",
                extra={"chain": chain, "date": date_str},
            )
            return None

        dfs: list[pd.DataFrame] = []

        for key in parquet_keys:
            s3_url = f"s3://{self._bucket}/{key}"

            def _read_parquet() -> pd.DataFrame:
                table = pq.read_table(s3_url)
                return table.to_pandas()

            df = _retry_on_failure(_read_parquet)
            if df is None:
                # Should be rare for blocks; treat as missing file and continue.
                logger.warning(
                    "Parquet block file not found during read, skipping",
                    extra={"chain": chain, "date": date_str, "key": key},
                )
                continue

            dfs.append(df)

        if not dfs:
            logger.warning(
                "No readable parquet block files for chain/date",
                extra={"chain": chain, "date": date_str},
            )
            return None

        result = pd.concat(dfs, ignore_index=True)

        logger.info(
            "Blocks fetched",
            extra={
                "chain": chain,
                "date": date_str,
                "rows": int(len(result)),
            },
        )

        return result

    def fetch_daily_transactions(
        self,
        chain: str,
        target_date: date,
        limit: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch daily transactions data for a given chain and date.

        WARNING:
            Transaction datasets can be very large (hundreds of thousands to
            millions of rows). The optional `limit` parameter allows callers
            to minimize loaded data for early aggregation (data minimization).

        S3 path structure:
            s3://aws-public-blockchain/v1.0/btc/transactions/date=YYYY-MM-DD/
            s3://aws-public-blockchain/v1.0/eth/transactions/date=YYYY-MM-DD/

        Args:
            chain: "BTC" or "ETH".
            target_date: Date to fetch transactions for.
            limit: Optional maximum number of transactions to return.
                   - None: return all rows.
                   - 0: return an empty DataFrame.
                   - < 0: raises ValueError.

        Returns:
            pandas.DataFrame with transactions data, or None if no data exists.

        Raises:
            ValueError: If limit is negative.
            PermissionError: If access to S3 is denied (403).
            Exception: For other non-retriable or unrecoverable errors.
        """
        if limit is not None and limit < 0:
            raise ValueError(f"limit must be >= 0 or None, got {limit}")

        prefix = get_chain_s3_prefix(chain)
        date_str = target_date.isoformat()
        s3_prefix = f"{prefix}/transactions/date={date_str}/"

        def _list_objects() -> dict:
            return self._s3_client.list_objects_v2(
                Bucket=self._bucket,
                Prefix=s3_prefix,
                MaxKeys=100,
            )

        response = _retry_on_failure(_list_objects)

        if response is None:
            logger.warning(
                "No transactions found for chain/date (404)",
                extra={"chain": chain, "date": date_str},
            )
            return None

        contents = response.get("Contents")
        if not contents:
            logger.warning(
                "No transactions found for chain/date",
                extra={"chain": chain, "date": date_str},
            )
            return None

        parquet_keys = [obj["Key"] for obj in contents if obj.get("Key", "").endswith(".parquet")]
        if not parquet_keys:
            logger.warning(
                "No parquet transaction files found for chain/date",
                extra={"chain": chain, "date": date_str},
            )
            return None

        dfs: list[pd.DataFrame] = []

        for key in parquet_keys:
            s3_url = f"s3://{self._bucket}/{key}"

            def _read_parquet() -> pd.DataFrame:
                table = pq.read_table(s3_url)
                return table.to_pandas()

            df = _retry_on_failure(_read_parquet)
            if df is None:
                logger.warning(
                    "Parquet transaction file not found during read, skipping",
                    extra={"chain": chain, "date": date_str, "key": key},
                )
                continue

            dfs.append(df)

        if not dfs:
            logger.warning(
                "No readable parquet transaction files for chain/date",
                extra={"chain": chain, "date": date_str},
            )
            return None

        result = pd.concat(dfs, ignore_index=True)

        if limit is not None:
            if limit == 0:
                logger.warning(
                    "Transactions limited to 0 rows (empty DataFrame requested)",
                    extra={
                        "chain": chain,
                        "date": date_str,
                        "total_available": int(len(result)),
                    },
                )
                # Preserve schema but drop all rows
                result = result.head(0)
            elif len(result) > limit:
                logger.warning(
                    "Transactions limited",
                    extra={
                        "chain": chain,
                        "date": date_str,
                        "limit": int(limit),
                        "total_available": int(len(result)),
                    },
                )
                result = result.head(limit)

        logger.info(
            "Transactions fetched",
            extra={
                "chain": chain,
                "date": date_str,
                "rows": int(len(result)),
                "limit": limit,
            },
        )

        return result

    def check_data_availability(self, chain: str, target_date: date) -> bool:
        """
        Check if any data exists for a given chain and date without loading it.

        This performs a lightweight list operation against the blocks prefix:
            {prefix}/blocks/date=YYYY-MM-DD/

        Args:
            chain: "BTC" or "ETH".
            target_date: Date to check.

        Returns:
            True if at least one object exists for this chain/date, False otherwise.

        Raises:
            PermissionError: If access to S3 is denied (403).
            Exception: For other non-retriable or unrecoverable errors.
        """
        prefix = get_chain_s3_prefix(chain)
        date_str = target_date.isoformat()
        s3_prefix = f"{prefix}/blocks/date={date_str}/"

        def _list_objects() -> dict:
            return self._s3_client.list_objects_v2(
                Bucket=self._bucket,
                Prefix=s3_prefix,
                MaxKeys=1,
            )

        response = _retry_on_failure(_list_objects)

        if response is None:
            logger.debug(
                "Data availability check: no objects (404)",
                extra={"chain": chain, "date": date_str},
            )
            return False

        contents = response.get("Contents")
        available = bool(contents)

        logger.info(
            "Data availability check",
            extra={
                "chain": chain,
                "date": date_str,
                "available": available,
            },
        )

        return available


# ------------------------------------------------------------------------- #
# Module-level convenience API (singleton pattern)
# ------------------------------------------------------------------------- #

_s3_client_singleton: Optional[S3Client] = None


def get_s3_client() -> S3Client:
    """
    Get or create the module-level S3Client singleton.

    Returns:
        S3Client: Shared instance for the current process.
    """
    global _s3_client_singleton
    if _s3_client_singleton is None:
        _s3_client_singleton = S3Client()
    return _s3_client_singleton


def fetch_blocks_for_date(chain: str, target_date: date) -> Optional[pd.DataFrame]:
    """
    Convenience wrapper to fetch daily blocks for a given chain and date.

    Args:
        chain: "BTC" or "ETH".
        target_date: Date to fetch.

    Returns:
        pandas.DataFrame with blocks data, or None if no data.
    """
    client = get_s3_client()
    return client.fetch_daily_blocks(chain, target_date)


def fetch_transactions_for_date(
    chain: str,
    target_date: date,
    limit: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """
    Convenience wrapper to fetch daily transactions for a given chain and date.

    Args:
        chain: "BTC" or "ETH".
        target_date: Date to fetch.
        limit: Optional maximum number of transactions to return.

    Returns:
        pandas.DataFrame with transactions data, or None if no data.
    """
    client = get_s3_client()
    return client.fetch_daily_transactions(chain, target_date, limit=limit)
