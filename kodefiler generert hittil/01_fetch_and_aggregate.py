"""
pipelines/01_fetch_and_aggregate.py

Daily pipeline step 1:
- Determine date range to process
- For each date and chain, fetch minimal raw data from AWS Public Blockchain Data (S3)
- Aggregate to daily metrics
- Store daily aggregates to local tmp/ as intermediate data for later steps

This script:
- Respects data minimization (no raw blockchain data persisted locally)
- Uses only AWS Public Blockchain Data as raw source
- Writes only daily aggregated data to tmp/
- Is designed to be invoked by cron:

    01 01 * * *  python3 /app/01_fetch_and_aggregate.py

Architecture & requirements references:
- See governing documents and architecture spec :contentReference[oaicite:0]{index=0}
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, List, Optional, Tuple

from config.settings import CHAINS  # type: ignore

from src.services.data_service import DataService  # type: ignore
from src.utils import validators  # type: ignore
from src.utils.logger import get_logger  # type: ignore

# Note: DataService is responsible for:
# - Connecting to AWS Public Blockchain Data (S3)
# - Fetching minimal required raw data (in-memory)
# - Fetching daily close prices from AWS Public Blockchain-compatible sources
# - Aggregating to daily metrics (including price columns)
# - Storing ONLY daily aggregated results to tmp/
#
# This script orchestrates those steps and ensures that:
# - Only needed dates are processed
# - No raw blockchain data is written to disk
# - Basic logging and error handling is applied


logger = get_logger(__name__)

LOCKFILE_PATH = "/tmp/01_fetch_and_aggregate.lock"

FailedItem = Tuple[date, str]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse CLI arguments.

    Supported modes:
    - Default (no args): process "yesterday" in UTC.
    - --date YYYY-MM-DD: process exactly that date.
    - --start-date / --end-date: process inclusive date range.
    """
    parser = argparse.ArgumentParser(
        description="Daily fetch & aggregate pipeline for ON-CHAIN SUPER SIGNALS™"
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Process a single UTC date (YYYY-MM-DD). "
        "Mutually exclusive with --start-date/--end-date.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (UTC, YYYY-MM-DD) for backfill/range runs.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date (UTC, YYYY-MM-DD) for backfill/range runs. "
        "Defaults to yesterday if omitted and --start-date is provided.",
    )

    args = parser.parse_args(argv)

    # Basic mutual exclusivity check
    if args.date and (args.start_date or args.end_date):
        parser.error("--date cannot be combined with --start-date/--end-date")

    return args


def _parse_iso_date(date_str: str) -> date:
    """Parse an ISO date string or raise ValueError with a clear message."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError as exc:  # pragma: no cover - trivial
        raise ValueError(f"Invalid date format '{date_str}', expected YYYY-MM-DD") from exc


def determine_date_range(args: argparse.Namespace) -> List[date]:
    """
    Determine which UTC dates to process.

    Rules:
    - If --date is provided: process that single date.
    - If --start-date (and optionally --end-date) is provided: process inclusive range.
      If --end-date is omitted, default to yesterday UTC.
    - If no args: default to yesterday UTC.

    Note: We intentionally do NOT introspect local storage for "latest processed date"
    to avoid introducing extra persistent state. Backfill logic is handled via
    explicit CLI arguments or separate scripts.

    Future dates (after yesterday UTC) are rejected to avoid attempting to
    process data that cannot yet exist in AWS Public Blockchain Data.
    """
    utc_today = datetime.now(timezone.utc).date()
    yesterday = utc_today - timedelta(days=1)

    if args.date:
        target = _parse_iso_date(args.date)
        if target > yesterday:
            raise ValueError(f"Cannot process future dates: {target}")
        validators.validate_date_range(target, target)
        return [target]

    if args.start_date:
        start = _parse_iso_date(args.start_date)
        if args.end_date:
            end = _parse_iso_date(args.end_date)
        else:
            end = yesterday

        if start > yesterday or end > yesterday:
            raise ValueError(
                f"Cannot process future dates in range: {start} to {end} (yesterday is {yesterday})"
            )

        validators.validate_date_range(start, end)
        # Inclusive range
        num_days = (end - start).days + 1
        if num_days <= 0:
            raise ValueError("End date must be on or after start date")

        return [start + timedelta(days=i) for i in range(num_days)]

    # Default: only yesterday
    validators.validate_date_range(yesterday, yesterday)
    return [yesterday]


def process_date_for_chain(
    data_service: DataService,
    chain: str,
    target_date: date,
) -> None:
    """
    Fetch and aggregate data for a single chain and date.

    Steps:
    1. Validate chain name.
    2. Fetch minimal raw data for (chain, date) from S3 via DataService.
    3. Fetch daily close price from AWS Public Blockchain-compatible data via DataService.
    4. Aggregate to daily metrics via DataService.
    5. Validate aggregated daily metrics.
    6. Store aggregated daily metrics to tmp/ via DataService.

    Data minimization:
    - Raw blockchain data is never persisted to disk by this function.
    - Only daily aggregated results are passed to store_to_tmp().
    - If aggregation yields no rows, a placeholder row with an is_valid flag
      is written so that downstream steps can detect the anomaly.
    """
    validators.validate_chain(chain)

    logger.info("Fetching raw data from S3", extra={"chain": chain, "date": str(target_date)})

    # Step 1: Fetch minimal raw data from AWS Public Blockchain Data
    fetch_start = time.time()
    raw_data = data_service.fetch_latest_blockchain_data(chain=chain, date=target_date)
    fetch_duration = time.time() - fetch_start

    if raw_data is None:
        logger.warning(
            "No raw data returned from S3 for given date; writing placeholder",
            extra={
                "chain": chain,
                "date": str(target_date),
                "duration_sec": round(fetch_duration, 2),
            },
        )
        # Placeholder so downstream pipelines can detect missing data for this day/chain
        data_service.store_placeholder_for_missing_day(chain=chain, date=target_date)
        return

    # We deliberately avoid logging contents of raw_data to preserve data minimization.
    try:
        length = len(raw_data)  # type: ignore[assignment]
    except TypeError:
        length = -1  # length unknown; don't fail just due to this

    logger.info(
        "Raw data fetched from S3",
        extra={
            "chain": chain,
            "date": str(target_date),
            "records": length,
            "duration_sec": round(fetch_duration, 2),
        },
    )

    # Step 2: Fetch daily price (from AWS Public Blockchain-compatible data sources)
    price_fetch_start = time.time()
    daily_price = data_service.fetch_daily_price(chain=chain, date=target_date)
    price_fetch_duration = time.time() - price_fetch_start

    logger.info(
        "Daily price fetched",
        extra={
            "chain": chain,
            "date": str(target_date),
            "has_price": daily_price is not None,
            "duration_sec": round(price_fetch_duration, 2),
        },
    )

    # Explicit handling of missing price data
    if daily_price is None:
        logger.error(
            "No price data available; writing placeholder for day",
            extra={"chain": chain, "date": str(target_date)},
        )
        data_service.store_placeholder_for_missing_day(chain=chain, date=target_date)
        return

    # Step 3: Aggregate in-memory to daily metrics (including price)
    logger.info(
        "Aggregating raw data to daily metrics",
        extra={"chain": chain, "date": str(target_date)},
    )
    agg_start = time.time()
    daily_agg = data_service.aggregate_to_daily(
        raw_data=raw_data,
        chain=chain,
        date=target_date,
        daily_price=daily_price,
    )
    agg_duration = time.time() - agg_start

    # Again, avoid logging full aggregated content; only basic metadata.
    if hasattr(daily_agg, "__len__"):
        agg_len = len(daily_agg)  # type: ignore[assignment]
    else:
        agg_len = -1

    if agg_len == 0:
        logger.warning(
            "Aggregation produced no rows; writing placeholder for invalid day",
            extra={
                "chain": chain,
                "date": str(target_date),
                "duration_sec": round(agg_duration, 2),
            },
        )
        data_service.store_placeholder_for_missing_day(chain=chain, date=target_date)
        return

    logger.info(
        "Daily aggregation complete",
        extra={
            "chain": chain,
            "date": str(target_date),
            "rows": agg_len,
            "duration_sec": round(agg_duration, 2),
        },
    )

    # Step 4: Validate aggregated daily metrics
    validators.validate_daily_aggregates(daily_agg, expected_date=target_date)

    # Step 5: Store aggregated result to tmp/ as intermediate data
    logger.info(
        "Storing aggregated daily data to tmp/",
        extra={"chain": chain, "date": str(target_date)},
    )
    store_start = time.time()
    data_service.store_to_tmp(daily_agg)
    store_duration = time.time() - store_start

    logger.info(
        "Successfully processed chain/date",
        extra={
            "chain": chain,
            "date": str(target_date),
            "store_duration_sec": round(store_duration, 2),
        },
    )


def run_pipeline(
    dates: Iterable[date],
    data_service: Optional[DataService] = None,
) -> Tuple[bool, List[FailedItem]]:
    """
    Run the fetch & aggregate pipeline for all configured chains and the given dates.

    Returns:
        (all_success, failures):
            all_success: True if all (date, chain) combinations succeeded.
            failures: list of (date, chain) pairs that failed.
    """
    if data_service is None:
        data_service = DataService()

    failures: List[FailedItem] = []

    for target_date in dates:
        logger.info("Starting processing for date", extra={"date": str(target_date)})
        for chain in CHAINS:
            try:
                process_date_for_chain(data_service, chain, target_date)
            except Exception as exc:
                logger.error(
                    "Error while processing chain/date",
                    extra={"chain": chain, "date": str(target_date), "error": repr(exc)},
                    exc_info=True,
                )
                failures.append((target_date, chain))
                # Continue with other chains and dates

    all_success = len(failures) == 0
    if not all_success:
        logger.warning(
            "One or more chain/date combinations failed",
            extra={"num_failed": len(failures)},
        )

    return all_success, failures


def _is_pid_running(pid: int) -> bool:
    """
    Check if a PID appears to be running on this system.

    Uses os.kill(pid, 0) which does not actually send a signal but
    performs error checking.
    """
    try:
        # Signal 0 does not actually kill the process but will raise if it doesn't exist.
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # We don't have permission to signal this PID, but it exists.
        return True
    else:
        return True


def _acquire_lock() -> bool:
    """
    Acquire a simple PID-based lock to avoid overlapping runs.

    Returns:
        True if the lock was acquired, False otherwise.
    """
    try:
        # O_EXCL ensures we fail if the file already exists.
        fd = os.open(LOCKFILE_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as lockfile:
            lockfile.write(str(os.getpid()))
        logger.info("Acquired pipeline lock", extra={"lockfile": LOCKFILE_PATH})
        return True
    except FileExistsError:
        # Detect and handle stale lockfiles.
        try:
            with open(LOCKFILE_PATH, "r") as lockfile:
                content = lockfile.read().strip()
            pid = int(content)
        except Exception as exc:
            logger.warning(
                "Lockfile exists but PID is invalid; treating as stale",
                extra={"lockfile": LOCKFILE_PATH, "content": content if "content" in locals() else None},
                exc_info=True,
            )
            try:
                os.remove(LOCKFILE_PATH)
                logger.info(
                    "Removed stale lockfile with invalid PID",
                    extra={"lockfile": LOCKFILE_PATH},
                )
                # Retry once after removing stale lock
                return _acquire_lock()
            except Exception as remove_exc:
                logger.error(
                    "Failed to remove stale lockfile",
                    extra={"lockfile": LOCKFILE_PATH, "error": repr(remove_exc)},
                    exc_info=True,
                )
                return False

        if not _is_pid_running(pid):
            logger.warning(
                "Stale lockfile detected (PID not running); removing",
                extra={"lockfile": LOCKFILE_PATH, "pid": pid},
            )
            try:
                os.remove(LOCKFILE_PATH)
                logger.info(
                    "Removed stale lockfile",
                    extra={"lockfile": LOCKFILE_PATH, "pid": pid},
                )
                # Retry once after removing stale lock
                return _acquire_lock()
            except Exception as remove_exc:
                logger.error(
                    "Failed to remove stale lockfile",
                    extra={"lockfile": LOCKFILE_PATH, "error": repr(remove_exc)},
                    exc_info=True,
                )
                return False

        logger.error(
            "Lockfile already exists; another instance may be running",
            extra={"lockfile": LOCKFILE_PATH, "pid": pid},
        )
        return False
    except Exception as exc:
        logger.error(
            "Failed to acquire lockfile",
            extra={"lockfile": LOCKFILE_PATH, "error": repr(exc)},
            exc_info=True,
        )
        return False


def _release_lock() -> None:
    """Release the PID-based lock if it exists."""
    try:
        if os.path.exists(LOCKFILE_PATH):
            os.remove(LOCKFILE_PATH)
            logger.info("Released pipeline lock", extra={"lockfile": LOCKFILE_PATH})
    except Exception as exc:
        logger.error(
            "Failed to remove lockfile",
            extra={"lockfile": LOCKFILE_PATH, "error": repr(exc)},
            exc_info=True,
        )


def main(argv: Optional[List[str]] = None) -> int:
    """
    Entry point for the daily fetch & aggregate pipeline.

    Returns:
        int: Exit code
            0 on full success,
            2 on partial failure (some chains/dates failed),
            1 on unrecoverable error.
    """
    logger.info("01_fetch_and_aggregate pipeline started")

    if not _acquire_lock():
        # Another instance is likely running or lock could not be acquired.
        return 1

    pipeline_start = time.time()
    data_service = DataService()

    try:
        args = parse_args(argv)
        dates = determine_date_range(args)
        logger.info(
            "Determined date range to process",
            extra={
                "start_date": str(min(dates)),
                "end_date": str(max(dates)),
                "num_days": len(dates),
            },
        )
        all_success, failures = run_pipeline(dates, data_service=data_service)
    except Exception as exc:
        logger.error(
            "Pipeline failed with an unrecoverable error",
            extra={"error": repr(exc)},
            exc_info=True,
        )
        try:
            logger.info("Cleaning up tmp/ after failure")
            data_service.cleanup_tmp()
        except Exception as cleanup_exc:
            logger.error(
                "Cleanup after failure also failed",
                extra={"error": repr(cleanup_exc)},
                exc_info=True,
            )
        total_duration = round(time.time() - pipeline_start, 2)
        logger.info(
            "01_fetch_and_aggregate pipeline failed",
            extra={"total_duration_sec": total_duration},
        )
        return 1
    finally:
        _release_lock()

    total_duration = round(time.time() - pipeline_start, 2)

    if not all_success:
        # Note: tmp/ is NOT cleaned up on partial failure.
        # This allows downstream pipelines to use successfully processed chains.
        # The 04_cleanup.py script is responsible for full cleanup after all
        # pipeline steps (including inference) have completed.
        logger.error(
            "01_fetch_and_aggregate completed with partial failures",
            extra={
                "failed": [f"{d.isoformat()}:{c}" for (d, c) in failures],
                "total_duration_sec": total_duration,
            },
        )
        return 2

    logger.info(
        "01_fetch_and_aggregate pipeline completed successfully",
        extra={"total_duration_sec": total_duration},
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - direct execution
    sys.exit(main())
