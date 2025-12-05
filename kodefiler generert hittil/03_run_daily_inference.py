"""
pipelines/03_run_daily_inference.py

Main script for running the daily inference pipeline for the
ON-CHAIN SUPER SIGNALS™ project.

Responsibilities
----------------
- Determine which chains and date to process.
- Instantiate service-layer components (ModelService, SignalService).
- Call SignalService to:
  - Load aggregated daily features from tmp/
  - Run model inference to obtain risk scores
  - Generate complete daily signals (regime, risk state, indices, alerts)
  - Persist signals to DuckDB
- Log overall success/failure status.

Data minimization
-----------------
- This script NEVER reads or writes raw blockchain data.
- It delegates all data access to service/utility layers.
- Only aggregated daily features (from tmp/) and derived signals (DuckDB)
  are touched indirectly via those layers.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from typing import List, Sequence

from config.settings import CHAINS, INFERENCE_DATE_OFFSET_DAYS
from src.services.model_service import ModelService
from src.services.signal_service import SignalService, FeatureFileNotFoundError
from src.utils import validators
from src.utils.logger import get_logger

_logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Argument parsing                                                            #
# --------------------------------------------------------------------------- #


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the daily inference pipeline.

    Parameters
    ----------
    argv:
        Optional explicit sequence of arguments (primarily for testing).
        If None, argparse will read from sys.argv.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes:
        - date: Optional[str] (ISO date "YYYY-MM-DD")
        - chains: Optional[List[str]] (comma-separated chain list)
    """
    parser = argparse.ArgumentParser(
        description="Run daily inference and signal generation for ON-CHAIN SUPER SIGNALS™"
    )

    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help=(
            "Target date for inference in ISO format (YYYY-MM-DD). "
            f"If omitted, defaults to (today - {INFERENCE_DATE_OFFSET_DAYS} days) in UTC."
        ),
    )

    parser.add_argument(
        "--chains",
        type=str,
        default=None,
        help=(
            "Comma-separated list of chains to process (e.g. 'BTC,ETH'). "
            f"If omitted, all configured chains are processed: {','.join(CHAINS)}."
        ),
    )

    return parser.parse_args(list(argv) if argv is not None else None)


# --------------------------------------------------------------------------- #
# Core orchestration logic                                                    #
# --------------------------------------------------------------------------- #


def _resolve_target_date(date_str: str | None) -> date:
    """
    Resolve the target date for inference.

    If date_str is None, defaults to (today - INFERENCE_DATE_OFFSET_DAYS) in UTC.
    The resulting date is validated centrally via validators.

    Parameters
    ----------
    date_str:
        Optional ISO date string ("YYYY-MM-DD").

    Returns
    -------
    date
        The resolved target date.

    Raises
    ------
    ValueError
        If the provided date_str is invalid or in the future.
    """
    if date_str is None:
        # Default to an offset relative to current UTC date.
        today_utc = datetime.utcnow().date()
        target = today_utc - timedelta(days=INFERENCE_DATE_OFFSET_DAYS)
        _logger.info(
            "No --date provided; defaulting to offset from today",
            extra={
                "today_utc": str(today_utc),
                "target_date": str(target),
                "offset_days": INFERENCE_DATE_OFFSET_DAYS,
            },
        )
        return target

    try:
        parsed = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError as exc:
        _logger.error(
            "Invalid date format; expected YYYY-MM-DD",
            extra={"date_str": date_str},
        )
        raise ValueError(
            f"Invalid date format: {date_str!r}. Expected YYYY-MM-DD."
        ) from exc

    try:
        # Delegate "not in future" validation to shared utility.
        validators.validate_date_not_in_future(parsed)
    except ValueError as exc:
        today_utc = datetime.utcnow().date()
        _logger.error(
            "Target date is in the future",
            extra={
                "target_date": str(parsed),
                "today_utc": str(today_utc),
            },
        )
        raise

    return parsed


def _resolve_chains(chains_arg: str | None) -> List[str]:
    """
    Resolve which chains should be processed.

    Parameters
    ----------
    chains_arg:
        Optional comma-separated string of chain names.

    Returns
    -------
    List[str]
        Uppercase chain identifiers to process.

    Raises
    ------
    ValueError
        If any requested chain is not supported.
    """
    if chains_arg is None:
        _logger.info(
            "No --chains provided; defaulting to all configured chains",
            extra={"chains": CHAINS},
        )
        return list(CHAINS)

    raw_chains = [c.strip() for c in chains_arg.split(",") if c.strip()]
    if not raw_chains:
        _logger.error(
            "Empty --chains argument after parsing",
            extra={"chains_arg": chains_arg},
        )
        raise ValueError("At least one chain must be specified in --chains.")

    chains: List[str] = []
    for chain in raw_chains:
        chain_upper = chain.upper()
        # Centralized chain validation (includes membership in CHAINS).
        validators.validate_chain(chain_upper)
        chains.append(chain_upper)

    _logger.info("Resolved chains for processing", extra={"chains": chains})
    return chains


def _run_inference_for_chain(
    signal_service: SignalService,
    chain: str,
    target_date: date,
) -> None:
    """
    Run daily inference and signal generation for a single chain and date.

    Parameters
    ----------
    signal_service:
        Configured SignalService instance.
    chain:
        Blockchain chain identifier (e.g., "BTC", "ETH").
    target_date:
        Target date for which to generate the signal.

    Raises
    ------
    RuntimeError
        If SignalService reports a non-feature-related error during processing.
    FeatureFileNotFoundError
        If required feature files for the given chain/date are missing.
    """
    _logger.info(
        "Starting daily inference for chain",
        extra={"chain": chain, "target_date": target_date.isoformat()},
    )

    try:
        signal_service.generate_and_store_daily_signal(
            chain=chain,
            target_date=target_date,
        )
    except FeatureFileNotFoundError as exc:
        # Upstream fetch/aggregate pipeline (01_fetch_and_aggregate.py) likely failed.
        _logger.error(
            "Missing feature file for chain/date; "
            "check 01_fetch_and_aggregate.py for failures.",
            extra={
                "chain": chain,
                "target_date": target_date.isoformat(),
                "error": str(exc),
            },
        )
        raise
    except RuntimeError:
        # SignalService is expected to log detailed error information.
        _logger.error(
            "SignalService reported a failure for chain",
            extra={"chain": chain, "target_date": target_date.isoformat()},
        )
        raise

    _logger.info(
        "Completed daily inference for chain",
        extra={"chain": chain, "target_date": target_date.isoformat()},
    )


def main(argv: Sequence[str] | None = None) -> int:
    """
    Entry point for the daily inference pipeline.

    This function is intentionally "thin" and focused on orchestration:
    - Parse CLI arguments
    - Resolve target date and chains
    - Instantiate service objects
    - Loop over chains and invoke SignalService

    Parameters
    ----------
    argv:
        Optional explicit arguments for testability. If None, uses sys.argv.

    Returns
    -------
    int
        Exit code (0 for success, 1 if any chain failed).
    """
    try:
        args = _parse_args(argv)
        target_date = _resolve_target_date(args.date)
        chains = _resolve_chains(args.chains)
    except ValueError as exc:
        _logger.error("Argument resolution failed", extra={"error": str(exc)})
        return 1

    _logger.info(
        "Starting daily inference pipeline",
        extra={
            "target_date": target_date.isoformat(),
            "chains": chains,
        },
    )

    # Instantiate service-layer components.
    model_service = ModelService()
    signal_service = SignalService(
        model_service=model_service,
    )

    failed_chains: List[str] = []

    for chain in chains:
        try:
            _run_inference_for_chain(signal_service, chain, target_date)
        except (RuntimeError, FeatureFileNotFoundError):
            failed_chains.append(chain)
            # Continue with remaining chains; failure is reflected in exit code.

    if failed_chains:
        _logger.error(
            "Daily inference pipeline completed with failures",
            extra={
                "failed_chains": failed_chains,
                "target_date": target_date.isoformat(),
            },
        )
        return 1

    _logger.info(
        "Daily inference pipeline completed successfully for all chains",
        extra={"chains": chains, "target_date": target_date.isoformat()},
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
