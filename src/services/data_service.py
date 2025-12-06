"""
Data service layer for the ON-CHAIN SUPER SIGNALS™ project.

This module orchestrates:
- Fetching raw blockchain data for a single day from AWS S3
- Delegating daily feature computation to chain-specific feature engines
- Validating aggregated features
- Writing daily aggregates to the temporary filesystem (tmp/)
- Cleaning up temporary data

Strict data minimization:
- Raw blockchain data is NEVER written to disk from this layer.
- Only daily aggregated feature data is written as Parquet to tmp/.
"""

from __future__ import annotations

import time
from datetime import date
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type

import polars as pl

from src.config.settings import CHAINS, TMP_DIR
from src.features.base import AggregatedFeatures, BaseFeature, RawBlockchainData
from src.utils import aws_client, validators
from src.utils.logger import get_logger


class DataService:
    """
    Orchestrates daily data fetching, feature computation, validation,
    and storage of aggregated features for a single chain and date.

    Responsibilities
    ----------------
    - Fetch raw blocks/transactions for a given chain/date from AWS S3.
    - Delegate feature computation to the appropriate FeatureEngine.
    - Validate the resulting daily feature vector.
    - Persist only aggregated features to tmp/ as Parquet.
    - Provide helpers for loading aggregates and cleaning tmp/.
    """

    def __init__(self) -> None:
        """
        Initialize the DataService.

        Responsibilities:
        - Initialize logger.
        - Dynamically import chain-specific feature engine classes
          (BTCFeatureEngine, ETHFeatureEngine) for later instantiation.
        """
        self._logger = get_logger(self.__class__.__name__)
        self._logger.info("Initializing DataService")

        # Mapping from chain symbol (e.g., "BTC") to FeatureEngine class.
        self._feature_engines: Dict[str, Type[BaseFeature]] = {}

        # S3 retry configuration (exponential backoff).
        self._max_s3_retries: int = 3
        self._s3_backoff_base: float = 2.0

        self._register_feature_engines()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _register_feature_engines(self) -> None:
        """
        Dynamically import and register chain-specific feature engines.

        The mapping is intentionally explicit to avoid accidental imports
        and to make it clear which chains are supported.
        """
        engine_specs = {
            "BTC": ("src.features.btc_features", "BTCFeatureEngine"),
            "ETH": ("src.features.eth_features", "ETHFeatureEngine"),
        }

        for chain, (module_path, class_name) in engine_specs.items():
            try:
                module = import_module(module_path)
                engine_cls = getattr(module, class_name)
                if not issubclass(engine_cls, BaseFeature):
                    raise TypeError(
                        f"{class_name} from {module_path} does not inherit from BaseFeature"
                    )
                self._feature_engines[chain] = engine_cls
                self._logger.info(
                    "Registered feature engine",
                    extra={"chain": chain, "class": f"{module_path}.{class_name}"},
                )
            except (ImportError, AttributeError, TypeError) as exc:
                self._logger.error(
                    "Failed to register feature engine",
                    extra={
                        "chain": chain,
                        "module": module_path,
                        "class": class_name,
                        "error": repr(exc),
                    },
                )

    def _get_feature_engine(self, chain: str) -> BaseFeature:
        """
        Return an instance of the chain-specific feature engine.

        Args:
            chain:
                Chain symbol (case-insensitive), e.g. "BTC" or "ETH".

        Returns:
            An instantiated BaseFeature subclass for the given chain.

        Raises:
            ValueError:
                If the chain is not supported or the engine is not registered.
        """
        chain_upper = chain.upper()
        engine_cls = self._feature_engines.get(chain_upper)

        if engine_cls is None:
            self._logger.error(
                "No feature engine registered for chain",
                extra={"chain": chain_upper},
            )
            raise ValueError(f"Unsupported or unregistered chain: {chain}")

        return engine_cls(chain=chain_upper)

    def _get_temp_path(self, chain: str, target_date: date) -> Path:
        """
        Build the standardized temporary Parquet path for a chain/date.

        Convention:
            TMP_DIR / "{YYYY-MM-DD}_{CHAIN}_features.parquet"

        Example:
            2025-12-04_BTC_features.parquet
        """
        chain_upper = chain.upper()
        date_str = target_date.isoformat()

        tmp_root = Path(TMP_DIR)
        return tmp_root / f"{date_str}_{chain_upper}_features.parquet"

    def _fetch_with_retry(
        self,
        fetch_fn: Callable[..., Any],
        description: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Fetch wrapper with simple exponential backoff retry logic.

        Parameters
        ----------
        fetch_fn:
            Callable S3 fetch function (e.g. aws_client.fetch_blocks_for_date).
        description:
            Human-readable description for logging ("blocks", "transactions").
        *args, **kwargs:
            Positional and keyword arguments passed to `fetch_fn`.

        Returns
        -------
        Any
            Result of `fetch_fn` on success, or None after final failure.
        """
        for attempt in range(1, self._max_s3_retries + 1):
            try:
                return fetch_fn(*args, **kwargs)
            except Exception as exc:
                if attempt >= self._max_s3_retries:
                    self._logger.error(
                        "Failed to fetch %s from S3 after retries",
                        description,
                        extra={"attempts": attempt, "error": repr(exc)},
                    )
                    return None

                self._logger.warning(
                    "Error fetching %s from S3; retrying",
                    description,
                    extra={
                        "attempt": attempt,
                        "max_retries": self._max_s3_retries,
                        "error": repr(exc),
                    },
                )
                # Exponential backoff: base^(attempt-1) seconds.
                sleep_seconds = self._s3_backoff_base ** (attempt - 1)
                time.sleep(sleep_seconds)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def aggregate_to_daily(
        self,
        chain: str,
        target_date: date,
        raw_data: RawBlockchainData,
    ) -> AggregatedFeatures:
        """
        Compute daily aggregated features from raw blockchain data.

        This is a thin wrapper around the chain-specific feature engine's
        compute(...) method to reflect the architecture's explicit
        aggregate_to_daily(raw_data) step.

        Parameters
        ----------
        chain:
            Chain symbol (case-insensitive), e.g. "BTC" or "ETH".
        target_date:
            Target calendar date for aggregation.
        raw_data:
            Mapping with at least "blocks" and "transactions" entries.

        Returns
        -------
        AggregatedFeatures
            Mapping of feature name → float.
        """
        chain_upper = chain.upper()
        date_str = target_date.isoformat()

        engine = self._get_feature_engine(chain_upper)

        self._logger.info(
            "Computing daily features via aggregate_to_daily",
            extra={"chain": chain_upper, "date": date_str},
        )

        try:
            features: AggregatedFeatures = engine.compute(raw_data, target_date)
        except ValueError as exc:
            self._logger.error(
                "Feature engine returned invalid data",
                extra={"chain": chain_upper, "date": date_str, "error": repr(exc)},
            )
            raise

        return features

    def store_to_tmp(
        self,
        chain: str,
        target_date: date,
        features: AggregatedFeatures,
    ) -> Path:
        """
        Persist aggregated features as a single-row Parquet file in tmp/.

        This implements the architecture-specified store_to_tmp(data)
        behavior as a separate, testable unit.

        The write is done atomically:
        - First write to a temporary .tmp file.
        - Then rename to the final Parquet path.

        Parameters
        ----------
        chain:
            Chain symbol (case-insensitive), e.g. "BTC" or "ETH".
        target_date:
            Target calendar date.
        features:
            Mapping of feature name → float to persist.

        Returns
        -------
        Path
            Path to the final Parquet file.
        """
        chain_upper = chain.upper()
        tmp_path = self._get_temp_path(chain_upper, target_date)
        tmp_path.parent.mkdir(parents=True, exist_ok=True)

        df = pl.DataFrame([features])

        tmp_path_tmp = tmp_path.with_suffix(tmp_path.suffix + ".tmp")

        try:
            df.write_parquet(tmp_path_tmp)
            tmp_path_tmp.replace(tmp_path)
        finally:
            # Ensure no stale .tmp file is left behind on failure.
            if tmp_path_tmp.exists() and not tmp_path_tmp.samefile(tmp_path):
                try:
                    tmp_path_tmp.unlink()
                except OSError as exc:
                    self._logger.error(
                        "Failed to delete temporary Parquet file after write failure",
                        extra={"path": str(tmp_path_tmp), "error": repr(exc)},
                    )

        self._logger.info(
            "Daily aggregates stored to tmp (atomic write)",
            extra={
                "chain": chain_upper,
                "date": target_date.isoformat(),
                "path": str(tmp_path),
            },
        )

        return tmp_path

    def fetch_and_aggregate(self, chain: str, target_date: date) -> AggregatedFeatures:
        """
        Fetch raw data for a chain/date, compute features, validate and persist.

        Steps:
        1. Validate chain name and target date (must not be in the future).
        2. Fetch raw blocks/transactions from AWS S3 via aws_client, with retry.
        3. Delegate feature computation to the appropriate FeatureEngine
           via aggregate_to_daily().
        4. Validate the resulting AggregatedFeatures.
        5. Write a single-row Parquet file to tmp/ via store_to_tmp().
        6. Return the AggregatedFeatures mapping.

        Data minimization:
        - Raw blocks/transactions remain in memory only.
        - Only aggregated features are written to disk.
        """
        validators.validate_chain(chain)
        chain_upper = chain.upper()

        if chain_upper not in CHAINS:
            raise ValueError(f"Unsupported chain: {chain}")

        if target_date > date.today():
            raise ValueError(
                f"target_date {target_date.isoformat()} cannot be in the future"
            )

        date_str = target_date.isoformat()

        # 1. Fetch raw blocks/transactions with retry.
        self._logger.info(
            "Fetching raw blockchain data from S3",
            extra={"chain": chain_upper, "date": date_str},
        )

        blocks_df = self._fetch_with_retry(
            aws_client.fetch_blocks_for_date,
            "blocks",
            chain_upper,
            target_date,
        )
        tx_df = self._fetch_with_retry(
            aws_client.fetch_transactions_for_date,
            "transactions",
            chain_upper,
            target_date,
        )

        # Normalize empty DataFrames to None and log.
        if blocks_df is not None:
            try:
                if len(blocks_df) == 0:
                    self._logger.warning(
                        "Empty blocks DataFrame returned from S3",
                        extra={"chain": chain_upper, "date": date_str},
                    )
                    blocks_df = None
            except TypeError:
                # If len(...) is not supported, leave as-is.
                pass

        if tx_df is not None:
            try:
                if len(tx_df) == 0:
                    self._logger.warning(
                        "Empty transactions DataFrame returned from S3",
                        extra={"chain": chain_upper, "date": date_str},
                    )
                    tx_df = None
            except TypeError:
                pass

        # If both sources failed or are empty, fail fast with a clear error.
        if blocks_df is None and tx_df is None:
            self._logger.error(
                "Both block and transaction fetches failed or returned empty data",
                extra={"chain": chain_upper, "date": date_str},
            )
            raise RuntimeError(
                f"Unable to fetch any raw data for {chain_upper} on {date_str}"
            )

        raw_data: RawBlockchainData = {
            "blocks": blocks_df,
            "transactions": tx_df,
        }

        # 2. Delegate to chain-specific feature engine via aggregate_to_daily.
        features: AggregatedFeatures = self.aggregate_to_daily(
            chain_upper, target_date, raw_data
        )

        # 3. Validate aggregated features.
        validators.validate_daily_aggregates(features, target_date)

        # 4. Persist aggregated features as a single-row Parquet in tmp/.
        self.store_to_tmp(chain_upper, target_date, features)

        return features

    def load_daily_aggregates(
        self,
        chain: str,
        target_date: date,
    ) -> Optional[AggregatedFeatures]:
        """
        Load daily aggregates for a chain/date from tmp/.

        Behavior:
        - If the Parquet file does not exist, returns None.
        - If the file exists but is empty, returns None.
        - On success, returns a Dict[str, float] with feature values.
        """
        validators.validate_chain(chain)
        chain_upper = chain.upper()
        if chain_upper not in CHAINS:
            raise ValueError(f"Unsupported chain: {chain}")

        tmp_path = self._get_temp_path(chain_upper, target_date)

        if not tmp_path.is_file():
            self._logger.warning(
                "No daily aggregates file found",
                extra={"chain": chain_upper, "date": target_date.isoformat()},
            )
            return None

        try:
            df = pl.read_parquet(tmp_path)
        except (OSError, pl.exceptions.PolarsError) as exc:
            self._logger.error(
                "Failed to read daily aggregates Parquet",
                extra={
                    "chain": chain_upper,
                    "date": target_date.isoformat(),
                    "path": str(tmp_path),
                    "error": repr(exc),
                },
            )
            raise

        if df.height == 0:
            self._logger.warning(
                "Daily aggregates file is empty",
                extra={"chain": chain_upper, "date": target_date.isoformat()},
            )
            return None

        row = df.to_dicts()[0]
        features: AggregatedFeatures = {}

        for key, value in row.items():
            try:
                features[str(key)] = float(value) if value is not None else float("nan")
            except (TypeError, ValueError):
                self._logger.warning(
                    "Non-numeric value in aggregates; coercing to NaN",
                    extra={
                        "chain": chain_upper,
                        "date": target_date.isoformat(),
                        "feature": str(key),
                        "value": repr(value),
                    },
                )
                features[str(key)] = float("nan")

        return features

    def cleanup_tmp(self, target_date: Optional[date] = None) -> None:
        """
        Delete temporary aggregation files in TMP_DIR.

        To avoid deleting historical aggregates that might still be useful
        if the pipeline is re-run, this method only removes files whose
        filename starts with the given target_date (or today's date if
        omitted), following the convention:

            "{YYYY-MM-DD}_{CHAIN}_features.parquet"

        Directories are left untouched.
        """
        tmp_root = Path(TMP_DIR)

        if not tmp_root.exists():
            self._logger.info(
                "TMP_DIR does not exist; nothing to clean up",
                extra={"tmp_dir": str(tmp_root)},
            )
            return

        if target_date is None:
            target_date = date.today()
        date_prefix = target_date.isoformat()

        deleted_files = 0

        for path in tmp_root.iterdir():
            if not path.is_file():
                continue

            name = path.name

            # Only delete files that match today's (or specified) date prefix.
            if not name.startswith(date_prefix):
                continue

            try:
                path.unlink()
                deleted_files += 1
            except OSError as exc:
                self._logger.error(
                    "Failed to delete temporary file",
                    extra={"path": str(path), "error": repr(exc)},
                )

        self._logger.info(
            "Temporary files cleanup completed",
            extra={
                "tmp_dir": str(tmp_root),
                "files_deleted": deleted_files,
                "date_prefix": date_prefix,
            },
        )
