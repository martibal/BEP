from __future__ import annotations

import logging
import math
from datetime import date
from typing import Any, Mapping

import pandas as pd  # NOTE: Temporary Pandas dependency for aws_client interop; will be removed once DataService emits Polars exclusively. :contentReference[oaicite:0]{index=0}
import polars as pl

from src.features.base import BaseFeature, AggregatedFeatures, RawBlockchainData
from src.utils.logger import get_logger  # Centralized logging configuration. :contentReference[oaicite:1]{index=1}

# BTCFeatureEngine implementation aligned with:
# - Final requirements specification (Styringsdokument BEP). :contentReference[oaicite:2]{index=2}
# - Architecture document (Arkitektur Claude). :contentReference[oaicite:3]{index=3}
# - BTC feature engine review feedback. :contentReference[oaicite:4]{index=4}


class BTCFeatureEngine(BaseFeature):
    """
    BTC-specific feature engine.

    Responsibilities:
    - Convert incoming raw blockchain data (potentially in Pandas) to Polars.
      (Temporary responsibility until DataService standardizes on Polars.)
    - Compute a minimal set of BTC daily on-chain features using Polars.
    - Return a single aggregated feature vector (dict[str, float]) per day.

    The minimal feature set (BTC_*) covers:
    - Total block count                → structural activity / security
    - Total transaction count          → chain usage / throughput
    - Total output value (BTC)         → volume / whale activity proxy
    - Mean fee rate (sat/vbyte)        → fee pressure / mempool stress proxy
    - Active addresses count (proxy)   → on-chain user activity
    """

    def __init__(self) -> None:
        super().__init__(chain="BTC")
        # Use centralized logger utility to ensure consistent formatting & config.
        self._logger = get_logger(f"{__name__}.BTCFeatureEngine")

    def compute(self, raw_data: RawBlockchainData, target_date: date) -> AggregatedFeatures:
        """
        Compute BTC daily aggregated features for the given date.

        Parameters
        ----------
        raw_data:
            RawBlockchainData containing at least the keys:
            - "blocks": block-level data (Polars or Pandas DataFrame)
            - "transactions": transaction-level data (Polars or Pandas DataFrame)
            DataService is responsible for providing a consistent mapping.
        target_date:
            The date these aggregates correspond to. Currently not used for filtering
            (DataService is expected to provide already date-scoped data) but included
            for future chain/date-specific logic.

        Returns
        -------
        AggregatedFeatures:
            A dict[str, float] with a single feature-vector for the given date.
        """
        if not isinstance(raw_data, Mapping):
            raise TypeError(
                "BTCFeatureEngine.compute expected raw_data to be a Mapping "
                "with 'blocks' and 'transactions' keys."
            )

        self._logger.debug("Starting BTC feature computation for %s", target_date)

        blocks_pl = self._to_polars(raw_data.get("blocks"), source="blocks")
        tx_pl = self._to_polars(raw_data.get("transactions"), source="transactions")

        # Ensure non-None Polars DataFrames
        if blocks_pl is None:
            self._logger.warning(
                "Blocks raw_data was None for %s; using empty Polars DataFrame.", target_date
            )
            blocks_pl = pl.DataFrame()

        if tx_pl is None:
            self._logger.warning(
                "Transactions raw_data was None for %s; using empty Polars DataFrame.", target_date
            )
            tx_pl = pl.DataFrame()

        if blocks_pl.is_empty() and tx_pl.is_empty():
            self._logger.warning(
                "Both blocks and transactions DataFrames are empty for %s; "
                "returning NaN features to signal missing raw data.",
                target_date,
            )

        # Minimal feature set (all float values, names prefixed with 'BTC_')
        features: AggregatedFeatures = {
            "BTC_total_block_count": self._compute_total_block_count(blocks_pl),
            "BTC_total_transactions_count": self._compute_total_transactions_count(tx_pl),
            "BTC_total_output_value_btc": self._compute_total_output_value_btc(tx_pl),
            "BTC_mean_transaction_fee_rate_sat_per_vbyte": self._compute_mean_fee_rate_sat_per_vbyte(
                tx_pl
            ),
            "BTC_active_addresses_count": self._compute_active_addresses_count(tx_pl),
        }

        cleaned_features: AggregatedFeatures = {}
        for name, value in features.items():
            cleaned_features[name] = self._sanitize_feature_value(name, value)

        self._logger.debug(
            "Completed BTC feature computation for %s with features: %s",
            target_date,
            cleaned_features,
        )

        # Explicitly drop references to potentially large raw DataFrames to aid GC
        # and reinforce that no raw data should be retained beyond this method. :contentReference[oaicite:5]{index=5}
        del blocks_pl, tx_pl

        return cleaned_features

    # -------------------------------------------------------------------------
    # Polars conversion helpers (temporary until DataService standardizes on Polars)
    # -------------------------------------------------------------------------

    def _to_polars(self, df_like: Any, source: str) -> pl.DataFrame | None:
        """
        Convert incoming DataFrame-like object to Polars DataFrame.

        Handles:
        - None → returns None
        - pd.DataFrame → pl.from_pandas(...)
        - pl.DataFrame → returned as-is

        Any other type results in a warning and an empty Polars DataFrame.

        NOTE: Per architectural review, Pandas-to-Polars conversion should
        eventually move to DataService or a dedicated RawDataConverter
        once upstream modules are refactored. :contentReference[oaicite:6]{index=6}
        """
        if df_like is None:
            return None

        if isinstance(df_like, pl.DataFrame):
            self._logger.debug("Received Polars DataFrame for %s.", source)
            return df_like

        if isinstance(df_like, pd.DataFrame):
            self._logger.debug("Converting Pandas DataFrame to Polars for %s.", source)
            # Using from_pandas ensures we move all heavy computation to Polars.
            return pl.from_pandas(df_like)

        self._logger.warning(
            "Unexpected data type for %s: %s. Using empty Polars DataFrame.",
            source,
            type(df_like),
        )
        return pl.DataFrame()

    # -------------------------------------------------------------------------
    # Feature computation helpers
    # -------------------------------------------------------------------------

    def _compute_total_transactions_count(self, tx_pl: pl.DataFrame) -> float:
        """
        Total number of transactions for the day.

        For missing/empty datasets, returns NaN to signal missing raw data
        rather than "zero activity". :contentReference[oaicite:7]{index=7}
        """
        if tx_pl is None or tx_pl.is_empty():
            self._logger.warning(
                "Transaction DataFrame is empty or None; "
                "setting total_transactions_count=NaN to signal missing data."
            )
            return float("nan")

        count = float(tx_pl.height)
        self._logger.debug("Computed total_transactions_count=%s", count)
        return count

    def _compute_total_block_count(self, blocks_pl: pl.DataFrame) -> float:
        """
        Total number of blocks for the day.

        For missing/empty datasets, returns NaN to signal missing raw data
        rather than "zero activity". :contentReference[oaicite:8]{index=8}
        """
        if blocks_pl is None or blocks_pl.is_empty():
            self._logger.warning(
                "Blocks DataFrame is empty or None; "
                "setting total_block_count=NaN to signal missing data."
            )
            return float("nan")

        count = float(blocks_pl.height)
        self._logger.debug("Computed total_block_count=%s", count)
        return count

    def _compute_mean_block_size_bytes(self, blocks_pl: pl.DataFrame) -> float:
        """
        Mean block size in bytes.

        Attempts to infer a reasonable size column from common names:
        - "size", "block_size", "block_size_bytes"

        If no appropriate column exists or dataset is empty, returns NaN.
        """
        if blocks_pl.is_empty():
            self._logger.info(
                "Blocks DataFrame is empty; setting mean_block_size_bytes=NaN (no blocks)."
            )
            return float("nan")

        size_col = self._find_first_matching_column(
            blocks_pl,
            candidates=("size", "block_size", "block_size_bytes"),
        )
        if size_col is None:
            self._logger.warning(
                "No block size column found in blocks DataFrame; "
                "setting mean_block_size_bytes=NaN."
            )
            return float("nan")

        try:
            result = blocks_pl.select(pl.col(size_col).cast(pl.Float64).mean()).to_dict(
                as_series=False
            )
            mean_value = result.get(size_col, [float("nan")])[0]
            mean_float = float(mean_value)
            self._logger.debug(
                "Computed mean_block_size_bytes=%s using column '%s'", mean_float, size_col
            )
            return mean_float
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failed to compute mean_block_size_bytes from column '%s': %s",
                size_col,
                exc,
            )
            return float("nan")

    def _compute_mean_fee_rate_sat_per_vbyte(self, tx_pl: pl.DataFrame) -> float:
        """
        Mean transaction fee rate in satoshis per virtual byte (sat/vbyte).

        Attempts to infer:
        - Fee column from: "fee_sat", "fee_satoshi", "fee", "total_fee_sat"
        - Virtual size column from: "vsize", "virtual_size", "vbytes", "size_vbytes"

        Filters out rows where vbytes <= 0 to avoid division by zero.

        If necessary columns are missing or dataset is empty, returns NaN.
        """
        if tx_pl.is_empty():
            self._logger.info(
                "Transaction DataFrame is empty; "
                "setting mean_transaction_fee_rate_sat_per_vbyte=NaN."
            )
            return float("nan")

        fee_col = self._find_first_matching_column(
            tx_pl,
            candidates=("fee_sat", "fee_satoshi", "fee", "total_fee_sat"),
        )
        vbytes_col = self._find_first_matching_column(
            tx_pl,
            candidates=("vsize", "virtual_size", "vbytes", "size_vbytes"),
        )

        if fee_col is None or vbytes_col is None:
            self._logger.warning(
                "Missing required columns for fee rate computation "
                "(fee_col=%s, vbytes_col=%s); returning NaN.",
                fee_col,
                vbytes_col,
            )
            return float("nan")

        try:
            fee_expr = pl.col(fee_col).cast(pl.Float64)
            vbytes_expr = pl.col(vbytes_col).cast(pl.Float64)

            fee_df = (
                tx_pl.select(
                    [
                        fee_expr.alias("fee_sat"),
                        vbytes_expr.alias("vbytes"),
                    ]
                )
                .filter(pl.col("vbytes") > 0)
                .with_columns(
                    (pl.col("fee_sat") / pl.col("vbytes")).alias("fee_sat_per_vbyte")
                )
            )

            if fee_df.is_empty():
                self._logger.info(
                    "No valid rows for fee rate (after vbytes>0 filter); returning NaN."
                )
                return float("nan")

            mean_val = fee_df.select(pl.col("fee_sat_per_vbyte").mean()).to_dict(
                as_series=False
            )["fee_sat_per_vbyte"][0]
            mean_float = float(mean_val)
            self._logger.debug(
                "Computed mean_transaction_fee_rate_sat_per_vbyte=%s using fee_col='%s', "
                "vbytes_col='%s'",
                mean_float,
                fee_col,
                vbytes_col,
            )
            return mean_float
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failed to compute mean_transaction_fee_rate_sat_per_vbyte: %s", exc)
            return float("nan")

    def _compute_total_output_value_btc(self, tx_pl: pl.DataFrame) -> float:
        """
        Total output value for the day, in BTC.

        Attempts to infer an output value column from:
        - BTC-denominated: "output_value_btc", "value_btc", "out_value_btc", "output_total_btc"
        - Satoshi-denominated: "output_value_sat", "value_sat", "out_value_sat", "output_total_sat"
          (converted to BTC via / 1e8)

        For missing/empty datasets, returns NaN to signal missing raw data.
        """
        if tx_pl.is_empty():
            self._logger.warning(
                "Transaction DataFrame is empty; setting total_output_value_btc=NaN."
            )
            return float("nan")

        btc_col = self._find_first_matching_column(
            tx_pl,
            candidates=("output_value_btc", "value_btc", "out_value_btc", "output_total_btc"),
        )
        sat_col = self._find_first_matching_column(
            tx_pl,
            candidates=("output_value_sat", "value_sat", "out_value_sat", "output_total_sat"),
        )

        if btc_col is None and sat_col is None:
            self._logger.warning(
                "No suitable output value column found for total_output_value_btc; returning NaN."
            )
            return float("nan")

        try:
            if btc_col is not None:
                expr = pl.col(btc_col).cast(pl.Float64)
            else:
                # Fallback: use satoshi column and convert to BTC
                expr = (pl.col(sat_col).cast(pl.Float64) / 1e8).alias("output_value_btc")

            result = tx_pl.select(expr.sum()).to_dict(as_series=False)
            # Extract the single sum value regardless of the column name
            sum_value = next(iter(result.values()))[0]
            sum_float = float(sum_value)
            self._logger.debug(
                "Computed total_output_value_btc=%s using %s column.",
                sum_float,
                "btc" if btc_col is not None else f"sat ({sat_col} / 1e8)",
            )
            return sum_float
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failed to compute total_output_value_btc: %s", exc)
            return float("nan")

    def _compute_active_addresses_count(self, tx_pl: pl.DataFrame) -> float:
        """
        Proxy for active addresses: number of unique sender addresses
        observed in the transaction dataset for the day. :contentReference[oaicite:9]{index=9}

        Column candidates for sender address include (case-insensitive):
        - "sender"
        - "from_address"
        - "address_from"
        - "input_address"
        - "spending_address"

        Returns NaN if no suitable column is present or dataset is empty.
        """
        if tx_pl.is_empty():
            self._logger.warning(
                "Transaction DataFrame is empty; setting active_addresses_count=NaN."
            )
            return float("nan")

        addr_col = self._find_first_matching_column(
            tx_pl,
            candidates=(
                "sender",
                "from_address",
                "address_from",
                "input_address",
                "spending_address",
            ),
        )
        if addr_col is None:
            self._logger.warning(
                "No sender address column found in transactions DataFrame; "
                "setting active_addresses_count=NaN."
            )
            return float("nan")

        try:
            # Filter out null addresses before counting uniques
            active_df = tx_pl.filter(pl.col(addr_col).is_not_null())
            if active_df.is_empty():
                self._logger.info(
                    "Sender address column '%s' contains no non-null values; "
                    "setting active_addresses_count=NaN.",
                    addr_col,
                )
                return float("nan")

            n_unique_df = active_df.select(pl.col(addr_col).n_unique())
            # The resulting DataFrame has a single column with the same name
            n_unique_val = n_unique_df.to_dict(as_series=False)[addr_col][0]
            active_count = float(n_unique_val)
            self._logger.debug(
                "Computed active_addresses_count=%s using sender column '%s'.",
                active_count,
                addr_col,
            )
            return active_count
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failed to compute active_addresses_count from column '%s': %s",
                addr_col,
                exc,
            )
            return float("nan")

    # -------------------------------------------------------------------------
    # Utility helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _find_first_matching_column(
        df: pl.DataFrame,
        candidates: tuple[str, ...],
    ) -> str | None:
        """
        Find the first column in `df` whose lowercase name matches one of `candidates`.

        Returns the actual column name (with original casing) or None if none found.
        """
        if df.is_empty() and not df.columns:
            return None

        lower_to_actual: dict[str, str] = {col.lower(): col for col in df.columns}
        for candidate in candidates:
            actual = lower_to_actual.get(candidate.lower())
            if actual is not None:
                return actual
        return None

    def _sanitize_feature_value(self, name: str, value: float | int | None) -> float:
        """
        Ensure feature values are valid floats.

        - Converts ints to float.
        - For None, logs and returns NaN.
        - For non-finite numbers (NaN/inf), logs and returns NaN.

        DataService is responsible for downstream validation, but this method
        aims to keep outputs well-formed.
        """
        if value is None:
            self._logger.warning("Feature '%s' is None; converting to NaN.", name)
            return float("nan")

        try:
            as_float = float(value)
        except (TypeError, ValueError):
            self._logger.warning(
                "Feature '%s' has non-numeric value %r; converting to NaN.", name, value
            )
            return float("nan")

        if not math.isfinite(as_float):
            self._logger.warning(
                "Feature '%s' computed as non-finite (%s); converting to NaN.", name, as_float
            )
            return float("nan")

        return as_float
