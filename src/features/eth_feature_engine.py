"""
ETH-specific feature engine for the ON-CHAIN SUPER SIGNALS™ project.

This module implements the Ethereum (ETH) feature computation logic as
specified in the architecture and requirements documents. It computes
daily, aggregated on-chain features from in-memory raw blockchain data.

Data minimization:
- Consumes only in-memory raw data (blocks/transactions and optional
  helper frames) provided by the DataService (backed by AWS Public
  Blockchain Data on S3).
- Does NOT write any raw blockchain data to disk.
- Returns only daily aggregated feature values for persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Set, Tuple

import polars as pl

from src.features.base import (
    AggregatedFeatures,
    BaseFeature,
    RawBlockchainData,
)
from src.utils.logger import get_logger


# Heuristic constants derived from review feedback and architecture.
_WEI_PER_ETH: float = 1e18
_WEI_PER_GWEI: float = 1e9
# Whale threshold in ETH (heuristic, may be tuned via config later).
_WHALE_THRESHOLD_ETH: float = 1_000.0
# Beacon chain deposit contract address (canonical, on-chain constant).
_BEACON_DEPOSIT_CONTRACT: str = "0x00000000219ab540356cbb839cbe05303d7705fa"


@dataclass(frozen=True)
class _ETHFeatureNames:
    """
    Container for canonical ETH feature names.

    These names MUST align with:
    - config/features.yaml (eth section)
    - Downstream model and signal-generation logic

    Canonical ETH features:
        - whale_flow_1d
        - exchange_netflow_7d
        - gas_price_median
        - active_addresses
        - smart_contract_calls
        - eth2_staking_flow
    """

    WHALE_FLOW_1D: str = "whale_flow_1d"
    EXCHANGE_NETFLOW_7D: str = "exchange_netflow_7d"
    GAS_PRICE_MEDIAN: str = "gas_price_median"
    ACTIVE_ADDRESSES: str = "active_addresses"
    SMART_CONTRACT_CALLS: str = "smart_contract_calls"
    ETH2_STAKING_FLOW: str = "eth2_staking_flow"


_FEATURE_NAMES = _ETHFeatureNames()


class ETHFeatureEngine(BaseFeature):
    """
    Ethereum-specific feature engine.

    Responsibilities:
    - Accept raw ETH blocks/transactions for a single day.
    - Compute daily aggregated features defined in _ETHFeatureNames.
    - Return a FeatureMapping (dict[str, float]) suitable for persistence.

    Input structure (raw_data):
        {
            "blocks": pl.DataFrame | None,
            "transactions": pl.DataFrame | None,
            # Optional helper frames (if provided by DataService):
            # "staking": pl.DataFrame | None,
            # "supply": pl.DataFrame | None,
        }

    Edge cases:
    - If raw_data is missing or empty, all features are returned as NaN.
    - If required columns for a specific feature are missing, that feature
      is set to NaN, and a log entry is written (ERROR for core columns).
    """

    def __init__(
        self,
        chain: str = "ETH",
        exchange_netflow_window_days: Optional[int] = None,
    ) -> None:
        """
        Initialize ETHFeatureEngine.

        Args:
            chain:
                Chain symbol; forced to "ETH" internally.
            exchange_netflow_window_days:
                Retained for backward compatibility but NOT used for
                raw-data rolling windows. The feature engine now computes
                only the current day's exchange netflow, in line with the
                review feedback. The 7-day rolling aggregation is the
                responsibility of downstream services.
        """
        # Enforce ETH chain explicitly, regardless of caller input.
        super().__init__(chain="ETH")
        self.logger = get_logger(self.__class__.__name__)

        # Pre-compute expected feature name set.
        self._expected_feature_names: Set[str] = {
            _FEATURE_NAMES.WHALE_FLOW_1D,
            _FEATURE_NAMES.EXCHANGE_NETFLOW_7D,
            _FEATURE_NAMES.GAS_PRICE_MEDIAN,
            _FEATURE_NAMES.ACTIVE_ADDRESSES,
            _FEATURE_NAMES.SMART_CONTRACT_CALLS,
            _FEATURE_NAMES.ETH2_STAKING_FLOW,
        }

        # Keep attribute for compatibility; no longer used for raw 7D windows.
        self._exchange_netflow_window_days: Optional[int] = (
            int(exchange_netflow_window_days)
            if exchange_netflow_window_days is not None
            else None
        )

    # ------------------------------------------------------------------ #
    # Contract required by architecture / BaseFeature
    # ------------------------------------------------------------------ #

    @property
    def expected_feature_names(self) -> Set[str]:
        """
        Return the set of feature names produced by this engine.

        This property is used by higher-level validation logic to ensure
        consistency between configuration, training data, and runtime
        feature vectors.
        """
        return self._expected_feature_names

    def get_feature_names(self) -> List[str]:
        """
        Implementation of BaseFeature.get_feature_names.

        Returns:
            A sorted list of canonical feature names.
        """
        return sorted(self._expected_feature_names)

    # ------------------------------------------------------------------ #
    # Public API (abstract method implementation)
    # ------------------------------------------------------------------ #

    def compute(
        self,
        raw_data: RawBlockchainData,
        target_date: date,
        context: Optional[Dict[str, Any]] = None,
    ) -> AggregatedFeatures:
        """
        Compute all ETH features for a single day.

        Args:
            raw_data:
                Dict-like structure with keys:
                    - "blocks": pl.DataFrame or None
                    - "transactions": pl.DataFrame or None
                    - Optional helper frames (ignored if missing):
                        * "staking"
                        * "supply"
            target_date:
                The date these features correspond to.
            context:
                Optional context dict for the day, may contain:
                    - "eth_price_usd": float
                      Daily ETH price in USD used for USD-based features
                      (e.g., whale_flow_1d). If omitted, ETH-based
                      units are used instead.

        Returns:
            Dict[str, float]: A mapping from feature name to value.

        Behavior:
            - If raw_data is missing or empty, returns all-NaN feature mapping.
            - Any feature with missing input columns is set to NaN.
            - No raw data is written to disk; all computation is in-memory.
        """
        self.logger.info(
            "Computing ETH features",
            extra={"date": target_date.isoformat()},
        )

        blocks_df, tx_df, staking_df, supply_df = self._extract_dataframes(raw_data)

        if (
            (blocks_df is None or blocks_df.is_empty())
            and (tx_df is None or tx_df.is_empty())
            and (staking_df is None or staking_df.is_empty())
            and (supply_df is None or supply_df.is_empty())
        ):
            self.logger.warning(
                "Empty raw ETH data for date; returning all-NaN features",
                extra={"date": target_date.isoformat()},
            )
            return self._nan_feature_mapping()

        eth_price_usd: Optional[float] = None
        if context is not None and "eth_price_usd" in context:
            try:
                eth_price_usd = float(context["eth_price_usd"])
            except (TypeError, ValueError):
                self.logger.error(
                    "Invalid eth_price_usd in context; ignoring price",
                    extra={"value": context.get("eth_price_usd")},
                )
                eth_price_usd = None

        # Compute each feature individually. All helpers must be deterministic
        # and MUST NOT write to disk or retain references beyond this call.
        features: Dict[str, float] = {
            _FEATURE_NAMES.WHALE_FLOW_1D: self._compute_whale_flow_1d(
                tx_df, eth_price_usd
            ),
            _FEATURE_NAMES.EXCHANGE_NETFLOW_7D: self._compute_exchange_netflow_7d(
                tx_df
            ),
            _FEATURE_NAMES.GAS_PRICE_MEDIAN: self._compute_gas_price_median(
                blocks_df, tx_df
            ),
            _FEATURE_NAMES.ACTIVE_ADDRESSES: self._compute_active_addresses(tx_df),
            _FEATURE_NAMES.SMART_CONTRACT_CALLS: self._compute_smart_contract_calls(
                tx_df
            ),
            _FEATURE_NAMES.ETH2_STAKING_FLOW: self._compute_eth2_staking_flow(tx_df),
        }

        # Ensure all expected features are present; if any are missing, fill with NaN.
        for name in self._expected_feature_names:
            if name not in features:
                self.logger.warning(
                    "Feature missing from computation; filling with NaN",
                    extra={"feature": name, "date": target_date.isoformat()},
                )
                features[name] = float("nan")

        self.logger.info(
            "ETH features computed",
            extra={
                "date": target_date.isoformat(),
                "num_features": len(features),
            },
        )

        return features

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _extract_dataframes(
        self,
        raw_data: RawBlockchainData,
    ) -> Tuple[
        Optional[pl.DataFrame],
        Optional[pl.DataFrame],
        Optional[pl.DataFrame],
        Optional[pl.DataFrame],
    ]:
        """
        Extract blocks, transactions and optional helper DataFrames
        from raw_data safely.

        Expects:
            raw_data: dict with optional keys:
                - 'blocks'
                - 'transactions'
                - 'staking'
                - 'supply'

        Returns:
            (blocks_df, tx_df, staking_df, supply_df), where each may be None.

        Raises:
            TypeError: If raw_data is not a dict-like structure.
        """
        if not isinstance(raw_data, dict):
            self.logger.error(
                "ETHFeatureEngine.compute expects raw_data as dict",
                extra={"type": type(raw_data).__name__},
            )
            raise TypeError(
                "ETHFeatureEngine.compute expects raw_data as Dict[str, Any] "
                "containing at least 'blocks' and 'transactions' keys"
            )

        def _as_df_or_none(value: Any, name: str) -> Optional[pl.DataFrame]:
            if value is None:
                return None
            if isinstance(value, pl.DataFrame):
                return value
            self.logger.warning(
                "Unexpected type for '%s' in raw_data; expected Polars DataFrame or None",
                name,
                extra={"type": type(value).__name__},
            )
            return None

        blocks_df = _as_df_or_none(raw_data.get("blocks"), "blocks")
        tx_df = _as_df_or_none(raw_data.get("transactions"), "transactions")
        staking_df = _as_df_or_none(raw_data.get("staking"), "staking")
        supply_df = _as_df_or_none(raw_data.get("supply"), "supply")

        return blocks_df, tx_df, staking_df, supply_df

    def _nan_feature_mapping(self) -> Dict[str, float]:
        """
        Return a FeatureMapping where all features are set to NaN.

        This is used when raw data is missing or unusable, so that
        downstream components still receive a structurally valid row.
        """
        return {name: float("nan") for name in self._expected_feature_names}

    # ------------------------------------------------------------------ #
    # Feature-specific computation methods
    # ------------------------------------------------------------------ #

    def _compute_whale_flow_1d(
        self,
        tx_df: Optional[pl.DataFrame],
        eth_price_usd: Optional[float],
    ) -> float:
        """
        Compute 1-day whale flow for ETH.

        Inputs (raw, protocol-level):
            - 'value' (int/float, Wei): Transaction value in Wei.

        Heuristic (from review):
            - "Whale" = transaction with value_wei > WHALE_THRESHOLD_ETH * 1e18.

        Units:
            - If eth_price_usd is provided, returns whale flow in USD.
            - Otherwise, returns whale flow in ETH.

        Strategy:
            - Filter transactions where value >= threshold_wei.
            - Sum the value for those transactions.
        """
        if tx_df is None or tx_df.is_empty():
            self.logger.warning("_compute_whale_flow_1d: empty transactions DataFrame")
            return float("nan")

        if "value" not in tx_df.columns:
            self.logger.error(
                "_compute_whale_flow_1d: missing 'value' column; returning NaN"
            )
            return float("nan")

        threshold_wei = _WHALE_THRESHOLD_ETH * _WEI_PER_ETH

        whales = tx_df.filter(pl.col("value") >= threshold_wei)
        if whales.is_empty():
            return 0.0

        total_wei = (
            whales.select(pl.col("value").sum())
            .to_series()
            .item()
        )
        if total_wei is None:
            return float("nan")

        total_eth = float(total_wei) / _WEI_PER_ETH

        if eth_price_usd is not None:
            return total_eth * eth_price_usd

        # Fallback to ETH units if no price is provided.
        self.logger.warning(
            "_compute_whale_flow_1d: eth_price_usd not provided; returning flow in ETH"
        )
        return total_eth

    def _compute_exchange_netflow_7d(
        self,
        tx_df: Optional[pl.DataFrame],
    ) -> float:
        """
        Compute daily exchange netflow placeholder.

        Review feedback:
            - Raw AWS data does not contain exchange labels.
            - No open, static exchange address list is currently wired into
              the architecture.
            - 7-day rolling aggregation MUST NOT be implemented over raw
              transaction history due to memory constraints.

        Correct behavior per review:
            - The feature engine should NOT attempt to infer exchanges from
              unavailable labels.
            - The 7-day rolling aggregation belongs in downstream services,
              operating on daily aggregates.
            - Until a deterministic, allowed heuristic or config-based
              exchange list is introduced, this feature is marked as
              "pending" and returns 0.0 with a clear log entry.

        Returns:
            0.0 as a placeholder, or NaN if tx_df is None and we cannot
            even confirm daily data presence.
        """
        if tx_df is None or tx_df.is_empty():
            self.logger.warning(
                "_compute_exchange_netflow_7d: no transaction data; returning NaN"
            )
            return float("nan")

        self.logger.error(
            "_compute_exchange_netflow_7d: exchange address labeling is not "
            "implemented and external commercial datasets are forbidden. "
            "Returning 0.0 as a temporary placeholder. This feature must be "
            "revisited once an allowed heuristic or config-based mapping is "
            "defined."
        )
        return 0.0

    def _compute_gas_price_median(
        self,
        blocks_df: Optional[pl.DataFrame],
        tx_df: Optional[pl.DataFrame],
    ) -> float:
        """
        Compute median gas price for the day, normalized to Gwei.

        Inputs (raw, protocol-level):
            Preferred:
                - tx_df['gas_price'] (Wei)
            Fallbacks:
                - blocks_df['base_fee_per_gas'] (Wei)
                - blocks_df['gas_price_median'] (Wei)

        Strategy:
            - Use transaction-level gas_price median if available.
            - Else use block-level base_fee_per_gas median.
            - Else use gas_price_median mean.
            - Convert Wei → Gwei before returning to avoid unit confusion.

        Returns:
            Median gas price in Gwei, or NaN if no suitable columns exist.
        """
        if tx_df is not None and not tx_df.is_empty() and "gas_price" in tx_df.columns:
            median_wei = tx_df["gas_price"].median()
            if median_wei is None:
                return float("nan")
            return float(median_wei) / _WEI_PER_GWEI

        if blocks_df is not None and not blocks_df.is_empty():
            if "base_fee_per_gas" in blocks_df.columns:
                median_wei = blocks_df["base_fee_per_gas"].median()
                if median_wei is None:
                    return float("nan")
                return float(median_wei) / _WEI_PER_GWEI

            if "gas_price_median" in blocks_df.columns:
                mean_wei = blocks_df["gas_price_median"].mean()
                if mean_wei is None:
                    return float("nan")
                return float(mean_wei) / _WEI_PER_GWEI

        self.logger.warning(
            "_compute_gas_price_median: no suitable columns found; returning NaN"
        )
        return float("nan")

    def _compute_active_addresses(self, tx_df: Optional[pl.DataFrame]) -> float:
        """
        Compute the number of active addresses for ETH for the day.

        Inputs (raw, protocol-level):
            - 'from_address'
            - 'to_address'

        Strategy:
            - Count the number of unique non-null addresses across all
              available address columns using vectorized operations.
            - If no address columns are present or tx_df is empty, return NaN.
        """
        if tx_df is None or tx_df.is_empty():
            self.logger.warning(
                "_compute_active_addresses: empty transactions DataFrame"
            )
            return float("nan")

        address_cols = [
            col for col in ("from_address", "to_address") if col in tx_df.columns
        ]
        if not address_cols:
            self.logger.warning(
                "_compute_active_addresses: no address columns found; returning NaN"
            )
            return float("nan")

        series_list = [tx_df[col] for col in address_cols]
        combined = pl.concat(series_list).drop_nulls()
        unique_count = combined.cast(pl.Utf8).n_unique()

        return float(unique_count)

    def _compute_smart_contract_calls(
        self,
        tx_df: Optional[pl.DataFrame],
    ) -> float:
        """
        Compute the number of smart contract calls for the day.

        Review feedback:
            - Raw txs do NOT contain is_contract_call or to_address_type.
            - Correct behavior is to inspect protocol-level fields only.

        Inputs (raw, protocol-level):
            - 'input' (hex string): Non-empty input implies contract interaction.
            - 'to_address' (optional): Null/None indicates contract creation.

        Strategy:
            - A transaction is treated as a contract call/creation if:
                * input is non-null AND input != "0x" (has calldata), OR
                * to_address is null/None (contract creation).
            - If 'input' is missing, return NaN (schema issue).
        """
        if tx_df is None or tx_df.is_empty():
            self.logger.warning(
                "_compute_smart_contract_calls: empty transactions DataFrame"
            )
            return float("nan")

        if "input" not in tx_df.columns:
            self.logger.error(
                "_compute_smart_contract_calls: missing 'input' column; returning NaN"
            )
            return float("nan")

        has_input = pl.col("input").is_not_null() & (pl.col("input") != "0x")

        to_is_null = (
            pl.col("to_address").is_null()
            if "to_address" in tx_df.columns
            else pl.lit(False)
        )

        contract_mask = has_input | to_is_null
        count = tx_df.filter(contract_mask).height
        return float(count)

    def _compute_eth2_staking_flow(
        self,
        tx_df: Optional[pl.DataFrame],
    ) -> float:
        """
        Compute ETH2 staking flow for the day (ETH).

        Review feedback:
            - Relying on staking_df or is_staking_deposit flags violates
              the requirement to use only raw, protocol-level fields.
            - Correct behavior is to detect deposits to the canonical
              Beacon Chain Deposit Contract address.

        Inputs (raw, protocol-level):
            - 'to_address'
            - 'value' (Wei)

        Strategy:
            - Filter transactions where to_address == BEACON_DEPOSIT_CONTRACT.
            - Sum 'value' for those transactions and convert Wei → ETH.
            - If required columns are missing or tx_df is empty, return NaN.
        """
        if tx_df is None or tx_df.is_empty():
            self.logger.warning(
                "_compute_eth2_staking_flow: empty transactions DataFrame"
            )
            return float("nan")

        required_cols = {"to_address", "value"}
        missing = required_cols.difference(set(tx_df.columns))
        if missing:
            self.logger.error(
                "_compute_eth2_staking_flow: missing required columns; returning NaN",
                extra={"missing_columns": sorted(missing)},
            )
            return float("nan")

        # Normalize addresses to lowercase for comparison.
        df = tx_df.with_columns(
            pl.col("to_address").cast(pl.Utf8).str.to_lowercase()
        )

        deposits = df.filter(pl.col("to_address") == _BEACON_DEPOSIT_CONTRACT)
        if deposits.is_empty():
            return 0.0

        total_wei = (
            deposits.select(pl.col("value").sum())
            .to_series()
            .item()
        )
        if total_wei is None:
            return float("nan")

        total_eth = float(total_wei) / _WEI_PER_ETH
        return total_eth
