"""
BTC-specific feature engine for the ON-CHAIN SUPER SIGNALS™ project.

This module implements the Bitcoin (BTC) feature computation logic as
specified in the architecture and requirements documents. It computes
daily, aggregated on-chain features from in-memory raw blockchain data.

Data minimization:
- Consumes only in-memory raw data (blocks/transactions) provided by the
  DataService (backed by AWS Public Blockchain Data on S3).
- Does NOT write any raw blockchain data to disk.
- Returns only daily aggregated feature values for persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.features.base import (
    AggregatedFeatures,
    BaseFeature,
    RawBlockchainData,
)
from src.utils.logger import get_logger


@dataclass(frozen=True)
class _BTCFeatureNames:
    """
    Container for canonical BTC feature names.

    These names MUST align with:
    - config/features.yaml (btc section)
    - Downstream model and signal-generation logic
    """

    WHALE_FLOW_1D: str = "whale_flow_1d"
    EXCHANGE_NETFLOW_7D: str = "exchange_netflow_7d"
    MEMPOOL_STRESS: str = "mempool_stress"
    FEE_PRESSURE: str = "fee_pressure"
    ACTIVE_ADDRESSES: str = "active_addresses"
    UTXO_AGE_1Y: str = "utxo_age_1y"
    REVIVED_SUPPLY: str = "revived_supply"


_FEATURE_NAMES = _BTCFeatureNames()


class BTCFeatureEngine(BaseFeature):
    """
    Bitcoin-specific feature engine.

    Responsibilities:
    - Accept raw BTC blocks/transactions for a single day.
    - Compute daily aggregated features defined in _BTCFeatureNames.
    - Return a FeatureMapping (dict[str, float]) suitable for persistence.

    Input structure (raw_data):
        {
            "blocks": pd.DataFrame | None,
            "transactions": pd.DataFrame | None,
        }

    Edge cases:
    - If raw_data is missing or empty, all features are returned as NaN.
    - If required columns for a specific feature are missing, that feature
      is set to NaN, and a warning is logged.
    """

    def __init__(self, chain: str = "BTC") -> None:
        # Enforce BTC chain explicitly, regardless of caller input.
        super().__init__(chain="BTC")
        self.logger = get_logger(self.__class__.__name__)

        # Pre-compute expected feature name set.
        self._expected_feature_names: Set[str] = {
            _FEATURE_NAMES.WHALE_FLOW_1D,
            _FEATURE_NAMES.EXCHANGE_NETFLOW_7D,
            _FEATURE_NAMES.MEMPOOL_STRESS,
            _FEATURE_NAMES.FEE_PRESSURE,
            _FEATURE_NAMES.ACTIVE_ADDRESSES,
            _FEATURE_NAMES.UTXO_AGE_1Y,
            _FEATURE_NAMES.REVIVED_SUPPLY,
        }

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
    ) -> AggregatedFeatures:
        """
        Compute all BTC features for a single day.

        Args:
            raw_data:
                Dict-like structure with keys:
                    - "blocks": pd.DataFrame or None
                    - "transactions": pd.DataFrame or None
            target_date:
                The date these features correspond to.

        Returns:
            Dict[str, float]: A mapping from feature name to value.

        Behavior:
            - If raw_data is missing or empty, returns all-NaN feature mapping.
            - Any feature with missing input columns is set to NaN.
            - No raw data is written to disk; all computation is in-memory.
        """
        self.logger.info(
            "Computing BTC features",
            extra={"date": target_date.isoformat()},
        )

        blocks_df, tx_df = self._extract_dataframes(raw_data)

        if (blocks_df is None or blocks_df.empty) and (tx_df is None or tx_df.empty):
            self.logger.warning(
                "Empty raw BTC data for date; returning all-NaN features",
                extra={"date": target_date.isoformat()},
            )
            return self._nan_feature_mapping()

        # Compute each feature individually. All helpers must be deterministic
        # and MUST NOT write to disk or retain references beyond this call.
        features: Dict[str, float] = {
            _FEATURE_NAMES.WHALE_FLOW_1D: self.compute_whale_flow_1d(tx_df),
            _FEATURE_NAMES.EXCHANGE_NETFLOW_7D: self.compute_exchange_netflow(tx_df),
            _FEATURE_NAMES.MEMPOOL_STRESS: self.compute_mempool_stress(blocks_df, tx_df),
            _FEATURE_NAMES.FEE_PRESSURE: self.compute_fee_pressure(blocks_df, tx_df),
            _FEATURE_NAMES.ACTIVE_ADDRESSES: self.compute_active_addresses(tx_df),
            _FEATURE_NAMES.UTXO_AGE_1Y: self.compute_utxo_age_1y(tx_df),
            _FEATURE_NAMES.REVIVED_SUPPLY: self.compute_revived_supply(tx_df),
        }

        # Ensure all expected features are present; if any are missing, fill with NaN.
        for name in self._expected_feature_names:
            if name not in features:
                self.logger.warning(
                    "Feature missing from computation; filling with NaN",
                    extra={"feature": name, "date": target_date.isoformat()},
                )
                features[name] = float("nan")

        # No validation of NaN/inf here; that is handled by validators at a higher level.
        self.logger.info(
            "BTC features computed",
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
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Extract blocks and transactions DataFrames from raw_data safely.

        Expects:
            raw_data: dict with optional 'blocks' and 'transactions' keys.

        Returns:
            (blocks_df, tx_df), where each may be None if absent.

        Raises:
            TypeError: If raw_data is not a dict-like structure.
        """
        if not isinstance(raw_data, dict):
            self.logger.error(
                "BTCFeatureEngine.compute expects raw_data as dict",
                extra={"type": type(raw_data).__name__},
            )
            raise TypeError(
                "BTCFeatureEngine.compute expects raw_data as Dict[str, Any] "
                "containing 'blocks' and 'transactions' keys"
            )

        blocks_df: Optional[pd.DataFrame] = None
        tx_df: Optional[pd.DataFrame] = None

        blocks = raw_data.get("blocks")
        txs = raw_data.get("transactions")

        if isinstance(blocks, pd.DataFrame):
            blocks_df = blocks
        elif blocks is not None:
            self.logger.warning(
                "Unexpected type for 'blocks' in raw_data; expected DataFrame or None",
                extra={"type": type(blocks).__name__},
            )

        if isinstance(txs, pd.DataFrame):
            tx_df = txs
        elif txs is not None:
            self.logger.warning(
                "Unexpected type for 'transactions' in raw_data; expected DataFrame or None",
                extra={"type": type(txs).__name__},
            )

        return blocks_df, tx_df

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

    def compute_whale_flow_1d(self, tx_df: Optional[pd.DataFrame]) -> float:
        """
        Compute 1-day whale flow (USD).

        Expected columns (if available):
            - 'is_whale' (bool): Whether the transaction involves a "whale".
            - 'value_usd' (float): USD value of the transaction.

        If the required columns are missing or tx_df is empty, returns NaN.
        """
        if tx_df is None or tx_df.empty:
            self.logger.warning("compute_whale_flow_1d: empty transactions DataFrame")
            return float("nan")

        required_cols = {"is_whale", "value_usd"}
        missing = required_cols.difference(tx_df.columns)
        if missing:
            self.logger.warning(
                "compute_whale_flow_1d: missing columns; returning NaN",
                extra={"missing_columns": sorted(missing)},
            )
            return float("nan")

        whale_tx = tx_df[tx_df["is_whale"] == True]  # noqa: E712
        if whale_tx.empty:
            return 0.0

        value = float(whale_tx["value_usd"].sum())
        return value

    def compute_exchange_netflow(self, tx_df: Optional[pd.DataFrame]) -> float:
        """
        Compute 7-day exchange netflow (approximation based on available data).

        Expected columns (if available):
            - 'exchange_in_usd' (float): Flow into exchanges for the day (USD).
            - 'exchange_out_usd' (float): Flow out of exchanges for the day (USD).

        Netflow definition:
            netflow = exchange_in_usd - exchange_out_usd

        If required columns are missing or tx_df is empty, returns NaN.
        """
        if tx_df is None or tx_df.empty:
            self.logger.warning("compute_exchange_netflow: empty transactions DataFrame")
            return float("nan")

        required_cols = {"exchange_in_usd", "exchange_out_usd"}
        missing = required_cols.difference(tx_df.columns)
        if missing:
            self.logger.warning(
                "compute_exchange_netflow: missing columns; returning NaN",
                extra={"missing_columns": sorted(missing)},
            )
            return float("nan")

        in_sum = float(tx_df["exchange_in_usd"].sum())
        out_sum = float(tx_df["exchange_out_usd"].sum())
        return in_sum - out_sum

    def compute_mempool_stress(
        self,
        blocks_df: Optional[pd.DataFrame],
        tx_df: Optional[pd.DataFrame],
    ) -> float:
        """
        Compute a mempool stress indicator.

        Preferred inputs (if available):
            - blocks_df['tx_count'] (per-block transaction count)
            - Fallback: total number of transactions in tx_df

        Strategy:
            - If blocks_df has 'tx_count', use the daily sum of tx_count.
            - Else if tx_df is available, use the number of transactions.
            - Otherwise, return NaN.

        This avoids relying on pre-aggregated daily fields and keeps the
        aggregation responsibility within this engine.
        """
        if blocks_df is not None and not blocks_df.empty and "tx_count" in blocks_df.columns:
            tx_sum = float(blocks_df["tx_count"].sum())
            return tx_sum

        if tx_df is not None and not tx_df.empty:
            # Fallback approximation: treat row count as total tx count.
            return float(len(tx_df))

        self.logger.warning(
            "compute_mempool_stress: no suitable columns found; returning NaN"
        )
        return float("nan")

    def compute_fee_pressure(
        self,
        blocks_df: Optional[pd.DataFrame],
        tx_df: Optional[pd.DataFrame],
    ) -> float:
        """
        Compute a fee pressure indicator.

        Preferred inputs (if available):
            - blocks_df['total_fees_btc'] and 'block_subsidy_btc' or 'block_reward_btc'
            - tx_df['fee_rate'] or 'fee']

        Strategy:
            - If blocks_df has both 'total_fees_btc' and a reward column,
              compute fee_pressure = mean(total_fees_btc / reward) over blocks
              with non-zero reward.
            - Else fall back to mean 'fee_rate' or 'fee' in tx_df.
            - Otherwise, return NaN.
        """
        if blocks_df is not None and not blocks_df.empty:
            fee_col: Optional[str] = None
            reward_col: Optional[str] = None

            if "total_fees_btc" in blocks_df.columns:
                fee_col = "total_fees_btc"
            if "block_reward_btc" in blocks_df.columns:
                reward_col = "block_reward_btc"
            elif "block_subsidy_btc" in blocks_df.columns:
                reward_col = "block_subsidy_btc"

            if fee_col and reward_col:
                rewards = blocks_df[reward_col]
                non_zero_mask = rewards != 0
                if non_zero_mask.any():
                    ratio = blocks_df.loc[non_zero_mask, fee_col] / rewards.loc[non_zero_mask]
                    if ratio.notna().any():
                        return float(ratio.mean())
                else:
                    self.logger.warning(
                        "compute_fee_pressure: all rewards are zero; cannot compute ratio"
                    )

        if tx_df is not None and not tx_df.empty:
            if "fee_rate" in tx_df.columns:
                return float(tx_df["fee_rate"].mean())
            if "fee" in tx_df.columns:
                return float(tx_df["fee"].mean())

        self.logger.warning(
            "compute_fee_pressure: no suitable columns found; returning NaN"
        )
        return float("nan")

    def compute_active_addresses(self, tx_df: Optional[pd.DataFrame]) -> float:
        """
        Compute the number of active addresses for the day in a UTXO-aware way.

        Preferred columns (any subset):
            - 'input_address', 'output_address'
            - 'input_script_pubkey', 'output_script_pubkey'
            - 'input_pubkey_hash', 'output_pubkey_hash'

        Strategy:
            - Aggregate unique non-null input references (inputs) and
              unique non-null output destinations (outputs).
            - If no UTXO-style columns are present, fall back to any
              'from_address'/'to_address' columns for compatibility.
            - If no address-like columns are present or tx_df is empty,
              return NaN.
        """
        if tx_df is None or tx_df.empty:
            self.logger.warning("compute_active_addresses: empty transactions DataFrame")
            return float("nan")

        utxo_input_cols = [
            col
            for col in (
                "input_address",
                "input_script_pubkey",
                "input_pubkey_hash",
            )
            if col in tx_df.columns
        ]
        utxo_output_cols = [
            col
            for col in (
                "output_address",
                "output_script_pubkey",
                "output_pubkey_hash",
            )
            if col in tx_df.columns
        ]

        address_cols: List[str] = utxo_input_cols + utxo_output_cols

        if not address_cols:
            # Backwards-compatible fallback for simplified schemas.
            fallback_cols = [
                col
                for col in ("from_address", "to_address")
                if col in tx_df.columns
            ]
            if not fallback_cols:
                self.logger.warning(
                    "compute_active_addresses: no address-like columns found; returning NaN"
                )
                return float("nan")

            self.logger.warning(
                "compute_active_addresses: using fallback from/to address columns; "
                "underlying schema may be non-UTXO-accurate"
            )
            address_cols = fallback_cols

        unique_addresses: Set[str] = set()
        for col in address_cols:
            series = tx_df[col].dropna()
            # Convert to string to ensure hashability and consistency.
            unique_addresses.update(series.astype(str).tolist())

        return float(len(unique_addresses))

    def compute_utxo_age_1y(self, tx_df: Optional[pd.DataFrame]) -> float:
        """
        Compute a UTXO age metric representing outputs older than 1 year.

        Expected columns (if available):
            - 'spent_output_age_days' (float or int)
            - 'value_btc' or 'value_usd'

        Strategy:
            - Filter rows where spent_output_age_days >= 365.
            - Sum the chosen value column.
            - If required columns are missing or tx_df is empty, return NaN.
        """
        if tx_df is None or tx_df.empty:
            self.logger.warning("compute_utxo_age_1y: empty transactions DataFrame")
            return float("nan")

        if "spent_output_age_days" not in tx_df.columns:
            self.logger.warning(
                "compute_utxo_age_1y: missing 'spent_output_age_days'; returning NaN"
            )
            return float("nan")

        value_col: Optional[str] = None
        if "value_usd" in tx_df.columns:
            value_col = "value_usd"
        elif "value_btc" in tx_df.columns:
            value_col = "value_btc"

        if value_col is None:
            self.logger.warning(
                "compute_utxo_age_1y: no suitable value column; returning NaN"
            )
            return float("nan")

        mask = tx_df["spent_output_age_days"] >= 365
        older_utxos = tx_df[mask]
        if older_utxos.empty:
            return 0.0

        return float(older_utxos[value_col].sum())

    def _approximate_revived_supply(self, tx_df: Optional[pd.DataFrame]) -> float:
        """
        Internal helper to approximate revived supply (USD) for BTC.

        Expected columns (if available):
            - 'revived_supply_usd'
        or fallback:
            - 'spent_output_age_days'
            - 'value_usd'

        Strategy:
            - If 'revived_supply_usd' exists, sum it.
            - Else approximate by summing 'value_usd' where
              spent_output_age_days >= 365.
            - If required columns are missing or tx_df is empty, return NaN.
        """
        if tx_df is None or tx_df.empty:
            self.logger.warning("_approximate_revived_supply: empty transactions DataFrame")
            return float("nan")

        if "revived_supply_usd" in tx_df.columns:
            return float(tx_df["revived_supply_usd"].sum())

        if "spent_output_age_days" in tx_df.columns and "value_usd" in tx_df.columns:
            mask = tx_df["spent_output_age_days"] >= 365
            revived = tx_df[mask]
            if revived.empty:
                return 0.0
            return float(revived["value_usd"].sum())

        self.logger.warning(
            "_approximate_revived_supply: no suitable columns found; returning NaN"
        )
        return float("nan")

    def compute_revived_supply(self, tx_df: Optional[pd.DataFrame]) -> float:
        """
        Compute revived supply (USD) for BTC.

        This is a thin wrapper around `_approximate_revived_supply` to keep
        the public API aligned with the architecture while centralizing the
        core computation logic in a single place.
        """
        return self._approximate_revived_supply(tx_df)
