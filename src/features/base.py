"""
Abstract base classes and type definitions for feature engineering in
the ON-CHAIN SUPER SIGNALS™ project.

This module defines the foundational abstractions for all blockchain
feature computations. All chain-specific feature engines (BTC, ETH)
must inherit from BaseFeature and implement the required abstract
methods.

Design principles:
- Features are computed from in-memory raw blockchain data.
- Each feature represents a daily aggregate (one value per day).
- Raw blockchain data is never written to disk from this layer.
- Implementations must be deterministic and testable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Union, List

import pandas as pd
import polars as pl

from config.settings import CHAINS
from src.utils.logger import get_logger

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# Raw blockchain data can be represented as:
# - pandas DataFrame
# - polars DataFrame
# - a dictionary-like structure (for small, pre-aggregated inputs)
RawBlockchainData = Union[pd.DataFrame, pl.DataFrame, Dict[str, Any]]

# Canonical feature mapping: one daily value per feature name.
FeatureMapping = Dict[str, float]

# Implementations of compute() may return:
# - a FeatureMapping
# - a single-row pandas DataFrame
# - a single-row polars DataFrame
AggregatedFeatures = Union[FeatureMapping, pd.DataFrame, pl.DataFrame]


# ---------------------------------------------------------------------------
# Optional feature metadata
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    """
    Metadata for a single blockchain feature.

    Attributes:
        name:
            Feature name (e.g. "whale_flow_1d").
        description:
            Human-readable description of the feature.
        normalize:
            Whether this feature should be normalized.
        strict_validation:
            Whether this feature requires strict validation.

            - True  -> downstream logic MUST treat NaN/inf as invalid
                       and either raise or impute deterministically.
            - False -> NaN/inf may be tolerated or handled in a more
                       permissive way by downstream components.
    """
    name: str
    description: str
    normalize: bool = True
    strict_validation: bool = True


# ---------------------------------------------------------------------------
# Base abstract feature class
# ---------------------------------------------------------------------------

class BaseFeature(ABC):
    """
    Abstract base class for all blockchain feature computations.

    All chain-specific feature classes (BTC, ETH) must inherit from this
    class and implement the abstract methods.

    Design principles:
    - Features are computed from in-memory raw blockchain data.
    - Each feature represents a daily aggregate (one value per day).
    - Features must be reproducible and testable.
    - Raw data is never persisted; only aggregated features are stored.
    """

    def __init__(self, chain: str) -> None:
        """
        Initialize a feature engine for a given blockchain chain.

        Args:
            chain: Chain symbol ("BTC" or "ETH", case-insensitive).

        Raises:
            TypeError: If chain is None.
            ValueError: If chain is not one of the supported CHAINS.
        """
        if chain is None:
            raise TypeError("chain must not be None")

        chain_str = str(chain).strip()
        if not chain_str:
            raise ValueError("chain must be a non-empty string")

        chain_upper = chain_str.upper()
        if chain_upper not in CHAINS:
            raise ValueError(
                f"Unsupported chain: '{chain_str}'. Supported chains: {CHAINS}"
            )

        self.chain: str = chain_upper
        self.logger = get_logger(self.__class__.__name__)

        self.logger.debug(
            "Feature engine initialized",
            extra={"chain": self.chain},
        )

    # ------------------------------------------------------------------ #
    # Abstract API to be implemented by subclasses
    # ------------------------------------------------------------------ #

    @abstractmethod
    def compute(
        self,
        raw_data: RawBlockchainData,
        target_date: date,
    ) -> AggregatedFeatures:
        """
        Compute all features for a single day from raw blockchain data.

        This method MUST be implemented by subclasses.

        Implementations are expected to:
            - Validate that `raw_data` is not None and not empty.
            - Compute all relevant features for the given chain and date.
            - Return either:
                * a dict mapping feature_name -> float, or
                * a single-row DataFrame (pandas or polars) whose columns
                  are exactly the feature names returned by get_feature_names().

        Contract:
            - The set of feature names MUST match exactly the names returned
              by `get_feature_names()` (no more, no less).
            - Each feature represents a single daily aggregate value.

        Args:
            raw_data:
                Raw blockchain data (blocks and/or transactions)
                for the given target_date. Must be in-memory.
            target_date:
                The date the features correspond to.

        Returns:
            AggregatedFeatures:
                Either a FeatureMapping or a single-row DataFrame with
                feature_name -> value entries.

        Raises:
            ValueError:
                If raw_data is invalid (e.g., missing required columns),
                or if the implementation cannot produce the required
                daily aggregates.
        """
        raise NotImplementedError

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Return a list of feature names produced by this engine.

        Implementations are expected to return a stable, complete list
        of feature names, for example:

            return [
                "whale_flow_1d",
                "exchange_netflow_7d",
                "mempool_stress",
                "active_addresses",
            ]

        The set of names returned here is treated as the contract for
        aggregate_daily: any result returned by compute(...) must have
        exactly these feature names (no more, no less).

        Returns:
            A list of feature names as strings.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Concrete wrapper for consistent logging and validation
    # ------------------------------------------------------------------ #

    def aggregate_daily(
        self,
        raw_data: RawBlockchainData,
        target_date: date,
    ) -> FeatureMapping:
        """
        Compute daily aggregated features with standardized logging and checks.

        This method wraps the abstract `compute` method to provide:
            - Consistent logging around feature computation.
            - Validation of the returned feature structure.
            - Enforcement that the result represents a single daily aggregate.
            - A canonical return format: Dict[str, float].

        Args:
            raw_data: Raw blockchain data for the given date.
            target_date: Date the features are computed for.

        Returns:
            FeatureMapping:
                A dict mapping feature names (as returned by
                `get_feature_names()`) to their daily aggregate values.

        Raises:
            ValueError:
                - If `compute` returns None.
                - If `compute` returns an empty structure.
                - If a DataFrame result does not contain exactly one row.
                - If the feature names do not match get_feature_names().
            Exception:
                - Any exception raised by `compute` is logged and re-raised.
        """
        self.logger.info(
            "Computing features",
            extra={
                "chain": self.chain,
                "date": target_date.isoformat(),
            },
        )

        try:
            features = self.compute(raw_data, target_date)
        except Exception as exc:
            self.logger.error(
                "Feature computation failed",
                extra={
                    "chain": self.chain,
                    "date": target_date.isoformat(),
                    "error": repr(exc),
                },
                exc_info=True,
            )
            # Propagate the original exception; pipeline logic decides how to handle it.
            raise

        # Validate returned features
        if features is None:
            raise ValueError("compute() returned None")

        expected_names = self.get_feature_names()
        expected_name_set = set(expected_names)

        # Canonical mapping that will be returned
        feature_mapping: FeatureMapping

        if isinstance(features, dict):
            if not features:
                raise ValueError("compute() returned empty features")

            actual_name_set = set(features.keys())
            if actual_name_set != expected_name_set:
                raise ValueError(
                    f"compute() returned feature keys {sorted(actual_name_set)}, "
                    f"but expected {sorted(expected_name_set)}"
                )

            # Normalize to Dict[str, float]
            feature_mapping = {
                name: float(features[name]) for name in expected_names
            }

        elif isinstance(features, pd.DataFrame):
            if features.empty:
                raise ValueError("compute() returned empty DataFrame")

            row_count = features.shape[0]
            columns = list(features.columns)

            if row_count != 1:
                raise ValueError(
                    f"compute() must return a single-row DataFrame "
                    f"(got {row_count} rows)"
                )
            if not columns:
                raise ValueError("compute() returned DataFrame with no columns")

            actual_name_set = set(columns)
            if actual_name_set != expected_name_set:
                raise ValueError(
                    f"compute() returned columns {sorted(actual_name_set)}, "
                    f"but expected {sorted(expected_name_set)}"
                )

            # Take the single row and normalize to Dict[str, float]
            row = features.iloc[0]
            feature_mapping = {
                name: float(row[name]) for name in expected_names
            }

        elif isinstance(features, pl.DataFrame):
            if features.height == 0:
                raise ValueError("compute() returned empty DataFrame")

            row_count = features.height
            columns = list(features.columns)

            if row_count != 1:
                raise ValueError(
                    f"compute() must return a single-row DataFrame "
                    f"(got {row_count} rows)"
                )
            if not columns:
                raise ValueError("compute() returned DataFrame with no columns")

            actual_name_set = set(columns)
            if actual_name_set != expected_name_set:
                raise ValueError(
                    f"compute() returned columns {sorted(actual_name_set)}, "
                    f"but expected {sorted(expected_name_set)}"
                )

            # polars: get the first row as a dict
            row_dict = features.row(0, named=True)
            feature_mapping = {
                name: float(row_dict[name]) for name in expected_names
            }

        else:
            # Unsupported/opaque feature structure -> rejected
            raise ValueError(
                "compute() returned an unsupported feature structure; "
                "expected dict or single-row DataFrame."
            )

        num_features = len(feature_mapping)

        self.logger.info(
            "Features computed",
            extra={
                "chain": self.chain,
                "date": target_date.isoformat(),
                "num_features": num_features,
            },
        )

        return feature_mapping
