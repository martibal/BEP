"""
Feature normalization utilities for the ON-CHAIN SUPER SIGNALS™ project.

This module implements the FeatureNormalizer, responsible for:
- Fitting normalization parameters (mean/std) on training data.
- Applying Z-score normalization to new data.
- Persisting/loading normalization parameters as JSON.

Data minimization:
- Only aggregated statistics (mean/std per feature) are stored.
- No raw feature vectors or blockchain data are written to disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import polars as pl

from src.utils.logger import get_logger


@dataclass
class FeatureNormalizer:
    """
    Z-score feature normalizer (mean=0, std=1).

    Responsibilities
    ----------------
    - Compute mean and std for a set of features from a training DataFrame.
    - Apply normalization to new DataFrames using stored parameters.
    - Persist and load normalization parameters as JSON.

    Notes
    -----
    - Standard deviation is computed with ddof=0 to match scikit-learn's
      StandardScaler behavior.
    - If std == 0 (constant feature), normalized values are set to 0.0 to
      avoid division by zero and infinities.
    - NaNs in input data are preserved through transform.
    - Infinite values (±inf) are excluded from statistics with a warning.
    """

    params: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def fit(self, data: pl.DataFrame, features: List[str]) -> None:
        """
        Fit normalization parameters (mean, std) for the given features.

        Parameters
        ----------
        data:
            Training Polars DataFrame containing the feature columns.
        features:
            List of column names to compute normalization for.

        Raises
        ------
        ValueError
            If the input DataFrame is empty or the feature list is empty.
        """
        if data is None or not isinstance(data, pl.DataFrame):
            self.logger.error("fit called with invalid data (must be Polars DataFrame)")
            raise ValueError("Cannot fit normalizer without a Polars DataFrame")

        if data.is_empty():
            self.logger.error("fit called with empty DataFrame")
            raise ValueError("Cannot fit normalizer on empty DataFrame")

        if not features:
            self.logger.error("fit called with empty feature list")
            raise ValueError("Cannot fit normalizer with empty feature list")

        self.params.clear()

        for feature in features:
            if feature not in data.columns:
                self.logger.warning(
                    "Feature '%s' not found in DataFrame during fit; skipping",
                    feature,
                )
                continue

            series = data[feature]

            # Drop nulls first.
            non_null = series.drop_nulls()

            if non_null.is_empty():
                self.logger.warning(
                    "Feature '%s' has no non-null values during fit; "
                    "treating as constant (mean=0.0, std=0.0)",
                    feature,
                )
                mean_val = 0.0
                std_val = 0.0
                self.params[feature] = {"mean": mean_val, "std": std_val}
                continue

            values = non_null.to_numpy()
            finite_mask = np.isfinite(values)

            if not np.all(finite_mask):
                num_bad = int((~finite_mask).sum())
                self.logger.warning(
                    "Feature '%s' contains %d non-finite values (inf/-inf); "
                    "excluding them from statistics",
                    feature,
                    num_bad,
                )
                values = values[finite_mask]

            if values.size == 0:
                # After removing inf/-inf, nothing left — treat as constant.
                self.logger.warning(
                    "Feature '%s' has no finite values after filtering; "
                    "treating as constant (mean=0.0, std=0.0)",
                    feature,
                )
                mean_val = 0.0
                std_val = 0.0
            else:
                mean_val = float(values.mean())
                # ddof=0 to match StandardScaler
                std_val = float(values.std(ddof=0))

                if values.size < 2:
                    self.logger.warning(
                        "Feature '%s' has <2 finite samples during fit; "
                        "std may be unreliable (std=%s)",
                        feature,
                        std_val,
                    )

                if not np.isfinite(std_val):
                    self.logger.warning(
                        "Feature '%s' produced non-finite std (%s); overriding to 0.0",
                        feature,
                        std_val,
                    )
                    std_val = 0.0

            # Ensure we never store NaN in params (for JSON safety).
            if not np.isfinite(mean_val):
                self.logger.warning(
                    "Feature '%s' produced non-finite mean (%s); overriding to 0.0",
                    feature,
                    mean_val,
                )
                mean_val = 0.0

            self.params[feature] = {"mean": mean_val, "std": std_val}

        if not self.params:
            self.logger.error(
                "fit completed but no parameters were computed; "
                "check feature list and DataFrame columns"
            )

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Apply Z-score normalization to the DataFrame using fitted parameters.

        Parameters
        ----------
        data:
            Input Polars DataFrame. May contain one or more rows.

        Returns
        -------
        pl.DataFrame
            New DataFrame with normalized values for all features present
            in both the data and self.params. Other columns are left untouched.

        Raises
        ------
        ValueError
            If normalization parameters have not been fitted or loaded.
        """
        if not self.params:
            self.logger.error(
                "transform called before parameters were fitted/loaded"
            )
            raise ValueError("Normalization parameters are not initialized")

        if data is None or not isinstance(data, pl.DataFrame):
            self.logger.error("transform called with invalid data (must be Polars DataFrame)")
            raise ValueError("DataFrame cannot be None and must be Polars")

        if data.is_empty():
            self.logger.warning("transform called with empty DataFrame")
            return data.clone()

        result = data

        # Normalize only columns we have parameters for; others are passed through.
        for feature, stats in self.params.items():
            if feature not in result.columns:
                self.logger.error(
                    "Missing required feature '%s' in input data; "
                    "skipping normalization for this feature.",
                    feature,
                )
                continue

            mean_val = stats.get("mean", 0.0)
            std_val = stats.get("std", 0.0)

            if std_val == 0.0 or not np.isfinite(std_val):
                # Constant or invalid std: set all non-null entries to 0.0.
                self.logger.warning(
                    "Feature '%s' has std=%s; setting normalized values to 0.0",
                    feature,
                    std_val,
                )
                expr = (
                    pl.when(pl.col(feature).is_null())
                    .then(pl.lit(None))
                    .otherwise(pl.lit(0.0))
                    .alias(feature)
                )
            else:
                expr = ((pl.col(feature) - mean_val) / std_val).alias(feature)

            result = result.with_columns(expr)

        return result

    def save_params(self, path: Path) -> None:
        """
        Save normalization parameters to a JSON file.

        Parameters
        ----------
        path:
            Output path for the JSON file. Parent directories are created
            if they do not exist.

        Raises
        ------
        ValueError
            If no parameters have been fitted (self.params is empty).
        """
        if not self.params:
            self.logger.error(
                "save_params called with empty params; refusing to write file"
            )
            raise ValueError("Cannot save empty normalization parameters")

        path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure all values are basic floats and JSON-safe (no NaN/inf).
        payload: Dict[str, Dict[str, float]] = {}
        for feature, stats in self.params.items():
            mean_raw = stats.get("mean", 0.0)
            std_raw = stats.get("std", 0.0)

            mean_val = float(mean_raw)
            std_val = float(std_raw)

            if not np.isfinite(mean_val):
                self.logger.warning(
                    "Non-finite mean detected for feature '%s' during save; "
                    "overriding to 0.0",
                    feature,
                )
                mean_val = 0.0

            if not np.isfinite(std_val):
                self.logger.warning(
                    "Non-finite std detected for feature '%s' during save; "
                    "overriding to 0.0",
                    feature,
                )
                std_val = 0.0

            payload[feature] = {"mean": mean_val, "std": std_val}

        # Atomic write: write to temporary file, then rename.
        tmp_path = path.with_suffix(path.suffix + ".tmp")

        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)

        tmp_path.replace(path)

        self.logger.info(
            "Normalization parameters saved",
            extra={"filename": path.name, "num_features": len(payload)},
        )

    def load_params(self, path: Path) -> None:
        """
        Load normalization parameters from a JSON file.

        Parameters
        ----------
        path:
            Path to the JSON file containing normalization parameters.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file content is not valid JSON or has unexpected format.
        """
        if not path.is_file():
            self.logger.error(
                "Normalization parameters file not found",
                extra={"filename": path.name},
            )
            raise FileNotFoundError(f"Normalization parameters file not found: {path}")

        try:
            with path.open("r", encoding="utf-8") as f:
                raw: Mapping[str, Any] = json.load(f)
        except json.JSONDecodeError as exc:
            self.logger.error(
                "Failed to decode normalization parameters JSON",
                extra={"filename": path.name, "error": str(exc)},
            )
            raise ValueError(
                f"Invalid JSON in normalization parameters file: {path}"
            ) from exc

        if not isinstance(raw, dict):
            self.logger.error(
                "Unexpected JSON structure for normalization parameters",
                extra={"filename": path.name, "type": type(raw).__name__},
            )
            raise ValueError("Normalization parameters JSON must be an object/dict")

        parsed: Dict[str, Dict[str, float]] = {}
        for feature, stats in raw.items():
            if not isinstance(stats, dict):
                self.logger.warning(
                    "Skipping feature '%s' due to invalid stats format", feature
                )
                continue

            mean_raw = stats.get("mean", 0.0)
            std_raw = stats.get("std", 0.0)

            # Handle None/null and non-numeric values gracefully.
            try:
                if mean_raw is None or (
                    isinstance(mean_raw, float) and np.isnan(mean_raw)
                ):
                    mean_val = 0.0
                else:
                    mean_val = float(mean_raw)

                if std_raw is None or (
                    isinstance(std_raw, float) and np.isnan(std_raw)
                ):
                    std_val = 0.0
                else:
                    std_val = float(std_raw)
            except (TypeError, ValueError):
                self.logger.warning(
                    "Skipping feature '%s' due to non-numeric mean/std", feature
                )
                continue

            # Enforce finiteness.
            if not np.isfinite(mean_val):
                self.logger.warning(
                    "Non-finite mean loaded for feature '%s'; overriding to 0.0",
                    feature,
                )
                mean_val = 0.0

            if not np.isfinite(std_val):
                self.logger.warning(
                    "Non-finite std loaded for feature '%s'; overriding to 0.0",
                    feature,
                )
                std_val = 0.0

            parsed[feature] = {"mean": mean_val, "std": std_val}

        if not parsed:
            self.logger.error(
                "No valid normalization parameters found in file",
                extra={"filename": path.name},
            )
            raise ValueError("No valid normalization parameters loaded")

        self.params = parsed

        self.logger.info(
            "Normalization parameters loaded",
            extra={"filename": path.name, "num_features": len(self.params)},
        )
