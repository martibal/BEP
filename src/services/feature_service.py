from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Any, List

import math

import pandas as pd
import yaml

from config.settings import CHAINS, NORMALIZATION_DIR
from src.features.normalizer import FeatureNormalizer
from src.services.data_service import DataService
from src.utils import validators
from src.utils.logger import get_logger

_logger = get_logger(__name__)


class FeatureService:
    """
    Orchestrates feature computation, normalization, and validation.

    Responsibilities
    ----------------
    - Orchestrate daily feature computation for one or more chains.
    - Coordinate with DataService to load aggregated daily features.
      NOTE: This implementation expects that daily aggregated features
      have already been computed and stored by the
      pipelines/01_fetch_and_aggregate.py script into the tmp/ layer,
      and are then exposed via DataService.load_daily_aggregates(...).
    - Apply normalization via FeatureNormalizer.
    - Validate feature vectors before they are passed further downstream.
    - Provide a clean interface for pipelines and other services.

    Notes
    -----
    - This service never accesses S3 directly; all data access is delegated
      to DataService.
    - This service never persists raw blockchain data. Only aggregated,
      daily features are handled in-memory.
    - This service does not contain any model prediction logic.
    """

    def __init__(
        self,
        data_service: Optional[DataService] = None,
        normalizer: Optional[FeatureNormalizer] = None,
    ) -> None:
        """
        Initialize the FeatureService.

        Parameters
        ----------
        data_service : Optional[DataService]
            DataService instance for data retrieval and aggregation.
            If None, a new DataService instance is created.

        normalizer : Optional[FeatureNormalizer]
            FeatureNormalizer instance for normalization.
            If None, a new FeatureNormalizer instance is created.
        """
        self._data_service: DataService = data_service if data_service is not None else DataService()
        self._normalizer: FeatureNormalizer = normalizer if normalizer is not None else FeatureNormalizer()
        self._has_normalizer_params: bool = False
        self._feature_definitions: Dict[str, List[str]] = {}

        # Load feature definitions from config/features.yaml
        self._load_feature_definitions()

        # Attempt to load normalization params on startup.
        try:
            self.load_normalizer_params()
            self._has_normalizer_params = True
        except FileNotFoundError:
            # First run is allowed to proceed without normalization params.
            self._has_normalizer_params = False
            _logger.info(
                "No normalization parameters found during FeatureService initialization; "
                "service will operate with raw features until params are fitted and saved."
            )
        except ValueError as exc:
            # Corrupt or invalid params file is a configuration issue.
            self._has_normalizer_params = False
            _logger.error(
                "Failed to load normalization parameters during FeatureService initialization.",
                extra={"error": str(exc)},
            )

        _logger.info(
            "FeatureService initialized",
            extra={"has_normalizer_params": self._has_normalizer_params},
        )

    # -------------------------------------------------------------------------
    # Internal helpers for configuration
    # -------------------------------------------------------------------------

    def _load_feature_definitions(self) -> None:
        """
        Load feature definitions per chain from config/features.yaml.

        The expected YAML structure is:

            btc:
              - whale_flow_1d
              - exchange_netflow_7d
              - mempool_stress
              - fee_pressure
              - active_addresses
              - utxo_age_1y
              - revived_supply

            eth:
              - whale_flow_1d
              - exchange_netflow_7d
              - gas_price_median
              - active_addresses
              - smart_contract_calls
              - eth2_staking_flow

        The loaded definitions are stored in self._feature_definitions as
        a mapping from UPPERCASE chain symbol to list of feature names in
        the order defined in the YAML file.
        """
        try:
            # Resolve path relative to project root: config/features.yaml
            base_dir = Path(__file__).resolve().parents[2]
            features_path = base_dir / "config" / "features.yaml"

            if not features_path.is_file():
                _logger.error(
                    "Feature definitions file not found at expected location.",
                    extra={"path": str(features_path)},
                )
                self._feature_definitions = {}
                return

            with features_path.open("r", encoding="utf-8") as f:
                raw_defs = yaml.safe_load(f) or {}

            feature_definitions: Dict[str, List[str]] = {}

            if not isinstance(raw_defs, dict):
                _logger.error(
                    "Feature definitions YAML has invalid format (expected mapping).",
                    extra={"type": type(raw_defs).__name__},
                )
                self._feature_definitions = {}
                return

            for chain_key, feature_list in raw_defs.items():
                if not isinstance(chain_key, str):
                    continue
                if not isinstance(feature_list, list):
                    continue

                upper_chain = chain_key.upper().strip()
                cleaned_features: List[str] = []
                for name in feature_list:
                    if isinstance(name, str):
                        cleaned_features.append(name.strip())
                if cleaned_features:
                    feature_definitions[upper_chain] = cleaned_features

            self._feature_definitions = feature_definitions

            _logger.info(
                "Loaded feature definitions from config/features.yaml.",
                extra={
                    "chains": sorted(self._feature_definitions.keys()),
                    "definition_counts": {
                        chain: len(names) for chain, names in self._feature_definitions.items()
                    },
                },
            )
        except Exception as exc:  # noqa: BLE001
            _logger.error(
                "Failed to load feature definitions from config/features.yaml.",
                extra={"error": str(exc)},
            )
            self._feature_definitions = {}

    def get_expected_features(self, chain: str) -> List[str]:
        """
        Return the ordered list of expected feature names for a given chain.

        Parameters
        ----------
        chain : str
            Chain symbol ("BTC" or "ETH"), case-insensitive.

        Returns
        -------
        List[str]
            Ordered list of feature names defined for the chain.

        Raises
        ------
        ValueError
            If no feature definitions are available for the given chain.
        """
        if not isinstance(chain, str):
            raise ValueError("Chain must be a string when requesting expected features.")

        chain_upper = chain.upper().strip()
        expected = self._feature_definitions.get(chain_upper)

        if expected is None:
            raise ValueError(
                f"No feature definitions available for chain '{chain_upper}'. "
                "Ensure config/features.yaml is correctly configured."
            )

        return list(expected)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def compute_daily_features(
        self,
        chain: str,
        target_date: date,
        normalize: bool = True,
    ) -> Dict[str, float]:
        """
        Compute and return (optionally normalized) daily features for one chain and date.

        Steps
        -----
        1. Validate chain (must be present in CHAINS).
        2. Validate target_date (must not be in the future, using UTC date).
        3. Fetch aggregated features from DataService.load_daily_aggregates().
        4. Validate that the returned feature set matches the expected feature
           specification for the chain (no missing required features).
           Extra/unknown features are ignored (with a warning).
        5. If normalize=True and normalization params are available: apply normalization.
           - If normalize=True but params are not loaded: raise RuntimeError.
        6. Validate the resulting feature vector via validators.validate_feature_vector().
        7. Return features as Dict[str, float], ordered according to the
           feature definitions in config/features.yaml.

        Parameters
        ----------
        chain : str
            Chain symbol ("BTC" or "ETH"), case-insensitive.
        target_date : date
            Date for feature computation.
        normalize : bool
            If True (default), normalize features. Set to False during training
            to obtain raw features.

        Returns
        -------
        Dict[str, float]
            Mapping from feature name to value.

        Raises
        ------
        ValueError
            If chain is not supported or target_date is in the future.
        RuntimeError
            If DataService does not return data, features are incomplete,
            or normalization fails.
        """
        if not isinstance(chain, str):
            raise ValueError("Chain must be a string.")

        chain_upper = chain.upper().strip()

        # Validate chain against settings.
        if chain_upper not in CHAINS:
            raise ValueError(f"Unsupported chain '{chain}'. Supported chains: {CHAINS}.")

        # Validate date is not in the future (based on UTC).
        today_utc = datetime.now(timezone.utc).date()
        if target_date > today_utc:
            raise ValueError(
                f"target_date {target_date.isoformat()} cannot be in the future "
                f"(today is {today_utc.isoformat()} in UTC)."
            )

        _logger.info(
            "Computing daily features",
            extra={"chain": chain_upper, "date": target_date.isoformat()},
        )

        # Retrieve expected feature names for this chain.
        try:
            expected_features = self.get_expected_features(chain_upper)
        except ValueError as exc:
            _logger.error(
                "Failed to compute features (missing feature definitions).",
                extra={
                    "chain": chain_upper,
                    "date": target_date.isoformat(),
                    "error": str(exc),
                },
            )
            raise RuntimeError("Feature definitions missing for chain.") from exc

        # Step 3: Fetch aggregated daily features from DataService.
        try:
            aggregated = self._data_service.load_daily_aggregates(chain_upper, target_date)
        except Exception as exc:  # noqa: BLE001
            _logger.error(
                "Failed to compute features (DataService.load_daily_aggregates() failed).",
                extra={
                    "chain": chain_upper,
                    "date": target_date.isoformat(),
                    "error": str(exc),
                },
            )
            raise RuntimeError("Failed to load daily aggregates from DataService.") from exc

        if aggregated is None:
            _logger.warning(
                "Missing data for date",
                extra={"chain": chain_upper, "date": target_date.isoformat()},
            )
            raise RuntimeError(f"No aggregated feature data available for {chain_upper} on {target_date}.")

        # Support aggregated as either pandas DataFrame (with a single row) or dict.
        features: Dict[str, float]

        if isinstance(aggregated, pd.DataFrame):
            if aggregated.empty:
                _logger.warning(
                    "Missing data for date (empty DataFrame)",
                    extra={"chain": chain_upper, "date": target_date.isoformat()},
                )
                raise RuntimeError(
                    f"Aggregated feature DataFrame is empty for {chain_upper} on {target_date}."
                )
            # Use the first row; drop any 'date' column if present.
            first_row = aggregated.iloc[0]
            features = {
                col: float(first_row[col])
                for col in aggregated.columns
                if col != "date"
            }
        elif isinstance(aggregated, dict):
            # Directly convert to float values.
            try:
                features = {k: float(v) for k, v in aggregated.items()}
            except (TypeError, ValueError) as exc:
                _logger.error(
                    "Failed to compute features (non-numeric values in aggregated dict).",
                    extra={
                        "chain": chain_upper,
                        "date": target_date.isoformat(),
                        "error": str(exc),
                    },
                )
                raise RuntimeError("Aggregated features contain non-numeric values.") from exc
        else:
            _logger.error(
                "Failed to compute features (unsupported aggregated data type).",
                extra={
                    "chain": chain_upper,
                    "date": target_date.isoformat(),
                    "error": f"Unsupported type: {type(aggregated)!r}",
                },
            )
            raise RuntimeError(
                f"Unsupported aggregated data type from DataService: {type(aggregated)!r}."
            )

        if not features:
            _logger.warning(
                "Missing data for date (no feature columns after extraction).",
                extra={"chain": chain_upper, "date": target_date.isoformat()},
            )
            raise RuntimeError(
                f"No feature columns available for {chain_upper} on {target_date}."
            )

        # Step 4: Enforce expected feature set: detect missing and extra features,
        # and order features consistently according to config/features.yaml.
        feature_names_present = set(features.keys())
        expected_set = set(expected_features)

        missing_features = expected_set - feature_names_present
        if missing_features:
            _logger.error(
                "Missing required features for chain/date.",
                extra={
                    "chain": chain_upper,
                    "date": target_date.isoformat(),
                    "missing_features": sorted(missing_features),
                },
            )
            raise RuntimeError(
                f"Missing required features for {chain_upper} on {target_date}: "
                f"{sorted(missing_features)}"
            )

        extra_features = feature_names_present - expected_set
        if extra_features:
            _logger.warning(
                "Extra/unknown features returned from DataService; they will be ignored.",
                extra={
                    "chain": chain_upper,
                    "date": target_date.isoformat(),
                    "extra_features": sorted(extra_features),
                },
            )

        # Reconstruct ordered feature dict using the expected feature list,
        # ignoring any extra features.
        ordered_features: Dict[str, float] = {}
        for name in expected_features:
            ordered_features[name] = float(features[name])

        features = ordered_features

        # Step 5: Optional normalization.
        if normalize:
            if self._has_normalizer_params:
                try:
                    features = self.normalize_features(features)
                except ValueError as exc:
                    _logger.error(
                        "Failed to compute features (normalization produced invalid values).",
                        extra={
                            "chain": chain_upper,
                            "date": target_date.isoformat(),
                            "error": str(exc),
                        },
                    )
                    raise RuntimeError("Normalization failed due to invalid feature values.") from exc
            else:
                # Normalization requested but parameters are not available:
                # this is a hard failure to avoid silent misuse of raw features.
                _logger.error(
                    "Normalization requested but parameters are not loaded.",
                    extra={"chain": chain_upper, "date": target_date.isoformat()},
                )
                raise RuntimeError(
                    "Normalization was requested but parameters are not loaded. "
                    "Run fit_normalizer() or load_normalizer_params() first."
                )

        # Step 6: Validate resulting feature vector strictly.
        try:
            validators.validate_feature_vector(features)
        except ValueError as exc:
            _logger.error(
                "Failed to compute features (feature validation failed).",
                extra={
                    "chain": chain_upper,
                    "date": target_date.isoformat(),
                    "error": str(exc),
                },
            )
            raise RuntimeError("Feature validation failed for computed feature vector.") from exc

        _logger.debug(
            "Features computed",
            extra={"feature_count": len(features)},
        )
        return features

    def compute_features_for_date_range(
        self,
        chain: str,
        start_date: date,
        end_date: date,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """
        Compute features for a date interval and return as a pandas DataFrame.

        Used primarily for:
        - Model training (normalize=False to obtain raw features).
        - Backfilling of historical features.
        - Batch processing.

        Steps
        -----
        1. Validate inputs.
        2. Iterate over each date in the interval.
        3. For each date: call compute_daily_features().
        4. Collect results into a pandas DataFrame.
        5. Return a DataFrame with a 'date' column plus feature columns.

        Parameters
        ----------
        chain : str
            Chain symbol.
        start_date : date
            Start date (inclusive).
        end_date : date
            End date (inclusive).
        normalize : bool
            If True, normalize each feature vector.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: ["date", "feature1", "feature2", ...].
            Rows are sorted chronologically (oldest first).
            Dates without data are skipped (with a warning log).

        Raises
        ------
        ValueError
            If start_date > end_date or inputs are invalid.
        RuntimeError
            If all dates fail (no data available for any date).
        """
        if not isinstance(chain, str):
            raise ValueError("Chain must be a string.")

        chain_upper = chain.upper().strip()

        if chain_upper not in CHAINS:
            raise ValueError(f"Unsupported chain '{chain}'. Supported chains: {CHAINS}.")

        # Validate date range.
        if start_date > end_date:
            raise ValueError(
                f"start_date {start_date.isoformat()} cannot be after end_date {end_date.isoformat()}."
            )

        validators.validate_date_range(start_date, end_date)

        results: List[Dict[str, Any]] = []
        failed_dates: List[date] = []

        current_date = start_date
        while current_date <= end_date:
            try:
                features = self.compute_daily_features(
                    chain=chain_upper,
                    target_date=current_date,
                    normalize=normalize,
                )
            except RuntimeError:
                # compute_daily_features logs specific reason; here we only record failure.
                failed_dates.append(current_date)
            except ValueError:
                # Validation or input-related errors; treat as failure for this date.
                failed_dates.append(current_date)
            else:
                # Successful computation: prepend date into the record.
                row: Dict[str, Any] = {"date": current_date}
                row.update(features)
                results.append(row)

            current_date = current_date + timedelta(days=1)

        if not results:
            # All dates failed: log each failure and raise RuntimeError.
            for d in failed_dates:
                _logger.error(
                    "Failed to compute features for date in range.",
                    extra={"chain": chain_upper, "date": d.isoformat()},
                )
            raise RuntimeError(
                f"Failed to compute features for any date in range "
                f"{start_date.isoformat()} to {end_date.isoformat()} for chain {chain_upper}."
            )

        if failed_dates:
            # Not all dates had data; log a high-level warning summarizing.
            _logger.warning(
                "Some dates in range had missing or invalid data.",
                extra={
                    "chain": chain_upper,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "failed_dates": [d.isoformat() for d in failed_dates],
                },
            )

        df = pd.DataFrame(results)
        # Ensure chronological sorting by date.
        df = df.sort_values("date").reset_index(drop=True)
        return df

    def normalize_features(
        self,
        features: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Normalize a feature vector using loaded normalization parameters.

        This is a thin wrapper around self._normalizer.transform() for external use.

        Parameters
        ----------
        features : Dict[str, float]
            Raw feature vector.

        Returns
        -------
        Dict[str, float]
            Normalized feature vector.

        Raises
        ------
        ValueError
            If features are empty, non-numeric, or contain NaN/inf after normalization.
        RuntimeError
            If normalization parameters are not loaded.
        """
        if not self._has_normalizer_params:
            raise RuntimeError(
                "Normalization parameters are not loaded. "
                "Call load_normalizer_params() or fit_normalizer() first."
            )

        if not features:
            raise ValueError("Empty features dict")

        # Ensure numeric input values before normalization.
        cleaned_input: Dict[str, float] = {}
        for key, value in features.items():
            try:
                cleaned_input[key] = float(value)
            except (TypeError, ValueError) as exc:
                _logger.error(
                    "Non-numeric value encountered before normalization.",
                    extra={"feature": key, "error": str(exc)},
                )
                raise ValueError(f"Feature '{key}' has non-numeric value: {value!r}") from exc

        # Optional: if the normalizer exposes a set of known features, filter unknown ones.
        known_features = None
        if hasattr(self._normalizer, "known_features"):
            try:
                known_features = set(getattr(self._normalizer, "known_features"))
            except TypeError:
                known_features = None

        if known_features is not None:
            filtered_input: Dict[str, float] = {}
            for key, value in cleaned_input.items():
                if key in known_features:
                    filtered_input[key] = value
                else:
                    _logger.warning(
                        "Unknown feature in input; ignoring feature.",
                        extra={"feature": key},
                    )
            cleaned_input = filtered_input

        if not cleaned_input:
            raise ValueError("Empty features dict after filtering unknown features.")

        # Delegate to normalizer.
        try:
            normalized = self._normalizer.transform(cleaned_input)
        except Exception as exc:  # noqa: BLE001
            _logger.error(
                "Error during feature normalization.",
                extra={"error": str(exc)},
            )
            raise RuntimeError("Normalizer.transform() failed.") from exc

        if not isinstance(normalized, dict):
            raise ValueError(
                f"FeatureNormalizer.transform() must return a dict[str, float], got {type(normalized)!r}."
            )

        # Validate that all normalized values are finite numbers.
        normalized_output: Dict[str, float] = {}
        for key, value in normalized.items():
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                _logger.error(
                    "Non-numeric value encountered after normalization.",
                    extra={"feature": key, "error": str(exc)},
                )
                raise ValueError(
                    f"Normalized feature '{key}' has non-numeric value: {value!r}"
                ) from exc

            if not math.isfinite(numeric_value):
                _logger.error(
                    "NaN or infinite value encountered after normalization.",
                    extra={"feature": key},
                )
                raise ValueError(
                    f"Normalized feature '{key}' has non-finite value: {numeric_value!r}"
                )
            normalized_output[key] = numeric_value

        return normalized_output

    def validate_features(
        self,
        features: Dict[str, float],
        chain: Optional[str] = None,
    ) -> bool:
        """
        Validate a feature vector for basic sanity.

        Checks
        ------
        - No NaN values.
        - No inf values.
        - All values are numeric.
        - Values are within loose numeric bounds.
        - If chain is provided, validate that the feature names match the
          expected feature set from config/features.yaml.

        Parameters
        ----------
        features : Dict[str, float]
            Feature vector to validate.
        chain : Optional[str]
            Optional chain symbol ("BTC" or "ETH"). When provided, the feature
            names are checked against the expected feature names for this chain.

        Returns
        -------
        bool
            True if valid, False otherwise.

        Notes
        -----
        - Logs warnings for invalid data but does not raise exceptions.
        - For stricter validation (including raising exceptions), use
          validators.validate_feature_vector() directly.
        """
        if not features:
            _logger.warning("Empty features dict provided to validate_features().")
            return False

        is_valid = True

        # If chain context is provided, validate feature names against the spec.
        if chain is not None:
            try:
                expected_features = set(self.get_expected_features(chain))
            except ValueError as exc:
                _logger.warning(
                    "Unable to validate feature names against spec (missing definitions).",
                    extra={"chain": str(chain), "error": str(exc)},
                )
            else:
                present_features = set(features.keys())
                missing = expected_features - present_features
                extra = present_features - expected_features

                if missing:
                    _logger.warning(
                        "Feature vector is missing expected features.",
                        extra={
                            "chain": str(chain).upper(),
                            "missing_features": sorted(missing),
                        },
                    )
                    is_valid = False

                if extra:
                    _logger.warning(
                        "Feature vector contains extra/unknown features.",
                        extra={
                            "chain": str(chain).upper(),
                            "extra_features": sorted(extra),
                        },
                    )
                    # Presence of extra features is considered a validation issue.
                    is_valid = False

        # Basic numeric and finiteness checks.
        for key, value in features.items():
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                _logger.warning(
                    "Feature has non-numeric value.",
                    extra={"feature": key, "value": repr(value)},
                )
                is_valid = False
                continue

            if not math.isfinite(numeric_value):
                _logger.warning(
                    "Feature has NaN or infinite value.",
                    extra={"feature": key, "value": numeric_value},
                )
                is_valid = False
                continue

            # Very loose "reasonable bounds" check to catch extreme values.
            if abs(numeric_value) > 1e9:
                _logger.warning(
                    "Feature value appears to be out of reasonable bounds.",
                    extra={"feature": key, "value": numeric_value},
                )
                is_valid = False

        # Finally, re-use the stricter validator in a non-raising fashion.
        try:
            validators.validate_feature_vector(features)
        except ValueError as exc:
            _logger.warning(
                "validators.validate_feature_vector() reported invalid features.",
                extra={"error": str(exc)},
            )
            is_valid = False

        return is_valid

    def fit_normalizer(
        self,
        training_data: pd.DataFrame,
        save_params: bool = True,
    ) -> None:
        """
        Fit normalization parameters on training data.

        Used during model training to compute mean/std per feature.

        Steps
        -----
        1. Call self._normalizer.fit(training_data).
        2. If save_params=True, save parameters to NORMALIZATION_DIR.

        Parameters
        ----------
        training_data : pd.DataFrame
            DataFrame with feature columns (no date column).
        save_params : bool
            If True (default), save parameters to disk.

        Raises
        ------
        ValueError
            If training_data is empty or has an invalid format.
        """
        if not isinstance(training_data, pd.DataFrame):
            raise ValueError("training_data must be a pandas DataFrame.")

        if training_data.empty:
            raise ValueError("training_data must not be empty.")

        try:
            self._normalizer.fit(training_data)
        except Exception as exc:  # noqa: BLE001
            _logger.error(
                "Failed to fit normalizer on training data.",
                extra={"error": str(exc)},
            )
            raise ValueError("Failed to fit normalizer on training data.") from exc

        if save_params:
            try:
                self._normalizer.save_params(NORMALIZATION_DIR)
                self._has_normalizer_params = True
            except Exception as exc:  # noqa: BLE001
                _logger.error(
                    "Failed to save normalization parameters to disk.",
                    extra={"error": str(exc)},
                )
                # Raising here is intentional; failure to save params is a configuration
                # or I/O problem and should be addressed before proceeding.
                raise ValueError("Failed to save normalization parameters.") from exc

    def load_normalizer_params(self) -> None:
        """
        Load normalization parameters from disk.

        This is called automatically on startup if a params file exists,
        and can also be called explicitly to load updated parameters.

        Raises
        ------
        FileNotFoundError
            If the params file does not exist.
        ValueError
            If the params file has an invalid format.
        """
        # Delegate to normalizer. Any FileNotFoundError / ValueError will propagate
        # to the caller, which is the desired behavior for explicit calls.
        self._normalizer.load_params(NORMALIZATION_DIR)
        self._has_normalizer_params = True
