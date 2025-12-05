from __future__ import annotations

from datetime import date
from typing import Any, Dict, Optional

import numpy as np
import polars as pl

from config import settings
from src.models.model_loader import ModelLoader
from src.models.signal_generator import SignalGenerator
from src.services.data_service import DataService
from src.utils import validators
from src.utils.exceptions import ConfigurationError
from src.utils.logger import get_logger


class SignalService:
    """
    Orchestrates the daily inference and signal generation flow for a single
    chain and date.

    Responsibilities
    ----------------
    - Load aggregated daily features from tmp/ via DataService.
    - Load chain-specific ML model and normalization parameters via ModelLoader.
    - Normalize features into a model-ready NumPy array.
    - Run model.predict() (through an abstraction) to obtain a raw risk score.
    - Delegate signal generation to SignalGenerator.
    - Validate the final signals structure via validators.

    Constraints
    -----------
    - This service MUST NOT persist anything (no DB writes, no file writes).
      Persistence of signals belongs in a higher-level orchestration layer
      (e.g., SignalService / pipelines/03_run_daily_inference.py).
    - This service MUST NOT contain business rules for signal semantics
      (regimes, alerts, smoothing, hysteresis); these belong in SignalGenerator.
    """

    def __init__(self) -> None:
        """
        Initialize SignalService with required collaborators:

        - Logger (centralized via src.utils.logger.get_logger).
        - ModelLoader: loads ML models and normalization parameters.
        - SignalGenerator: converts raw model output + features to final signals.
        - DataService: loads aggregated features from tmp/ for a given chain/date.
        """
        self._logger = get_logger(f"{__name__}.SignalService")

        self._model_loader = ModelLoader()
        self._signal_generator = SignalGenerator()
        self._data_service = DataService()

        # Validate and cache imputation_value configuration eagerly.
        # Even though imputering av rådata nå ligger i DataService/FeatureService,
        # sikrer vi her at en eventuell bruk av settings.IMPUTATION_VALUE andre steder
        # ikke feiler på grunn av en ikke-numerisk konfigurasjon.
        self._imputation_value: Optional[float]
        if hasattr(settings, "IMPUTATION_VALUE"):
            raw_imputation_value = getattr(settings, "IMPUTATION_VALUE")
            if raw_imputation_value is None:
                raise ValueError(
                    "settings.IMPUTATION_VALUE must not be None; "
                    "it must be configured to a numeric value."
                )
            try:
                self._imputation_value = float(raw_imputation_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "settings.IMPUTATION_VALUE must be a numeric type; "
                    f"got {raw_imputation_value!r} of type "
                    f"{type(raw_imputation_value).__name__}."
                ) from exc
        else:
            # If the setting is fully absent, treat as unused.
            self._imputation_value = None

        self._logger.debug(
            "SignalService initialized with ModelLoader=%s, SignalGenerator=%s, DataService=%s",
            type(self._model_loader).__name__,
            type(self._signal_generator).__name__,
            type(self._data_service).__name__,
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run_daily_inference(self, chain: str, target_date: date) -> Optional[Dict[str, Any]]:
        """
        Execute the full inference flow for one chain and date.

        Steps
        -----
        1. Load aggregated daily features from tmp/ using DataService.
        2. Load chain-specific model and normalization parameters via ModelLoader.
        3. Normalize features using _normalize_features(...).
        4. Validate normalized features.
        5. Obtain scalar prediction via model/ModelLoader abstraction.
        6. Generate final signals via SignalGenerator.generate_all(...).
        7. Validate final signals via validators.validate_signals(...).

        Parameters
        ----------
        chain:
            Chain identifier, e.g. "BTC" or "ETH".
        target_date:
            The date of the features and resulting signals.

        Returns
        -------
        Optional[Dict[str, Any]]
            Validated signal dictionary for the given chain/date, or None if
            features are missing or a recoverable error occurs.
        """
        chain_key = chain.upper()
        self._logger.info(
            "Starting daily inference for chain='%s', date='%s'.",
            chain_key,
            target_date,
        )

        # ------------------------------------------------------------------
        # 1. Load aggregated features
        # ------------------------------------------------------------------
        try:
            features_df = self._data_service.load_daily_aggregates(chain_key, target_date)
        except (OSError, ValueError, RuntimeError) as exc:
            self._logger.error(
                "Error while loading daily aggregates for chain='%s', date='%s': %s",
                chain_key,
                target_date,
                exc,
            )
            return None

        if features_df is None:
            self._logger.info(
                "No aggregated features found for chain='%s', date='%s'; "
                "skipping inference.",
                chain_key,
                target_date,
            )
            return None

        if not isinstance(features_df, pl.DataFrame):
            self._logger.error(
                "Unsupported features type '%s' for chain='%s', date='%s'; "
                "expected Polars DataFrame.",
                type(features_df).__name__,
                chain_key,
                target_date,
            )
            return None

        if features_df.is_empty():
            self._logger.info(
                "Aggregated features DataFrame is empty for chain='%s', date='%s'; "
                "skipping inference.",
                chain_key,
                target_date,
            )
            return None

        # ------------------------------------------------------------------
        # 2. Load model and normalization parameters
        # ------------------------------------------------------------------
        try:
            model = self._model_loader.load_model(chain_key)
            norm_params = self._model_loader.load_normalization_params(chain_key)
        except (OSError, IOError, ValueError) as exc:
            self._logger.error(
                "Failed to load model or normalization params for chain='%s': %s",
                chain_key,
                exc,
            )
            return None

        # ------------------------------------------------------------------
        # 3. Normalize features
        # ------------------------------------------------------------------
        try:
            normalized_data = self._normalize_features(features_df, norm_params)
            if normalized_data.shape[0] != 1:
                # Defensive double-check for daily, single-row inference.
                raise ValueError(
                    "Expected exactly one feature row for daily inference, "
                    f"got {normalized_data.shape[0]} rows."
                )
        except (ValueError, TypeError, ConfigurationError) as exc:
            self._logger.error(
                "Normalization failed for chain='%s', date='%s': %s",
                chain_key,
                target_date,
                exc,
            )
            return None

        # ------------------------------------------------------------------
        # 4. Validate normalized features (degenerate / over-imputed safeguards)
        # ------------------------------------------------------------------
        try:
            validators.validate_input_features(normalized_data)
        except ValueError as exc:
            self._logger.error(
                "Input feature validation failed for chain='%s', date='%s': %s",
                chain_key,
                target_date,
                exc,
            )
            return None

        # ------------------------------------------------------------------
        # 5. Model prediction via abstraction
        # ------------------------------------------------------------------
        try:
            raw_score = self._get_scalar_prediction(model, normalized_data)
        except (ValueError, TypeError, RuntimeError) as exc:
            self._logger.error(
                "Model prediction failed for chain='%s', date='%s': %s",
                chain_key,
                target_date,
                exc,
            )
            return None

        self._logger.debug(
            "Model raw_score for chain='%s', date='%s': %s",
            chain_key,
            target_date,
            raw_score,
        )

        # ------------------------------------------------------------------
        # 6. Signal generation (delegated)
        # ------------------------------------------------------------------
        try:
            final_signals = self._signal_generator.generate_all(
                raw_score=raw_score,
                features=features_df,
                target_date=target_date,
                chain=chain_key,
            )
        except (ValueError, RuntimeError) as exc:
            self._logger.error(
                "Signal generation failed for chain='%s', date='%s': %s",
                chain_key,
                target_date,
                exc,
            )
            return None

        if not isinstance(final_signals, dict):
            self._logger.error(
                "SignalGenerator.generate_all returned non-dict result for "
                "chain='%s', date='%s': %r",
                chain_key,
                target_date,
                type(final_signals),
            )
            return None

        # ------------------------------------------------------------------
        # 7. Validate final signals
        # ------------------------------------------------------------------
        try:
            validators.validate_signals(final_signals)
        except ValueError as exc:
            self._logger.error(
                "Signal validation failed for chain='%s', date='%s': %s",
                chain_key,
                target_date,
                exc,
            )
            return None

        self._logger.info(
            "Daily inference completed successfully for chain='%s', date='%s'.",
            chain_key,
            target_date,
        )
        return final_signals

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _normalize_features(
        self,
        features: pl.DataFrame,
        params: Dict[str, Dict[str, float]],
    ) -> np.ndarray:
        """
        Normalize features using Z-score/StandardScaler semantics.

        Steps
        -----
        1. For each feature listed in `params`, select the corresponding column
           from the Polars DataFrame.
        2. Validate that each feature has numeric mean/std in `params`.
        3. Apply standard scaling:
               X_norm = (X - mean) / std
        4. Edge cases:
           - If std == 0 or not finite, set X_norm = 0 for that column.
           - If a feature column is missing in `features`, it is skipped with a warning.
        5. Build a new Polars DataFrame containing only the normalized feature
           columns (in deterministic order based on `params` keys) and convert
           to a NumPy array of dtype float32.

        Parameters
        ----------
        features:
            Polars DataFrame containing one row per date (usually one row here).
        params:
            Dict[feature_name, {"mean": float, "std": float}]

        Returns
        -------
        np.ndarray
            2D NumPy array of normalized features, compatible with model.predict().
        """
        if features.is_empty():
            raise ValueError("Cannot normalize an empty features DataFrame.")

        feature_exprs: list[pl.Expr] = []
        ordered_feature_names: list[str] = []

        for feature_name, stats in params.items():
            if feature_name not in features.columns:
                # Missing feature in current dataset; log and continue.
                self._logger.warning(
                    "Feature '%s' from normalization params is missing in "
                    "current features DataFrame; skipping normalization for this feature.",
                    feature_name,
                )
                continue

            if not isinstance(stats, dict):
                raise ConfigurationError(
                    f"Normalization params for feature '{feature_name}' must be a dict "
                    f"with 'mean' and 'std' keys; got {type(stats).__name__}."
                )

            if "mean" not in stats or "std" not in stats:
                raise ConfigurationError(
                    f"Normalization params for feature '{feature_name}' must contain "
                    "'mean' and 'std' keys."
                )

            try:
                mean = float(stats["mean"])
                std = float(stats["std"])
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(
                    f"Normalization params for feature '{feature_name}' must be numeric; "
                    f"got mean={stats.get('mean')!r}, std={stats.get('std')!r}."
                ) from exc

            # The DataService contract guarantees pl.Float32 columns for features.
            # We therefore do not cast here, to avoid hiding type errors from collaborators.
            col_expr = pl.col(feature_name)

            if std == 0.0 or not np.isfinite(std):
                # Degenerate standard deviation: set normalized column to 0.
                self._logger.warning(
                    "Std for feature '%s' is zero or non-finite (std=%s); "
                    "setting normalized values to 0.",
                    feature_name,
                    std,
                )
                norm_expr = pl.lit(0.0).alias(feature_name)
            else:
                norm_expr = ((col_expr - mean) / std).alias(feature_name)

            feature_exprs.append(norm_expr)
            ordered_feature_names.append(feature_name)

        if not feature_exprs:
            raise ValueError(
                "No overlapping features between input DataFrame and normalization parameters."
            )

        # Select only the normalized feature columns in deterministic order.
        normalized_df = features.select(feature_exprs)
        normalized_array = normalized_df.to_numpy(dtype=np.float32)

        if normalized_array.ndim != 2:
            raise ValueError(
                "Normalized feature array must be 2D, got shape "
                f"{normalized_array.shape} instead."
            )

        if normalized_array.shape[0] != 1:
            # Enforce single-row semantics for daily inference at this level as well.
            raise ValueError(
                "Expected normalized feature array with exactly one row for daily "
                f"inference, got shape {normalized_array.shape}."
            )

        self._logger.debug(
            "Feature normalization completed. Output shape: %s; features: %s",
            normalized_array.shape,
            ordered_feature_names,
        )
        return normalized_array

    def _get_scalar_prediction(self, model: Any, normalized_data: np.ndarray) -> float:
        """
        Obtain a scalar prediction for a single-row feature vector.

        Prefer delegating to ModelLoader.get_scalar_prediction if available,
        otherwise fall back to a robust local interpretation of model.predict(...)
        output. This preserves the architectural intent that model-specific
        output parsing lives close to the model-loading layer, while remaining
        backwards compatible with loaders that do not yet implement the helper.
        """
        # Prefer delegation to ModelLoader if it exposes the helper.
        get_scalar = getattr(self._model_loader, "get_scalar_prediction", None)
        if callable(get_scalar):
            self._logger.debug(
                "Delegating scalar prediction to ModelLoader.get_scalar_prediction."
            )
            raw_score = get_scalar(model, normalized_data)
            try:
                return float(raw_score)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "ModelLoader.get_scalar_prediction returned a non-numeric value: "
                    f"{raw_score!r}"
                ) from exc

        # Fallback path: call model.predict() here and interpret output shape robustly.
        self._logger.debug(
            "ModelLoader.get_scalar_prediction not available; "
            "using local prediction parsing logic."
        )
        try:
            raw_pred = model.predict(normalized_data)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Model prediction failed: {exc}") from exc

        # Interpret the raw prediction output and reduce to a scalar.
        if isinstance(raw_pred, np.ndarray):
            if raw_pred.ndim == 0:
                return float(raw_pred)
            if raw_pred.ndim == 1 and raw_pred.size == 1:
                return float(raw_pred[0])
            if raw_pred.ndim == 2 and raw_pred.shape == (1, 1):
                return float(raw_pred[0, 0])
            raise ValueError(
                f"Unexpected prediction array shape {raw_pred.shape} for single-row inference."
            )

        if isinstance(raw_pred, (list, tuple)):
            if len(raw_pred) != 1:
                raise ValueError(
                    f"Expected single-element prediction sequence, got length {len(raw_pred)}."
                )
            return float(raw_pred[0])

        # Assume scalar-like value.
        try:
            return float(raw_pred)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Unable to interpret model prediction output as a scalar float: "
                f"value={raw_pred!r}"
            ) from exc
