from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from joblib import load as joblib_load

from src.config.settings import MODEL_DIR, NORMALIZER_PARAMS_PATH
from src.models.predictor import Predictor as ModelPredictor
from src.utils.logger import get_logger

__all__ = ["ModelLoader"]

logger = get_logger(__name__)


class ModelLoader:
    """
    Abstraction layer for loading ML models and normalization parameters.

    Responsibilities
    ----------------
    - Load chain-specific ML models from disk.
    - Load normalization parameters (mean/std per feature) from disk.
    - Provide a helper to obtain a scalar prediction from a model.
    - Cache loaded models and parameters in memory for reuse.

    Notes
    -----
    - Read-only: does not write any files.
    - Operates only on derived artifacts (model binaries, JSON configs).
    - No access to raw blockchain data.
    """

    def __init__(self) -> None:
        """
        Initialize ModelLoader with in-memory caches and collaborators.
        """
        self._model_dir = Path(MODEL_DIR)
        self._norm_params_path = Path(NORMALIZER_PARAMS_PATH)

        # Cache for loaded models, keyed by chain (e.g., "BTC", "ETH").
        self._model_cache: Dict[str, Any] = {}

        # Cache for normalization parameters, keyed by chain.
        # Structure per chain: Dict[feature_name, {"mean": float, "std": float}]
        self._norm_params_cache: Dict[str, Dict[str, Dict[str, float]]] = {}

        # Optional collaborator: ModelPredictor for delegated model loading.
        self._predictor = ModelPredictor()

        logger.debug(
            "ModelLoader initialized with MODEL_DIR=%s, NORMALIZER_PARAMS_PATH=%s",
            str(self._model_dir),
            str(self._norm_params_path),
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def load_model(self, chain: str) -> Any:
        """
        Load a trained model for the given chain from disk.

        Parameters
        ----------
        chain : str
            Chain symbol ("BTC" or "ETH"), case-insensitive.

        Returns
        -------
        Any
            Model object that supports predict(...) and optionally
            predict_proba(...).

        Raises
        ------
        FileNotFoundError
            If version.json or the model file does not exist.
        ValueError
            If version.json is invalid or does not contain the given chain.
        RuntimeError
            If the model cannot be loaded due to corruption or other I/O errors.
        """
        if not isinstance(chain, str):
            raise ValueError("chain must be a string.")

        chain_key = chain.upper().strip()

        # Return from cache if already loaded.
        if chain_key in self._model_cache:
            logger.debug("Model for chain '%s' returned from cache.", chain_key)
            return self._model_cache[chain_key]

        # Read version metadata.
        version_info = self._read_version_json()
        if chain_key not in version_info:
            logger.warning(
                "Chain '%s' not found in version.json.",
                chain_key,
            )
            raise ValueError(
                f"Chain '{chain_key}' not found in version.json. "
                "Cannot determine latest model version."
            )

        chain_meta = version_info[chain_key]
        if not isinstance(chain_meta, dict) or "latest_version" not in chain_meta:
            logger.error(
                "Invalid version metadata for chain '%s' in version.json: %r",
                chain_key,
                chain_meta,
            )
            raise ValueError(
                f"Invalid version metadata for chain '{chain_key}' in version.json."
            )

        latest_version = chain_meta["latest_version"]
        try:
            version_int = int(latest_version)
        except (TypeError, ValueError) as exc:
            logger.error(
                "latest_version for chain '%s' is not an integer: %r",
                chain_key,
                latest_version,
            )
            raise ValueError(
                f"latest_version for chain '{chain_key}' must be an integer, "
                f"got {latest_version!r}."
            ) from exc

        # Construct model file path.
        # Filenames follow the convention: btc_model_v{version}.bin, eth_model_v{version}.bin
        chain_lower = chain_key.lower()
        model_filename = f"{chain_lower}_model_v{version_int}.bin"
        model_path = self._model_dir / model_filename

        if not model_path.is_file():
            logger.error(
                "Model file does not exist for chain '%s': %s",
                chain_key,
                str(model_path),
            )
            raise FileNotFoundError(
                f"Model file not found for chain '{chain_key}': {model_path}"
            )

        # Load model from disk using either ModelPredictor or direct joblib.
        try:
            logger.info(
                "Loading model for chain '%s' from file: %s",
                chain_key,
                str(model_path),
            )

            # Preferred path: delegate loading to ModelPredictor if it provides a compatible method.
            load_method = getattr(self._predictor, "load_model", None)
            if callable(load_method):
                model = load_method(str(model_path))
            else:
                # Fallback: direct joblib.load if predictor does not expose load_model.
                model = joblib_load(str(model_path))

        except FileNotFoundError:
            # Re-raise with additional logging.
            logger.error(
                "Model file not found while attempting to load for chain '%s': %s",
                chain_key,
                str(model_path),
            )
            raise
        except Exception as exc:  # noqa: BLE001
            # Any other failure is treated as a runtime error (e.g., corrupt file).
            logger.critical(
                "Failed to load model for chain '%s' from %s: %s",
                chain_key,
                str(model_path),
                str(exc),
            )
            raise RuntimeError(
                f"Failed to load model for chain '{chain_key}' from {model_path}: {exc}"
            ) from exc

        # Cache and return.
        self._model_cache[chain_key] = model
        logger.debug(
            "Model for chain '%s' loaded and cached successfully.",
            chain_key,
        )
        return model

    def load_normalization_params(self, chain: str) -> Dict[str, Dict[str, float]]:
        """
        Load normalization parameters for the given chain.

        Parameters
        ----------
        chain : str
            Chain symbol ("BTC" or "ETH"), case-insensitive.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Mapping from feature name to a dict with keys "mean" and "std",
            both floats. Example:
                {
                    "whale_flow_1d": {"mean": 0.1, "std": 0.02},
                    "exchange_netflow_7d": {"mean": 1.5, "std": 0.3},
                    ...
                }

        Raises
        ------
        FileNotFoundError
            If the normalization parameters file cannot be found.
        ValueError
            If the parameters file is invalid or does not contain the given chain.
        """
        if not isinstance(chain, str):
            raise ValueError("chain must be a string.")

        chain_key = chain.upper().strip()

        # Return from cache if already loaded.
        if chain_key in self._norm_params_cache:
            logger.debug(
                "Normalization parameters for chain '%s' returned from cache.",
                chain_key,
            )
            return self._norm_params_cache[chain_key]

        if not self._norm_params_path.is_file():
            logger.error(
                "Normalization parameters file not found at path: %s",
                str(self._norm_params_path),
            )
            raise FileNotFoundError(
                f"Normalization parameters file not found: {self._norm_params_path}"
            )

        try:
            with self._norm_params_path.open("r", encoding="utf-8") as f:
                raw_params = json.load(f)
        except json.JSONDecodeError as exc:
            logger.error(
                "Invalid JSON in normalization parameters file '%s': %s",
                str(self._norm_params_path),
                str(exc),
            )
            raise ValueError(
                f"Invalid JSON in normalization parameters file: {self._norm_params_path}"
            ) from exc

        if not isinstance(raw_params, dict):
            logger.error(
                "Normalization parameters JSON must be a mapping; got %s.",
                type(raw_params).__name__,
            )
            raise ValueError("Normalization parameters JSON must be a mapping.")

        if chain_key not in raw_params:
            logger.error(
                "Normalization parameters for chain '%s' not found in file '%s'.",
                chain_key,
                str(self._norm_params_path),
            )
            raise ValueError(
                f"Normalization parameters for chain '{chain_key}' not found "
                f"in {self._norm_params_path}."
            )

        chain_params_raw = raw_params[chain_key]
        if not isinstance(chain_params_raw, dict):
            logger.error(
                "Normalization params for chain '%s' must be a mapping; got %s.",
                chain_key,
                type(chain_params_raw).__name__,
            )
            raise ValueError(
                f"Normalization parameters for chain '{chain_key}' must be a mapping."
            )

        # Validate and coerce mean/std to float; raise if any feature is invalid.
        validated_params: Dict[str, Dict[str, float]] = {}
        for feature_name, stats in chain_params_raw.items():
            if not isinstance(feature_name, str):
                logger.warning(
                    "Skipping normalization params with non-string feature name for chain '%s': %r",
                    chain_key,
                    feature_name,
                )
                continue

            if not isinstance(stats, dict):
                logger.warning(
                    "Normalization stats for feature '%s' (chain '%s') must be a dict; got %s.",
                    feature_name,
                    chain_key,
                    type(stats).__name__,
                )
                raise ValueError(
                    f"Normalization stats for feature '{feature_name}' "
                    f"(chain '{chain_key}') must be a dict."
                )

            if "mean" not in stats or "std" not in stats:
                logger.warning(
                    "Normalization stats for feature '%s' (chain '%s') missing "
                    "'mean' or 'std' keys.",
                    feature_name,
                    chain_key,
                )
                raise ValueError(
                    f"Normalization stats for feature '{feature_name}' "
                    f"(chain '{chain_key}') must contain 'mean' and 'std'."
                )

            try:
                mean_val = float(stats["mean"])
                std_val = float(stats["std"])
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "Non-numeric mean/std for feature '%s' (chain '%s'): mean=%r, std=%r",
                    feature_name,
                    chain_key,
                    stats.get("mean"),
                    stats.get("std"),
                )
                raise ValueError(
                    f"Normalization stats for feature '{feature_name}' "
                    f"(chain '{chain_key}') must be numeric."
                ) from exc

            validated_params[feature_name] = {"mean": mean_val, "std": std_val}

        if not validated_params:
            logger.error(
                "No valid normalization parameters found for chain '%s' in file '%s'.",
                chain_key,
                str(self._norm_params_path),
            )
            raise ValueError(
                f"No valid normalization parameters found for chain '{chain_key}'."
            )

        # Cache and return.
        self._norm_params_cache[chain_key] = validated_params
        logger.debug(
            "Normalization parameters for chain '%s' loaded and cached successfully.",
            chain_key,
        )
        return validated_params

    def get_scalar_prediction(self, model: Any, X: np.ndarray) -> float:
        """
        Obtain a scalar prediction from the given model and input array.

        Parameters
        ----------
        model : Any
            Trained model object. May expose predict_proba(X) and/or predict(X).
        X : np.ndarray
            Input feature array with shape (1, n_features).

        Returns
        -------
        float
            Scalar prediction, typically a probability in [0, 1] or a raw score.

        Raises
        ------
        ValueError
            If X has invalid shape or if the model's prediction output cannot
            be interpreted as a scalar.
        RuntimeError
            If the model's prediction method raises an unexpected exception.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy ndarray.")

        if X.ndim != 2 or X.shape[0] != 1:
            raise ValueError(
                f"X must have shape (1, n_features) for single-row inference; got {X.shape}."
            )

        # 1) Try predict_proba if available (typical for classifiers).
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "model.predict_proba(...) failed: %s",
                    str(exc),
                )
                raise RuntimeError(f"model.predict_proba failed: {exc}") from exc

            proba_arr = np.asarray(proba)
            if proba_arr.ndim == 2:
                if proba_arr.shape[0] != 1:
                    raise ValueError(
                        "predict_proba returned array with unexpected number of rows: "
                        f"{proba_arr.shape[0]}."
                    )
                # Common case: shape (1, 2) for binary classifier; take class 1.
                if proba_arr.shape[1] == 2:
                    return float(proba_arr[0, 1])
                # Fallback: take probability of the last class.
                return float(proba_arr[0, -1])
            elif proba_arr.ndim == 1:
                if proba_arr.size == 1:
                    return float(proba_arr[0])
                return float(proba_arr[-1])
            elif proba_arr.ndim == 0:
                return float(proba_arr)

            raise ValueError(
                f"predict_proba returned array with unsupported shape: {proba_arr.shape}."
            )

        # 2) Fallback to predict(...) for models without predict_proba.
        if not hasattr(model, "predict"):
            raise ValueError(
                "Model does not implement predict_proba or predict; cannot obtain scalar prediction."
            )

        try:
            pred = model.predict(X)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "model.predict(...) failed: %s",
                str(exc),
            )
            raise RuntimeError(f"model.predict failed: {exc}") from exc

        pred_arr = np.asarray(pred)

        if pred_arr.ndim == 2:
            if pred_arr.shape[0] == 1 and pred_arr.shape[1] == 1:
                return float(pred_arr[0, 0])
            if pred_arr.shape[0] == 1:
                # If multiple outputs for a single row, take the last one.
                return float(pred_arr[0, -1])
            raise ValueError(
                f"predict returned array with unsupported shape for single-row inference: {pred_arr.shape}."
            )
        if pred_arr.ndim == 1:
            if pred_arr.size == 1:
                return float(pred_arr[0])
            # If multiple outputs, take the last element as a conservative default.
            return float(pred_arr[-1])
        if pred_arr.ndim == 0:
            return float(pred_arr)

        raise ValueError(
            f"predict returned array with unsupported shape: {pred_arr.shape}."
        )

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _read_version_json(self) -> Dict[str, Dict[str, Any]]:
        """
        Read and parse version.json from the model directory.

        Expected JSON structure:
            {
                "BTC": {"latest_version": 3},
                "ETH": {"latest_version": 2}
            }

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Parsed JSON mapping chain symbols to metadata dicts.

        Raises
        ------
        FileNotFoundError
            If version.json does not exist.
        ValueError
            If the JSON is invalid or not a mapping.
        """
        version_path = self._model_dir / "version.json"

        if not version_path.is_file():
            logger.error(
                "version.json not found in MODEL_DIR: %s",
                str(version_path),
            )
            raise FileNotFoundError(
                f"version.json not found in MODEL_DIR: {version_path}"
            )

        try:
            with version_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            logger.error(
                "Invalid JSON in version.json at '%s': %s",
                str(version_path),
                str(exc),
            )
            raise ValueError(
                f"Invalid JSON in version.json: {version_path}"
            ) from exc

        if not isinstance(data, dict):
            logger.error(
                "version.json must contain a JSON object (mapping); got %s.",
                type(data).__name__,
            )
            raise ValueError("version.json must contain a JSON object (mapping).")

        logger.debug(
            "version.json loaded successfully from %s with chains: %s",
            str(version_path),
            list(data.keys()),
        )
        return data
