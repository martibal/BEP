"""
src/models/trainer.py

ModelTrainer: core training logic for ON-CHAIN SUPER SIGNALS™.

Responsibilities
----------------
- Initialize a chain-specific ML model based on configuration.
- Train the model with support for validation-based early stopping.
- Persist the trained model as a binary artifact in MODEL_DIR.

Data minimization
-----------------
- Operates only on aggregated, in-memory numpy arrays (features/targets).
- NEVER reads, writes, or persists raw blockchain data.
- Only writes model binaries (.bin) to MODEL_DIR.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np

# Top-level imports of ML libraries (per review requirement A.1)
try:  # pragma: no cover - environment dependent
    import xgboost as xgb
except ImportError:  # pragma: no cover - environment dependent
    xgb = None

try:  # pragma: no cover - environment dependent
    import lightgbm as lgb
except ImportError:  # pragma: no cover - environment dependent
    lgb = None

from src.config.settings import MODEL_DIR
from src.models.predictor import _ModelLike
from src.utils import validators
from src.utils.logger import get_logger

_logger = get_logger(__name__)


class TrainingError(Exception):
    """
    Domain-specific exception for all model training related errors.

    Used to wrap lower-level ML library exceptions and configuration issues
    in a consistent, catchable error type.
    """


class ModelTrainer:
    """
    Encapsulates model initialization, training and artifact persistence
    for a single chain.

    Primary public API (per architectural spec)
    -------------------------------------------
    - __init__(chain, config)
    - train(X_train, y_train, X_val, y_val) -> trained model
    - save_model_artifact(model, model_version) -> Path

    Backwards compatibility
    -----------------------
    For compatibility with earlier pipeline code, the following thin wrappers
    are also provided:

    - train_model(X_train, y_train, config) -> trained model
    - save_model(model, path) -> None

    These wrappers delegate to the new, more explicit API where possible.
    """

    def __init__(self, chain: str, config: Dict[str, Any]) -> None:
        """
        Initialize a ModelTrainer for a specific chain.

        Parameters
        ----------
        chain:
            Chain identifier ("BTC" or "ETH").
        config:
            Configuration dictionary for this chain's model, expected to
            include:
              - "model_type": "XGBoostClassifier" | "LightGBMClassifier"
              - "hyperparameters": dict of model hyperparameters
              - "early_stopping_rounds": int (optional)
        """
        validators.validate_chain(chain)
        chain_upper = chain.upper()
        if chain_upper not in {"BTC", "ETH"}:
            raise TrainingError(
                f"ModelTrainer supports only BTC and ETH at this time, got: {chain}"
            )

        self.chain = chain_upper
        (
            self.model_type,
            self.hyperparameters,
            self.early_stopping_rounds,
        ) = self._normalize_config(config)

        _logger.info(
            "ModelTrainer initialized",
            extra={
                "chain": self.chain,
                "model_type": self.model_type,
                "has_hyperparameters": bool(self.hyperparameters),
                "early_stopping_rounds": self.early_stopping_rounds,
            },
        )

    # ------------------------------------------------------------------ #
    # Public API (spec)                                                  #
    # ------------------------------------------------------------------ #

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> _ModelLike:
        """
        Train a model with validation-based early stopping.

        Parameters
        ----------
        X_train:
            2D numpy array of training features.
        y_train:
            1D numpy array of training targets (typically binary 0/1).
        X_val:
            2D numpy array of validation features.
        y_val:
            1D numpy array of validation targets.

        Returns
        -------
        _ModelLike
            The trained model instance (e.g. XGBClassifier or LGBMClassifier).

        Raises
        ------
        TrainingError
            If inputs are empty/invalid or if model.fit() fails.
        """
        X_train, y_train, X_val, y_val = self._validate_and_cast_inputs(
            X_train, y_train, X_val, y_val
        )
        model = self._get_model()

        early_stopping_rounds = self.early_stopping_rounds
        eval_set = [(X_val, y_val)]

        _logger.info(
            "Starting model training with early stopping",
            extra={
                "chain": self.chain,
                "model_type": self.model_type,
                "train_rows": int(X_train.shape[0]),
                "val_rows": int(X_val.shape[0]),
                "features": int(X_train.shape[1]),
                "early_stopping_rounds": early_stopping_rounds,
            },
        )

        try:
            if self.model_type == "XGBoostClassifier":
                model = self._fit_xgboost(
                    model,
                    X_train,
                    y_train,
                    eval_set,
                    early_stopping_rounds=early_stopping_rounds,
                )
            elif self.model_type == "LightGBMClassifier":
                model = self._fit_lightgbm(
                    model,
                    X_train,
                    y_train,
                    eval_set,
                    early_stopping_rounds=early_stopping_rounds,
                )
            else:  # pragma: no cover - guarded by _get_model
                raise TrainingError(f"Unsupported model_type: {self.model_type}")
        except TrainingError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            _logger.error(
                "Unexpected error during model.fit()",
                extra={
                    "chain": self.chain,
                    "model_type": self.model_type,
                    "error": str(exc),
                },
            )
            raise TrainingError(f"Model training failed: {exc}") from exc

        best_iteration = self._get_best_iteration(model)
        _logger.info(
            "Model training completed",
            extra={
                "chain": self.chain,
                "model_type": self.model_type,
                "best_iteration": best_iteration,
            },
        )

        return model

    def save_model_artifact(self, model: _ModelLike, model_version: str) -> Path:
        """
        Persist the trained model as a binary artifact in MODEL_DIR.

        Data minimization:
        - Only the model binary is stored.
        - No training data or raw blockchain data is persisted.

        Parameters
        ----------
        model:
            Trained model instance to serialize.
        model_version:
            Semantic version string (e.g. "1.0.0") provided by caller.

        Returns
        -------
        Path
            Full path to the saved model file.

        Raises
        ------
        TrainingError
            If the model cannot be serialized or written to disk.
        """
        model_dir = Path(MODEL_DIR)
        filename = f"{self.chain.lower()}_model_v{model_version}.bin"
        model_path = model_dir / filename

        # Ensure target directory exists before saving (robust I/O)
        try:
            model_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover - defensive
            _logger.error(
                "Failed to create model directory",
                extra={"chain": self.chain, "path": str(model_path.parent), "error": str(exc)},
            )
            raise TrainingError(
                f"Failed to create model directory {model_path.parent}"
            ) from exc

        try:
            joblib.dump(model, model_path)
        except Exception as exc:  # pragma: no cover - defensive
            _logger.error(
                "Failed to save model artifact",
                extra={"chain": self.chain, "path": str(model_path), "error": str(exc)},
            )
            raise TrainingError(
                f"Failed to save model artifact to {model_path}"
            ) from exc

        _logger.info(
            "Model artifact saved",
            extra={
                "chain": self.chain,
                "path": str(model_path),
                "model_version": model_version,
            },
        )
        return model_path

    # ------------------------------------------------------------------ #
    # Backwards compatible wrappers (for existing pipeline code)         #
    # ------------------------------------------------------------------ #

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        config: Optional[Dict[str, Any]] = None,
    ) -> _ModelLike:
        """
        Backwards compatible wrapper used by older training pipeline.

        This method does *not* use a validation set or early stopping.
        It simply:
        - Optionally merges `config` into the existing configuration,
        - Instantiates the model,
        - Fits on (X_train, y_train),
        - Returns the trained model.

        Parameters
        ----------
        X_train:
            2D numpy array of training features.
        y_train:
            1D numpy array of training targets.
        config:
            Optional configuration dictionary with model hyperparameters
            (e.g. xgboost section from model_config.yaml).

        Returns
        -------
        _ModelLike
            Trained model instance.

        Raises
        ------
        TrainingError
            On invalid inputs or training failure.
        """
        if config:
            # Normalize provided config hyperparameters to avoid numpy scalar issues.
            extra_hp = self._normalize_hyperparameters(config)
            merged_hp = {**self.hyperparameters, **extra_hp}
            self.hyperparameters = merged_hp

        X_train = self._ensure_2d_array(X_train, "X_train")
        y_train = self._ensure_1d_array(y_train, "y_train")

        if X_train.shape[0] == 0 or y_train.shape[0] == 0:
            raise TrainingError("Training data must not be empty")

        if X_train.shape[0] != y_train.shape[0]:
            raise TrainingError(
                f"X_train and y_train must have the same number of rows "
                f"(got {X_train.shape[0]} vs {y_train.shape[0]})"
            )

        model = self._get_model()

        _logger.info(
            "Starting model training via legacy train_model() wrapper",
            extra={
                "chain": self.chain,
                "model_type": self.model_type,
                "train_rows": int(X_train.shape[0]),
                "features": int(X_train.shape[1]),
            },
        )

        try:
            model.fit(X_train, y_train)
        except Exception as exc:  # pragma: no cover - defensive
            _logger.error(
                "Model.fit() failed in train_model wrapper",
                extra={"chain": self.chain, "error": str(exc)},
            )
            raise TrainingError(f"Model training failed: {exc}") from exc

        _logger.info(
            "Model training via train_model() completed",
            extra={"chain": self.chain, "model_type": self.model_type},
        )
        return model

    @staticmethod
    def save_model(model: _ModelLike, path: str | Path) -> None:
        """
        Backwards compatible static wrapper to persist a model using a given path.

        Parameters
        ----------
        model:
            Trained model instance.
        path:
            Target file path (string or Path).

        Raises
        ------
        TrainingError
            If serialization fails.
        """
        target_path = Path(path)
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover - defensive
            _logger.error(
                "Failed to create directory in save_model()",
                extra={"path": str(target_path.parent), "error": str(exc)},
            )
            raise TrainingError(
                f"Failed to create directory {target_path.parent}"
            ) from exc

        try:
            joblib.dump(model, target_path)
        except Exception as exc:  # pragma: no cover - defensive
            _logger.error(
                "Failed to save model via legacy save_model() wrapper",
                extra={"path": str(target_path), "error": str(exc)},
            )
            raise TrainingError(f"Failed to save model to {target_path}") from exc

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_config(config: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Optional[int]]:
        """
        Normalize the raw config dict into explicit components.

        Fills in sensible defaults for missing fields and normalizes
        hyperparameters (including conversion of numpy scalar types).
        """
        model_type = config.get("model_type", "XGBoostClassifier")
        hyperparameters = config.get("hyperparameters", {}) or {}
        early_stopping_rounds = config.get("early_stopping_rounds", 50)

        if not isinstance(model_type, str):
            raise TrainingError("config['model_type'] must be a string")

        if not isinstance(hyperparameters, Dict):
            raise TrainingError("config['hyperparameters'] must be a dict")

        if early_stopping_rounds is not None and not isinstance(
            early_stopping_rounds, int
        ):
            raise TrainingError(
                "config['early_stopping_rounds'] must be an int or None"
            )

        # Ensure all hyperparameters are plain Python types (no numpy scalars).
        normalized_hp = ModelTrainer._normalize_hyperparameters(hyperparameters)

        return model_type, normalized_hp, early_stopping_rounds

    @staticmethod
    def _normalize_hyperparameters(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure hyperparameters are JSON- and joblib-serializable.

        In particular, convert numpy scalar types (e.g. np.float64, np.int64)
        to native Python float/int, and recursively normalize containers.
        """

        def _convert(value: Any) -> Any:
            # numpy scalar types → native Python
            if isinstance(value, np.generic):
                return value.item()
            # simple containers
            if isinstance(value, list):
                return [_convert(v) for v in value]
            if isinstance(value, tuple):
                return tuple(_convert(v) for v in value)
            if isinstance(value, dict):
                return {k: _convert(v) for k, v in value.items()}
            return value

        return {k: _convert(v) for k, v in params.items()}

    def _get_model(self) -> _ModelLike:
        """
        Instantiate an untrained model based on configuration.

        Supported model_type values:
        - "XGBoostClassifier"
        - "LightGBMClassifier"

        Returns
        -------
        _ModelLike
            Untrained model instance.

        Raises
        ------
        TrainingError
            If the model_type is unsupported or required libraries are missing.
        """
        model_type = self.model_type
        hp = dict(self.hyperparameters)  # shallow copy

        if model_type == "XGBoostClassifier":
            if xgb is None:  # pragma: no cover - environment dependent
                raise TrainingError(
                    "xgboost is not installed but is required for model_type "
                    "'XGBoostClassifier'"
                )

            # Reasonable defaults; user-supplied hyperparameters override.
            default_hp = {
                "n_estimators": 500,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "tree_method": "hist",
                "n_jobs": -1,
            }
            default_hp.update(hp)
            model = xgb.XGBClassifier(**default_hp)

        elif model_type == "LightGBMClassifier":
            if lgb is None:  # pragma: no cover - environment dependent
                raise TrainingError(
                    "lightgbm is not installed but is required for model_type "
                    "'LightGBMClassifier'"
                )

            default_hp = {
                "n_estimators": 500,
                "max_depth": -1,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "boosting_type": "gbdt",
                "objective": "binary",
                "n_jobs": -1,
            }
            default_hp.update(hp)
            model = lgb.LGBMClassifier(**default_hp)

        else:
            raise TrainingError(f"Unsupported model_type: {model_type}")

        _logger.info(
            "Initialized model",
            extra={
                "chain": self.chain,
                "model_type": model_type,
                "hyperparameters": self.hyperparameters,
            },
        )
        return model

    @staticmethod
    def _ensure_2d_array(arr: np.ndarray, name: str) -> np.ndarray:
        """Ensure input is a 2D numpy array."""
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)

        if arr.ndim != 2:
            raise TrainingError(f"{name} must be a 2D array, got shape {arr.shape}")

        return arr

    @staticmethod
    def _ensure_1d_array(arr: np.ndarray, name: str) -> np.ndarray:
        """Ensure input is a 1D numpy array."""
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)

        # Accept (n, 1) as well and reshape to (n,)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.reshape(-1)

        if arr.ndim != 1:
            raise TrainingError(f"{name} must be a 1D array, got shape {arr.shape}")

        return arr

    def _validate_and_cast_inputs(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Validate and normalize train/val inputs."""
        X_train = self._ensure_2d_array(X_train, "X_train")
        X_val = self._ensure_2d_array(X_val, "X_val")
        y_train = self._ensure_1d_array(y_train, "y_train")
        y_val = self._ensure_1d_array(y_val, "y_val")

        if X_train.shape[0] == 0 or y_train.shape[0] == 0:
            raise TrainingError("Training data must not be empty")

        if X_val.shape[0] == 0 or y_val.shape[0] == 0:
            raise TrainingError("Validation data must not be empty")

        if X_train.shape[0] != y_train.shape[0]:
            raise TrainingError(
                f"X_train and y_train must have the same number of rows "
                f"(got {X_train.shape[0]} vs {y_train.shape[0]})"
            )

        if X_val.shape[0] != y_val.shape[0]:
            raise TrainingError(
                f"X_val and y_val must have the same number of rows "
                f"(got {X_val.shape[0]} vs {y_val.shape[0]})"
            )

        if X_train.shape[1] != X_val.shape[1]:
            raise TrainingError(
                f"X_train and X_val must have the same number of features "
                f"(got {X_train.shape[1]} vs {X_val.shape[1]})"
            )

        return X_train, y_train, X_val, y_val

    @staticmethod
    def _fit_xgboost(
        model: _ModelLike,
        X_train: np.ndarray,
        y_train: np.ndarray,
        eval_set: list[tuple[np.ndarray, np.ndarray]],
        early_stopping_rounds: Optional[int],
    ) -> _ModelLike:
        """
        Fit XGBoost classifier with optional early stopping.

        Uses eval_set and explicitly sets eval_metric="auc" to align with
        the AUC-based evaluation requirement.
        """
        fit_kwargs: Dict[str, Any] = {
            "X": X_train,
            "y": y_train,
            "eval_set": eval_set,
            "verbose": False,
            "eval_metric": "auc",
        }
        if early_stopping_rounds is not None and early_stopping_rounds > 0:
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds

        try:
            model.fit(**fit_kwargs)
        except Exception as exc:  # pragma: no cover - delegated to TrainingError above
            raise TrainingError(f"XGBoost training failed: {exc}") from exc

        return model

    @staticmethod
    def _fit_lightgbm(
        model: _ModelLike,
        X_train: np.ndarray,
        y_train: np.ndarray,
        eval_set: list[tuple[np.ndarray, np.ndarray]],
        early_stopping_rounds: Optional[int],
    ) -> _ModelLike:
        """Fit LightGBM classifier with optional early stopping."""
        fit_kwargs: Dict[str, Any] = {
            "X": X_train,
            "y": y_train,
            "eval_set": eval_set,
            "eval_metric": "auc",
            "verbose": -1,
        }
        if early_stopping_rounds is not None and early_stopping_rounds > 0:
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds

        try:
            model.fit(**fit_kwargs)
        except Exception as exc:  # pragma: no cover - delegated to TrainingError above
            raise TrainingError(f"LightGBM training failed: {exc}") from exc

        return model

    @staticmethod
    def _get_best_iteration(model: _ModelLike) -> Optional[int]:
        """
        Attempt to extract the best iteration / best_ntree_limit from the model.

        Returns None if the model does not expose an applicable attribute.
        """
        for attr in ("best_iteration", "best_ntree_limit", "best_iteration_"):
            if hasattr(model, attr):
                try:
                    value = getattr(model, attr)
                    return int(value) if value is not None else None
                except (TypeError, ValueError):
                    continue
        return None
