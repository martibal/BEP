"""
Predictor layer for ON-CHAIN SUPER SIGNALS™ project.

This module is part of the 'models' layer and contains all domain-specific logic
related to machine learning model artifacts:
- Version resolution (via version.json).
- Model file path construction.
- Robust model loading (including caching and error handling for corruption).
- Core inference logic (predict_proba handling, class selection, scaling to [0, 100]).

This component is used by ModelService.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import joblib
import numpy as np
import pandas as pd
from src.config.settings import MODEL_DIR
from src.utils.logger import get_logger


class ModelLoadingError(Exception):
    """Custom exception raised when model files are missing or corrupt."""
    pass


@runtime_checkable
class _ModelLike(Protocol):
    """
    Protocol for loaded ML models.
    """
    def predict(self, X: Any) -> Any:  # pragma: no cover
        ...

    def predict_proba(self, X: Any) -> Any:  # pragma: no cover
        ...


class Predictor:
    """
    Handles all aspects of ML model loading, caching, and running core inference.
    """
    def __init__(self) -> None:
        self._logger = get_logger(self.__class__.__name__)
        # Cache of loaded models per chain (e.g., {"BTC": <model object>}).
        self._models: Dict[str, _ModelLike] = {}
        self._model_versions: Dict[str, Optional[int]] = {}

    def _read_latest_model_version(self, chain: str) -> int:
        """Read the latest model version for a given chain from version.json."""
        chain_upper = chain.upper()
        version_path = Path(MODEL_DIR) / "version.json"

        if not version_path.is_file():
            self._logger.error(
                "Model version metadata file not found", extra={"path": str(version_path)},
            )
            raise ModelLoadingError(f"Model version metadata file not found: {version_path}")

        try:
            with version_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        except json.JSONDecodeError as exc:
            raise ModelLoadingError(f"Invalid JSON in version metadata: {version_path}") from exc

        # Simplified logic to retrieve the version
        entry = meta.get(chain_upper)
        if entry is None:
            raise ModelLoadingError(f"No version entry for chain {chain_upper} in {version_path}")
        
        version_val = entry.get("latest_version") if isinstance(entry, dict) else entry

        try:
            return int(version_val)
        except (TypeError, ValueError):
            raise ModelLoadingError(
                f"Invalid version value for chain {chain_upper} in {version_path}: {version_val!r}"
            )

    def _get_model_path(self, chain: str, version: Optional[int] = None) -> tuple[Path, int]:
        """
        Construct the path to a trained model artifact and resolve the version.
        
        Returns
        -------
        tuple[Path, int]
            Fully qualified path and the resolved version number.
        """
        chain_upper = chain.upper()
        model_dir = Path(MODEL_DIR)

        if version is None:
            resolved_version = self._read_latest_model_version(chain_upper)
        else:
            try:
                resolved_version = int(version)
            except (TypeError, ValueError):
                raise ValueError(f"Invalid model version: {version!r}")

        # Naming convention based on architecture: <MODEL_DIR>/<CHAIN>_model_v{version}.bin
        filename = f"{chain_upper}_model_v{resolved_version}.bin"
        return model_dir / filename, resolved_version

    def load_model(
        self,
        chain: str,
        version: Optional[int] = None,
        force_reload: bool = False,
    ) -> _ModelLike:
        """
        Load (or retrieve from cache) the trained model for a given chain.
        """
        chain_upper = chain.upper()

        # Resolve path and effective version
        try:
            model_path, effective_version = self._get_model_path(chain_upper, version)
        except (ValueError, ModelLoadingError) as exc:
            self._logger.error(f"Failed to resolve model path: {exc}")
            raise ModelLoadingError(f"Failed to resolve model path: {exc}")

        # Reuse cached model if appropriate.
        cached_model = self._models.get(chain_upper)
        cached_version = self._model_versions.get(chain_upper)

        if (
            cached_model is not None
            and not force_reload
            and cached_version == effective_version
        ):
            self._logger.debug(
                "Using cached model",
                extra={"chain": chain_upper, "version": cached_version},
            )
            return cached_model

        self._logger.info(
            "Loading model from disk",
            extra={"chain": chain_upper, "version": effective_version, "path": str(model_path)},
        )

        if not model_path.exists():
            raise ModelLoadingError(f"Model file not found: {model_path}")

        try:
            # CRITICAL FIX (C.3): Robust error handling for joblib.load
            model = joblib.load(model_path)
        except (EOFError, joblib.externals.loky.process_original_exception.RemoteTraceback) as exc:
            self._logger.critical(
                "CRITICAL: Model file appears corrupt or incomplete.",
                extra={"path": str(model_path), "error": str(exc)},
            )
            raise ModelLoadingError(f"Corrupt model file: {model_path}. Error: {exc}")
        except Exception as exc:
             self._logger.critical(
                "CRITICAL: Unexpected error during model loading.",
                extra={"path": str(model_path), "error": str(exc), "type": type(exc).__name__},
            )
             raise ModelLoadingError(f"Unexpected error loading model: {model_path}. Error: {exc}")
             

        if not isinstance(model, _ModelLike):
             self._logger.warning(
                "Loaded model does not formally implement _ModelLike protocol",
                extra={"chain": chain_upper, "type": type(model).__name__},
            )

        # Cache model and associated version.
        self._models[chain_upper] = model  # type: ignore[assignment]
        self._model_versions[chain_upper] = effective_version

        return model # type: ignore[return-value]
    
    def get_risk_score(self, chain: str, X: np.ndarray, model_version: Optional[int] = None) -> float:
        """
        Runs prediction on a normalized feature array (X) and returns the risk score [0-100].
        
        Parameters
        ----------
        chain: Chain symbol ("BTC", "ETH").
        X: The 2D numpy array of normalized features (ready for model input).
        model_version: Optional explicit model version to use.
        
        Returns
        -------
        float: Risk score in the range [0, 100].
        
        Raises
        ------
        ValueError: If model does not support prediction API.
        ModelLoadingError: If the model cannot be loaded.
        """
        chain_upper = chain.upper()
        model = self.load_model(chain_upper, version=model_version)

        # Core Inference Logic (Moved from ModelService)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            proba_arr = np.asarray(proba)

            if proba_arr.ndim == 1:
                p1_raw = float(proba_arr[0])
            elif proba_arr.ndim == 2 and proba_arr.shape[0] >= 1:
                # Find index of the positive class (1). Most binary classifiers use index 1 for the second class.
                idx_positive: int = 1 
                
                # More robust class indexing based on model metadata if available
                classes = getattr(model, "classes_", None)
                if classes is not None and len(classes) == 2:
                    # Assumes the target risk class (e.g., '1') is the positive class.
                    if np.asarray(classes)[0] in (0, '0', 'NO_RISK'):
                        idx_positive = 1
                    elif np.asarray(classes)[1] in (0, '0', 'NO_RISK'):
                        idx_positive = 0
                
                p1_raw = float(proba_arr[0, idx_positive])
            else:
                raise ValueError("Unexpected shape from predict_proba output")

            # Clip raw probability to [0, 1] before scaling.
            p1_clipped = float(np.clip(p1_raw, 0.0, 1.0))
            risk_score = p1_clipped * 100.0

        elif hasattr(model, "predict"):
            raw_pred = model.predict(X)
            raw_val = float(np.asarray(raw_pred)[0])

            # Treat raw prediction as probability-like; clip to [0, 1] and scale to [0, 100].
            prob_like = float(np.clip(raw_val, 0.0, 1.0))
            risk_score = prob_like * 100.0
        else:
            raise ValueError(
                "Loaded model does not expose predict or predict_proba; "
                "cannot compute risk score."
            )
        
        # Final safety clamp to [0, 100].
        if not np.isfinite(risk_score):
             raise ValueError("Non-finite risk score produced after prediction.")
        
        return float(np.clip(risk_score, 0.0, 100.0))
