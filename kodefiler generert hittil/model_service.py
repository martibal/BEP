"""
Model service layer for the ON-CHAIN SUPER SIGNALS™ project.

This module is responsible for orchestrating model-based inference:

- Loading and applying feature normalization parameters via FeatureNormalizer.
- Orchestrating model loading and prediction via the Predictor layer.
- Returning normalized risk scores suitable for downstream signal generation.

Strict data minimization:
- This service operates *only* on aggregated feature data (pd.DataFrame).
- It NEVER reads, writes, or persists raw blockchain data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import NORMALIZER_PARAMS_PATH
from src.features.normalizer import FeatureNormalizer
from src.models.predictor import Predictor, ModelLoadingError
from src.utils import validators
from src.utils.logger import get_logger


class ModelService:
    """
    Service responsible for orchestrating feature normalization and model inference.

    Responsibilities
    ----------------
    - Initialize logger.
    - Initialize FeatureNormalizer and Predictor instances.
    - Load and manage normalization parameters.
    - Orchestrate inference: validate -> normalize -> predict (via Predictor).
    - Ensure that only aggregated data and model artifacts are read/written.
    """

    def __init__(self) -> None:
        """
        Initialize the ModelService and its dependencies.
        """
        self._logger = get_logger(self.__class__.__name__)
        self._logger.info("Initializing ModelService")

        # Feature normalizer instance (parameters are loaded on demand).
        self._normalizer: FeatureNormalizer = FeatureNormalizer()
        
        # Predictor handles all model loading, caching, and core prediction logic (Architectural Fix A.1)
        self._predictor: Predictor = Predictor()

    # --------------------------------------------------------------------- #
    # Normalizer handling
    # --------------------------------------------------------------------- #

    def load_normalizer(self, params_path: Optional[Path] = None) -> None:
        """
        Load normalization parameters into the internal FeatureNormalizer.
        """
        if params_path is None:
            params_path = Path(NORMALIZER_PARAMS_PATH)

        params_path = Path(params_path)

        self._logger.info(
            "Loading normalization parameters",
            extra={"path": str(params_path)},
        )
        self._normalizer.load_params(params_path)
        self._logger.info("Normalization parameters loaded successfully")

    # --------------------------------------------------------------------- #
    # Inference Orchestration
    # --------------------------------------------------------------------- #

    def run_inference(
        self,
        chain: str,
        features: pd.DataFrame,
        model_version: Optional[int] = None,
    ) -> float:
        """
        Run model inference for a given chain on aggregated feature data.

        Steps
        -----
        1. Validate `features` is non-empty.
        2. Apply Z-score normalization.
        3. Validate the normalized feature vector for NaN/inf.
        4. Pass the normalized vector to the Predictor layer for prediction.

        Parameters
        ----------
        chain:
            Chain symbol ("BTC", "ETH", case-insensitive).
        features:
            Aggregated feature DataFrame (typically one row).
        model_version:
            Optional explicit model version to use. If None, the latest version
            is resolved via version.json.

        Returns
        -------
        float
            Risk score in the range [0, 100].

        Raises
        ------
        ValueError
            If input is invalid, normalization fails, or the model API is incompatible.
        ModelLoadingError
            If the model file is missing or corrupt.
        """
        chain_upper = chain.upper()

        if features is None or features.empty:
            raise ValueError("features must be a non-empty DataFrame")

        self._logger.debug(
            "Starting inference orchestration",
            extra={"chain": chain_upper},
        )

        # 1. Normalize features (Normalizer will raise if params missing).
        try:
            normalized = self._normalizer.transform(features)
        except ValueError as exc:
            self._logger.error("Normalization failed: %s", str(exc))
            raise

        # 2. Validate normalized feature vector for NaN/inf before prediction.
        validators.validate_feature_vector(normalized)

        # 3. Run prediction via the dedicated Predictor (Architectural Fix A.1).
        X = normalized.values
        
        try:
            risk_score = self._predictor.get_risk_score(
                chain_upper, 
                X, 
                model_version=model_version
            )
        except (ValueError, ModelLoadingError) as exc:
            # Catch errors from the Predictor layer and log before re-raising.
            self._logger.error(
                "Predictor failed to compute score: %s",
                str(exc),
                extra={"chain": chain_upper},
            )
            raise
        
        # Log final result
        self._logger.info(
            "Inference completed successfully",
            extra={
                "chain": chain_upper,
                "risk_score": risk_score,
                # Note: We rely on Predictor's internal versioning but the log is here.
            },
        )

        return risk_score