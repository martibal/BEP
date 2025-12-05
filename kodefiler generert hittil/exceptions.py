from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional
import json


class OnChainSignalsError(Exception):
    """
    Base exception for all ON-CHAIN SUPER SIGNALS™ errors.

    Attributes
    ----------
    message : str
        Human-readable error description.
    details : Optional[Dict[str, Any]]
        Additional context about the error.
    """

    def __init__(
        self,
        message: Optional[str],
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Ensure message is always a string; treat None as empty string.
        if message is None:
            self.message: str = ""
        else:
            self.message = str(message)

        # Details are stored as-is; to_dict() will handle serialization safety.
        self.details: Optional[Dict[str, Any]] = details

        super().__init__(self.message)

    def __str__(self) -> str:
        """
        Return formatted error message with details if available.
        """
        if self.details:
            return f"{self.message} (details={self.details})"
        return self.message

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize exception to a dictionary suitable for logging or API responses.

        The returned dict is designed to be JSON-serializable. If `details`
        contains non-serializable values, a repr() fallback is used.

        Returns
        -------
        Dict[str, Any]
            Dictionary with keys:
            - "error_type": Name of the exception class.
            - "message": Error message string.
            - "details": Original details if serializable, otherwise repr(details).
            - "timestamp": ISO 8601 UTC timestamp with 'Z' suffix.
        """
        # Compute a safe representation of details.
        safe_details: Any
        if self.details is None:
            safe_details = None
        else:
            # First try to ensure JSON-serializability.
            try:
                json.dumps(self.details)
                safe_details = self.details
            except TypeError:
                # Fallback to repr() if details are not JSON-serializable.
                safe_details = repr(self.details)

        # Generate an ISO 8601 UTC timestamp with 'Z' suffix.
        ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": safe_details,
            "timestamp": ts,
        }


class ConfigurationError(OnChainSignalsError):
    """
    Raised when configuration is missing, invalid, or inconsistent.

    Examples
    --------
    - Missing required environment variable.
    - Invalid value in settings.py.
    - YAML parse error in features.yaml.
    - Normalization parameters file has wrong format.
    """


class DataError(OnChainSignalsError):
    """Base class for data-related errors."""


class DataFetchError(DataError):
    """Raised when data cannot be fetched (S3, network, etc.)."""


class DataValidationError(DataError):
    """Raised when data fails validation (NaN, wrong types, etc.)."""


class DataNotFoundError(DataError):
    """Raised when expected data does not exist."""


class ModelError(OnChainSignalsError):
    """Base class for ML model-related errors."""


class ModelLoadingError(ModelError):
    """Raised when model cannot be loaded (file not found, corrupt, etc.)."""


class ModelPredictionError(ModelError):
    """Raised when model prediction fails."""


class ModelTrainingError(ModelError):
    """Raised when model training fails."""


class FeatureError(OnChainSignalsError):
    """Base class for feature computation errors."""


class FeatureComputationError(FeatureError):
    """Raised when feature computation fails."""


class FeatureNormalizationError(FeatureError):
    """Raised when feature normalization fails."""


class ServiceError(OnChainSignalsError):
    """Base class for service-layer errors."""


class SignalGenerationError(ServiceError):
    """Raised when signal generation fails."""


class DatabaseError(ServiceError):
    """Raised when database operations fail."""
