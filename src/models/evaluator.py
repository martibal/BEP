"""
src/models/evaluator.py

Model evaluation utilities for the ON-CHAIN SUPER SIGNALS™ project.

Responsibilities:
- Compute classification metrics (AUC, precision, recall, F1).
- Compute drawdown hit-rate (per architecture requirements).
- Compute risk-adjusted return metrics (Sharpe, Sortino improvement).

Data minimization:
- Operates only on aggregated numpy arrays.
- NEVER accesses raw blockchain data.
- NEVER persists evaluation data to disk.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.utils.logger import get_logger

_logger = get_logger(__name__)


class ModelEvaluator:
    """
    Evaluate ML models for the ON-CHAIN SUPER SIGNALS™ project.

    Responsibilities:
    - Compute classification metrics (AUC, precision, recall, F1).
    - Compute drawdown hit-rate based on the target definition.
    - Compute risk-adjusted return metrics (Sharpe, Sortino).

    Note:
    - Default values for `target_drawdown_pct` and `target_window_days` are
      aligned with the architecture (5% and 21 days), but the *actual* values
      MUST be injected by the caller (e.g. pipelines/02_train_model.py) using
      config/model_config.yaml. This class itself must not read config.
    """

    def __init__(self, target_drawdown_pct: float = 5.0, target_window_days: int = 21) -> None:
        """
        Initialize a ModelEvaluator.

        Parameters
        ----------
        target_drawdown_pct:
            Percentage value for the drawdown threshold (default: 5%).
            Per architecture: 3–8%, optimized and injected from config by caller.
        target_window_days:
            Number of days for the forward drawdown window (default: 21).
        """
        if target_drawdown_pct <= 0:
            raise ValueError("target_drawdown_pct must be positive")
        if target_window_days <= 0:
            raise ValueError("target_window_days must be positive")

        self.target_drawdown_pct = float(target_drawdown_pct)
        self.target_window_days = int(target_window_days)

        _logger.info(
            "Initialized ModelEvaluator",
            extra={
                "target_drawdown_pct": self.target_drawdown_pct,
                "target_window_days": self.target_window_days,
            },
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        returns: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point for complete model evaluation.

        Parameters
        ----------
        y_true:
            Ground truth targets (0/1 for drawdown events).
        y_pred_proba:
            Predicted probabilities from the model in [0, 1].
        returns:
            Optional: daily returns for risk-adjusted metrics.

        Returns
        -------
        Dict[str, Any]
            Dictionary with keys:
            - "auc": float (0–1, may be NaN if only one class).
            - "drawdown_hitrate": float (0–1, may be NaN if no drawdowns).
            - "precision": float.
            - "recall": float.
            - "f1_score": float.
            - "sharpe_improvement": Optional[float].
            - "sortino_improvement": Optional[float].
            - "confusion_matrix": Dict[str, int].
        """
        # Validate basic classification inputs
        self._validate_arrays(y_true, y_pred_proba, allow_empty=False, is_proba=True)

        # Classification metrics
        auc = self.compute_auc(y_true, y_pred_proba)

        # Default threshold 0.5 for probability → class
        y_pred = (y_pred_proba >= 0.5).astype(int)

        drawdown_hitrate = self.compute_drawdown_hitrate(
            y_true=y_true,
            y_pred=y_pred,
        )

        # Use zero_division=0 for stability when there are no positive predictions
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Confusion matrix (labels 0/1)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        # Risk-adjusted metrics
        sharpe_improvement: Optional[float] = None
        sortino_improvement: Optional[float] = None

        if returns is not None:
            returns = np.asarray(returns)
            if returns.size == 0:
                _logger.info("Returns array is empty; Sharpe/Sortino improvements set to None")
            elif not np.isfinite(returns).all():
                _logger.warning(
                    "Returns array contains NaN/inf; Sharpe/Sortino improvements set to None",
                )
            else:
                # In this implementation, "improvement" is interpreted as the model
                # Sharpe/Sortino relative to a risk-free baseline (0). For an explicit
                # HODL baseline, use compute_sharpe_improvement() with two arrays.
                try:
                    sharpe_improvement = self.compute_sharpe_ratio(returns)
                except ValueError as exc:
                    _logger.warning(
                        "Failed to compute Sharpe ratio",
                        extra={"error": str(exc)},
                    )
                    sharpe_improvement = None

                try:
                    sortino_improvement = self.compute_sortino_ratio(returns)
                except ValueError as exc:
                    _logger.warning(
                        "Failed to compute Sortino ratio",
                        extra={"error": str(exc)},
                    )
                    sortino_improvement = None

        metrics: Dict[str, Any] = {
            "auc": float(auc) if auc is not None else np.nan,
            "drawdown_hitrate": float(drawdown_hitrate) if drawdown_hitrate is not None else np.nan,
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "sharpe_improvement": sharpe_improvement,
            "sortino_improvement": sortino_improvement,
            "confusion_matrix": {
                "true_positive": int(tp),
                "false_positive": int(fp),
                "true_negative": int(tn),
                "false_negative": int(fn),
            },
        }

        _logger.info(
            "Model evaluation completed",
            extra={
                "auc": metrics["auc"],
                "drawdown_hitrate": metrics["drawdown_hitrate"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "sharpe_improvement": sharpe_improvement,
                "sortino_improvement": sortino_improvement,
            },
        )

        return metrics

    # -------------------------------------------------------------------------
    # Metric helpers
    # -------------------------------------------------------------------------

    def compute_auc(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> float:
        """
        Compute Area Under the ROC Curve (AUC).

        Edge cases
        ----------
        - If y_true contains only one class: return NaN with a warning.
        - If arrays are empty: raise ValueError.

        Returns
        -------
        float
            AUC score between 0 and 1 (or NaN if only one class).
        """
        y_true = np.asarray(y_true)
        y_pred_proba = np.asarray(y_pred_proba)

        if y_true.size == 0 or y_pred_proba.size == 0:
            raise ValueError("Cannot compute AUC on empty arrays")

        # Check for NaN/inf in both y_true and y_pred_proba
        if not np.isfinite(y_true).all() or not np.isfinite(y_pred_proba).all():
            raise ValueError("Input contains NaN or inf values; cannot compute AUC")

        unique_classes = np.unique(y_true)
        if unique_classes.size < 2:
            _logger.warning(
                "AUC is undefined when y_true contains only one class",
                extra={"unique_classes": unique_classes.tolist()},
            )
            return float("nan")

        try:
            auc = roc_auc_score(y_true, y_pred_proba)
        except ValueError as exc:
            # sklearn can still raise ValueError in some special cases
            _logger.warning(
                "Failed to compute AUC via roc_auc_score",
                extra={"error": str(exc)},
            )
            return float("nan")

        return float(auc)

    def compute_drawdown_hitrate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.5,
    ) -> float:
        """
        Compute hit-rate for drawdown predictions.

        Definition
        ----------
        hit-rate = (correctly predicted drawdowns) / (total actual drawdowns)
        This is essentially recall for the drawdown (positive) class.

        Parameters
        ----------
        y_true:
            Actual drawdown events (0/1).
        y_pred:
            Predicted classes or probabilities.
            If not strictly binary, values are interpreted as probabilities and
            `threshold` is used to binarize.
        threshold:
            Threshold to convert probability into a binary class if needed.

        Returns
        -------
        float
            Hit-rate between 0 and 1 (or NaN if no actual drawdowns).
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Validate length, finiteness, and binary nature of y_true as early as possible
        self._validate_arrays(y_true, y_pred, allow_empty=False, is_proba=False)

        # If y_pred is not strictly binary, treat it as probabilities
        if not np.array_equal(np.unique(y_pred), np.array([0, 1])):
            # Validate that this actually looks like probabilities
            if (y_pred < 0).any() or (y_pred > 1).any():
                raise ValueError("y_pred values must be in [0, 1] when used as probabilities")
            y_pred = (y_pred >= threshold).astype(int)

        true_drawdowns = (y_true == 1)
        n_true_drawdowns = int(true_drawdowns.sum())

        if n_true_drawdowns == 0:
            _logger.warning(
                "No positive drawdown events in y_true; drawdown hit-rate is undefined",
            )
            return float("nan")

        predicted_drawdowns = (y_pred == 1)
        hits = int(np.logical_and(true_drawdowns, predicted_drawdowns).sum())

        hitrate = hits / n_true_drawdowns
        return float(hitrate)

    def compute_sharpe_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        annualization_factor: float = 252.0,
    ) -> float:
        """
        Compute annualized Sharpe ratio.

        Formula
        -------
        Sharpe = (mean(returns) - risk_free_rate) / std(returns) * sqrt(annualization_factor)

        Edge cases
        ----------
        - std(returns) == 0: return 0.0 with a warning.
        - Empty returns: raise ValueError.

        Note
        ----
        Uses sample standard deviation (ddof=1), which is standard in
        financial metrics, even though some libraries default to ddof=0.
        """
        returns = np.asarray(returns, dtype=float)

        if returns.size == 0:
            raise ValueError("Cannot compute Sharpe ratio on empty returns array")

        if not np.isfinite(returns).all():
            raise ValueError("Returns array contains NaN or inf values")

        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns, ddof=1))  # sample std

        if std_ret == 0.0:
            _logger.warning(
                "Standard deviation of returns is zero; Sharpe ratio set to 0.0",
                extra={"mean_return": mean_ret},
            )
            return 0.0

        excess_return = mean_ret - risk_free_rate
        sharpe = excess_return / std_ret * np.sqrt(annualization_factor)
        return float(sharpe)

    def compute_sortino_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        annualization_factor: float = 252.0,
    ) -> float:
        """
        Compute annualized Sortino ratio (using downside volatility only).

        Edge cases
        ----------
        - No negative returns: return inf with a warning.
        """
        returns = np.asarray(returns, dtype=float)

        if returns.size == 0:
            raise ValueError("Cannot compute Sortino ratio on empty returns array")

        if not np.isfinite(returns).all():
            raise ValueError("Returns array contains NaN or inf values")

        # Downside returns (below risk-free rate)
        downside_mask = returns < risk_free_rate
        downside_returns = returns[downside_mask]

        if downside_returns.size == 0:
            _logger.warning(
                "No negative (downside) returns; Sortino ratio is infinite",
            )
            return float("inf")

        mean_ret = float(np.mean(returns))
        downside_std = float(np.std(downside_returns, ddof=1))  # sample std

        if downside_std == 0.0:
            _logger.warning(
                "Downside standard deviation is zero; Sortino ratio set to 0.0",
                extra={"mean_return": mean_ret},
            )
            return 0.0

        excess_return = mean_ret - risk_free_rate
        sortino = excess_return / downside_std * np.sqrt(annualization_factor)
        return float(sortino)

    def compute_sharpe_improvement(
        self,
        model_returns: np.ndarray,
        baseline_returns: np.ndarray,
    ) -> float:
        """
        Compare Sharpe ratio between model strategy and HODL baseline.

        Returns
        -------
        float
            Difference in Sharpe ratio: (model_sharpe - baseline_sharpe).
        """
        model_sharpe = self.compute_sharpe_ratio(model_returns)
        baseline_sharpe = self.compute_sharpe_ratio(baseline_returns)
        improvement = model_sharpe - baseline_sharpe

        _logger.info(
            "Computed Sharpe improvement",
            extra={
                "model_sharpe": model_sharpe,
                "baseline_sharpe": baseline_sharpe,
                "sharpe_improvement": improvement,
            },
        )

        return float(improvement)

    # -------------------------------------------------------------------------
    # Internal validation helpers
    # -------------------------------------------------------------------------

    def _validate_arrays(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        allow_empty: bool = False,
        is_proba: bool = False,
    ) -> None:
        """
        Internal validation of input arrays.

        Checks
        ------
        - Not None.
        - Same shape.
        - No NaN/inf.
        - Not empty (unless allow_empty=True).
        - For y_true: only 0/1.
        - For probabilities: values must be in [0, 1].
        """
        if y_true is None or y_pred is None:
            raise ValueError("y_true and y_pred must not be None")

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if not allow_empty and (y_true.size == 0 or y_pred.size == 0):
            raise ValueError("Input arrays must not be empty")

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"y_true and y_pred must have the same shape; "
                f"got {y_true.shape} and {y_pred.shape}"
            )

        if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
            raise ValueError("Input arrays contain NaN or inf values")

        # y_true must only contain 0/1
        unique_true = np.unique(y_true)
        if not np.all(np.isin(unique_true, [0, 1])):
            raise ValueError("y_true must contain only binary values {0, 1}")

        if is_proba:
            # For probabilities, values must be in [0, 1]
            if (y_pred < 0.0).any() or (y_pred > 1.0).any():
                raise ValueError("Probabilities must be in [0, 1]")
