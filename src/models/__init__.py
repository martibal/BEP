"""
Machine learning models package for ON-CHAIN SUPER SIGNALS™.

Contains:
- trainer: Model training logic (XGBoost/LightGBM)
- predictor: Inference and prediction
- evaluator: Performance metrics (AUC, hit-rate, Sharpe)
- signal_generator: Convert predictions to signals
- model_loader: Model loading and versioning
"""

from src.models import evaluator, model_loader, predictor, signal_generator, trainer

__all__ = ["evaluator", "model_loader", "predictor", "signal_generator", "trainer"]
