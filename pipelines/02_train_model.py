"""
pipelines/02_train_model.py

Main script for periodic model training for the
ON-CHAIN SUPER SIGNALS™ project.

Responsibilities
----------------
- Load historical aggregated feature/target data (no raw blockchain data).
- Split data into train/validation sets according to config/model_config.yaml.
- Fit FeatureNormalizer on training features.
- Train the ML model (delegated to src.models.trainer).
- Evaluate model performance (delegated to src.models.evaluator).
- Persist model, normalizer params, and version metadata to disk.
- Persist performance metrics to DuckDB via db_handler.write_performance_summary.

Data minimization
-----------------
- This script NEVER reads or writes raw blockchain data.
- It only operates on historical aggregated feature/target data.
- It only writes:
  - Model binaries (.bin) to MODEL_DIR
  - Normalization params (.json) to NORMALIZATION_DIR
  - Version metadata (version.json) to MODEL_DIR
  - Aggregated performance metrics to DuckDB (performance_summary table)
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from config.settings import CHAINS, MODEL_DIR, NORMALIZATION_DIR
from src.features.normalizer import FeatureNormalizer
from src.models.evaluator import ModelEvaluator
from src.models.trainer import ModelTrainer
from src.utils import db_handler, validators
from src.utils.logger import get_logger

_logger = get_logger(__name__)

# Global constant: minimum acceptable AUC for "production" status
MIN_PRODUCTION_AUC = 0.65


# --------------------------------------------------------------------------- #
# Argument parsing                                                            #
# --------------------------------------------------------------------------- #


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the model training pipeline.

    Parameters
    ----------
    argv:
        Optional explicit sequence of arguments (primarily for testing).
        If None, argparse will read from sys.argv.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes:
        - chains: Optional[str] (comma-separated chain list, e.g. "BTC,ETH")
        - date: Optional[str] (ISO date "YYYY-MM-DD", end of training window)
    """
    parser = argparse.ArgumentParser(
        description="Train/update ML models for ON-CHAIN SUPER SIGNALS™"
    )

    parser.add_argument(
        "--chains",
        type=str,
        default=None,
        help=(
            "Comma-separated list of chains to train (e.g. 'BTC,ETH'). "
            f"If omitted, all configured chains are trained: {','.join(CHAINS)}."
        ),
    )

    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help=(
            "Optional end date for the training window in ISO format (YYYY-MM-DD). "
            "If omitted, uses the test_end date from config/model_config.yaml."
        ),
    )

    return parser.parse_args(list(argv) if argv is not None else None)


# --------------------------------------------------------------------------- #
# Config & utility helpers                                                    #
# --------------------------------------------------------------------------- #


def _load_model_config() -> Dict[str, Any]:
    """
    Load model and training configuration from config/model_config.yaml.

    Returns
    -------
    Dict[str, Any]
        Parsed YAML configuration as a nested dictionary.

    Raises
    ------
    RuntimeError
        If the config file cannot be read or parsed.
    """
    base_dir = Path(__file__).resolve().parents[1]  # project root
    config_path = base_dir / "config" / "model_config.yaml"

    if not config_path.is_file():
        _logger.error(
            "model_config.yaml not found",
            extra={"path": str(config_path)},
        )
        raise RuntimeError(f"Missing model config file at {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except Exception as exc:  # pragma: no cover - defensive
        _logger.error(
            "Failed to parse model_config.yaml",
            extra={"path": str(config_path), "error": str(exc)},
        )
        raise RuntimeError("Failed to parse model_config.yaml") from exc

    _logger.info(
        "Loaded model_config.yaml",
        extra={"path": str(config_path)},
    )
    return config


def _resolve_target_end_date(config: Dict[str, Any], date_str: str | None) -> date:
    """
    Resolve the end date for the training window.

    If date_str is None, defaults to the 'test_end' date from model_config.yaml.
    The resulting date is validated to ensure it is not in the future.

    Parameters
    ----------
    config:
        Model configuration dictionary loaded from model_config.yaml.
    date_str:
        Optional ISO date string.

    Returns
    -------
    date
        The resolved end date for the training window.
    """
    if date_str is None:
        training_cfg = config.get("training", {})
        test_end_str = training_cfg.get("test_end")
        if not test_end_str:
            _logger.error(
                "Missing 'test_end' in training config; cannot resolve end date",
                extra={"training_config": training_cfg},
            )
            raise RuntimeError("training.test_end must be defined in model_config.yaml")
        try:
            end = datetime.strptime(test_end_str, "%Y-%m-%d").date()
        except ValueError as exc:
            _logger.error(
                "Invalid 'test_end' format in model_config.yaml; expected YYYY-MM-DD",
                extra={"test_end": test_end_str},
            )
            raise RuntimeError(
                f"Invalid training.test_end format: {test_end_str!r}"
            ) from exc
    else:
        try:
            end = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError as exc:
            _logger.error(
                "Invalid --date format; expected YYYY-MM-DD",
                extra={"date_str": date_str},
            )
            raise ValueError(
                f"Invalid date format: {date_str!r}. Expected YYYY-MM-DD."
            ) from exc

    validators.validate_date_not_in_future(end)
    return end


def _resolve_chains(chains_arg: str | None) -> List[str]:
    """
    Resolve which chains should be trained.

    Parameters
    ----------
    chains_arg:
        Optional comma-separated string of chain names.

    Returns
    -------
    List[str]
        Uppercase chain identifiers to train.

    Raises
    ------
    ValueError
        If any requested chain is not supported.
    """
    if chains_arg is None:
        _logger.info(
            "No --chains provided; defaulting to all configured chains",
            extra={"chains": CHAINS},
        )
        return list(CHAINS)

    raw_chains = [c.strip() for c in chains_arg.split(",") if c.strip()]
    if not raw_chains:
        _logger.error(
            "Empty --chains argument after parsing",
            extra={"chains_arg": chains_arg},
        )
        raise ValueError("At least one chain must be specified in --chains.")

    chains: List[str] = []
    for chain in raw_chains:
        chain_upper = chain.upper()
        validators.validate_chain(chain_upper)
        chains.append(chain_upper)

    _logger.info("Resolved chains for training", extra={"chains": chains})
    return chains


# --------------------------------------------------------------------------- #
# Data loading & preparation                                                  #
# --------------------------------------------------------------------------- #


def _load_historical_features(chain: str) -> pd.DataFrame:
    """
    Load historical aggregated feature/target data for a given chain.

    This function must NOT load or persist any raw blockchain data.
    It is expected that a separate pipeline (e.g. backfill_historical.py)
    has already computed and stored daily aggregated features + target.

    Implementation assumption
    -------------------------
    - Delegates to db_handler.get_historical_features(chain)
      which must return a pandas.DataFrame containing at least:
        - 'date' (datetime/date)
        - 'target' (binary or continuous, depending on model)
        - All configured feature columns.

    Parameters
    ----------
    chain:
        Blockchain identifier (e.g. "BTC", "ETH").

    Returns
    -------
    pd.DataFrame
        Historical feature/target data.

    Raises
    ------
    RuntimeError
        If no data is found or the returned object is not a DataFrame.
    """
    try:
        df = db_handler.get_historical_features(chain=chain)
    except AttributeError as exc:
        _logger.error(
            "db_handler.get_historical_features not implemented "
            "(required for training pipeline)",
            extra={"chain": chain, "error": str(exc)},
        )
        raise RuntimeError(
            "db_handler.get_historical_features(chain=...) must be implemented "
            "to support model training."
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        _logger.error(
            "Unexpected error while loading historical features",
            extra={"chain": chain, "error": str(exc)},
        )
        raise RuntimeError(
            f"Failed to load historical features for chain {chain}"
        ) from exc

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        _logger.error(
            "Historical feature dataset is missing or empty",
            extra={"chain": chain},
        )
        raise RuntimeError(f"No historical features available for chain {chain}")

    if "date" not in df.columns or "target" not in df.columns:
        _logger.error(
            "Historical feature dataset missing required columns 'date'/'target'",
            extra={"chain": chain, "columns": list(df.columns)},
        )
        raise RuntimeError(
            "Historical features must contain 'date' and 'target' columns."
        )

    # Pipeline-level validation of large historical dataset
    try:
        validators.validate_historical_features(df)
    except ValueError as exc:
        _logger.error(
            "Historical feature dataset failed validation",
            extra={"chain": chain, "error": str(exc)},
        )
        raise RuntimeError(
            f"Historical features for chain {chain} did not pass validation"
        ) from exc

    _logger.info(
        "Loaded historical features",
        extra={"chain": chain, "rows": len(df.index)},
    )
    return df


def _prepare_data(
    df: pd.DataFrame,
    config: Dict[str, Any],
    end_date: date,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split historical data into Train and Validation sets.

    The split is based on the date ranges in config["training"]:
        - train_start, train_end
        - val_start, val_end

    The end_date parameter is used to cap the validation end date
    (e.g. if training should only use data up to a specific day).

    Parameters
    ----------
    df:
        Historical feature/target DataFrame with 'date' and 'target' columns.
    config:
        Model configuration dictionary from model_config.yaml.
    end_date:
        End date for the training window (cap for validation period).

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]
        (X_train, y_train, X_val, y_val)

    Raises
    ------
    RuntimeError
        If required config fields are missing or resulting sets are empty.
    """
    training_cfg = config.get("training", {})
    required_keys = [
        "train_start",
        "train_end",
        "val_start",
        "val_end",
    ]
    missing_keys = [k for k in required_keys if k not in training_cfg]
    if missing_keys:
        _logger.error(
            "Missing keys in training config",
            extra={"missing_keys": missing_keys, "training_config": training_cfg},
        )
        raise RuntimeError(
            f"Missing keys in training config: {', '.join(missing_keys)}"
        )

    # Ensure 'date' is datetime.date
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.date

    train_start = datetime.strptime(training_cfg["train_start"], "%Y-%m-%d").date()
    train_end = datetime.strptime(training_cfg["train_end"], "%Y-%m-%d").date()
    val_start = datetime.strptime(training_cfg["val_start"], "%Y-%m-%d").date()
    val_end = datetime.strptime(training_cfg["val_end"], "%Y-%m-%d").date()

    # Cap validation end-date by the resolved end_date (if earlier).
    if end_date < val_end:
        _logger.info(
            "Capping validation end date by provided end_date",
            extra={"val_end": str(val_end), "end_date": str(end_date)},
        )
        val_end = end_date

    # Filter by date ranges
    train_mask = (df["date"] >= train_start) & (df["date"] <= train_end)
    val_mask = (df["date"] >= val_start) & (df["date"] <= val_end)

    train_df = df.loc[train_mask].copy()
    val_df = df.loc[val_mask].copy()

    if train_df.empty or val_df.empty:
        _logger.error(
            "Train or validation set is empty after date filtering",
            extra={
                "train_rows": len(train_df.index),
                "val_rows": len(val_df.index),
                "train_start": str(train_start),
                "train_end": str(train_end),
                "val_start": str(val_start),
                "val_end": str(val_end),
            },
        )
        raise RuntimeError("Train/validation sets must not be empty")

    # Separate features and target
    feature_cols = [c for c in train_df.columns if c not in ("date", "target")]
    X_train = train_df[feature_cols].reset_index(drop=True)
    y_train = train_df["target"].reset_index(drop=True)
    X_val = val_df[feature_cols].reset_index(drop=True)
    y_val = val_df["target"].reset_index(drop=True)

    _logger.info(
        "Prepared train/validation splits",
        extra={
            "train_rows": len(X_train.index),
            "val_rows": len(X_val.index),
            "feature_count": len(feature_cols),
        },
    )
    return X_train, y_train, X_val, y_val


# --------------------------------------------------------------------------- #
# Training & evaluation                                                       #
# --------------------------------------------------------------------------- #


def _sanitize_training_data(
    X_train: pd.DataFrame,
) -> pd.DataFrame:
    """
    Defensively handle NaN/Inf in training features.

    Strategy:
    - Replace +/-inf with NaN.
    - If the fraction of rows containing any NaN is <= 5%, drop those rows.
    - Otherwise, impute NaN values with per-feature median.

    Parameters
    ----------
    X_train:
        Training feature matrix.

    Returns
    -------
    pd.DataFrame
        Cleaned training feature matrix.
    """
    X = X_train.replace([np.inf, -np.inf], np.nan)
    nan_rows = X.isna().any(axis=1)
    nan_row_count = int(nan_rows.sum())
    total_rows = int(len(X.index))
    frac_nan_rows = (nan_row_count / total_rows) if total_rows > 0 else 0.0

    if nan_row_count == 0:
        return X

    if frac_nan_rows <= 0.05:
        _logger.warning(
            "Dropping rows with NaN/Inf in training data",
            extra={
                "nan_rows": nan_row_count,
                "total_rows": total_rows,
                "fraction": frac_nan_rows,
            },
        )
        X_clean = X.loc[~nan_rows].reset_index(drop=True)
    else:
        _logger.warning(
            "High proportion of rows with NaN/Inf; imputing with feature medians",
            extra={
                "nan_rows": nan_row_count,
                "total_rows": total_rows,
                "fraction": frac_nan_rows,
            },
        )
        X_clean = X.copy()
        medians = X_clean.median(axis=0, numeric_only=True)
        X_clean = X_clean.fillna(medians)

    return X_clean


def _train_and_evaluate(
    chain: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    normalizer: FeatureNormalizer,
    config: Dict[str, Any],
) -> Tuple[Any, Dict[str, Any]]:
    """
    Main training logic.

    Steps
    -----
    1. Defensive cleaning of X_train for NaN/Inf.
    2. Fit FeatureNormalizer on cleaned X_train.
    3. Transform X_train and X_val.
    4. Train model via ModelTrainer.
    5. Evaluate on validation set via ModelEvaluator.

    Parameters
    ----------
    chain:
        Blockchain identifier.
    X_train, y_train:
        Training features/target.
    X_val, y_val:
        Validation features/target.
    normalizer:
        FeatureNormalizer instance (unfitted).
    config:
        Model configuration dictionary from model_config.yaml.

    Returns
    -------
    Tuple[Any, Dict[str, Any]]
        (trained_model, metrics_dict)
    """
    _logger.info(
        "Starting training",
        extra={
            "chain": chain,
            "train_rows": len(X_train.index),
            "val_rows": len(X_val.index),
        },
    )

    # 1) Clean training data
    X_train_clean = _sanitize_training_data(X_train)

    # 2) Fit normalizer
    normalizer.fit(X_train_clean)

    # 3) Transform train/validation features
    X_train_norm = normalizer.transform(X_train_clean)
    X_val_norm = normalizer.transform(X_val)

    # 4) Train model
    trainer = ModelTrainer()
    model_cfg = config.get("xgboost", {})  # or LightGBM config; trainer decides
    model = trainer.train_model(X_train_norm, y_train, model_cfg)

    # 5) Evaluate model
    evaluator = ModelEvaluator()
    y_val_pred = (
        model.predict_proba(X_val_norm)[:, 1]
        if hasattr(model, "predict_proba")
        else model.predict(X_val_norm)
    )

    metrics: Dict[str, Any] = {}
    try:
        metrics["auc"] = evaluator.compute_auc(y_val, y_val_pred)
    except Exception as exc:  # pragma: no cover - defensive
        _logger.error(
            "Failed to compute AUC on validation set",
            extra={"chain": chain, "error": str(exc)},
        )
        metrics["auc"] = None

    # Drawdown hit-rate and Sharpe improvement are model/target specific and
    # may rely on additional series (e.g. returns). We assume ModelEvaluator
    # provides the appropriate logic; they may be None if not available.
    try:
        metrics["drawdown_hitrate"] = evaluator.compute_drawdown_hitrate(
            y_val,
            y_val_pred,
            threshold=config.get("training", {}).get("target_drawdown_pct", 5),
        )
    except Exception as exc:  # pragma: no cover - defensive
        _logger.error(
            "Failed to compute drawdown hit-rate",
            extra={"chain": chain, "error": str(exc)},
        )
        metrics["drawdown_hitrate"] = None

    try:
        metrics["sharpe_improvement"] = evaluator.compute_sharpe_improvement(
            returns=None, signals=None  # Placeholder; evaluator may handle None internally.
        )
    except Exception:
        metrics["sharpe_improvement"] = None

    _logger.info(
        "Training and evaluation completed",
        extra={"chain": chain, "metrics": metrics},
    )

    # Critical performance check (for versioning)
    auc = metrics.get("auc")
    if auc is not None and auc < MIN_PRODUCTION_AUC:
        _logger.critical(
            "Model AUC below production threshold; model will not be set as active",
            extra={"chain": chain, "auc": auc, "threshold": MIN_PRODUCTION_AUC},
        )

    return model, metrics


# --------------------------------------------------------------------------- #
# Artifact persistence & performance logging                                  #
# --------------------------------------------------------------------------- #


def _compute_next_model_version(chain: str) -> str:
    """
    Compute the next model version for a given chain.

    Strategy:
    - If db_handler.get_latest_model_version(chain) exists and returns e.g. "1.0.3",
      increment the patch component → "1.0.4".
    - If no existing version, default to "1.0.0".

    Parameters
    ----------
    chain:
        Blockchain identifier.

    Returns
    -------
    str
        Next model version string.
    """
    try:
        current_version = db_handler.get_latest_model_version(chain=chain)
    except AttributeError:
        _logger.warning(
            "db_handler.get_latest_model_version not implemented; "
            "defaulting to '1.0.0'",
            extra={"chain": chain},
        )
        return "1.0.0"
    except Exception as exc:  # pragma: no cover - defensive
        _logger.error(
            "Error while fetching latest model version",
            extra={"chain": chain, "error": str(exc)},
        )
        return "1.0.0"

    if not current_version:
        return "1.0.0"

    parts = str(current_version).split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0
    except (ValueError, IndexError):
        _logger.warning(
            "Unexpected version format; resetting to '1.0.0'",
            extra={"chain": chain, "current_version": current_version},
        )
        return "1.0.0"

    patch += 1
    next_version = f"{major}.{minor}.{patch}"
    _logger.info(
        "Computed next model version",
        extra={"chain": chain, "current_version": current_version, "next_version": next_version},
    )
    return next_version


def _save_artifacts(
    chain: str,
    model: Any,
    normalizer: FeatureNormalizer,
    model_version: str,
    metrics: Dict[str, Any],
) -> None:
    """
    Save trained model, normalization parameters, and update version.json.

    Dataminimering:
    - Only model binaries, normalization params, and version metadata are stored.
    - No raw blockchain data or training datasets are written to disk.

    Parameters
    ----------
    chain:
        Blockchain identifier.
    model:
        Trained model instance.
    normalizer:
        Fitted FeatureNormalizer.
    model_version:
        Version string for the trained model.
    metrics:
        Evaluation metrics (used to decide active/production status).
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(NORMALIZATION_DIR, exist_ok=True)

    chain_lower = chain.lower()
    model_path = os.path.join(MODEL_DIR, f"{chain_lower}_model_v{model_version}.bin")
    norm_path = os.path.join(
        NORMALIZATION_DIR, f"{chain_lower}_normalizer_params_v{model_version}.json"
    )

    # Save model
    try:
        trainer = ModelTrainer()
        trainer.save_model(model, model_path)
    except (IOError, OSError) as exc:
        _logger.error(
            "Failed to save model binary",
            extra={"chain": chain, "path": model_path, "error": str(exc)},
        )
        raise RuntimeError(f"Failed to save model for chain {chain}") from exc

    # Save normalization params
    try:
        normalizer.save_params(norm_path)
    except (IOError, OSError) as exc:
        _logger.error(
            "Failed to save normalizer params",
            extra={"chain": chain, "path": norm_path, "error": str(exc)},
        )
        raise RuntimeError(
            f"Failed to save normalizer params for chain {chain}"
        ) from exc

    # Update version.json (only if AUC >= threshold), using atomic write.
    version_file = os.path.join(MODEL_DIR, "version.json")
    version_data: Dict[str, str] = {}
    if os.path.exists(version_file):
        try:
            with open(version_file, "r", encoding="utf-8") as f:
                version_data = json.load(f) or {}
        except Exception as exc:  # pragma: no cover - defensive
            _logger.error(
                "Failed to read existing version.json; starting fresh",
                extra={"path": version_file, "error": str(exc)},
            )
            version_data = {}

    auc = metrics.get("auc")
    auc_met = bool(auc is not None and auc >= MIN_PRODUCTION_AUC)

    if auc_met:
        version_data[chain] = model_version
        _logger.info(
            "Setting model as active in version.json",
            extra={"chain": chain, "model_version": model_version, "auc": auc},
        )
    else:
        _logger.warning(
            "Model not set as active due to insufficient or missing AUC",
            extra={"chain": chain, "model_version": model_version, "auc": auc},
        )

    tmp_version_file = f"{version_file}.tmp"
    try:
        with open(tmp_version_file, "w", encoding="utf-8") as f:
            json.dump(version_data, f, indent=2, sort_keys=True)
        os.replace(tmp_version_file, version_file)
    except (IOError, OSError) as exc:
        _logger.error(
            "Failed to update version.json atomically",
            extra={"path": version_file, "error": str(exc)},
        )
        raise RuntimeError("Failed to update version.json") from exc

    _logger.info(
        "New model version activation evaluated",
        extra={
            "chain": chain,
            "model_version": model_version,
            "auc": auc,
            "auc_met": auc_met,
        },
    )


def _log_performance(
    chain: str,
    metrics: Dict[str, Any],
    model_version: str,
) -> None:
    """
    Persist performance metrics to the performance_summary table in DuckDB.

    Parameters
    ----------
    chain:
        Blockchain identifier.
    metrics:
        Metrics dictionary from _train_and_evaluate().
    model_version:
        Trained model version string.
    """
    payload = {
        "chain": chain,
        "model_version": model_version,
        "auc": metrics.get("auc"),
        "hitrate": metrics.get("drawdown_hitrate"),
        "sharpe_ratio": metrics.get("sharpe_improvement"),
    }

    try:
        db_handler.write_performance_summary(**payload)
    except Exception as exc:  # pragma: no cover - defensive
        _logger.error(
            "Failed to write performance summary to DuckDB",
            extra={"chain": chain, "model_version": model_version, "error": str(exc)},
        )
        raise RuntimeError(
            f"Failed to persist performance summary for chain {chain}"
        ) from exc

    _logger.info(
        "Persisted performance summary",
        extra={"chain": chain, "model_version": model_version, "metrics": payload},
    )


# --------------------------------------------------------------------------- #
# Main orchestration                                                          #
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    """
    Entry point for the model training pipeline.

    Steps
    -----
    - Parse CLI arguments.
    - Load model/training configuration.
    - Resolve target end date and chains.
    - For each chain:
        * Load historical feature/target data.
        * Prepare train/validation splits.
        * Train and evaluate model.
        * Compute next model version.
        * Save model & normalizer artifacts.
        * Log performance summary to DuckDB.

    Parameters
    ----------
    argv:
        Optional explicit arguments for testability. If None, uses sys.argv.

    Returns
    -------
    int
        Exit code (0 for success, 1 if any chain failed).
    """
    try:
        args = _parse_args(argv)
        config = _load_model_config()
        end_date = _resolve_target_end_date(config, args.date)
        chains = _resolve_chains(args.chains)
    except (ValueError, RuntimeError) as exc:
        _logger.error(
            "Argument or configuration resolution failed",
            extra={"error": str(exc)},
        )
        return 1

    _logger.info(
        "Starting model training pipeline",
        extra={"chains": chains, "end_date": end_date.isoformat()},
    )

    failed_chains: List[str] = []

    for chain in chains:
        try:
            historical_df = _load_historical_features(chain)
            X_train, y_train, X_val, y_val = _prepare_data(
                historical_df, config, end_date
            )

            normalizer = FeatureNormalizer()
            model, metrics = _train_and_evaluate(
                chain=chain,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                normalizer=normalizer,
                config=config,
            )

            model_version = _compute_next_model_version(chain)
            _save_artifacts(
                chain=chain,
                model=model,
                normalizer=normalizer,
                model_version=model_version,
                metrics=metrics,
            )
            _log_performance(chain=chain, metrics=metrics, model_version=model_version)
        except Exception as exc:  # pragma: no cover - defensive
            _logger.error(
                "Model training pipeline failed for chain",
                extra={"chain": chain, "error": str(exc)},
            )
            failed_chains.append(chain)
            continue

    if failed_chains:
        _logger.error(
            "Model training pipeline completed with failures",
            extra={"failed_chains": failed_chains},
        )
        return 1

    _logger.info("Model training pipeline completed successfully", extra={"chains": chains})
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    import sys

    sys.exit(main())
