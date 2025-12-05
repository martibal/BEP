"""
SignalService: orchestrates daily signal generation for ON-CHAIN SUPER SIGNALS™.

Ansvar:
- Laste daglige aggregerte features fra tmp/ via DataService
- Hente historiske signaler fra DuckDB via db_handler
- Kjøre modell-inferens via ModelService for å få rå risk_score
- Generere endelige signaler (regime, risk state, indekser, alerts) via SignalGenerator
- Persistere daglig signal til DuckDB via db_handler

Viktige begrensninger:
- Ingen håndtering av rå blockchain-data (kun aggregerte features)
- Ingen direkte fil-I/O utover det DataService og db_handler gjør
- All historisk informasjon hentes fra DuckDB (ikke fra rådata)
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

from config.settings import CHAINS
from src.models.signal_generator import DailySignal, SignalGenerator
from src.utils import validators
from src.utils.logger import get_logger

_logger = get_logger(__name__)


class SignalService:
    """
    Orkestrerer hele inferens- og signalgenereringsflyten for én kjede og dato.

    Avhengigheter injiseres ved konstruksjon for å holde klassen testbar og
    følge arkitekturkravene om tydelig separasjon av ansvar.
    """

    def __init__(
        self,
        db_handler: Any,
        data_service: Any,
        model_service: Any,
        signal_generator: Optional[SignalGenerator] = None,
    ) -> None:
        """
        Initialiserer SignalService med nødvendige avhengigheter.

        Parametre
        ---------
        db_handler:
            Objekt (eller modul) som tilbyr DuckDB-operasjoner, forventes å ha:
            - get_historical_signals(chain, target_date, lookback_days)
            - write_daily_signal(chain, signal_dict)
            - write_alerts(chain, alerts)
        data_service:
            DataService-instans med:
            - load_daily_aggregates(chain, target_date) -> Dict[str, float]
        model_service:
            ModelService-instans med:
            - run_inference(features_df: pd.DataFrame) -> float
        signal_generator:
            Valgfri SignalGenerator-instans. Hvis None, opprettes en ny.
        """
        self._db_handler = db_handler
        self._data_service = data_service
        self._model_service = model_service
        self._signal_generator = signal_generator or SignalGenerator()

        _logger.info(
            "SignalService initialized",
            extra={
                "has_db_handler": self._db_handler is not None,
                "has_data_service": self._data_service is not None,
                "has_model_service": self._model_service is not None,
            },
        )

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def generate_and_store_daily_signal(self, chain: str, target_date: date) -> None:
        """
        Utfører hele den daglige inferens- og persistensflyten for én kjede.

        Steg:
        1. Valider chain og dato.
        2. Last daglige aggregerte features fra DataService.
        3. Hent historiske signaler fra DuckDB via db_handler.
        4. Konstruer feature-DataFrame og kjør modell-inferens via ModelService.
        5. Generer endelig signal via SignalGenerator.
        6. Persister signalet og eventuelle alerts til DuckDB via db_handler.

        Ved kritiske feil logges det på ERROR-nivå og det kastes RuntimeError.
        """
        self._validate_inputs(chain, target_date)
        chain_upper = chain.upper()

        _logger.info(
            "Starting daily signal generation",
            extra={"chain": chain_upper, "date": target_date.isoformat()},
        )

        # 1) Laste daglige features
        daily_features = self._load_daily_features(chain_upper, target_date)

        # 2) Laste historiske signaler (for EWMA/hysterese)
        lookback_days = self._signal_generator.smoothing_days * 6
        historical_signals_df = self._load_historical_signals(
            chain_upper, target_date, lookback_days
        )

        # 3) Modell-inferens → rå risk_score
        features_df = pd.DataFrame([daily_features])
        risk_score = self._run_model_inference(features_df)

        # 4) Generere endelig signal via SignalGenerator
        daily_signal = self._generate_final_signal(
            chain_upper,
            target_date,
            risk_score,
            daily_features,
            historical_signals_df,
        )

        # 5) Persistere til DuckDB
        self._persist_daily_signal(chain_upper, daily_signal)
        self._persist_alerts(chain_upper, daily_signal)

        _logger.info(
            "Daily signal generated and stored successfully",
            extra={
                "chain": chain_upper,
                "date": target_date.isoformat(),
                "risk_score": daily_signal.risk_score,
                "regime": daily_signal.regime,
                "risk_state": daily_signal.risk_state,
            },
        )

    # ------------------------------------------------------------------ #
    # Internal steps                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_inputs(chain: str, target_date: date) -> None:
        """Validerer kjede og dato i henhold til styringsdokumentet."""
        validators.validate_chain(chain)
        chain_upper = chain.upper()
        if chain_upper not in CHAINS:
            raise ValueError(f"Unsupported chain: {chain}")

        if not isinstance(target_date, date):
            raise TypeError("target_date must be a datetime.date instance")

        if target_date > date.today():
            raise ValueError(
                f"target_date {target_date.isoformat()} cannot be in the future"
            )

    def _load_daily_features(self, chain: str, target_date: date) -> Dict[str, float]:
        """
        Laster daglige aggregerte features fra DataService.

        Forventet:
        - DataService.load_daily_aggregates(chain, target_date) → Dict[str, float]

        Ved manglende data kastes RuntimeError.
        """
        try:
            features = self._data_service.load_daily_aggregates(chain, target_date)
        except FileNotFoundError as exc:
            _logger.error(
                "Daily aggregates not found; fetch/aggregate pipeline may have failed",
                extra={
                    "chain": chain,
                    "date": target_date.isoformat(),
                    "error": str(exc),
                },
            )
            raise RuntimeError(
                f"Missing daily aggregates for {chain} on {target_date.isoformat()}"
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive
            _logger.error(
                "Unexpected error while loading daily aggregates",
                extra={
                    "chain": chain,
                    "date": target_date.isoformat(),
                    "error": str(exc),
                },
            )
            raise RuntimeError(
                f"Failed to load daily aggregates for {chain} "
                f"on {target_date.isoformat()}"
            ) from exc

        if not isinstance(features, dict):
            _logger.error(
                "DataService returned non-dict daily features",
                extra={
                    "chain": chain,
                    "date": target_date.isoformat(),
                    "type": type(features).__name__,
                },
            )
            raise RuntimeError("DataService.load_daily_aggregates must return a dict")

        # Sikkerhetsvalidering av feature-vektor (NaN/inf etc.)
        try:
            validators.validate_feature_vector(features)
        except ValueError as exc:
            _logger.error(
                "Invalid feature vector (NaN/inf) for daily aggregates",
                extra={
                    "chain": chain,
                    "date": target_date.isoformat(),
                    "error": str(exc),
                },
            )
            raise RuntimeError(
                f"Invalid feature vector for {chain} on {target_date.isoformat()}"
            ) from exc

        _logger.debug(
            "Loaded daily features",
            extra={
                "chain": chain,
                "date": target_date.isoformat(),
                "feature_count": len(features),
            },
        )
        return features

    def _load_historical_signals(
        self,
        chain: str,
        target_date: date,
        lookback_days: int,
    ) -> pd.DataFrame:
        """
        Henter historiske signaler fra DuckDB via db_handler.

        Forventet:
        - db_handler.get_historical_signals(chain, target_date, lookback_days)
          → pd.DataFrame (kan være tom, men med korrekt schema).

        Eventuelle DB-feil pakkes i RuntimeError.
        """
        try:
            historical_df = self._db_handler.get_historical_signals(
                chain=chain,
                target_date=target_date,
                lookback_days=lookback_days,
            )
        except Exception as exc:  # pragma: no cover - defensive
            _logger.error(
                "Failed to fetch historical signals",
                extra={
                    "chain": chain,
                    "date": target_date.isoformat(),
                    "lookback_days": lookback_days,
                    "error": str(exc),
                },
            )
            raise RuntimeError(
                f"Failed to fetch historical signals for {chain} "
                f"on {target_date.isoformat()}"
            ) from exc

        if historical_df is None:
            # Normaliser til tom DataFrame for å matche spesifikasjonen.
            historical_df = pd.DataFrame()

        if not isinstance(historical_df, pd.DataFrame):
            _logger.error(
                "db_handler.get_historical_signals returned non-DataFrame",
                extra={
                    "chain": chain,
                    "date": target_date.isoformat(),
                    "type": type(historical_df).__name__,
                },
            )
            raise RuntimeError(
                "db_handler.get_historical_signals must return a pandas.DataFrame"
            )

        _logger.debug(
            "Loaded historical signals",
            extra={
                "chain": chain,
                "date": target_date.isoformat(),
                "lookback_days": lookback_days,
                "rows": len(historical_df.index),
            },
        )
        return historical_df

    def _run_model_inference(self, features_df: pd.DataFrame) -> float:
        """
        Kjør modell-inferens via ModelService.

        Forventet:
        - ModelService.run_inference(features_df: pd.DataFrame) → float (risk_score)

        Feil:
        - Hvis ModelService kaster ValueError eller spesifikke modellfeil,
          logges dette og pakkes i RuntimeError.
        """
        try:
            risk_score = self._model_service.run_inference(features_df)
        except ValueError as exc:
            _logger.error(
                "Model inference failed due to invalid input features",
                extra={"error": str(exc)},
            )
            raise RuntimeError("Model inference failed (invalid features)") from exc
        except Exception as exc:  # pragma: no cover - defensive
            _logger.error(
                "Unexpected error during model inference",
                extra={"error": str(exc)},
            )
            # Bevar opprinnelig unntakstype for enklere feilsøking.
            raise

        if not isinstance(risk_score, (int, float)):
            _logger.error(
                "ModelService.run_inference returned non-numeric risk_score",
                extra={"type": type(risk_score).__name__},
            )
            raise RuntimeError("ModelService.run_inference must return a float")

        _logger.debug(
            "Model inference completed",
            extra={"raw_risk_score": float(risk_score)},
        )
        return float(risk_score)

    def _generate_final_signal(
        self,
        chain: str,
        target_date: date,
        risk_score: float,
        daily_features: Dict[str, float],
        historical_signals_df: pd.DataFrame,
    ) -> DailySignal:
        """
        Genererer endelig daglig signal via SignalGenerator.

        Spesifikasjon:
        - SignalGenerator skal bruke historiske signaler for EWMA smoothing,
          hysterese og regime-rate-limiting.

        Implementasjon:
        - Konverterer historiske signaler fra DataFrame til liste av dicts
          i tråd med SignalGenerator-implementasjonen.
        """
        # Konverter historiske signaler til liste av dicts (nyeste først antas
        # å være siste rad i DataFrame, så vi sorterer eksplisitt hvis mulig).
        historical_records = self._historical_df_to_records(historical_signals_df)

        daily_signal = self._signal_generator.generate_daily_signal(
            chain=chain,
            target_date=target_date,
            raw_risk_score=risk_score,
            features=daily_features,
            historical_signals=historical_records,
        )

        if not isinstance(daily_signal, DailySignal):
            _logger.error(
                "SignalGenerator.generate_daily_signal returned unexpected type",
                extra={"type": type(daily_signal).__name__},
            )
            raise RuntimeError("SignalGenerator must return a DailySignal instance")

        return daily_signal

    @staticmethod
    def _historical_df_to_records(df: pd.DataFrame) -> list[Dict[str, Any]]:
        """
        Konverterer historiske signaler fra DataFrame til liste av dicts.

        Forventet sortering:
        - SignalGenerator forventer historiske signaler med nyeste først.
        - Hvis DataFrame har en kolonne 'date', sorterer vi eksplisitt på den
          i synkende rekkefølge.
        """
        if df.empty:
            return []

        if "date" in df.columns:
            df_sorted = df.sort_values("date", ascending=False)
        else:
            df_sorted = df

        return df_sorted.to_dict(orient="records")

    def _persist_daily_signal(self, chain: str, daily_signal: DailySignal) -> None:
        """
        Persisterer daglig signal til DuckDB via db_handler.

        Spesifikasjon:
        - Kall db_handler.write_daily_signal(chain, signal_dict) med et
          dictionary som matcher daily_signals-schemaet.
        """
        signal_dict = asdict(daily_signal)
        payload = {
            "chain": chain,
            "date": signal_dict["date"],
            "risk_score": signal_dict["risk_score"],
            "regime": signal_dict["regime"],
            "risk_state": signal_dict["risk_state"],
            "whale_index": signal_dict["whale_index"],
            "stress_index": signal_dict["stress_index"],
            "momentum_score": signal_dict["momentum_score"],
            "updated_at": datetime.now(timezone.utc),
        }

        try:
            self._db_handler.write_daily_signal(chain=chain, signal=payload)
        except Exception as exc:  # pragma: no cover - defensive
            _logger.error(
                "Failed to persist daily signal to DuckDB",
                extra={
                    "chain": chain,
                    "date": daily_signal.date.isoformat(),
                    "error": str(exc),
                },
            )
            raise RuntimeError(
                f"Failed to persist daily signal for {chain} "
                f"on {daily_signal.date.isoformat()}"
            ) from exc

    def _persist_alerts(self, chain: str, daily_signal: DailySignal) -> None:
        """
        Persisterer alle alerts fra DailySignal til alerts-tabellen via db_handler.

        Spesifikasjon:
        - Hvis daily_signal.alerts er tom, gjør ingenting.
        - Ellers kall db_handler.write_alerts(chain, alerts) med en liste av dicts
          som matcher alerts-schemaet (uten id/created_at).
        """
        if not daily_signal.alerts:
            _logger.debug(
                "No alerts generated for daily signal; skipping alert persistence",
                extra={
                    "chain": chain,
                    "date": daily_signal.date.isoformat(),
                },
            )
            return

        alerts_payload = []
        for alert in daily_signal.alerts:
            alert_dict = asdict(alert)
            alerts_payload.append(
                {
                    "chain": chain,
                    "date": alert_dict["date"],
                    "alert_type": alert_dict["alert_type"],
                    "message": alert_dict["message"],
                }
            )

        try:
            self._db_handler.write_alerts(chain=chain, alerts=alerts_payload)
        except Exception as exc:  # pragma: no cover - defensive
            _logger.error(
                "Failed to persist alerts to DuckDB",
                extra={
                    "chain": chain,
                    "date": daily_signal.date.isoformat(),
                    "alert_count": len(alerts_payload),
                    "error": str(exc),
                },
            )
            raise RuntimeError(
                f"Failed to persist alerts for {chain} "
                f"on {daily_signal.date.isoformat()}"
            ) from exc
