"""
Signal generation module for the ON-CHAIN SUPER SIGNALS™ project.

This module converts raw model predictions (risk_score 0–100) and daily
aggregated features into complete, stable daily signals suitable for
storage in the daily_signals DuckDB table.

Responsibilities
----------------
- EWMA smoothing of daily risk_score
- Regime classification (Bull / Neutral / Bear)
- Risk state classification (Risk ON / Neutral / Risk OFF)
- Secondary indices:
  - whale_index (0–100)
  - stress_index (0–100)
  - momentum_score (0–100)
- Alert generation:
  - whale_spike
  - stress_event
  - momentum_collapse
  - regime_shift
- Stability mechanisms:
  - Hysteresis on regime changes
  - Rate limiting: max 1 regime change per 48 timer

Data minimization:
- Operates exclusively on aggregated daily features (Dict[str, float]).
- No access to raw blockchain data.
- No persistence; storage is handled by services / db_handler.

Architecture:
- Lives in models/ layer (domain logic).
- Called by src.services.signal_service.SignalService.
- Uses config.settings.CHAINS and src.utils.validators for validation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from config.settings import CHAINS
from src.utils import validators
from src.utils.logger import get_logger

_logger = get_logger(__name__)


@dataclass
class Alert:
    """Representerer én alert for en gitt kjede og dato."""

    chain: str
    date: date
    alert_type: str  # "whale_spike" | "stress_event" | "momentum_collapse" | "regime_shift"
    message: str


@dataclass
class DailySignal:
    """Representerer et komplett daglig signal for én kjede."""

    chain: str
    date: date
    risk_score: float  # 0-100, etter smoothing
    regime: str  # "Bull" | "Neutral" | "Bear"
    risk_state: str  # "Risk ON" | "Neutral" | "Risk OFF"
    whale_index: float  # 0-100
    stress_index: float  # 0-100
    momentum_score: float  # 0-100
    alerts: List[Alert]  # Genererte alerts for denne dagen


class SignalGenerator:
    """
    Konverterer rå modellprediksjoner til stabile markedssignaler.

    Parametre (kan overstyres via constructor, default-verdier fra
    arkitekturdokumentet / model_config.yaml):

    - smoothing_days: Antall dager for EWMA smoothing (default: 5)
    - hysteresis_threshold: Minimum poengendring for regime-skifte (default: 10.0)
    - min_regime_change_hours: Minimum tid mellom regime-endringer (default: 48)
    """

    def __init__(
        self,
        smoothing_days: int = 5,
        hysteresis_threshold: float = 10.0,
        min_regime_change_hours: int = 48,
    ) -> None:
        if smoothing_days < 1:
            _logger.warning(
                "smoothing_days < 1; clamping to 1",
                extra={"smoothing_days": smoothing_days},
            )
            smoothing_days = 1

        if min_regime_change_hours < 0:
            _logger.warning(
                "min_regime_change_hours < 0; clamping to 0",
                extra={"min_regime_change_hours": min_regime_change_hours},
            )
            min_regime_change_hours = 0

        self.smoothing_days = smoothing_days
        self.hysteresis_threshold = float(hysteresis_threshold)
        self.min_regime_change_hours = int(min_regime_change_hours)

        self._alpha = 2.0 / (self.smoothing_days + 1.0)

        _logger.info(
            "Initialized SignalGenerator",
            extra={
                "smoothing_days": self.smoothing_days,
                "hysteresis_threshold": self.hysteresis_threshold,
                "min_regime_change_hours": self.min_regime_change_hours,
            },
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def generate_daily_signal(
        self,
        chain: str,
        target_date: date,
        raw_risk_score: float,
        features: Dict[str, float],
        historical_signals: Optional[List[Dict[str, Any]]] = None,
    ) -> DailySignal:
        """
        Generer et komplett daglig signal for én kjede.

        Steg:
        1. Valider input (chain, dato, rå score).
        2. EWMA smoothing av risk_score basert på historiske signaler.
        3. Regime-klassifisering (Bull/Neutral/Bear).
        4. Hysterese mot forrige regime.
        5. Rate limiting (maks 1 endring per 48 timer).
        6. Risk state-klassifisering (Risk ON/Neutral/Risk OFF).
        7. Beregning av whale_index, stress_index, momentum_score.
        8. Generering av alerts.
        9. Returner DailySignal.

        Edge cases:
        - Ingen historiske signaler: hopp over smoothing/hysterese/rate limiting.
        - NaN/inf eller manglende feature-nøkler: bruk fallback 50.0 med warning.
        - risk_score utenfor [0, 100]: clamp til [0, 100] med warning.
        """
        validators.validate_chain(chain)
        chain_upper = chain.upper()
        if chain_upper not in CHAINS:
            raise ValueError(f"Unsupported chain: {chain}")

        if target_date > date.today():
            raise ValueError(
                f"target_date {target_date.isoformat()} cannot be in the future"
            )

        # Sørg for at features alltid er et dictionary, også hvis caller sender inn None.
        features = features or {}

        # Ensure historical_signals is a list (None og [] behandles likt).
        historical_signals = historical_signals or []

        # 1. Normaliser og clamp rå risk_score til [0, 100].
        current_score = self._sanitize_risk_score(raw_risk_score)

        # 2. EWMA smoothing basert på historiske scorer (nyeste først).
        historical_scores = self._extract_historical_scores(historical_signals)
        smoothed_score = self.apply_smoothing(current_score, historical_scores)

        # 3. Regime basert på smoothed score.
        new_regime = self.risk_score_to_regime(smoothed_score)

        # 4. Hysterese mot forrige regime.
        previous_signal = historical_signals[0] if historical_signals else None
        previous_regime = (
            previous_signal.get("regime") if previous_signal is not None else None
        )
        previous_score = (
            float(previous_signal.get("risk_score"))
            if previous_signal is not None
            and isinstance(previous_signal.get("risk_score"), (int, float))
            else None
        )

        regime_after_hysteresis = self.apply_hysteresis(
            new_regime=new_regime,
            previous_regime=previous_regime,
            current_score=smoothed_score,
            previous_score=previous_score,
        )

        # 5. Rate limiting – maks 1 regime-endring per min_regime_change_hours.
        final_regime = self.check_rate_limit(
            new_regime=regime_after_hysteresis,
            historical_signals=historical_signals,
            target_date=target_date,
        )

        if previous_regime and final_regime != previous_regime:
            _logger.info(
                "Regime change detected",
                extra={
                    "chain": chain_upper,
                    "date": target_date.isoformat(),
                    "previous_regime": previous_regime,
                    "new_regime": final_regime,
                },
            )

        # 6. Risk state basert på smoothed score.
        risk_state = self.compute_risk_state(smoothed_score)

        # 7. Sekundære indekser.
        whale_index = self.compute_whale_index(chain_upper, features)
        stress_index = self.compute_stress_index(chain_upper, features)
        momentum_score = self.compute_momentum_score(chain_upper, features)

        # Foreløpig signal uten alerts (alerts genereres i neste steg).
        current_signal = DailySignal(
            chain=chain_upper,
            date=target_date,
            risk_score=smoothed_score,
            regime=final_regime,
            risk_state=risk_state,
            whale_index=whale_index,
            stress_index=stress_index,
            momentum_score=momentum_score,
            alerts=[],
        )

        # 8. Alerts basert på dagens og gårsdagens signal + features.
        alerts = self.generate_alerts(
            chain=chain_upper,
            target_date=target_date,
            current_signal=current_signal,
            previous_signal=previous_signal,
            features=features,
        )
        current_signal.alerts = alerts

        _logger.info(
            "Generated daily signal",
            extra={
                "chain": chain_upper,
                "date": target_date.isoformat(),
                "risk_score": smoothed_score,
                "regime": final_regime,
                "risk_state": risk_state,
                "whale_index": whale_index,
                "stress_index": stress_index,
                "momentum_score": momentum_score,
                "alerts_count": len(alerts),
            },
        )

        return current_signal

    # ------------------------------------------------------------------ #
    # Smoothing & classification
    # ------------------------------------------------------------------ #

    def apply_smoothing(
        self,
        current_score: float,
        historical_scores: List[float],
    ) -> float:
        """
        EWMA smoothing av risk_score.

        Input:
        - current_score: Dagens rå risk_score (allerede clampet til [0, 100]).
        - historical_scores: Liste av tidligere risk_scores (nyeste først).

        Viktig:
        - historical_scores forventes å være tidligere SMOOTHED risk_score-verdier
          generert av denne klassen (ikke rå modelloutput). Vi antar at alle
          historiske rader allerede er prosessert gjennom samme pipeline.

        Logikk:
        - alpha = 2 / (smoothing_days + 1)
        - Hvis historical_scores er tom, returner current_score uendret.
        - Ellers:
            previous_smoothed = historical_scores[0]
            smoothed = alpha * current_score + (1 - alpha) * previous_smoothed
        """
        if not historical_scores:
            _logger.debug(
                "No historical scores provided; skipping smoothing",
                extra={"current_score": current_score},
            )
            return current_score

        previous_smoothed = float(historical_scores[0])
        smoothed = self._alpha * current_score + (1.0 - self._alpha) * previous_smoothed

        _logger.debug(
            "Applied EWMA smoothing",
            extra={
                "current_score": current_score,
                "previous_smoothed": previous_smoothed,
                "alpha": self._alpha,
                "smoothed_score": smoothed,
            },
        )

        return self._clamp_0_100(smoothed)

    def risk_score_to_regime(self, risk_score: float) -> str:
        """
        Konverter smoothed risk_score til regime.

        - "Bull" hvis score < 33
        - "Neutral" hvis 33 ≤ score < 66
        - "Bear" hvis score ≥ 66
        """
        if risk_score < 33.0:
            regime = "Bull"
        elif risk_score < 66.0:
            regime = "Neutral"
        else:
            regime = "Bear"

        _logger.debug(
            "Mapped risk_score to regime",
            extra={"risk_score": risk_score, "regime": regime},
        )
        return regime

    def apply_hysteresis(
        self,
        new_regime: str,
        previous_regime: Optional[str],
        current_score: float,
        previous_score: Optional[float],
    ) -> str:
        """
        Hysterese på regime-endringer.

        Logikk:
        - Hvis previous_regime er None, returner new_regime.
        - Hvis |current_score - previous_score| < hysteresis_threshold,
          behold previous_regime.
        - Ellers returner new_regime.
        """
        if previous_regime is None or previous_score is None:
            return new_regime

        score_delta = abs(current_score - float(previous_score))

        _logger.debug(
            "Evaluating hysteresis",
            extra={
                "new_regime": new_regime,
                "previous_regime": previous_regime,
                "current_score": current_score,
                "previous_score": previous_score,
                "score_delta": score_delta,
                "threshold": self.hysteresis_threshold,
            },
        )

        if score_delta < self.hysteresis_threshold:
            return previous_regime
        return new_regime

    def check_rate_limit(
        self,
        new_regime: str,
        historical_signals: List[Dict[str, Any]],
        target_date: date,
    ) -> str:
        """
        Rate limiting av regime-endringer.

        Input:
        - new_regime: Regime etter hysterese.
        - historical_signals: Signaler fra siste 48+ timer (nyeste først).
        - target_date: Datoen vi genererer signal for (brukes som referansetid).

        Logikk:
        - Hvis ingen historikk: returner new_regime.
        - previous_regime = historiske_signaler[0]["regime"]
        - Hvis new_regime == previous_regime: ingen endring, returner new_regime.
        - Finn tidspunkt for siste regime-endring (basert på created_at hvis
          tilgjengelig, ellers date-felt).
        - Hvis tid fra siste endring til target_date < min_regime_change_hours:
            behold previous_regime
          ellers:
            tillat endring (returner new_regime).
        """
        if not historical_signals:
            return new_regime

        latest = historical_signals[0]
        previous_regime = latest.get("regime")
        if previous_regime is None:
            return new_regime

        if new_regime == previous_regime:
            return new_regime

        # Finn tidspunkt for siste regime-endring.
        last_regime = previous_regime
        last_change_ts: Optional[datetime] = None

        for signal in historical_signals:
            regime = signal.get("regime")
            if regime is None:
                continue

            created_at = signal.get("created_at")
            if isinstance(created_at, datetime):
                ts = created_at
            else:
                # Fallback til date-felt dersom created_at mangler.
                d = signal.get("date")
                if isinstance(d, date):
                    ts = datetime.combine(d, datetime.min.time())
                else:
                    # Hvis verken created_at eller date er gyldig, hopp over.
                    continue

            if regime != last_regime:
                # Dette signalet representerer overgang inn i last_regime.
                last_change_ts = ts
                break

        # Hvis vi ikke fant eksplisitt regime-endring, bruk eldste timestamp.
        if last_change_ts is None:
            oldest = historical_signals[-1]
            created_at = oldest.get("created_at")
            if isinstance(created_at, datetime):
                last_change_ts = created_at
            else:
                d = oldest.get("date")
                if isinstance(d, date):
                    last_change_ts = datetime.combine(d, datetime.min.time())

        if last_change_ts is None:
            # Manglende tidsinformasjon → ingen rate limiting.
            return new_regime

        # Referansetid er target_date (ikke siste historiske signal).
        current_ts = datetime.combine(target_date, datetime.min.time())
        hours_since_change = (current_ts - last_change_ts).total_seconds() / 3600.0

        _logger.debug(
            "Evaluating regime rate limiting",
            extra={
                "new_regime": new_regime,
                "previous_regime": previous_regime,
                "hours_since_change": hours_since_change,
                "min_regime_change_hours": self.min_regime_change_hours,
            },
        )

        if hours_since_change < self.min_regime_change_hours:
            return previous_regime
        return new_regime

    def compute_risk_state(self, risk_score: float) -> str:
        """
        Klassifiser risk state basert på smoothed risk_score.

        - "Risk ON" hvis score < 40
        - "Neutral" hvis 40 ≤ score < 60
        - "Risk OFF" hvis score ≥ 60
        """
        if risk_score < 40.0:
            state = "Risk ON"
        elif risk_score < 60.0:
            state = "Neutral"
        else:
            state = "Risk OFF"

        _logger.debug(
            "Mapped risk_score to risk_state",
            extra={"risk_score": risk_score, "risk_state": state},
        )
        return state

    # ------------------------------------------------------------------ #
    # Secondary indices
    # ------------------------------------------------------------------ #

    def compute_whale_index(self, chain: str, features: Dict[str, float]) -> float:
        """
        Beregn whale_index (0–100).

        Baseres på:
        - whale_flow_1d
        - exchange_netflow_7d

        Høyere verdi = mer whale-aktivitet.

        chain-parameteren er eksplisitt, men logikken er per nå identisk for
        BTC og ETH. Den er inkludert for å være tydelig på kjede-kontekst.
        """
        whale_flow = self._safe_feature_value(
            features, "whale_flow_1d", "whale_index"
        )
        netflow_7d = self._safe_feature_value(
            features, "exchange_netflow_7d", "whale_index"
        )

        # Heuristisk normalisering:
        # - Stor absolutt whale_flow → høyere index.
        # - Stor negativ exchange_netflow_7d (outflows) → høyere index.
        combined = abs(whale_flow) + max(0.0, -netflow_7d)

        index = self._normalize_positive_signal(combined)
        _logger.debug(
            "Computed whale_index",
            extra={
                "chain": chain,
                "whale_flow_1d": whale_flow,
                "exchange_netflow_7d": netflow_7d,
                "combined": combined,
                "whale_index": index,
            },
        )
        return index

    def compute_stress_index(self, chain: str, features: Dict[str, float]) -> float:
        """
        Beregn stress_index (0–100).

        Logikk:
        - BTC: Baseres på mempool_stress og fee_pressure.
        - ETH: Baseres på gas_price_median.
        - Andre kjeder (hvis aktuelt): fallback til generisk logikk.
        - Normaliseres til 0–100 skala.
        """
        chain_upper = chain.upper()

        if chain_upper == "BTC":
            mempool_stress = self._safe_feature_value(
                features, "mempool_stress", "stress_index"
            )
            fee_pressure = self._safe_feature_value(
                features, "fee_pressure", "stress_index"
            )
            combined = mempool_stress + fee_pressure
        elif chain_upper == "ETH":
            gas_price = self._safe_feature_value(
                features, "gas_price_median", "stress_index"
            )
            combined = gas_price
        else:
            # Fallback til tidligere nøkkelbasert logikk for eventuelle andre kjeder.
            has_mempool = "mempool_stress" in features or "fee_pressure" in features
            has_gas_price = "gas_price_median" in features

            if has_mempool:
                mempool_stress = self._safe_feature_value(
                    features, "mempool_stress", "stress_index"
                )
                fee_pressure = self._safe_feature_value(
                    features, "fee_pressure", "stress_index"
                )
                combined = mempool_stress + fee_pressure
            elif has_gas_price:
                gas_price = self._safe_feature_value(
                    features, "gas_price_median", "stress_index"
                )
                combined = gas_price
            else:
                _logger.warning(
                    "Missing stress-related features; using fallback for stress_index",
                    extra={"chain": chain_upper},
                )
                combined = 50.0

        index = self._normalize_positive_signal(combined)
        _logger.debug(
            "Computed stress_index",
            extra={
                "chain": chain_upper,
                "combined_stress_signal": combined,
                "stress_index": index,
            },
        )
        return index

    def compute_momentum_score(self, chain: str, features: Dict[str, float]) -> float:
        """
        Beregn momentum_score (0–100).

        Logikk:
        - Baseres på active_addresses trend.
        - BTC: kombineres med UTXO age (utxo_age_1y).
        - ETH: kombineres med smart_contract_calls.
        - Normaliseres til 0–100 skala.
        """
        chain_upper = chain.upper()

        active_addr = self._safe_feature_value(
            features, "active_addresses", "momentum_score"
        )

        if chain_upper == "BTC":
            utxo_age = self._safe_feature_value(
                features, "utxo_age_1y", "momentum_score"
            )
            combined = active_addr - utxo_age
        elif chain_upper == "ETH":
            smart_calls = self._safe_feature_value(
                features, "smart_contract_calls", "momentum_score"
            )
            combined = active_addr + smart_calls
        else:
            # Fallback til tidligere nøkkelbasert logikk.
            if "utxo_age_1y" in features:  # BTC-proxy
                utxo_age = self._safe_feature_value(
                    features, "utxo_age_1y", "momentum_score"
                )
                combined = active_addr - utxo_age
            elif "smart_contract_calls" in features:  # ETH-proxy
                smart_calls = self._safe_feature_value(
                    features, "smart_contract_calls", "momentum_score"
                )
                combined = active_addr + smart_calls
            else:
                _logger.warning(
                    "Missing chain-specific momentum features; using active_addresses only",
                    extra={"chain": chain_upper},
                )
                combined = active_addr

        index = self._normalize_positive_signal(combined)
        _logger.debug(
            "Computed momentum_score",
            extra={
                "chain": chain_upper,
                "combined_momentum_signal": combined,
                "momentum_score": index,
            },
        )
        return index

    # ------------------------------------------------------------------ #
    # Alerts
    # ------------------------------------------------------------------ #

    def generate_alerts(
        self,
        chain: str,
        target_date: date,
        current_signal: DailySignal,
        previous_signal: Optional[Dict[str, Any]],
        features: Dict[str, float],  # reserved for future use (e.g. feature-based alerts)
    ) -> List[Alert]:
        """
        Generer alerts basert på dagens og forrige signal.

        Alert-typer og triggere:
        - whale_spike:
            whale_index øker > 20 poeng på én dag.
        - stress_event:
            stress_index > 80.
        - momentum_collapse:
            momentum_score faller > 25 poeng på én dag.
        - regime_shift:
            regime endres fra forrige dag.

        Merk:
        - features-parameteren er forberedt for fremtidige feature-baserte
          alerts, men er per nå ikke i aktiv bruk.
        """
        alerts: List[Alert] = []

        prev_whale = (
            float(previous_signal.get("whale_index"))
            if previous_signal is not None
            and isinstance(previous_signal.get("whale_index"), (int, float))
            else None
        )
        prev_momentum = (
            float(previous_signal.get("momentum_score"))
            if previous_signal is not None
            and isinstance(previous_signal.get("momentum_score"), (int, float))
            else None
        )
        prev_regime = (
            previous_signal.get("regime") if previous_signal is not None else None
        )

        # whale_spike
        if prev_whale is not None:
            delta_whale = current_signal.whale_index - prev_whale
            if delta_whale > 20.0:
                alerts.append(
                    Alert(
                        chain=chain,
                        date=target_date,
                        alert_type="whale_spike",
                        message=(
                            f"Whale activity spike: whale_index up by "
                            f"{delta_whale:.1f} points vs previous day."
                        ),
                    )
                )

        # stress_event
        if current_signal.stress_index > 80.0:
            alerts.append(
                Alert(
                    chain=chain,
                    date=target_date,
                    alert_type="stress_event",
                    message=(
                        f"On-chain stress elevated: stress_index at "
                        f"{current_signal.stress_index:.1f}."
                    ),
                )
            )

        # momentum_collapse
        if prev_momentum is not None:
            delta_momentum = prev_momentum - current_signal.momentum_score
            if delta_momentum > 25.0:
                alerts.append(
                    Alert(
                        chain=chain,
                        date=target_date,
                        alert_type="momentum_collapse",
                        message=(
                            f"Momentum collapse: momentum_score down by "
                            f"{delta_momentum:.1f} points vs previous day."
                        ),
                    )
                )

        # regime_shift
        if prev_regime is not None and prev_regime != current_signal.regime:
            alerts.append(
                Alert(
                    chain=chain,
                    date=target_date,
                    alert_type="regime_shift",
                    message=(
                                f"Regime shift from {prev_regime} to "
                                f"{current_signal.regime}."
                    ),
                )
            )

        _logger.debug(
            "Generated alerts",
            extra={
                "chain": chain,
                "date": target_date.isoformat(),
                "alerts_count": len(alerts),
                "alert_types": [a.alert_type for a in alerts],
            },
        )

        return alerts

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _clamp_0_100(value: float) -> float:
        """Clamp en verdi til [0, 100]."""
        return max(0.0, min(100.0, value))

    def _sanitize_risk_score(self, raw_score: float) -> float:
        """
        Rydd opp i rå risk_score:

        - Hvis NaN/inf → fallback til 50.0 med warning.
        - Clamp til [0, 100] med warning hvis utenfor intervallet.
        """
        if not isinstance(raw_score, (int, float)) or math.isnan(raw_score):
            _logger.warning(
                "Non-finite or non-numeric raw_risk_score; using fallback 50.0",
                extra={"raw_risk_score": raw_score},
            )
            return 50.0

        if not math.isfinite(raw_score):
            _logger.warning(
                "Infinite raw_risk_score; using fallback 50.0",
                extra={"raw_risk_score": raw_score},
            )
            return 50.0

        if raw_score < 0.0 or raw_score > 100.0:
            _logger.warning(
                "raw_risk_score outside [0, 100]; clamping",
                extra={"raw_risk_score": raw_score},
            )
        return self._clamp_0_100(float(raw_score))

    @staticmethod
    def _extract_historical_scores(
        historical_signals: List[Dict[str, Any]],
    ) -> List[float]:
        """Hent risk_score-verdier fra historiske signaler (nyeste først)."""
        scores: List[float] = []
        for signal in historical_signals:
            value = signal.get("risk_score")
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                scores.append(float(value))
        return scores

    def _safe_feature_value(
        self,
        features: Dict[str, float],
        key: str,
        index_name: str,
        fallback: float = 50.0,
    ) -> float:
        """
        Hent feature-verdi med robust feilhåndtering.

        - Manglende nøkkel → warning + fallback.
        - NaN/inf → warning + fallback.
        - Ikke-numerisk → warning + fallback.
        """
        if key not in features:
            _logger.warning(
                "Missing feature key for index computation; using fallback",
                extra={"feature_key": key, "index_name": index_name},
            )
            return fallback

        raw_value = features.get(key)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            _logger.warning(
                "Non-numeric feature value for index computation; using fallback",
                extra={
                    "feature_key": key,
                    "index_name": index_name,
                    "value": raw_value,
                },
            )
            return fallback

        if not math.isfinite(value):
            _logger.warning(
                "Non-finite feature value for index computation; using fallback",
                extra={
                    "feature_key": key,
                    "index_name": index_name,
                    "value": value,
                },
            )
            return fallback

        return value

    @staticmethod
    def _normalize_positive_signal(signal_value: float) -> float:
        """
        Heuristisk normalisering av en (typisk positiv) signalverdi til [0, 100].

        Bruker en enkel tanh-baserte komprimering rundt 0:

            index = 50 + 50 * tanh(signal_value / scale)

        med scale satt til en moderat verdi for å unngå ekstreme hopp.
        """
        scale = 10.0  # konservativ skala; kan tunes ved behov
        z = signal_value / scale
        normalized = 50.0 + 50.0 * math.tanh(z)
        return max(0.0, min(100.0, normalized))
