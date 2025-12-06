"""
DuckDB handler for the ON-CHAIN SUPER SIGNALS™ project.

Denne modulen er det eneste grensesnittet mot DuckDB-databasen (signals.duckdb).

Ansvar:
- Initialisere databaseskjema (daily_signals, performance_summary, alerts)
- Håndtere alle CRUD-operasjoner mot disse tabellene
- Håndheve dataminimering (kun aggregerte data lagres)
- Sørge for trygg bruk av DuckDB via parameteriserte spørringer

Spesifikasjon og krav:
- Se db_handler-spesifikasjonen fra arkitekt (Claude)
- Se styringsdokumentet for dataminimering og lagringskrav
- Se arkitekturdokumentet for DuckDB-schema og helhetlig dataflyt

"""

from __future__ import annotations

import math
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import duckdb

from src.config.settings import CHAINS, DB_PATH
from src.utils import validators
from src.utils.logger import get_logger

_logger = get_logger(__name__)

# Tillatte verdier for regime og risk_state iht. spesifikasjon
_ALLOWED_REGIMES = {"Bull", "Neutral", "Bear"}
_ALLOWED_RISK_STATES = {"Risk ON", "Neutral", "Risk OFF"}

# Tillatte alert-typer
_ALLOWED_ALERT_TYPES = {
    "whale_spike",
    "stress_event",
    "momentum_collapse",
    "regime_shift",
}


# ===================================================================== #
#  MODUL-NIVÅ FUNKSJONER – TILKOBLING & INIT                            #
# ===================================================================== #


def get_connection(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """
    Returnerer en DuckDB-tilkobling til signals.duckdb.

    Parametre
    ---------
    read_only:
        Om tilkoblingen skal være skrivebeskyttet.

    Returnerer
    ----------
    duckdb.DuckDBPyConnection

    Kaster
    ------
    FileNotFoundError
        Hvis DB_PATH ikke eksisterer og read_only=True.
    duckdb.Error
        Ved tilkoblingsfeil.
    """
    db_path = Path(DB_PATH)

    if read_only and not db_path.exists():
        _logger.error(
            "Attempted read-only connection to non-existent database",
            extra={"db_path": str(db_path)},
        )
        raise FileNotFoundError(f"DuckDB file does not exist: {db_path}")

    if not read_only:
        db_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        conn = duckdb.connect(str(db_path), read_only=read_only)
        _logger.debug(
            "Established DuckDB connection",
            extra={"db_path": str(db_path), "read_only": read_only},
        )
        return conn
    except duckdb.Error as exc:  # pragma: no cover - fatal connection issues
        _logger.error(
            "Failed to connect to DuckDB",
            extra={"db_path": str(db_path), "read_only": read_only, "error": str(exc)},
        )
        raise


@contextmanager
def _get_connection(read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    """
    Context manager rundt get_connection() for å sikre korrekt lukking.

    Dette følger anbefalingen om å bruke én tilkobling per script-kjøring,
    men gir samtidig et enkelt og sikkert API i utils-laget.
    """
    conn = get_connection(read_only=read_only)
    try:
        yield conn
    finally:
        conn.close()


def init_database() -> None:
    """
    Oppretter alle nødvendige tabeller hvis de ikke finnes.

    Tabeller som opprettes:
    - daily_signals
    - performance_summary
    - alerts

    Idempotent: Kan kalles flere ganger uten sideeffekter.

    Kaster
    ------
    duckdb.Error
        Ved skjemafeil.
    PermissionError
        Hvis prosessen mangler skrivetilgang.
    """
    with _get_connection(read_only=False) as conn:
        try:
            _logger.info("Initializing DuckDB schema if missing")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_signals (
                    chain VARCHAR NOT NULL,
                    date DATE NOT NULL,
                    risk_score FLOAT NOT NULL,
                    regime VARCHAR NOT NULL,
                    risk_state VARCHAR NOT NULL,
                    whale_index FLOAT NOT NULL,
                    stress_index FLOAT NOT NULL,
                    momentum_score FLOAT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (chain, date)
                );
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS performance_summary (
                    chain VARCHAR NOT NULL,
                    model_version VARCHAR NOT NULL,
                    auc FLOAT,
                    hitrate FLOAT,
                    sharpe_ratio FLOAT,
                    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (chain, model_version)
                );
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY,
                    chain VARCHAR NOT NULL,
                    date DATE NOT NULL,
                    alert_type VARCHAR NOT NULL,
                    message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            _logger.info("DuckDB schema initialized (or already present)")
        except duckdb.Error as exc:
            # Grov sjekk på rettighet-problemer
            msg = str(exc).lower()
            if "permission" in msg or "readonly" in msg:
                _logger.error(
                    "Permission error while initializing database schema",
                    extra={"error": str(exc)},
                )
                raise PermissionError(
                    "Insufficient permissions to initialize DuckDB schema"
                ) from exc

            _logger.error(
                "Failed to initialize DuckDB schema",
                extra={"error": str(exc)},
            )
            raise


# ===================================================================== #
#  HJELPEFUNKSJONER – VALIDERING & KONVERTERING                         #
# ===================================================================== #


def _validate_chain(chain: str) -> str:
    """Validerer chain-navn og returnerer uppercased versjon."""
    validators.validate_chain(chain)
    chain_upper = chain.upper()
    if chain_upper not in CHAINS:
        raise ValueError(f"Unsupported chain: {chain}")
    return chain_upper


def _validate_not_future_date(d: date, field_name: str = "date") -> None:
    """Sikrer at dato ikke ligger i fremtiden."""
    if d > date.today():
        raise ValueError(f"{field_name} {d.isoformat()} cannot be in the future")


def _validate_score_0_100(value: float, field_name: str) -> None:
    """
    Sikrer at en score er endelig (ikke NaN/inf) og ligger innenfor [0, 100].

    Dette adresserer review-kommentaren om eksplisitt håndtering av NaN/inf
    i lagrede scorer.
    """
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite, got {value}")
    if not (0.0 <= value <= 100.0):
        raise ValueError(f"{field_name} must be in [0, 100], got {value}")


def _validate_prob_0_1(value: Optional[float], field_name: str) -> None:
    """Sikrer at en sannsynlighet (hvis satt) ligger innenfor [0, 1]."""
    if value is None:
        return
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{field_name} must be in [0, 1], got {value}")


def _row_to_dict(cursor: duckdb.DuckDBPyConnection, row: Any) -> Dict[str, Any]:
    """
    Konverterer én rad fra DuckDB til dict basert på cursor.description.

    Brukes for å unngå duplisert kolonnehåndtering i flere funksjoner.
    """
    columns = [col[0] for col in cursor.description]
    return dict(zip(columns, row))


def _rows_to_dicts(
    cursor: duckdb.DuckDBPyConnection,
    rows: List[Any],
) -> List[Dict[str, Any]]:
    """
    Konverterer flere rader fra DuckDB til liste av dicts.

    Brukes for å erstatte gjentatt zip()-logikk i query-funksjoner.
    """
    columns = [col[0] for col in cursor.description]
    return [dict(zip(columns, row)) for row in rows]


# ===================================================================== #
#  DAILY_SIGNALS OPERASJONER                                            #
# ===================================================================== #


def insert_daily_signals(
    chain: str,
    date_: date,
    risk_score: float,
    regime: str,
    risk_state: str,
    whale_index: float,
    stress_index: float,
    momentum_score: float,
) -> None:
    """
    Setter inn eller oppdaterer én rad i daily_signals.

    Bruker UPSERT (INSERT OR REPLACE) for å håndtere re-kjøringer.
    Skrivingen gjøres innen en eksplisitt transaksjon for å sikre atomisitet.

    Parametre
    ---------
    chain:
        "BTC" eller "ETH"
    date_:
        Dato for signalet
    risk_score:
        Risikoscore 0-100
    regime:
        "Bull", "Neutral", eller "Bear"
    risk_state:
        "Risk ON", "Neutral", eller "Risk OFF"
    whale_index:
        0-100
    stress_index:
        0-100
    momentum_score:
        0-100

    Kaster
    ------
    ValueError
        Ved ugyldig input.
    duckdb.Error
        Ved databasefeil.
    """
    chain_upper = _validate_chain(chain)
    _validate_not_future_date(date_, "date")
    _validate_score_0_100(risk_score, "risk_score")
    _validate_score_0_100(whale_index, "whale_index")
    _validate_score_0_100(stress_index, "stress_index")
    _validate_score_0_100(momentum_score, "momentum_score")

    if regime not in _ALLOWED_REGIMES:
        raise ValueError(f"regime must be one of {_ALLOWED_REGIMES}, got {regime}")

    if risk_state not in _ALLOWED_RISK_STATES:
        raise ValueError(
            f"risk_state must be one of {_ALLOWED_RISK_STATES}, got {risk_state}"
        )

    with _get_connection(read_only=False) as conn:
        try:
            _logger.debug(
                "Inserting daily_signals row (UPSERT)",
                extra={
                    "chain": chain_upper,
                    "date": date_.isoformat(),
                    "risk_score": risk_score,
                    "regime": regime,
                    "risk_state": risk_state,
                },
            )
            conn.execute("BEGIN;")
            conn.execute(
                """
                INSERT OR REPLACE INTO daily_signals (
                    chain,
                    date,
                    risk_score,
                    regime,
                    risk_state,
                    whale_index,
                    stress_index,
                    momentum_score
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                [
                    chain_upper,
                    date_,
                    float(risk_score),
                    regime,
                    risk_state,
                    float(whale_index),
                    float(stress_index),
                    float(momentum_score),
                ],
            )
            conn.execute("COMMIT;")
            _logger.info(
                "Inserted/updated daily_signals row",
                extra={"chain": chain_upper, "date": date_.isoformat()},
            )
        except duckdb.Error as exc:
            try:
                conn.execute("ROLLBACK;")
            except duckdb.Error:
                # Hvis rollback feiler, logges det men overdøver ikke opprinnelig feil.
                _logger.error(
                    "Rollback failed after insert_daily_signals error",
                    extra={"chain": chain_upper, "date": date_.isoformat()},
                )
            _logger.error(
                "Failed to insert/update daily_signals row",
                extra={
                    "chain": chain_upper,
                    "date": date_.isoformat(),
                    "error": str(exc),
                },
            )
            raise


def get_latest_signal(chain: str) -> Optional[Dict[str, Any]]:
    """
    Henter siste signalrad for gitt kjede.

    Parametre
    ---------
    chain:
        "BTC" eller "ETH"

    Returnerer
    ----------
    Optional[Dict[str, Any]]
        Dict med alle kolonner fra daily_signals, eller None hvis ingen finnes.
        Format:
        {
          "chain": str,
          "date": datetime.date,
          "risk_score": float,
          "regime": str,
          "risk_state": str,
          "whale_index": float,
          "stress_index": float,
          "momentum_score": float,
          "created_at": datetime
        }

    Kaster
    ------
    ValueError
        Hvis chain er ugyldig.
    duckdb.Error
        Ved databasefeil.
    """
    chain_upper = _validate_chain(chain)

    with _get_connection(read_only=True) as conn:
        try:
            cursor = conn.execute(
                """
                SELECT
                    chain,
                    date,
                    risk_score,
                    regime,
                    risk_state,
                    whale_index,
                    stress_index,
                    momentum_score,
                    created_at
                FROM daily_signals
                WHERE chain = ?
                ORDER BY date DESC
                LIMIT 1;
                """,
                [chain_upper],
            )
            row = cursor.fetchone()
            if row is None:
                _logger.warning(
                    "No latest signal found for chain",
                    extra={"chain": chain_upper},
                )
                return None

            result = _row_to_dict(cursor, row)

            _logger.info(
                "Fetched latest signal",
                extra={"chain": chain_upper, "date": str(result.get("date"))},
            )
            return result
        except duckdb.Error as exc:
            _logger.error(
                "Failed to fetch latest signal",
                extra={"chain": chain_upper, "error": str(exc)},
            )
            raise


def query_signals(
    chain: str,
    start_date: date,
    end_date: date,
) -> List[Dict[str, Any]]:
    """
    Henter signalhistorikk for gitt kjede og datointervall.

    Merk:
    Arkitekturdokumentet spesifiserer opprinnelig returtype pd.DataFrame.
    For å holde utils-laget uavhengig av pandas returneres her en
    List[Dict[str, Any]]. Kallere som trenger DataFrame kan enkelt
    konvertere resultatet eksplisitt.

    Parametre
    ---------
    chain:
        "BTC" eller "ETH"
    start_date:
        Startdato (inklusiv)
    end_date:
        Sluttdato (inklusiv)

    Returnerer
    ----------
    List[Dict[str, Any]]
        Liste av dicts, sortert etter dato (eldste først).
        Tom liste hvis ingen data finnes.

    Kaster
    ------
    ValueError
        Ved ugyldig chain eller datointervall.
    duckdb.Error
        Ved databasefeil.
    """
    chain_upper = _validate_chain(chain)
    validators.validate_date_range(start_date, end_date)

    with _get_connection(read_only=True) as conn:
        try:
            cursor = conn.execute(
                """
                SELECT
                    chain,
                    date,
                    risk_score,
                    regime,
                    risk_state,
                    whale_index,
                    stress_index,
                    momentum_score,
                    created_at
                FROM daily_signals
                WHERE chain = ?
                  AND date BETWEEN ? AND ?
                ORDER BY date ASC;
                """,
                [chain_upper, start_date, end_date],
            )
            rows = cursor.fetchall()
            if not rows:
                _logger.warning(
                    "No signals found for chain in date range",
                    extra={
                        "chain": chain_upper,
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                    },
                )
                return []

            result = _rows_to_dicts(cursor, rows)

            _logger.info(
                "Fetched signals for chain in date range",
                extra={
                    "chain": chain_upper,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "count": len(result),
                },
            )
            return result
        except duckdb.Error as exc:
            _logger.error(
                "Failed to query signals",
                extra={
                    "chain": chain_upper,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "error": str(exc),
                },
            )
            raise


def get_signal_history(chain: str, days: int = 30) -> List[Dict[str, Any]]:
    """
    Henter de siste N dagene med signaler for gitt kjede.

    Parametre
    ---------
    chain:
        "BTC" eller "ETH"
    days:
        Antall dager å hente (default 30)

    Returnerer
    ----------
    List[Dict[str, Any]]
        Liste av dicts, sortert etter dato (nyeste først).

    Kaster
    ------
    ValueError
        Hvis chain er ugyldig eller days < 1.
    duckdb.Error
        Ved databasefeil.
    """
    if days < 1:
        raise ValueError("days must be >= 1")

    chain_upper = _validate_chain(chain)

    with _get_connection(read_only=True) as conn:
        try:
            cursor = conn.execute(
                """
                SELECT
                    chain,
                    date,
                    risk_score,
                    regime,
                    risk_state,
                    whale_index,
                    stress_index,
                    momentum_score,
                    created_at
                FROM daily_signals
                WHERE chain = ?
                ORDER BY date DESC
                LIMIT ?;
                """,
                [chain_upper, days],
            )
            rows = cursor.fetchall()
            if not rows:
                _logger.warning(
                    "No signal history found for chain",
                    extra={"chain": chain_upper, "days": days},
                )
                return []

            result = _rows_to_dicts(cursor, rows)

            _logger.info(
                "Fetched signal history",
                extra={"chain": chain_upper, "days": days, "count": len(result)},
            )
            return result
        except duckdb.Error as exc:
            _logger.error(
                "Failed to fetch signal history",
                extra={"chain": chain_upper, "days": days, "error": str(exc)},
            )
            raise


# ===================================================================== #
#  ALERTS OPERASJONER                                                   #
# ===================================================================== #


def insert_alert(
    chain: str,
    date_: date,
    alert_type: str,
    message: str,
) -> int:
    """
    Setter inn en ny alert og returnerer dens ID.

    Skrivingen gjøres innen en eksplisitt transaksjon for å sikre atomisitet.

    Parametre
    ---------
    chain:
        "BTC" eller "ETH"
    date_:
        Dato for alerten
    alert_type:
        En av:
        - "whale_spike"
        - "stress_event"
        - "momentum_collapse"
        - "regime_shift"
    message:
        Beskrivende melding

    Returnerer
    ----------
    int
        ID for den nye alerten.

    Kaster
    ------
    ValueError
        Ved ugyldig input.
    duckdb.Error
        Ved databasefeil.
    """
    chain_upper = _validate_chain(chain)
    _validate_not_future_date(date_, "date")

    if alert_type not in _ALLOWED_ALERT_TYPES:
        raise ValueError(
            f"alert_type must be one of {_ALLOWED_ALERT_TYPES}, got {alert_type}"
        )

    with _get_connection(read_only=False) as conn:
        try:
            _logger.debug(
                "Inserting alert",
                extra={
                    "chain": chain_upper,
                    "date": date_.isoformat(),
                    "alert_type": alert_type,
                },
            )
            conn.execute("BEGIN;")
            row = conn.execute(
                """
                INSERT INTO alerts (
                    chain,
                    date,
                    alert_type,
                    message
                )
                VALUES (?, ?, ?, ?)
                RETURNING id;
                """,
                [chain_upper, date_, alert_type, message],
            ).fetchone()
            conn.execute("COMMIT;")
            alert_id = int(row[0])

            _logger.info(
                "Inserted alert",
                extra={
                    "id": alert_id,
                    "chain": chain_upper,
                    "date": date_.isoformat(),
                    "alert_type": alert_type,
                },
            )
            return alert_id
        except duckdb.Error as exc:
            try:
                conn.execute("ROLLBACK;")
            except duckdb.Error:
                _logger.error(
                    "Rollback failed after insert_alert error",
                    extra={"chain": chain_upper, "date": date_.isoformat()},
                )
            _logger.error(
                "Failed to insert alert",
                extra={
                    "chain": chain_upper,
                    "date": date_.isoformat(),
                    "alert_type": alert_type,
                    "error": str(exc),
                },
            )
            raise


def get_alerts_for_date(
    date_: date,
    chain: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Henter alle alerts for en gitt dato.

    Parametre
    ---------
    date_:
        Dato å hente alerts for.
    chain:
        Valgfritt filter på kjede.

    Returnerer
    ----------
    List[Dict[str, Any]]
        Liste av alert-dicts.
    """
    params: List[Any] = [date_]
    where_clause = "date = ?"

    chain_upper: Optional[str] = None
    if chain is not None:
        chain_upper = _validate_chain(chain)
        where_clause += " AND chain = ?"
        params.append(chain_upper)

    with _get_connection(read_only=True) as conn:
        try:
            cursor = conn.execute(
                f"""
                SELECT
                    id,
                    chain,
                    date,
                    alert_type,
                    message,
                    created_at
                FROM alerts
                WHERE {where_clause}
                ORDER BY created_at ASC;
                """,
                params,
            )
            rows = cursor.fetchall()
            if not rows:
                _logger.warning(
                    "No alerts found for date",
                    extra={
                        "date": date_.isoformat(),
                        "chain": chain_upper,
                    },
                )
                return []

            result = _rows_to_dicts(cursor, rows)

            _logger.info(
                "Fetched alerts for date",
                extra={
                    "date": date_.isoformat(),
                    "chain": chain_upper,
                    "count": len(result),
                },
            )
            return result
        except duckdb.Error as exc:
            _logger.error(
                "Failed to fetch alerts for date",
                extra={
                    "date": date_.isoformat(),
                    "chain": chain_upper,
                    "error": str(exc),
                },
            )
            raise


def get_recent_alerts(days: int = 7) -> List[Dict[str, Any]]:
    """
    Henter alle alerts fra de siste N dagene.

    Parametre
    ---------
    days:
        Antall dager (default 7)

    Returnerer
    ----------
    List[Dict[str, Any]]
        Liste av alert-dicts, sortert etter dato (nyeste først).
    """
    if days < 1:
        raise ValueError("days must be >= 1")

    end_date = date.today()
    start_date = end_date - timedelta(days=days - 1)

    with _get_connection(read_only=True) as conn:
        try:
            cursor = conn.execute(
                """
                SELECT
                    id,
                    chain,
                    date,
                    alert_type,
                    message,
                    created_at
                FROM alerts
                WHERE date BETWEEN ? AND ?
                ORDER BY date DESC, created_at DESC;
                """,
                [start_date, end_date],
            )
            rows = cursor.fetchall()
            if not rows:
                _logger.warning(
                    "No recent alerts found",
                    extra={"days": days},
                )
                return []

            result = _rows_to_dicts(cursor, rows)

            _logger.info(
                "Fetched recent alerts",
                extra={"days": days, "count": len(result)},
            )
            return result
        except duckdb.Error as exc:
            _logger.error(
                "Failed to fetch recent alerts",
                extra={"days": days, "error": str(exc)},
            )
            raise


# ===================================================================== #
#  PERFORMANCE_SUMMARY OPERASJONER                                      #
# ===================================================================== #


def insert_performance_summary(
    chain: str,
    model_version: str,
    auc: Optional[float],
    hitrate: Optional[float],
    sharpe_ratio: Optional[float],
) -> None:
    """
    Setter inn treningsmetadata for en modellversjon.

    Bruker UPSERT for å håndtere re-trening av samme versjon.
    Skrivingen gjøres innen en eksplisitt transaksjon for å sikre atomisitet.

    Parametre
    ---------
    chain:
        "BTC" eller "ETH"
    model_version:
        Versjonsnummer som streng (f.eks. "1", "2")
    auc:
        AUC-score (kan være None før evaluering)
    hitrate:
        Hit-rate for drawdowns
    sharpe_ratio:
        Sharpe ratio forbedring

    Validering
    ----------
    - auc, hvis oppgitt, bør være [0, 1]
    - hitrate, hvis oppgitt, bør være [0, 1]

    Kaster
    ------
    ValueError
        Ved ugyldig input.
    duckdb.Error
        Ved databasefeil.
    """
    chain_upper = _validate_chain(chain)
    _validate_prob_0_1(auc, "auc")
    _validate_prob_0_1(hitrate, "hitrate")

    with _get_connection(read_only=False) as conn:
        try:
            _logger.debug(
                "Inserting performance_summary row (UPSERT)",
                extra={
                    "chain": chain_upper,
                    "model_version": model_version,
                    "auc": auc,
                    "hitrate": hitrate,
                    "sharpe_ratio": sharpe_ratio,
                },
            )
            conn.execute("BEGIN;")
            conn.execute(
                """
                INSERT OR REPLACE INTO performance_summary (
                    chain,
                    model_version,
                    auc,
                    hitrate,
                    sharpe_ratio
                )
                VALUES (?, ?, ?, ?, ?);
                """,
                [
                    chain_upper,
                    str(model_version),
                    auc,
                    hitrate,
                    sharpe_ratio,
                ],
            )
            conn.execute("COMMIT;")
            _logger.info(
                "Inserted/updated performance_summary row",
                extra={"chain": chain_upper, "model_version": model_version},
            )
        except duckdb.Error as exc:
            try:
                conn.execute("ROLLBACK;")
            except duckdb.Error:
                _logger.error(
                    "Rollback failed after insert_performance_summary error",
                    extra={"chain": chain_upper, "model_version": model_version},
                )
            _logger.error(
                "Failed to insert/update performance_summary row",
                extra={
                    "chain": chain_upper,
                    "model_version": model_version,
                    "error": str(exc),
                },
            )
            raise


def get_latest_model_performance(chain: str) -> Optional[Dict[str, Any]]:
    """
    Henter performance-data for siste modellversjon.

    "Siste" er definert som raden med høyest trained_at-timestamp for kjeden,
    i tråd med review-kommentar om at re-trening av samme versjon skal
    overskrive tidligere rader.

    Parametre
    ---------
    chain:
        "BTC" eller "ETH"

    Returnerer
    ----------
    Optional[Dict[str, Any]]
        Dict med performance-data eller None.
    """
    chain_upper = _validate_chain(chain)

    with _get_connection(read_only=True) as conn:
        try:
            cursor = conn.execute(
                """
                SELECT
                    chain,
                    model_version,
                    auc,
                    hitrate,
                    sharpe_ratio,
                    trained_at
                FROM performance_summary
                WHERE chain = ?
                ORDER BY trained_at DESC
                LIMIT 1;
                """,
                [chain_upper],
            )
            row = cursor.fetchone()
            if row is None:
                _logger.warning(
                    "No model performance found for chain",
                    extra={"chain": chain_upper},
                )
                return None

            result = _row_to_dict(cursor, row)

            _logger.info(
                "Fetched latest model performance",
                extra={
                    "chain": chain_upper,
                    "model_version": result.get("model_version"),
                },
            )
            return result
        except duckdb.Error as exc:
            _logger.error(
                "Failed to fetch latest model performance",
                extra={"chain": chain_upper, "error": str(exc)},
            )
            raise


# ===================================================================== #
#  VEDLIKEHOLDSFUNKSJONER                                               #
# ===================================================================== #


def delete_tmp_tables() -> None:
    """
    Sletter eventuelle midlertidige tabeller.

    OBS:
    - Denne funksjonen skal IKKE slette de tre permanente tabellene
      (daily_signals, performance_summary, alerts).
    - Den er kun for opprydding av ad-hoc tabeller brukt under beregninger.

    Merk:
    DuckDB støtter ikke parameterisering av tabellnavn i DROP TABLE,
    derfor brukes f-string for qualified_name. Verdiene som brukes kommer
    kun fra information_schema.tables og ikke fra brukerinput, og er derfor
    vurdert som trygge mot SQL-injeksjon.
    """
    with _get_connection(read_only=False) as conn:
        try:
            # Finn alle tabeller i 'temp'-schema eller med navn som starter på 'tmp_'
            cursor = conn.execute(
                """
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE
                    (table_schema = 'temp' OR table_name ILIKE 'tmp_%')
                ORDER BY table_schema, table_name;
                """
            )
            rows = cursor.fetchall()
            if not rows:
                _logger.info("No temporary tables found to delete")
                return

            deleted = 0
            for schema, name in rows:
                # Ekstra sikring: ikke rør permanente hovedtabeller
                if name in {"daily_signals", "performance_summary", "alerts"}:
                    continue

                qualified_name = (
                    f'"{schema}"."{name}"' if schema and schema != "main" else f'"{name}"'
                )
                _logger.debug(
                    "Dropping temporary table",
                    extra={"schema": schema, "name": name},
                )
                conn.execute(f"DROP TABLE IF EXISTS {qualified_name};")
                deleted += 1

            _logger.info(
                "Temporary tables cleanup completed",
                extra={"deleted_tables": deleted},
            )
        except duckdb.Error as exc:
            _logger.error(
                "Failed to delete temporary tables",
                extra={"error": str(exc)},
            )
            raise


def vacuum_database() -> None:
    """
    Kjører VACUUM på databasen for å frigjøre plass.

    Bør kjøres månedlig per arkitekturspesifikasjonen.
    """
    with _get_connection(read_only=False) as conn:
        try:
            _logger.info("Running VACUUM on DuckDB database")
            conn.execute("VACUUM;")
            _logger.info("VACUUM completed successfully")
        except duckdb.Error as exc:
            _logger.error(
                "Failed to run VACUUM on database",
                extra={"error": str(exc)},
            )
            raise
