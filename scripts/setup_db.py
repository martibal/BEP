from __future__ import annotations

"""
scripts/setup_db.py

Hensikt:
Orkestrere engangs-oppsettet av prosjektmiljøet.
1. Opprette nødvendige filsystemkataloger.
2. Initialisere DuckDB-databasens skjema.
"""

import sys
from pathlib import Path

from config import settings
from src.utils import db_handler
from src.utils.logger import get_logger, setup_logging

# Initialiser logger (konfigureres i main via setup_logging()).
LOGGER = get_logger("setup")


def _ensure_directories() -> None:
    """
    Opprett eller verifiser nødvendige filsystemkataloger.

    Basert på arkitekturen skal følgende kataloger finnes:
    - data/db/ (forelder til DB_PATH)
    - data/tmp/ (TMP_DIR)
    - data/models/ (MODEL_DIR)
    - data/normalization/ (NORMALIZATION_DIR, dersom definert)
    - logs/ (LOG_DIR)

    Denne funksjonen er bevisst plassert utenfor config-laget for å unngå
    operasjonell logikk i config/settings.py.

    Merk:
    - DB_PATH, TMP_DIR, MODEL_DIR og LOG_DIR er obligatoriske og hentes
      direkte fra settings. Hvis noen av disse mangler, vil en AttributeError
      bli kastet, som er korrekt for kritisk manglende konfigurasjon.
    - NORMALIZATION_DIR er valgfri og håndteres derfor defensivt.
    """
    dirs_to_ensure: set[Path] = set()

    # Foreldrekatalog til DuckDB-fil (obligatorisk konfigurasjon).
    db_path = Path(settings.DB_PATH)
    if db_path.parent:
        dirs_to_ensure.add(db_path.parent)

    # Midlertidige filer (obligatorisk konfigurasjon).
    tmp_dir = Path(settings.TMP_DIR)
    dirs_to_ensure.add(tmp_dir)

    # Modellkatalog (obligatorisk konfigurasjon).
    model_dir = Path(settings.MODEL_DIR)
    dirs_to_ensure.add(model_dir)

    # Loggkatalog (obligatorisk konfigurasjon).
    log_dir = Path(settings.LOG_DIR)
    dirs_to_ensure.add(log_dir)

    # Normaliseringsparametre (valgfritt, avhengig av settings).
    if hasattr(settings, "NORMALIZATION_DIR"):
        normalization_dir = Path(settings.NORMALIZATION_DIR)
        dirs_to_ensure.add(normalization_dir)

    for directory in dirs_to_ensure:
        directory.mkdir(parents=True, exist_ok=True)


def _initialize_db() -> None:
    """
    Initialiser DuckDB-databasens skjema via db_handler.

    Denne funksjonen kapsler kall til db_handler.initialize_database()
    for å forbedre testbarhet og lesbarhet.

    Idempotens
    ----------
    For å verifisere at database-initialisering er idempotent (dvs. trygt
    kan kjøres flere ganger uten feil), kalles initialize_database() to
    ganger på rad. Den andre kallet skal fullføre uten å kaste unntak hvis
    alle CREATE TABLE-setninger er definert med IF NOT EXISTS i henhold til
    arkitekturen.
    """
    LOGGER.info("Initializing DuckDB database schema (first pass)...")
    db_handler.initialize_database()
    LOGGER.info("DuckDB database schema initialized/verified (first pass).")

    LOGGER.info("Verifying idempotent DuckDB database initialization (second pass)...")
    db_handler.initialize_database()
    LOGGER.info(
        "DuckDB database schema initialization verified as idempotent (second pass completed)."
    )


def main() -> int:
    """
    Hovedinngangspunkt for oppsett-scriptet.
    Utfører katalogopprettelse og databaseinitialisering.

    Flyt
    ----
    1. Sørg for at alle nødvendige kataloger eksisterer.
    2. Initialiser logging.
    3. Kjør database-initialisering (inkludert idempotens-verifisering).
    4. Logg resultat og returner exit-kode.

    Returnerer
    ---------
    int
        Exit code (0 for suksess, 1 for feil).
    """
    # 1. Opprett kataloger før logging settes opp, slik at loggmappen
    #    garantert eksisterer når setup_logging() konfigurerer filhandlere.
    _ensure_directories()

    # 2. Sikre at logging er satt opp før ytterligere bruk av LOGGER.
    setup_logging()

    try:
        LOGGER.info("Starting ON-CHAIN SUPER SIGNALS™ environment setup.")

        LOGGER.info("Filesystem directories created/verified.")
        _initialize_db()

    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "FATAL: Environment setup failed.",
            extra={"error_type": type(exc).__name__, "details": str(exc)},
            exc_info=True,
        )
        # Returner feilkode 1 ved feil.
        return 1

    LOGGER.info(
        "ON-CHAIN SUPER SIGNALS™ environment setup completed successfully. "
        "System is ready for use."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
