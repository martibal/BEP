from __future__ import annotations

import sys
import time
from datetime import date

from src.services.data_service import DataService
from src.services.model_service import ModelService
from src.utils import db_handler, validators
from src.utils.logger import get_logger

LOGGER = get_logger("pipelines.04_cleanup")


def _is_monthly_vacuum_due() -> bool:
    """
    Determine whether a monthly VACUUM operation is due.

    Logic
    -----
    - Returns True if today's calendar day is the first of the month.
    - Otherwise returns False.

    The scheduler (cron) is assumed to run this script daily. By tying the
    VACUUM schedule to the first day of each month, we ensure a predictable
    and simple maintenance cadence without additional state.
    """
    today = date.today()
    return today.day == 1


def _run_full_cleanup(data_service: DataService) -> None:
    """
    Orchestrate the full cleanup and maintenance routine.

    Steps
    -----
    1. Log the start of the cleanup.
    2. Cleanup temporary DuckDB tables via db_handler.cleanup_temporary_tables().
       - This must drop any intermediate or temporary tables that may remain
         from previous failed or partial pipeline runs.
    3. Cleanup temporary files via DataService.cleanup_tmp(target_date=None).
       - This must delete all temporary files in TMP_DIR, regardless of date.
       - If cleanup fails, the error is logged and the function will still
         attempt to perform subsequent steps, but will ultimately raise an
         exception so that the pipeline exit code can indicate failure.
    4. Archive or clean up old model versions via ModelService.archive_old_models().
       - This prevents unbounded growth of model binaries in MODEL_DIR and
         implements the retention policy defined in the requirements.
    5. Ensure any orphaned pipeline lock files are removed via
       validators.force_release_pipeline_lock().
       - This is a safety net in case earlier pipeline stages crashed and
         did not release the lock properly.
    6. If a monthly VACUUM is due:
       - Log that VACUUM is starting.
       - Call db_handler.vacuum_database().
       - Log success or failure.
    7. If any of the steps (temporary table cleanup, file cleanup, model
       archival, lock cleanup, or VACUUM) failed, raise a RuntimeError so
       that the caller can return a non-zero exit code.

    Raises
    ------
    RuntimeError
        If any of the cleanup steps fail.
    """
    LOGGER.info("Starting full cleanup routine.")

    temp_table_cleanup_failed = False
    cleanup_failed = False
    model_archive_failed = False
    lock_cleanup_failed = False
    vacuum_failed = False

    # Step 1: Cleanup temporary DuckDB tables
    try:
        LOGGER.info("Starting DuckDB temporary tables cleanup via db_handler.cleanup_temporary_tables().")
        db_handler.cleanup_temporary_tables()
        LOGGER.info("DuckDB temporary tables cleanup completed successfully.")
    except Exception as exc:  # noqa: BLE001
        temp_table_cleanup_failed = True
        LOGGER.error("DuckDB temporary tables cleanup failed: %s", exc)

    # Step 2: Cleanup temporary files
    try:
        LOGGER.info("Starting temporary files cleanup via DataService.cleanup_tmp().")
        # target_date=None ensures all temporary files are cleaned up,
        # not just for a specific date.
        data_service.cleanup_tmp(target_date=None)
        LOGGER.info("Temporary files cleanup completed successfully.")
    except Exception as exc:  # noqa: BLE001
        cleanup_failed = True
        LOGGER.error("Temporary files cleanup failed: %s", exc)

    # Step 3: Archive or clean up old model versions
    model_service = ModelService()
    try:
        LOGGER.info("Starting model archival/cleanup via ModelService.archive_old_models().")
        model_service.archive_old_models()
        LOGGER.info("Model archival/cleanup completed successfully.")
    except Exception as exc:  # noqa: BLE001
        model_archive_failed = True
        LOGGER.error("Model archival/cleanup failed: %s", exc)

    # Step 4: Ensure any orphaned pipeline lock files are removed
    try:
        LOGGER.info("Ensuring pipeline lock is released (if present) via validators.force_release_pipeline_lock().")
        validators.force_release_pipeline_lock()
        LOGGER.info("Pipeline lock release completed successfully.")
    except Exception as exc:  # noqa: BLE001
        lock_cleanup_failed = True
        LOGGER.error("Pipeline lock release failed: %s", exc)

    # Step 5: Monthly VACUUM, if due
    if _is_monthly_vacuum_due():
        LOGGER.info("Monthly VACUUM is due, executing database optimization.")
        try:
            db_handler.vacuum_database()
            LOGGER.info("Database VACUUM completed successfully.")
        except Exception as exc:  # noqa: BLE001
            vacuum_failed = True
            LOGGER.error("Database VACUUM failed: %s", exc)
    else:
        LOGGER.debug("Monthly VACUUM not due today, skipping VACUUM step.")

    # Step 6: Evaluate overall success and raise if anything failed
    if any(
        (
            temp_table_cleanup_failed,
            cleanup_failed,
            model_archive_failed,
            lock_cleanup_failed,
            vacuum_failed,
        )
    ):
        # All failures are considered critical for the purposes of this script.
        # We raise a RuntimeError so that the caller (main) can return a non-zero
        # exit code for cron/monitoring.
        raise RuntimeError(
            "Full cleanup routine did not complete successfully "
            f"(temp_table_cleanup_failed={temp_table_cleanup_failed}, "
            f"cleanup_failed={cleanup_failed}, "
            f"model_archive_failed={model_archive_failed}, "
            f"lock_cleanup_failed={lock_cleanup_failed}, "
            f"vacuum_failed={vacuum_failed})."
        )

    LOGGER.info("Full cleanup routine completed successfully.")


def main() -> int:
    """
    Main entrypoint for the 04_cleanup pipeline script.

    Steps
    -----
    1. Record the start time.
    2. Log that the pipeline has started.
    3. Instantiate required services (DataService).
    4. Execute the full cleanup routine inside a try/except block.
    5. On success:
       - Log completion including total duration.
       - Return exit code 0.
    6. On failure:
       - Log the error and duration.
       - Return exit code 1.

    Returns
    -------
    int
        Exit code suitable for use with sys.exit():
        - 0 indicates success.
        - 1 indicates that an error occurred.
    """
    pipeline_start = time.time()
    LOGGER.info("Starting 04_cleanup pipeline.")

    data_service = DataService()

    try:
        _run_full_cleanup(data_service=data_service)
    except Exception as exc:  # noqa: BLE001
        duration_seconds = time.time() - pipeline_start
        LOGGER.error("04_cleanup pipeline failed: %s", exc)
        LOGGER.info(
            "04_cleanup pipeline finished with errors in %.2f seconds.",
            duration_seconds,
        )
        return 1

    duration_seconds = time.time() - pipeline_start
    LOGGER.info(
        "04_cleanup pipeline completed successfully in %.2f seconds.",
        duration_seconds,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
