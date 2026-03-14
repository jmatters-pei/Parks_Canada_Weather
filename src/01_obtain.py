"""Step 1 skeleton for Obtain stage of OSE pipeline."""

from __future__ import annotations

from pathlib import Path

from config import ECCC_CACHE_DIR, get_hobolink_dropzones, ensure_directories, setup_logging


def main() -> int:
    """Initialize obtain-stage scaffolding and print a run summary."""
    logger = setup_logging("01_obtain")
    ensure_directories()

    hobolink_dropzones = get_hobolink_dropzones()
    missing_dropzones = [
        station_name
        for station_name, folder in hobolink_dropzones.items()
        if not Path(folder).exists()
    ]

    logger.info("Obtain scaffolding initialized.")
    logger.info("ECCC cache directory: %s", ECCC_CACHE_DIR)
    logger.info("HOBOlink station folders expected: %d", len(hobolink_dropzones))

    if missing_dropzones:
        logger.warning(
            "Missing HOBOlink drop-zone folders: %s",
            ", ".join(missing_dropzones),
        )
    else:
        logger.info("All HOBOlink drop-zone station folders are present.")

    logger.info("Summary: obtain skeleton ran successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
