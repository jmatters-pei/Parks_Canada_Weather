"""Step 1 skeleton for Explore stage of OSE pipeline."""

from __future__ import annotations

from config import FIGURES_DIR, OUTPUTS_DIR, ensure_directories, setup_logging


def main() -> int:
    """Initialize explore-stage scaffolding and print a run summary."""
    logger = setup_logging("03_explore")
    ensure_directories()

    logger.info("Explore scaffolding initialized.")
    logger.info("Outputs directory: %s", OUTPUTS_DIR)
    logger.info("Figures directory: %s", FIGURES_DIR)
    logger.info("Summary: explore skeleton ran successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
