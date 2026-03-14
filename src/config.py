"""Shared configuration for the OSE pipeline scripts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SCRUBBED_DIR = DATA_DIR / "scrubbed"
MANIFEST_DIR = SCRUBBED_DIR / "_manifests"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

ECCC_CACHE_DIR = RAW_DIR / "ECCC_Stanhope"

# HOBOlink data are manually dropped in these station folders.
HOBOLINK_STATIONS: List[str] = [
    "Cavendish",
    "Greenwich",
    "North Rustico Wharf",
    "Stanley Bridge Wharf",
    "Tracadie Wharf",
]

CANONICAL_VARIABLES: Dict[str, str] = {
    "temp": "air_temperature_c",
    "rh": "relative_humidity_pct",
    "wind_speed": "wind_speed_kmh",
    "wind_dir": "wind_direction_deg",
    "rain": "precipitation_mm",
}


def ensure_directories() -> None:
    """Create required runtime directories if they are missing."""
    required_dirs = [
        SCRUBBED_DIR,
        MANIFEST_DIR,
        OUTPUTS_DIR,
        FIGURES_DIR,
        ECCC_CACHE_DIR,
    ]
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)


def get_hobolink_dropzones() -> Dict[str, Path]:
    """Return expected HOBOlink drop-zone folders by station name."""
    return {station: RAW_DIR / station for station in HOBOLINK_STATIONS}


def setup_logging(script_name: str) -> logging.Logger:
    """Configure and return a standard logger for pipeline scripts."""
    logger = logging.getLogger(script_name)
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    return logger
