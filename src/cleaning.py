from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import re
import time
import warnings
from tempfile import NamedTemporaryFile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

# -----------------------------------------------------------------------------
# Configuration constants
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SCRUBBED_DIR = DATA_DIR / "scrubbed"
MANIFEST_DIR = SCRUBBED_DIR / "_manifests"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
LOGS_DIR = OUTPUTS_DIR / "logs"

ECCC_CACHE_DIR = RAW_DIR / "ECCC_Stanhope"
ECCC_FWI_CACHE_DIR = RAW_DIR / "ECCC_Stanhope_FWI"
ECCC_STANHOPE_CLIMATE_ID = 8300590
ECCC_STANHOPE_NAME = "Stanhope"

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

HOBOLINK_MANIFEST = MANIFEST_DIR / "01_obtain_hobolink_files.csv"
ECCC_MANIFEST = MANIFEST_DIR / "01_obtain_eccc_periods.csv"
ECCC_FWI_MANIFEST = MANIFEST_DIR / "01_obtain_eccc_fwi_daily_periods.csv"
SCHEMA_INVENTORY = MANIFEST_DIR / "01_schema_inventory.csv"

OUTPUT_HOURLY = SCRUBBED_DIR / "02_hourly_weather_utc.csv"
OUTPUT_MISSINGNESS = SCRUBBED_DIR / "02_missingness_hourly_summary.csv"
OUTPUT_QC_COUNTS = SCRUBBED_DIR / "02_qc_out_of_range_counts.csv"
OUTPUT_PRECIP_LOG = SCRUBBED_DIR / "02_precip_semantics_log.csv"
SCRUB_RUNS_MANIFEST = MANIFEST_DIR / "02_scrub_runs.csv"

MANIFEST_COLUMNS = [
    "source",
    "station_raw",
    "station_slug",
    "year",
    "period",
    "file_name",
    "file_path",
    "size_bytes",
    "sha256",
    "file_mtime_ns",
    "ingested_at_utc",
    "status",
    "error_message",
    "schema_hash",
    "coverage_start_date",
    "coverage_end_date",
]

SCHEMA_COLUMNS = [
    "source",
    "schema_hash",
    "columns_json",
    "first_seen_utc",
    "last_seen_utc",
    "seen_count",
]

STATUS_OK = "ok"
STATUS_FAILED_READ = "failed_read"
STATUS_FAILED_DOWNLOAD = "failed_download"

READ_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")
NA_STRINGS = ["", "NA", "N/A", "na", "null", "NULL"]
HOURLY_FREQ = "h"

CSV_MIME_HINTS = (
    "text/csv",
    "application/csv",
    "application/octet-stream",
    "application/vnd.ms-excel",
    "application/force-download",
    "text/plain",
)

FWI_REQUIRED_COLUMNS = ("ffmc", "dmc", "dc", "isi", "bui", "fwi")
FWI_API_BASE = "https://api.weather.gc.ca/collections/climate-hourly/items"
FWI_PAGE_LIMIT = 500
FWI_REQUEST_DELAY_SECONDS = 1.0
FWI_REQUEST_TIMEOUT_SECONDS = 30
FWI_REQUEST_MAX_RETRIES = 3
FWI_REQUEST_RETRY_BACKOFF_SECONDS = 2.0
FWI_SEASON_START_MONTH = 6
FWI_SEASON_START_DAY = 1
FWI_SEASON_END_MONTH = 9
FWI_SEASON_END_DAY = 30
FFMC_START = 85.0
DMC_START = 6.0
DC_START = 15.0
FWI_CONTINUOUS_INTERPOLATION_LIMIT_DAYS = 2
FWI_PRECIP_ZERO_FILL_MAX_GAP_DAYS = 2

TEMP_BOUNDS = (-50.0, 50.0)
RH_MIN = 0.0
RH_CAP_MAX = 100.0
RH_SOFT_MAX = 105.0
WIND_SPEED_MIN = 0.0
WIND_DIR_BOUNDS = (0.0, 360.0)

STEP_THRESHOLDS = {
    CANONICAL_VARIABLES["temp"]: 15.0,
    CANONICAL_VARIABLES["rh"]: 30.0,
    CANONICAL_VARIABLES["wind_speed"]: 60.0,
}

CONTINUOUS_FILL_VARS = [
    CANONICAL_VARIABLES["temp"],
    CANONICAL_VARIABLES["rh"],
    CANONICAL_VARIABLES["wind_speed"],
]

CANONICAL_ORDER = [
    CANONICAL_VARIABLES["temp"],
    CANONICAL_VARIABLES["rh"],
    CANONICAL_VARIABLES["wind_speed"],
    CANONICAL_VARIABLES["wind_dir"],
    CANONICAL_VARIABLES["rain"],
]

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def utc_now_iso() -> str:
    """Return current UTC timestamp in stable ISO format for manifests/logs."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def station_slug(station_name: str) -> str:
    """Convert a human-readable station name into a filesystem-safe identifier."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(station_name).strip())
    return slug.strip("_")


def setup_logging(log_path: Path) -> logging.Logger:
    """Configure one shared logger that writes to console and a daily log file."""
    logger = logging.getLogger("cleaning")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def ensure_directories(raw_dir: Path, scrubbed_dir: Path, outputs_dir: Path) -> None:
    """Create required output folders so downstream writes never fail on missing paths."""
    (scrubbed_dir / "_manifests").mkdir(parents=True, exist_ok=True)
    (outputs_dir / "logs").mkdir(parents=True, exist_ok=True)
    (outputs_dir / "figures").mkdir(parents=True, exist_ok=True)
    (outputs_dir / "tables").mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)


def compute_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute a content hash used to fingerprint source files in manifests."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def response_looks_like_csv(response: requests.Response, csv_mime_hints: tuple[str, ...]) -> bool:
    """Reject HTML/error pages and keep only responses that look structurally like CSV."""
    content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
    if "text/html" in content_type:
        return False
    if content_type and content_type not in csv_mime_hints:
        # Keep a content sniff fallback for non-standard download MIME types.
        pass

    probe = response.text.lstrip("\ufeff\n\r\t ")
    probe_head = probe[:500]
    if "<!doctype html" in probe_head.lower() or "<html" in probe_head.lower():
        return False

    first_line = probe_head.splitlines()[0] if probe_head.splitlines() else ""
    if "," not in first_line:
        return False

    csv_markers = (
        "date/time",
        "station name",
        "year",
        "month",
        "day",
    )
    marker_hits = sum(marker in probe_head.lower() for marker in csv_markers)
    return marker_hits >= 2


def download_to_file_atomic(
    url: str,
    destination: Path,
    csv_mime_hints: tuple[str, ...],
    timeout_seconds: int = 60,
) -> None:
    """Download text payload and atomically replace destination to avoid partial files."""
    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()

    if not response_looks_like_csv(response, csv_mime_hints):
        raise ValueError("Response did not look like CSV payload.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", delete=False, dir=destination.parent, encoding="utf-8", newline="") as tmp:
        tmp.write(response.text)
        temp_path = Path(tmp.name)

    temp_path.replace(destination)


def get_eccc_download_mode(climate_id: int, probe_year: int, logger: logging.Logger) -> str:
    """Probe ECCC endpoint capabilities (currently informational; monthly flow is used)."""
    annual_url = (
        "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"
        f"?format=csv&climate_id={climate_id}&Year={probe_year}&timeframe=1"
    )
    try:
        response = requests.get(annual_url, timeout=30)
        if response.status_code == 200 and response_looks_like_csv(response, CSV_MIME_HINTS):
            logger.info("ECCC endpoint supports annual hourly download mode.")
            return "annual"
    except Exception as exc:  # noqa: BLE001
        logger.warning("Annual ECCC probe failed; falling back to monthly mode: %s", exc)

    logger.info("ECCC endpoint will use monthly hourly download mode.")
    return "monthly"


def previous_month(year: int, month: int) -> Tuple[int, int]:
    """Return the previous calendar month as (year, month)."""
    if month == 1:
        return year - 1, 12
    return year, month - 1


def build_required_month_periods(start_year: int, end_year: int) -> List[Tuple[int, int]]:
    """Build month periods between start/end years, capped at the current UTC month."""
    now_utc = datetime.now(timezone.utc)
    capped_end_year = min(end_year, now_utc.year)
    periods: List[Tuple[int, int]] = []
    for year in range(start_year, capped_end_year + 1):
        max_month = 12
        if year == now_utc.year:
            max_month = now_utc.month
        for month in range(1, max_month + 1):
            periods.append((year, month))
    return periods


def recent_refresh_periods() -> set[Tuple[int, int]]:
    """Return current and previous month so recent ECCC data can be refreshed."""
    now_utc = datetime.now(timezone.utc)
    current = (now_utc.year, now_utc.month)
    previous = previous_month(now_utc.year, now_utc.month)
    return {current, previous}


def build_eccc_url(climate_id: int, year: int, month: Optional[int]) -> str:
    """Create an ECCC bulk CSV URL for hourly data at monthly resolution."""
    base = (
        "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"
        f"?format=csv&climate_id={climate_id}&Year={year}&timeframe=1"
    )
    if month is None:
        return base
    return f"{base}&Month={month}&Day=1"


def download_eccc_periods(
    *,
    station_name: str,
    climate_id: int,
    start_year: int,
    end_year: int,
    logger: logging.Logger,
    dry_run: bool,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Download/refresh Stanhope hourly ECCC CSVs and return manifest + schema rows."""
    manifest_rows: List[Dict[str, object]] = []
    schema_rows: List[Dict[str, object]] = []

    # Force monthly management so we can refresh only the most recent two months.
    get_eccc_download_mode(climate_id, end_year, logger)
    periods = build_required_month_periods(start_year, end_year)
    refresh_set = recent_refresh_periods()

    for year, month in periods:
        period = f"{year}-{month:02d}"
        file_name = f"ECCC_{station_slug(station_name)}_{period}_hourly.csv"
        destination = ECCC_CACHE_DIR / file_name
        should_refresh = (year, month) in refresh_set

        if destination.exists() and not should_refresh:
            # Reuse historical files outside the rolling refresh window.
            size_bytes = destination.stat().st_size
            sha256_value = compute_sha256(destination)
            schema_status, error_message, schema_hash, columns = inspect_csv_schema(destination, "eccc")
            final_status = STATUS_OK if schema_status == STATUS_OK else STATUS_FAILED_READ
        else:
            if dry_run:
                dry_msg = "dry-run skipped ECCC internet fetch"
                if should_refresh and destination.exists():
                    dry_msg = "dry-run skipped ECCC refresh for recent month"
                manifest_rows.append(
                    create_manifest_row(
                        source="eccc",
                        station_raw=station_name,
                        year=year,
                        period=period,
                        file_path=destination,
                        size_bytes=0,
                        sha256_value="",
                        status=STATUS_FAILED_DOWNLOAD,
                        error_message=dry_msg,
                        schema_hash=None,
                    )
                )
                continue

            url = build_eccc_url(climate_id=climate_id, year=year, month=month)
            try:
                # Download first, then inspect schema before marking the file as usable.
                download_to_file_atomic(url=url, destination=destination, csv_mime_hints=CSV_MIME_HINTS)
                size_bytes = destination.stat().st_size
                sha256_value = compute_sha256(destination)
                schema_status, error_message, schema_hash, columns = inspect_csv_schema(destination, "eccc")
                final_status = STATUS_OK if schema_status == STATUS_OK else STATUS_FAILED_READ
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed ECCC download for period %s: %s", period, exc)
                manifest_rows.append(
                    create_manifest_row(
                        source="eccc",
                        station_raw=station_name,
                        year=year,
                        period=period,
                        file_path=destination,
                        size_bytes=0,
                        sha256_value="",
                        status=STATUS_FAILED_DOWNLOAD,
                        error_message=str(exc),
                        schema_hash=None,
                    )
                )
                continue

        if schema_hash and columns:
            schema_rows.append(
                {
                    "source": "eccc",
                    "schema_hash": schema_hash,
                    "columns_json": json.dumps(columns, ensure_ascii=True),
                }
            )

        manifest_rows.append(
            create_manifest_row(
                source="eccc",
                station_raw=station_name,
                year=year,
                period=period,
                file_path=destination,
                size_bytes=size_bytes,
                sha256_value=sha256_value,
                status=final_status,
                error_message=error_message,
                schema_hash=schema_hash,
            )
        )

    return manifest_rows, schema_rows


def schema_hash_from_columns(columns: List[str]) -> str:
    """Generate a deterministic hash for a column layout to track schema drift."""
    payload = json.dumps({"columns": columns}, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_scanned_rows(csv_path: Path, max_scan_lines: int = 100) -> Tuple[List[List[str]], str]:
    """Read a small CSV sample with fallback encodings for robust header detection."""
    last_error: Optional[Exception] = None
    for encoding in READ_ENCODINGS:
        try:
            with csv_path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.reader(handle)
                rows: List[List[str]] = []
                for i, row in enumerate(reader):
                    rows.append(row)
                    if i >= max_scan_lines:
                        break
            return rows, encoding
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise ValueError("Unable to scan csv rows.")


def _normalize_header_columns(raw_columns: List[str]) -> List[str]:
    """Trim blank/duplicate header names and make duplicate names explicit."""
    cleaned: List[str] = []
    for i, column in enumerate(raw_columns, start=1):
        text = str(column).strip()
        if not text:
            text = f"unnamed_{i}"
        cleaned.append(text)

    seen: Dict[str, int] = {}
    out: List[str] = []
    for column in cleaned:
        seen[column] = seen.get(column, 0) + 1
        if seen[column] == 1:
            out.append(column)
        else:
            out.append(f"{column}__dup{seen[column] - 1}")
    return out


def detect_header_and_columns(csv_path: Path, max_scan_lines: int = 100) -> Tuple[int, List[str]]:
    """Detect likely header row and return normalized column names."""
    rows, _ = _read_scanned_rows(csv_path, max_scan_lines=max_scan_lines)

    candidate_row: Optional[int] = None
    candidate_columns: List[str] = []
    for idx, row in enumerate(rows):
        stripped = [cell.strip() for cell in row]
        non_empty = [cell for cell in stripped if cell]
        if len(non_empty) < 2:
            continue
        joined = " ".join(non_empty).lower()
        if "date" in joined and "time" in joined:
            candidate_row = idx
            candidate_columns = stripped
            break

    if candidate_row is None:
        best_score = -1
        for idx, row in enumerate(rows):
            stripped = [cell.strip() for cell in row]
            score = len([cell for cell in stripped if cell])
            if score >= 4 and score > best_score:
                best_score = score
                candidate_row = idx
                candidate_columns = stripped

    if candidate_row is None:
        raise ValueError("Could not detect a valid header row.")

    columns = _normalize_header_columns(candidate_columns)
    if not columns:
        raise ValueError("Detected header row but no column names were found.")
    return candidate_row, columns


def validate_timestamp_columns(source: str, columns: List[str]) -> None:
    """Enforce timestamp column requirements per source system."""
    lower_columns = {column.strip().lower() for column in columns}
    if source == "hobolink":
        has_split = "date" in lower_columns and "time" in lower_columns
        has_combined = any(token in lower_columns for token in ("date/time", "date time", "datetime", "timestamp"))
        if not has_split and not has_combined:
            raise ValueError("HOBOlink file is missing required Date and/or Time columns.")
        return
    if source == "eccc_fwi_daily":
        has_date_like = any("date" in column for column in lower_columns)
        if not has_date_like:
            raise ValueError("FWI daily file is missing a required date-like column.")
        return
    has_datetime = any("date/time" in col for col in lower_columns)
    has_date_and_time = "date" in lower_columns and "time" in lower_columns
    if not has_datetime and not has_date_and_time:
        raise ValueError("ECCC file is missing a Date/Time-equivalent timestamp column.")


def validate_required_fwi_columns(columns: List[str]) -> None:
    """Verify expected daily FWI code fields are present before accepting the file."""
    normalized = [re.sub(r"[^a-z0-9]+", "", column.lower()) for column in columns]
    has_date = any("date" in column for column in normalized)
    if not has_date:
        raise ValueError("FWI daily file is missing required date column.")

    missing = [
        code.upper()
        for code in FWI_REQUIRED_COLUMNS
        if not any(column == code or column.startswith(code) for column in normalized)
    ]
    if missing:
        raise ValueError("FWI daily file missing required FWI code columns: " + ", ".join(missing))


def inspect_csv_schema(csv_path: Path, source: str) -> Tuple[str, Optional[str], Optional[str], Optional[List[str]]]:
    """Run lightweight schema checks and return status, errors, and schema fingerprint."""
    if not csv_path.exists() or csv_path.stat().st_size <= 0:
        return STATUS_FAILED_READ, "File does not exist or is empty.", None, None
    try:
        _, columns = detect_header_and_columns(csv_path)
        validate_timestamp_columns(source, columns)
        if source == "eccc_fwi_daily":
            validate_required_fwi_columns(columns)
        return STATUS_OK, None, schema_hash_from_columns(columns), columns
    except Exception as exc:  # noqa: BLE001
        return STATUS_FAILED_READ, str(exc), None, None


def create_manifest_row(
    *,
    source: str,
    station_raw: str,
    year: Optional[int],
    period: str,
    file_path: Path,
    size_bytes: int,
    sha256_value: str,
    file_mtime_ns: Optional[int] = None,
    status: str,
    error_message: Optional[str],
    schema_hash: Optional[str],
    coverage_start_date: Optional[str] = None,
    coverage_end_date: Optional[str] = None,
) -> Dict[str, object]:
    """Build one standard manifest record used by obtain/scrub lineage tracking."""
    return {
        "source": source,
        "station_raw": station_raw,
        "station_slug": station_slug(station_raw),
        "year": int(year) if year is not None else pd.NA,
        "period": period,
        "file_name": file_path.name,
        "file_path": str(file_path),
        "size_bytes": size_bytes,
        "sha256": sha256_value,
        "file_mtime_ns": int(file_mtime_ns) if file_mtime_ns is not None else pd.NA,
        "ingested_at_utc": utc_now_iso(),
        "status": status,
        "error_message": error_message,
        "schema_hash": schema_hash,
        "coverage_start_date": coverage_start_date,
        "coverage_end_date": coverage_end_date,
    }


def extract_year(text: str) -> Optional[int]:
    """Extract the first four-digit year from text if present."""
    match = re.search(r"(19|20)\d{2}", text)
    if match:
        return int(match.group())
    return None


def status_is_usable(status: object) -> bool:
    """Treat any non-failed status as eligible for downstream processing."""
    if pd.isna(status):
        return False
    text = str(status).strip().lower()
    if not text:
        return False
    return not text.startswith("failed_")


def find_hobolink_datetime_columns(columns: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Locate HOBOlink datetime fields (combined or split date/time columns)."""
    lowered = {column.lower(): column for column in columns}

    combined_candidates = ["date/time", "date time", "datetime", "timestamp"]
    for candidate in combined_candidates:
        if candidate in lowered:
            return lowered[candidate], None, None

    date_col = lowered.get("date")
    time_col = lowered.get("time")
    if date_col and time_col:
        return None, date_col, time_col

    return None, None, None


def read_hobolink_datetime_bounds(csv_path: Path) -> Optional[Tuple[date, date]]:
    """Read only datetime bounds from a HOBOlink file for coverage planning."""
    header_row, columns = detect_header_and_columns(csv_path)
    datetime_col, date_col, time_col = find_hobolink_datetime_columns(columns)
    if datetime_col is None and (date_col is None or time_col is None):
        return None

    last_error: Optional[Exception] = None
    for encoding in READ_ENCODINGS:
        try:
            df = pd.read_csv(csv_path, skiprows=header_row, encoding=encoding, engine="python")
            df.columns = _normalize_header_columns([str(column) for column in df.columns])

            if datetime_col and datetime_col in df.columns:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Could not infer format, so each element will be parsed individually",
                        category=UserWarning,
                    )
                    parsed = pd.to_datetime(df[datetime_col], errors="coerce", utc=True)
            elif date_col and time_col and date_col in df.columns and time_col in df.columns:
                combined = df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip()
                parsed = pd.to_datetime(combined, errors="coerce", utc=True, format="mixed")
            else:
                return None

            parsed = parsed.dropna()
            if parsed.empty:
                return None

            return parsed.min().date(), parsed.max().date()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    return None


def derive_hobolink_coverage_bounds(
    hobolink_rows: List[Dict[str, object]], logger: logging.Logger
) -> Optional[Tuple[date, date]]:
    """Estimate global HOBOlink start/end dates across all usable files."""
    cached_starts: List[date] = []
    cached_ends: List[date] = []
    for row in hobolink_rows:
        if not status_is_usable(row.get("status")):
            continue
        start_text = str(row.get("coverage_start_date", "")).strip()
        end_text = str(row.get("coverage_end_date", "")).strip()
        if not start_text or not end_text:
            continue
        try:
            cached_starts.append(date.fromisoformat(start_text))
            cached_ends.append(date.fromisoformat(end_text))
        except ValueError:
            continue

    if cached_starts and cached_ends:
        min_seen = min(cached_starts)
        max_seen = max(cached_ends)
        logger.info(
            "HOBOlink coverage bounds derived from manifest cache (%s files): %s to %s",
            len(cached_starts),
            min_seen,
            max_seen,
        )
        return min_seen, max_seen

    min_seen: Optional[date] = None
    max_seen: Optional[date] = None
    scanned_files = 0

    for row in hobolink_rows:
        if not status_is_usable(row.get("status")):
            continue
        file_path_text = str(row.get("file_path", "")).strip()
        if not file_path_text:
            continue
        csv_path = Path(file_path_text)
        if not csv_path.exists() or csv_path.suffix.lower() != ".csv":
            continue

        try:
            bounds = read_hobolink_datetime_bounds(csv_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping HOBOlink bounds parse failure for %s: %s", csv_path, exc)
            continue

        if bounds is None:
            continue

        scanned_files += 1
        file_min, file_max = bounds
        if min_seen is None or file_min < min_seen:
            min_seen = file_min
        if max_seen is None or file_max > max_seen:
            max_seen = file_max

    if min_seen is None or max_seen is None:
        logger.warning("Unable to derive HOBOlink coverage bounds from usable files.")
        return None

    logger.info(
        "HOBOlink coverage bounds derived from %s files: %s to %s",
        scanned_files,
        min_seen,
        max_seen,
    )
    return min_seen, max_seen


# -----------------------------------------------------------------------------
# FWI scientific formulas and data extraction helpers
# -----------------------------------------------------------------------------

def rh_from_dewpoint(temperature_c: float, dew_point_c: float) -> float:
    """Estimate relative humidity from temperature and dew point when RH is absent."""
    return 100 * math.exp(
        (17.625 * dew_point_c / (243.04 + dew_point_c))
        - (17.625 * temperature_c / (243.04 + temperature_c))
    )


def ffmc_code(temp_c: float, rh_pct: float, wind_kmh: float, rain_mm: float, ffmc_prev: float) -> float:
    """Compute the Fine Fuel Moisture Code (FFMC) for one day."""
    mo = 147.2 * (101 - ffmc_prev) / (59.5 + ffmc_prev)
    if rain_mm > 0.5:
        rf = rain_mm - 0.5
        if mo > 150:
            mo = (
                mo
                + 42.5 * rf * math.exp(-100 / (251 - mo)) * (1 - math.exp(-6.93 / rf))
                + 0.0015 * (mo - 150) ** 2 * rf ** 0.5
            )
        else:
            mo = mo + 42.5 * rf * math.exp(-100 / (251 - mo)) * (1 - math.exp(-6.93 / rf))
        if mo > 250:
            mo = 250

    ed = 0.942 * rh_pct**0.679 + 11 * math.exp((rh_pct - 100) / 10) + 0.18 * (21.1 - temp_c) * (1 - math.exp(-0.115 * rh_pct))
    ew = 0.618 * rh_pct**0.753 + 10 * math.exp((rh_pct - 100) / 10) + 0.18 * (21.1 - temp_c) * (1 - math.exp(-0.115 * rh_pct))
    if mo > ed:
        ko = 0.424 * (1 - (rh_pct / 100) ** 1.7) + 0.0694 * wind_kmh**0.5 * (1 - (rh_pct / 100) ** 8)
        kd = ko * 0.581 * math.exp(0.0365 * temp_c)
        m = ed + (mo - ed) * 10 ** (-kd)
    elif mo < ew:
        ko = 0.424 * (1 - ((100 - rh_pct) / 100) ** 1.7) + 0.0694 * wind_kmh**0.5 * (1 - ((100 - rh_pct) / 100) ** 8)
        kw = ko * 0.581 * math.exp(0.0365 * temp_c)
        m = ew - (ew - mo) * 10 ** (-kw)
    else:
        m = mo

    return 59.5 * (250 - m) / (147.2 + m)


def dmc_code(temp_c: float, rh_pct: float, rain_mm: float, dmc_prev: float, month: int) -> float:
    """Compute the Duff Moisture Code (DMC) for one day."""
    day_length_factors = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.8, 6.3]
    if temp_c < -1.1:
        temp_c = -1.1
    rk = 1.894 * (temp_c + 1.1) * (100 - rh_pct) * day_length_factors[month - 1] * 1e-4

    if rain_mm > 1.5:
        rw = 0.92 * rain_mm - 1.27
        if rw <= 0:
            return dmc_prev + rk

        wmi = 20 + 280 / math.exp(0.023 * dmc_prev)
        if dmc_prev <= 33:
            b = 100 / (0.5 + 0.3 * dmc_prev)
        elif dmc_prev <= 65:
            b = 14 - 1.3 * math.log(dmc_prev)
        else:
            b = 6.2 * math.log(dmc_prev) - 17.2
        wmr = wmi + 1000 * rw / (48.77 + b * rw)
        if wmr <= 20:
            pr = 0.0
        else:
            pr = 43.43 * (5.6348 - math.log(wmr - 20))
    else:
        pr = dmc_prev

    if pr < 0:
        pr = 0
    return pr + rk


def dc_code(temp_c: float, rain_mm: float, dc_prev: float, month: int) -> float:
    """Compute the Drought Code (DC) for one day."""
    seasonal_factors = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]
    if temp_c < -2.8:
        temp_c = -2.8
    pe = (0.36 * (temp_c + 2.8) + seasonal_factors[month - 1]) / 2
    if pe < 0:
        pe = 0

    if rain_mm > 2.8:
        rw = 0.83 * rain_mm - 1.27
        if rw <= 0:
            return dc_prev + pe

        smi = 800 * math.exp(-dc_prev / 400)
        log_arg = 1 + 3.937 * rw / smi
        if log_arg <= 0:
            dr = 0
        else:
            dr = dc_prev - 400 * math.log(log_arg)
        if dr < 0:
            dr = 0
    else:
        dr = dc_prev

    return dr + pe


def isi_index(wind_kmh: float, ffmc_value: float) -> float:
    """Compute Initial Spread Index (ISI) from wind and FFMC."""
    fuel_moisture = 147.2 * (101 - ffmc_value) / (59.5 + ffmc_value)
    spread_factor = 19.115 * math.exp(-0.1386 * fuel_moisture) * (1 + fuel_moisture**5.31 / 4.93e7)
    return spread_factor * math.exp(0.05039 * wind_kmh)


def bui_index(dmc_value: float, dc_value: float) -> float:
    """Compute Build-Up Index (BUI) from DMC and DC."""
    denominator = dmc_value + 0.4 * dc_value
    if denominator <= 0:
        return 0.0
    if dmc_value <= 0.4 * dc_value:
        return 0.8 * dmc_value * dc_value / denominator
    bui_value = dmc_value - ((1 - 0.8 * dc_value / denominator) * (0.92 + (0.0114 * dmc_value) ** 1.7))
    return max(bui_value, 0)


def fwi_index(isi_value: float, bui_value: float) -> float:
    """Compute final Fire Weather Index (FWI) from ISI and BUI."""
    if bui_value <= 80:
        bb = 0.1 * isi_value * (0.626 * bui_value**0.809 + 2)
    else:
        bb = 0.1 * isi_value * (1000 / (25 + 108.64 * math.exp(-0.023 * bui_value)))
    if bb <= 1:
        return bb
    return math.exp(2.72 * (0.434 * math.log(bb)) ** 0.647)


def fetch_hourly_range(climate_id: int | str, start_dt: str, end_dt: str, logger: logging.Logger) -> List[Dict[str, object]]:
    """Fetch paged hourly records from ECCC API with retry and progress logging."""
    records: List[Dict[str, object]] = []
    offset = 0
    expected_total: Optional[int] = None

    while True:
        params = {
            "CLIMATE_IDENTIFIER": str(climate_id),
            "datetime": f"{start_dt}/{end_dt}",
            "f": "json",
            "limit": FWI_PAGE_LIMIT,
            "offset": offset,
            "sortby": "LOCAL_DATE",
        }
        payload: Dict[str, object] = {}
        for attempt in range(1, FWI_REQUEST_MAX_RETRIES + 1):
            try:
                response = requests.get(
                    FWI_API_BASE,
                    params=params,
                    timeout=FWI_REQUEST_TIMEOUT_SECONDS,
                    headers={"User-Agent": "fwi-research/1.0"},
                )
                response.raise_for_status()
                payload = response.json()
                break
            except requests.RequestException as exc:
                if attempt >= FWI_REQUEST_MAX_RETRIES:
                    raise RuntimeError(
                        f"Stanhope FWI API request failed at offset {offset} after {FWI_REQUEST_MAX_RETRIES} attempts: {exc}"
                    ) from exc

                sleep_seconds = FWI_REQUEST_RETRY_BACKOFF_SECONDS * attempt
                logger.warning(
                    "Stanhope FWI API request retry %s/%s at offset %s after error: %s",
                    attempt,
                    FWI_REQUEST_MAX_RETRIES,
                    offset,
                    exc,
                )
                time.sleep(sleep_seconds)

        features = payload.get("features", [])
        if not isinstance(features, list):
            raise RuntimeError(
                f"Stanhope FWI API returned unexpected payload type for features at offset {offset}."
            )

        if expected_total is None:
            number_matched = payload.get("numberMatched")
            if number_matched is not None:
                expected_total = int(number_matched)

        records.extend(features)
        total = expected_total if expected_total is not None else len(records)
        logger.info("Stanhope FWI fetch progress: %s/%s records", len(records), total)

        if len(records) >= total or not features:
            break

        offset += FWI_PAGE_LIMIT
        if FWI_REQUEST_DELAY_SECONDS > 0:
            time.sleep(FWI_REQUEST_DELAY_SECONDS)

    if expected_total is not None and len(records) < expected_total:
        raise RuntimeError(
            "Stanhope FWI API paging completed with incomplete record count: "
            f"expected {expected_total}, got {len(records)}."
        )

    return records


def parse_local_datetime(local_date_text: str) -> Optional[datetime]:
    """Parse LOCAL_DATE text from API payloads into Python datetime."""
    text = str(local_date_text).strip()
    if not text:
        return None
    for token in (" ", "T"):
        candidate = text.replace(" ", token)
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            continue
    return None


def extract_daily_fwi_inputs(records: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    """Construct noon-based daily FWI inputs from hourly API records."""
    by_dt: Dict[datetime, Dict[str, object]] = {}
    for feature in records:
        properties = feature.get("properties")
        if not isinstance(properties, dict):
            continue
        dt = parse_local_datetime(properties.get("LOCAL_DATE", ""))
        if dt is None:
            continue
        by_dt[dt] = properties

    if not by_dt:
        return {}

    dates = sorted({stamp.date() for stamp in by_dt})
    daily: Dict[str, Dict[str, float]] = {}

    for day in dates:
        noon_properties: Optional[Dict[str, object]] = None
        for hour in (12, 11, 13):
            # Prefer strict noon, but accept +/- 1 hour fallback if noon is missing.
            candidate = datetime(day.year, day.month, day.day, hour)
            if candidate in by_dt:
                noon_properties = by_dt[candidate]
                break
        if noon_properties is None:
            continue

        temp = noon_properties.get("TEMP")
        wind = noon_properties.get("WIND_SPEED")
        rh = noon_properties.get("RELATIVE_HUMIDITY")
        dew_point = noon_properties.get("DEW_POINT_TEMP")

        if temp is None or wind is None:
            continue
        if rh is None and dew_point is not None:
            rh = rh_from_dewpoint(float(temp), float(dew_point))
        if rh is None:
            continue

        temp_value = float(temp)
        rh_value = max(1.0, min(100.0, float(rh)))
        wind_value = float(wind)

        precip_total = 0.0
        noon_stamp = datetime(day.year, day.month, day.day, 12)
        for offset_hours in range(1, 25):
            stamp = noon_stamp - timedelta(hours=24 - offset_hours)
            hourly = by_dt.get(stamp)
            if not hourly:
                continue
            precip = hourly.get("PRECIP_AMOUNT")
            if precip is None:
                continue
            precip_total += float(precip)

        daily[str(day)] = {"t": temp_value, "h": rh_value, "w": wind_value, "p": precip_total}

    return daily


def build_fwi_daily_driver_table(
    daily_inputs: Dict[str, Dict[str, float]],
    *,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Build a continuous daily FWI driver table with bounded imputations and flags."""
    full_index = pd.date_range(start=start_date, end=end_date, freq="D")
    base = pd.DataFrame({"date": full_index})

    if daily_inputs:
        observed_rows = []
        for date_text, values in daily_inputs.items():
            observed_rows.append(
                {
                    "date": pd.Timestamp(date.fromisoformat(date_text)),
                    "t": float(values["t"]),
                    "h": float(values["h"]),
                    "w": float(values["w"]),
                    "p": float(values["p"]),
                }
            )
        observed_df = pd.DataFrame(observed_rows)
    else:
        observed_df = pd.DataFrame(columns=["date", "t", "h", "w", "p"])

    driver = base.merge(observed_df, on="date", how="left")

    for variable in ("t", "h", "w", "p"):
        driver[f"{variable}_observed"] = driver[variable].notna()

    # Fill only short interior gaps to preserve long-outage visibility.
    for variable in ("t", "h", "w"):
        driver[variable] = pd.to_numeric(driver[variable], errors="coerce").interpolate(
            method="linear",
            limit=FWI_CONTINUOUS_INTERPOLATION_LIMIT_DAYS,
            limit_area="inside",
        )

    precip_missing_before = driver["p"].isna()
    precip_fill_mask = short_gap_mask(driver["p"], max_gap=FWI_PRECIP_ZERO_FILL_MAX_GAP_DAYS) & precip_missing_before
    driver.loc[precip_fill_mask, "p"] = 0.0

    linear_imputed = pd.Series(False, index=driver.index, dtype="boolean")
    for variable in ("t", "h", "w"):
        linear_imputed = linear_imputed | (~driver[f"{variable}_observed"] & driver[variable].notna())
    driver["imputed_day"] = (linear_imputed | precip_fill_mask).astype("boolean")

    driver["imputation_method"] = "none"
    driver.loc[linear_imputed, "imputation_method"] = "linear"
    driver.loc[precip_fill_mask, "imputation_method"] = "zero_fill"
    driver.loc[linear_imputed & precip_fill_mask, "imputation_method"] = "linear+zero_fill"

    driver["unresolved_gap"] = driver[["t", "h", "w", "p"]].isna().any(axis=1)
    driver.loc[driver["unresolved_gap"], "imputation_method"] = "unresolved_gap"

    return driver


def compute_fwi_daily_records(
    daily_inputs: Dict[str, Dict[str, float]],
    *,
    start_date: date,
    end_date: date,
) -> List[Dict[str, object]]:
    """Run sequential daily FWI calculations over a continuous daily driver timeline."""
    ffmc_value = FFMC_START
    dmc_value = DMC_START
    dc_value = DC_START
    results: List[Dict[str, object]] = []
    driver = build_fwi_daily_driver_table(daily_inputs, start_date=start_date, end_date=end_date)
    season_invalid = False

    for _, row in driver.iterrows():
        current_day = row["date"].date()
        date_text = current_day.isoformat()
        season_end = date(current_day.year, FWI_SEASON_END_MONTH, FWI_SEASON_END_DAY)
        if current_day > season_end:
            continue

        if bool(row["unresolved_gap"]):
            season_invalid = True

        if season_invalid:
            results.append(
                {
                    "Date": date_text,
                    "T_noon": round(float(row["t"]), 1) if pd.notna(row["t"]) else pd.NA,
                    "RH_noon": round(float(row["h"]), 0) if pd.notna(row["h"]) else pd.NA,
                    "Wind_noon": round(float(row["w"]), 0) if pd.notna(row["w"]) else pd.NA,
                    "Precip_24h": round(float(row["p"]), 1) if pd.notna(row["p"]) else pd.NA,
                    "FFMC": pd.NA,
                    "DMC": pd.NA,
                    "DC": pd.NA,
                    "ISI": pd.NA,
                    "BUI": pd.NA,
                    "FWI": pd.NA,
                    "fwi_valid": False,
                    "imputed_day": bool(row["imputed_day"]),
                    "imputation_method": str(row["imputation_method"]),
                }
            )
            continue

        values = {"t": float(row["t"]), "h": float(row["h"]), "w": float(row["w"]), "p": float(row["p"])}
        month = int(date_text[5:7])

        ffmc_value = ffmc_code(values["t"], values["h"], values["w"], values["p"], ffmc_value)
        dmc_value = dmc_code(values["t"], values["h"], values["p"], dmc_value, month)
        dc_value = dc_code(values["t"], values["p"], dc_value, month)

        isi_value = isi_index(values["w"], ffmc_value)
        bui_value = bui_index(dmc_value, dc_value)
        fwi_value = fwi_index(isi_value, bui_value)

        results.append(
            {
                "Date": date_text,
                "T_noon": round(values["t"], 1),
                "RH_noon": round(values["h"], 0),
                "Wind_noon": round(values["w"], 0),
                "Precip_24h": round(values["p"], 1),
                "FFMC": round(ffmc_value, 2),
                "DMC": round(dmc_value, 2),
                "DC": round(dc_value, 2),
                "ISI": round(isi_value, 2),
                "BUI": round(bui_value, 2),
                "FWI": round(fwi_value, 2),
                "fwi_valid": True,
                "imputed_day": bool(row["imputed_day"]),
                "imputation_method": str(row["imputation_method"]),
            }
        )

    return results


def get_fwi_date_bounds(csv_path: Path) -> Optional[Tuple[date, date]]:
    """Get first/last dates from a precomputed daily FWI CSV."""
    try:
        frame = pd.read_csv(csv_path, usecols=["Date"])
    except Exception:  # noqa: BLE001
        return None

    if frame.empty:
        return None

    parsed = pd.to_datetime(frame["Date"], format="%Y-%m-%d", errors="coerce")
    parsed = parsed.dropna()
    if parsed.empty:
        return None

    return parsed.min().date(), parsed.max().date()


def compute_stanhope_fwi_daily_file(
    *,
    station_name: str,
    climate_id: int | str,
    year: int,
    start_date: date,
    end_date: date,
    destination: Path,
    logger: logging.Logger,
) -> Tuple[str, int, str, Optional[str], Optional[str], Optional[List[str]]]:
    """Compute one year of Stanhope daily FWI and save to CSV with schema checks."""
    start_dt = f"{start_date:%Y-%m-%d}T00:00:00"
    end_dt = f"{end_date:%Y-%m-%d}T23:59:59"

    logger.info("Computing Stanhope daily FWI for %s (%s to %s).", year, start_dt[:10], end_dt[:10])

    try:
        records = fetch_hourly_range(climate_id=climate_id, start_dt=start_dt, end_dt=end_dt, logger=logger)
    except Exception as exc:  # noqa: BLE001
        return STATUS_FAILED_DOWNLOAD, 0, "", f"FWI hourly API fetch failed: {exc}", None, None

    if not records:
        return STATUS_FAILED_DOWNLOAD, 0, "", "No hourly API records returned for requested period.", None, None

    daily_inputs = extract_daily_fwi_inputs(records)
    if not daily_inputs:
        return STATUS_FAILED_READ, 0, "", "No complete daily noon inputs found to compute FWI.", None, None

    results = compute_fwi_daily_records(
        daily_inputs,
        start_date=start_date,
        end_date=end_date,
    )
    if not results:
        return STATUS_FAILED_READ, 0, "", "FWI computation produced no records.", None, None

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    size_bytes = destination.stat().st_size
    sha256_value = compute_sha256(destination)
    schema_status, error_message, schema_hash, columns = inspect_csv_schema(destination, "eccc_fwi_daily")
    if schema_status == STATUS_FAILED_READ:
        return STATUS_FAILED_READ, size_bytes, sha256_value, error_message, schema_hash, columns

    logger.info("Stanhope daily FWI computed for %s: %s days saved to %s", year, len(results), destination)
    return STATUS_OK, size_bytes, sha256_value, None, schema_hash, columns


def download_eccc_fwi_daily_periods(
    *,
    station_name: str,
    station_id: int | str,
    start_year: int,
    end_year: int,
    end_date: date,
    logger: logging.Logger,
    dry_run: bool,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Create/update annual Stanhope daily FWI files and their manifest rows."""
    manifest_rows: List[Dict[str, object]] = []
    schema_rows: List[Dict[str, object]] = []

    for year in range(start_year, end_year + 1):
        period = str(year)
        file_name = f"ECCC_{station_slug(station_name)}_{period}_daily_fwi.csv"
        destination = ECCC_FWI_CACHE_DIR / file_name

        year_start = date(year, FWI_SEASON_START_MONTH, FWI_SEASON_START_DAY)
        season_end = date(year, FWI_SEASON_END_MONTH, FWI_SEASON_END_DAY)
        year_end = min(season_end, end_date)

        if year_end < year_start:
            error_message = (
                "FWI compute window invalid: end date precedes configured seasonal start "
                f"({year_start.isoformat()})."
            )
            manifest_rows.append(
                create_manifest_row(
                    source="eccc_fwi_daily",
                    station_raw=station_name,
                    year=year,
                    period=period,
                    file_path=destination,
                    size_bytes=0,
                    sha256_value="",
                    status=STATUS_FAILED_READ,
                    error_message=error_message,
                    schema_hash=None,
                )
            )
            continue

        recompute_existing = False
        if destination.exists():
            bounds = get_fwi_date_bounds(destination)
            if bounds is None:
                recompute_existing = True
            else:
                first_date, last_date = bounds
                # We need to recompute if the file starts too late, or if it doesn't
                # extend far enough to cover the known year_end requirement.
                if first_date.month > FWI_SEASON_START_MONTH or last_date < year_end:
                    recompute_existing = True

        if destination.exists() and not recompute_existing:
            size_bytes = destination.stat().st_size
            sha256_value = compute_sha256(destination)
            schema_status, error_message, schema_hash, columns = inspect_csv_schema(destination, "eccc_fwi_daily")
            final_status = STATUS_OK if schema_status == STATUS_OK else STATUS_FAILED_READ
        else:
            if dry_run:
                manifest_rows.append(
                    create_manifest_row(
                        source="eccc_fwi_daily",
                        station_raw=station_name,
                        year=year,
                        period=period,
                        file_path=destination,
                        size_bytes=0,
                        sha256_value="",
                        status=STATUS_FAILED_DOWNLOAD,
                        error_message="dry-run skipped Stanhope daily FWI compute",
                        schema_hash=None,
                    )
                )
                continue

            final_status, size_bytes, sha256_value, error_message, schema_hash, columns = compute_stanhope_fwi_daily_file(
                station_name=station_name,
                climate_id=station_id,
                year=year,
                start_date=year_start,
                end_date=year_end,
                destination=destination,
                logger=logger,
            )

        if schema_hash and columns:
            schema_rows.append(
                {
                    "source": "eccc_fwi_daily",
                    "schema_hash": schema_hash,
                    "columns_json": json.dumps(columns, ensure_ascii=True),
                }
            )

        manifest_rows.append(
            create_manifest_row(
                source="eccc_fwi_daily",
                station_raw=station_name,
                year=year,
                period=period,
                file_path=destination,
                size_bytes=size_bytes,
                sha256_value=sha256_value,
                status=final_status,
                error_message=error_message,
                schema_hash=schema_hash,
            )
        )

    return manifest_rows, schema_rows


def update_schema_inventory(schema_records: List[Dict[str, object]], schema_inventory_path: Path) -> None:
    """Append and de-duplicate observed schemas over time for data governance."""
    if not schema_records:
        return
    now_utc = utc_now_iso()
    new_df = pd.DataFrame(schema_records)
    grouped = (
        new_df.groupby(["source", "schema_hash", "columns_json"], dropna=False)
        .size()
        .reset_index(name="seen_count")
    )
    grouped["first_seen_utc"] = now_utc
    grouped["last_seen_utc"] = now_utc
    grouped = grouped[SCHEMA_COLUMNS]

    if schema_inventory_path.exists():
        existing = pd.read_csv(schema_inventory_path)
        for col in SCHEMA_COLUMNS:
            if col not in existing.columns:
                existing[col] = pd.NA
        merged = pd.concat([existing[SCHEMA_COLUMNS], grouped], ignore_index=True)
    else:
        merged = grouped

    merged = (
        merged.groupby(["source", "schema_hash", "columns_json"], dropna=False)
        .agg(
            first_seen_utc=("first_seen_utc", "min"),
            last_seen_utc=("last_seen_utc", "max"),
            seen_count=("seen_count", "sum"),
        )
        .reset_index()
    )
    merged = merged[SCHEMA_COLUMNS]
    merged.to_csv(schema_inventory_path, index=False)


# -----------------------------------------------------------------------------
# Obtain stage: discover local files + optionally fetch online ECCC datasets
# -----------------------------------------------------------------------------

def run_obtain(
    raw_dir: Path,
    manifest_dir: Path,
    logger: logging.Logger,
    dry_run: bool,
    start_year: Optional[int],
    end_year: Optional[int],
    skip_eccc_download: bool,
) -> pd.DataFrame:
    """Build source inventory and manifests used as the scrub stage input contract."""
    manifest_dir.mkdir(parents=True, exist_ok=True)
    schema_records: List[Dict[str, object]] = []

    existing_hobolink_cache: Dict[str, Dict[str, object]] = {}
    if HOBOLINK_MANIFEST.exists():
        try:
            existing_hobolink_manifest = pd.read_csv(HOBOLINK_MANIFEST)
            if not existing_hobolink_manifest.empty:
                existing_hobolink_manifest = existing_hobolink_manifest[
                    existing_hobolink_manifest["source"].astype(str).str.lower() == "hobolink"
                ].copy()
                for _, cached_row in existing_hobolink_manifest.iterrows():
                    file_path_text = str(cached_row.get("file_path", "")).strip()
                    if file_path_text:
                        existing_hobolink_cache[file_path_text] = cached_row.to_dict()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unable to load existing HOBOlink manifest cache: %s", exc)

    hobolink_rows: List[Dict[str, object]] = []
    hobolink_years: List[int] = []
    for station in HOBOLINK_STATIONS:
        # Walk each station tree and register every CSV with schema diagnostics.
        station_dir = raw_dir / station
        if not station_dir.exists():
            logger.warning("Missing HOBOlink drop-zone: %s", station_dir)
            continue
        for csv_path in sorted(station_dir.rglob("*.csv")):
            stat_result = csv_path.stat()
            size_bytes = stat_result.st_size
            file_mtime_ns = int(stat_result.st_mtime_ns)
            year = extract_year(csv_path.name) or extract_year(str(csv_path.parent))
            if year is not None:
                hobolink_years.append(year)
            period = str(year) if year is not None else "unknown"

            schema_status: str
            err: Optional[str]
            schema_hash: Optional[str]
            cols: Optional[List[str]]
            sha256_value: str
            coverage_start_date: Optional[str] = None
            coverage_end_date: Optional[str] = None

            cache_hit = False
            cached = existing_hobolink_cache.get(str(csv_path))
            if cached is not None:
                cached_size_raw = cached.get("size_bytes")
                cached_mtime_raw = cached.get("file_mtime_ns")
                cached_sha = str(cached.get("sha256", "")).strip()
                try:
                    cached_size = int(float(cached_size_raw))
                except Exception:  # noqa: BLE001
                    cached_size = None
                try:
                    cached_mtime = int(float(cached_mtime_raw))
                except Exception:  # noqa: BLE001
                    cached_mtime = None

                if cached_size == size_bytes and cached_mtime == file_mtime_ns and cached_sha:
                    cache_hit = True
                    sha256_value = cached_sha
                    schema_status = str(cached.get("status", STATUS_OK)).strip() or STATUS_OK
                    err_text = str(cached.get("error_message", "")).strip()
                    err = err_text or None
                    schema_hash_text = str(cached.get("schema_hash", "")).strip()
                    schema_hash = schema_hash_text or None
                    coverage_start_text = str(cached.get("coverage_start_date", "")).strip()
                    coverage_end_text = str(cached.get("coverage_end_date", "")).strip()
                    coverage_start_date = coverage_start_text or None
                    coverage_end_date = coverage_end_text or None
                    cols = None

            if not cache_hit:
                schema_status, err, schema_hash, cols = inspect_csv_schema(csv_path, "hobolink")
                sha256_value = compute_sha256(csv_path)
                try:
                    bounds = read_hobolink_datetime_bounds(csv_path)
                    if bounds is not None:
                        coverage_start_date = bounds[0].isoformat()
                        coverage_end_date = bounds[1].isoformat()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Skipping HOBOlink bounds parse failure for %s: %s", csv_path, exc)

            row = create_manifest_row(
                source="hobolink",
                station_raw=station,
                year=year,
                period=period,
                file_path=csv_path,
                size_bytes=size_bytes,
                sha256_value=sha256_value,
                file_mtime_ns=file_mtime_ns,
                status=schema_status,
                error_message=err,
                schema_hash=schema_hash,
                coverage_start_date=coverage_start_date,
                coverage_end_date=coverage_end_date,
            )
            hobolink_rows.append(row)
            if schema_hash and cols:
                schema_records.append(
                    {
                        "source": "hobolink",
                        "schema_hash": schema_hash,
                        "columns_json": json.dumps(cols, ensure_ascii=True),
                    }
                )

    if start_year is not None and end_year is not None and start_year > end_year:
        raise ValueError(f"Invalid year range: start_year ({start_year}) > end_year ({end_year}).")

    inferred_start = min(hobolink_years) if hobolink_years else datetime.now(timezone.utc).year
    inferred_end = max(hobolink_years) if hobolink_years else datetime.now(timezone.utc).year
    effective_start_year = start_year if start_year is not None else inferred_start
    effective_end_year = end_year if end_year is not None else inferred_end

    logger.info(
        "ECCC internet fetch coverage window: %s to %s",
        effective_start_year,
        effective_end_year,
    )

    coverage_bounds = derive_hobolink_coverage_bounds(hobolink_rows, logger)
    if coverage_bounds is not None:
        hobolink_start, hobolink_end = coverage_bounds
        fwi_start_year = hobolink_start.year
        fwi_end_year = hobolink_end.year
        fwi_end_date = hobolink_end
    else:
        fwi_start_year = effective_start_year
        fwi_end_year = effective_end_year
        fwi_end_date = date(effective_end_year, FWI_SEASON_END_MONTH, FWI_SEASON_END_DAY)

    eccc_rows: List[Dict[str, object]] = []
    if skip_eccc_download:
        logger.info("Skipping ECCC internet fetch by user request (--skip-eccc-download).")
    else:
        downloaded_rows, downloaded_schema_rows = download_eccc_periods(
            station_name=ECCC_STANHOPE_NAME,
            climate_id=ECCC_STANHOPE_CLIMATE_ID,
            start_year=effective_start_year,
            end_year=effective_end_year,
            logger=logger,
            dry_run=dry_run,
        )
        eccc_rows.extend(downloaded_rows)
        schema_records.extend(downloaded_schema_rows)

    fwi_rows: List[Dict[str, object]] = []
    if skip_eccc_download:
        logger.info("Skipping Stanhope daily FWI compute by user request (--skip-eccc-download).")
    else:
        downloaded_fwi_rows, downloaded_fwi_schema_rows = download_eccc_fwi_daily_periods(
            station_name=ECCC_STANHOPE_NAME,
            station_id=ECCC_STANHOPE_CLIMATE_ID,
            start_year=fwi_start_year,
            end_year=fwi_end_year,
            end_date=fwi_end_date,
            logger=logger,
            dry_run=dry_run,
        )
        fwi_rows.extend(downloaded_fwi_rows)
        schema_records.extend(downloaded_fwi_schema_rows)

    if not dry_run:
        pd.DataFrame(hobolink_rows, columns=MANIFEST_COLUMNS).to_csv(HOBOLINK_MANIFEST, index=False)
        pd.DataFrame(eccc_rows, columns=MANIFEST_COLUMNS).to_csv(ECCC_MANIFEST, index=False)
        pd.DataFrame(fwi_rows, columns=MANIFEST_COLUMNS).to_csv(ECCC_FWI_MANIFEST, index=False)
        update_schema_inventory(schema_records, SCHEMA_INVENTORY)

    logger.info("Obtain summary: HOBOlink files=%s | ECCC files=%s | ECCC FWI files=%s", len(hobolink_rows), len(eccc_rows), len(fwi_rows))
    inventory = pd.concat(
        [
            pd.DataFrame(hobolink_rows),
            pd.DataFrame(eccc_rows),
        ],
        ignore_index=True,
    )
    return inventory


# -----------------------------------------------------------------------------
# Scrub stage helpers: parse, map columns, normalize precipitation and aggregate
# -----------------------------------------------------------------------------

def normalize_header_columns(raw_columns: Iterable[str]) -> List[str]:
    """Public wrapper for header normalization helper."""
    return _normalize_header_columns(list(raw_columns))


def detect_hobolink_header_and_encoding(csv_path: Path, max_scan_lines: int = 120) -> Tuple[int, str]:
    """Infer HOBOlink header row index and text encoding from a short scan."""
    for encoding in READ_ENCODINGS:
        try:
            with csv_path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.reader(handle)
                scan_rows: List[List[str]] = []
                for i, row in enumerate(reader):
                    scan_rows.append(row)
                    if i >= max_scan_lines:
                        break

            for idx, row in enumerate(scan_rows):
                lowered = [cell.strip().lower() for cell in row]
                if "date" in lowered and "time" in lowered:
                    return idx, encoding

            best_idx = -1
            best_score = -1
            for idx, row in enumerate(scan_rows):
                score = len([cell for cell in row if str(cell).strip()])
                if score >= 4 and score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx >= 0:
                return best_idx, encoding
        except UnicodeDecodeError:
            continue

    raise ValueError("Unable to detect HOBOlink header row.")


def find_column(df: pd.DataFrame, required_tokens: List[str], banned_tokens: List[str] | None = None) -> List[str]:
    """Find columns whose names contain all required tokens and none of the banned tokens."""
    banned_tokens = banned_tokens or []
    matches: List[str] = []
    for col in df.columns:
        low = col.lower()
        if all(token in low for token in required_tokens) and all(token not in low for token in banned_tokens):
            matches.append(col)
    return matches


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Convert a dataframe column to numeric, coercing bad values to NaN."""
    return pd.to_numeric(df[column], errors="coerce")


def choose_best_column(df: pd.DataFrame, candidates: List[str]) -> str | None:
    """Pick candidate with the highest usable numeric ratio."""
    if not candidates:
        return None
    best_name = None
    best_ratio = -1.0
    for col in candidates:
        ratio = numeric_series(df, col).notna().mean()
        if ratio > best_ratio:
            best_ratio = ratio
            best_name = col
    return best_name


def parse_hobolink_file(path: Path, row: pd.Series, logger: logging.Logger) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    """Parse one HOBOlink CSV into canonical weather variables in UTC."""
    header_row, encoding = detect_hobolink_header_and_encoding(path)
    raw = pd.read_csv(
        path,
        skiprows=header_row,
        encoding=encoding,
        na_values=NA_STRINGS,
        keep_default_na=True,
        engine="python",
        on_bad_lines="skip",
    )
    raw.columns = normalize_header_columns(raw.columns)

    if "Date" in raw.columns and "Time" in raw.columns:
        # First try strict timestamp format; fall back if too many parse failures.
        dt_text = raw["Date"].astype(str).str.strip() + " " + raw["Time"].astype(str).str.strip()
        timestamp = pd.to_datetime(dt_text, format="%m/%d/%Y %H:%M:%S %z", errors="coerce", utc=True)
        if float(timestamp.isna().mean()) > 0.05:
            timestamp = pd.to_datetime(dt_text, errors="coerce", utc=True, cache=True)
    else:
        datetime_cols = [c for c in raw.columns if "date/time" in c.lower() or "datetime" in c.lower()]
        if not datetime_cols:
            raise ValueError(f"HOBOlink file does not contain parseable Date/Time columns: {path}")
        timestamp = pd.to_datetime(raw[datetime_cols[0]], errors="coerce", utc=True)

    mapped: Dict[str, pd.Series] = {}
    mapping_log: List[Dict[str, object]] = []

    temp_candidates = find_column(raw, ["temperature", "°c"], banned_tokens=["water", "dew point"])
    rh_candidates = find_column(raw, ["rh"], banned_tokens=["threshold"])
    wind_speed_kmh_candidates = find_column(raw, ["wind", "speed", "km/h"], banned_tokens=["gust"])
    wind_speed_ms_candidates = find_column(raw, ["wind", "speed", "m/s"], banned_tokens=["gust"])
    wind_dir_candidates = find_column(raw, ["wind direction"])
    rain_candidates = [
        c for c in raw.columns if "rain" in c.lower() and "mm" in c.lower() and "accumulated" not in c.lower()
    ]
    rain_candidates += [
        c
        for c in raw.columns
        if "accumulated rain" in c.lower() and "mm" in c.lower() and c not in rain_candidates
    ]

    selected_temp = choose_best_column(raw, temp_candidates)
    selected_rh = choose_best_column(raw, rh_candidates)
    selected_wind_speed = choose_best_column(raw, wind_speed_kmh_candidates)
    wind_speed_in_ms = False
    if selected_wind_speed is None:
        selected_wind_speed = choose_best_column(raw, wind_speed_ms_candidates)
        wind_speed_in_ms = selected_wind_speed is not None
    selected_wind_dir = choose_best_column(raw, wind_dir_candidates)
    selected_rain = choose_best_column(raw, rain_candidates)

    selected = {
        CANONICAL_VARIABLES["temp"]: selected_temp,
        CANONICAL_VARIABLES["rh"]: selected_rh,
        CANONICAL_VARIABLES["wind_speed"]: selected_wind_speed,
        CANONICAL_VARIABLES["wind_dir"]: selected_wind_dir,
        CANONICAL_VARIABLES["rain"]: selected_rain,
    }

    for canonical_var, source_col in selected.items():
        if source_col is None:
            mapped[canonical_var] = pd.Series(pd.NA, index=raw.index, dtype="Float64")
            mapping_log.append(
                {
                    "file_path": str(path),
                    "canonical_variable": canonical_var,
                    "selected_column": None,
                    "non_null_fraction": 0.0,
                }
            )
            continue

        numeric = numeric_series(raw, source_col)
        if canonical_var == CANONICAL_VARIABLES["wind_speed"] and wind_speed_in_ms:
            numeric = numeric * 3.6
        mapped[canonical_var] = numeric
        mapping_log.append(
            {
                "file_path": str(path),
                "canonical_variable": canonical_var,
                "selected_column": source_col,
                "non_null_fraction": float(numeric.notna().mean()),
            }
        )

    out = pd.DataFrame(
        {
            "datetime_utc": timestamp,
            "station_raw": row.get("station_raw", ""),
            "station_slug": row.get("station_slug", station_slug(str(row.get("station_raw", "")))),
            "source": "hobolink",
            CANONICAL_VARIABLES["temp"]: mapped[CANONICAL_VARIABLES["temp"]],
            CANONICAL_VARIABLES["rh"]: mapped[CANONICAL_VARIABLES["rh"]],
            CANONICAL_VARIABLES["wind_speed"]: mapped[CANONICAL_VARIABLES["wind_speed"]],
            CANONICAL_VARIABLES["wind_dir"]: mapped[CANONICAL_VARIABLES["wind_dir"]],
            CANONICAL_VARIABLES["rain"]: mapped[CANONICAL_VARIABLES["rain"]],
        }
    )
    out = out.dropna(subset=["datetime_utc"]).copy()

    if not out.empty and not out["datetime_utc"].is_monotonic_increasing:
        out = out.sort_values("datetime_utc").reset_index(drop=True)
        logger.warning("Sorted non-monotonic timestamps in file %s", path)

    return out, mapping_log


def parse_eccc_file(path: Path, row: pd.Series) -> pd.DataFrame:
    """Parse one ECCC hourly CSV and map source fields to canonical names."""
    raw = pd.read_csv(path, na_values=NA_STRINGS, keep_default_na=True, low_memory=False)

    if "Date/Time (LST)" not in raw.columns:
        raise ValueError(f"ECCC file is missing Date/Time (LST): {path}")

    dt_local = pd.to_datetime(raw["Date/Time (LST)"], errors="coerce")
    atlantic_lst = timezone(timedelta(hours=-4))
    dt_utc = dt_local.dt.tz_localize(atlantic_lst).dt.tz_convert(timezone.utc)

    out = pd.DataFrame(
        {
            "datetime_utc": dt_utc,
            "station_raw": row.get("station_raw", "Stanhope"),
            "station_slug": row.get("station_slug", station_slug(str(row.get("station_raw", "Stanhope")))),
            "source": "eccc",
            CANONICAL_VARIABLES["temp"]: pd.to_numeric(raw.get("Temp (°C)"), errors="coerce"),
            CANONICAL_VARIABLES["rh"]: pd.to_numeric(raw.get("Rel Hum (%)"), errors="coerce"),
            CANONICAL_VARIABLES["wind_speed"]: pd.to_numeric(raw.get("Wind Spd (km/h)"), errors="coerce"),
            CANONICAL_VARIABLES["wind_dir"]: pd.to_numeric(raw.get("Wind Dir (10s deg)"), errors="coerce")
            * 10.0,
            CANONICAL_VARIABLES["rain"]: pd.to_numeric(raw.get("Precip. Amount (mm)"), errors="coerce"),
        }
    )
    out = out.dropna(subset=["datetime_utc"]).copy()
    return out.sort_values("datetime_utc").reset_index(drop=True)


def classify_and_normalize_precip(series: pd.Series, source: str) -> Tuple[pd.Series, Dict[str, object]]:
    """Detect whether precipitation is cumulative and convert to hourly increments if needed."""
    precip = pd.to_numeric(series, errors="coerce")

    if source == "eccc":
        return precip, {
            "interpretation": "hourly_incremental",
            "negative_diff_count": 0,
            "reset_count": 0,
            "non_negative_diff_ratio": 1.0,
            "dynamic_range_mm": float((precip.max() - precip.min()) if precip.notna().any() else 0.0),
        }

    diffs = precip.diff()
    valid_diffs = diffs.dropna()
    if valid_diffs.empty:
        return precip, {
            "interpretation": "incremental_assumed",
            "negative_diff_count": 0,
            "reset_count": 0,
            "non_negative_diff_ratio": 1.0,
            "dynamic_range_mm": float((precip.max() - precip.min()) if precip.notna().any() else 0.0),
        }

    non_negative_ratio = float((valid_diffs >= -0.05).mean())
    reset_count = int((valid_diffs <= -1.0).sum())
    negative_diff_count = int((valid_diffs < 0).sum())
    dynamic_range = float((precip.max() - precip.min()) if precip.notna().any() else 0.0)
    is_cumulative = non_negative_ratio >= 0.95 and (reset_count >= 1 or dynamic_range >= 30.0)

    if not is_cumulative:
        return precip, {
            "interpretation": "incremental",
            "negative_diff_count": negative_diff_count,
            "reset_count": reset_count,
            "non_negative_diff_ratio": non_negative_ratio,
            "dynamic_range_mm": dynamic_range,
        }

    incremental = diffs.copy()
    incremental[(diffs < 0)] = pd.NA
    return incremental, {
        "interpretation": "cumulative_to_incremental",
        "negative_diff_count": negative_diff_count,
        "reset_count": reset_count,
        "non_negative_diff_ratio": non_negative_ratio,
        "dynamic_range_mm": dynamic_range,
    }


def aggregate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate parsed observations to hourly resolution by station/source."""
    if df.empty:
        return df

    work = df.copy()
    work["datetime_utc"] = pd.to_datetime(work["datetime_utc"], utc=True, errors="coerce")
    work = work.dropna(subset=["datetime_utc"])
    work["datetime_utc"] = work["datetime_utc"].dt.floor(HOURLY_FREQ)

    wind_speed_col = CANONICAL_VARIABLES["wind_speed"]
    wind_dir_col = CANONICAL_VARIABLES["wind_dir"]

    valid_dir = work[wind_dir_col].notna() & work[wind_speed_col].notna() & (work[wind_speed_col] > 0)
    # Convert direction+speed into vector components for circular averaging.
    rad = work[wind_dir_col].astype(float).map(math.radians)
    work["_wind_x"] = pd.NA
    work["_wind_y"] = pd.NA
    work.loc[valid_dir, "_wind_x"] = work.loc[valid_dir, wind_speed_col] * rad[valid_dir].map(math.cos)
    work.loc[valid_dir, "_wind_y"] = work.loc[valid_dir, wind_speed_col] * rad[valid_dir].map(math.sin)

    grouped = (
        work.groupby(["station_raw", "station_slug", "source", "datetime_utc"], as_index=False)
        .agg(
            air_temperature_c=(CANONICAL_VARIABLES["temp"], "mean"),
            relative_humidity_pct=(CANONICAL_VARIABLES["rh"], "mean"),
            wind_speed_kmh=(CANONICAL_VARIABLES["wind_speed"], "mean"),
            precipitation_mm=(
                CANONICAL_VARIABLES["rain"],
                lambda s: pd.to_numeric(s, errors="coerce").sum(min_count=1),
            ),
            _wind_x=("_wind_x", "sum"),
            _wind_y=("_wind_y", "sum"),
            _dir_obs=(wind_dir_col, lambda s: int(s.notna().sum())),
        )
        .sort_values("datetime_utc")
        .reset_index(drop=True)
    )

    dir_values: List[float] = []
    for _, row in grouped.iterrows():
        if row["_dir_obs"] <= 0:
            dir_values.append(float("nan"))
            continue
        x_val = row["_wind_x"]
        y_val = row["_wind_y"]
        if pd.isna(x_val) or pd.isna(y_val) or (float(x_val) == 0.0 and float(y_val) == 0.0):
            dir_values.append(float("nan"))
            continue
        angle = math.degrees(math.atan2(float(y_val), float(x_val)))
        dir_values.append(angle % 360.0)

    grouped[CANONICAL_VARIABLES["wind_dir"]] = pd.Series(dir_values, dtype="Float64")
    grouped = grouped.drop(columns=["_wind_x", "_wind_y", "_dir_obs"])
    return grouped


def missing_run_lengths(mask: pd.Series) -> List[int]:
    """Return lengths of contiguous missing-value runs."""
    lengths: List[int] = []
    run = 0
    for value in mask.fillna(False).tolist():
        if value:
            run += 1
        elif run > 0:
            lengths.append(run)
            run = 0
    if run > 0:
        lengths.append(run)
    return lengths


def short_gap_mask(series: pd.Series, max_gap: int = 2) -> pd.Series:
    """Flag interior missing runs whose lengths are small enough to be filled safely."""
    missing = series.isna()
    eligible = pd.Series(False, index=series.index, dtype="boolean")
    i = 0
    n = len(series)
    while i < n:
        if not missing.iloc[i]:
            i += 1
            continue
        start = i
        while i < n and missing.iloc[i]:
            i += 1
        end = i - 1
        run_len = end - start + 1
        is_interior = start > 0 and i < n
        if run_len <= max_gap and is_interior:
            eligible.iloc[start : end + 1] = True
    return eligible


def apply_step_filter(series: pd.Series, threshold: float) -> Tuple[pd.Series, pd.Series]:
    """Drop implausible jumps by comparing each value to previous valid observation."""
    values = pd.to_numeric(series, errors="coerce").copy()
    failed = pd.Series(False, index=values.index, dtype="boolean")
    previous_valid = None
    for idx, value in values.items():
        if pd.isna(value):
            continue
        if previous_valid is None:
            previous_valid = float(value)
            continue
        if abs(float(value) - previous_valid) > threshold:
            values.at[idx] = pd.NA
            failed.at[idx] = True
            continue
        previous_valid = float(value)
    return values, failed


def apply_qc_and_fill(station_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    """Apply range/step QC rules and fill only short interior gaps (<=2 hours)."""
    df = station_df.copy().sort_values("datetime_utc").reset_index(drop=True)
    qc_rows: List[Dict[str, object]] = []

    for var in CANONICAL_ORDER:
        df[f"{var}_failed_qc"] = pd.Series(False, index=df.index, dtype="boolean")
        df[f"{var}_filled_short_gap"] = pd.Series(False, index=df.index, dtype="boolean")

    temp_col = CANONICAL_VARIABLES["temp"]
    rh_col = CANONICAL_VARIABLES["rh"]
    ws_col = CANONICAL_VARIABLES["wind_speed"]
    wd_col = CANONICAL_VARIABLES["wind_dir"]
    rain_col = CANONICAL_VARIABLES["rain"]

    temp_invalid = (df[temp_col] < TEMP_BOUNDS[0]) | (df[temp_col] > TEMP_BOUNDS[1])
    df.loc[temp_invalid, temp_col] = pd.NA
    df.loc[temp_invalid, f"{temp_col}_failed_qc"] = True

    rh_negative = df[rh_col] < RH_MIN
    rh_soft_cap = (df[rh_col] > RH_CAP_MAX) & (df[rh_col] <= RH_SOFT_MAX)
    rh_extreme = df[rh_col] > RH_SOFT_MAX
    df.loc[rh_negative | rh_extreme, rh_col] = pd.NA
    df.loc[rh_negative | rh_extreme, f"{rh_col}_failed_qc"] = True
    df.loc[rh_soft_cap, rh_col] = RH_CAP_MAX

    ws_invalid = df[ws_col] < WIND_SPEED_MIN
    df.loc[ws_invalid, ws_col] = pd.NA
    df.loc[ws_invalid, f"{ws_col}_failed_qc"] = True

    wd_invalid = (df[wd_col] < WIND_DIR_BOUNDS[0]) | (df[wd_col] > WIND_DIR_BOUNDS[1])
    df.loc[wd_invalid, wd_col] = pd.NA
    df.loc[wd_invalid, f"{wd_col}_failed_qc"] = True
    df[wd_col] = pd.to_numeric(df[wd_col], errors="coerce") % 360.0

    rain_negative = df[rain_col] < 0
    df.loc[rain_negative, rain_col] = pd.NA
    df.loc[rain_negative, f"{rain_col}_failed_qc"] = True

    for var, threshold in STEP_THRESHOLDS.items():
        filtered, failed = apply_step_filter(df[var], threshold)
        df[var] = filtered
        df.loc[failed, f"{var}_failed_qc"] = True

    for var in CONTINUOUS_FILL_VARS:
        # Policy: interpolate only short interior gaps so long outages remain visible.
        before_missing = df[var].isna()
        filled = pd.to_numeric(df[var], errors="coerce").interpolate(method="linear", limit=2, limit_area="inside")
        df[var] = filled
        after_filled = before_missing & df[var].notna()
        df.loc[after_filled, f"{var}_filled_short_gap"] = True

    dir_missing_eligible = short_gap_mask(df[wd_col], max_gap=2)
    dir_nearest = pd.to_numeric(df[wd_col], errors="coerce").interpolate(method="nearest", limit=2, limit_area="inside")
    dir_fill_mask = dir_missing_eligible & df[wd_col].isna() & dir_nearest.notna() & (df[ws_col] > 0)
    df.loc[dir_fill_mask, wd_col] = dir_nearest[dir_fill_mask]
    df.loc[dir_fill_mask, f"{wd_col}_filled_short_gap"] = True

    calm_mask = (df[ws_col] == 0) | df[ws_col].isna()
    df.loc[calm_mask, wd_col] = pd.NA

    station = str(df["station_slug"].iloc[0]) if not df.empty else "unknown"
    for var in CANONICAL_ORDER:
        qc_rows.append(
            {
                "station_slug": station,
                "variable": var,
                "failed_qc_count": int(df[f"{var}_failed_qc"].fillna(False).sum()),
                "filled_short_gap_count": int(df[f"{var}_filled_short_gap"].fillna(False).sum()),
                "missing_after_qc_fill": int(df[var].isna().sum()),
            }
        )

    return df, qc_rows


def build_complete_hourly_grid(station_df: pd.DataFrame) -> pd.DataFrame:
    """Expand a station series to a complete hourly timeline between min/max timestamps."""
    if station_df.empty:
        return station_df
    station_df = station_df.sort_values("datetime_utc").reset_index(drop=True)
    start = station_df["datetime_utc"].min()
    end = station_df["datetime_utc"].max()
    full_index = pd.date_range(start=start, end=end, freq=HOURLY_FREQ, tz=timezone.utc)

    keys = station_df[["station_raw", "station_slug", "source"]].iloc[0].to_dict()
    grid = pd.DataFrame({"datetime_utc": full_index})
    for key, value in keys.items():
        grid[key] = value
    return grid.merge(station_df, on=["station_raw", "station_slug", "source", "datetime_utc"], how="left")


def summarize_missingness(station_df: pd.DataFrame) -> List[Dict[str, object]]:
    """Summarize missingness and gap-run structure by variable for one station."""
    rows: List[Dict[str, object]] = []
    if station_df.empty:
        return rows
    station = str(station_df["station_slug"].iloc[0])
    total_hours = len(station_df)
    for var in CANONICAL_ORDER:
        miss_mask = station_df[var].isna()
        runs = missing_run_lengths(miss_mask)
        rows.append(
            {
                "station_slug": station,
                "variable": var,
                "total_hours": total_hours,
                "missing_hours": int(miss_mask.sum()),
                "missing_pct": float(miss_mask.mean() * 100.0),
                "max_missing_run_hours": int(max(runs) if runs else 0),
                "run_lengths_json": json.dumps(runs),
            }
        )
    return rows


def cast_output_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast output columns to stable dtypes for consistent downstream I/O."""
    out = df.copy()
    for var in CANONICAL_ORDER:
        out[var] = pd.to_numeric(out[var], errors="coerce").astype("Float32")
        out[f"{var}_failed_qc"] = out[f"{var}_failed_qc"].astype("boolean")
        out[f"{var}_filled_short_gap"] = out[f"{var}_filled_short_gap"].astype("boolean")
    return out


def assert_output_schema(df: pd.DataFrame) -> None:
    """Validate final scrub schema and station-hour uniqueness contract."""
    required = ["station_raw", "station_slug", "source", "datetime_utc"] + CANONICAL_ORDER
    for var in CANONICAL_ORDER:
        required.append(f"{var}_failed_qc")
        required.append(f"{var}_filled_short_gap")
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in scrub output: {missing}")
    dupes = df.duplicated(subset=["station_slug", "datetime_utc"])
    if bool(dupes.any()):
        raise ValueError(f"Duplicate station/hour keys detected: {int(dupes.sum())}")


def manifest_signature(latest_inventory: pd.DataFrame) -> str:
    """Create a deterministic signature of inputs used for this scrub run."""
    if latest_inventory.empty:
        return ""
    subset = latest_inventory[["source", "file_path", "sha256", "status"]].fillna("")
    ordered = subset.sort_values(["source", "file_path"]).to_dict(orient="records")
    payload = json.dumps(ordered, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def append_scrub_run_manifest(
    *,
    run_id: str,
    run_timestamp_utc: str,
    inventory: pd.DataFrame,
    output_path: Path,
    output_df: pd.DataFrame,
    obtain_signature: str,
    scrub_runs_manifest: Path,
) -> None:
    """Append run-level metadata so outputs can be traced back to exact inputs."""
    station_ranges = (
        output_df.groupby("station_slug", dropna=False)["datetime_utc"].agg(["min", "max"]).reset_index()
    )
    station_range_payload = []
    for _, row in station_ranges.iterrows():
        station_range_payload.append(
            {
                "station_slug": row["station_slug"],
                "min_datetime_utc": row["min"].isoformat() if pd.notna(row["min"]) else None,
                "max_datetime_utc": row["max"].isoformat() if pd.notna(row["max"]) else None,
            }
        )

    record = {
        "run_timestamp_utc": run_timestamp_utc,
        "run_id": run_id,
        "input_hobolink_files": int((inventory["source"] == "hobolink").sum()),
        "input_eccc_files": int((inventory["source"] == "eccc").sum()),
        "obtain_manifest_signature": obtain_signature,
        "output_file_path": str(output_path),
        "output_rows": int(len(output_df)),
        "station_ranges_json": json.dumps(station_range_payload, ensure_ascii=True),
    }

    new_df = pd.DataFrame([record])
    if scrub_runs_manifest.exists():
        existing = pd.read_csv(scrub_runs_manifest)
        merged = pd.concat([existing, new_df], ignore_index=True)
    else:
        merged = new_df
    merged.to_csv(scrub_runs_manifest, index=False)


def run_scrub(
    inventory: pd.DataFrame,
    *,
    output_hourly: Path,
    output_missingness: Path,
    output_qc_counts: Path,
    output_precip_log: Path,
    scrub_runs_manifest: Path,
    logger: logging.Logger,
    dry_run: bool,
) -> Dict[str, object]:
    """Execute parse -> normalize -> aggregate -> QC/fill -> write scrubbed artifacts."""
    if inventory.empty:
        raise ValueError("No usable input files found after obtain discovery.")

    inventory = inventory[inventory["status"] == STATUS_OK].copy()
    if inventory.empty:
        raise ValueError("All discovered files failed schema checks.")

    all_hourly_chunks: List[pd.DataFrame] = []
    precip_log_rows: List[Dict[str, object]] = []

    for _, manifest_row in inventory.iterrows():
        # Parse each source file into a common schema before hourly aggregation.
        source = str(manifest_row.get("source", "")).strip().lower()
        file_path = Path(str(manifest_row.get("file_path", "")))
        if not file_path.exists():
            logger.warning("Skipping missing file listed in manifest: %s", file_path)
            continue

        if source == "hobolink":
            parsed, _ = parse_hobolink_file(file_path, manifest_row, logger)
        elif source == "eccc":
            parsed = parse_eccc_file(file_path, manifest_row)
        else:
            continue

        if parsed.empty:
            logger.warning("Parsed data was empty for %s", file_path)
            continue

        precip_norm, precip_meta = classify_and_normalize_precip(parsed[CANONICAL_VARIABLES["rain"]], source)
        parsed[CANONICAL_VARIABLES["rain"]] = precip_norm
        precip_log_rows.append(
            {
                "station_slug": parsed["station_slug"].iloc[0],
                "source": source,
                "file_path": str(file_path),
                **precip_meta,
            }
        )

        hourly = aggregate_hourly(parsed)
        all_hourly_chunks.append(hourly)

    if not all_hourly_chunks:
        raise ValueError("No hourly data generated from discovered files.")

    hourly_all = pd.concat(all_hourly_chunks, ignore_index=True)
    hourly_all = (
        hourly_all.groupby(["station_raw", "station_slug", "source", "datetime_utc"], as_index=False)
        .agg(
            air_temperature_c=(CANONICAL_VARIABLES["temp"], "mean"),
            relative_humidity_pct=(CANONICAL_VARIABLES["rh"], "mean"),
            wind_speed_kmh=(CANONICAL_VARIABLES["wind_speed"], "mean"),
            wind_direction_deg=(CANONICAL_VARIABLES["wind_dir"], "mean"),
            precipitation_mm=(
                CANONICAL_VARIABLES["rain"],
                lambda s: pd.to_numeric(s, errors="coerce").sum(min_count=1),
            ),
        )
        .sort_values(["station_slug", "datetime_utc"])
        .reset_index(drop=True)
    )

    station_outputs: List[pd.DataFrame] = []
    missingness_rows: List[Dict[str, object]] = []
    qc_rows: List[Dict[str, object]] = []

    for _, station_df in hourly_all.groupby("station_slug", dropna=False):
        station_df = station_df.sort_values("datetime_utc").reset_index(drop=True)
        expanded = build_complete_hourly_grid(station_df)
        cleaned, station_qc_rows = apply_qc_and_fill(expanded)
        station_outputs.append(cleaned)
        missingness_rows.extend(summarize_missingness(cleaned))
        qc_rows.extend(station_qc_rows)

    final_df = pd.concat(station_outputs, ignore_index=True)
    final_df = cast_output_dtypes(final_df)
    final_df = final_df.sort_values(["station_slug", "datetime_utc"]).reset_index(drop=True)
    assert_output_schema(final_df)

    if not dry_run:
        # Persist primary cleaned table plus quality diagnostics and run metadata.
        write_df = final_df.copy()
        write_df["datetime_utc"] = pd.to_datetime(write_df["datetime_utc"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        write_df.to_csv(output_hourly, index=False)
        pd.DataFrame(missingness_rows).to_csv(output_missingness, index=False)
        pd.DataFrame(qc_rows).to_csv(output_qc_counts, index=False)
        pd.DataFrame(precip_log_rows).to_csv(output_precip_log, index=False)

        append_scrub_run_manifest(
            run_id=f"scrub_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}",
            run_timestamp_utc=utc_now_iso(),
            inventory=inventory,
            output_path=output_hourly,
            output_df=final_df,
            obtain_signature=manifest_signature(inventory),
            scrub_runs_manifest=scrub_runs_manifest,
        )

    return {
        "rows": int(len(final_df)),
        "stations": int(final_df["station_slug"].nunique(dropna=True)),
        "date_min": final_df["datetime_utc"].min(),
        "date_max": final_df["datetime_utc"].max(),
        "missing_pct": {
            var: float(final_df[var].isna().mean() * 100.0)
            for var in [CANONICAL_VARIABLES["temp"], CANONICAL_VARIABLES["rh"], CANONICAL_VARIABLES["wind_speed"], CANONICAL_VARIABLES["rain"]]
        },
    }


# -----------------------------------------------------------------------------
# CLI wiring and business-facing run summary
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Define command-line options for running this standalone cleaning pipeline."""
    parser = argparse.ArgumentParser(
        description="Run combined obtain + scrub pipeline into scrubbed hourly outputs.",
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="Raw input directory root.")
    parser.add_argument("--scrubbed-dir", type=Path, default=SCRUBBED_DIR, help="Scrubbed output directory.")
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR, help="Outputs directory for logs/tables/figures.")
    parser.add_argument("--start-year", type=int, default=None, help="Start year for ECCC internet fetch window.")
    parser.add_argument("--end-year", type=int, default=None, help="End year for ECCC internet fetch window.")
    parser.add_argument(
        "--skip-eccc-download",
        action="store_true",
        help="Skip ECCC internet fetch and use only files already present in data/raw/ECCC_Stanhope.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs if present.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize without writing files.")
    return parser.parse_args()


def main() -> int:
    """Program entrypoint: run obtain + scrub stages and print a decision-oriented summary."""
    args = parse_args()

    raw_dir = args.raw_dir.resolve()
    scrubbed_dir = args.scrubbed_dir.resolve()
    outputs_dir = args.outputs_dir.resolve()
    manifest_dir = scrubbed_dir / "_manifests"

    output_hourly = scrubbed_dir / OUTPUT_HOURLY.name
    output_missingness = scrubbed_dir / OUTPUT_MISSINGNESS.name
    output_qc_counts = scrubbed_dir / OUTPUT_QC_COUNTS.name
    output_precip_log = scrubbed_dir / OUTPUT_PRECIP_LOG.name
    scrub_runs_manifest = manifest_dir / SCRUB_RUNS_MANIFEST.name

    ensure_directories(raw_dir, scrubbed_dir, outputs_dir)

    log_path = outputs_dir / "logs" / f"cleaning_{datetime.now(timezone.utc):%Y%m%d}.log"
    logger = setup_logging(log_path)

    if args.force:
        logger.info("--force provided; existing scrubbed outputs will be overwritten.")
    elif not args.dry_run:
        logger.info("Overwrite mode is default; existing scrubbed outputs/manifests will be replaced.")

    logger.info("Step A (obtain): discovering local files and building manifests.")
    inventory = run_obtain(
        raw_dir=raw_dir,
        manifest_dir=manifest_dir,
        logger=logger,
        dry_run=args.dry_run,
        start_year=args.start_year,
        end_year=args.end_year,
        skip_eccc_download=args.skip_eccc_download,
    )

    logger.info("Step B (scrub): parsing, UTC normalization, QC, fill<=2h, hourly aggregation.")
    summary = run_scrub(
        inventory,
        output_hourly=output_hourly,
        output_missingness=output_missingness,
        output_qc_counts=output_qc_counts,
        output_precip_log=output_precip_log,
        scrub_runs_manifest=scrub_runs_manifest,
        logger=logger,
        dry_run=args.dry_run,
    )

    date_min = summary["date_min"]
    date_max = summary["date_max"]
    logger.info("Cleaning completed successfully.")
    logger.info("Summary rows=%s stations=%s range=%s -> %s", summary["rows"], summary["stations"], date_min, date_max)
    logger.info("Missingness (%%): %s", summary["missing_pct"])
    if args.dry_run:
        logger.info("Dry run mode enabled: no files were written.")
    else:
        logger.info("Wrote hourly: %s", output_hourly)
        logger.info("Wrote missingness: %s", output_missingness)
        logger.info("Wrote QC counts: %s", output_qc_counts)
        logger.info("Wrote precip log: %s", output_precip_log)

    variable_labels = {
        "air_temperature_c": "Air temperature",
        "relative_humidity_pct": "Relative humidity",
        "wind_speed_kmh": "Wind speed",
        "precipitation_mm": "Precipitation",
    }
    core_variables = ["air_temperature_c", "relative_humidity_pct", "wind_speed_kmh"]

    core_missing = [float(summary["missing_pct"].get(var, float("nan"))) for var in core_variables]
    core_missing_clean = [value for value in core_missing if not math.isnan(value)]
    avg_core_missing = sum(core_missing_clean) / len(core_missing_clean) if core_missing_clean else float("nan")

    if math.isnan(avg_core_missing):
        readiness_label = "unknown"
        readiness_note = "Core readiness could not be assessed from missingness."
    elif avg_core_missing <= 5.0:
        readiness_label = "strong"
        readiness_note = "Core weather coverage is strong for operational fire-weather use."
    elif avg_core_missing <= 15.0:
        readiness_label = "moderate"
        readiness_note = "Core weather coverage is usable, but data gaps should be monitored."
    else:
        readiness_label = "at-risk"
        readiness_note = "Core weather coverage is weak and may reduce confidence in downstream decisions."

    worst_var = None
    worst_missing = -1.0
    for var, value in summary["missing_pct"].items():
        value_float = float(value)
        if value_float > worst_missing:
            worst_missing = value_float
            worst_var = var

    print("=== CLEANING SUMMARY ===")
    print(f"Rows: {summary['rows']}")
    print(f"Stations: {summary['stations']}")
    print(f"Date range (UTC): {date_min} -> {date_max}")
    print("Missingness (%):")
    for key, val in summary["missing_pct"].items():
        label = variable_labels.get(key, key)
        print(f"  - {label}: {val:.2f}%")

    print("\n=== BUSINESS INTERPRETATION ===")
    print(f"Core readiness status: {readiness_label.upper()}")
    print(readiness_note)
    if worst_var is not None:
        print(
            "Highest data-gap risk: "
            f"{variable_labels.get(worst_var, worst_var)} ({worst_missing:.2f}% missing)."
        )

    if args.dry_run:
        print("\nMode: dry-run (no files written)")
        print("This run is suitable for pre-flight validation before generating deliverables.")
    else:
        print("\nArtifacts written:")
        print(f"  - Hourly weather: {output_hourly}")
        print(f"  - Missingness summary: {output_missingness}")
        print(f"  - QC out-of-range counts: {output_qc_counts}")
        print(f"  - Precip semantics log: {output_precip_log}")
        print("\nNext business step: run analysis.ipynb to produce decision-ready station redundancy and FWI risk outputs.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
