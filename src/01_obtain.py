"""Step 2 obtain stage for local HOBOlink discovery and ECCC ingestion."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import time
import warnings
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

from config import (
    ECCC_CACHE_DIR,
    ECCC_FWI_CACHE_DIR,
    ECCC_STANHOPE_CLIMATE_ID,
    ECCC_STANHOPE_NAME,
    LOGS_DIR,
    MANIFEST_DIR,
    get_hobolink_dropzones,
    ensure_directories,
    setup_logging,
)
from obtain_utils import (
    acquire_obtain_lock,
    append_manifest_rows,
    compute_sha256,
    download_to_file_atomic,
    latest_manifest_index,
    load_manifest,
    release_obtain_lock,
    response_looks_like_csv,
    update_schema_inventory,
    utc_now_iso,
)

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
    "ingested_at_utc",
    "status",
    "error_message",
    "schema_hash",
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
STATUS_DOWNLOADED = "downloaded"
STATUS_SKIPPED_UNCHANGED = "skipped_unchanged"
STATUS_FAILED_READ = "failed_read"
STATUS_FAILED_DOWNLOAD = "failed_download"

HOBOLINK_MANIFEST = MANIFEST_DIR / "01_obtain_hobolink_files.csv"
ECCC_MANIFEST = MANIFEST_DIR / "01_obtain_eccc_periods.csv"
ECCC_FWI_MANIFEST = MANIFEST_DIR / "01_obtain_eccc_fwi_daily_periods.csv"
SCHEMA_INVENTORY = MANIFEST_DIR / "01_schema_inventory.csv"
OBTAIN_LOCK_PATH = LOGS_DIR / "01_obtain.lock"

CSV_MIME_HINTS = (
    "text/csv",
    "application/csv",
    "application/octet-stream",
    "application/vnd.ms-excel",
    "application/force-download",
    "text/plain",
)

READ_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")
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


def station_slug(station_name: str) -> str:
    """Normalize station names for manifest tracking."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", station_name.strip())
    return slug.strip("_")


def extract_year(text: str) -> Optional[int]:
    """Extract the first year token from text."""
    match = re.search(r"(19|20)\d{2}", text)
    if match:
        return int(match.group())
    return None


def schema_hash_from_columns(columns: List[str]) -> str:
    """Deterministically hash ordered column names."""
    payload = json.dumps({"columns": columns}, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_scanned_rows(
    csv_path: Path,
    max_scan_lines: int,
) -> Tuple[List[List[str]], str]:
    """Read a small scan window using fallback encodings."""
    last_error: Optional[Exception] = None
    for encoding in READ_ENCODINGS:
        try:
            with csv_path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.reader(handle)
                scanned_rows: List[List[str]] = []
                for index, row in enumerate(reader):
                    scanned_rows.append(row)
                    if index >= max_scan_lines:
                        break
            return scanned_rows, encoding
        except UnicodeDecodeError as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    raise ValueError("Unable to read file with supported encodings.")


def _normalize_header_columns(raw_columns: List[str]) -> List[str]:
    """Normalize and disambiguate header names for downstream validation/hash."""
    cleaned: List[str] = []
    for index, column in enumerate(raw_columns, start=1):
        text = column.strip()
        if not text:
            text = f"unnamed_{index}"
        cleaned.append(text)

    seen: Counter[str] = Counter()
    unique_columns: List[str] = []
    for column in cleaned:
        seen[column] += 1
        if seen[column] == 1:
            unique_columns.append(column)
        else:
            unique_columns.append(f"{column}__dup{seen[column] - 1}")

    return unique_columns


def detect_header_and_columns(csv_path: Path, max_scan_lines: int = 80) -> Tuple[int, List[str]]:
    """Detect CSV header row dynamically for both HOBOlink and ECCC files."""
    scanned_rows, _ = _read_scanned_rows(csv_path=csv_path, max_scan_lines=max_scan_lines)

    candidate_row = None
    candidate_columns: List[str] = []
    for index, row in enumerate(scanned_rows):
        stripped = [cell.strip() for cell in row]
        non_empty = [cell for cell in stripped if cell]
        if len(non_empty) < 2:
            continue

        joined = " ".join(non_empty).lower()
        has_date = "date" in joined
        has_time = "time" in joined
        if has_date and has_time:
            candidate_row = index
            candidate_columns = stripped
            break

    if candidate_row is None:
        best_score = -1
        for index, row in enumerate(scanned_rows):
            stripped = [cell.strip() for cell in row]
            non_empty = [cell for cell in stripped if cell]
            # Choose the densest non-empty row as a fallback header candidate.
            score = len(non_empty)
            if score >= 4 and score > best_score:
                best_score = score
                candidate_row = index
                candidate_columns = stripped

    if candidate_row is None:
        raise ValueError("Could not detect a valid header row within the first scan lines.")

    columns = _normalize_header_columns(candidate_columns)
    if not columns:
        raise ValueError("Header row was detected but no column names were found.")

    return candidate_row, columns


def validate_timestamp_columns(source: str, columns: List[str]) -> None:
    """Validate timestamp columns expected by each source."""
    lower_columns = {column.strip().lower() for column in columns}

    if source == "hobolink":
        has_split = "date" in lower_columns and "time" in lower_columns
        has_combined = any(
            token in lower_columns
            for token in ("date/time", "date time", "datetime", "timestamp")
        )
        if not has_split and not has_combined:
            raise ValueError("HOBOlink file is missing required Date and/or Time columns.")
        return

    if source == "eccc_fwi_daily":
        has_date_like = any("date" in column for column in lower_columns)
        if not has_date_like:
            raise ValueError("FWI daily file is missing a required date-like column.")
        return

    has_datetime = any("date/time" in column for column in lower_columns)
    has_date_and_time = "date" in lower_columns and "time" in lower_columns
    if not has_datetime and not has_date_and_time:
        raise ValueError("ECCC file is missing a Date/Time-equivalent timestamp column.")


def validate_required_fwi_columns(columns: List[str]) -> None:
    """Require daily FWI outputs to include date plus six official code columns."""
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
        raise ValueError(
            "FWI daily file missing required FWI code columns: " + ", ".join(missing)
        )


def inspect_csv_schema(csv_path: Path, source: str) -> Tuple[str, Optional[str], Optional[str], Optional[List[str]]]:
    """Run parse and schema checks against one CSV file."""
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
    status: str,
    error_message: Optional[str],
    schema_hash: Optional[str],
) -> Dict[str, object]:
    """Create a standard manifest row dictionary."""
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
        "ingested_at_utc": utc_now_iso(),
        "status": status,
        "error_message": error_message,
        "schema_hash": schema_hash,
    }


def infer_hobolink_years(files: Iterable[Path]) -> List[int]:
    """Infer available years from discovered HOBOlink files."""
    years: List[int] = []
    for csv_path in files:
        inferred = extract_year(csv_path.name)
        if inferred is None:
            inferred = extract_year(str(csv_path.parent))
        if inferred is not None:
            years.append(inferred)
    return sorted(set(years))


def classify_unchanged(
    file_path: Path,
    latest_index: Dict[str, Dict[str, object]],
) -> Tuple[bool, Optional[str]]:
    """Determine whether to skip expensive re-hashing for unchanged files."""
    previous = latest_index.get(str(file_path))
    if previous is None:
        return False, None

    prev_size = previous.get("size_bytes")
    prev_hash = previous.get("sha256")
    if pd.isna(prev_hash) or not str(prev_hash).strip():
        return False, None

    if int(prev_size) == file_path.stat().st_size:
        return True, str(prev_hash)
    return False, None


def discover_hobolink(
    dropzones: Dict[str, Path],
    latest_index: Dict[str, Dict[str, object]],
    logger,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[int]]:
    """Discover and validate HOBOlink files from expected station drop-zones."""
    manifest_rows: List[Dict[str, object]] = []
    schema_rows: List[Dict[str, object]] = []
    all_files: List[Path] = []

    for station_name, station_folder in dropzones.items():
        if not station_folder.exists():
            logger.warning("Missing HOBOlink drop-zone: %s", station_folder)
            continue

        csv_files = sorted(station_folder.rglob("*.csv"))
        all_files.extend(csv_files)
        for csv_file in csv_files:
            size_bytes = csv_file.stat().st_size
            year = extract_year(csv_file.name) or extract_year(str(csv_file.parent))
            period = str(year) if year is not None else "unknown"

            unchanged, cached_hash = classify_unchanged(csv_file, latest_index)
            if unchanged and cached_hash:
                sha256_value = cached_hash
                base_status = STATUS_SKIPPED_UNCHANGED
            else:
                sha256_value = compute_sha256(csv_file)
                base_status = STATUS_OK

            schema_status, error_message, schema_hash, columns = inspect_csv_schema(
                csv_file,
                "hobolink",
            )

            final_status = base_status if schema_status != STATUS_FAILED_READ else STATUS_FAILED_READ

            if schema_hash and columns:
                schema_rows.append(
                    {
                        "source": "hobolink",
                        "schema_hash": schema_hash,
                        "columns_json": json.dumps(columns, ensure_ascii=True),
                    }
                )

            manifest_rows.append(
                create_manifest_row(
                    source="hobolink",
                    station_raw=station_name,
                    year=year,
                    period=period,
                    file_path=csv_file,
                    size_bytes=size_bytes,
                    sha256_value=sha256_value,
                    status=final_status,
                    error_message=error_message,
                    schema_hash=schema_hash,
                )
            )

    years = infer_hobolink_years(all_files)
    return manifest_rows, schema_rows, years


def status_is_usable(status: object) -> bool:
    """Return True when a manifest status indicates readable data."""
    if pd.isna(status):
        return False
    text = str(status).strip().lower()
    if not text:
        return False
    return not text.startswith("failed_")


def find_hobolink_datetime_columns(columns: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Identify combined or split date/time columns from normalized header names."""
    lowered = {column.lower(): column for column in columns}

    combined_candidates = [
        "date/time",
        "date time",
        "datetime",
        "timestamp",
    ]
    for candidate in combined_candidates:
        if candidate in lowered:
            return lowered[candidate], None, None

    date_col = lowered.get("date")
    time_col = lowered.get("time")
    if date_col and time_col:
        return None, date_col, time_col

    return None, None, None


def read_hobolink_datetime_bounds(csv_path: Path) -> Optional[Tuple[date, date]]:
    """Read one HOBOlink CSV and return min/max observation dates found."""
    header_row, columns = detect_header_and_columns(csv_path)
    datetime_col, date_col, time_col = find_hobolink_datetime_columns(columns)
    if datetime_col is None and (date_col is None or time_col is None):
        return None

    last_error: Optional[Exception] = None
    for encoding in READ_ENCODINGS:
        try:
            df = pd.read_csv(
                csv_path,
                skiprows=header_row,
                encoding=encoding,
                engine="python",
            )
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
                parsed = pd.to_datetime(
                    combined,
                    errors="coerce",
                    utc=True,
                    format="mixed",
                )
            else:
                return None

            parsed = parsed.dropna()
            if parsed.empty:
                return None

            min_date = parsed.min().date()
            max_date = parsed.max().date()
            return min_date, max_date
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    return None


def derive_hobolink_coverage_bounds(
    hobolink_rows: List[Dict[str, object]],
    logger,
) -> Optional[Tuple[date, date]]:
    """Compute global min/max HOBOlink timestamps across usable files."""
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


def rh_from_dewpoint(temperature_c: float, dew_point_c: float) -> float:
    """Compute relative humidity (%) from temperature and dew point in Celsius."""
    return 100 * math.exp(
        (17.625 * dew_point_c / (243.04 + dew_point_c))
        - (17.625 * temperature_c / (243.04 + temperature_c))
    )


def ffmc_code(temp_c: float, rh_pct: float, wind_kmh: float, rain_mm: float, ffmc_prev: float) -> float:
    """Fine Fuel Moisture Code (Van Wagner 1987)."""
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

    ed = (
        0.942 * rh_pct**0.679
        + 11 * math.exp((rh_pct - 100) / 10)
        + 0.18 * (21.1 - temp_c) * (1 - math.exp(-0.115 * rh_pct))
    )
    ew = (
        0.618 * rh_pct**0.753
        + 10 * math.exp((rh_pct - 100) / 10)
        + 0.18 * (21.1 - temp_c) * (1 - math.exp(-0.115 * rh_pct))
    )
    if mo > ed:
        ko = 0.424 * (1 - (rh_pct / 100) ** 1.7) + 0.0694 * wind_kmh**0.5 * (1 - (rh_pct / 100) ** 8)
        kd = ko * 0.581 * math.exp(0.0365 * temp_c)
        m = ed + (mo - ed) * 10 ** (-kd)
    elif mo < ew:
        ko = (
            0.424 * (1 - ((100 - rh_pct) / 100) ** 1.7)
            + 0.0694 * wind_kmh**0.5 * (1 - ((100 - rh_pct) / 100) ** 8)
        )
        kw = ko * 0.581 * math.exp(0.0365 * temp_c)
        m = ew - (ew - mo) * 10 ** (-kw)
    else:
        m = mo

    return 59.5 * (250 - m) / (147.2 + m)


def dmc_code(temp_c: float, rh_pct: float, rain_mm: float, dmc_prev: float, month: int) -> float:
    """Duff Moisture Code (Van Wagner 1987)."""
    day_length_factors = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.8, 6.3]
    if temp_c < -1.1:
        temp_c = -1.1
    rk = 1.894 * (temp_c + 1.1) * (100 - rh_pct) * day_length_factors[month - 1] * 1e-4

    if rain_mm > 1.5:
        # Use canonical DMC effective rain term based on total daily rain.
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
    """Drought Code (Van Wagner 1987)."""
    seasonal_factors = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]
    if temp_c < -2.8:
        temp_c = -2.8
    pe = (0.36 * (temp_c + 2.8) + seasonal_factors[month - 1]) / 2
    if pe < 0:
        pe = 0

    if rain_mm > 2.8:
        # Use canonical DC effective rain term based on total daily rain.
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
    """Initial Spread Index from wind and FFMC."""
    fuel_moisture = 147.2 * (101 - ffmc_value) / (59.5 + ffmc_value)
    spread_factor = 19.115 * math.exp(-0.1386 * fuel_moisture) * (1 + fuel_moisture**5.31 / 4.93e7)
    return spread_factor * math.exp(0.05039 * wind_kmh)


def bui_index(dmc_value: float, dc_value: float) -> float:
    """Buildup Index from DMC and DC."""
    denominator = dmc_value + 0.4 * dc_value
    if denominator <= 0:
        return 0.0

    if dmc_value <= 0.4 * dc_value:
        return 0.8 * dmc_value * dc_value / denominator
    bui_value = dmc_value - (
        (1 - 0.8 * dc_value / denominator)
        * (0.92 + (0.0114 * dmc_value) ** 1.7)
    )
    return max(bui_value, 0)


def fwi_index(isi_value: float, bui_value: float) -> float:
    """Fire Weather Index from ISI and BUI."""
    if bui_value <= 80:
        bb = 0.1 * isi_value * (0.626 * bui_value**0.809 + 2)
    else:
        bb = 0.1 * isi_value * (1000 / (25 + 108.64 * math.exp(-0.023 * bui_value)))
    if bb <= 1:
        return bb
    return math.exp(2.72 * (0.434 * math.log(bb)) ** 0.647)


def fetch_hourly_range(climate_id: int | str, start_dt: str, end_dt: str, logger) -> List[Dict[str, object]]:
    """Fetch all hourly station records for a datetime range from MSC GeoMet API."""
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
                        f"Stanhope FWI API request failed at offset {offset} after "
                        f"{FWI_REQUEST_MAX_RETRIES} attempts: {exc}"
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
    """Parse LOCAL_DATE string from GeoMet payload into datetime."""
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
    """Build per-day noon weather inputs plus trailing 24h precipitation."""
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

        daily[str(day)] = {
            "t": temp_value,
            "h": rh_value,
            "w": wind_value,
            "p": precip_total,
        }

    return daily


def compute_fwi_daily_records(daily_inputs: Dict[str, Dict[str, float]]) -> List[Dict[str, object]]:
    """Compute daily FWI metrics sequentially from daily noon weather inputs."""
    ffmc_value = FFMC_START
    dmc_value = DMC_START
    dc_value = DC_START
    results: List[Dict[str, object]] = []

    for date_text in sorted(daily_inputs.keys()):
        current_day = date.fromisoformat(date_text)
        season_end = date(current_day.year, FWI_SEASON_END_MONTH, FWI_SEASON_END_DAY)
        if current_day > season_end:
            continue

        values = daily_inputs[date_text]
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
            }
        )

    return results


def get_fwi_date_bounds(csv_path: Path) -> Optional[Tuple[date, date]]:
    """Return min/max Date values from a computed daily FWI CSV, if available."""
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
    logger,
) -> Tuple[str, int, str, Optional[str], Optional[str], Optional[List[str]]]:
    """Compute and write one annual daily FWI CSV for Stanhope from hourly API data."""
    start_dt = f"{start_date:%Y-%m-%d}T00:00:00"
    end_dt = f"{end_date:%Y-%m-%d}T23:59:59"

    logger.info(
        "Computing Stanhope daily FWI for %s (%s to %s).",
        year,
        start_dt[:10],
        end_dt[:10],
    )

    try:
        records = fetch_hourly_range(
            climate_id=climate_id,
            start_dt=start_dt,
            end_dt=end_dt,
            logger=logger,
        )
    except Exception as exc:  # noqa: BLE001
        return (
            STATUS_FAILED_DOWNLOAD,
            0,
            "",
            f"FWI hourly API fetch failed: {exc}",
            None,
            None,
        )

    if not records:
        return (
            STATUS_FAILED_DOWNLOAD,
            0,
            "",
            "No hourly API records returned for requested period.",
            None,
            None,
        )

    daily_inputs = extract_daily_fwi_inputs(records)
    if not daily_inputs:
        return (
            STATUS_FAILED_READ,
            0,
            "",
            "No complete daily noon inputs found to compute FWI.",
            None,
            None,
        )

    results = compute_fwi_daily_records(daily_inputs)
    if not results:
        return (
            STATUS_FAILED_READ,
            0,
            "",
            "FWI computation produced no records.",
            None,
            None,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    output_min = results[0]["Date"]
    output_max = results[-1]["Date"]
    logger.info(
        "Stanhope daily FWI date bounds for %s output: min(Date)=%s max(Date)=%s",
        year,
        output_min,
        output_max,
    )

    size_bytes = destination.stat().st_size
    sha256_value = compute_sha256(destination)
    schema_status, error_message, schema_hash, columns = inspect_csv_schema(destination, "eccc_fwi_daily")
    if schema_status == STATUS_FAILED_READ:
        return STATUS_FAILED_READ, size_bytes, sha256_value, error_message, schema_hash, columns

    logger.info(
        "Stanhope daily FWI computed for %s: %s days saved to %s",
        year,
        len(results),
        destination,
    )
    return STATUS_DOWNLOADED, size_bytes, sha256_value, None, schema_hash, columns


def get_eccc_download_mode(climate_id: int, probe_year: int, logger) -> str:
    """Prefer annual downloads when supported by endpoint, else use monthly."""
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


def build_eccc_url(climate_id: int, year: int, month: Optional[int]) -> str:
    """Build ECCC bulk download URL for monthly or annual mode."""
    base = (
        "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"
        f"?format=csv&climate_id={climate_id}&Year={year}&timeframe=1"
    )
    if month is None:
        return base
    return f"{base}&Month={month}&Day=1"


def download_eccc_fwi_daily_periods(
    *,
    station_name: str,
    station_id: int | str,
    start_year: int,
    end_year: int,
    end_date: date,
    latest_index: Dict[str, Dict[str, object]],
    logger,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Compute annual daily Stanhope FWI files from observed hourly ECCC inputs."""
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
                logger.info(
                    "Recomputing %s because cached daily FWI has no valid Date bounds.",
                    destination.name,
                )
            else:
                first_date, last_date = bounds
                if first_date.month < FWI_SEASON_START_MONTH or last_date > season_end:
                    recompute_existing = True
                    logger.info(
                        "Recomputing %s because cached daily FWI date bounds (%s to %s) "
                        "violate seasonal window (%s to %s).",
                        destination.name,
                        first_date,
                        last_date,
                        year_start,
                        season_end,
                    )

        if destination.exists() and not recompute_existing:
            size_bytes = destination.stat().st_size
            unchanged, cached_hash = classify_unchanged(destination, latest_index)
            if unchanged and cached_hash:
                sha256_value = cached_hash
                base_status = STATUS_SKIPPED_UNCHANGED
            else:
                sha256_value = compute_sha256(destination)
                base_status = STATUS_OK

            schema_status, error_message, schema_hash, columns = inspect_csv_schema(
                destination,
                "eccc_fwi_daily",
            )
            final_status = base_status if schema_status != STATUS_FAILED_READ else STATUS_FAILED_READ
        else:
            final_status, size_bytes, sha256_value, error_message, schema_hash, columns = (
                compute_stanhope_fwi_daily_file(
                    station_name=station_name,
                    climate_id=station_id,
                    year=year,
                    start_date=year_start,
                    end_date=year_end,
                    destination=destination,
                    logger=logger,
                )
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


def download_eccc_periods(
    *,
    station_name: str,
    climate_id: int,
    start_year: int,
    end_year: int,
    latest_index: Dict[str, Dict[str, object]],
    logger,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Download or detect ECCC files incrementally within requested year range."""
    manifest_rows: List[Dict[str, object]] = []
    schema_rows: List[Dict[str, object]] = []

    mode = get_eccc_download_mode(climate_id, end_year, logger)
    periods: List[Tuple[int, Optional[int]]] = []
    for year in range(start_year, end_year + 1):
        if mode == "annual":
            periods.append((year, None))
        else:
            for month in range(1, 13):
                periods.append((year, month))

    for year, month in periods:
        period = f"{year}" if month is None else f"{year}-{month:02d}"
        file_name = f"ECCC_{station_slug(station_name)}_{period}_hourly.csv"
        destination = ECCC_CACHE_DIR / file_name

        if destination.exists():
            size_bytes = destination.stat().st_size
            unchanged, cached_hash = classify_unchanged(destination, latest_index)
            if unchanged and cached_hash:
                sha256_value = cached_hash
                base_status = STATUS_SKIPPED_UNCHANGED
            else:
                sha256_value = compute_sha256(destination)
                base_status = STATUS_OK

            schema_status, error_message, schema_hash, columns = inspect_csv_schema(
                destination,
                "eccc",
            )
            final_status = base_status if schema_status != STATUS_FAILED_READ else STATUS_FAILED_READ
        else:
            url = build_eccc_url(climate_id=climate_id, year=year, month=month)
            try:
                download_to_file_atomic(
                    url=url,
                    destination=destination,
                    csv_mime_hints=CSV_MIME_HINTS,
                )
                size_bytes = destination.stat().st_size
                sha256_value = compute_sha256(destination)
                schema_status, error_message, schema_hash, columns = inspect_csv_schema(
                    destination,
                    "eccc",
                )
                final_status = (
                    STATUS_DOWNLOADED
                    if schema_status != STATUS_FAILED_READ
                    else STATUS_FAILED_READ
                )
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


def parse_args() -> argparse.Namespace:
    """Parse command-line options for obtain execution."""
    parser = argparse.ArgumentParser(description="Run Step 2 obtain ingestion.")
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    """Run local HOBOlink discovery and incremental ECCC downloads."""
    args = parse_args()
    ensure_directories()
    log_path = LOGS_DIR / f"obtain_{datetime.now(timezone.utc):%Y%m%d}.log"
    logger = setup_logging("01_obtain", log_file_path=log_path)

    if not acquire_obtain_lock(OBTAIN_LOCK_PATH, logger):
        return 1

    try:
        logger.info("Step 2 Obtain started.")
        logger.info("ECCC Stanhope climate identifier: %s", ECCC_STANHOPE_CLIMATE_ID)
        logger.info("Stanhope daily FWI source: computed locally from ECCC hourly API observations.")

        hobolink_history = load_manifest(HOBOLINK_MANIFEST, MANIFEST_COLUMNS)
        eccc_history = load_manifest(ECCC_MANIFEST, MANIFEST_COLUMNS)
        eccc_fwi_history = load_manifest(ECCC_FWI_MANIFEST, MANIFEST_COLUMNS)
        hobolink_index = latest_manifest_index(hobolink_history)
        eccc_index = latest_manifest_index(eccc_history)
        eccc_fwi_index = latest_manifest_index(eccc_fwi_history)

        hobolink_dropzones = get_hobolink_dropzones()
        hobolink_rows, hobolink_schema_rows, _ = discover_hobolink(
            dropzones=hobolink_dropzones,
            latest_index=hobolink_index,
            logger=logger,
        )

        coverage_bounds = derive_hobolink_coverage_bounds(hobolink_rows, logger)
        if coverage_bounds is None:
            logger.error("Cannot compute Stanhope FWI window because HOBOlink coverage bounds were not found.")
            return 1
        hobolink_start, hobolink_end = coverage_bounds


        # Set default start year to 2021, end year to current UTC year if not provided
        default_start_year = 2021
        default_end_year = datetime.now(timezone.utc).year
        start_year = args.start_year if args.start_year is not None else default_start_year
        end_year = args.end_year if args.end_year is not None else default_end_year
        if start_year > end_year:
            logger.error("Invalid year range: start_year (%s) > end_year (%s).", start_year, end_year)
            return 1

        fwi_start_year = hobolink_start.year
        fwi_end_year = hobolink_end.year
        fwi_end_date = hobolink_end
        logger.info(
            "Stanhope FWI compute window forced to HOBOlink coverage: %s-01-01 to %s",
            fwi_start_year,
            fwi_end_date,
        )

        logger.info("ECCC period coverage: %s to %s", start_year, end_year)

        eccc_rows, eccc_schema_rows = download_eccc_periods(
            station_name=ECCC_STANHOPE_NAME,
            climate_id=ECCC_STANHOPE_CLIMATE_ID,
            start_year=start_year,
            end_year=end_year,
            latest_index=eccc_index,
            logger=logger,
        )

        eccc_fwi_rows, eccc_fwi_schema_rows = download_eccc_fwi_daily_periods(
            station_name=ECCC_STANHOPE_NAME,
            station_id=ECCC_STANHOPE_CLIMATE_ID,
            start_year=fwi_start_year,
            end_year=fwi_end_year,
            end_date=fwi_end_date,
            latest_index=eccc_fwi_index,
            logger=logger,
        )

        append_manifest_rows(HOBOLINK_MANIFEST, hobolink_rows, MANIFEST_COLUMNS)
        append_manifest_rows(ECCC_MANIFEST, eccc_rows, MANIFEST_COLUMNS)
        append_manifest_rows(ECCC_FWI_MANIFEST, eccc_fwi_rows, MANIFEST_COLUMNS)
        update_schema_inventory(
            hobolink_schema_rows + eccc_schema_rows + eccc_fwi_schema_rows,
            SCHEMA_INVENTORY,
            SCHEMA_COLUMNS,
        )

        hobolink_status = pd.Series([row["status"] for row in hobolink_rows]).value_counts().to_dict()
        eccc_status = pd.Series([row["status"] for row in eccc_rows]).value_counts().to_dict()
        eccc_fwi_status = pd.Series([row["status"] for row in eccc_fwi_rows]).value_counts().to_dict()

        logger.info("Summary: HOBOlink files processed = %s", len(hobolink_rows))
        logger.info("Summary: HOBOlink statuses = %s", hobolink_status)
        logger.info("Summary: ECCC periods processed = %s", len(eccc_rows))
        logger.info("Summary: ECCC statuses = %s", eccc_status)
        logger.info("Summary: ECCC FWI daily periods processed = %s", len(eccc_fwi_rows))
        logger.info("Summary: ECCC FWI daily statuses = %s", eccc_fwi_status)
        logger.info(
            "Next steps: run 02_scrub.py for UTC normalization, missing-value handling, and hourly resampling."
        )
        logger.info("Step 2 Obtain completed. Logs written to %s", log_path)
        return 0
    except KeyboardInterrupt:
        logger.error("Obtain interrupted before completion. Exiting cleanly.")
        return 130
    finally:
        release_obtain_lock(OBTAIN_LOCK_PATH, logger)


if __name__ == "__main__":
    raise SystemExit(main())
