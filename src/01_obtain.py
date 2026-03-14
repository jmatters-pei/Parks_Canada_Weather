"""Step 2 obtain stage for local HOBOlink discovery and ECCC ingestion."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

from config import (
    ECCC_CACHE_DIR,
    ECCC_STANHOPE_CLIMATE_ID,
    ECCC_STANHOPE_NAME,
    LOGS_DIR,
    MANIFEST_DIR,
    get_hobolink_dropzones,
    ensure_directories,
    setup_logging,
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

HOBOLINK_MANIFEST = MANIFEST_DIR / "obtain_hobolink_files.csv"
ECCC_MANIFEST = MANIFEST_DIR / "obtain_eccc_periods.csv"
SCHEMA_INVENTORY = MANIFEST_DIR / "schema_inventory.csv"

CSV_MIME_HINTS = (
    "text/csv",
    "application/csv",
    "application/octet-stream",
    "application/vnd.ms-excel",
    "application/force-download",
    "text/plain",
)

READ_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


def utc_now_iso() -> str:
    """Return UTC timestamp as ISO 8601 string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def compute_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 for a file."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    """Load a manifest or return an empty dataframe with standard columns."""
    if manifest_path.exists():
        return pd.read_csv(manifest_path)
    return pd.DataFrame(columns=MANIFEST_COLUMNS)


def append_manifest_rows(manifest_path: Path, rows: List[Dict[str, object]]) -> None:
    """Append new rows to a manifest without dropping historical records."""
    if not rows:
        return

    new_df = pd.DataFrame(rows)
    for column in MANIFEST_COLUMNS:
        if column not in new_df.columns:
            new_df[column] = None
    new_df = new_df[MANIFEST_COLUMNS]

    if manifest_path.exists():
        existing_df = pd.read_csv(manifest_path)
        output_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        output_df = new_df

    output_df.to_csv(manifest_path, index=False)


def latest_manifest_index(manifest_df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    """Create a latest-by-file-path index for incremental checks."""
    latest: Dict[str, Dict[str, object]] = {}
    if manifest_df.empty:
        return latest

    for row in manifest_df.to_dict(orient="records"):
        file_path = str(row.get("file_path", "")).strip()
        if file_path:
            latest[file_path] = row
    return latest


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

    has_datetime = any("date/time" in column for column in lower_columns)
    has_date_and_time = "date" in lower_columns and "time" in lower_columns
    if not has_datetime and not has_date_and_time:
        raise ValueError("ECCC file is missing a Date/Time-equivalent timestamp column.")


def inspect_csv_schema(csv_path: Path, source: str) -> Tuple[str, Optional[str], Optional[str], Optional[List[str]]]:
    """Run parse and schema checks against one CSV file."""
    if not csv_path.exists() or csv_path.stat().st_size <= 0:
        return STATUS_FAILED_READ, "File does not exist or is empty.", None, None

    try:
        _, columns = detect_header_and_columns(csv_path)
        validate_timestamp_columns(source, columns)
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


def update_schema_inventory(schema_records: List[Dict[str, object]]) -> None:
    """Merge schema records into schema inventory while preserving history."""
    if not schema_records:
        return

    now_utc = utc_now_iso()
    new_df = pd.DataFrame(schema_records)
    if new_df.empty:
        return

    grouped = (
        new_df.groupby(["source", "schema_hash", "columns_json"], dropna=False)
        .size()
        .reset_index(name="seen_count")
    )
    grouped["first_seen_utc"] = now_utc
    grouped["last_seen_utc"] = now_utc
    grouped = grouped[SCHEMA_COLUMNS]

    if SCHEMA_INVENTORY.exists():
        existing = pd.read_csv(SCHEMA_INVENTORY)
        for column in SCHEMA_COLUMNS:
            if column not in existing.columns:
                existing[column] = pd.NA
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
    merged.to_csv(SCHEMA_INVENTORY, index=False)


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


def response_looks_like_csv(response: requests.Response) -> bool:
    """Quickly check if HTTP response appears to be CSV."""
    content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
    if "text/html" in content_type:
        return False
    if content_type and content_type not in CSV_MIME_HINTS:
        # Fall back to content sniffing for non-standard download MIME types.
        pass

    probe = response.text.lstrip("\ufeff\n\r\t ")
    probe_head = probe[:500]
    if "<!doctype html" in probe_head.lower() or "<html" in probe_head.lower():
        return False

    first_line = probe_head.splitlines()[0] if probe_head.splitlines() else ""
    if "," not in first_line:
        return False

    csv_markers = ("date/time", "station name", "year", "month", "day")
    marker_hits = sum(marker in probe_head.lower() for marker in csv_markers)
    return marker_hits >= 2


def download_to_file_atomic(url: str, destination: Path, timeout_seconds: int = 60) -> None:
    """Download text content atomically via temp file then move into place."""
    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()

    if not response_looks_like_csv(response):
        raise ValueError("Response did not look like CSV payload.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", delete=False, dir=destination.parent, encoding="utf-8", newline="") as tmp:
        tmp.write(response.text)
        temp_path = Path(tmp.name)

    temp_path.replace(destination)


def get_eccc_download_mode(climate_id: int, probe_year: int, logger) -> str:
    """Prefer annual downloads when supported by endpoint, else use monthly."""
    annual_url = (
        "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"
        f"?format=csv&climate_id={climate_id}&Year={probe_year}&timeframe=1"
    )
    try:
        response = requests.get(annual_url, timeout=30)
        if response.status_code == 200 and response_looks_like_csv(response):
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
                download_to_file_atomic(url=url, destination=destination)
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

    logger.info("Step 2 Obtain started.")
    logger.info("ECCC Stanhope climate identifier: %s", ECCC_STANHOPE_CLIMATE_ID)

    hobolink_history = load_manifest(HOBOLINK_MANIFEST)
    eccc_history = load_manifest(ECCC_MANIFEST)
    hobolink_index = latest_manifest_index(hobolink_history)
    eccc_index = latest_manifest_index(eccc_history)

    hobolink_dropzones = get_hobolink_dropzones()
    hobolink_rows, hobolink_schema_rows, hobolink_years = discover_hobolink(
        dropzones=hobolink_dropzones,
        latest_index=hobolink_index,
        logger=logger,
    )


    # Set default start year to 2021, end year to current UTC year if not provided
    default_start_year = 2021
    default_end_year = datetime.now(timezone.utc).year
    start_year = args.start_year if args.start_year is not None else default_start_year
    end_year = args.end_year if args.end_year is not None else default_end_year
    if start_year > end_year:
        logger.error("Invalid year range: start_year (%s) > end_year (%s).", start_year, end_year)
        return 1

    logger.info("ECCC period coverage: %s to %s", start_year, end_year)

    eccc_rows, eccc_schema_rows = download_eccc_periods(
        station_name=ECCC_STANHOPE_NAME,
        climate_id=ECCC_STANHOPE_CLIMATE_ID,
        start_year=start_year,
        end_year=end_year,
        latest_index=eccc_index,
        logger=logger,
    )

    append_manifest_rows(HOBOLINK_MANIFEST, hobolink_rows)
    append_manifest_rows(ECCC_MANIFEST, eccc_rows)
    update_schema_inventory(hobolink_schema_rows + eccc_schema_rows)

    hobolink_status = pd.Series([row["status"] for row in hobolink_rows]).value_counts().to_dict()
    eccc_status = pd.Series([row["status"] for row in eccc_rows]).value_counts().to_dict()

    logger.info("Summary: HOBOlink files processed = %s", len(hobolink_rows))
    logger.info("Summary: HOBOlink statuses = %s", hobolink_status)
    logger.info("Summary: ECCC periods processed = %s", len(eccc_rows))
    logger.info("Summary: ECCC statuses = %s", eccc_status)
    logger.info(
        "Next steps: run 02_scrub.py for UTC normalization, missing-value handling, and hourly resampling."
    )
    logger.info("Step 2 Obtain completed. Logs written to %s", log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
