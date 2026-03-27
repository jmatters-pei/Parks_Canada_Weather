"""Combined obtain + scrub pipeline entrypoint (self-contained).

This script intentionally duplicates key logic from Step 2 (obtain) and Step 3 (scrub)
without importing other repo modules, so it can run independently.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

# -----------------------------------------------------------------------------
# Copied/adapted configuration constants (do not import config.py)
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
STATUS_FAILED_READ = "failed_read"

READ_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")
NA_STRINGS = ["", "NA", "N/A", "na", "null", "NULL"]
HOURLY_FREQ = "h"

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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def station_slug(station_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(station_name).strip())
    return slug.strip("_")


def setup_logging(log_path: Path) -> logging.Logger:
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
    (scrubbed_dir / "_manifests").mkdir(parents=True, exist_ok=True)
    (outputs_dir / "logs").mkdir(parents=True, exist_ok=True)
    (outputs_dir / "figures").mkdir(parents=True, exist_ok=True)
    (outputs_dir / "tables").mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)


def compute_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def schema_hash_from_columns(columns: List[str]) -> str:
    payload = json.dumps({"columns": columns}, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_scanned_rows(csv_path: Path, max_scan_lines: int = 100) -> Tuple[List[List[str]], str]:
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
    lower_columns = {column.strip().lower() for column in columns}
    if source == "hobolink":
        has_split = "date" in lower_columns and "time" in lower_columns
        has_combined = any(token in lower_columns for token in ("date/time", "date time", "datetime", "timestamp"))
        if not has_split and not has_combined:
            raise ValueError("HOBOlink file is missing required Date and/or Time columns.")
        return
    has_datetime = any("date/time" in col for col in lower_columns)
    has_date_and_time = "date" in lower_columns and "time" in lower_columns
    if not has_datetime and not has_date_and_time:
        raise ValueError("ECCC file is missing a Date/Time-equivalent timestamp column.")


def inspect_csv_schema(csv_path: Path, source: str) -> Tuple[str, Optional[str], Optional[str], Optional[List[str]]]:
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


def extract_year(text: str) -> Optional[int]:
    match = re.search(r"(19|20)\d{2}", text)
    if match:
        return int(match.group())
    return None


def update_schema_inventory(schema_records: List[Dict[str, object]], schema_inventory_path: Path) -> None:
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


def run_obtain(raw_dir: Path, manifest_dir: Path, logger: logging.Logger, dry_run: bool) -> pd.DataFrame:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    schema_records: List[Dict[str, object]] = []

    hobolink_rows: List[Dict[str, object]] = []
    for station in HOBOLINK_STATIONS:
        station_dir = raw_dir / station
        if not station_dir.exists():
            logger.warning("Missing HOBOlink drop-zone: %s", station_dir)
            continue
        for csv_path in sorted(station_dir.rglob("*.csv")):
            size_bytes = csv_path.stat().st_size
            year = extract_year(csv_path.name) or extract_year(str(csv_path.parent))
            period = str(year) if year is not None else "unknown"
            schema_status, err, schema_hash, cols = inspect_csv_schema(csv_path, "hobolink")
            row = create_manifest_row(
                source="hobolink",
                station_raw=station,
                year=year,
                period=period,
                file_path=csv_path,
                size_bytes=size_bytes,
                sha256_value=compute_sha256(csv_path),
                status=schema_status,
                error_message=err,
                schema_hash=schema_hash,
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

    eccc_rows: List[Dict[str, object]] = []
    if ECCC_CACHE_DIR.exists():
        for csv_path in sorted(ECCC_CACHE_DIR.glob("*.csv")):
            size_bytes = csv_path.stat().st_size
            year = extract_year(csv_path.name)
            month_match = re.search(r"_(\d{4})-(\d{2})", csv_path.name)
            period = f"{month_match.group(1)}-{month_match.group(2)}" if month_match else (str(year) if year else "unknown")
            schema_status, err, schema_hash, cols = inspect_csv_schema(csv_path, "eccc")
            row = create_manifest_row(
                source="eccc",
                station_raw="Stanhope",
                year=year,
                period=period,
                file_path=csv_path,
                size_bytes=size_bytes,
                sha256_value=compute_sha256(csv_path),
                status=schema_status,
                error_message=err,
                schema_hash=schema_hash,
            )
            eccc_rows.append(row)
            if schema_hash and cols:
                schema_records.append(
                    {
                        "source": "eccc",
                        "schema_hash": schema_hash,
                        "columns_json": json.dumps(cols, ensure_ascii=True),
                    }
                )

    fwi_rows: List[Dict[str, object]] = []
    if ECCC_FWI_CACHE_DIR.exists():
        for csv_path in sorted(ECCC_FWI_CACHE_DIR.glob("*.csv")):
            size_bytes = csv_path.stat().st_size
            year = extract_year(csv_path.name)
            row = create_manifest_row(
                source="eccc_fwi_daily",
                station_raw="Stanhope",
                year=year,
                period=str(year) if year else "unknown",
                file_path=csv_path,
                size_bytes=size_bytes,
                sha256_value=compute_sha256(csv_path),
                status=STATUS_OK,
                error_message=None,
                schema_hash=None,
            )
            fwi_rows.append(row)

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


def normalize_header_columns(raw_columns: Iterable[str]) -> List[str]:
    return _normalize_header_columns(list(raw_columns))


def detect_hobolink_header_and_encoding(csv_path: Path, max_scan_lines: int = 120) -> Tuple[int, str]:
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
    banned_tokens = banned_tokens or []
    matches: List[str] = []
    for col in df.columns:
        low = col.lower()
        if all(token in low for token in required_tokens) and all(token not in low for token in banned_tokens):
            matches.append(col)
    return matches


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def choose_best_column(df: pd.DataFrame, candidates: List[str]) -> str | None:
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
    if df.empty:
        return df

    work = df.copy()
    work["datetime_utc"] = pd.to_datetime(work["datetime_utc"], utc=True, errors="coerce")
    work = work.dropna(subset=["datetime_utc"])
    work["datetime_utc"] = work["datetime_utc"].dt.floor(HOURLY_FREQ)

    wind_speed_col = CANONICAL_VARIABLES["wind_speed"]
    wind_dir_col = CANONICAL_VARIABLES["wind_dir"]

    valid_dir = work[wind_dir_col].notna() & work[wind_speed_col].notna() & (work[wind_speed_col] > 0)
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
    out = df.copy()
    for var in CANONICAL_ORDER:
        out[var] = pd.to_numeric(out[var], errors="coerce").astype("Float32")
        out[f"{var}_failed_qc"] = out[f"{var}_failed_qc"].astype("boolean")
        out[f"{var}_filled_short_gap"] = out[f"{var}_filled_short_gap"].astype("boolean")
    return out


def assert_output_schema(df: pd.DataFrame) -> None:
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
    if inventory.empty:
        raise ValueError("No usable input files found after obtain discovery.")

    inventory = inventory[inventory["status"] == STATUS_OK].copy()
    if inventory.empty:
        raise ValueError("All discovered files failed schema checks.")

    all_hourly_chunks: List[pd.DataFrame] = []
    precip_log_rows: List[Dict[str, object]] = []

    for _, manifest_row in inventory.iterrows():
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run combined obtain + scrub pipeline into scrubbed hourly outputs.",
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="Raw input directory root.")
    parser.add_argument("--scrubbed-dir", type=Path, default=SCRUBBED_DIR, help="Scrubbed output directory.")
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR, help="Outputs directory for logs/tables/figures.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs if present.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize without writing files.")
    return parser.parse_args()


def main() -> int:
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

    output_targets = [
        output_hourly,
        output_missingness,
        output_qc_counts,
        output_precip_log,
        manifest_dir / HOBOLINK_MANIFEST.name,
        manifest_dir / ECCC_MANIFEST.name,
        manifest_dir / ECCC_FWI_MANIFEST.name,
    ]

    if not args.force and not args.dry_run:
        existing = [path for path in output_targets if path.exists()]
        if existing:
            logger.error("Output files already exist. Re-run with --force to overwrite.")
            for path in existing:
                logger.error("Existing output: %s", path)
            return 2

    logger.info("Step A (obtain): discovering local files and building manifests.")
    inventory = run_obtain(raw_dir=raw_dir, manifest_dir=manifest_dir, logger=logger, dry_run=args.dry_run)

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
