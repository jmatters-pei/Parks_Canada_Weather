"""Step 3 scrub stage: hourly UTC normalization, QC, and short-gap filling."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from config import (
    CANONICAL_VARIABLES,
    LOGS_DIR,
    MANIFEST_DIR,
    SCRUBBED_DIR,
    ensure_directories,
    setup_logging,
)

HOBOLINK_MANIFEST = MANIFEST_DIR / "01_obtain_hobolink_files.csv"
ECCC_MANIFEST = MANIFEST_DIR / "01_obtain_eccc_periods.csv"
SCRUB_RUNS_MANIFEST = MANIFEST_DIR / "02_scrub_runs.csv"

OUTPUT_HOURLY = SCRUBBED_DIR / "02_hourly_weather_utc.csv"
OUTPUT_MISSINGNESS = SCRUBBED_DIR / "02_missingness_hourly_summary.csv"
OUTPUT_QC_COUNTS = SCRUBBED_DIR / "02_qc_out_of_range_counts.csv"
OUTPUT_PRECIP_LOG = SCRUBBED_DIR / "02_precip_semantics_log.csv"

READ_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")
NA_STRINGS = ["", "NA", "N/A", "na", "null", "NULL"]

HOURLY_FREQ = "h"

STATUS_BAD_PREFIXES = ("failed_",)

# QC thresholds and bounds (easy to tune)
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
    """Return the current UTC timestamp as ISO-8601 string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def station_slug(station_name: str) -> str:
    """Normalize station names to a slug used in keys/manifests."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", station_name.strip())
    return slug.strip("_")


def normalize_header_columns(raw_columns: Iterable[str]) -> List[str]:
    """Strip and deduplicate column names while preserving order."""
    cleaned: List[str] = []
    for index, column in enumerate(raw_columns, start=1):
        text = str(column).strip()
        if not text:
            text = f"unnamed_{index}"
        cleaned.append(text)

    seen: Dict[str, int] = {}
    unique: List[str] = []
    for column in cleaned:
        count = seen.get(column, 0) + 1
        seen[column] = count
        if count == 1:
            unique.append(column)
        else:
            unique.append(f"{column}__dup{count - 1}")

    return unique


def load_manifest(path: Path) -> pd.DataFrame:
    """Read a manifest CSV, returning an empty frame when missing."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def latest_manifest_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest row per file path using append order semantics."""
    if df.empty or "file_path" not in df.columns:
        return pd.DataFrame(columns=df.columns)
    latest = df.dropna(subset=["file_path"]).copy()
    latest["file_path"] = latest["file_path"].astype(str)
    return latest.drop_duplicates(subset=["file_path"], keep="last")


def status_is_usable(status: object) -> bool:
    """Return True for statuses that are safe to consume in scrub."""
    if pd.isna(status):
        return False
    text = str(status).strip().lower()
    if not text:
        return False
    return not any(text.startswith(prefix) for prefix in STATUS_BAD_PREFIXES)


def build_input_inventory(logger) -> pd.DataFrame:
    """Build latest valid file inventory from obtain manifests."""
    hobolink_df = latest_manifest_rows(load_manifest(HOBOLINK_MANIFEST))
    eccc_df = latest_manifest_rows(load_manifest(ECCC_MANIFEST))

    if not hobolink_df.empty:
        hobolink_df = hobolink_df[hobolink_df["status"].map(status_is_usable)].copy()
    if not eccc_df.empty:
        eccc_df = eccc_df[eccc_df["status"].map(status_is_usable)].copy()

    use_cols = [
        "source",
        "station_raw",
        "station_slug",
        "year",
        "period",
        "file_name",
        "file_path",
        "sha256",
        "status",
    ]
    inventory = pd.concat([hobolink_df, eccc_df], ignore_index=True)
    if inventory.empty:
        return inventory

    for col in use_cols:
        if col not in inventory.columns:
            inventory[col] = pd.NA
    inventory = inventory[use_cols].copy()

    logger.info("Input inventory size: %s files", len(inventory))
    source_counts = inventory["source"].value_counts(dropna=False).to_dict()
    station_counts = inventory["station_slug"].value_counts(dropna=False).to_dict()
    period_counts = inventory.groupby(["source", "period"], dropna=False).size().to_dict()
    logger.info("Inventory by source: %s", source_counts)
    logger.info("Inventory by station: %s", station_counts)
    logger.info("Inventory by source/period: %s", period_counts)

    return inventory


def manifest_signature(latest_inventory: pd.DataFrame) -> str:
    """Compute deterministic signature of latest obtain rows used in scrub."""
    if latest_inventory.empty:
        return ""
    subset = latest_inventory[["source", "file_path", "sha256", "status"]].fillna("")
    ordered = subset.sort_values(["source", "file_path"]).to_dict(orient="records")
    payload = json.dumps(ordered, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def detect_hobolink_header_and_encoding(csv_path: Path, max_scan_lines: int = 120) -> Tuple[int, str]:
    """Detect HOBOlink header row and matching encoding with fallbacks."""
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
                non_empty = [cell.strip() for cell in row if str(cell).strip()]
                score = len(non_empty)
                if score >= 4 and score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx >= 0:
                return best_idx, encoding
        except UnicodeDecodeError:
            continue

    raise ValueError("Unable to detect HOBOlink header row with supported encodings.")


def find_column(df: pd.DataFrame, required_tokens: List[str], banned_tokens: List[str] | None = None) -> List[str]:
    """Return matching columns that include all required tokens and no banned tokens."""
    banned_tokens = banned_tokens or []
    matches: List[str] = []
    for col in df.columns:
        low = col.lower()
        if all(token in low for token in required_tokens) and all(
            token not in low for token in banned_tokens
        ):
            matches.append(col)
    return matches


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Convert a series to numeric with coercion for parser robustness."""
    return pd.to_numeric(df[column], errors="coerce")


def choose_best_column(df: pd.DataFrame, candidates: List[str]) -> str | None:
    """Choose best candidate by non-null fraction of numeric parse."""
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


def parse_hobolink_file(path: Path, row: pd.Series, logger) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    """Read, normalize, and canonicalize one HOBOlink raw CSV file."""
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
        # Fast-path: HOBOlink exports commonly look like:
        #   Date="11/01/2022", Time="00:00:00 -0300"
        # Using an explicit format is dramatically faster than dateutil inference.
        dt_text = raw["Date"].astype(str).str.strip() + " " + raw["Time"].astype(str).str.strip()
        timestamp = pd.to_datetime(
            dt_text,
            format="%m/%d/%Y %H:%M:%S %z",
            errors="coerce",
            utc=True,
        )
        # Fallback for any non-standard rows/files.
        if float(timestamp.isna().mean()) > 0.05:
            timestamp = pd.to_datetime(dt_text, errors="coerce", utc=True, cache=True)
    else:
        datetime_cols = [c for c in raw.columns if "date/time" in c.lower() or "datetime" in c.lower()]
        if not datetime_cols:
            raise ValueError("HOBOlink file does not contain parseable Date/Time columns.")
        timestamp = pd.to_datetime(raw[datetime_cols[0]], errors="coerce", utc=True)

    mapped: Dict[str, pd.Series] = {}
    mapping_log: List[Dict[str, object]] = []

    temp_candidates = find_column(raw, ["temperature", "°c"], banned_tokens=["water", "dew point"])
    rh_candidates = find_column(raw, ["rh"], banned_tokens=["threshold"])
    wind_speed_kmh_candidates = find_column(
        raw,
        ["wind", "speed", "km/h"],
        banned_tokens=["gust"],
    )
    wind_speed_ms_candidates = find_column(
        raw,
        ["wind", "speed", "m/s"],
        banned_tokens=["gust"],
    )
    wind_dir_candidates = find_column(raw, ["wind direction"])
    rain_candidates = [
        c
        for c in raw.columns
        if "rain" in c.lower() and "mm" in c.lower() and "accumulated" not in c.lower()
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

    duplicate_count = int(
        out.duplicated(subset=["datetime_utc", "station_slug", "source"] + CANONICAL_ORDER).sum()
    )
    if duplicate_count > 0:
        logger.info("Detected %s exact duplicate raw rows in %s", duplicate_count, path)

    return out, mapping_log


def parse_eccc_file(path: Path, row: pd.Series) -> pd.DataFrame:
    """Read and canonicalize one ECCC CSV file."""
    raw = pd.read_csv(
        path,
        na_values=NA_STRINGS,
        keep_default_na=True,
        low_memory=False,
    )

    if "Date/Time (LST)" not in raw.columns:
        raise ValueError("ECCC file is missing Date/Time (LST) column.")

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
    out = out.sort_values("datetime_utc").reset_index(drop=True)
    return out


def classify_and_normalize_precip(series: pd.Series, source: str) -> Tuple[pd.Series, Dict[str, object]]:
    """Classify precipitation semantics and normalize to incremental mm."""
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
    """Aggregate sub-hourly records to hourly UTC using variable-specific rules."""
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
            precipitation_mm=(CANONICAL_VARIABLES["rain"], lambda s: s.sum(min_count=1)),
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
    """Return run lengths for True segments in a boolean mask."""
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
    """Identify interior missing runs of length <= max_gap."""
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
    """Invalidate abrupt one-hour jumps while skipping across missing values."""
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
    """Apply bounds QC, rate-of-change QC, and short-gap fill policy per station."""
    df = station_df.copy()
    df = df.sort_values("datetime_utc").reset_index(drop=True)

    qc_rows: List[Dict[str, object]] = []

    for var in CANONICAL_ORDER:
        df[f"{var}_failed_qc"] = pd.Series(False, index=df.index, dtype="boolean")
        df[f"{var}_filled_short_gap"] = pd.Series(False, index=df.index, dtype="boolean")

    # Bounds QC first
    temp_col = CANONICAL_VARIABLES["temp"]
    temp_invalid = (df[temp_col] < TEMP_BOUNDS[0]) | (df[temp_col] > TEMP_BOUNDS[1])
    df.loc[temp_invalid, temp_col] = pd.NA
    df.loc[temp_invalid, f"{temp_col}_failed_qc"] = True

    rh_col = CANONICAL_VARIABLES["rh"]
    rh_negative = df[rh_col] < RH_MIN
    rh_soft_cap = (df[rh_col] > RH_CAP_MAX) & (df[rh_col] <= RH_SOFT_MAX)
    rh_extreme = df[rh_col] > RH_SOFT_MAX
    df.loc[rh_negative | rh_extreme, rh_col] = pd.NA
    df.loc[rh_negative | rh_extreme, f"{rh_col}_failed_qc"] = True
    df.loc[rh_soft_cap, rh_col] = RH_CAP_MAX

    wind_speed_col = CANONICAL_VARIABLES["wind_speed"]
    ws_invalid = df[wind_speed_col] < WIND_SPEED_MIN
    df.loc[ws_invalid, wind_speed_col] = pd.NA
    df.loc[ws_invalid, f"{wind_speed_col}_failed_qc"] = True

    wind_dir_col = CANONICAL_VARIABLES["wind_dir"]
    wd_invalid = (df[wind_dir_col] < WIND_DIR_BOUNDS[0]) | (df[wind_dir_col] > WIND_DIR_BOUNDS[1])
    df.loc[wd_invalid, wind_dir_col] = pd.NA
    df.loc[wd_invalid, f"{wind_dir_col}_failed_qc"] = True
    df[wind_dir_col] = pd.to_numeric(df[wind_dir_col], errors="coerce") % 360.0

    rain_col = CANONICAL_VARIABLES["rain"]
    rain_negative = df[rain_col] < 0
    df.loc[rain_negative, rain_col] = pd.NA
    df.loc[rain_negative, f"{rain_col}_failed_qc"] = True

    # Rate-of-change filters on hourly station series
    for var, threshold in STEP_THRESHOLDS.items():
        filtered, failed = apply_step_filter(df[var], threshold)
        df[var] = filtered
        df.loc[failed, f"{var}_failed_qc"] = True

    # Fill short gaps for continuous variables with linear interpolation
    for var in CONTINUOUS_FILL_VARS:
        before_missing = df[var].isna()
        filled = pd.to_numeric(df[var], errors="coerce").interpolate(
            method="linear",
            limit=2,
            limit_area="inside",
        )
        df[var] = filled
        after_filled = before_missing & df[var].notna()
        df.loc[after_filled, f"{var}_filled_short_gap"] = True

    # Wind direction short-gap fill (nearest), but never through calm wind hours.
    dir_missing_eligible = short_gap_mask(df[wind_dir_col], max_gap=2)
    dir_nearest = pd.to_numeric(df[wind_dir_col], errors="coerce").interpolate(
        method="nearest",
        limit=2,
        limit_area="inside",
    )
    dir_fill_mask = (
        dir_missing_eligible
        & df[wind_dir_col].isna()
        & dir_nearest.notna()
        & (df[wind_speed_col] > 0)
    )
    df.loc[dir_fill_mask, wind_dir_col] = dir_nearest[dir_fill_mask]
    df.loc[dir_fill_mask, f"{wind_dir_col}_filled_short_gap"] = True

    # Do not carry direction through calm hours.
    calm_mask = (df[wind_speed_col] == 0) | df[wind_speed_col].isna()
    df.loc[calm_mask, wind_dir_col] = pd.NA

    # Re-apply simple bounds after fill.
    post_temp_invalid = (df[temp_col] < TEMP_BOUNDS[0]) | (df[temp_col] > TEMP_BOUNDS[1])
    df.loc[post_temp_invalid, temp_col] = pd.NA
    df.loc[post_temp_invalid, f"{temp_col}_failed_qc"] = True

    post_rh_negative = df[rh_col] < RH_MIN
    post_rh_soft_cap = (df[rh_col] > RH_CAP_MAX) & (df[rh_col] <= RH_SOFT_MAX)
    post_rh_extreme = df[rh_col] > RH_SOFT_MAX
    df.loc[post_rh_negative | post_rh_extreme, rh_col] = pd.NA
    df.loc[post_rh_negative | post_rh_extreme, f"{rh_col}_failed_qc"] = True
    df.loc[post_rh_soft_cap, rh_col] = RH_CAP_MAX

    post_ws_invalid = df[wind_speed_col] < WIND_SPEED_MIN
    df.loc[post_ws_invalid, wind_speed_col] = pd.NA
    df.loc[post_ws_invalid, f"{wind_speed_col}_failed_qc"] = True

    post_wd_invalid = (df[wind_dir_col] < WIND_DIR_BOUNDS[0]) | (df[wind_dir_col] > WIND_DIR_BOUNDS[1])
    df.loc[post_wd_invalid, wind_dir_col] = pd.NA
    df.loc[post_wd_invalid, f"{wind_dir_col}_failed_qc"] = True
    df[wind_dir_col] = pd.to_numeric(df[wind_dir_col], errors="coerce") % 360.0

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
    """Expand station data to complete hourly UTC index from min to max."""
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

    merged = grid.merge(
        station_df,
        on=["station_raw", "station_slug", "source", "datetime_utc"],
        how="left",
    )
    return merged


def summarize_missingness(station_df: pd.DataFrame) -> List[Dict[str, object]]:
    """Compute per-variable missingness summaries for one station."""
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
    """Apply memory-efficient nullable dtypes for numeric and flag columns."""
    out = df.copy()
    for var in CANONICAL_ORDER:
        out[var] = pd.to_numeric(out[var], errors="coerce").astype("Float32")
        out[f"{var}_failed_qc"] = out[f"{var}_failed_qc"].astype("boolean")
        out[f"{var}_filled_short_gap"] = out[f"{var}_filled_short_gap"].astype("boolean")
    return out


def assert_output_schema(df: pd.DataFrame) -> None:
    """Run required schema and key integrity checks before writing outputs."""
    required_columns = ["station_raw", "station_slug", "source", "datetime_utc"]
    required_columns.extend(CANONICAL_ORDER)
    for var in CANONICAL_ORDER:
        required_columns.append(f"{var}_failed_qc")
        required_columns.append(f"{var}_filled_short_gap")

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in scrub output: {missing}")

    dupes = df.duplicated(subset=["station_slug", "datetime_utc"])
    if bool(dupes.any()):
        count = int(dupes.sum())
        raise ValueError(f"Duplicate station/hour keys detected: {count}")


def append_scrub_run_manifest(
    *,
    run_id: str,
    run_timestamp_utc: str,
    inventory: pd.DataFrame,
    output_path: Path,
    output_df: pd.DataFrame,
    obtain_signature: str,
) -> None:
    """Append run-level scrub metadata for reproducibility."""
    station_ranges = (
        output_df.groupby("station_slug", dropna=False)["datetime_utc"]
        .agg(["min", "max"])
        .reset_index()
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

    row = {
        "run_timestamp_utc": run_timestamp_utc,
        "run_id": run_id,
        "input_hobolink_files": int((inventory["source"] == "hobolink").sum()),
        "input_eccc_files": int((inventory["source"] == "eccc").sum()),
        "obtain_manifest_signature": obtain_signature,
        "output_file_path": str(output_path),
        "output_rows": int(len(output_df)),
        "station_ranges_json": json.dumps(station_range_payload, ensure_ascii=True),
    }

    new_df = pd.DataFrame([row])
    if SCRUB_RUNS_MANIFEST.exists():
        existing = pd.read_csv(SCRUB_RUNS_MANIFEST)
        merged = pd.concat([existing, new_df], ignore_index=True)
    else:
        merged = new_df
    merged.to_csv(SCRUB_RUNS_MANIFEST, index=False)


def main() -> int:
    """Run scrub stage end-to-end and write hourly UTC outputs + QA tables."""
    ensure_directories()
    log_path = LOGS_DIR / f"scrub_{datetime.now(timezone.utc):%Y%m%d}.log"
    logger = setup_logging("02_scrub", log_file_path=log_path)

    logger.info("Step 3 Scrub started.")

    inventory = build_input_inventory(logger)
    if inventory.empty:
        logger.error("No usable input files found in obtain manifests.")
        return 1

    obtain_sig = manifest_signature(inventory)
    logger.info("Obtain latest-row signature: %s", obtain_sig)

    all_hourly_chunks: List[pd.DataFrame] = []
    precip_log_rows: List[Dict[str, object]] = []
    mapping_logs: List[Dict[str, object]] = []

    for _, manifest_row in inventory.iterrows():
        source = str(manifest_row.get("source", "")).strip().lower()
        file_path = Path(str(manifest_row.get("file_path", "")))
        if not file_path.exists():
            logger.warning("Skipping missing file listed in manifest: %s", file_path)
            continue

        try:
            if source == "hobolink":
                parsed, per_file_mapping_log = parse_hobolink_file(file_path, manifest_row, logger)
                mapping_logs.extend(per_file_mapping_log)
            elif source == "eccc":
                parsed = parse_eccc_file(file_path, manifest_row)
            else:
                logger.warning("Skipping unknown source '%s' for %s", source, file_path)
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
            logger.info(
                "Precip semantics %s | station=%s | file=%s | resets=%s | neg_diffs=%s",
                precip_meta["interpretation"],
                parsed["station_slug"].iloc[0],
                file_path.name,
                precip_meta["reset_count"],
                precip_meta["negative_diff_count"],
            )

            hourly = aggregate_hourly(parsed)
            all_hourly_chunks.append(hourly)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed processing file %s: %s", file_path, exc)
            continue

    if not all_hourly_chunks:
        logger.error("No hourly data generated from input inventory.")
        return 1

    hourly_all = pd.concat(all_hourly_chunks, ignore_index=True)
    merge_work = hourly_all.copy()
    merge_work["_dir_valid"] = (
        merge_work[CANONICAL_VARIABLES["wind_dir"]].notna()
        & merge_work[CANONICAL_VARIABLES["wind_speed"]].notna()
        & (merge_work[CANONICAL_VARIABLES["wind_speed"]] > 0)
    )
    merge_rad = pd.to_numeric(merge_work[CANONICAL_VARIABLES["wind_dir"]], errors="coerce").map(math.radians)
    merge_work["_wind_x"] = pd.NA
    merge_work["_wind_y"] = pd.NA
    merge_work.loc[merge_work["_dir_valid"], "_wind_x"] = (
        merge_work.loc[merge_work["_dir_valid"], CANONICAL_VARIABLES["wind_speed"]]
        * merge_rad[merge_work["_dir_valid"]].map(math.cos)
    )
    merge_work.loc[merge_work["_dir_valid"], "_wind_y"] = (
        merge_work.loc[merge_work["_dir_valid"], CANONICAL_VARIABLES["wind_speed"]]
        * merge_rad[merge_work["_dir_valid"]].map(math.sin)
    )

    hourly_all = (
        merge_work.groupby(["station_raw", "station_slug", "source", "datetime_utc"], as_index=False)
        .agg(
            air_temperature_c=(CANONICAL_VARIABLES["temp"], "mean"),
            relative_humidity_pct=(CANONICAL_VARIABLES["rh"], "mean"),
            wind_speed_kmh=(CANONICAL_VARIABLES["wind_speed"], "mean"),
            precipitation_mm=(CANONICAL_VARIABLES["rain"], lambda s: s.sum(min_count=1)),
            _wind_x=("_wind_x", "sum"),
            _wind_y=("_wind_y", "sum"),
            _dir_obs=("_dir_valid", "sum"),
        )
        .sort_values(["station_slug", "datetime_utc"])
        .reset_index(drop=True)
    )

    merged_dir_values: List[float] = []
    for _, row in hourly_all.iterrows():
        if row["_dir_obs"] <= 0:
            merged_dir_values.append(float("nan"))
            continue

        x_val = row["_wind_x"]
        y_val = row["_wind_y"]
        if pd.isna(x_val) or pd.isna(y_val) or (float(x_val) == 0.0 and float(y_val) == 0.0):
            merged_dir_values.append(float("nan"))
            continue

        angle = math.degrees(math.atan2(float(y_val), float(x_val)))
        merged_dir_values.append(angle % 360.0)

    hourly_all[CANONICAL_VARIABLES["wind_dir"]] = pd.Series(merged_dir_values, dtype="Float64")
    hourly_all = hourly_all.drop(columns=["_wind_x", "_wind_y", "_dir_obs"])

    station_outputs: List[pd.DataFrame] = []
    missingness_rows: List[Dict[str, object]] = []
    qc_rows: List[Dict[str, object]] = []

    for station_slug_key, station_df in hourly_all.groupby("station_slug", dropna=False):
        station_df = station_df.sort_values("datetime_utc").reset_index(drop=True)
        expanded = build_complete_hourly_grid(station_df)
        cleaned, station_qc_rows = apply_qc_and_fill(expanded)

        station_outputs.append(cleaned)
        missingness_rows.extend(summarize_missingness(cleaned))
        qc_rows.extend(station_qc_rows)

        for row in station_qc_rows:
            logger.info(
                "QC summary | station=%s | var=%s | failed_qc=%s | filled_short_gap=%s",
                station_slug_key,
                row["variable"],
                row["failed_qc_count"],
                row["filled_short_gap_count"],
            )

    final_df = pd.concat(station_outputs, ignore_index=True)
    final_df = cast_output_dtypes(final_df)
    final_df = final_df.sort_values(["station_slug", "datetime_utc"]).reset_index(drop=True)

    assert_output_schema(final_df)

    # Write datetime as ISO-8601 UTC string in CSV output.
    write_df = final_df.copy()
    write_df["datetime_utc"] = pd.to_datetime(write_df["datetime_utc"], utc=True).dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    write_df.to_csv(OUTPUT_HOURLY, index=False)
    pd.DataFrame(missingness_rows).to_csv(OUTPUT_MISSINGNESS, index=False)
    pd.DataFrame(qc_rows).to_csv(OUTPUT_QC_COUNTS, index=False)
    pd.DataFrame(precip_log_rows).to_csv(OUTPUT_PRECIP_LOG, index=False)

    run_timestamp = utc_now_iso()
    run_id = f"scrub_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    append_scrub_run_manifest(
        run_id=run_id,
        run_timestamp_utc=run_timestamp,
        inventory=inventory,
        output_path=OUTPUT_HOURLY,
        output_df=final_df,
        obtain_signature=obtain_sig,
    )

    if mapping_logs:
        logger.info("HOBOlink column mapping samples: %s", mapping_logs[:20])

    logger.info("Wrote scrub output: %s", OUTPUT_HOURLY)
    logger.info("Wrote missingness summary: %s", OUTPUT_MISSINGNESS)
    logger.info("Wrote QC summary: %s", OUTPUT_QC_COUNTS)
    logger.info("Wrote precip semantics log: %s", OUTPUT_PRECIP_LOG)
    logger.info("Wrote scrub run manifest: %s", SCRUB_RUNS_MANIFEST)
    logger.info(
        "Summary: generated %s station-hour rows across %s stations.",
        len(final_df),
        final_df["station_slug"].nunique(dropna=True),
    )
    logger.info("Next steps: run 03_explore.py for QA visualization and downstream analysis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
