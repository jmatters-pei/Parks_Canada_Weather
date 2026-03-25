"""Step 5 model stage: daily FWI modeling, segmentation, and Stanhope validation."""

from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from config import (
    CANONICAL_VARIABLES,
    ECCC_FWI_CACHE_DIR,
    LOGS_DIR,
    SCRUBBED_DIR,
    TABLES_DIR,
    ensure_directories,
    setup_logging,
)

INPUT_HOURLY = SCRUBBED_DIR / "02_hourly_weather_utc.csv"
REFERENCE_FWI_DIR = ECCC_FWI_CACHE_DIR
OUTPUT_MODEL_DAILY = TABLES_DIR / "04_model_fwi_daily.csv"
OUTPUT_VALIDATION = TABLES_DIR / "04_model_fwi_validation_summary.csv"

HALIFAX_TZ = ZoneInfo("America/Halifax")
UTC_TZ = ZoneInfo("UTC")

SEASON_START_MONTH = 6
SEASON_START_DAY = 1
SEASON_END_MONTH = 9
SEASON_END_DAY = 30

FFMC_START = 85.0
DMC_START = 6.0
DC_START = 15.0

CORE_NOON_VARS = [
    CANONICAL_VARIABLES["temp"],
    CANONICAL_VARIABLES["rh"],
    CANONICAL_VARIABLES["wind_speed"],
]

ALL_CANONICAL_VARS = [
    CANONICAL_VARIABLES["temp"],
    CANONICAL_VARIABLES["rh"],
    CANONICAL_VARIABLES["wind_speed"],
    CANONICAL_VARIABLES["rain"],
]

MODEL_CODE_COLUMNS = ["ffmc", "dmc", "dc", "isi", "bui", "fwi"]


def _first_non_null(series: pd.Series) -> object:
    """Return the first non-null value from a series or NA when empty."""
    non_null = series.dropna()
    if non_null.empty:
        return pd.NA
    return non_null.iloc[0]


def _coerce_bool(series: pd.Series) -> pd.Series:
    """Robustly coerce mixed-type columns to boolean."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    truthy = {"true", "1", "t", "yes", "y"}
    normalized = series.fillna(False).astype(str).str.strip().str.lower()
    return normalized.isin(truthy)


def _is_in_fwi_season(local_day: date) -> bool:
    """Return True only for June 1 through September 30 (inclusive)."""
    start = date(local_day.year, SEASON_START_MONTH, SEASON_START_DAY)
    end = date(local_day.year, SEASON_END_MONTH, SEASON_END_DAY)
    return start <= local_day <= end


def _required_columns() -> List[str]:
    """Return required scrubbed schema columns for Step 5 modeling."""
    columns = ["datetime_utc", "station_raw", "station_slug", "source"]
    columns.extend(ALL_CANONICAL_VARS)
    columns.append(CANONICAL_VARIABLES["wind_dir"])
    for var in ALL_CANONICAL_VARS + [CANONICAL_VARIABLES["wind_dir"]]:
        columns.append(f"{var}_failed_qc")
        columns.append(f"{var}_filled_short_gap")
    return columns


def load_and_validate_hourly(path: Path) -> pd.DataFrame:
    """Load scrubbed hourly CSV with strict schema and key validation."""
    if not path.exists():
        raise FileNotFoundError(f"Required input dataset is missing: {path}")

    df = pd.read_csv(path, low_memory=False)
    missing = [col for col in _required_columns() if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in hourly scrubbed dataset: {missing}")

    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
    bad_dt = int(df["datetime_utc"].isna().sum())
    if bad_dt > 0:
        raise ValueError(f"Found {bad_dt} rows with invalid datetime_utc values")

    duplicated_keys = df.duplicated(subset=["station_slug", "datetime_utc"])
    duplicate_count = int(duplicated_keys.sum())
    if duplicate_count > 0:
        raise ValueError(
            "Found duplicated (station_slug, datetime_utc) keys in scrubbed input: "
            f"{duplicate_count}"
        )

    for var in ALL_CANONICAL_VARS + [CANONICAL_VARIABLES["wind_dir"]]:
        df[var] = pd.to_numeric(df[var], errors="coerce")
        df[f"{var}_failed_qc"] = _coerce_bool(df[f"{var}_failed_qc"])
        df[f"{var}_filled_short_gap"] = _coerce_bool(df[f"{var}_filled_short_gap"])

    return df.sort_values(["station_slug", "datetime_utc"]).reset_index(drop=True)


def apply_qc_gating(df: pd.DataFrame) -> pd.DataFrame:
    """Apply strict failed-QC gating before any extraction or aggregation."""
    gated = df.copy()
    for var in ALL_CANONICAL_VARS:
        failed_col = f"{var}_failed_qc"
        gated.loc[gated[failed_col], var] = np.nan
    return gated


def build_season_scaffold(df: pd.DataFrame) -> pd.DataFrame:
    """Build one row per station and in-season local day."""
    work = df.copy()
    work["datetime_local"] = work["datetime_utc"].dt.tz_convert(HALIFAX_TZ)
    work["date_local"] = work["datetime_local"].dt.date
    work["local_hour"] = work["datetime_local"].dt.hour

    season_mask = work["date_local"].map(_is_in_fwi_season)
    season = work[season_mask].copy()
    if season.empty:
        raise ValueError("No in-season rows found after local date conversion.")

    daily = (
        season.groupby(["station_slug", "date_local"], dropna=False)
        .agg(
            station_raw=("station_raw", _first_non_null),
            source=("source", _first_non_null),
        )
        .reset_index()
    )
    return season, daily


def build_noon_features(season_hourly: pd.DataFrame) -> pd.DataFrame:
    """Extract strict local noon weather inputs at exactly hour 12."""
    noon = season_hourly[season_hourly["local_hour"] == 12].copy()

    duplicate_noon = noon.duplicated(subset=["station_slug", "date_local"])
    duplicate_count = int(duplicate_noon.sum())
    if duplicate_count > 0:
        raise ValueError(
            "Found duplicate local-noon rows for station/date combinations: "
            f"{duplicate_count}"
        )

    rename_map = {
        "datetime_local": "datetime_noon_local",
        "datetime_utc": "datetime_noon_utc",
        CANONICAL_VARIABLES["temp"]: "air_temperature_c_noon",
        CANONICAL_VARIABLES["rh"]: "relative_humidity_pct_noon",
        CANONICAL_VARIABLES["wind_speed"]: "wind_speed_kmh_noon",
    }

    keep_columns = ["station_slug", "date_local"] + list(rename_map.keys()) + [
        f"{CANONICAL_VARIABLES['temp']}_filled_short_gap",
        f"{CANONICAL_VARIABLES['rh']}_filled_short_gap",
        f"{CANONICAL_VARIABLES['wind_speed']}_filled_short_gap",
    ]
    noon = noon[keep_columns].rename(columns=rename_map)

    noon["noon_core_present"] = (
        noon[["air_temperature_c_noon", "relative_humidity_pct_noon", "wind_speed_kmh_noon"]]
        .notna()
        .all(axis=1)
    )

    noon["any_noon_core_filled_short_gap"] = noon[
        [
            f"{CANONICAL_VARIABLES['temp']}_filled_short_gap",
            f"{CANONICAL_VARIABLES['rh']}_filled_short_gap",
            f"{CANONICAL_VARIABLES['wind_speed']}_filled_short_gap",
        ]
    ].any(axis=1)

    drop_fill_cols = [
        f"{CANONICAL_VARIABLES['temp']}_filled_short_gap",
        f"{CANONICAL_VARIABLES['rh']}_filled_short_gap",
        f"{CANONICAL_VARIABLES['wind_speed']}_filled_short_gap",
    ]
    noon = noon.drop(columns=drop_fill_cols)

    return noon


def _expected_precip_window_utc(local_day: date) -> pd.DatetimeIndex:
    """Return exact local 13:00->12:00 trailing 24-hour window in UTC stamps."""
    start_local = datetime.combine(local_day - timedelta(days=1), time(13, 0), tzinfo=HALIFAX_TZ)
    end_local = datetime.combine(local_day, time(12, 0), tzinfo=HALIFAX_TZ)
    local_index = pd.date_range(start=start_local, end=end_local, freq="h", tz=HALIFAX_TZ)
    return local_index.tz_convert(UTC_TZ)


def build_precip_features(season_hourly: pd.DataFrame) -> pd.DataFrame:
    """Build strict trailing 24-hour precipitation completeness and sums."""
    rows: List[Dict[str, object]] = []

    for station_slug, station_df in season_hourly.groupby("station_slug", dropna=False):
        precip_series = (
            station_df.set_index("datetime_utc")[CANONICAL_VARIABLES["rain"]]
            .sort_index()
            .astype(float)
        )
        date_values = sorted(set(station_df["date_local"].tolist()))

        for local_day in date_values:
            expected_window_utc = _expected_precip_window_utc(local_day)
            window_values = precip_series.reindex(expected_window_utc)
            is_complete = len(window_values) == 24 and bool(window_values.notna().all())
            precip_sum = float(window_values.sum()) if is_complete else np.nan

            rows.append(
                {
                    "station_slug": station_slug,
                    "date_local": local_day,
                    "precip_window_complete_24h": is_complete,
                    "precip_24h_sum_mm": precip_sum,
                }
            )

    return pd.DataFrame(rows)


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


def run_fwi_state_machine(scaffold: pd.DataFrame) -> pd.DataFrame:
    """Run recursive FWI equations on eligible contiguous in-season segments."""
    modeled = scaffold.copy()
    modeled = modeled.sort_values(["station_slug", "date_local"]).reset_index(drop=True)

    modeled["eligible"] = modeled["noon_core_present"] & modeled["precip_window_complete_24h"]
    modeled["season_year"] = pd.to_datetime(modeled["date_local"]).dt.year

    modeled["segment_id"] = pd.NA
    modeled["continuity_gap_days"] = pd.NA
    modeled["continuity_reset_applied"] = False

    for code_col in MODEL_CODE_COLUMNS:
        modeled[code_col] = np.nan

    for (station_slug, season_year), group in modeled.groupby(["station_slug", "season_year"], sort=False):
        indices = list(group.index)
        prev_eligible_day: date | None = None
        segment_number = 0

        ffmc_prev = FFMC_START
        dmc_prev = DMC_START
        dc_prev = DC_START

        for idx in indices:
            row = modeled.loc[idx]
            local_day = row["date_local"]

            if not bool(row["eligible"]):
                prev_eligible_day = None
                continue

            if prev_eligible_day is None:
                continuity_gap_days = pd.NA
                needs_reset = True
            else:
                continuity_gap_days = (local_day - prev_eligible_day).days
                needs_reset = bool(continuity_gap_days > 1)

            if needs_reset:
                segment_number += 1
                ffmc_prev = FFMC_START
                dmc_prev = DMC_START
                dc_prev = DC_START

            temp_noon = float(row["air_temperature_c_noon"])
            rh_noon = float(row["relative_humidity_pct_noon"])
            wind_noon = float(row["wind_speed_kmh_noon"])
            precip_24h = float(row["precip_24h_sum_mm"])
            month = int(local_day.month)

            ffmc_prev = ffmc_code(temp_noon, rh_noon, wind_noon, precip_24h, ffmc_prev)
            dmc_prev = dmc_code(temp_noon, rh_noon, precip_24h, dmc_prev, month)
            dc_prev = dc_code(temp_noon, precip_24h, dc_prev, month)
            isi_value = isi_index(wind_noon, ffmc_prev)
            bui_value = bui_index(dmc_prev, dc_prev)
            fwi_value = fwi_index(isi_value, bui_value)

            modeled.at[idx, "segment_id"] = f"{season_year}_{segment_number}"
            modeled.at[idx, "continuity_gap_days"] = continuity_gap_days
            modeled.at[idx, "continuity_reset_applied"] = bool(needs_reset)
            modeled.at[idx, "ffmc"] = ffmc_prev
            modeled.at[idx, "dmc"] = dmc_prev
            modeled.at[idx, "dc"] = dc_prev
            modeled.at[idx, "isi"] = isi_value
            modeled.at[idx, "bui"] = bui_value
            modeled.at[idx, "fwi"] = fwi_value

            prev_eligible_day = local_day

    modeled = modeled.drop(columns=["eligible", "season_year"])
    return modeled


def assemble_model_daily_table(hourly: pd.DataFrame) -> pd.DataFrame:
    """Build auditable daily scaffold and run FWI recursion on eligible segments."""
    season_hourly, daily = build_season_scaffold(hourly)
    noon = build_noon_features(season_hourly)
    precip = build_precip_features(season_hourly)

    model_daily = daily.merge(noon, on=["station_slug", "date_local"], how="left")
    model_daily = model_daily.merge(precip, on=["station_slug", "date_local"], how="left")

    model_daily["noon_core_present"] = _coerce_bool(model_daily["noon_core_present"])
    model_daily["precip_window_complete_24h"] = _coerce_bool(model_daily["precip_window_complete_24h"])
    model_daily["any_noon_core_filled_short_gap"] = _coerce_bool(
        model_daily["any_noon_core_filled_short_gap"]
    )

    modeled = run_fwi_state_machine(model_daily)

    ordered_cols = [
        "station_slug",
        "station_raw",
        "source",
        "date_local",
        "datetime_noon_local",
        "datetime_noon_utc",
        "air_temperature_c_noon",
        "relative_humidity_pct_noon",
        "wind_speed_kmh_noon",
        "precip_24h_sum_mm",
        "ffmc",
        "dmc",
        "dc",
        "isi",
        "bui",
        "fwi",
        "noon_core_present",
        "precip_window_complete_24h",
        "any_noon_core_filled_short_gap",
        "segment_id",
        "continuity_gap_days",
        "continuity_reset_applied",
    ]

    modeled = modeled.sort_values(["station_slug", "date_local"]).reset_index(drop=True)
    return modeled[ordered_cols]


def load_reference_fwi(path_dir: Path) -> pd.DataFrame:
    """Load cached official Stanhope daily FWI files."""
    if not path_dir.exists():
        raise FileNotFoundError(f"Reference FWI folder is missing: {path_dir}")

    files = sorted(path_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in reference FWI folder: {path_dir}")

    frames: List[pd.DataFrame] = []
    for file_path in files:
        frame = pd.read_csv(file_path, low_memory=False)
        required = ["Date", "FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]
        missing = [col for col in required if col not in frame.columns]
        if missing:
            raise ValueError(f"Reference file {file_path} is missing columns: {missing}")

        frame = frame[required].copy()
        frame["date_local"] = pd.to_datetime(frame["Date"], errors="coerce").dt.date
        bad_dates = int(frame["date_local"].isna().sum())
        if bad_dates > 0:
            raise ValueError(f"Reference file {file_path} has {bad_dates} invalid Date values")

        frame["season_year"] = pd.to_datetime(frame["date_local"]).dt.year
        frame["source_file"] = file_path.name
        frames.append(frame)

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["date_local"], keep="last").reset_index(drop=True)
    return out


def _safe_metrics(diff: pd.Series) -> Dict[str, float | int]:
    """Compute MAE, RMSE, and bias for a difference vector."""
    valid = diff.dropna()
    n = int(valid.shape[0])
    if n == 0:
        return {"n_pairs": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan}
    mae = float(valid.abs().mean())
    rmse = float(np.sqrt((valid**2).mean()))
    bias = float(valid.mean())
    return {"n_pairs": n, "mae": mae, "rmse": rmse, "bias": bias}


def build_validation_summary(model_daily: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Join modeled Stanhope rows with official reference and summarize errors."""
    stanhope_mask = model_daily["station_slug"].astype(str).str.strip().str.casefold() == "stanhope"
    stanhope_model = model_daily[stanhope_mask].copy()

    if stanhope_model.empty:
        raise ValueError("No Stanhope rows found in modeled daily dataset for validation.")

    stanhope_model["season_year"] = pd.to_datetime(stanhope_model["date_local"]).dt.year

    merged = stanhope_model.merge(reference, on="date_local", how="inner", suffixes=("_model", "_ref"))
    if merged.empty:
        raise ValueError("Stanhope validation join is empty; check date alignment.")

    rows: List[Dict[str, object]] = []
    for code in MODEL_CODE_COLUMNS:
        code_upper = code.upper()
        diff = merged[code] - merged[code_upper]
        overall = _safe_metrics(diff)
        rows.append(
            {
                "scope": "overall",
                "season_year": pd.NA,
                "station_slug": "Stanhope",
                "code": code_upper,
                **overall,
            }
        )

        for season_year, season_df in merged.groupby("season_year_model", dropna=False):
            season_diff = season_df[code] - season_df[code_upper]
            seasonal = _safe_metrics(season_diff)
            rows.append(
                {
                    "scope": "season",
                    "season_year": int(season_year),
                    "station_slug": "Stanhope",
                    "code": code_upper,
                    **seasonal,
                }
            )

    summary = pd.DataFrame(rows)
    return summary.sort_values(["code", "scope", "season_year"], na_position="first").reset_index(drop=True)


def main() -> int:
    """Run Step 5 model stage end-to-end."""
    ensure_directories()
    log_path = LOGS_DIR / f"model_fwi_{datetime.now(timezone.utc):%Y%m%d}.log"
    logger = setup_logging("04_model_fwi", log_file_path=log_path)

    logger.info("Step 5 Model FWI started.")
    logger.info("Input scrubbed hourly file: %s", INPUT_HOURLY)

    hourly = load_and_validate_hourly(INPUT_HOURLY)
    logger.info("Loaded hourly rows=%s stations=%s", len(hourly), hourly["station_slug"].nunique())

    hourly = apply_qc_gating(hourly)
    model_daily = assemble_model_daily_table(hourly)

    OUTPUT_MODEL_DAILY.parent.mkdir(parents=True, exist_ok=True)
    model_daily.to_csv(OUTPUT_MODEL_DAILY, index=False)

    ref_fwi = load_reference_fwi(REFERENCE_FWI_DIR)
    validation = build_validation_summary(model_daily, ref_fwi)
    validation.to_csv(OUTPUT_VALIDATION, index=False)

    eligible_count = int((model_daily["noon_core_present"] & model_daily["precip_window_complete_24h"]).sum())
    logger.info("Wrote table: %s", OUTPUT_MODEL_DAILY)
    logger.info("Wrote table: %s", OUTPUT_VALIDATION)
    logger.info(
        "Summary | rows=%s | stations=%s | eligible_days=%s | validation_rows=%s",
        len(model_daily),
        model_daily["station_slug"].nunique(),
        eligible_count,
        len(validation),
    )
    logger.info(
        "Next steps: inspect validation summary for systematic Stanhope bias and use model output for Step 6 redundancy and uncertainty analyses."
    )
    logger.info("Step 5 Model FWI completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
