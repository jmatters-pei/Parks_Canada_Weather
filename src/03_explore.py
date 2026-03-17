"""Step 4 explore stage: FWI readiness checks and redundancy-oriented EDA."""

from __future__ import annotations

import itertools
import math
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from zoneinfo import ZoneInfo

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import (
    CANONICAL_VARIABLES,
    FIGURES_DIR,
    LOGS_DIR,
    SCRUBBED_DIR,
    TABLES_DIR,
    ensure_directories,
    setup_logging,
)

matplotlib.use("Agg")

INPUT_HOURLY = SCRUBBED_DIR / "hourly_weather_utc.csv"

CORE_VARS = [
    CANONICAL_VARIABLES["temp"],
    CANONICAL_VARIABLES["rh"],
    CANONICAL_VARIABLES["wind_speed"],
]
ALL_VARS = [
    CANONICAL_VARIABLES["temp"],
    CANONICAL_VARIABLES["rh"],
    CANONICAL_VARIABLES["wind_speed"],
    CANONICAL_VARIABLES["wind_dir"],
    CANONICAL_VARIABLES["rain"],
]
PAIRWISE_VARS = [
    CANONICAL_VARIABLES["temp"],
    CANONICAL_VARIABLES["rh"],
    CANONICAL_VARIABLES["wind_speed"],
]

HALIFAX_TZ = ZoneInfo("America/Halifax")
MIN_PAIRWISE_OVERLAP_HOURS = 720

MONTH_LABELS = {
    1: "jan",
    2: "feb",
    3: "mar",
    4: "apr",
    5: "may",
    6: "jun",
    7: "jul",
    8: "aug",
    9: "sep",
    10: "oct",
    11: "nov",
    12: "dec",
}

WEST_TO_EAST_STATIONS = [
    "Stanley_Bridge_Wharf",
    "Cavendish",
    "North_Rustico_Wharf",
    "Stanhope",
    "Tracadie_Wharf",
    "Greenwich",
]


def _ordered_station_categories(values: Iterable[object]) -> List[str]:
    present = {str(value) for value in values if pd.notna(value)}
    ordered_known = [station for station in WEST_TO_EAST_STATIONS if station in present]
    ordered_unknown = sorted(present - set(WEST_TO_EAST_STATIONS))
    return ordered_known + ordered_unknown


def _apply_station_order(df: pd.DataFrame, column: str = "station_slug") -> pd.DataFrame:
    out = df.copy()
    categories = _ordered_station_categories(out[column].tolist())
    out[column] = pd.Categorical(out[column], categories=categories, ordered=True)
    return out


def _required_columns() -> List[str]:
    columns = ["datetime_utc", "station_slug", "station_raw", "source"] + ALL_VARS
    for var in ALL_VARS:
        columns.append(f"{var}_failed_qc")
        columns.append(f"{var}_filled_short_gap")
    return columns


def load_hourly_dataset(path: Path) -> pd.DataFrame:
    """Load scrubbed hourly weather data and enforce core schema integrity."""
    if not path.exists():
        raise FileNotFoundError(f"Scrubbed hourly dataset is missing: {path}")

    df = pd.read_csv(path, low_memory=False)
    missing_columns = [col for col in _required_columns() if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in scrubbed dataset: {missing_columns}")

    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
    if df["datetime_utc"].isna().any():
        invalid_count = int(df["datetime_utc"].isna().sum())
        raise ValueError(f"Found {invalid_count} rows with invalid datetime_utc values")

    duplicated_keys = df.duplicated(subset=["station_slug", "datetime_utc"])
    if bool(duplicated_keys.any()):
        raise ValueError(
            "Found duplicated (station_slug, datetime_utc) keys: "
            f"{int(duplicated_keys.sum())}"
        )

    for var in ALL_VARS:
        df[var] = pd.to_numeric(df[var], errors="coerce")
        df[f"{var}_failed_qc"] = df[f"{var}_failed_qc"].fillna(False).astype(bool)
        df[f"{var}_filled_short_gap"] = df[f"{var}_filled_short_gap"].fillna(False).astype(bool)

    df = _apply_station_order(df, "station_slug")
    df = df.sort_values(["station_slug", "datetime_utc"]).reset_index(drop=True)
    return df


def build_coverage_overall(df: pd.DataFrame) -> pd.DataFrame:
    """Build station x variable coverage/QC summary table."""
    rows: List[Dict[str, object]] = []
    for (station_slug, station_raw), station_df in df.groupby(["station_slug", "station_raw"], dropna=False):
        total_hours = len(station_df)
        for var in ALL_VARS:
            missing_hours = int(station_df[var].isna().sum())
            failed_qc_count = int(station_df[f"{var}_failed_qc"].sum())
            filled_short_gap_count = int(station_df[f"{var}_filled_short_gap"].sum())
            rows.append(
                {
                    "station_slug": station_slug,
                    "station_raw": station_raw,
                    "variable": var,
                    "total_hours": total_hours,
                    "missing_hours": missing_hours,
                    "missing_pct": (missing_hours / total_hours * 100.0) if total_hours else np.nan,
                    "failed_qc_count": failed_qc_count,
                    "failed_qc_pct": (failed_qc_count / total_hours * 100.0) if total_hours else np.nan,
                    "filled_short_gap_count": filled_short_gap_count,
                    "filled_short_gap_pct": (
                        (filled_short_gap_count / total_hours * 100.0) if total_hours else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_coverage_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Build monthly station x variable coverage/QC summary table."""
    work = df.copy()
    work["year_month"] = work["datetime_utc"].dt.tz_localize(None).dt.to_period("M").astype(str)

    rows: List[Dict[str, object]] = []
    group_cols = ["station_slug", "station_raw", "year_month"]
    for keys, group in work.groupby(group_cols, dropna=False):
        station_slug, station_raw, year_month = keys
        total_hours = len(group)
        for var in ALL_VARS:
            missing_hours = int(group[var].isna().sum())
            failed_qc_count = int(group[f"{var}_failed_qc"].sum())
            filled_short_gap_count = int(group[f"{var}_filled_short_gap"].sum())
            rows.append(
                {
                    "station_slug": station_slug,
                    "station_raw": station_raw,
                    "year_month": year_month,
                    "variable": var,
                    "total_hours": total_hours,
                    "missing_hours": missing_hours,
                    "missing_pct": (missing_hours / total_hours * 100.0) if total_hours else np.nan,
                    "failed_qc_count": failed_qc_count,
                    "failed_qc_pct": (failed_qc_count / total_hours * 100.0) if total_hours else np.nan,
                    "filled_short_gap_count": filled_short_gap_count,
                    "filled_short_gap_pct": (
                        (filled_short_gap_count / total_hours * 100.0) if total_hours else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


def _plot_missingness_heatmap(coverage_overall: pd.DataFrame, output_path: Path) -> None:
    ordered = _apply_station_order(coverage_overall, "station_slug")
    heat = ordered.pivot(index="station_slug", columns="variable", values="missing_pct")
    plt.figure(figsize=(11, 6))
    sns.heatmap(heat, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={"label": "Missing (%)"})
    plt.title("Station Missingness by Variable")
    plt.xlabel("Variable")
    plt.ylabel("Station")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _plot_missingness_timeline(coverage_monthly: pd.DataFrame, output_path: Path) -> None:
    subset = coverage_monthly[coverage_monthly["variable"].isin(CORE_VARS)].copy()
    subset = _apply_station_order(subset, "station_slug")
    subset["month_dt"] = pd.to_datetime(subset["year_month"] + "-01", utc=True, errors="coerce")

    fig, axes = plt.subplots(len(CORE_VARS), 1, figsize=(12, 10), sharex=True)
    if len(CORE_VARS) == 1:
        axes = [axes]

    for axis, variable in zip(axes, CORE_VARS):
        var_df = subset[subset["variable"] == variable].copy()
        sns.lineplot(
            data=var_df,
            x="month_dt",
            y="missing_pct",
            hue="station_slug",
            hue_order=_ordered_station_categories(var_df["station_slug"].tolist()),
            marker="o",
            linewidth=1.2,
            ax=axis,
        )
        axis.set_ylabel("Missing (%)")
        axis.set_title(f"Monthly Missingness: {variable}")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Month")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _plot_distributions(df: pd.DataFrame, output_path: Path) -> None:
    ordered_df = _apply_station_order(df, "station_slug")
    fig, axes = plt.subplots(1, len(CORE_VARS), figsize=(16, 6), sharex=False)
    if len(CORE_VARS) == 1:
        axes = [axes]

    for axis, variable in zip(axes, CORE_VARS):
        sns.boxplot(
            data=ordered_df,
            x="station_slug",
            y=variable,
            order=_ordered_station_categories(ordered_df["station_slug"].tolist()),
            ax=axis,
        )
        axis.set_title(f"Distribution: {variable}")
        axis.set_xlabel("Station")
        axis.set_ylabel(variable)
        axis.tick_params(axis="x", rotation=35)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _select_summer_month(df: pd.DataFrame) -> Tuple[int, int]:
    """Pick a July month with the strongest cross-station availability."""
    july = df[df["datetime_utc"].dt.month == 7].copy()
    if july.empty:
        fallback = df["datetime_utc"].dt.to_period("M").value_counts().sort_index()
        top = fallback.index[-1]
        return int(top.year), int(top.month)

    july["year"] = july["datetime_utc"].dt.year
    july["missing_core"] = july[CORE_VARS].isna().sum(axis=1)
    score = july.groupby("year", dropna=False).agg(hours=("datetime_utc", "count"), miss=("missing_core", "sum"))
    score["rank_score"] = score["hours"] - score["miss"]
    best_year = int(score.sort_values(["rank_score", "hours"], ascending=False).index[0])
    return best_year, 7


def _plot_summer_timeseries(df: pd.DataFrame, output_path: Path) -> Tuple[int, int]:
    year, month = _select_summer_month(df)
    subset = df[(df["datetime_utc"].dt.year == year) & (df["datetime_utc"].dt.month == month)].copy()
    subset = _apply_station_order(subset, "station_slug")

    fig, axes = plt.subplots(len(CORE_VARS), 1, figsize=(14, 10), sharex=True)
    if len(CORE_VARS) == 1:
        axes = [axes]

    for axis, variable in zip(axes, CORE_VARS):
        sns.lineplot(
            data=subset,
            x="datetime_utc",
            y=variable,
            hue="station_slug",
            hue_order=_ordered_station_categories(subset["station_slug"].tolist()),
            linewidth=0.9,
            ax=axis,
        )
        axis.set_title(f"{variable} overlay for {year}-{month:02d}")
        axis.set_ylabel(variable)
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("UTC Timestamp")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return year, month


def _plot_wind_roses(df: pd.DataFrame, output_path: Path) -> None:
    stations = _ordered_station_categories(df["station_slug"].dropna().tolist())
    if not stations:
        return

    speed_bins = [0, 5, 10, 20, 30, np.inf]
    speed_labels = ["0-5", "5-10", "10-20", "20-30", "30+"]
    sector_count = 16
    sector_step = 360.0 / sector_count
    theta = np.deg2rad(np.arange(0, 360, sector_step))
    width = np.deg2rad(sector_step * 0.9)

    n_cols = 3
    n_rows = math.ceil(len(stations) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.8 * n_rows), subplot_kw={"projection": "polar"})
    axes_arr = np.atleast_1d(axes).flatten()

    colors = sns.color_palette("YlGnBu", n_colors=len(speed_labels))

    for idx, station in enumerate(stations):
        axis = axes_arr[idx]
        s_df = df[df["station_slug"] == station].copy()
        s_df = s_df[
            s_df[CANONICAL_VARIABLES["wind_dir"]].notna() & s_df[CANONICAL_VARIABLES["wind_speed"]].notna()
        ].copy()
        s_df = s_df[s_df[CANONICAL_VARIABLES["wind_speed"]] > 0]

        if s_df.empty:
            axis.set_title(f"{station} (no wind data)")
            axis.set_axis_off()
            continue

        s_df["sector_idx"] = ((s_df[CANONICAL_VARIABLES["wind_dir"]] % 360) / sector_step).astype(int)
        s_df["speed_bin"] = pd.cut(
            s_df[CANONICAL_VARIABLES["wind_speed"]],
            bins=speed_bins,
            labels=speed_labels,
            include_lowest=True,
            right=False,
        )
        table = (
            s_df.groupby(["sector_idx", "speed_bin"], observed=False)
            .size()
            .unstack(fill_value=0)
            .reindex(index=range(sector_count), columns=speed_labels, fill_value=0)
        )
        freq_pct = table.div(table.values.sum()) * 100.0

        base = np.zeros(sector_count)
        for speed_label, color in zip(speed_labels, colors):
            values = freq_pct[speed_label].to_numpy(dtype=float)
            axis.bar(theta, values, width=width, bottom=base, color=color, edgecolor="white", linewidth=0.3)
            base += values

        axis.set_theta_zero_location("N")
        axis.set_theta_direction(-1)
        axis.set_title(station)

    for idx in range(len(stations), len(axes_arr)):
        axes_arr[idx].set_axis_off()

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color) for color in colors]
    fig.legend(handles, speed_labels, title="Wind speed (km/h)", loc="lower center", ncol=len(speed_labels))
    fig.suptitle("Wind Roses by Station (frequency %)")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    plt.savefig(output_path, dpi=180)
    plt.close()


def build_noon_readiness(df: pd.DataFrame) -> pd.DataFrame:
    """Compute DST-aware local-noon readiness metrics for FWI prerequisites."""
    rows: List[Dict[str, object]] = []
    for station_slug, station_df in df.groupby("station_slug", dropna=False):
        s_df = station_df.copy()
        s_df["datetime_local"] = s_df["datetime_utc"].dt.tz_convert(HALIFAX_TZ)
        s_df["local_date"] = s_df["datetime_local"].dt.date
        s_df["local_hour"] = s_df["datetime_local"].dt.hour

        noon_rows = s_df[s_df["local_hour"] == 12].copy()
        days_with_noon_row = int(noon_rows["local_date"].nunique())

        noon_core_ready = noon_rows[CORE_VARS].notna().all(axis=1)
        days_noon_ready_core = int(noon_rows.loc[noon_core_ready, "local_date"].nunique())

        precip_series = s_df.set_index("datetime_utc")[CANONICAL_VARIABLES["rain"]]
        precip_ready_days = 0
        for local_day in sorted(noon_rows["local_date"].unique().tolist()):
            start_local = datetime.combine(local_day - timedelta(days=1), time(13, 0), tzinfo=HALIFAX_TZ)
            window_local = pd.date_range(start=start_local, periods=24, freq="h", tz=HALIFAX_TZ)
            window_utc = window_local.tz_convert(timezone.utc)
            window_vals = precip_series.reindex(window_utc)
            if int(window_vals.notna().sum()) == 24:
                precip_ready_days += 1

        rows.append(
            {
                "station_slug": station_slug,
                "days_with_noon_row": days_with_noon_row,
                "days_noon_ready_core": days_noon_ready_core,
                "days_noon_core_not_ready": days_with_noon_row - days_noon_ready_core,
                "days_precip_ready_24h": precip_ready_days,
                "noon_core_ready_pct": (
                    days_noon_ready_core / days_with_noon_row * 100.0 if days_with_noon_row else np.nan
                ),
                "precip_ready_24h_pct": (
                    precip_ready_days / days_with_noon_row * 100.0 if days_with_noon_row else np.nan
                ),
            }
        )

    out = pd.DataFrame(rows)
    out = _apply_station_order(out, "station_slug")
    return out.sort_values("station_slug").reset_index(drop=True)


def _plot_noon_readiness(noon_readiness: pd.DataFrame, output_path: Path) -> None:
    ordered = _apply_station_order(noon_readiness, "station_slug")
    plot_df = ordered.melt(
        id_vars=["station_slug"],
        value_vars=["noon_core_ready_pct", "precip_ready_24h_pct"],
        var_name="metric",
        value_name="pct",
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=plot_df,
        x="station_slug",
        y="pct",
        hue="metric",
        order=_ordered_station_categories(plot_df["station_slug"].tolist()),
    )
    plt.ylabel("Readiness (%)")
    plt.xlabel("Station")
    plt.title("Local-Noon FWI Readiness (America/Halifax)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _pearson(series_a: pd.Series, series_b: pd.Series) -> float:
    if len(series_a) < 2:
        return np.nan
    return float(series_a.corr(series_b))


def _rmse(series_a: pd.Series, series_b: pd.Series) -> float:
    return float(np.sqrt(np.mean((series_a - series_b) ** 2)))


def _bias(series_a: pd.Series, series_b: pd.Series) -> float:
    return float((series_a - series_b).mean())


def _season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def _common_timeline(station_a: pd.DataFrame, station_b: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    left = station_a[["datetime_utc", *columns]].copy()
    right = station_b[["datetime_utc", *columns]].copy()
    rename_a = {col: f"{col}_a" for col in columns}
    rename_b = {col: f"{col}_b" for col in columns}
    left = left.rename(columns=rename_a)
    right = right.rename(columns=rename_b)
    merged = left.merge(right, on="datetime_utc", how="inner")
    return merged


def build_pairwise_similarity(
    df: pd.DataFrame,
    min_overlap_hours: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute pairwise station similarity metrics for redundancy scouting."""
    station_frames = {slug: chunk.copy() for slug, chunk in df.groupby("station_slug", dropna=False)}
    stations = sorted(station_frames.keys())

    rows_core: List[Dict[str, object]] = []
    rows_precip: List[Dict[str, object]] = []
    rows_wind_uv: List[Dict[str, object]] = []
    rows_seasonal: List[Dict[str, object]] = []

    for station_a, station_b in itertools.combinations(stations, 2):
        left = station_frames[station_a]
        right = station_frames[station_b]

        merged_presence = _common_timeline(left, right, [])
        overlap_hours_total = len(merged_presence)

        # Temperature/RH/wind speed metrics
        for variable in PAIRWISE_VARS:
            merged = _common_timeline(left, right, [variable])
            valid = merged[f"{variable}_a"].notna() & merged[f"{variable}_b"].notna()
            n_overlap = int(valid.sum())
            if n_overlap < min_overlap_hours:
                continue

            va = merged.loc[valid, f"{variable}_a"]
            vb = merged.loc[valid, f"{variable}_b"]

            rows_core.append(
                {
                    "station_a": station_a,
                    "station_b": station_b,
                    "variable": variable,
                    "n_overlap_hours": n_overlap,
                    "overlap_hours_total": overlap_hours_total,
                    "concurrent_availability_pct": (
                        n_overlap / overlap_hours_total * 100.0 if overlap_hours_total else np.nan
                    ),
                    "pearson_r": _pearson(va, vb),
                    "rmse": _rmse(va, vb),
                    "mean_bias_a_minus_b": _bias(va, vb),
                }
            )

            seasonal = merged.loc[valid, ["datetime_utc", f"{variable}_a", f"{variable}_b"]].copy()
            seasonal["month"] = seasonal["datetime_utc"].dt.month
            seasonal["season"] = seasonal["month"].map(_season_from_month)
            for season_name, season_df in seasonal.groupby("season", dropna=False):
                if season_df.empty:
                    continue
                sea_a = season_df[f"{variable}_a"]
                sea_b = season_df[f"{variable}_b"]
                rows_seasonal.append(
                    {
                        "station_a": station_a,
                        "station_b": station_b,
                        "variable": variable,
                        "season": season_name,
                        "n_overlap_hours": int(len(season_df)),
                        "pearson_r": _pearson(sea_a, sea_b),
                        "rmse": _rmse(sea_a, sea_b),
                        "mean_bias_a_minus_b": _bias(sea_a, sea_b),
                    }
                )

        # Precipitation metrics
        precip_var = CANONICAL_VARIABLES["rain"]
        merged_precip = _common_timeline(left, right, [precip_var])
        precip_valid = merged_precip[f"{precip_var}_a"].notna() & merged_precip[f"{precip_var}_b"].notna()
        n_overlap_precip = int(precip_valid.sum())
        if n_overlap_precip >= min_overlap_hours:
            pa = merged_precip.loc[precip_valid, f"{precip_var}_a"]
            pb = merged_precip.loc[precip_valid, f"{precip_var}_b"]
            rows_precip.append(
                {
                    "station_a": station_a,
                    "station_b": station_b,
                    "variable": precip_var,
                    "n_overlap_hours": n_overlap_precip,
                    "overlap_hours_total": overlap_hours_total,
                    "concurrent_availability_pct": (
                        n_overlap_precip / overlap_hours_total * 100.0 if overlap_hours_total else np.nan
                    ),
                    "pearson_r": _pearson(pa, pb),
                    "rmse": _rmse(pa, pb),
                    "mean_bias_a_minus_b": _bias(pa, pb),
                }
            )

        # Wind direction vector components (u, v)
        dir_var = CANONICAL_VARIABLES["wind_dir"]
        ws_var = CANONICAL_VARIABLES["wind_speed"]
        merged_wind = _common_timeline(left, right, [ws_var, dir_var])
        wind_valid = (
            merged_wind[f"{ws_var}_a"].notna()
            & merged_wind[f"{dir_var}_a"].notna()
            & merged_wind[f"{ws_var}_b"].notna()
            & merged_wind[f"{dir_var}_b"].notna()
        )
        n_overlap_wind = int(wind_valid.sum())
        if n_overlap_wind >= min_overlap_hours:
            work = merged_wind.loc[wind_valid].copy()
            rad_a = np.deg2rad(work[f"{dir_var}_a"])
            rad_b = np.deg2rad(work[f"{dir_var}_b"])
            work["u_a"] = work[f"{ws_var}_a"] * np.cos(rad_a)
            work["v_a"] = work[f"{ws_var}_a"] * np.sin(rad_a)
            work["u_b"] = work[f"{ws_var}_b"] * np.cos(rad_b)
            work["v_b"] = work[f"{ws_var}_b"] * np.sin(rad_b)

            for component in ("u", "v"):
                rows_wind_uv.append(
                    {
                        "station_a": station_a,
                        "station_b": station_b,
                        "component": component,
                        "n_overlap_hours": n_overlap_wind,
                        "overlap_hours_total": overlap_hours_total,
                        "concurrent_availability_pct": (
                            n_overlap_wind / overlap_hours_total * 100.0 if overlap_hours_total else np.nan
                        ),
                        "pearson_r": _pearson(work[f"{component}_a"], work[f"{component}_b"]),
                        "rmse": _rmse(work[f"{component}_a"], work[f"{component}_b"]),
                        "mean_bias_a_minus_b": _bias(work[f"{component}_a"], work[f"{component}_b"]),
                    }
                )

    return (
        pd.DataFrame(rows_core),
        pd.DataFrame(rows_precip),
        pd.DataFrame(rows_wind_uv),
        pd.DataFrame(rows_seasonal),
    )


def _corr_matrix(df: pd.DataFrame, variable: str, min_periods: int = MIN_PAIRWISE_OVERLAP_HOURS) -> pd.DataFrame:
    wide = (
        df[["datetime_utc", "station_slug", variable]]
        .pivot(index="datetime_utc", columns="station_slug", values=variable)
    )
    station_order = _ordered_station_categories(wide.columns.tolist())
    wide = wide.reindex(columns=station_order)
    matrix = wide.corr(min_periods=min_periods)
    return matrix.reindex(index=station_order, columns=station_order)


def _plot_corr_heatmap(matrix: pd.DataFrame, title: str, output_path: Path) -> None:
    plt.figure(figsize=(8.5, 7))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, square=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _plot_monthly_precip_corr_heatmaps(df: pd.DataFrame, output_dir: Path) -> List[Path]:
    """Create one precipitation correlation heatmap per month pooled across years."""
    outputs: List[Path] = []
    for month in range(1, 13):
        month_df = df[df["datetime_utc"].dt.month == month].copy()
        matrix = _corr_matrix(month_df, CANONICAL_VARIABLES["rain"])
        month_slug = MONTH_LABELS[month]
        out_path = output_dir / f"explore_corr_precip_{month:02d}_{month_slug}.png"
        _plot_corr_heatmap(
            matrix,
            f"Correlation Matrix: Precipitation ({month_slug.upper()} pooled across years)",
            out_path,
        )
        outputs.append(out_path)
    return outputs


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write DataFrame to CSV (empty frames still keep headers where possible)."""
    if df.empty:
        df.to_csv(path, index=False)
        return
    df.to_csv(path, index=False)


def main() -> int:
    """Run the Explore stage end-to-end."""
    ensure_directories()
    log_path = LOGS_DIR / f"explore_{datetime.now(timezone.utc):%Y%m%d}.log"
    logger = setup_logging("03_explore", log_file_path=log_path)

    logger.info("Step 4 Explore started.")
    logger.info("Input scrubbed dataset: %s", INPUT_HOURLY)

    sns.set_theme(style="whitegrid")

    df = load_hourly_dataset(INPUT_HOURLY)
    logger.info("Loaded scrubbed rows=%s stations=%s", len(df), df["station_slug"].nunique())

    coverage_overall = build_coverage_overall(df)
    coverage_monthly = build_coverage_monthly(df)
    noon_readiness = build_noon_readiness(df)
    pairwise_core, pairwise_precip, pairwise_wind_uv, pairwise_seasonal = build_pairwise_similarity(
        df, min_overlap_hours=MIN_PAIRWISE_OVERLAP_HOURS
    )

    write_csv(coverage_overall, TABLES_DIR / "explore_coverage_overall.csv")
    write_csv(coverage_monthly, TABLES_DIR / "explore_coverage_monthly.csv")
    write_csv(noon_readiness, TABLES_DIR / "explore_noon_readiness.csv")
    write_csv(pairwise_core, TABLES_DIR / "explore_pairwise_core_metrics.csv")
    write_csv(pairwise_precip, TABLES_DIR / "explore_pairwise_precip_metrics.csv")
    write_csv(pairwise_wind_uv, TABLES_DIR / "explore_pairwise_wind_uv_metrics.csv")
    write_csv(pairwise_seasonal, TABLES_DIR / "explore_pairwise_seasonal_metrics.csv")

    _plot_missingness_heatmap(coverage_overall, FIGURES_DIR / "explore_missingness_heatmap.png")
    _plot_missingness_timeline(coverage_monthly, FIGURES_DIR / "explore_missingness_timeline.png")
    _plot_distributions(df, FIGURES_DIR / "explore_distributions.png")
    selected_year, selected_month = _plot_summer_timeseries(df, FIGURES_DIR / "explore_timeseries_summer.png")
    _plot_wind_roses(df, FIGURES_DIR / "explore_wind_roses.png")
    _plot_noon_readiness(noon_readiness, FIGURES_DIR / "explore_noon_readiness.png")

    corr_temp = _corr_matrix(df, CANONICAL_VARIABLES["temp"])
    corr_rh = _corr_matrix(df, CANONICAL_VARIABLES["rh"])
    corr_wind = _corr_matrix(df, CANONICAL_VARIABLES["wind_speed"])
    corr_precip = _corr_matrix(df, CANONICAL_VARIABLES["rain"])
    _plot_corr_heatmap(corr_temp, "Correlation Matrix: Air Temperature", FIGURES_DIR / "explore_corr_temp.png")
    _plot_corr_heatmap(corr_rh, "Correlation Matrix: Relative Humidity", FIGURES_DIR / "explore_corr_rh.png")
    _plot_corr_heatmap(
        corr_wind,
        "Correlation Matrix: Wind Speed",
        FIGURES_DIR / "explore_corr_wind_speed.png",
    )
    _plot_corr_heatmap(
        corr_precip,
        "Correlation Matrix: Precipitation",
        FIGURES_DIR / "explore_corr_precip.png",
    )
    monthly_precip_corr_files = _plot_monthly_precip_corr_heatmaps(df, FIGURES_DIR)

    logger.info("Wrote table: %s", TABLES_DIR / "explore_coverage_overall.csv")
    logger.info("Wrote table: %s", TABLES_DIR / "explore_coverage_monthly.csv")
    logger.info("Wrote table: %s", TABLES_DIR / "explore_noon_readiness.csv")
    logger.info("Wrote table: %s", TABLES_DIR / "explore_pairwise_core_metrics.csv")
    logger.info("Wrote table: %s", TABLES_DIR / "explore_pairwise_precip_metrics.csv")
    logger.info("Wrote table: %s", TABLES_DIR / "explore_pairwise_wind_uv_metrics.csv")
    logger.info("Wrote table: %s", TABLES_DIR / "explore_pairwise_seasonal_metrics.csv")

    logger.info("Wrote figure: %s", FIGURES_DIR / "explore_missingness_heatmap.png")
    logger.info("Wrote figure: %s", FIGURES_DIR / "explore_missingness_timeline.png")
    logger.info("Wrote figure: %s", FIGURES_DIR / "explore_distributions.png")
    logger.info("Wrote figure: %s", FIGURES_DIR / "explore_timeseries_summer.png")
    logger.info("Wrote figure: %s", FIGURES_DIR / "explore_wind_roses.png")
    logger.info("Wrote figure: %s", FIGURES_DIR / "explore_noon_readiness.png")
    logger.info("Wrote figure: %s", FIGURES_DIR / "explore_corr_temp.png")
    logger.info("Wrote figure: %s", FIGURES_DIR / "explore_corr_rh.png")
    logger.info("Wrote figure: %s", FIGURES_DIR / "explore_corr_wind_speed.png")
    logger.info("Wrote figure: %s", FIGURES_DIR / "explore_corr_precip.png")
    for file_path in monthly_precip_corr_files:
        logger.info("Wrote figure: %s", file_path)

    logger.info(
        "Summary | rows=%s | stations=%s | summer_overlay_month=%s-%02d | pairwise_core_rows=%s",
        len(df),
        df["station_slug"].nunique(),
        selected_year,
        selected_month,
        len(pairwise_core),
    )
    logger.info("Step 4 Explore completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
