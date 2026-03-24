"""Step 6 model stage: station redundancy modeling and reference benchmarking."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from zoneinfo import ZoneInfo

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import (
    BENCHMARK_REFERENCE_STATION_SLUG,
    CANONICAL_VARIABLES,
    FWI_SEASON_MONTHS,
    HOBOLINK_STATIONS,
    LOGS_DIR,
    MIN_ALIGNED_HOURS_PER_WINDOW,
    MIN_COMPLETE_CASE_FRACTION,
    REDUNDANCY_INCLUDE_LOG1P_PRECIP_SENSITIVITY,
    REDUNDANCY_INCLUDE_WIND_DIRECTION_FEATURES,
    REDUNDANCY_RANDOM_SEED,
    SCRUBBED_DIR,
    SEASON_BY_MONTH,
    TABLES_DIR,
    FIGURES_DIR,
    ensure_directories,
    setup_logging,
    station_name_to_slug,
)

matplotlib.use("Agg")

INPUT_HOURLY = SCRUBBED_DIR / "hourly_weather_utc.csv"
OUTPUT_EXPLAINED = TABLES_DIR / "model_pca_explained_variance.csv"
OUTPUT_LOADINGS = TABLES_DIR / "model_pca_loadings.csv"
OUTPUT_SCORES = TABLES_DIR / "model_pca_scores.csv"
OUTPUT_STATION_SUMMARY = TABLES_DIR / "model_pca_station_summary.csv"
OUTPUT_BENCHMARKS = TABLES_DIR / "model_hourly_benchmarks.csv"
OUTPUT_BIPLOT = FIGURES_DIR / "model_pca_biplot.png"

HALIFAX_TZ = ZoneInfo("America/Halifax")

PRIMARY_FEATURES = [
    CANONICAL_VARIABLES["temp"],
    CANONICAL_VARIABLES["rh"],
    CANONICAL_VARIABLES["wind_speed"],
    CANONICAL_VARIABLES["rain"],
]

BENCHMARK_METRICS = ["mae", "rmse", "bias", "pearson_r", "spearman_r", "ks_stat"]


def _required_columns() -> List[str]:
    columns = ["datetime_utc", "station_slug", "station_raw", "source"] + PRIMARY_FEATURES
    columns.append(CANONICAL_VARIABLES["wind_dir"])
    for var in PRIMARY_FEATURES + [CANONICAL_VARIABLES["wind_dir"]]:
        columns.append(f"{var}_failed_qc")
    return columns


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    truthy = {"true", "1", "t", "yes", "y"}
    normalized = series.fillna(False).astype(str).str.strip().str.lower()
    return normalized.isin(truthy)


def _write_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def _ks_statistic(series_a: pd.Series, series_b: pd.Series) -> float:
    a = np.sort(series_a.to_numpy(dtype=float))
    b = np.sort(series_b.to_numpy(dtype=float))
    if len(a) == 0 or len(b) == 0:
        return np.nan

    values = np.sort(np.unique(np.concatenate([a, b])))
    cdf_a = np.searchsorted(a, values, side="right") / len(a)
    cdf_b = np.searchsorted(b, values, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _season_from_local_month(month: int) -> str:
    return SEASON_BY_MONTH.get(month, "unknown")


def _fwi_window_from_local_month(month: int) -> str:
    return "fwi_season" if month in FWI_SEASON_MONTHS else "off_season"


def load_and_validate(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required scrubbed dataset is missing: {path}")

    df = pd.read_csv(path, low_memory=False)
    missing = [col for col in _required_columns() if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in scrubbed dataset: {missing}")

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

    for var in PRIMARY_FEATURES + [CANONICAL_VARIABLES["wind_dir"]]:
        df[var] = pd.to_numeric(df[var], errors="coerce")
        failed_col = f"{var}_failed_qc"
        df[failed_col] = _coerce_bool(df[failed_col])

    return df.sort_values(["station_slug", "datetime_utc"]).reset_index(drop=True)


def apply_qc_gating(df: pd.DataFrame) -> pd.DataFrame:
    gated = df.copy()
    for var in PRIMARY_FEATURES + [CANONICAL_VARIABLES["wind_dir"]]:
        failed_col = f"{var}_failed_qc"
        if failed_col in gated.columns:
            gated.loc[gated[failed_col], var] = np.nan
    return gated


def add_time_windows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    local_dt = out["datetime_utc"].dt.tz_convert(HALIFAX_TZ)
    out["season"] = local_dt.dt.month.map(_season_from_local_month)
    out["fwi_window"] = local_dt.dt.month.map(_fwi_window_from_local_month)
    return out


def validate_station_roster(df: pd.DataFrame) -> Tuple[str, List[str], List[str]]:
    stations_present = sorted(df["station_slug"].dropna().astype(str).unique().tolist())
    if BENCHMARK_REFERENCE_STATION_SLUG not in stations_present:
        raise ValueError(
            "Reference station is missing from scrubbed dataset: "
            f"{BENCHMARK_REFERENCE_STATION_SLUG}"
        )

    expected_park_slugs = sorted(station_name_to_slug(name) for name in HOBOLINK_STATIONS)
    missing_parks = [slug for slug in expected_park_slugs if slug not in stations_present]
    return BENCHMARK_REFERENCE_STATION_SLUG, expected_park_slugs, missing_parks


def build_feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    feature_sets: Dict[str, List[str]] = {"core": PRIMARY_FEATURES.copy()}

    if REDUNDANCY_INCLUDE_WIND_DIRECTION_FEATURES:
        dir_col = CANONICAL_VARIABLES["wind_dir"]
        if dir_col in df.columns and df[dir_col].notna().any():
            radians = np.deg2rad(df[dir_col])
            df["wind_dir_sin"] = np.sin(radians)
            df["wind_dir_cos"] = np.cos(radians)
            feature_sets["core_plus_wind_dir_uv"] = PRIMARY_FEATURES + ["wind_dir_sin", "wind_dir_cos"]

    if REDUNDANCY_INCLUDE_LOG1P_PRECIP_SENSITIVITY:
        rain_col = CANONICAL_VARIABLES["rain"]
        df["precipitation_log1p_mm"] = np.log1p(df[rain_col].clip(lower=0))
        feature_sets["core_log1p_precip"] = [
            CANONICAL_VARIABLES["temp"],
            CANONICAL_VARIABLES["rh"],
            CANONICAL_VARIABLES["wind_speed"],
            "precipitation_log1p_mm",
        ]

    return feature_sets


def _window_slices(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    windows: Dict[str, pd.DataFrame] = {
        "overall": df,
        "winter": df[df["season"] == "winter"],
        "spring": df[df["season"] == "spring"],
        "summer": df[df["season"] == "summer"],
        "fall": df[df["season"] == "fall"],
        "fwi_season": df[df["fwi_window"] == "fwi_season"],
        "off_season": df[df["fwi_window"] == "off_season"],
    }
    return windows


def _append_pca_skip_rows(
    explained_rows: List[Dict[str, object]],
    loading_rows: List[Dict[str, object]],
    score_rows: List[Dict[str, object]],
    station_summary_rows: List[Dict[str, object]],
    *,
    window: str,
    feature_set: str,
    reason: str,
    n_rows_window: int,
    complete_case_fraction: float,
    n_complete_rows: int,
) -> None:
    explained_rows.append(
        {
            "window": window,
            "feature_set": feature_set,
            "component": pd.NA,
            "explained_variance_ratio": np.nan,
            "explained_variance_cumulative": np.nan,
            "status": "skipped_insufficient_overlap",
            "reason": reason,
            "n_rows_window": n_rows_window,
            "n_complete_rows": n_complete_rows,
            "complete_case_fraction": complete_case_fraction,
        }
    )
    loading_rows.append(
        {
            "window": window,
            "feature_set": feature_set,
            "component": pd.NA,
            "variable": pd.NA,
            "loading": np.nan,
            "status": "skipped_insufficient_overlap",
            "reason": reason,
        }
    )
    score_rows.append(
        {
            "window": window,
            "feature_set": feature_set,
            "datetime_utc": pd.NA,
            "station_slug": pd.NA,
            "pc": pd.NA,
            "score": np.nan,
            "status": "skipped_insufficient_overlap",
            "reason": reason,
        }
    )
    station_summary_rows.append(
        {
            "window": window,
            "feature_set": feature_set,
            "station_slug": pd.NA,
            "pc": pd.NA,
            "n_rows": 0,
            "mean": np.nan,
            "std": np.nan,
            "q25": np.nan,
            "median": np.nan,
            "q75": np.nan,
            "status": "skipped_insufficient_overlap",
            "reason": reason,
        }
    )


def run_pca_models(df: pd.DataFrame, feature_sets: Dict[str, List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    explained_rows: List[Dict[str, object]] = []
    loading_rows: List[Dict[str, object]] = []
    score_rows: List[Dict[str, object]] = []
    station_summary_rows: List[Dict[str, object]] = []

    overall_core_for_biplot = pd.DataFrame()
    windows = _window_slices(df)

    for window_name, window_df in windows.items():
        n_rows_window = int(len(window_df))
        for feature_set_name, feature_columns in feature_sets.items():
            modeling_columns = ["datetime_utc", "station_slug", *feature_columns]
            model_df = window_df[modeling_columns].dropna(subset=feature_columns).copy()

            n_complete_rows = int(len(model_df))
            complete_case_fraction = (
                n_complete_rows / n_rows_window if n_rows_window > 0 else np.nan
            )

            if (
                n_complete_rows < MIN_ALIGNED_HOURS_PER_WINDOW
                or (pd.notna(complete_case_fraction) and complete_case_fraction < MIN_COMPLETE_CASE_FRACTION)
            ):
                reason = (
                    f"insufficient data (n_complete_rows={n_complete_rows}, "
                    f"complete_case_fraction={complete_case_fraction:.3f})"
                )
                _append_pca_skip_rows(
                    explained_rows,
                    loading_rows,
                    score_rows,
                    station_summary_rows,
                    window=window_name,
                    feature_set=feature_set_name,
                    reason=reason,
                    n_rows_window=n_rows_window,
                    complete_case_fraction=float(complete_case_fraction) if pd.notna(complete_case_fraction) else np.nan,
                    n_complete_rows=n_complete_rows,
                )
                continue

            x = model_df[feature_columns].to_numpy(dtype=float)
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x)

            n_components = min(len(feature_columns), 4)
            pca = PCA(n_components=n_components, random_state=REDUNDANCY_RANDOM_SEED)
            scores = pca.fit_transform(x_scaled)

            cumulative = np.cumsum(pca.explained_variance_ratio_)
            for component_idx, (evr, evr_cum) in enumerate(
                zip(pca.explained_variance_ratio_, cumulative),
                start=1,
            ):
                explained_rows.append(
                    {
                        "window": window_name,
                        "feature_set": feature_set_name,
                        "component": f"PC{component_idx}",
                        "explained_variance_ratio": float(evr),
                        "explained_variance_cumulative": float(evr_cum),
                        "status": "ok",
                        "reason": "",
                        "n_rows_window": n_rows_window,
                        "n_complete_rows": n_complete_rows,
                        "complete_case_fraction": float(complete_case_fraction),
                    }
                )

            for component_idx in range(pca.n_components_):
                comp_name = f"PC{component_idx + 1}"
                for variable, value in zip(feature_columns, pca.components_[component_idx]):
                    loading_rows.append(
                        {
                            "window": window_name,
                            "feature_set": feature_set_name,
                            "component": comp_name,
                            "variable": variable,
                            "loading": float(value),
                            "status": "ok",
                            "reason": "",
                        }
                    )

            score_columns = [f"PC{i + 1}" for i in range(scores.shape[1])]
            score_df = model_df[["datetime_utc", "station_slug"]].copy()
            for idx, col_name in enumerate(score_columns):
                score_df[col_name] = scores[:, idx]

            score_long = score_df.melt(
                id_vars=["datetime_utc", "station_slug"],
                value_vars=score_columns,
                var_name="pc",
                value_name="score",
            )
            score_long["window"] = window_name
            score_long["feature_set"] = feature_set_name
            score_long["status"] = "ok"
            score_long["reason"] = ""
            score_rows.extend(
                score_long[
                    [
                        "window",
                        "feature_set",
                        "datetime_utc",
                        "station_slug",
                        "pc",
                        "score",
                        "status",
                        "reason",
                    ]
                ].to_dict(orient="records")
            )

            station_group = score_df.groupby("station_slug", dropna=False)
            for station_slug, station_df in station_group:
                for col_name in score_columns:
                    series = station_df[col_name].astype(float)
                    station_summary_rows.append(
                        {
                            "window": window_name,
                            "feature_set": feature_set_name,
                            "station_slug": station_slug,
                            "pc": col_name,
                            "n_rows": int(len(series)),
                            "mean": float(series.mean()),
                            "std": float(series.std(ddof=1)) if len(series) > 1 else np.nan,
                            "q25": float(series.quantile(0.25)),
                            "median": float(series.quantile(0.50)),
                            "q75": float(series.quantile(0.75)),
                            "status": "ok",
                            "reason": "",
                        }
                    )

            if window_name == "overall" and feature_set_name == "core":
                overall_core_for_biplot = pd.DataFrame(
                    {
                        "station_slug": score_df["station_slug"],
                        "PC1": score_df["PC1"],
                        "PC2": score_df["PC2"] if "PC2" in score_df.columns else np.nan,
                    }
                )
                if pca.n_components_ >= 2:
                    loadings_for_plot = pd.DataFrame(
                        {
                            "variable": feature_columns,
                            "PC1": pca.components_[0],
                            "PC2": pca.components_[1],
                        }
                    )
                    overall_core_for_biplot = overall_core_for_biplot.merge(
                        loadings_for_plot.assign(_join_key=1), how="cross"
                    )

    explained_df = pd.DataFrame(explained_rows)
    loading_df = pd.DataFrame(loading_rows)
    scores_df = pd.DataFrame(score_rows)
    station_summary_df = pd.DataFrame(station_summary_rows)

    return explained_df, loading_df, scores_df, station_summary_df, overall_core_for_biplot


def _benchmark_value_rows(
    *,
    window: str,
    station_slug: str,
    reference_station_slug: str,
    variable: str,
    status: str,
    reason: str,
    aligned_hours: int,
    complete_case_fraction: float,
    values: Dict[str, float],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for metric in BENCHMARK_METRICS:
        rows.append(
            {
                "window": window,
                "feature_set": "core",
                "station_slug": station_slug,
                "reference_station_slug": reference_station_slug,
                "variable": variable,
                "metric": metric,
                "value": values.get(metric, np.nan),
                "aligned_hours": aligned_hours,
                "complete_case_fraction": complete_case_fraction,
                "status": status,
                "reason": reason,
            }
        )
    return rows


def run_benchmarking(df: pd.DataFrame, reference_slug: str, expected_park_slugs: Iterable[str]) -> pd.DataFrame:
    benchmark_rows: List[Dict[str, object]] = []
    windows = _window_slices(df)

    present_stations = set(df["station_slug"].dropna().astype(str).tolist())

    for window_name, window_df in windows.items():
        ref_df = window_df[window_df["station_slug"] == reference_slug]

        for park_slug in sorted(set(expected_park_slugs)):
            if park_slug not in present_stations:
                for variable in PRIMARY_FEATURES:
                    benchmark_rows.extend(
                        _benchmark_value_rows(
                            window=window_name,
                            station_slug=park_slug,
                            reference_station_slug=reference_slug,
                            variable=variable,
                            status="skipped_station_missing",
                            reason="Parks station missing from scrubbed dataset",
                            aligned_hours=0,
                            complete_case_fraction=np.nan,
                            values={},
                        )
                    )
                continue

            park_df = window_df[window_df["station_slug"] == park_slug]
            for variable in PRIMARY_FEATURES:
                merged = (
                    park_df[["datetime_utc", variable]]
                    .rename(columns={variable: "park_value"})
                    .merge(
                        ref_df[["datetime_utc", variable]].rename(columns={variable: "ref_value"}),
                        on="datetime_utc",
                        how="inner",
                    )
                )

                aligned_hours = int(len(merged))
                valid = merged["park_value"].notna() & merged["ref_value"].notna()
                valid_df = merged.loc[valid].copy()
                valid_hours = int(len(valid_df))
                complete_case_fraction = valid_hours / aligned_hours if aligned_hours > 0 else np.nan

                if (
                    valid_hours < MIN_ALIGNED_HOURS_PER_WINDOW
                    or (pd.notna(complete_case_fraction) and complete_case_fraction < MIN_COMPLETE_CASE_FRACTION)
                ):
                    reason = (
                        f"insufficient overlap (valid_hours={valid_hours}, "
                        f"complete_case_fraction={complete_case_fraction:.3f})"
                    )
                    benchmark_rows.extend(
                        _benchmark_value_rows(
                            window=window_name,
                            station_slug=park_slug,
                            reference_station_slug=reference_slug,
                            variable=variable,
                            status="skipped_insufficient_overlap",
                            reason=reason,
                            aligned_hours=aligned_hours,
                            complete_case_fraction=complete_case_fraction,
                            values={},
                        )
                    )
                    continue

                park_values = valid_df["park_value"].astype(float)
                ref_values = valid_df["ref_value"].astype(float)

                values = {
                    "mae": float((park_values - ref_values).abs().mean()),
                    "rmse": float(np.sqrt(np.mean((park_values - ref_values) ** 2))),
                    "bias": float((park_values - ref_values).mean()),
                    "pearson_r": float(park_values.corr(ref_values)),
                    "spearman_r": float(park_values.rank().corr(ref_values.rank())),
                    "ks_stat": _ks_statistic(park_values, ref_values),
                }

                benchmark_rows.extend(
                    _benchmark_value_rows(
                        window=window_name,
                        station_slug=park_slug,
                        reference_station_slug=reference_slug,
                        variable=variable,
                        status="ok",
                        reason="",
                        aligned_hours=aligned_hours,
                        complete_case_fraction=complete_case_fraction,
                        values=values,
                    )
                )

    return pd.DataFrame(benchmark_rows)


def plot_biplot(scores_df: pd.DataFrame, loadings_df: pd.DataFrame, output_path: Path) -> bool:
    score_subset = scores_df[
        (scores_df["window"] == "overall")
        & (scores_df["feature_set"] == "core")
        & (scores_df["status"] == "ok")
        & (scores_df["pc"].isin(["PC1", "PC2"]))
    ].copy()

    if score_subset.empty:
        return False

    wide_scores = (
        score_subset.pivot_table(
            index=["datetime_utc", "station_slug"],
            columns="pc",
            values="score",
            aggfunc="first",
        )
        .reset_index()
    )

    if "PC1" not in wide_scores.columns or "PC2" not in wide_scores.columns:
        return False

    loading_subset = loadings_df[
        (loadings_df["window"] == "overall")
        & (loadings_df["feature_set"] == "core")
        & (loadings_df["status"] == "ok")
        & (loadings_df["component"].isin(["PC1", "PC2"]))
    ].copy()

    if loading_subset.empty:
        return False

    loadings_wide = (
        loading_subset.pivot_table(index="variable", columns="component", values="loading", aggfunc="first")
        .reset_index()
    )

    if "PC1" not in loadings_wide.columns or "PC2" not in loadings_wide.columns:
        return False

    plt.figure(figsize=(10, 8))

    station_groups = wide_scores.groupby("station_slug", dropna=False)
    for station_slug, station_df in station_groups:
        plt.scatter(station_df["PC1"], station_df["PC2"], alpha=0.35, s=18, label=station_slug)

    scale = 2.5
    for row in loadings_wide.itertuples(index=False):
        x = float(row.PC1) * scale
        y = float(row.PC2) * scale
        plt.arrow(0, 0, x, y, color="black", width=0.004, head_width=0.08, alpha=0.8)
        plt.text(x * 1.08, y * 1.08, str(row.variable), fontsize=9)

    plt.axhline(0, color="grey", linewidth=0.8)
    plt.axvline(0, color="grey", linewidth=0.8)
    plt.title("PCA Biplot (overall, core features)")
    plt.xlabel("PC1 score")
    plt.ylabel("PC2 score")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return True


def main() -> int:
    ensure_directories()
    log_path = LOGS_DIR / f"model_redundancy_{datetime.now(timezone.utc):%Y%m%d}.log"
    logger = setup_logging("05_model_redundancy", log_file_path=log_path)

    logger.info("Step 6 Model Redundancy started.")
    logger.info("Input scrubbed dataset: %s", INPUT_HOURLY)
    logger.info(
        "Config | reference=%s min_aligned_hours=%s min_complete_case_fraction=%.2f seed=%s",
        BENCHMARK_REFERENCE_STATION_SLUG,
        MIN_ALIGNED_HOURS_PER_WINDOW,
        MIN_COMPLETE_CASE_FRACTION,
        REDUNDANCY_RANDOM_SEED,
    )

    df = load_and_validate(INPUT_HOURLY)
    df = apply_qc_gating(df)
    df = add_time_windows(df)

    reference_slug, expected_park_slugs, missing_parks = validate_station_roster(df)
    if missing_parks:
        logger.warning("Expected Parks stations missing from scrubbed data: %s", missing_parks)

    feature_sets = build_feature_sets(df)
    logger.info("Feature sets configured: %s", sorted(feature_sets.keys()))

    explained_df, loadings_df, scores_df, station_summary_df, _ = run_pca_models(df, feature_sets)
    benchmarks_df = run_benchmarking(df, reference_slug, expected_park_slugs)

    biplot_written = plot_biplot(scores_df, loadings_df, OUTPUT_BIPLOT)

    _write_csv(explained_df, OUTPUT_EXPLAINED)
    _write_csv(loadings_df, OUTPUT_LOADINGS)
    _write_csv(scores_df, OUTPUT_SCORES)
    _write_csv(station_summary_df, OUTPUT_STATION_SUMMARY)
    _write_csv(benchmarks_df, OUTPUT_BENCHMARKS)

    logger.info("Wrote table: %s", OUTPUT_EXPLAINED)
    logger.info("Wrote table: %s", OUTPUT_LOADINGS)
    logger.info("Wrote table: %s", OUTPUT_SCORES)
    logger.info("Wrote table: %s", OUTPUT_STATION_SUMMARY)
    logger.info("Wrote table: %s", OUTPUT_BENCHMARKS)
    if biplot_written:
        logger.info("Wrote figure: %s", OUTPUT_BIPLOT)
    else:
        logger.warning("Skipped biplot generation because PC1/PC2 results were unavailable.")

    ok_benchmarks = int((benchmarks_df.get("status") == "ok").sum()) if not benchmarks_df.empty else 0
    skipped_benchmarks = (
        int((benchmarks_df.get("status") != "ok").sum()) if not benchmarks_df.empty else 0
    )
    logger.info(
        "Summary | rows=%s stations=%s pca_rows=%s benchmark_rows=%s benchmark_ok=%s benchmark_skipped=%s",
        len(df),
        df["station_slug"].nunique(),
        len(scores_df),
        len(benchmarks_df),
        ok_benchmarks,
        skipped_benchmarks,
    )
    logger.info("Step 6 Model Redundancy completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
