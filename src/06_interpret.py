"""Step 7 interpret stage: probabilistic risk and station removal strategy."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from zoneinfo import ZoneInfo

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import (
    FIGURES_DIR,
    FWI_SEASON_MONTHS,
    INTERPRET_ALPHA,
    INTERPRET_BOOTSTRAP_BLOCK_DAYS,
    INTERPRET_BOOTSTRAP_N_RESAMPLES,
    INTERPRET_CANDIDATE_STATION_SLUGS,
    INTERPRET_DAILY_THRESHOLDS,
    INTERPRET_HOURLY_THRESHOLDS,
    INTERPRET_MIN_ALIGNED_DAILY_PER_WINDOW,
    INTERPRET_MIN_ALIGNED_HOURLY_PER_WINDOW,
    INTERPRET_RANDOM_SEED,
    INTERPRET_REFERENCE_STATION_SLUG,
    LOGS_DIR,
    PLANS_DIR,
    SCRUBBED_DIR,
    SEASON_BY_MONTH,
    TABLES_DIR,
    ensure_directories,
    setup_logging,
)

matplotlib.use("Agg")

INPUT_HOURLY = SCRUBBED_DIR / "hourly_weather_utc.csv"
INPUT_DAILY_FWI = TABLES_DIR / "model_fwi_daily.csv"
INPUT_BENCHMARKS = TABLES_DIR / "model_hourly_benchmarks.csv"
INPUT_PCA_LOADINGS = TABLES_DIR / "model_pca_loadings.csv"
INPUT_PCA_EXPLAINED = TABLES_DIR / "model_pca_explained_variance.csv"

OUTPUT_RISK_TABLE = TABLES_DIR / "interpret_redundancy_risk.csv"
OUTPUT_KDE_FIGURE = FIGURES_DIR / "interpret_kde_error_distribution.png"
OUTPUT_FINDINGS_MD = PLANS_DIR / "07_interpret_findings.md"

HALIFAX_TZ = ZoneInfo("America/Halifax")
WINDOWS = ["overall", "winter", "spring", "summer", "fall", "fwi_season"]


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    truthy = {"true", "1", "t", "yes", "y"}
    normalized = series.fillna(False).astype(str).str.strip().str.lower()
    return normalized.isin(truthy)


def _window_mask(df: pd.DataFrame, window: str) -> pd.Series:
    if window == "overall":
        return pd.Series(True, index=df.index)
    if window == "fwi_season":
        return df["month_local"].isin(FWI_SEASON_MONTHS)
    season_months = {month for month, season in SEASON_BY_MONTH.items() if season == window}
    return df["month_local"].isin(season_months)


def _required_hourly_columns() -> List[str]:
    cols = ["datetime_utc", "station_slug"]
    cols.extend(INTERPRET_HOURLY_THRESHOLDS.keys())
    for variable in INTERPRET_HOURLY_THRESHOLDS.keys():
        cols.append(f"{variable}_failed_qc")
    return cols


def load_hourly(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required hourly file is missing: {path}")

    hourly = pd.read_csv(path, low_memory=False)
    missing = [col for col in _required_hourly_columns() if col not in hourly.columns]
    if missing:
        raise ValueError(f"Missing required hourly columns: {missing}")

    hourly["datetime_utc"] = pd.to_datetime(hourly["datetime_utc"], utc=True, errors="coerce")
    bad_dt = int(hourly["datetime_utc"].isna().sum())
    if bad_dt > 0:
        raise ValueError(f"Found {bad_dt} invalid datetime_utc values in hourly file")

    dupes = int(hourly.duplicated(subset=["station_slug", "datetime_utc"]).sum())
    if dupes > 0:
        raise ValueError(f"Found duplicated (station_slug, datetime_utc) keys: {dupes}")

    for variable in INTERPRET_HOURLY_THRESHOLDS:
        hourly[variable] = pd.to_numeric(hourly[variable], errors="coerce")
        failed_col = f"{variable}_failed_qc"
        hourly[failed_col] = _coerce_bool(hourly[failed_col])
        hourly.loc[hourly[failed_col], variable] = np.nan

    local_dt = hourly["datetime_utc"].dt.tz_convert(HALIFAX_TZ)
    hourly["month_local"] = local_dt.dt.month
    return hourly.sort_values(["station_slug", "datetime_utc"]).reset_index(drop=True)


def _required_daily_columns() -> List[str]:
    return ["station_slug", "date_local", *INTERPRET_DAILY_THRESHOLDS.keys()]


def load_daily(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required daily FWI file is missing: {path}")

    daily = pd.read_csv(path, low_memory=False)
    missing = [col for col in _required_daily_columns() if col not in daily.columns]
    if missing:
        raise ValueError(f"Missing required daily columns: {missing}")

    daily["date_local"] = pd.to_datetime(daily["date_local"], errors="coerce")
    bad_dates = int(daily["date_local"].isna().sum())
    if bad_dates > 0:
        raise ValueError(f"Found {bad_dates} invalid date_local values in daily file")

    for variable in INTERPRET_DAILY_THRESHOLDS:
        daily[variable] = pd.to_numeric(daily[variable], errors="coerce")

    daily["month_local"] = daily["date_local"].dt.month
    return daily.sort_values(["station_slug", "date_local"]).reset_index(drop=True)


def _rmse(values: np.ndarray) -> float:
    if values.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(values**2)))


def _bias(values: np.ndarray) -> float:
    if values.size == 0:
        return np.nan
    return float(np.mean(values))


def _exceed_prob(values: np.ndarray, threshold: float) -> float:
    if values.size == 0:
        return np.nan
    return float(np.mean(np.abs(values) > threshold))


def _circular_block_bootstrap_sample(
    values: np.ndarray,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = values.size
    if n == 0:
        return values

    effective_block = max(1, min(block_length, n))
    starts = rng.integers(0, n, size=int(np.ceil(n / effective_block)))

    sampled_blocks: List[np.ndarray] = []
    for start in starts:
        stop = start + effective_block
        if stop <= n:
            sampled_blocks.append(values[start:stop])
        else:
            tail = values[start:]
            head = values[: stop - n]
            sampled_blocks.append(np.concatenate([tail, head]))

    sample = np.concatenate(sampled_blocks)[:n]
    return sample


def _bootstrap_ci(
    values: np.ndarray,
    threshold: float,
    block_length: int,
    n_resamples: int,
    alpha: float,
    rng: np.random.Generator,
) -> Dict[str, float]:
    if values.size == 0:
        return {
            "exceed_prob_ci_low": np.nan,
            "exceed_prob_ci_high": np.nan,
            "rmse_ci_low": np.nan,
            "rmse_ci_high": np.nan,
            "bias_ci_low": np.nan,
            "bias_ci_high": np.nan,
        }

    exceed_samples = np.empty(n_resamples, dtype=float)
    rmse_samples = np.empty(n_resamples, dtype=float)
    bias_samples = np.empty(n_resamples, dtype=float)

    for i in range(n_resamples):
        sample = _circular_block_bootstrap_sample(values, block_length, rng)
        exceed_samples[i] = _exceed_prob(sample, threshold)
        rmse_samples[i] = _rmse(sample)
        bias_samples[i] = _bias(sample)

    lo = alpha / 2.0
    hi = 1.0 - alpha / 2.0

    return {
        "exceed_prob_ci_low": float(np.quantile(exceed_samples, lo)),
        "exceed_prob_ci_high": float(np.quantile(exceed_samples, hi)),
        "rmse_ci_low": float(np.quantile(rmse_samples, lo)),
        "rmse_ci_high": float(np.quantile(rmse_samples, hi)),
        "bias_ci_low": float(np.quantile(bias_samples, lo)),
        "bias_ci_high": float(np.quantile(bias_samples, hi)),
    }


def _skipped_row(
    *,
    candidate: str,
    reference: str,
    window: str,
    variable: str,
    threshold: float,
    n_aligned: int,
) -> Dict[str, object]:
    return {
        "candidate_station_slug": candidate,
        "reference_station_slug": reference,
        "window": window,
        "variable": variable,
        "threshold": threshold,
        "n_aligned": n_aligned,
        "exceed_prob": np.nan,
        "rmse": np.nan,
        "bias": np.nan,
        "exceed_prob_ci_low": np.nan,
        "exceed_prob_ci_high": np.nan,
        "rmse_ci_low": np.nan,
        "rmse_ci_high": np.nan,
        "bias_ci_low": np.nan,
        "bias_ci_high": np.nan,
        "frequency": pd.NA,
        "status": "skipped_insufficient_overlap",
        "decision_safe_to_remove": False,
    }


def _metric_row(
    *,
    candidate: str,
    reference: str,
    window: str,
    variable: str,
    threshold: float,
    values: np.ndarray,
    frequency: str,
    block_length: int,
    rng: np.random.Generator,
) -> Dict[str, object]:
    ci = _bootstrap_ci(
        values=values,
        threshold=threshold,
        block_length=block_length,
        n_resamples=INTERPRET_BOOTSTRAP_N_RESAMPLES,
        alpha=INTERPRET_ALPHA,
        rng=rng,
    )

    return {
        "candidate_station_slug": candidate,
        "reference_station_slug": reference,
        "window": window,
        "variable": variable,
        "threshold": threshold,
        "n_aligned": int(values.size),
        "exceed_prob": _exceed_prob(values, threshold),
        "rmse": _rmse(values),
        "bias": _bias(values),
        **ci,
        "frequency": frequency,
        "status": "ok",
        "decision_safe_to_remove": False,
    }


def _aligned_delta_frame(
    data: pd.DataFrame,
    time_key: str,
    candidate: str,
    reference: str,
    variables: Iterable[str],
    window: str,
) -> pd.DataFrame:
    window_df = data.loc[_window_mask(data, window)].copy()

    candidate_df = window_df[window_df["station_slug"] == candidate][[time_key, *variables]].copy()
    reference_df = window_df[window_df["station_slug"] == reference][[time_key, *variables]].copy()

    if candidate_df.empty or reference_df.empty:
        return pd.DataFrame(columns=[time_key, *variables])

    candidate_df = candidate_df.rename(columns={v: f"{v}_cand" for v in variables})
    reference_df = reference_df.rename(columns={v: f"{v}_ref" for v in variables})

    merged = candidate_df.merge(reference_df, on=time_key, how="inner")
    for variable in variables:
        merged[variable] = merged[f"{variable}_cand"] - merged[f"{variable}_ref"]

    keep_cols = [time_key, *variables]
    return merged[keep_cols].sort_values(time_key).reset_index(drop=True)


def analyze_frequency(
    *,
    data: pd.DataFrame,
    time_key: str,
    frequency: str,
    thresholds: Dict[str, float],
    reference_slug: str,
    candidate_slugs: Iterable[str],
    min_aligned: int,
    block_days: int,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    kde_rows: List[Dict[str, object]] = []

    if frequency == "hourly":
        block_length = max(1, block_days * 24)
    else:
        block_length = max(1, block_days)

    variable_list = list(thresholds.keys())

    for candidate in candidate_slugs:
        for window in WINDOWS:
            delta_df = _aligned_delta_frame(
                data=data,
                time_key=time_key,
                candidate=candidate,
                reference=reference_slug,
                variables=variable_list,
                window=window,
            )

            for variable, threshold in thresholds.items():
                valid = delta_df[variable].dropna().to_numpy(dtype=float)
                if valid.size < min_aligned:
                    rows.append(
                        _skipped_row(
                            candidate=candidate,
                            reference=reference_slug,
                            window=window,
                            variable=variable,
                            threshold=threshold,
                            n_aligned=int(valid.size),
                        )
                    )
                    rows[-1]["frequency"] = frequency
                    continue

                rows.append(
                    _metric_row(
                        candidate=candidate,
                        reference=reference_slug,
                        window=window,
                        variable=variable,
                        threshold=threshold,
                        values=valid,
                        frequency=frequency,
                        block_length=block_length,
                        rng=rng,
                    )
                )

                for value in valid:
                    kde_rows.append(
                        {
                            "candidate_station_slug": candidate,
                            "variable": variable,
                            "delta": value,
                            "frequency": frequency,
                        }
                    )

            any_delta = delta_df[variable_list].dropna(how="any")
            any_threshold = np.nan
            any_values = np.nan
            if not any_delta.empty:
                crit_matrix = np.column_stack(
                    [np.abs(any_delta[var].to_numpy(dtype=float)) > thresholds[var] for var in variable_list]
                )
                any_values = crit_matrix.any(axis=1).astype(float)

            if isinstance(any_values, np.ndarray) and any_values.size >= min_aligned:
                ci = _bootstrap_ci(
                    values=any_values,
                    threshold=0.5,
                    block_length=block_length,
                    n_resamples=INTERPRET_BOOTSTRAP_N_RESAMPLES,
                    alpha=INTERPRET_ALPHA,
                    rng=rng,
                )
                rows.append(
                    {
                        "candidate_station_slug": candidate,
                        "reference_station_slug": reference_slug,
                        "window": window,
                        "variable": "any_critical",
                        "threshold": any_threshold,
                        "n_aligned": int(any_values.size),
                        "exceed_prob": float(np.mean(any_values > 0.5)),
                        "rmse": np.nan,
                        "bias": np.nan,
                        "exceed_prob_ci_low": ci["exceed_prob_ci_low"],
                        "exceed_prob_ci_high": ci["exceed_prob_ci_high"],
                        "rmse_ci_low": np.nan,
                        "rmse_ci_high": np.nan,
                        "bias_ci_low": np.nan,
                        "bias_ci_high": np.nan,
                        "frequency": frequency,
                        "status": "ok",
                        "decision_safe_to_remove": False,
                    }
                )
            else:
                rows.append(
                    {
                        "candidate_station_slug": candidate,
                        "reference_station_slug": reference_slug,
                        "window": window,
                        "variable": "any_critical",
                        "threshold": any_threshold,
                        "n_aligned": int(any_values.size) if isinstance(any_values, np.ndarray) else 0,
                        "exceed_prob": np.nan,
                        "rmse": np.nan,
                        "bias": np.nan,
                        "exceed_prob_ci_low": np.nan,
                        "exceed_prob_ci_high": np.nan,
                        "rmse_ci_low": np.nan,
                        "rmse_ci_high": np.nan,
                        "bias_ci_low": np.nan,
                        "bias_ci_high": np.nan,
                        "frequency": frequency,
                        "status": "skipped_insufficient_overlap",
                        "decision_safe_to_remove": False,
                    }
                )

    return pd.DataFrame(rows), pd.DataFrame(kde_rows)


def _finalize_decisions(risk_df: pd.DataFrame) -> pd.DataFrame:
    out = risk_df.copy()

    decisions: Dict[str, bool] = {}
    for candidate, group in out.groupby("candidate_station_slug", dropna=False):
        all_ok = bool((group["status"] == "ok").all())
        if not all_ok:
            decisions[str(candidate)] = False
            continue

        exceeded = group["exceed_prob"] >= INTERPRET_ALPHA
        decisions[str(candidate)] = not bool(exceeded.fillna(True).any())

    out["decision_safe_to_remove"] = out["candidate_station_slug"].map(decisions).fillna(False)
    return out


def build_kde_figure(kde_df: pd.DataFrame, output_path: Path) -> None:
    if kde_df.empty:
        raise ValueError("No valid deltas were available to build KDE figure.")

    stations = sorted(kde_df["candidate_station_slug"].dropna().astype(str).unique().tolist())
    variables = sorted(kde_df["variable"].dropna().astype(str).unique().tolist())

    nrows = max(1, len(variables))
    ncols = max(1, len(stations))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.4 * ncols, 2.5 * nrows), squeeze=False)

    for row_idx, variable in enumerate(variables):
        for col_idx, station in enumerate(stations):
            axis = axes[row_idx][col_idx]
            subset = kde_df[
                (kde_df["variable"] == variable)
                & (kde_df["candidate_station_slug"] == station)
            ]["delta"].dropna()

            if subset.empty:
                axis.text(0.5, 0.5, "No data", ha="center", va="center", transform=axis.transAxes)
            elif subset.nunique() <= 1:
                axis.hist(subset.to_numpy(dtype=float), bins=10, color="#7f7f7f", alpha=0.7)
            else:
                sns.kdeplot(x=subset.to_numpy(dtype=float), ax=axis, fill=True, color="#4c78a8", linewidth=1.4)

            axis.axvline(0, color="black", linestyle="--", linewidth=0.8)
            axis.set_title(f"{station} | {variable}", fontsize=9)
            axis.set_xlabel("Delta (candidate - reference)")
            axis.set_ylabel("Density")
            axis.grid(alpha=0.2)

    fig.suptitle("Phase 7 Delta Distributions (KDE where supported)", fontsize=12)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def write_findings_markdown(
    risk_df: pd.DataFrame,
    pca_loadings: pd.DataFrame,
    pca_explained: pd.DataFrame,
    benchmarks: pd.DataFrame,
    output_path: Path,
) -> None:
    lines: List[str] = []
    generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines.append("# 07_interpret_findings.md")
    lines.append("")
    lines.append("## Phase 7 Interpretation Findings")
    lines.append("")
    lines.append(f"Generated: {generated_utc}")
    lines.append("")
    lines.append("### Operational Definition of Critical Loss")
    lines.append("")
    lines.append(f"- Probability target: exceedance probability below alpha={INTERPRET_ALPHA:.2f}.")
    lines.append("- Exceedance definition: P(|Delta X| > threshold) using empirical exceedance rates.")
    lines.append("- Confidence intervals: 95% block-bootstrap CIs with 7-day blocks.")
    lines.append("- Decision rule: conservative worst-case across windows and metrics.")
    lines.append("")

    lines.append("### Thresholds Used")
    lines.append("")
    lines.append("- Hourly thresholds:")
    for variable, value in INTERPRET_HOURLY_THRESHOLDS.items():
        lines.append(f"  - {variable}: {value}")
    lines.append("- Daily thresholds:")
    for variable, value in INTERPRET_DAILY_THRESHOLDS.items():
        lines.append(f"  - {variable}: {value}")
    lines.append("")

    lines.append("### Candidate Recommendations")
    lines.append("")
    for candidate in sorted(risk_df["candidate_station_slug"].dropna().unique().tolist()):
        candidate_rows = risk_df[risk_df["candidate_station_slug"] == candidate].copy()
        decision = bool(candidate_rows["decision_safe_to_remove"].iloc[0]) if not candidate_rows.empty else False

        ok_rows = candidate_rows[candidate_rows["status"] == "ok"].copy()
        if ok_rows.empty:
            lines.append(f"- {candidate}: Not Safe to Remove (no evaluable rows).")
            continue

        worst_idx = ok_rows["exceed_prob"].fillna(-np.inf).idxmax()
        worst_row = ok_rows.loc[worst_idx]
        lines.append(
            "- "
            f"{candidate}: {'Safe to Remove' if decision else 'Not Safe to Remove'} "
            f"(worst window={worst_row['window']}, variable={worst_row['variable']}, "
            f"exceed_prob={worst_row['exceed_prob']:.4f}, "
            f"CI=[{worst_row['exceed_prob_ci_low']:.4f}, {worst_row['exceed_prob_ci_high']:.4f}])."
        )

    skipped_count = int((risk_df["status"] != "ok").sum())
    lines.append("")
    lines.append("### Data Gating and Overlap Notes")
    lines.append("")
    lines.append(f"- Skipped rows due to overlap gating: {skipped_count}.")
    lines.append(
        f"- Minimum aligned sample thresholds: hourly={INTERPRET_MIN_ALIGNED_HOURLY_PER_WINDOW}, "
        f"daily={INTERPRET_MIN_ALIGNED_DAILY_PER_WINDOW}."
    )
    lines.append("")

    lines.append("### KDE Interpretation Notes")
    lines.append("")
    lines.append("- KDE curves visualize delta distributions only; exceedance probabilities are based on ECDF rates.")
    lines.append("- Precipitation distributions are often zero-inflated, so KDE shape is only a supporting visual cue.")
    lines.append("- See figure: outputs/figures/interpret_kde_error_distribution.png")
    lines.append("")

    lines.append("### PCA Context (Phase 6)")
    lines.append("")
    if not pca_explained.empty:
        subset = pca_explained[
            (pca_explained.get("window") == "overall")
            & (pca_explained.get("feature_set") == "core")
            & (pca_explained.get("status") == "ok")
        ].copy()
        if not subset.empty and "component" in subset.columns:
            pc2 = subset[subset["component"].isin(["PC1", "PC2"])].copy()
            if not pc2.empty and "explained_variance_ratio" in pc2.columns:
                cumulative = float(pc2["explained_variance_ratio"].sum())
                lines.append(f"- Overall core PC1+PC2 explained variance: {cumulative:.4f}.")

    if not pca_loadings.empty:
        load = pca_loadings[
            (pca_loadings.get("window") == "overall")
            & (pca_loadings.get("feature_set") == "core")
            & (pca_loadings.get("component") == "PC1")
            & (pca_loadings.get("status") == "ok")
        ].copy()
        if not load.empty and "loading" in load.columns:
            load["abs_loading"] = load["loading"].abs()
            top = load.sort_values("abs_loading", ascending=False).head(2)
            if not top.empty:
                top_text = ", ".join(
                    f"{row.variable} ({row.loading:.3f})" for row in top.itertuples(index=False)
                )
                lines.append(f"- Strongest PC1 loading context variables: {top_text}.")
    lines.append("")

    lines.append("### Hourly Benchmark Context (Phase 6)")
    lines.append("")
    if not benchmarks.empty:
        bench = benchmarks[
            (benchmarks.get("window") == "overall")
            & (benchmarks.get("metric") == "rmse")
            & (benchmarks.get("status") == "ok")
            & (benchmarks.get("station_slug").isin(INTERPRET_CANDIDATE_STATION_SLUGS))
        ].copy()
        if not bench.empty:
            for row in bench.sort_values(["station_slug", "variable"]).itertuples(index=False):
                lines.append(
                    f"- {row.station_slug} vs {row.reference_station_slug} | "
                    f"{row.variable} RMSE={float(row.value):.3f} (aligned_hours={int(row.aligned_hours)})."
                )
    lines.append("")
    lines.append("### Next Step Contract")
    lines.append("")
    lines.append(
        "- Use outputs/tables/interpret_redundancy_risk.csv as the decision table for any station rationalization action."
    )
    lines.append(
        "- If any candidate is Not Safe to Remove, retain that station and prioritize targeted maintenance or variable-specific backup design."
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ensure_directories()
    log_path = LOGS_DIR / f"interpret_{datetime.now(timezone.utc):%Y%m%d}.log"
    logger = setup_logging("06_interpret", log_file_path=log_path)

    logger.info("Step 7 Interpret started.")
    logger.info("Config | reference=%s candidates=%s alpha=%.3f", INTERPRET_REFERENCE_STATION_SLUG, INTERPRET_CANDIDATE_STATION_SLUGS, INTERPRET_ALPHA)

    hourly = load_hourly(INPUT_HOURLY)
    daily = load_daily(INPUT_DAILY_FWI)

    rng = np.random.default_rng(INTERPRET_RANDOM_SEED)

    hourly_risk, hourly_kde = analyze_frequency(
        data=hourly,
        time_key="datetime_utc",
        frequency="hourly",
        thresholds=INTERPRET_HOURLY_THRESHOLDS,
        reference_slug=INTERPRET_REFERENCE_STATION_SLUG,
        candidate_slugs=INTERPRET_CANDIDATE_STATION_SLUGS,
        min_aligned=INTERPRET_MIN_ALIGNED_HOURLY_PER_WINDOW,
        block_days=INTERPRET_BOOTSTRAP_BLOCK_DAYS,
        rng=rng,
    )

    daily_risk, daily_kde = analyze_frequency(
        data=daily,
        time_key="date_local",
        frequency="daily",
        thresholds=INTERPRET_DAILY_THRESHOLDS,
        reference_slug=INTERPRET_REFERENCE_STATION_SLUG,
        candidate_slugs=INTERPRET_CANDIDATE_STATION_SLUGS,
        min_aligned=INTERPRET_MIN_ALIGNED_DAILY_PER_WINDOW,
        block_days=INTERPRET_BOOTSTRAP_BLOCK_DAYS,
        rng=rng,
    )

    risk_df = pd.concat([hourly_risk, daily_risk], ignore_index=True)
    risk_df = _finalize_decisions(risk_df)
    risk_df = risk_df.sort_values(
        ["candidate_station_slug", "frequency", "window", "variable"],
        na_position="last",
    ).reset_index(drop=True)

    kde_df = pd.concat([hourly_kde, daily_kde], ignore_index=True)

    OUTPUT_RISK_TABLE.parent.mkdir(parents=True, exist_ok=True)
    risk_df.to_csv(OUTPUT_RISK_TABLE, index=False)

    if not kde_df.empty:
        build_kde_figure(kde_df, OUTPUT_KDE_FIGURE)

    pca_loadings = _read_optional_csv(INPUT_PCA_LOADINGS)
    pca_explained = _read_optional_csv(INPUT_PCA_EXPLAINED)
    benchmarks = _read_optional_csv(INPUT_BENCHMARKS)
    write_findings_markdown(risk_df, pca_loadings, pca_explained, benchmarks, OUTPUT_FINDINGS_MD)

    decision_counts = risk_df[["candidate_station_slug", "decision_safe_to_remove"]].drop_duplicates()
    logger.info("Wrote table: %s", OUTPUT_RISK_TABLE)
    if not kde_df.empty:
        logger.info("Wrote figure: %s", OUTPUT_KDE_FIGURE)
    logger.info("Wrote findings: %s", OUTPUT_FINDINGS_MD)
    logger.info(
        "Summary | risk_rows=%s | kde_rows=%s | decisions=%s",
        len(risk_df),
        len(kde_df),
        decision_counts.to_dict(orient="records"),
    )
    logger.info("Step 7 Interpret completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
