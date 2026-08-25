"""Shortwave-defined nighttime and rule-robustness analysis.

This script preserves the previously defined four-regime framework.  It changes
only the nighttime sampling definition used by the persistent/diel criteria:
the former fixed 04:00--06:00 window is replaced by periods identified from the
measured shortwave radiation.  Calendar seasons are used only after daily
classification for interpretation, never as classification inputs.

DeltaT and MLD are treated as complementary temperature-structure metrics.
St and N2 remain threshold-independent stability diagnostics calculated by the
existing discrete-profile workflow.  Rib is not used.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_DATA = REPOSITORY_ROOT / "data" / "raw_or_input" / "full_year"
OUTPUT_DIR = REPOSITORY_ROOT / "results" / "reproduced" / "regime_classification"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW = PRIVATE_DATA / "meteorology_water_temperature_2024_10min.csv"
STABILITY = PRIVATE_DATA / "figure5_indices_5depth_source.csv"
PROFILE = REPOSITORY_ROOT / "data" / "processed" / "figures" / "Figure4_representative_profiles.csv"
MLD = REPOSITORY_ROOT / "results" / "reproduced" / "sensitivity_oat" / "MLD_all_scenarios_10min.csv"

LEGACY_CANDIDATES = [
    HERE / "base_regime_classification_analysis.py",
    HERE.parent / "14_RegimeClassification" / "regime_classification_analysis.py",
    WORKSPACE / "analysis_work" / "CSITE_revision_regime_classification_4A" / "regime_classification_analysis.py",
]
LEGACY_PATH = next((p for p in LEGACY_CANDIDATES if p.exists()), LEGACY_CANDIDATES[-1])

SHORTWAVE_COLUMN = "简易总辐射(W/m²)"
NIGHTTIME_THRESHOLD_W_M2 = 0.0
BASELINE_THRESHOLD_C = 0.20
DEPTH_MAX_M = 2.25
SENSOR_SPACING_M = 0.50
DECAY_MIN_RUN_DAYS = 7
DECAY_LINK_TO_MIXED_DAYS = 7
ROLLING_WINDOW_DAYS = 7

REGIME_ORDER = [
    "Near-isothermal mixing",
    "Persistent stratification",
    "Diurnally oscillating stratification",
    "Decaying stratification",
    "Transition/unclassified",
]


def load_legacy_module():
    if not LEGACY_PATH.exists():
        raise FileNotFoundError(f"Previous Stage-4A method file not found: {LEGACY_PATH}")
    spec = importlib.util.spec_from_file_location("legacy_regime", LEGACY_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load legacy module: {LEGACY_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LEGACY = load_legacy_module()


def read_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(RAW, parse_dates=["Date"], encoding="utf-8-sig").set_index("Date").sort_index()
    stability = pd.read_csv(STABILITY, parse_dates=["Date"], encoding="utf-8-sig").set_index("Date").sort_index()
    mld = pd.read_csv(MLD, parse_dates=["Date"], encoding="utf-8-sig").set_index("Date").sort_index()
    return raw, stability, mld


def shortwave_night_mask(rsw: pd.Series, threshold: float) -> pd.Series:
    """Return a measured-radiation nighttime mask.

    The baseline threshold is exactly zero because the sensor has no negative
    values, records exact zero on all 366 days, and has no positive values in
    (0, 1] W m-2.  Positive thresholds are retained only for diagnostics.
    """
    values = pd.to_numeric(rsw, errors="coerce")
    if threshold == 0:
        return values.eq(0)
    return values.le(threshold)


def align_shortwave(rsw: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    """Align measured shortwave to another time index without interpolation."""
    aligned = rsw.reindex(target_index)
    if aligned.isna().any():
        aligned = rsw.reindex(target_index, method="nearest", tolerance=pd.Timedelta(minutes=5))
    if aligned.isna().any():
        raise ValueError("Shortwave could not be aligned to all requested timestamps within 5 min.")
    return aligned


def daily_metrics_with_nighttime(
    raw: pd.DataFrame,
    stability: pd.DataFrame,
    mld: pd.DataFrame,
    nighttime_threshold: float,
) -> pd.DataFrame:
    """Add radiation-defined nighttime metrics to the unchanged daily metrics."""
    daily = LEGACY.daily_metrics(raw, stability, mld).copy()
    rsw = pd.to_numeric(raw[SHORTWAVE_COLUMN], errors="coerce")
    night_raw = shortwave_night_mask(rsw, nighttime_threshold)

    interval_hours = float(raw.index.to_series().diff().dropna().median().total_seconds() / 3600.0)
    daily["Nighttime_observations_n"] = night_raw.astype(int).resample("D").sum()
    daily["Nighttime_duration_h"] = daily["Nighttime_observations_n"] * interval_hours
    daily["Nighttime_Rsw_min_W_m2"] = rsw.loc[night_raw].resample("D").min()
    daily["Nighttime_Rsw_max_W_m2"] = rsw.loc[night_raw].resample("D").max()

    delta_t = pd.to_numeric(raw["25cm"], errors="coerce") - pd.to_numeric(raw["225cm"], errors="coerce")
    delta_night = delta_t.loc[night_raw]
    daily["DeltaT_night_mean_C"] = delta_night.resample("D").mean()
    daily["DeltaT_night_median_C"] = delta_night.resample("D").median()
    daily["DeltaT_night_min_C"] = delta_night.resample("D").min()
    daily["DeltaT_night_max_C"] = delta_night.resample("D").max()

    rsw_stability = align_shortwave(rsw, stability.index)
    night_stability = shortwave_night_mask(rsw_stability, nighttime_threshold)
    st = pd.to_numeric(stability["St(J/m2)"], errors="coerce")
    n2 = pd.to_numeric(stability["N2_max(1/s2)"], errors="coerce")
    daily["St_night_min_J_m2"] = st.loc[night_stability].resample("D").min()
    daily["St_night_median_J_m2"] = st.loc[night_stability].resample("D").median()
    daily["N2_night_min_s2"] = n2.loc[night_stability].resample("D").min()
    daily["N2_night_median_s2"] = n2.loc[night_stability].resample("D").median()

    rsw_mld = align_shortwave(rsw, mld.index)
    night_mld = shortwave_night_mask(rsw_mld, nighttime_threshold)
    for method in ["PCHIP", "linear"]:
        for threshold in [0.1, 0.2, 0.3]:
            tag = f"MLD_{method}_{threshold:.1f}C"
            values = pd.to_numeric(mld[tag], errors="coerce")
            selected = values.loc[night_mld]
            daily[f"{tag}_night_mean_m"] = selected.resample("D").mean()
            daily[f"{tag}_night_median_m"] = selected.resample("D").median()
            daily[f"{tag}_night_min_m"] = selected.resample("D").min()
            daily[f"{tag}_night_max_m"] = selected.resample("D").max()
            daily[f"{tag}_night_deepest_fraction"] = selected.eq(DEPTH_MAX_M).resample("D").mean()

    daily["DeltaT_night_7d_median_C"] = daily["DeltaT_night_median_C"].rolling(7, min_periods=4).median()
    daily["St_night_7d_median_J_m2"] = daily["St_night_median_J_m2"].rolling(7, min_periods=4).median()
    daily["N2_night_7d_median_s2"] = daily["N2_night_median_s2"].rolling(7, min_periods=4).median()
    daily["MLD_night_7d_median_m"] = daily["MLD_PCHIP_0.2C_night_median_m"].rolling(7, min_periods=4).median()

    # The median rule preserves the existing framework while changing only the
    # window definition.  The all-observations flag is a stricter diagnostic,
    # not an additional classification requirement.
    daily["Nocturnal_persistence_median_signature"] = (
        daily["DeltaT_night_median_C"].gt(BASELINE_THRESHOLD_C)
        & daily["MLD_PCHIP_0.2C_night_median_m"].lt(DEPTH_MAX_M)
    )
    daily["Nocturnal_all_observations_stratified_signature"] = (
        daily["DeltaT_night_min_C"].gt(BASELINE_THRESHOLD_C)
        & daily["MLD_PCHIP_0.2C_night_max_m"].lt(DEPTH_MAX_M)
    )
    return daily


def classify_with_nighttime(
    daily: pd.DataFrame,
    method: str = "PCHIP",
    threshold: float = BASELINE_THRESHOLD_C,
    state_min_run_days: int = 3,
    decay_compare_lag_days: int = 14,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the existing season-blind framework using radiation-defined night."""
    tag = f"MLD_{method}_{threshold:.1f}C"
    mean_col = f"{tag}_mean_m"
    amplitude_col = f"{tag}_amplitude_m"
    deepest_col = f"{tag}_deepest_fraction"
    night_col = f"{tag}_night_median_m"

    flags = pd.DataFrame(index=daily.index)
    flags["Mixed_daily_core"] = daily["Near_isothermal_fraction"].ge(0.50) & daily[deepest_col].ge(0.50)
    flags["Mixed_sustained"] = LEGACY.retain_runs(flags["Mixed_daily_core"], state_min_run_days)

    flags["Persistent_daily_core"] = (
        daily["DeltaT_night_median_C"].gt(BASELINE_THRESHOLD_C)
        & daily[night_col].lt(DEPTH_MAX_M)
    )
    flags["Persistent_sustained"] = LEGACY.retain_runs(flags["Persistent_daily_core"], state_min_run_days)

    flags["Diel_daily_core"] = (
        daily["DeltaT_max_C"].gt(BASELINE_THRESHOLD_C)
        & (daily["DeltaT_max_C"] - daily["DeltaT_night_median_C"]).ge(BASELINE_THRESHOLD_C)
        & daily["DeltaT_range_C"].ge(BASELINE_THRESHOLD_C)
        & daily[amplitude_col].ge(SENSOR_SPACING_M)
        & ~flags["Mixed_sustained"]
    )
    flags["Diel_sustained"] = LEGACY.retain_runs(flags["Diel_daily_core"], state_min_run_days)
    flags["Classification_MLD_amplitude_7d_median_m"] = daily[amplitude_col].rolling(7, min_periods=4).median()

    decay, decay_audit = LEGACY.qualifying_decay_episodes(
        daily,
        mixed=flags["Mixed_sustained"],
        mld_mean_col=mean_col,
        delta_threshold=BASELINE_THRESHOLD_C,
        compare_lag_days=decay_compare_lag_days,
    )
    flags["Decay_multimetric_episode"] = decay
    flags["Short_lived_mixed_interruption"] = (
        daily["DeltaT_night_median_C"].le(BASELINE_THRESHOLD_C)
        & daily[night_col].ge(DEPTH_MAX_M - 1e-9)
        & ~flags["Mixed_sustained"]
    )

    label = pd.Series("Transition/unclassified", index=daily.index, dtype="object")
    persistent = flags["Persistent_sustained"]
    diel = flags["Diel_sustained"]
    label.loc[persistent & ~diel] = "Persistent stratification"
    label.loc[diel & ~persistent] = "Diurnally oscillating stratification"
    overlap = persistent & diel
    label.loc[overlap & flags["Classification_MLD_amplitude_7d_median_m"].lt(SENSOR_SPACING_M)] = "Persistent stratification"
    label.loc[overlap & flags["Classification_MLD_amplitude_7d_median_m"].ge(SENSOR_SPACING_M)] = "Diurnally oscillating stratification"
    label.loc[flags["Decay_multimetric_episode"] & ~flags["Mixed_sustained"]] = "Decaying stratification"
    label.loc[flags["Mixed_sustained"]] = "Near-isothermal mixing"
    flags["Regime"] = label
    return flags, decay_audit


def label_comparison(old: pd.Series, new: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for regime in REGIME_ORDER:
        old_mask = old.eq(regime)
        new_mask = new.eq(regime)
        intersection = int((old_mask & new_mask).sum())
        union = int((old_mask | new_mask).sum())
        old_n = int(old_mask.sum())
        new_n = int(new_mask.sum())
        rows.append(
            {
                "Regime": regime,
                "Old_number_of_days": old_n,
                "New_number_of_days": new_n,
                "Difference_days": new_n - old_n,
                "Same_label_days": intersection,
                "Old_to_new_retention_percent": 100.0 * intersection / old_n if old_n else np.nan,
                "New_label_precision_percent": 100.0 * intersection / new_n if new_n else np.nan,
                "Percentage_agreement_Jaccard": 100.0 * intersection / union if union else np.nan,
            }
        )
    comparison = pd.DataFrame(rows)
    confusion = pd.crosstab(old.rename("Old_fixed_clock"), new.rename("New_shortwave_night"), margins=True)
    summary = pd.DataFrame(
        [
            {
                "Metric": "Overall daily label agreement",
                "Numerator_days": int(old.eq(new).sum()),
                "Denominator_days": int(len(old)),
                "Value_percent": 100.0 * float(old.eq(new).mean()),
            },
            {
                "Metric": "Changed daily labels",
                "Numerator_days": int(old.ne(new).sum()),
                "Denominator_days": int(len(old)),
                "Value_percent": 100.0 * float(old.ne(new).mean()),
            },
        ]
    )
    return comparison, confusion, summary


def posthoc_season(index: pd.DatetimeIndex) -> pd.Series:
    values = np.select(
        [index.month.isin([12, 1, 2]), index.month.isin([3, 4, 5]), index.month.isin([6, 7, 8])],
        ["Winter", "Spring", "Summer"],
        default="Autumn",
    )
    return pd.Series(values, index=index, name="Posthoc_season")


def dominant_regime(labels: pd.Series, season_mask: np.ndarray) -> str:
    counts = labels.loc[season_mask].value_counts()
    if counts.empty:
        return "No data"
    maximum = counts.max()
    winners = [x for x in REGIME_ORDER if counts.get(x, 0) == maximum]
    return " / ".join(winners)


def rule_sensitivity(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, object]] = []
    seasonal_rows: list[dict[str, object]] = []
    seasons = posthoc_season(daily.index)
    baseline_signature: tuple[str, str, str, str] | None = None

    for run_days in [2, 3, 5]:
        for trend_lag in [7, 14]:
            flags, _ = classify_with_nighttime(
                daily,
                state_min_run_days=run_days,
                decay_compare_lag_days=trend_lag,
            )
            labels = flags["Regime"]
            scenario = f"Persistence {run_days} d; 7-d metrics compared across {trend_lag} d"
            counts = labels.value_counts()
            winter = dominant_regime(labels, seasons.eq("Winter").to_numpy())
            spring = dominant_regime(labels, seasons.eq("Spring").to_numpy())
            summer = dominant_regime(labels, seasons.eq("Summer").to_numpy())

            autumn = labels.loc[seasons.eq("Autumn")]
            autumn_decay = int(autumn.eq("Decaying stratification").sum())
            autumn_diel = int(autumn.eq("Diurnally oscillating stratification").sum())
            september_mixed = int(labels.loc[labels.index.month == 9].eq("Near-isothermal mixing").sum())
            november_mixed = int(labels.loc[labels.index.month == 11].eq("Near-isothermal mixing").sum())
            autumn_transition_supported = autumn_decay > 0 and autumn_diel > 0 and november_mixed > september_mixed
            if autumn_transition_supported:
                autumn_interpretation = "Progressive transition/erosion with intermittent restratification"
            else:
                autumn_interpretation = "Operational evidence incomplete; inspect scenario details"

            signature = (winter, spring, summer, autumn_interpretation)
            if run_days == 3 and trend_lag == 14:
                baseline_signature = signature

            detail_rows.append(
                {
                    "Rule_scenario": scenario,
                    "Persistence_consecutive_days": run_days,
                    "Rolling_background_days": ROLLING_WINDOW_DAYS,
                    "Trend_comparison_interval_days": trend_lag,
                    "Near_isothermal_mixing_days": int(counts.get("Near-isothermal mixing", 0)),
                    "Persistent_stratification_days": int(counts.get("Persistent stratification", 0)),
                    "Diurnally_oscillating_days": int(counts.get("Diurnally oscillating stratification", 0)),
                    "Decaying_stratification_days": int(counts.get("Decaying stratification", 0)),
                    "Transition_unclassified_days": int(counts.get("Transition/unclassified", 0)),
                    "Winter_dominant_regime": winter,
                    "Spring_dominant_regime": spring,
                    "Summer_dominant_regime": summer,
                    "Autumn_interpretation": autumn_interpretation,
                    "Autumn_decay_days": autumn_decay,
                    "Autumn_diel_days": autumn_diel,
                    "September_mixed_days": september_mixed,
                    "November_mixed_days": november_mixed,
                    "Principal_interpretation_supported": (
                        winter == "Near-isothermal mixing"
                        and spring == "Persistent stratification"
                        and summer == "Diurnally oscillating stratification"
                        and autumn_transition_supported
                    ),
                }
            )
            table = pd.crosstab(labels, seasons).reindex(index=REGIME_ORDER, columns=["Winter", "Spring", "Summer", "Autumn"], fill_value=0)
            for regime in REGIME_ORDER:
                seasonal_rows.append(
                    {
                        "Rule_scenario": scenario,
                        "Regime": regime,
                        **{f"{season}_days": int(table.loc[regime, season]) for season in ["Winter", "Spring", "Summer", "Autumn"]},
                    }
                )

    detail = pd.DataFrame(detail_rows)
    if baseline_signature is None:
        raise RuntimeError("Baseline rule scenario was not generated.")
    detail["Whether_principal_seasonal_interpretation_changes"] = ~detail["Principal_interpretation_supported"]
    detail["Matches_baseline_interpretation_signature"] = detail.apply(
        lambda r: (
            r["Winter_dominant_regime"],
            r["Spring_dominant_regime"],
            r["Summer_dominant_regime"],
            r["Autumn_interpretation"],
        ) == baseline_signature,
        axis=1,
    )
    return detail, pd.DataFrame(seasonal_rows)


def threshold_distribution(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rsw = pd.to_numeric(raw[SHORTWAVE_COLUMN], errors="coerce")
    interval_h = float(raw.index.to_series().diff().dropna().median().total_seconds() / 3600.0)
    rows: list[dict[str, object]] = []
    for threshold in [0.0, 1.0, 5.0]:
        mask = shortwave_night_mask(rsw, threshold)
        counts = mask.astype(int).resample("D").sum()
        included = rsw.loc[mask]
        rows.append(
            {
                "Nighttime_rule": "Rsw = 0" if threshold == 0 else f"Rsw <= {threshold:g} W m-2",
                "Threshold_W_m2": threshold,
                "Included_observations": int(mask.sum()),
                "Fraction_of_all_observations_percent": 100.0 * float(mask.mean()),
                "Days_with_nighttime_observations": int((counts > 0).sum()),
                "Minimum_observations_per_day": int(counts.min()),
                "Median_observations_per_day": float(counts.median()),
                "Maximum_observations_per_day": int(counts.max()),
                "Minimum_duration_h_per_day": float(counts.min() * interval_h),
                "Median_duration_h_per_day": float(counts.median() * interval_h),
                "Maximum_duration_h_per_day": float(counts.max() * interval_h),
                "Exact_zero_observations_included": int(included.eq(0).sum()),
                "Positive_observations_included": int(included.gt(0).sum()),
            }
        )

    buckets = [
        ("Rsw < 0", rsw.lt(0)),
        ("Rsw = 0", rsw.eq(0)),
        ("0 < Rsw <= 1", rsw.gt(0) & rsw.le(1)),
        ("1 < Rsw <= 5", rsw.gt(1) & rsw.le(5)),
        ("5 < Rsw <= 10", rsw.gt(5) & rsw.le(10)),
        ("10 < Rsw <= 20", rsw.gt(10) & rsw.le(20)),
        ("Rsw > 20", rsw.gt(20)),
    ]
    bucket_rows = [
        {
            "Shortwave_bucket_W_m2": name,
            "Observations": int(mask.sum()),
            "Fraction_percent": 100.0 * float(mask.mean()),
            "Days_represented": int(raw.index[mask].normalize().nunique()),
        }
        for name, mask in buckets
    ]
    return pd.DataFrame(rows), pd.DataFrame(bucket_rows)


def threshold_classification_sensitivity(
    raw: pd.DataFrame,
    stability: pd.DataFrame,
    mld: pd.DataFrame,
    old_labels: pd.Series,
) -> tuple[pd.DataFrame, dict[float, tuple[pd.DataFrame, pd.DataFrame]]]:
    scenarios: dict[float, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for threshold in [0.0, 1.0, 5.0]:
        daily = daily_metrics_with_nighttime(raw, stability, mld, threshold)
        flags, _ = classify_with_nighttime(daily)
        scenarios[threshold] = (daily, flags)

    reference = scenarios[0.0][1]["Regime"]
    rows: list[dict[str, object]] = []
    for threshold, (_, flags) in scenarios.items():
        labels = flags["Regime"]
        counts = labels.value_counts()
        rows.append(
            {
                "Nighttime_rule": "Rsw = 0" if threshold == 0 else f"Rsw <= {threshold:g} W m-2",
                "Threshold_W_m2": threshold,
                "Near_isothermal_mixing_days": int(counts.get("Near-isothermal mixing", 0)),
                "Persistent_stratification_days": int(counts.get("Persistent stratification", 0)),
                "Diurnally_oscillating_days": int(counts.get("Diurnally oscillating stratification", 0)),
                "Decaying_stratification_days": int(counts.get("Decaying stratification", 0)),
                "Transition_unclassified_days": int(counts.get("Transition/unclassified", 0)),
                "Agreement_with_Rsw_zero_percent": 100.0 * float(labels.eq(reference).mean()),
                "Agreement_with_old_fixed_clock_percent": 100.0 * float(labels.eq(old_labels).mean()),
            }
        )
    return pd.DataFrame(rows), scenarios


def profile_audit(profile_path: Path, results: pd.DataFrame) -> pd.DataFrame:
    profiles = pd.read_csv(profile_path, encoding="utf-8-sig")
    profiles["target_time"] = pd.to_datetime(profiles["target_time"])
    rows: list[dict[str, object]] = []
    for (profile_label, target), group in profiles.groupby(["profile_label", "target_time"], sort=False):
        day = target.normalize()
        row = results.loc[day]
        rows.append(
            {
                "Profile_label_original": profile_label,
                "Target_time": target,
                "Profile_DeltaT_C": float(group["temperature_C"].iloc[0] - group["temperature_C"].iloc[-1]),
                "Old_fixed_clock_regime": row["Old_fixed_clock_regime"],
                "New_shortwave_night_regime": row["Regime"],
                "Daily_DeltaT_mean_C": row["DeltaT_mean_C"],
                "Nighttime_DeltaT_median_C": row["DeltaT_night_median_C"],
                "Daily_DeltaT_range_C": row["DeltaT_range_C"],
                "Daily_MLD_mean_m": row["MLD_PCHIP_0.2C_mean_m"],
                "Nighttime_MLD_median_m": row["MLD_PCHIP_0.2C_night_median_m"],
                "Daily_MLD_amplitude_m": row["MLD_PCHIP_0.2C_amplitude_m"],
                "Daily_St_mean_J_m2": row["St_mean_J_m2"],
                "Daily_N2_median_s2": row["N2_median_s2"],
                "Recommended_role": "Illustration only; not a classification input",
            }
        )
    audit = pd.DataFrame(rows)
    audit["Recommended_profile_name"] = audit["Profile_label_original"]
    autumn = audit["Profile_label_original"].eq("Autumn decay")
    audit.loc[autumn, "Recommended_profile_name"] = "Autumn transition with daytime restratification"
    return audit


def event_audit(results: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "DeltaT_mean_C",
        "DeltaT_night_min_C",
        "DeltaT_night_median_C",
        "Near_isothermal_fraction",
        "St_mean_J_m2",
        "St_night_min_J_m2",
        "St_night_median_J_m2",
        "N2_median_s2",
        "N2_night_median_s2",
        "MLD_PCHIP_0.2C_mean_m",
        "MLD_PCHIP_0.2C_night_median_m",
        "MLD_PCHIP_0.2C_deepest_fraction",
        "Mixed_daily_core",
        "Mixed_sustained",
        "Short_lived_mixed_interruption",
        "Regime",
    ]
    return results.loc["2024-05-01":"2024-05-10", columns].reset_index()


def main() -> None:
    raw, stability, mld = read_sources()
    old_daily = LEGACY.daily_metrics(raw, stability, mld)
    old_flags, _ = LEGACY.classify(old_daily, "PCHIP", BASELINE_THRESHOLD_C)
    old_labels = old_flags["Regime"]

    threshold_summary, low_value_buckets = threshold_distribution(raw)
    threshold_sensitivity, scenarios = threshold_classification_sensitivity(raw, stability, mld, old_labels)
    daily, flags = scenarios[NIGHTTIME_THRESHOLD_W_M2]
    results = daily.join(flags)
    results.insert(0, "Old_fixed_clock_regime", old_labels)
    results.insert(0, "Year_day", np.arange(1, len(results) + 1))
    results["Old_new_label_agreement"] = results["Old_fixed_clock_regime"].eq(results["Regime"])

    comparison, confusion, agreement_summary = label_comparison(old_labels, results["Regime"])
    rule_detail, rule_seasonal = rule_sensitivity(daily)
    periods = LEGACY.label_periods(results["Regime"])
    profiles = profile_audit(PROFILE, results)
    may_event = event_audit(results)

    autumn_decay = results.loc[
        results.index.month.isin([9, 10, 11]) & results["Regime"].eq("Decaying stratification"),
        [
            "DeltaT_mean_C",
            "DeltaT_night_median_C",
            "St_mean_J_m2",
            "N2_median_s2",
            "MLD_PCHIP_0.2C_mean_m",
            "MLD_PCHIP_0.2C_amplitude_m",
            "Regime",
        ],
    ].copy()
    autumn_decay.insert(0, "Candidate_status", "Strict daily decaying label; alternative only")

    results.reset_index().to_csv(OUTPUT_DIR / "revised_regime_classification_results.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    periods.to_csv(OUTPUT_DIR / "revised_regime_classification_periods.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    threshold_summary.to_csv(OUTPUT_DIR / "nighttime_threshold_distribution.csv", index=False, encoding="utf-8-sig")
    low_value_buckets.to_csv(OUTPUT_DIR / "nighttime_shortwave_low_value_buckets.csv", index=False, encoding="utf-8-sig")
    threshold_sensitivity.to_csv(OUTPUT_DIR / "nighttime_threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUTPUT_DIR / "nighttime_old_new_regime_comparison.csv", index=False, encoding="utf-8-sig")
    confusion.to_csv(OUTPUT_DIR / "nighttime_old_new_confusion_matrix.csv", encoding="utf-8-sig")
    agreement_summary.to_csv(OUTPUT_DIR / "nighttime_old_new_daily_agreement_summary.csv", index=False, encoding="utf-8-sig")
    rule_detail.to_csv(OUTPUT_DIR / "regime_rule_sensitivity_details.csv", index=False, encoding="utf-8-sig")
    rule_seasonal.to_csv(OUTPUT_DIR / "regime_rule_sensitivity_seasonal_counts.csv", index=False, encoding="utf-8-sig")
    profiles.to_csv(OUTPUT_DIR / "representative_profile_validation_revised.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M")
    may_event.to_csv(OUTPUT_DIR / "may3_6_event_audit.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    autumn_decay.reset_index().to_csv(OUTPUT_DIR / "autumn_decay_candidate_audit.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")

    rsw = pd.to_numeric(raw[SHORTWAVE_COLUMN], errors="coerce")
    may5 = results.loc[pd.Timestamp("2024-05-05")]
    oct25 = results.loc[pd.Timestamp("2024-10-25")]
    checks = {
        "source_files": {
            "raw_temperature_and_shortwave": str(RAW),
            "stability": str(STABILITY),
            "mld": str(MLD),
            "profiles": str(PROFILE),
            "legacy_method": str(LEGACY_PATH),
        },
        "nighttime_definition": {
            "selected_rule": "measured Rsw = 0 W m-2",
            "selected_threshold_W_m2": NIGHTTIME_THRESHOLD_W_M2,
            "negative_shortwave_observations": int(rsw.lt(0).sum()),
            "exact_zero_observations": int(rsw.eq(0).sum()),
            "positive_observations_0_to_1_W_m2": int((rsw.gt(0) & rsw.le(1)).sum()),
            "days_with_exact_zero": int(raw.index[rsw.eq(0)].normalize().nunique()),
            "zero_and_le1_masks_identical": bool(shortwave_night_mask(rsw, 0).equals(shortwave_night_mask(rsw, 1))),
        },
        "old_new": {
            "overall_daily_agreement_percent": float(agreement_summary.loc[0, "Value_percent"]),
            "changed_days": int(agreement_summary.loc[1, "Numerator_days"]),
            "old_counts": old_labels.value_counts().reindex(REGIME_ORDER, fill_value=0).to_dict(),
            "new_counts": results["Regime"].value_counts().reindex(REGIME_ORDER, fill_value=0).to_dict(),
        },
        "rule_sensitivity": {
            "scenario_count": int(len(rule_detail)),
            "all_principal_interpretations_supported": bool(rule_detail["Principal_interpretation_supported"].all()),
            "scenarios_changing_principal_interpretation": int(rule_detail["Whether_principal_seasonal_interpretation_changes"].sum()),
        },
        "events": {
            "may5_night_deltaT_median_C": float(may5["DeltaT_night_median_C"]),
            "may5_night_mld_median_m": float(may5["MLD_PCHIP_0.2C_night_median_m"]),
            "may5_short_lived_mixed_interruption": bool(may5["Short_lived_mixed_interruption"]),
            "may5_sustained_mixed": bool(may5["Mixed_sustained"]),
            "may6_to_may10_regimes": {
                str(day.date()): str(regime)
                for day, regime in results.loc["2024-05-06":"2024-05-10", "Regime"].items()
            },
            "oct25_new_regime": str(oct25["Regime"]),
            "oct25_old_regime": str(oct25["Old_fixed_clock_regime"]),
            "oct25_recommended_description": "Autumn transition with daytime restratification",
        },
        "quality": {
            "daily_rows": int(len(results)),
            "labels_total": int(results["Regime"].notna().sum()),
            "missing_core_metrics": int(results[["DeltaT_mean_C", "DeltaT_night_median_C", "St_mean_J_m2", "N2_median_s2", "MLD_PCHIP_0.2C_mean_m", "MLD_PCHIP_0.2C_night_median_m"]].isna().sum().sum()),
            "profiles_count": int(len(profiles)),
        },
    }
    (OUTPUT_DIR / "revised_regime_classification_quality_checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    assert len(results) == 366
    assert results["Regime"].notna().all()
    assert checks["quality"]["missing_core_metrics"] == 0
    assert checks["nighttime_definition"]["negative_shortwave_observations"] == 0
    assert checks["nighttime_definition"]["days_with_exact_zero"] == 366
    assert checks["nighttime_definition"]["zero_and_le1_masks_identical"]
    assert len(rule_detail) == 6
    print(json.dumps(checks, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
