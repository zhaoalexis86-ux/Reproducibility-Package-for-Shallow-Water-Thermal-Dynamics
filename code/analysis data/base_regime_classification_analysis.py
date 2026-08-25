"""Stage 4A: reproducible, season-blind thermal-regime evidence.

The classification rules use continuous physical metrics only. Calendar months
are introduced only after classification to compare the detected periods with
the manuscript's seasonal interpretation.

This script does not recalculate or alter the original St/N2 algorithms and it
does not use Rib as a classification criterion.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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

DEPTH_MAX_M = 2.25
SENSOR_SPACING_M = 0.50
BASELINE_THRESHOLD_C = 0.20
PREDAWN_START_HOUR = 4.0
PREDAWN_END_HOUR = 6.0
STATE_MIN_RUN_DAYS = 3
DECAY_MIN_RUN_DAYS = 7
ROLLING_WINDOW_DAYS = 7
DECAY_COMPARE_LAG_DAYS = 14
DECAY_LINK_TO_MIXED_DAYS = 7

REGIME_ORDER = [
    "Near-isothermal mixing",
    "Persistent stratification",
    "Diurnally oscillating stratification",
    "Decaying stratification",
    "Transition/unclassified",
]
REGIME_COLORS = {
    "Near-isothermal mixing": "#4C78A8",
    "Persistent stratification": "#ECA82C",
    "Diurnally oscillating stratification": "#E45756",
    "Decaying stratification": "#72B7B2",
    "Transition/unclassified": "#B8B8B8",
}


def predawn_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """Fixed local-time early-morning window; no seasonal information used."""
    hour = index.hour + index.minute / 60.0
    return (hour >= PREDAWN_START_HOUR) & (hour < PREDAWN_END_HOUR)


def retain_runs(mask: pd.Series, minimum: int) -> pd.Series:
    """Retain True runs whose duration is at least ``minimum`` days."""
    x = mask.fillna(False).astype(bool)
    group = x.ne(x.shift(fill_value=False)).cumsum()
    length = x.groupby(group).transform("sum")
    return x & length.ge(minimum)


def contiguous_periods(mask: pd.Series, label: str | None = None) -> pd.DataFrame:
    """Return inclusive date ranges for True runs."""
    x = mask.fillna(False).astype(bool)
    group = x.ne(x.shift(fill_value=False)).cumsum()
    records: list[dict[str, object]] = []
    for _, part in x.groupby(group):
        if not bool(part.iloc[0]):
            continue
        records.append(
            {
                "Regime": label or "True",
                "Start": part.index.min(),
                "End": part.index.max(),
                "Duration_days": int(len(part)),
            }
        )
    return pd.DataFrame(records)


def read_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(RAW, parse_dates=["Date"], encoding="utf-8-sig").set_index("Date").sort_index()
    stability = (
        pd.read_csv(STABILITY, parse_dates=["Date"], encoding="utf-8-sig")
        .set_index("Date")
        .sort_index()
    )
    mld = pd.read_csv(MLD, parse_dates=["Date"], encoding="utf-8-sig").set_index("Date").sort_index()
    return raw, stability, mld


def daily_metrics(raw: pd.DataFrame, stability: pd.DataFrame, mld: pd.DataFrame) -> pd.DataFrame:
    """Calculate the daily metrics used for thermal-regime classification."""
    full_days = pd.date_range("2024-01-01", "2024-12-31", freq="D", name="Date")
    daily = pd.DataFrame(index=full_days)

    delta_t = pd.to_numeric(raw["25cm"], errors="coerce") - pd.to_numeric(raw["225cm"], errors="coerce")
    delta_t_abs = delta_t.abs()
    predawn_delta_t = delta_t.loc[predawn_mask(delta_t.index)]
    daily["DeltaT_mean_C"] = delta_t.resample("D").mean()
    daily["DeltaT_max_C"] = delta_t.resample("D").max()
    daily["DeltaT_min_C"] = delta_t.resample("D").min()
    daily["DeltaT_predawn_mean_C"] = predawn_delta_t.resample("D").mean()
    daily["DeltaT_predawn_median_C"] = predawn_delta_t.resample("D").median()
    daily["DeltaT_range_C"] = delta_t.resample("D").apply(lambda x: x.max() - x.min())
    daily["Near_isothermal_fraction"] = delta_t_abs.le(BASELINE_THRESHOLD_C).resample("D").mean()

    st = pd.to_numeric(stability["St(J/m2)"], errors="coerce")
    n2 = pd.to_numeric(stability["N2_max(1/s2)"], errors="coerce")
    predawn_st = st.loc[predawn_mask(st.index)]
    predawn_n2 = n2.loc[predawn_mask(n2.index)]
    daily["St_mean_J_m2"] = st.resample("D").mean()
    daily["St_max_J_m2"] = st.resample("D").max()
    daily["St_predawn_min_J_m2"] = predawn_st.resample("D").min()
    daily["St_predawn_median_J_m2"] = predawn_st.resample("D").median()
    daily["St_range_J_m2"] = st.resample("D").apply(lambda x: x.max() - x.min())

    # Existing hourly N2_max is the strongest local density-gradient segment in
    # each profile. Its daily median represents typical local stability while
    # reducing sensitivity to one isolated hourly maximum.
    daily["N2_median_s2"] = n2.resample("D").median()
    daily["N2_max_s2"] = n2.resample("D").max()
    daily["N2_predawn_median_s2"] = predawn_n2.resample("D").median()
    daily["N2_range_s2"] = n2.resample("D").apply(lambda x: x.max() - x.min())

    predawn_mld = predawn_mask(mld.index)
    for method in ["PCHIP", "linear"]:
        for threshold in [0.1, 0.2, 0.3]:
            source = f"MLD_{method}_{threshold:.1f}C"
            values = pd.to_numeric(mld[source], errors="coerce")
            early = values.loc[predawn_mld]
            tag = source
            daily[f"{tag}_mean_m"] = values.resample("D").mean()
            daily[f"{tag}_min_m"] = values.resample("D").min()
            daily[f"{tag}_max_m"] = values.resample("D").max()
            daily[f"{tag}_amplitude_m"] = values.resample("D").apply(lambda x: x.max() - x.min())
            daily[f"{tag}_deepest_fraction"] = values.eq(DEPTH_MAX_M).resample("D").mean()
            daily[f"{tag}_predawn_median_m"] = early.resample("D").median()

    # Trailing 7-day statistics are operational (no future values) and are used
    # to distinguish a multi-day background state from isolated weather events.
    for source, target, operation in [
        ("DeltaT_mean_C", "DeltaT_mean_7d_median_C", "median"),
        ("DeltaT_predawn_median_C", "DeltaT_predawn_7d_median_C", "median"),
        ("St_mean_J_m2", "St_mean_7d_median_J_m2", "median"),
        ("N2_median_s2", "N2_median_7d_median_s2", "median"),
        ("MLD_PCHIP_0.2C_mean_m", "MLD_mean_7d_median_m", "median"),
        ("MLD_PCHIP_0.2C_amplitude_m", "MLD_amplitude_7d_median_m", "median"),
        ("Near_isothermal_fraction", "Near_isothermal_fraction_7d_mean", "mean"),
    ]:
        roll = daily[source].rolling(ROLLING_WINDOW_DAYS, min_periods=4)
        daily[target] = roll.median() if operation == "median" else roll.mean()
    return daily


def qualifying_decay_episodes(
    daily: pd.DataFrame,
    mixed: pd.Series,
    mld_mean_col: str,
    delta_threshold: float,
    compare_lag_days: int = DECAY_COMPARE_LAG_DAYS,
) -> tuple[pd.Series, pd.DataFrame]:
    """Identify multimetric erosion episodes without an absolute St/N2 cutoff.

    A candidate must show an increased 7-day near-isothermal occurrence and at
    least three of four directions: lower DeltaT, lower St, lower N2, deeper MLD.
    It must persist for >=7 days, begin from a stratified state, and connect to a
    sustained mixed episode during the episode or within the next seven days.
    """
    dt7 = daily["DeltaT_mean_C"].rolling(7, min_periods=4).median()
    st7 = daily["St_mean_J_m2"].rolling(7, min_periods=4).median()
    n27 = daily["N2_median_s2"].rolling(7, min_periods=4).median()
    mld7 = daily[mld_mean_col].rolling(7, min_periods=4).median()
    mixed7 = daily["Near_isothermal_fraction"].rolling(7, min_periods=4).mean()

    votes = pd.DataFrame(index=daily.index)
    votes["DeltaT_decreasing"] = dt7.lt(dt7.shift(compare_lag_days))
    votes["St_decreasing"] = st7.lt(st7.shift(compare_lag_days))
    votes["N2_decreasing"] = n27.lt(n27.shift(compare_lag_days))
    votes["MLD_deepening"] = mld7.gt(mld7.shift(compare_lag_days))
    candidate = mixed7.gt(mixed7.shift(compare_lag_days)) & votes.sum(axis=1).ge(3)
    candidate = retain_runs(candidate, DECAY_MIN_RUN_DAYS)

    accepted = pd.Series(False, index=daily.index)
    episode_records: list[dict[str, object]] = []
    for _, episode in candidate.groupby(candidate.ne(candidate.shift(fill_value=False)).cumsum()):
        if not bool(episode.iloc[0]):
            continue
        start = episode.index.min()
        end = episode.index.max()
        begins_stratified = bool(dt7.loc[start] > delta_threshold and mld7.loc[start] < DEPTH_MAX_M)
        link_end = min(daily.index.max(), end + pd.Timedelta(days=DECAY_LINK_TO_MIXED_DAYS))
        linked_to_mixed = bool(mixed.loc[start:link_end].any())
        keep = begins_stratified and linked_to_mixed
        if keep:
            accepted.loc[start:end] = True
        episode_records.append(
            {
                "Start": start,
                "End": end,
                "Duration_days": int((end - start).days + 1),
                "Begins_from_stratified_7d_state": begins_stratified,
                "Links_to_sustained_mixed_state": linked_to_mixed,
                "Accepted_as_decay_episode": keep,
                "Trend_comparison_lag_days": compare_lag_days,
            }
        )
    return accepted, pd.DataFrame(episode_records)


def classify(
    daily: pd.DataFrame,
    method: str,
    threshold: float,
    state_min_run_days: int = STATE_MIN_RUN_DAYS,
    decay_compare_lag_days: int = DECAY_COMPARE_LAG_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply season-blind physical criteria to one MLD scenario.

    ``threshold`` is the MLD-definition threshold only.  The independent
    surface-to-bottom mixing criterion remains 0.2 degC in all six scenarios,
    so the MLD sensitivity test changes only the intended calculation choice.
    """
    tag = f"MLD_{method}_{threshold:.1f}C"
    mean_col = f"{tag}_mean_m"
    amplitude_col = f"{tag}_amplitude_m"
    deepest_col = f"{tag}_deepest_fraction"
    predawn_col = f"{tag}_predawn_median_m"

    flags = pd.DataFrame(index=daily.index)
    flags["Mixed_daily_core"] = (
        daily["Near_isothermal_fraction"].ge(0.50)
        & daily[deepest_col].ge(0.50)
    )
    flags["Mixed_sustained"] = retain_runs(flags["Mixed_daily_core"], state_min_run_days)

    flags["Persistent_daily_core"] = (
        daily["DeltaT_predawn_median_C"].gt(BASELINE_THRESHOLD_C)
        & daily[predawn_col].lt(DEPTH_MAX_M)
    )
    flags["Persistent_sustained"] = retain_runs(flags["Persistent_daily_core"], state_min_run_days)

    flags["Diel_daily_core"] = (
        daily["DeltaT_max_C"].gt(BASELINE_THRESHOLD_C)
        & (daily["DeltaT_max_C"] - daily["DeltaT_predawn_median_C"]).ge(BASELINE_THRESHOLD_C)
        & daily["DeltaT_range_C"].ge(BASELINE_THRESHOLD_C)
        & daily[amplitude_col].ge(SENSOR_SPACING_M)
        & ~flags["Mixed_sustained"]
    )
    flags["Diel_sustained"] = retain_runs(flags["Diel_daily_core"], state_min_run_days)
    flags["Classification_MLD_amplitude_7d_median_m"] = daily[amplitude_col].rolling(7, min_periods=4).median()

    decay, decay_audit = qualifying_decay_episodes(
        daily,
        mixed=flags["Mixed_sustained"],
        mld_mean_col=mean_col,
        delta_threshold=BASELINE_THRESHOLD_C,
        compare_lag_days=decay_compare_lag_days,
    )
    flags["Decay_multimetric_episode"] = decay

    # A one-night mixing event is retained as an interruption rather than
    # promoted to a regime unless the three-day persistence rule is met.
    flags["Short_lived_mixed_interruption"] = (
        daily["DeltaT_predawn_median_C"].le(BASELINE_THRESHOLD_C)
        & daily[predawn_col].ge(DEPTH_MAX_M - 1e-9)
        & ~flags["Mixed_sustained"]
    )

    label = pd.Series("Transition/unclassified", index=daily.index, dtype="object")
    p = flags["Persistent_sustained"]
    diel = flags["Diel_sustained"]
    label.loc[p & ~diel] = "Persistent stratification"
    label.loc[diel & ~p] = "Diurnally oscillating stratification"
    # If both signatures occur, the trailing 7-day median MLD amplitude
    # determines the predominant pattern. The 0.5-m divide is one sensor
    # interval, i.e. the smallest directly resolved vertical displacement.
    overlap = p & diel
    label.loc[overlap & flags["Classification_MLD_amplitude_7d_median_m"].lt(SENSOR_SPACING_M)] = (
        "Persistent stratification"
    )
    label.loc[overlap & flags["Classification_MLD_amplitude_7d_median_m"].ge(SENSOR_SPACING_M)] = (
        "Diurnally oscillating stratification"
    )
    # Decay is a trajectory, not a single-state threshold. Sustained mixed days
    # have priority because they represent the endpoint already reached.
    label.loc[flags["Decay_multimetric_episode"] & ~flags["Mixed_sustained"]] = "Decaying stratification"
    label.loc[flags["Mixed_sustained"]] = "Near-isothermal mixing"
    flags["Regime"] = label
    return flags, decay_audit


def label_periods(labels: pd.Series) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for regime in REGIME_ORDER:
        periods = contiguous_periods(labels.eq(regime), regime)
        if not periods.empty:
            records.append(periods)
    return pd.concat(records, ignore_index=True).sort_values(["Start", "End"]) if records else pd.DataFrame()


def posthoc_season(month: int) -> str:
    if month in [12, 1, 2]:
        return "Winter (post-hoc)"
    if month in [3, 4, 5]:
        return "Spring (post-hoc)"
    if month in [6, 7, 8]:
        return "Summer (post-hoc)"
    return "Autumn (post-hoc)"


def mld_robustness(daily: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    records: list[dict[str, object]] = []
    labels_by_scenario: dict[str, pd.DataFrame] = {}
    for method in ["PCHIP", "linear"]:
        for threshold in [0.1, 0.2, 0.3]:
            flags, _ = classify(daily, method, threshold)
            scenario = f"{method}_{threshold:.1f}C"
            labels_by_scenario[scenario] = flags
            labels = flags["Regime"]
            counts = labels.value_counts()
            seasonal = pd.crosstab(
                labels,
                pd.Index([posthoc_season(x.month) for x in labels.index], name="Posthoc_season"),
            )
            periods = label_periods(labels)

            def max_period_days(regime: str) -> int:
                x = periods.loc[periods["Regime"].eq(regime), "Duration_days"]
                return int(x.max()) if len(x) else 0

            autumn_decay_days = int(
                ((labels == "Decaying stratification") & labels.index.month.isin([9, 10, 11])).sum()
            )
            # Calendar seasons are used here only to compare the already-derived
            # regime sequence with the manuscript, never to build the labels.
            winter_mixed = int(seasonal.get("Winter (post-hoc)", pd.Series()).get("Near-isothermal mixing", 0))
            spring_persistent = int(seasonal.get("Spring (post-hoc)", pd.Series()).get("Persistent stratification", 0))
            summer_diel = int(seasonal.get("Summer (post-hoc)", pd.Series()).get("Diurnally oscillating stratification", 0))
            interpretation_preserved = (
                max_period_days("Near-isothermal mixing") >= 7
                and max_period_days("Persistent stratification") >= 7
                and max_period_days("Diurnally oscillating stratification") >= 7
                and autumn_decay_days > 0
            )
            row: dict[str, object] = {
                "Scenario": scenario,
                "Method": method,
                "MLD_threshold_C": threshold,
                "Mixed_days": int(counts.get("Near-isothermal mixing", 0)),
                "Persistent_days": int(counts.get("Persistent stratification", 0)),
                "Diel_days": int(counts.get("Diurnally oscillating stratification", 0)),
                "Decay_days": int(counts.get("Decaying stratification", 0)),
                "Transition_unclassified_days": int(counts.get("Transition/unclassified", 0)),
                "Winter_mixed_days_posthoc": winter_mixed,
                "Spring_persistent_days_posthoc": spring_persistent,
                "Summer_diel_days_posthoc": summer_diel,
                "Autumn_decay_days_posthoc": autumn_decay_days,
                "Longest_mixed_run_days": max_period_days("Near-isothermal mixing"),
                "Longest_persistent_run_days": max_period_days("Persistent stratification"),
                "Longest_diel_run_days": max_period_days("Diurnally oscillating stratification"),
                "Regime_interpretation_preserved": interpretation_preserved,
            }
            records.append(row)
    return pd.DataFrame(records), labels_by_scenario


def representative_profile_audit(
    daily: pd.DataFrame,
    baseline_flags: pd.DataFrame,
    stability: pd.DataFrame,
    mld: pd.DataFrame,
) -> pd.DataFrame:
    profiles = pd.read_csv(PROFILE, encoding="utf-8-sig")
    profiles["target_time"] = pd.to_datetime(profiles["target_time"])
    records: list[dict[str, object]] = []
    for (profile_label, target), group in profiles.groupby(["profile_label", "target_time"], sort=False):
        day = target.normalize()
        nearest_st_pos = stability.index.get_indexer([target], method="nearest")[0]
        nearest_mld_pos = mld.index.get_indexer([target], method="nearest")[0]
        stab_time = stability.index[nearest_st_pos]
        mld_time = mld.index[nearest_mld_pos]
        row = daily.loc[day]
        flags = baseline_flags.loc[day]
        records.append(
            {
                "Profile_label": profile_label,
                "Target_time": target,
                "Profile_DeltaT_C": float(group["temperature_C"].iloc[0] - group["temperature_C"].iloc[-1]),
                "Stored_profile_DeltaT_C": float(group["delta_T"].iloc[0]),
                "Daily_regime": flags["Regime"],
                "Daily_DeltaT_mean_C": row["DeltaT_mean_C"],
                "Daily_DeltaT_predawn_median_C": row["DeltaT_predawn_median_C"],
                "Daily_DeltaT_range_C": row["DeltaT_range_C"],
                "Daily_MLD_mean_m": row["MLD_PCHIP_0.2C_mean_m"],
                "Daily_MLD_amplitude_m": row["MLD_PCHIP_0.2C_amplitude_m"],
                "Daily_St_mean_J_m2": row["St_mean_J_m2"],
                "Daily_N2_median_s2": row["N2_median_s2"],
                "Nearest_St_time": stab_time,
                "Nearest_St_J_m2": float(stability.iloc[nearest_st_pos]["St(J/m2)"]),
                "Nearest_N2_s2": float(stability.iloc[nearest_st_pos]["N2_max(1/s2)"]),
                "Nearest_MLD_time": mld_time,
                "Nearest_MLD_m": float(mld.iloc[nearest_mld_pos]["MLD_PCHIP_0.2C"]),
                "Mixed_flag": bool(flags["Mixed_sustained"]),
                "Persistent_flag": bool(flags["Persistent_sustained"]),
                "Diel_flag": bool(flags["Diel_sustained"]),
                "Decay_flag": bool(flags["Decay_multimetric_episode"]),
            }
        )
    result = pd.DataFrame(records)
    expected = {
        "Winter mixed": "Near-isothermal mixing",
        "Spring persistent": "Persistent stratification",
        "Summer diel": "Diurnally oscillating stratification",
        "Autumn decay": "Decaying stratification",
    }
    result["Expected_regime"] = result["Profile_label"].map(expected)
    result["Daily_label_matches_profile"] = result["Daily_regime"].eq(result["Expected_regime"])
    return result


def run_length_sensitivity(daily: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for run_days in [2, 3, 5]:
        flags, _ = classify(daily, "PCHIP", 0.2, state_min_run_days=run_days)
        counts = flags["Regime"].value_counts()
        records.append(
            {
                "Sensitivity_type": "State persistence run length",
                "Case": f"{run_days} consecutive days",
                "Mixed_days": int(counts.get("Near-isothermal mixing", 0)),
                "Persistent_days": int(counts.get("Persistent stratification", 0)),
                "Diel_days": int(counts.get("Diurnally oscillating stratification", 0)),
                "Decay_days": int(counts.get("Decaying stratification", 0)),
                "Transition_unclassified_days": int(counts.get("Transition/unclassified", 0)),
            }
        )
    for lag in [7, 14]:
        flags, _ = classify(daily, "PCHIP", 0.2, decay_compare_lag_days=lag)
        counts = flags["Regime"].value_counts()
        records.append(
            {
                "Sensitivity_type": "Decay trend comparison lag",
                "Case": f"{lag}-day lag between 7-day metrics",
                "Mixed_days": int(counts.get("Near-isothermal mixing", 0)),
                "Persistent_days": int(counts.get("Persistent stratification", 0)),
                "Diel_days": int(counts.get("Diurnally oscillating stratification", 0)),
                "Decay_days": int(counts.get("Decaying stratification", 0)),
                "Transition_unclassified_days": int(counts.get("Transition/unclassified", 0)),
            }
        )
    return pd.DataFrame(records)


def regime_metric_summary(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "DeltaT_mean_C",
        "DeltaT_predawn_median_C",
        "DeltaT_range_C",
        "St_mean_J_m2",
        "N2_median_s2",
        "MLD_PCHIP_0.2C_mean_m",
        "MLD_PCHIP_0.2C_amplitude_m",
        "MLD_PCHIP_0.2C_deepest_fraction",
    ]
    rows: list[dict[str, object]] = []
    for regime in REGIME_ORDER:
        subset = results.loc[results["Regime"].eq(regime)]
        for metric in metrics:
            rows.append(
                {
                    "Regime": regime,
                    "Metric": metric,
                    "N_days": int(subset[metric].notna().sum()),
                    "Median": float(subset[metric].median()),
                    "P25": float(subset[metric].quantile(0.25)),
                    "P75": float(subset[metric].quantile(0.75)),
                }
            )
    return pd.DataFrame(rows)


def make_figure(results: pd.DataFrame, profile_audit: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9.5,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    dates = results.index
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(7.4, 8.1),
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.15, 1.15, 0.42], "hspace": 0.12},
    )
    ax1, ax2, ax3, ax4 = axes

    ax1.fill_between(
        dates,
        results["DeltaT_min_C"],
        results["DeltaT_max_C"],
        color="#D9D9D9",
        alpha=0.6,
        linewidth=0,
        label="Daily min-max",
    )
    ax1.plot(dates, results["DeltaT_mean_7d_median_C"], color="#C23B33", lw=1.5, label="7-d median daily mean")
    ax1.plot(
        dates,
        results["DeltaT_predawn_7d_median_C"],
        color="#284B63",
        lw=1.3,
        ls="--",
        label="7-d median predawn",
    )
    ax1.axhline(BASELINE_THRESHOLD_C, color="black", lw=0.8, ls=":", label=r"$\Delta T$ criterion (0.2 $^\circ$C)")
    ax1.set_ylabel(r"$\Delta T$ ($^\circ$C)")
    ax1.set_ylim(-0.6, max(10.0, float(results["DeltaT_max_C"].max()) + 0.4))
    ax1.legend(loc="upper left", ncol=2, frameon=False, fontsize=8.2)
    ax1.text(0.006, 0.91, "(a)", transform=ax1.transAxes, fontweight="bold")

    ax2.plot(dates, results["St_mean_7d_median_J_m2"], color="#D07C00", lw=1.5)
    ax2.set_ylabel(r"$S_t$ (J m$^{-2}$)", color="#A05B00")
    ax2.tick_params(axis="y", labelcolor="#A05B00")
    ax2.set_ylim(bottom=-0.15)
    ax2r = ax2.twinx()
    ax2r.plot(dates, results["N2_median_7d_median_s2"] * 1000, color="#4C78A8", lw=1.25, ls="--")
    ax2r.set_ylabel(r"Daily median $N^2$ ($10^{-3}$ s$^{-2}$)", color="#315A84")
    ax2r.tick_params(axis="y", labelcolor="#315A84")
    ax2r.set_ylim(bottom=-0.1)
    ax2.text(0.006, 0.91, "(b)", transform=ax2.transAxes, fontweight="bold")

    ax3.plot(dates, results["MLD_mean_7d_median_m"], color="#2A9D8F", lw=1.5, label="7-d median daily-mean MLD")
    ax3.axhline(DEPTH_MAX_M, color="black", lw=0.8, ls=":", label="Deepest monitored layer")
    ax3.set_ylabel("MLD (m)")
    ax3.set_ylim(DEPTH_MAX_M + 0.1, 0.18)
    ax3r = ax3.twinx()
    ax3r.plot(dates, results["MLD_amplitude_7d_median_m"], color="#9C4DCC", lw=1.15, ls="--")
    ax3r.axhline(SENSOR_SPACING_M, color="#9C4DCC", lw=0.75, ls=":")
    ax3r.set_ylabel("7-d median MLD amplitude (m)", color="#7A35A4")
    ax3r.tick_params(axis="y", labelcolor="#7A35A4")
    ax3r.set_ylim(0, 2.1)
    ax3.legend(loc="upper left", frameon=False, fontsize=8.2)
    ax3.text(0.006, 0.91, "(c)", transform=ax3.transAxes, fontweight="bold")

    code = {regime: i for i, regime in enumerate(REGIME_ORDER)}
    z = np.array([code[x] for x in results["Regime"]], dtype=float)[None, :]
    cmap = ListedColormap([REGIME_COLORS[x] for x in REGIME_ORDER])
    left = mdates.date2num(dates.min()) - 0.5
    right = mdates.date2num(dates.max()) + 0.5
    ax4.imshow(z, aspect="auto", interpolation="nearest", cmap=cmap, extent=[left, right, 0, 1], vmin=-0.5, vmax=4.5)
    ax4.set_yticks([])
    ax4.set_ylabel("Regime", labelpad=15)
    ax4.text(0.006, 0.76, "(d)", transform=ax4.transAxes, fontweight="bold", color="white")

    profile_codes = {"Winter mixed": "W", "Spring persistent": "Sp", "Summer diel": "Su", "Autumn decay": "A"}
    for _, row in profile_audit.iterrows():
        when = pd.Timestamp(row["Target_time"])
        ax4.plot(when, 0.5, marker="v", ms=5.5, color="black", clip_on=False)
        ax4.text(when, 1.12, profile_codes[row["Profile_label"]], ha="center", va="bottom", fontsize=7.8)
    may5 = pd.Timestamp("2024-05-05")
    ax4.plot(may5, 0.5, marker="*", ms=8, color="black")
    ax4.annotate(
        "3-5 May interruption",
        xy=(may5, 0.5),
        xytext=(mdates.date2num(pd.Timestamp("2024-03-25")), -0.65),
        textcoords="data",
        arrowprops={"arrowstyle": "-", "lw": 0.7},
        fontsize=7.8,
        ha="left",
    )

    handles = [plt.Line2D([0], [0], color=REGIME_COLORS[x], lw=6, label=x) for x in REGIME_ORDER]
    ax4.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.82), ncol=3, frameon=False, fontsize=7.7)
    ax4.xaxis.set_major_locator(mdates.MonthLocator())
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax4.set_xlim(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31 23:59"))
    ax4.set_xlabel("Date in 2024")

    for ax in [ax1, ax2, ax3, ax4, ax2r, ax3r]:
        ax.grid(False)
    fig.subplots_adjust(left=0.105, right=0.88, top=0.985, bottom=0.15)
    fig.savefig(OUTPUT_DIR / "regime_classification_figure.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "regime_classification_figure.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    raw, stability, mld = read_sources()
    daily = daily_metrics(raw, stability, mld)
    baseline_flags, decay_audit = classify(daily, "PCHIP", BASELINE_THRESHOLD_C)
    results = daily.join(baseline_flags)
    results.insert(0, "Year_day", np.arange(1, len(results) + 1))

    periods = label_periods(results["Regime"])
    robustness, _ = mld_robustness(daily)
    profiles = representative_profile_audit(daily, baseline_flags, stability, mld)
    run_sensitivity = run_length_sensitivity(daily)
    metric_summary = regime_metric_summary(results)

    # Post-hoc seasonal counts are diagnostic only, never input to a rule.
    seasonal_counts = pd.crosstab(
        results["Regime"],
        pd.Index([posthoc_season(x.month) for x in results.index], name="Posthoc_season"),
    ).reindex(REGIME_ORDER, fill_value=0)
    summary = results["Regime"].value_counts().reindex(REGIME_ORDER, fill_value=0).rename("Days").reset_index()
    summary.columns = ["Regime", "Days"]
    summary["Fraction_of_year"] = summary["Days"] / len(results)

    output = results.reset_index()
    output.to_csv(OUTPUT_DIR / "regime_classification_results.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    periods.to_csv(OUTPUT_DIR / "regime_classification_periods.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    robustness.to_csv(OUTPUT_DIR / "regime_classification_MLD_robustness.csv", index=False, encoding="utf-8-sig")
    profiles.to_csv(OUTPUT_DIR / "representative_profile_validation.csv", index=False, encoding="utf-8-sig")
    run_sensitivity.to_csv(OUTPUT_DIR / "classification_rule_sensitivity.csv", index=False, encoding="utf-8-sig")
    metric_summary.to_csv(OUTPUT_DIR / "regime_metric_summary.csv", index=False, encoding="utf-8-sig")
    decay_audit.to_csv(OUTPUT_DIR / "decay_episode_audit.csv", index=False, encoding="utf-8-sig")
    seasonal_counts.to_csv(OUTPUT_DIR / "posthoc_seasonal_regime_counts.csv", encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "regime_counts_summary.csv", index=False, encoding="utf-8-sig")
    make_figure(results, profiles)

    may = results.loc["2024-05-01":"2024-05-08", [
        "DeltaT_mean_C",
        "DeltaT_predawn_median_C",
        "Near_isothermal_fraction",
        "St_mean_J_m2",
        "St_predawn_median_J_m2",
        "N2_median_s2",
        "N2_predawn_median_s2",
        "MLD_PCHIP_0.2C_mean_m",
        "MLD_PCHIP_0.2C_predawn_median_m",
        "MLD_PCHIP_0.2C_deepest_fraction",
        "Short_lived_mixed_interruption",
        "Regime",
    ]]
    checks = {
        "source_files": {"raw_temperature": str(RAW), "stability": str(STABILITY), "mld": str(MLD), "profiles": str(PROFILE)},
        "rules": {
            "deltaT_baseline_C": BASELINE_THRESHOLD_C,
            "predawn_local_time": "04:00 <= time < 06:00",
            "daily_majority_fraction": 0.50,
            "state_min_consecutive_days": STATE_MIN_RUN_DAYS,
            "MLD_amplitude_resolution_m": SENSOR_SPACING_M,
            "rolling_background_days": ROLLING_WINDOW_DAYS,
            "decay_min_consecutive_days": DECAY_MIN_RUN_DAYS,
            "decay_comparison_lag_days": DECAY_COMPARE_LAG_DAYS,
            "decay_link_to_mixed_days": DECAY_LINK_TO_MIXED_DAYS,
        },
        "quality": {
            "daily_rows": int(len(results)),
            "date_min": str(results.index.min().date()),
            "date_max": str(results.index.max().date()),
            "missing_baseline_daily_metrics": int(results[[
                "DeltaT_mean_C", "St_mean_J_m2", "N2_median_s2", "MLD_PCHIP_0.2C_mean_m"
            ]].isna().sum().sum()),
            "exclusive_label_total": int(summary["Days"].sum()),
            "all_six_MLD_scenarios_preserve_interpretation": bool(robustness["Regime_interpretation_preserved"].all()),
            "profile_daily_labels_all_match": bool(profiles["Daily_label_matches_profile"].all()),
            "may5_is_short_lived_interruption": bool(results.loc["2024-05-05", "Short_lived_mixed_interruption"]),
            "may5_not_sustained_mixed_regime": bool(not results.loc["2024-05-05", "Mixed_sustained"]),
        },
        "regime_counts": summary.to_dict(orient="records"),
        "posthoc_seasonal_counts": seasonal_counts.to_dict(),
        "representative_profiles": json.loads(profiles.to_json(orient="records", date_format="iso")),
        "mld_robustness": robustness.to_dict(orient="records"),
        "may_1_8": json.loads(may.reset_index().to_json(orient="records", date_format="iso")),
    }
    (OUTPUT_DIR / "regime_classification_quality_checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    assert len(results) == 366
    assert summary["Days"].sum() == 366
    assert checks["quality"]["missing_baseline_daily_metrics"] == 0
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
