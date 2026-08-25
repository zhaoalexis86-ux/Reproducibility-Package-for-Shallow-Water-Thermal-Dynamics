"""Pre-dawn robustness check for the established CSITE regime classification.

This script does not redesign the classification. It replaces the full-night
median used by the current shortwave-defined method with the final 1 h or 2 h
of the continuous zero-shortwave block immediately before the first measured
positive shortwave value of each day. Calendar seasons are used only after
classification for reporting.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
CURRENT_METHOD_PATH = HERE / "revised_regime_classification_analysis.py"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPOSITORY_ROOT / "results" / "reproduced" / "regime_classification"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_DELTA_C = 0.20
DEPTH_MAX_M = 2.25
SENSOR_SPACING_M = 0.50
NIGHT_THRESHOLD_W_M2 = 0.0
BASELINE_PERSISTENCE_DAYS = 3
MIXED_DIEL_PERSISTENCE_DAYS = 3

REGIME_ORDER = [
    "Near-isothermal mixing",
    "Persistent stratification",
    "Diurnally oscillating stratification",
    "Decaying stratification",
    "Transition/unclassified",
]


def load_current_module():
    spec = importlib.util.spec_from_file_location("current_regime", CURRENT_METHOD_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load current regime method: {CURRENT_METHOD_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CURRENT = load_current_module()


def detect_predawn_windows(
    rsw: pd.Series,
    night_threshold: float = NIGHT_THRESHOLD_W_M2,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Detect the final continuous dark block before measured sunrise.

    Measured sunrise is operationally the first record with Rsw above the
    selected night threshold. The pre-dawn block is the continuous run of
    nighttime records immediately preceding that first positive record.
    """
    values = pd.to_numeric(rsw, errors="coerce")
    interval = values.index.to_series().diff().dropna().median()
    if interval != pd.Timedelta(minutes=10):
        raise ValueError(f"Expected 10-min data; detected median interval {interval}.")

    masks = {
        "Predawn_1h": pd.Series(False, index=values.index),
        "Predawn_2h": pd.Series(False, index=values.index),
    }
    rows: list[dict[str, object]] = []
    for day, group in values.groupby(values.index.normalize()):
        group = group.sort_index()
        illuminated = group.gt(night_threshold)
        if not illuminated.any():
            rows.append({"Date": day, "Valid": False, "Reason": "No positive shortwave record"})
            continue
        sunrise_proxy = illuminated[illuminated].index[0]
        sunrise_position = int(group.index.get_loc(sunrise_proxy))
        cursor = sunrise_position - 1
        block_positions: list[int] = []
        while cursor >= 0:
            timestamp = group.index[cursor]
            if not bool(group.iloc[cursor] <= night_threshold):
                break
            if block_positions:
                later = group.index[block_positions[-1]]
                if later - timestamp != interval:
                    break
            block_positions.append(cursor)
            cursor -= 1
        block_positions = sorted(block_positions)
        block_index = group.index[block_positions] if block_positions else pd.DatetimeIndex([])
        valid = len(block_index) >= 12
        if valid:
            one_hour = block_index[-6:]
            two_hour = block_index[-12:]
            masks["Predawn_1h"].loc[one_hour] = True
            masks["Predawn_2h"].loc[two_hour] = True
            reason = "Complete"
        else:
            one_hour = block_index[-6:]
            two_hour = block_index
            reason = "Fewer than 12 continuous nighttime records"

        rows.append(
            {
                "Date": day,
                "Valid": valid,
                "Reason": reason,
                "Night_threshold_W_m2": night_threshold,
                "Sunrise_proxy_time": sunrise_proxy,
                "Sunrise_proxy_hour_local": sunrise_proxy.hour + sunrise_proxy.minute / 60.0,
                "First_positive_Rsw_W_m2": float(group.loc[sunrise_proxy]),
                "Predawn_continuous_block_start": block_index.min() if len(block_index) else pd.NaT,
                "Predawn_continuous_block_end": block_index.max() if len(block_index) else pd.NaT,
                "Predawn_continuous_block_observations": int(len(block_index)),
                "Predawn_continuous_block_duration_h": float(len(block_index) / 6.0),
                "Predawn_1h_start": one_hour.min() if len(one_hour) else pd.NaT,
                "Predawn_1h_end": one_hour.max() if len(one_hour) else pd.NaT,
                "Predawn_1h_observations": int(len(one_hour)),
                "Predawn_2h_start": two_hour.min() if len(two_hour) else pd.NaT,
                "Predawn_2h_end": two_hour.max() if len(two_hour) else pd.NaT,
                "Predawn_2h_observations": int(len(two_hour)),
            }
        )
    audit = pd.DataFrame(rows).set_index("Date").sort_index()
    return audit, masks


def interval_discrete_metric(
    series: pd.Series,
    window_audit: pd.DataFrame,
    prefix: str,
    operation: str,
) -> tuple[pd.Series, pd.Series]:
    """Summarize existing discrete observations inside each pre-dawn window."""
    result: dict[pd.Timestamp, float] = {}
    counts: dict[pd.Timestamp, int] = {}
    for day, row in window_audit.iterrows():
        start = row[f"{prefix}_start"]
        end = row[f"{prefix}_end"]
        if pd.isna(start) or pd.isna(end):
            result[day] = np.nan
            counts[day] = 0
            continue
        selected = series.loc[(series.index >= start) & (series.index <= end)].dropna()
        counts[day] = int(len(selected))
        if selected.empty:
            result[day] = np.nan
        elif operation == "median":
            result[day] = float(selected.median())
        elif operation == "min":
            result[day] = float(selected.min())
        elif operation == "max":
            result[day] = float(selected.max())
        else:
            raise ValueError(operation)
    return pd.Series(result), pd.Series(counts)


def add_predawn_metrics(
    daily: pd.DataFrame,
    raw: pd.DataFrame,
    stability: pd.DataFrame,
    mld: pd.DataFrame,
    window_audit: pd.DataFrame,
    masks: dict[str, pd.Series],
) -> pd.DataFrame:
    result = daily.copy()
    delta_t = pd.to_numeric(raw["25cm"], errors="coerce") - pd.to_numeric(raw["225cm"], errors="coerce")
    mld_values = pd.to_numeric(mld["MLD_PCHIP_0.2C"], errors="coerce")
    st = pd.to_numeric(stability["St(J/m2)"], errors="coerce")
    n2 = pd.to_numeric(stability["N2_max(1/s2)"], errors="coerce")

    for prefix, mask in masks.items():
        selected_delta = delta_t.loc[mask]
        selected_mld = mld_values.loc[mask.reindex(mld_values.index, fill_value=False)]
        result[f"{prefix}_DeltaT_median_C"] = selected_delta.resample("D").median()
        result[f"{prefix}_DeltaT_min_C"] = selected_delta.resample("D").min()
        result[f"{prefix}_MLD_median_m"] = selected_mld.resample("D").median()
        result[f"{prefix}_MLD_max_m"] = selected_mld.resample("D").max()

        st_median, st_count = interval_discrete_metric(st, window_audit, prefix, "median")
        n2_median, n2_count = interval_discrete_metric(n2, window_audit, prefix, "median")
        result[f"{prefix}_St_median_J_m2"] = st_median.reindex(result.index)
        result[f"{prefix}_N2_median_s2"] = n2_median.reindex(result.index)
        result[f"{prefix}_St_observations_n"] = st_count.reindex(result.index)
        result[f"{prefix}_N2_observations_n"] = n2_count.reindex(result.index)

        result[f"{prefix}_thermal_persistence_signature"] = (
            result[f"{prefix}_DeltaT_median_C"].gt(BASELINE_DELTA_C)
            & result[f"{prefix}_MLD_median_m"].lt(DEPTH_MAX_M)
        )
        result[f"{prefix}_strict_all_window_signature"] = (
            result[f"{prefix}_DeltaT_min_C"].gt(BASELINE_DELTA_C)
            & result[f"{prefix}_MLD_max_m"].lt(DEPTH_MAX_M)
        )
    return result


def classify_predawn(
    daily: pd.DataFrame,
    prefix: str,
    persistent_min_run_days: int = BASELINE_PERSISTENCE_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use pre-dawn metrics in the unchanged physical classification framework."""
    flags = pd.DataFrame(index=daily.index)
    flags["Mixed_daily_core"] = (
        daily["Near_isothermal_fraction"].ge(0.50)
        & daily["MLD_PCHIP_0.2C_deepest_fraction"].ge(0.50)
    )
    flags["Mixed_sustained"] = CURRENT.LEGACY.retain_runs(
        flags["Mixed_daily_core"], MIXED_DIEL_PERSISTENCE_DAYS
    )

    flags["Persistent_daily_core"] = (
        daily[f"{prefix}_DeltaT_median_C"].gt(BASELINE_DELTA_C)
        & daily[f"{prefix}_MLD_median_m"].lt(DEPTH_MAX_M)
    )
    flags["Persistent_sustained"] = CURRENT.LEGACY.retain_runs(
        flags["Persistent_daily_core"], persistent_min_run_days
    )

    flags["Diel_daily_core"] = (
        daily["DeltaT_max_C"].gt(BASELINE_DELTA_C)
        & (daily["DeltaT_max_C"] - daily[f"{prefix}_DeltaT_median_C"]).ge(BASELINE_DELTA_C)
        & daily["DeltaT_range_C"].ge(BASELINE_DELTA_C)
        & daily["MLD_PCHIP_0.2C_amplitude_m"].ge(SENSOR_SPACING_M)
        & ~flags["Mixed_sustained"]
    )
    flags["Diel_sustained"] = CURRENT.LEGACY.retain_runs(
        flags["Diel_daily_core"], MIXED_DIEL_PERSISTENCE_DAYS
    )
    flags["Classification_MLD_amplitude_7d_median_m"] = (
        daily["MLD_PCHIP_0.2C_amplitude_m"].rolling(7, min_periods=4).median()
    )

    decay, decay_audit = CURRENT.LEGACY.qualifying_decay_episodes(
        daily,
        mixed=flags["Mixed_sustained"],
        mld_mean_col="MLD_PCHIP_0.2C_mean_m",
        delta_threshold=BASELINE_DELTA_C,
        compare_lag_days=14,
    )
    flags["Decay_multimetric_episode"] = decay
    flags["Short_lived_mixed_interruption"] = (
        daily[f"{prefix}_DeltaT_median_C"].le(BASELINE_DELTA_C)
        & daily[f"{prefix}_MLD_median_m"].ge(DEPTH_MAX_M - 1e-9)
        & ~flags["Mixed_sustained"]
    )

    label = pd.Series("Transition/unclassified", index=daily.index, dtype="object")
    persistent = flags["Persistent_sustained"]
    diel = flags["Diel_sustained"]
    label.loc[persistent & ~diel] = "Persistent stratification"
    label.loc[diel & ~persistent] = "Diurnally oscillating stratification"
    overlap = persistent & diel
    label.loc[overlap & flags["Classification_MLD_amplitude_7d_median_m"].lt(SENSOR_SPACING_M)] = (
        "Persistent stratification"
    )
    label.loc[overlap & flags["Classification_MLD_amplitude_7d_median_m"].ge(SENSOR_SPACING_M)] = (
        "Diurnally oscillating stratification"
    )
    label.loc[flags["Decay_multimetric_episode"] & ~flags["Mixed_sustained"]] = "Decaying stratification"
    label.loc[flags["Mixed_sustained"]] = "Near-isothermal mixing"
    flags["Regime"] = label
    return flags, decay_audit


def posthoc_season(index: pd.DatetimeIndex) -> pd.Series:
    values = np.select(
        [index.month.isin([12, 1, 2]), index.month.isin([3, 4, 5]), index.month.isin([6, 7, 8])],
        ["Winter", "Spring", "Summer"],
        default="Autumn",
    )
    return pd.Series(values, index=index, name="Posthoc_season")


def dominant_regime(labels: pd.Series, season: str) -> str:
    seasons = posthoc_season(labels.index)
    counts = labels.loc[seasons.eq(season)].value_counts()
    if counts.empty:
        return "No data"
    maximum = counts.max()
    return " / ".join(r for r in REGIME_ORDER if counts.get(r, 0) == maximum)


def autumn_interpretation(labels: pd.Series) -> dict[str, object]:
    autumn = labels.loc[labels.index.month.isin([9, 10, 11])]
    decay = int(autumn.eq("Decaying stratification").sum())
    diel = int(autumn.eq("Diurnally oscillating stratification").sum())
    mixed = int(autumn.eq("Near-isothermal mixing").sum())
    sep_mixed = int(labels.loc[labels.index.month == 9].eq("Near-isothermal mixing").sum())
    nov_mixed = int(labels.loc[labels.index.month == 11].eq("Near-isothermal mixing").sum())
    supported = decay > 0 and diel > 0 and nov_mixed > sep_mixed
    return {
        "Autumn_decay_days": decay,
        "Autumn_diel_days": diel,
        "Autumn_mixed_days": mixed,
        "September_mixed_days": sep_mixed,
        "November_mixed_days": nov_mixed,
        "Autumn_progressive_transition_supported": supported,
    }


def jaccard_target(current: pd.Series, alternative: pd.Series, season: str, regime: str) -> float:
    seasons = posthoc_season(current.index)
    season_mask = seasons.eq(season)
    a = current.eq(regime) & season_mask
    b = alternative.eq(regime) & season_mask
    union = int((a | b).sum())
    return 100.0 * int((a & b).sum()) / union if union else np.nan


def consecutive_periods_within_season(labels: pd.Series, regime: str, season: str) -> pd.DataFrame:
    seasons = posthoc_season(labels.index)
    mask = labels.eq(regime) & seasons.eq(season)
    periods = CURRENT.LEGACY.contiguous_periods(mask, regime)
    if periods.empty:
        return periods
    periods.insert(0, "Season", season)
    return periods


def distribution_record(values: pd.Series, stem: str) -> dict[str, object]:
    values = values.dropna()
    return {
        f"{stem}_N": int(len(values)),
        f"{stem}_Min": float(values.min()),
        f"{stem}_P25": float(values.quantile(0.25)),
        f"{stem}_Median": float(values.median()),
        f"{stem}_P75": float(values.quantile(0.75)),
        f"{stem}_Max": float(values.max()),
    }


def spring_summary(
    daily: pd.DataFrame,
    method_records: list[tuple[str, pd.Series, pd.Series, pd.Series, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spring_mask = daily.index.month.isin([3, 4, 5])
    rows: list[dict[str, object]] = []
    episodes: list[pd.DataFrame] = []
    for method, labels, core, sustained, prefix in method_records:
        spring_labels = labels.loc[spring_mask]
        spring_core = core.loc[spring_mask]
        delta_col = "DeltaT_night_median_C" if method == "Current full-night" else f"{prefix}_DeltaT_median_C"
        mld_col = "MLD_PCHIP_0.2C_night_median_m" if method == "Current full-night" else f"{prefix}_MLD_median_m"
        st_col = "St_night_median_J_m2" if method == "Current full-night" else f"{prefix}_St_median_J_m2"
        n2_col = "N2_night_median_s2" if method == "Current full-night" else f"{prefix}_N2_median_s2"
        row: dict[str, object] = {
            "Method": method,
            "Spring_days": int(spring_mask.sum()),
            "Persistent_days": int(spring_labels.eq("Persistent stratification").sum()),
            "Persistent_days_fraction_percent": 100.0 * float(spring_labels.eq("Persistent stratification").mean()),
            "Nights_retaining_thermal_stratification": int(spring_core.sum()),
            "Nights_retaining_thermal_stratification_fraction_percent": 100.0 * float(spring_core.mean()),
        }
        row.update(distribution_record(daily.loc[spring_mask, delta_col], "DeltaT_C"))
        row.update(distribution_record(daily.loc[spring_mask, mld_col], "MLD_m"))
        row.update(distribution_record(daily.loc[spring_mask, st_col], "St_J_m2"))
        row.update(distribution_record(daily.loc[spring_mask, n2_col], "N2_s2"))
        evidence_mask = sustained & spring_mask
        evidence_periods = CURRENT.LEGACY.contiguous_periods(evidence_mask, "Persistent evidence")
        final_periods = consecutive_periods_within_season(labels, "Persistent stratification", "Spring")
        row["Persistent_evidence_episode_count"] = int(len(evidence_periods))
        row["Longest_persistent_evidence_episode_days"] = int(evidence_periods["Duration_days"].max()) if len(evidence_periods) else 0
        row["Final_persistent_label_segment_count"] = int(len(final_periods))
        rows.append(row)
        if len(evidence_periods):
            evidence_periods["Season"] = "Spring"
            evidence_periods.insert(0, "Episode_type", "Persistent_sustained evidence")
            evidence_periods.insert(0, "Method", method)
            episodes.append(evidence_periods)
        if len(final_periods):
            final_periods.insert(0, "Episode_type", "Final persistent regime label")
            final_periods.insert(0, "Method", method)
            episodes.append(final_periods)
    episode_table = pd.concat(episodes, ignore_index=True) if episodes else pd.DataFrame()
    return pd.DataFrame(rows), episode_table


def main() -> None:
    raw, stability, mld = CURRENT.read_sources()
    rsw = pd.to_numeric(raw[CURRENT.SHORTWAVE_COLUMN], errors="coerce")
    current_daily = CURRENT.daily_metrics_with_nighttime(raw, stability, mld, NIGHT_THRESHOLD_W_M2)
    current_flags, _ = CURRENT.classify_with_nighttime(current_daily)

    window_audit, masks = detect_predawn_windows(rsw)
    daily = add_predawn_metrics(current_daily, raw, stability, mld, window_audit, masks)
    flags_1h, _ = classify_predawn(daily, "Predawn_1h", 3)
    flags_2h, _ = classify_predawn(daily, "Predawn_2h", 3)

    labels = {
        "Current full-night": current_flags["Regime"],
        "Pre-dawn 1 h": flags_1h["Regime"],
        "Pre-dawn 2 h": flags_2h["Regime"],
    }

    count_rows: list[dict[str, object]] = []
    for regime in REGIME_ORDER:
        count_rows.append(
            {
                "Regime": regime,
                "Current_classification_days": int(labels["Current full-night"].eq(regime).sum()),
                "Predawn_1h_classification_days": int(labels["Pre-dawn 1 h"].eq(regime).sum()),
                "Predawn_2h_classification_days": int(labels["Pre-dawn 2 h"].eq(regime).sum()),
            }
        )
    count_table = pd.DataFrame(count_rows)

    agreement_rows: list[dict[str, object]] = []
    current = labels["Current full-night"]
    for method in ["Pre-dawn 1 h", "Pre-dawn 2 h"]:
        alternative = labels[method]
        autumn = autumn_interpretation(alternative)
        agreement_rows.append(
            {
                "Method": method,
                "Daily_classification_agreement_percent": 100.0 * float(current.eq(alternative).mean()),
                "Changed_daily_labels": int(current.ne(alternative).sum()),
                "Spring_persistent_day_Jaccard_percent": jaccard_target(current, alternative, "Spring", "Persistent stratification"),
                "Winter_mixed_day_Jaccard_percent": jaccard_target(current, alternative, "Winter", "Near-isothermal mixing"),
                "Summer_diel_day_Jaccard_percent": jaccard_target(current, alternative, "Summer", "Diurnally oscillating stratification"),
                "Winter_dominant_regime": dominant_regime(alternative, "Winter"),
                "Spring_dominant_regime": dominant_regime(alternative, "Spring"),
                "Summer_dominant_regime": dominant_regime(alternative, "Summer"),
                **autumn,
                "Principal_seasonal_interpretation_preserved": (
                    dominant_regime(alternative, "Winter") == "Near-isothermal mixing"
                    and dominant_regime(alternative, "Spring") == "Persistent stratification"
                    and dominant_regime(alternative, "Summer") == "Diurnally oscillating stratification"
                    and bool(autumn["Autumn_progressive_transition_supported"])
                ),
            }
        )
    agreement_table = pd.DataFrame(agreement_rows)

    seasons = posthoc_season(daily.index)
    seasonal_rows: list[dict[str, object]] = []
    for method, method_labels in labels.items():
        table = pd.crosstab(method_labels, seasons).reindex(index=REGIME_ORDER, columns=["Winter", "Spring", "Summer", "Autumn"], fill_value=0)
        for regime in REGIME_ORDER:
            seasonal_rows.append(
                {
                    "Method": method,
                    "Regime": regime,
                    **{f"{season}_days": int(table.loc[regime, season]) for season in ["Winter", "Spring", "Summer", "Autumn"]},
                }
            )
    seasonal_table = pd.DataFrame(seasonal_rows)

    sensitivity_rows: list[dict[str, object]] = []
    sensitivity_labels: dict[tuple[str, int], pd.Series] = {}
    for prefix, method in [("Predawn_1h", "Pre-dawn 1 h"), ("Predawn_2h", "Pre-dawn 2 h")]:
        for run_days in [2, 3, 5]:
            scenario_flags, _ = classify_predawn(daily, prefix, run_days)
            scenario_labels = scenario_flags["Regime"]
            sensitivity_labels[(method, run_days)] = scenario_labels
            counts = scenario_labels.value_counts()
            autumn = autumn_interpretation(scenario_labels)
            sensitivity_rows.append(
                {
                    "Method": method,
                    "Persistent_consecutive_days": run_days,
                    "Near_isothermal_mixing_days": int(counts.get("Near-isothermal mixing", 0)),
                    "Persistent_stratification_days": int(counts.get("Persistent stratification", 0)),
                    "Diurnally_oscillating_days": int(counts.get("Diurnally oscillating stratification", 0)),
                    "Decaying_stratification_days": int(counts.get("Decaying stratification", 0)),
                    "Transition_unclassified_days": int(counts.get("Transition/unclassified", 0)),
                    "Winter_dominant_regime": dominant_regime(scenario_labels, "Winter"),
                    "Spring_dominant_regime": dominant_regime(scenario_labels, "Spring"),
                    "Summer_dominant_regime": dominant_regime(scenario_labels, "Summer"),
                    **autumn,
                    "Principal_seasonal_interpretation_preserved": (
                        dominant_regime(scenario_labels, "Winter") == "Near-isothermal mixing"
                        and dominant_regime(scenario_labels, "Spring") == "Persistent stratification"
                        and dominant_regime(scenario_labels, "Summer") == "Diurnally oscillating stratification"
                        and bool(autumn["Autumn_progressive_transition_supported"])
                    ),
                }
            )
    sensitivity_table = pd.DataFrame(sensitivity_rows)

    spring_table, spring_episodes = spring_summary(
        daily,
        [
            ("Current full-night", current_flags["Regime"], current_flags["Persistent_daily_core"], current_flags["Persistent_sustained"], ""),
            ("Pre-dawn 1 h", flags_1h["Regime"], flags_1h["Persistent_daily_core"], flags_1h["Persistent_sustained"], "Predawn_1h"),
            ("Pre-dawn 2 h", flags_2h["Regime"], flags_2h["Persistent_daily_core"], flags_2h["Persistent_sustained"], "Predawn_2h"),
        ],
    )

    spring_columns = [
        "DeltaT_night_median_C",
        "MLD_PCHIP_0.2C_night_median_m",
        "St_night_median_J_m2",
        "N2_night_median_s2",
        "Predawn_1h_DeltaT_median_C",
        "Predawn_1h_DeltaT_min_C",
        "Predawn_1h_MLD_median_m",
        "Predawn_1h_MLD_max_m",
        "Predawn_1h_St_median_J_m2",
        "Predawn_1h_N2_median_s2",
        "Predawn_2h_DeltaT_median_C",
        "Predawn_2h_DeltaT_min_C",
        "Predawn_2h_MLD_median_m",
        "Predawn_2h_MLD_max_m",
        "Predawn_2h_St_median_J_m2",
        "Predawn_2h_N2_median_s2",
    ]
    spring_daily = daily.loc[daily.index.month.isin([3, 4, 5]), spring_columns].copy()
    spring_daily.insert(0, "Current_Regime", current_flags.loc[spring_daily.index, "Regime"])
    spring_daily.insert(1, "Predawn_1h_Regime", flags_1h.loc[spring_daily.index, "Regime"])
    spring_daily.insert(2, "Predawn_2h_Regime", flags_2h.loc[spring_daily.index, "Regime"])
    spring_daily["Current_persistent_core"] = current_flags.loc[spring_daily.index, "Persistent_daily_core"]
    spring_daily["Predawn_1h_persistent_core"] = flags_1h.loc[spring_daily.index, "Persistent_daily_core"]
    spring_daily["Predawn_2h_persistent_core"] = flags_2h.loc[spring_daily.index, "Persistent_daily_core"]
    spring_daily["Current_persistent_sustained"] = current_flags.loc[spring_daily.index, "Persistent_sustained"]
    spring_daily["Predawn_1h_persistent_sustained"] = flags_1h.loc[spring_daily.index, "Persistent_sustained"]
    spring_daily["Predawn_2h_persistent_sustained"] = flags_2h.loc[spring_daily.index, "Persistent_sustained"]

    may = daily.loc["2024-05-03":"2024-05-08", [
        "DeltaT_night_median_C",
        "MLD_PCHIP_0.2C_night_median_m",
        "St_night_median_J_m2",
        "N2_night_median_s2",
        "Predawn_1h_DeltaT_median_C",
        "Predawn_1h_DeltaT_min_C",
        "Predawn_1h_MLD_median_m",
        "Predawn_1h_MLD_max_m",
        "Predawn_1h_St_median_J_m2",
        "Predawn_1h_N2_median_s2",
        "Predawn_2h_DeltaT_median_C",
        "Predawn_2h_DeltaT_min_C",
        "Predawn_2h_MLD_median_m",
        "Predawn_2h_MLD_max_m",
        "Predawn_2h_St_median_J_m2",
        "Predawn_2h_N2_median_s2",
    ]].copy()
    may.insert(0, "Current_Regime", current_flags.loc[may.index, "Regime"])
    may.insert(1, "Predawn_1h_Regime", flags_1h.loc[may.index, "Regime"])
    may.insert(2, "Predawn_2h_Regime", flags_2h.loc[may.index, "Regime"])
    may["Predawn_1h_short_lived_mixed_interruption"] = flags_1h.loc[may.index, "Short_lived_mixed_interruption"]
    may["Predawn_2h_short_lived_mixed_interruption"] = flags_2h.loc[may.index, "Short_lived_mixed_interruption"]

    oct25 = pd.Timestamp("2024-10-25")
    oct25_time = pd.Timestamp("2024-10-25 15:00")
    delta_15 = float(pd.to_numeric(raw.loc[oct25_time, "25cm"]) - pd.to_numeric(raw.loc[oct25_time, "225cm"]))
    oct25_rows = []
    for method, method_labels, prefix in [
        ("Current full-night", current_flags["Regime"], ""),
        ("Pre-dawn 1 h", flags_1h["Regime"], "Predawn_1h"),
        ("Pre-dawn 2 h", flags_2h["Regime"], "Predawn_2h"),
    ]:
        predawn_delta = daily.loc[oct25, "DeltaT_night_median_C"] if not prefix else daily.loc[oct25, f"{prefix}_DeltaT_median_C"]
        predawn_mld = daily.loc[oct25, "MLD_PCHIP_0.2C_night_median_m"] if not prefix else daily.loc[oct25, f"{prefix}_MLD_median_m"]
        erosion_restratification = (
            predawn_delta <= BASELINE_DELTA_C
            and daily.loc[oct25, "DeltaT_max_C"] > BASELINE_DELTA_C
            and daily.loc[oct25, "DeltaT_range_C"] >= BASELINE_DELTA_C
            and daily.loc[oct25, "MLD_PCHIP_0.2C_amplitude_m"] >= SENSOR_SPACING_M
        )
        oct25_rows.append(
            {
                "Method": method,
                "Regime": method_labels.loc[oct25],
                "Predawn_DeltaT_median_C": float(predawn_delta),
                "Predawn_MLD_median_m": float(predawn_mld),
                "Daily_DeltaT_max_C": float(daily.loc[oct25, "DeltaT_max_C"]),
                "Daily_DeltaT_range_C": float(daily.loc[oct25, "DeltaT_range_C"]),
                "Daily_MLD_amplitude_m": float(daily.loc[oct25, "MLD_PCHIP_0.2C_amplitude_m"]),
                "DeltaT_at_15_00_C": delta_15,
                "Nighttime_erosion_followed_by_daytime_restratification": bool(erosion_restratification),
                "Recommended_description": "Autumn transition with daytime restratification",
            }
        )
    oct25_table = pd.DataFrame(oct25_rows)

    daily_results = daily.copy()
    daily_results.insert(0, "Current_Regime", current_flags["Regime"])
    daily_results.insert(1, "Predawn_1h_Regime", flags_1h["Regime"])
    daily_results.insert(2, "Predawn_2h_Regime", flags_2h["Regime"])
    daily_results["Current_vs_Predawn_1h_agreement"] = current_flags["Regime"].eq(flags_1h["Regime"])
    daily_results["Current_vs_Predawn_2h_agreement"] = current_flags["Regime"].eq(flags_2h["Regime"])
    daily_results["Predawn_1h_persistent_core"] = flags_1h["Persistent_daily_core"]
    daily_results["Predawn_2h_persistent_core"] = flags_2h["Persistent_daily_core"]

    count_table.to_csv(OUTPUT_DIR / "predawn_regime_count_comparison.csv", index=False, encoding="utf-8-sig")
    agreement_table.to_csv(OUTPUT_DIR / "predawn_agreement_summary.csv", index=False, encoding="utf-8-sig")
    seasonal_table.to_csv(OUTPUT_DIR / "predawn_seasonal_counts.csv", index=False, encoding="utf-8-sig")
    sensitivity_table.to_csv(OUTPUT_DIR / "predawn_persistence_sensitivity.csv", index=False, encoding="utf-8-sig")
    spring_table.to_csv(OUTPUT_DIR / "predawn_spring_summary.csv", index=False, encoding="utf-8-sig")
    spring_episodes.to_csv(OUTPUT_DIR / "predawn_spring_episodes.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    spring_daily.reset_index().to_csv(OUTPUT_DIR / "predawn_spring_persistence.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    may.reset_index().to_csv(OUTPUT_DIR / "predawn_event_May3_8.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    oct25_table.to_csv(OUTPUT_DIR / "predawn_oct25_audit.csv", index=False, encoding="utf-8-sig")
    window_audit.reset_index().to_csv(OUTPUT_DIR / "predawn_window_audit.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M")
    daily_results.reset_index().to_csv(OUTPUT_DIR / "predawn_daily_results.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")

    checks = {
        "source_files": {
            "current_method": str(CURRENT_METHOD_PATH),
            "raw": str(CURRENT.RAW),
            "stability": str(CURRENT.STABILITY),
            "mld": str(CURRENT.MLD),
        },
        "night_and_predawn": {
            "threshold_W_m2": NIGHT_THRESHOLD_W_M2,
            "negative_Rsw_observations": int(rsw.lt(0).sum()),
            "exact_zero_observations": int(rsw.eq(0).sum()),
            "positive_Rsw_0_to_1_W_m2": int((rsw.gt(0) & rsw.le(1)).sum()),
            "valid_predawn_days": int(window_audit["Valid"].sum()),
            "sunrise_proxy_hour_min": float(window_audit["Sunrise_proxy_hour_local"].min()),
            "sunrise_proxy_hour_median": float(window_audit["Sunrise_proxy_hour_local"].median()),
            "sunrise_proxy_hour_max": float(window_audit["Sunrise_proxy_hour_local"].max()),
            "one_hour_all_six_observations": bool(window_audit["Predawn_1h_observations"].eq(6).all()),
            "two_hour_all_twelve_observations": bool(window_audit["Predawn_2h_observations"].eq(12).all()),
            "one_hour_St_min_observations": int(daily["Predawn_1h_St_observations_n"].min()),
            "two_hour_St_min_observations": int(daily["Predawn_2h_St_observations_n"].min()),
        },
        "comparison": agreement_table.to_dict(orient="records"),
        "counts": count_table.to_dict(orient="records"),
        "spring": spring_table.to_dict(orient="records"),
        "persistence_sensitivity": sensitivity_table.to_dict(orient="records"),
        "may3_8": json.loads(may.reset_index().to_json(orient="records", date_format="iso")),
        "oct25": oct25_table.to_dict(orient="records"),
        "quality": {
            "daily_rows": int(len(daily_results)),
            "missing_predawn_thermal_metrics": int(daily[[
                "Predawn_1h_DeltaT_median_C",
                "Predawn_1h_MLD_median_m",
                "Predawn_2h_DeltaT_median_C",
                "Predawn_2h_MLD_median_m",
            ]].isna().sum().sum()),
            "missing_predawn_st_n2_metrics": int(daily[[
                "Predawn_1h_St_median_J_m2",
                "Predawn_1h_N2_median_s2",
                "Predawn_2h_St_median_J_m2",
                "Predawn_2h_N2_median_s2",
            ]].isna().sum().sum()),
            "all_two_predawn_methods_preserve_interpretation": bool(
                agreement_table["Principal_seasonal_interpretation_preserved"].all()
            ),
            "all_six_persistence_scenarios_preserve_interpretation": bool(
                sensitivity_table["Principal_seasonal_interpretation_preserved"].all()
            ),
        },
    }
    (OUTPUT_DIR / "predawn_quality_checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    assert len(daily_results) == 366
    assert checks["night_and_predawn"]["valid_predawn_days"] == 366
    assert checks["night_and_predawn"]["one_hour_all_six_observations"]
    assert checks["night_and_predawn"]["two_hour_all_twelve_observations"]
    assert checks["quality"]["missing_predawn_thermal_metrics"] == 0
    assert checks["quality"]["missing_predawn_st_n2_metrics"] == 0
    print(json.dumps(checks, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
