"""OAT uncertainty and sensitivity analysis for the CSITE study.

The script does not edit the manuscript data or the original calculation code.
It reproduces the archived Figure 6 baseline, varies one parameter at a time,
and independently recalculates MLD using PCHIP and linear interpolation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.optimize import brentq


SIGMA = 5.67e-8
ALBEDO = 0.07
EMISS_W = 0.97
CE = 1.3e-3
CH = 1.3e-3
WIND_EXPONENT = 1 / 7
ATM_CLEAR_COEFF = 0.642
CLOUD_COEFF = 0.17
DEPTHS = np.array([0.25, 0.75, 1.25, 1.75, 2.25])
TEMP_COLS = ["25cm", "75cm", "125cm", "175cm", "225cm"]
DT_SECONDS = 600.0
DELTA_H_ANNUAL_MJ_M2 = -4.753581788840919
ANALYSIS_DURATION_SECONDS = 365.99305555555554 * 86400.0

SEASON_MONTHS = {
    "Spring": [3, 4, 5],
    "Summer": [6, 7, 8],
    "Autumn": [9, 10, 11],
    "Winter": [12, 1, 2],
}

SOURCES = {
    "albedo": "https://doi.org/10.1029/2022JD038355",
    "emissivity": "https://doi.org/10.1029/2018JC014451",
    "transfer": "https://doi.org/10.1029/2021JD036099",
    "cloud": "https://doi.org/10.1016/j.solener.2010.01.012",
    "wind": "https://doi.org/10.1175/1520-0450(1994)033%3C0757:DTPLWP%3E2.0.CO;2",
    "mld": "https://doi.org/10.5194/hess-24-5559-2020",
    "pchip": "https://doi.org/10.1137/0717021",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def locate_package(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    return Path(__file__).resolve().parents[2]


def locate_by_pattern(root: Path, pattern: str) -> Path:
    hits = sorted(root.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No file matched: {root / pattern}")
    return hits[0]


def season_of(index: pd.DatetimeIndex) -> pd.Series:
    values = np.full(len(index), "Winter", dtype=object)
    months = index.month
    for season, members in SEASON_MONTHS.items():
        values[np.isin(months, members)] = season
    return pd.Series(values, index=index, name="Season")


def calculate_theoretical_solar_radiation(index: pd.DatetimeIndex) -> np.ndarray:
    doy = index.dayofyear.to_numpy()
    hour = index.hour.to_numpy()
    minute = index.minute.to_numpy()
    b = 2 * np.pi * (doy - 81) / 364.0
    eot = 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)
    local_time = hour + minute / 60.0
    tst = local_time + (114.31 - 120.0) * 4.0 / 60.0 + eot / 60.0
    omega = (tst - 12.0) * 15.0 * np.pi / 180.0
    delta = np.arcsin(0.39795 * np.cos(0.98563 * (doy - 173) * np.pi / 180.0))
    lat = np.deg2rad(30.47)
    sin_elevation = np.maximum(
        np.sin(lat) * np.sin(delta) + np.cos(lat) * np.cos(delta) * np.cos(omega), 0
    )
    return 0.75 * 1367.0 * (1 + 0.033 * np.cos(2 * np.pi * doy / 365.0)) * sin_elevation


def calc_vapor_pressure(temp_c: pd.Series) -> pd.Series:
    """Saturation vapour pressure (Pa), identical to the Figure 6 implementation."""
    return 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5)) * 100


def infer_cloud(index: pd.DatetimeIndex, shortwave: pd.Series, rso_scale: float) -> pd.Series:
    rso = calculate_theoretical_solar_radiation(index) * rso_scale
    cloud = pd.Series(np.nan, index=index, dtype=float)
    daytime = rso > 20
    ratio = np.clip(shortwave.loc[daytime].to_numpy() / rso[daytime], 0, 1)
    cloud.loc[daytime] = np.sqrt(1 - ratio)
    return cloud.interpolate(method="time").bfill().ffill()


def flux_summary(
    scenario_id: str,
    parameter: str,
    case: str,
    test_value: float,
    unit: str,
    qnet: pd.Series,
    components: dict[str, pd.Series],
    baseline_qnet: pd.Series,
    basis: str,
    source: str,
) -> tuple[dict, list[dict]]:
    mask = baseline_qnet.notna()
    q = qnet.where(mask)
    base = baseline_qnet.where(mask)
    annual_mean = q.mean()
    base_mean = base.mean()
    # Match the first-stage right-endpoint convention: intervals start at row 1.
    cumulative_valid = q.iloc[1:].sum() * DT_SECONDS / 1e6
    q_fill = q.interpolate(method="time", limit_area="inside")
    cumulative_cont = q_fill.iloc[1:].sum() * DT_SECONDS / 1e6
    row = {
        "Scenario_ID": scenario_id,
        "Parameter": parameter,
        "Case": case,
        "Test_value": test_value,
        "Unit": unit,
        "Annual_mean_Qnet_W_m2": annual_mean,
        "Annual_Qnet_change_W_m2": annual_mean - base_mean,
        "Annual_Qnet_change_percent": (annual_mean - base_mean) / abs(base_mean) * 100,
        "Cumulative_Qnet_valid_MJ_m2": cumulative_valid,
        "Cumulative_Qnet_continuous_MJ_m2": cumulative_cont,
        "Annual_residual_continuous_MJ_m2": cumulative_cont - DELTA_H_ANNUAL_MJ_M2,
        "Mean_residual_continuous_W_m2": (cumulative_cont - DELTA_H_ANNUAL_MJ_M2) * 1e6 / ANALYSIS_DURATION_SECONDS,
        "Mean_Qsw_W_m2": components["Q_sw"].where(mask).mean(),
        "Mean_Qlw_W_m2": components["Q_lw"].where(mask).mean(),
        "Mean_Qe_W_m2": components["Q_e"].where(mask).mean(),
        "Mean_Qh_W_m2": components["Q_h"].where(mask).mean(),
        "Range_basis": basis,
        "Source": source,
    }
    seasonal = []
    seasons = season_of(q.index)
    for season in ["Spring", "Summer", "Autumn", "Winter"]:
        smask = seasons.eq(season) & mask
        smean = q.loc[smask].mean()
        bmean = base.loc[smask].mean()
        seasonal.append({
            "Scenario_ID": scenario_id,
            "Parameter": parameter,
            "Case": case,
            "Season": season,
            "Mean_Qnet_W_m2": smean,
            "Change_W_m2": smean - bmean,
            "Change_percent_of_baseline_season": (smean - bmean) / abs(bmean) * 100,
            "Mean_Qsw_W_m2": components["Q_sw"].loc[smask].mean(),
            "Mean_Qlw_W_m2": components["Q_lw"].loc[smask].mean(),
            "Mean_Qe_W_m2": components["Q_e"].loc[smask].mean(),
            "Mean_Qh_W_m2": components["Q_h"].loc[smask].mean(),
        })
    return row, seasonal


def surface_oat(raw_path: Path, archived_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    raw = pd.read_csv(raw_path, index_col=0, parse_dates=True, encoding="utf-8-sig")
    raw = raw.loc[:, ~raw.columns.astype(str).str.startswith("Unnamed:")]
    raw = raw.sort_index()
    arc = pd.read_csv(archived_path, index_col=0, parse_dates=True, encoding="utf-8-sig")
    arc = arc.sort_index()
    if not raw.index.equals(arc.index):
        raise ValueError("Raw and archived Figure 6 timestamps do not match.")

    rad_col = next(c for c in raw.columns if "辐射" in c and "累计" not in c)
    Rsw = pd.to_numeric(raw[rad_col], errors="coerce").clip(lower=0)
    base = {k: pd.to_numeric(arc[k], errors="coerce") for k in ["Q_sw", "Q_lw", "Q_e", "Q_h", "Q_net"]}
    C0 = pd.to_numeric(arc["Cloud_Cover_C"], errors="coerce")
    Ta = pd.to_numeric(arc["Ta_C"], errors="coerce") if "Ta_C" in arc else pd.to_numeric(raw[next(c for c in raw if "大气温度" in c)], errors="coerce")
    Ts = pd.to_numeric(raw["25cm"], errors="coerce")
    rh_col = next(c for c in raw.columns if "湿度" in c)
    RH = pd.to_numeric(raw[rh_col], errors="coerce") / 100
    ea = calc_vapor_pressure(Ta) * RH
    tk = Ta + 273.15
    outgoing_unit = SIGMA * (Ts + 273.15) ** 4

    def total(q_sw=None, q_lw=None, q_e=None, q_h=None):
        return (base["Q_sw"] if q_sw is None else q_sw) + (base["Q_lw"] if q_lw is None else q_lw) + (base["Q_e"] if q_e is None else q_e) + (base["Q_h"] if q_h is None else q_h)

    rows, seasons = [], []
    specs = []
    specs.append(("BASE", "Baseline", "baseline", 1.0, "baseline", base["Q_net"], base, "Original archived calculation", "Original code"))
    for val, case in [(0.05, "low"), (ALBEDO, "baseline"), (0.10, "high")]:
        qsw = Rsw * (1 - val)
        comps = {**base, "Q_sw": qsw}
        specs.append((f"ALB_{val:.2f}", "Water albedo", case, val, "dimensionless", total(q_sw=qsw), comps, "Literature-informed 0.05–0.10 open-water sensitivity bracket; 0.07 retained", SOURCES["albedo"]))
    for val, case in [(0.96, "low"), (EMISS_W, "baseline"), (0.99, "high")]:
        qlw = base["Q_lw"] + (EMISS_W - val) * outgoing_unit
        comps = {**base, "Q_lw": qlw}
        specs.append((f"EMIS_{val:.2f}", "Water emissivity", case, val, "dimensionless", total(q_lw=qlw), comps, "Water thermal-infrared literature range 0.96–0.99; 0.97 retained", SOURCES["emissivity"]))
    for pname, col, baseval in [("Latent transfer coefficient CE", "Q_e", CE), ("Sensible transfer coefficient CH", "Q_h", CH)]:
        for pct in [-20, -10, 0, 10, 20]:
            val = baseval * (1 + pct / 100)
            changed = base[col] * val / baseval
            comps = {**base, col: changed}
            key = "CE" if col == "Q_e" else "CH"
        specs.append((f"{key}_{pct:+d}", pname, f"{pct:+d}%", val, "dimensionless", total(**{col.lower(): changed}), comps, "Prescribed ±10% and ±20% OAT bracket", SOURCES["transfer"]))
    for scale, case in [(0.9, "-10%"), (1.0, "baseline"), (1.1, "+10%")]:
        c = infer_cloud(raw.index, Rsw, scale)
        emiss = np.clip(ATM_CLEAR_COEFF * (ea / tk) ** (1 / 7) * (1 + CLOUD_COEFF * c**2), 0, 0.98)
        qlw = emiss * SIGMA * tk**4 - EMISS_W * outgoing_unit
        comps = {**base, "Q_lw": qlw}
        specs.append((f"RSO_{scale:.1f}", "Clear-sky radiation scale", case, scale, "multiplier", total(q_lw=qlw), comps, "Scenario assumption ±10%; cloud cover re-inferred from measured shortwave", SOURCES["cloud"]))
    for scale, case in [(0.8, "-20%"), (1.0, "baseline"), (1.2, "+20%")]:
        c = np.clip(C0 * scale, 0, 1)
        emiss = np.clip(ATM_CLEAR_COEFF * (ea / tk) ** (1 / 7) * (1 + CLOUD_COEFF * c**2), 0, 0.98)
        qlw = emiss * SIGMA * tk**4 - EMISS_W * outgoing_unit
        comps = {**base, "Q_lw": qlw}
        specs.append((f"CLOUD_{scale:.1f}", "Inferred cloud-cover scale", case, scale, "multiplier", total(q_lw=qlw), comps, "Scenario assumption ±20%; includes nighttime interpolated cloud series", SOURCES["cloud"]))
    for val, case in [(1 / 8, "1/8"), (WIND_EXPONENT, "1/7 baseline"), (1 / 6, "1/6")]:
        ratio = 5 ** (val - WIND_EXPONENT)
        qe, qh = base["Q_e"] * ratio, base["Q_h"] * ratio
        comps = {**base, "Q_e": qe, "Q_h": qh}
        specs.append((f"WIND_{case}", "2-m to 10-m wind exponent", case, val, "dimensionless", total(q_e=qe, q_h=qh), comps, "Tested 1/8, 1/7, and 1/6 values; consistent with over-water exponent uncertainty", SOURCES["wind"]))

    for scenario_id, parameter, case, value, unit, qnet, comps, basis, source in specs:
        row, sea = flux_summary(
            scenario_id, parameter, case, value, unit, qnet, comps,
            base["Q_net"], basis, source,
        )
        rows.append(row); seasons.extend(sea)
    detail = pd.DataFrame(rows).drop_duplicates(["Parameter", "Case"], keep="first")
    baseline_components = detail.loc[detail.Parameter.eq("Baseline")].iloc[0]
    for comp in ["Qsw", "Qlw", "Qe", "Qh"]:
        detail[f"Delta_mean_{comp}_W_m2"] = detail[f"Mean_{comp}_W_m2"] - baseline_components[f"Mean_{comp}_W_m2"]
    baseline_cont_residual = float(detail.loc[detail.Parameter.eq("Baseline"), "Mean_residual_continuous_W_m2"].iloc[0])
    detail["Residual_reduction_percent"] = (
        baseline_cont_residual - detail["Mean_residual_continuous_W_m2"]
    ) / baseline_cont_residual * 100
    seasonal = pd.DataFrame(seasons).merge(detail[["Scenario_ID"]], on="Scenario_ID", how="inner")

    summaries = []
    baseline_lookup = {
        "Water albedo": ALBEDO,
        "Water emissivity": EMISS_W,
        "Latent transfer coefficient CE": CE,
        "Sensible transfer coefficient CH": CH,
        "Clear-sky radiation scale": 1.0,
        "Inferred cloud-cover scale": 1.0,
        "2-m to 10-m wind exponent": WIND_EXPONENT,
    }
    for parameter, b in detail[detail.Parameter.ne("Baseline")].groupby("Parameter", sort=False):
        sb = seasonal[seasonal.Parameter.eq(parameter)]
        summaries.append({
            "Parameter": parameter,
            "Baseline": baseline_lookup[parameter],
            "Test_range": f"{b.Test_value.min():.6g}–{b.Test_value.max():.6g}",
            "Maximum_abs_change_annual_Qnet_W_m2": b.Annual_Qnet_change_W_m2.abs().max(),
            "Maximum_abs_change_annual_Qnet_percent": b.Annual_Qnet_change_percent.abs().max(),
            "Maximum_abs_seasonal_change_W_m2": sb.Change_W_m2.abs().max(),
            "Most_affected_season": sb.loc[sb.Change_W_m2.abs().idxmax(), "Season"],
            "Source": b.Source.iloc[0],
        })
    summary = pd.DataFrame(summaries).sort_values("Maximum_abs_change_annual_Qnet_W_m2", ascending=False)
    checks = {
        "archived_baseline_mean_Qnet_W_m2": float(base["Q_net"].mean()),
        "archived_baseline_missing_count": int(base["Q_net"].isna().sum()),
        "archived_baseline_valid_cumulative_MJ_m2": float(base["Q_net"].iloc[1:].sum() * DT_SECONDS / 1e6),
        "continuous_baseline_cumulative_MJ_m2": float(base["Q_net"].interpolate(method="time", limit_area="inside").iloc[1:].sum() * DT_SECONDS / 1e6),
        "max_abs_component_reconstruction_error_W_m2": float((total() - base["Q_net"]).abs().max()),
    }
    return detail, seasonal, summary, checks


def mld_one(temps: np.ndarray, threshold: float, method: str, max_depth=2.25) -> float:
    valid = np.isfinite(temps)
    if valid.sum() < 2:
        return np.nan
    t, z = temps[valid], DEPTHS[valid]
    target = t[0] - threshold
    if np.nanmin(t) > target:
        return max_depth
    diff = t - target
    for i in range(len(z) - 1):
        if diff[i] == 0:
            return float(z[i])
        if diff[i] * diff[i + 1] <= 0:
            if method == "linear":
                if t[i + 1] == t[i]:
                    return float(z[i + 1])
                return float(z[i] + (target - t[i]) * (z[i + 1] - z[i]) / (t[i + 1] - t[i]))
            f = PchipInterpolator(z, t, extrapolate=False)
            return float(brentq(lambda zz: float(f(zz) - target), z[i], z[i + 1]))
    return max_depth


def stratification_sensitivity(raw_path: Path, archived_fig3: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    raw = pd.read_csv(raw_path, index_col=0, parse_dates=True, encoding="utf-8-sig")
    raw = raw.loc[:, ~raw.columns.astype(str).str.startswith("Unnamed:")].sort_index()
    temps = raw[TEMP_COLS].apply(pd.to_numeric, errors="coerce")
    # Sensitivity uses the same analysis-copy gap treatment already documented in the revision.
    temps_filled = temps.interpolate(method="time", limit_area="inside")
    series = {}
    for method in ["PCHIP", "linear"]:
        for threshold in [0.1, 0.2, 0.3]:
            key = f"MLD_{method}_{threshold:.1f}C"
            series[key] = temps_filled.apply(lambda r: mld_one(r.to_numpy(float), threshold, method.lower()), axis=1)
    mld = pd.DataFrame(series, index=raw.index)
    mld.index.name = "Date"
    mld["Season"] = season_of(mld.index)
    dt_base = pd.to_numeric(raw["25cm"], errors="coerce") - pd.to_numeric(raw["225cm"], errors="coerce")
    mld["Delta_T_surface_bottom_C"] = dt_base
    mld["Regime_proxy"] = pd.cut(
        dt_base,
        bins=[-np.inf, 0.2, 1.0, np.inf],
        labels=["near-isothermal", "weak/intermediate stratification", "strong stratification"],
    ).astype("object")

    summary_rows = []
    for method in ["PCHIP", "linear"]:
        for threshold in [0.1, 0.2, 0.3]:
            col = f"MLD_{method}_{threshold:.1f}C"
            s = mld[col]
            daily = s.resample("D").agg(["mean", "min", "max"])
            daily["amplitude"] = daily["max"] - daily["min"]
            summer_amp = daily[daily.index.month.isin([6, 7, 8])]["amplitude"]
            for season in ["Annual", "Spring", "Summer", "Autumn", "Winter"]:
                x = s if season == "Annual" else s[mld.Season.eq(season)]
                summary_rows.append({
                    "Method": method,
                    "Threshold_C": threshold,
                    "Period": season,
                    "Mean_MLD_m": x.mean(),
                    "Median_MLD_m": x.median(),
                    "Minimum_MLD_m": x.min(),
                    "Maximum_MLD_m": x.max(),
                    "Bottom_reached_fraction_percent": x.eq(2.25).mean() * 100,
                    "Summer_daily_amplitude_mean_m": summer_amp.mean() if season == "Summer" else np.nan,
                    "Summer_daily_amplitude_median_m": summer_amp.median() if season == "Summer" else np.nan,
                })
    summary = pd.DataFrame(summary_rows)

    # PCHIP-vs-linear MLD differences at each threshold.
    comp_rows = []
    for threshold in [0.1, 0.2, 0.3]:
        diff = mld[f"MLD_PCHIP_{threshold:.1f}C"] - mld[f"MLD_linear_{threshold:.1f}C"]
        for season in ["Annual", "Spring", "Summer", "Autumn", "Winter"]:
            x = diff if season == "Annual" else diff[mld.Season.eq(season)]
            comp_rows.append({
                "Threshold_C": threshold,
                "Period": season,
                "Mean_signed_difference_m": x.mean(),
                "Mean_absolute_difference_m": x.abs().mean(),
                "Maximum_absolute_difference_m": x.abs().max(),
                "Fraction_abs_difference_gt_0.05m_percent": x.abs().gt(0.05).mean() * 100,
            })
    method_compare = pd.DataFrame(comp_rows)

    profile_times = {
        "Winter mixed": "2024-12-26 06:00",
        "Spring persistent": "2024-05-28 05:20",
        "Summer afternoon": "2024-08-18 15:00",
        "Summer night": "2024-08-19 02:10",
        "Autumn decay": "2024-10-25 15:00",
    }
    zgrid = np.linspace(0.25, 2.25, 201)
    profile_rows = []
    for label, when in profile_times.items():
        t = temps_filled.loc[pd.Timestamp(when)].to_numpy(float)
        pchip = PchipInterpolator(DEPTHS, t)(zgrid)
        linear = np.interp(zgrid, DEPTHS, t)
        for z, tp, tl in zip(zgrid, pchip, linear):
            profile_rows.append({"Profile": label, "Date": when, "Depth_m": z, "PCHIP_C": tp, "Linear_C": tl, "Difference_C": tp - tl})
    profiles = pd.DataFrame(profile_rows)

    # Whole-year profile interpolation comparison on a common 0.01 m grid.
    matrix = temps_filled.to_numpy(float).T
    pchip_all = PchipInterpolator(DEPTHS, matrix, axis=0, extrapolate=False)(zgrid)
    linear_all = interp1d(DEPTHS, matrix, axis=0, kind="linear", bounds_error=True)(zgrid)
    abs_diff = np.abs(pchip_all - linear_all)
    seasons = season_of(raw.index)
    profile_summary_rows = []
    for season in ["Annual", "Spring", "Summer", "Autumn", "Winter"]:
        subset = abs_diff if season == "Annual" else abs_diff[:, seasons.eq(season).to_numpy()]
        profile_summary_rows.append({
            "Period": season,
            "Profile_temperature_MAE_C": float(np.nanmean(subset)),
            "Profile_temperature_RMSE_C": float(np.sqrt(np.nanmean(subset**2))),
            "Profile_temperature_max_abs_difference_C": float(np.nanmax(subset)),
            "Comparison_depth_grid_m": "0.25–2.25 at 0.01 m",
        })
    profile_summary = pd.DataFrame(profile_summary_rows)

    # Audit against the archived Figure 3 MLD values (raw_mld records only).
    archived = pd.read_csv(archived_fig3, parse_dates=["Date"], encoding="utf-8-sig")
    archived = archived[archived.data_type.eq("raw_mld")].set_index("Date")["MLD"]
    common = archived.index.intersection(mld.index)
    audit_diff = (archived.loc[common] - mld.loc[common, "MLD_PCHIP_0.2C"]).abs()
    checks = {
        "temperature_gap_count_rows": int(temps.isna().any(axis=1).sum()),
        "mld_valid_count": int(mld["MLD_PCHIP_0.2C"].notna().sum()),
        "archived_fig3_mld_common_count": int(len(common)),
        "archived_fig3_max_abs_difference_m": float(audit_diff.max()),
        "archived_fig3_mean_abs_difference_m": float(audit_diff.mean()),
    }
    return mld, summary, method_compare, profiles, profile_summary, checks


def make_figures(surface_summary, seasonal, mld, mld_summary, method_compare, profiles, output: Path):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": 0.8, "pdf.fonttype": 42})
    blue, red, gold = "#376B95", "#E76354", "#E5A735"

    short_labels = {
        "Inferred cloud-cover scale": "Cloud-cover scale",
        "Water emissivity": "Water emissivity",
        "Latent transfer coefficient CE": r"Latent coefficient $C_E$",
        "Water albedo": "Water albedo",
        "Clear-sky radiation scale": "Clear-sky radiation",
        "2-m to 10-m wind exponent": "Wind-height exponent",
        "Sensible transfer coefficient CH": r"Sensible coefficient $C_H$",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.4), gridspec_kw={"wspace": 0.72})
    s = surface_summary.sort_values("Maximum_abs_change_annual_Qnet_W_m2")
    axes[0].barh([short_labels.get(x, x) for x in s.Parameter], s.Maximum_abs_change_annual_Qnet_W_m2, color=blue)
    axes[0].set_xlabel(r"Maximum $|\Delta Q_{net}|$ (W m$^{-2}$)")
    axes[0].set_title("(a) Annual-mean sensitivity", loc="left", fontweight="bold")
    pivot = seasonal[seasonal.Parameter.ne("Baseline")].groupby(["Parameter", "Season"])["Change_W_m2"].apply(lambda x: x.loc[x.abs().idxmax()]).unstack()
    im = axes[1].imshow(pivot.abs(), aspect="auto", cmap="YlOrRd")
    axes[1].set_xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    axes[1].set_yticks(range(len(pivot.index)), [short_labels.get(x, x) for x in pivot.index])
    axes[1].set_title("(b) Maximum seasonal change", loc="left", fontweight="bold")
    cb = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04); cb.set_label(r"$|\Delta Q_{net}|$ (W m$^{-2}$)")
    fig.savefig(output / "Figure_S1_surface_flux_sensitivity.png", dpi=600, bbox_inches="tight")
    fig.savefig(output / "Figure_S1_surface_flux_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.2), sharex=True, gridspec_kw={"hspace": 0.18})
    daily = mld[[f"MLD_PCHIP_{x:.1f}C" for x in [0.1, 0.2, 0.3]]].resample("D").mean()
    for c, color in zip(daily, [gold, "black", red]):
        axes[0].plot(daily.index, daily[c], lw=0.8, color=color, label=c.split("_")[-1])
    axes[0].invert_yaxis(); axes[0].set_ylabel("Daily mean MLD (m)"); axes[0].legend(ncol=3, frameon=False)
    axes[0].set_title("(a) MLD threshold sensitivity (PCHIP)", loc="left", fontweight="bold")
    diff = (mld["MLD_PCHIP_0.2C"] - mld["MLD_linear_0.2C"]).resample("D").mean()
    axes[1].plot(diff.index, diff, lw=0.8, color=blue)
    axes[1].axhline(0, color="0.4", lw=0.6); axes[1].set_ylabel("PCHIP − linear (m)")
    axes[1].set_title("(b) Interpolation-method difference at 0.2 °C", loc="left", fontweight="bold")
    fig.savefig(output / "Figure_S2_MLD_sensitivity.png", dpi=600, bbox_inches="tight")
    fig.savefig(output / "Figure_S2_MLD_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)

    labels = profiles.Profile.unique()
    fig, axes = plt.subplots(1, len(labels), figsize=(10.2, 3.3), sharey=True)
    for ax, label in zip(axes, labels):
        p = profiles[profiles.Profile.eq(label)]
        ax.plot(p.PCHIP_C, p.Depth_m, color=red, label="PCHIP")
        ax.plot(p.Linear_C, p.Depth_m, color=blue, ls="--", label="Linear")
        ax.scatter(p.loc[p.Depth_m.isin(DEPTHS), "PCHIP_C"], DEPTHS, s=9, color="black", zorder=3)
        ax.invert_yaxis(); ax.set_title(label, fontsize=8); ax.set_xlabel("Temperature (°C)")
    axes[0].set_ylabel("Depth (m)"); axes[-1].legend(frameon=False, fontsize=8)
    fig.savefig(output / "Figure_S3_profile_interpolation_comparison.png", dpi=600, bbox_inches="tight")
    fig.savefig(output / "Figure_S3_profile_interpolation_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", help="Repository root; defaults to the detected public-package root")
    parser.add_argument("--output", help="Output directory; defaults to results/reproduced/sensitivity_oat")
    args = parser.parse_args()
    package = locate_package(args.package)
    output = (
        Path(args.output).resolve()
        if args.output
        else package / "results" / "reproduced" / "sensitivity_oat"
    )
    output.mkdir(parents=True, exist_ok=True)
    raw = package / "data" / "raw_or_input" / "full_year" / "meteorology_water_temperature_2024_10min.csv"
    f6 = package / "results" / "reproduced" / "Figure6" / "Figure6_calculated_data_revised.csv"
    f3 = package / "results" / "reproduced" / "Figure3" / "Figure3.csv"
    missing_inputs = [path for path in (raw, f6, f3) if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError(
            "Required input(s) not found. Place the private annual input as documented in README.md "
            "and run Figure3.py and Figure6.py first: " + ", ".join(str(path) for path in missing_inputs)
        )

    surface, seasonal, table_s1, surface_checks = surface_oat(raw, f6)
    mld, mld_summary, method_compare, profiles, profile_summary, mld_checks = stratification_sensitivity(raw, f3)
    spring = mld_summary[(mld_summary.Method == "PCHIP") & (mld_summary.Period == "Spring")].set_index("Threshold_C")
    summer = mld_summary[(mld_summary.Method == "PCHIP") & (mld_summary.Period == "Summer")].set_index("Threshold_C")
    comp02 = method_compare[(method_compare.Threshold_C == 0.2) & (method_compare.Period == "Annual")].iloc[0]
    prof_annual = profile_summary[profile_summary.Period == "Annual"].iloc[0]
    table_s2 = pd.DataFrame([
        {
            "Parameter": "MLD temperature threshold", "Baseline": "0.2 °C", "Alternative_cases": "0.1 and 0.3 °C",
            "Quantitative_effect": f"Spring mean MLD {spring.loc[0.1,'Mean_MLD_m']:.3f}/{spring.loc[0.2,'Mean_MLD_m']:.3f}/{spring.loc[0.3,'Mean_MLD_m']:.3f} m; summer median daily amplitude {summer.loc[0.1,'Summer_daily_amplitude_median_m']:.3f}/{summer.loc[0.2,'Summer_daily_amplitude_median_m']:.3f}/{summer.loc[0.3,'Summer_daily_amplitude_median_m']:.3f} m",
            "Impact_on_conclusions": "MLD magnitude shifts systematically, but spring remains shallow/persistent and summer retains ~1 m diel oscillation; seasonal regime interpretation is robust.",
            "Impact_scope": "MLD only; St and N² remain discrete-observation calculations",
        },
        {
            "Parameter": "Vertical interpolation", "Baseline": "PCHIP", "Alternative_cases": "linear",
            "Quantitative_effect": f"Annual profile-temperature MAE {prof_annual.Profile_temperature_MAE_C:.3f} °C; annual MLD MAE {comp02.Mean_absolute_difference_m:.3f} m, maximum {comp02.Maximum_absolute_difference_m:.3f} m at 0.2 °C threshold",
            "Impact_on_conclusions": "Small shifts in exact crossing depth do not alter the seasonal evolution; representative profiles preserve the same ordering and regime interpretation.",
            "Impact_scope": "profile visualization and MLD only; St and N² excluded",
        },
    ])

    surface.to_csv(output / "surface_flux_OAT_detail.csv", index=False, encoding="utf-8-sig")
    seasonal.to_csv(output / "surface_flux_OAT_seasonal.csv", index=False, encoding="utf-8-sig")
    table_s1.to_csv(output / "Table_S1_heat_flux_parameter_sensitivity.csv", index=False, encoding="utf-8-sig")
    mld.reset_index().to_csv(output / "MLD_all_scenarios_10min.csv", index=False, encoding="utf-8-sig")
    mld.resample("D").mean(numeric_only=True).reset_index().to_csv(output / "MLD_all_scenarios_daily.csv", index=False, encoding="utf-8-sig")
    mld_summary.to_csv(output / "MLD_threshold_summary.csv", index=False, encoding="utf-8-sig")
    method_compare.to_csv(output / "interpolation_MLD_comparison.csv", index=False, encoding="utf-8-sig")
    profiles.to_csv(output / "representative_profile_interpolation.csv", index=False, encoding="utf-8-sig")
    profile_summary.to_csv(output / "profile_interpolation_full_year_summary.csv", index=False, encoding="utf-8-sig")
    table_s2.to_csv(output / "Table_S2_stratification_metric_sensitivity.csv", index=False, encoding="utf-8-sig")
    checks = {
        "inputs": {"raw": str(raw), "raw_sha256": sha256(raw), "figure6": str(f6), "figure6_sha256": sha256(f6), "figure3": str(f3), "figure3_sha256": sha256(f3)},
        "surface": surface_checks,
        "mld": mld_checks,
        "rules": {"N2_and_St_interpolation_sensitivity": "excluded", "OAT": True, "original_files_modified": False},
    }
    (output / "quality_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    make_figures(table_s1, seasonal, mld, mld_summary, method_compare, profiles, output)
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
