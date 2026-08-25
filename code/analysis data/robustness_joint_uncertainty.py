from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import qmc


SEASONS = ["Spring", "Summer", "Autumn", "Winter"]
COMPONENTS = ["Qsw", "Qlw", "Qe", "Qh"]
COMPONENT_COLS = {c: f"Mean_{c}_W_m2" for c in COMPONENTS}
DISPLAY = {"Qsw": "Qsw", "Qlw": "Qlw", "Qe": "Qe", "Qh": "Qh"}
BASELINE_STORAGE_TENDENCY_W_M2 = -0.15032609746570147
BASELINE_CONTINUOUS_QNET_W_M2 = 52.567178637692294
BASELINE_CONTINUOUS_RESIDUAL_W_M2 = 52.717504735158
SEED = 20260817
N_SAMPLES = 100_000

BOUNDS = {
    "albedo": (0.05, 0.10),
    "emissivity": (0.96, 0.99),
    "CE": (1.3e-3 * 0.8, 1.3e-3 * 1.2),
    "CH": (1.3e-3 * 0.8, 1.3e-3 * 1.2),
    "rso_scale": (0.9, 1.1),
    "cloud_scale": (0.8, 1.2),
    "wind_exponent": (1 / 8, 1 / 6),
}
BASELINE = {
    "albedo": 0.07,
    "emissivity": 0.97,
    "CE": 1.3e-3,
    "CH": 1.3e-3,
    "rso_scale": 1.0,
    "cloud_scale": 1.0,
    "wind_exponent": 1 / 7,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def import_stage1(script: Path):
    spec = importlib.util.spec_from_file_location("stage1_sensitivity", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import stage-1 script: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def season_of(index: pd.DatetimeIndex) -> np.ndarray:
    month = index.month
    return np.select(
        [np.isin(month, [3, 4, 5]), np.isin(month, [6, 7, 8]), np.isin(month, [9, 10, 11])],
        ["Spring", "Summer", "Autumn"],
        default="Winter",
    )


def dominant_input(row: pd.Series) -> str:
    values = {c: float(row[COMPONENT_COLS[c]]) for c in COMPONENTS}
    positive = {k: v for k, v in values.items() if v > 0}
    return max(positive, key=positive.get) if positive else "None"


def dominant_cooling(row: pd.Series) -> str:
    values = {c: float(row[COMPONENT_COLS[c]]) for c in COMPONENTS}
    negative = {k: abs(v) for k, v in values.items() if v < 0}
    return max(negative, key=negative.get) if negative else "None"


def cooling_ranking(row: pd.Series) -> str:
    values = {c: float(row[COMPONENT_COLS[c]]) for c in COMPONENTS}
    negative = sorted(((abs(v), k) for k, v in values.items() if v < 0), reverse=True)
    return " > ".join(k for _, k in negative)


def oat_robustness(stage1_results: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seasonal = pd.read_csv(stage1_results / "surface_flux_OAT_seasonal.csv")
    required = {"Scenario_ID", "Parameter", "Case", "Season", *COMPONENT_COLS.values()}
    missing = required.difference(seasonal.columns)
    if missing:
        raise ValueError(f"Missing stage-1 seasonal columns: {sorted(missing)}")

    baseline = seasonal[seasonal["Scenario_ID"].eq("BASE")].set_index("Season")
    rows = []
    for scenario_id, group in seasonal.groupby("Scenario_ID", sort=False):
        by_season = group.set_index("Season")
        def at(season: str, kind: str) -> str:
            row = by_season.loc[season]
            return dominant_input(row) if kind == "input" else dominant_cooling(row)
        spring_in = at("Spring", "input")
        summer_in = at("Summer", "input")
        summer_cool = at("Summer", "cool")
        winter_cool = at("Winter", "cool")
        base_tuple = (
            dominant_input(baseline.loc["Spring"]),
            dominant_input(baseline.loc["Summer"]),
            dominant_cooling(baseline.loc["Summer"]),
            dominant_cooling(baseline.loc["Winter"]),
        )
        current_tuple = (spring_in, summer_in, summer_cool, winter_cool)
        meta = group.iloc[0]
        rows.append({
            "Scenario_ID": scenario_id,
            "Parameter": meta["Parameter"],
            "Case": meta["Case"],
            "Spring_dominant_heat_input": spring_in,
            "Summer_dominant_heat_input": summer_in,
            "Summer_dominant_cooling_term": summer_cool,
            "Winter_dominant_cooling_term": winter_cool,
            "Autumn_cooling_ranking": cooling_ranking(by_season.loc["Autumn"]),
            "Baseline_core_ranking_changes": current_tuple != base_tuple,
        })
    dominance = pd.DataFrame(rows)
    dominance["Core_ranking_preserved_flag"] = (~dominance["Baseline_core_ranking_changes"]).astype(int)

    contribution_rows = []
    for _, row in seasonal.iterrows():
        vals = {c: float(row[COMPONENT_COLS[c]]) for c in COMPONENTS}
        absolute_total = sum(abs(v) for v in vals.values())
        positive_total = sum(v for v in vals.values() if v > 0)
        cooling_total = sum(abs(v) for v in vals.values() if v < 0)
        for comp, value in vals.items():
            role_total = positive_total if value >= 0 else cooling_total
            contribution_rows.append({
                "Scenario_ID": row["Scenario_ID"],
                "Parameter": row["Parameter"],
                "Case": row["Case"],
                "Season": row["Season"],
                "Component": comp,
                "Mean_flux_W_m2": value,
                "Signed_absolute_budget_share_percent": 100 * value / absolute_total,
                "Role_specific_share_percent": 100 * abs(value) / role_total if role_total else np.nan,
                "Role": "heat input" if value >= 0 else "cooling",
            })
    contributions = pd.DataFrame(contribution_rows)
    ranges = (
        contributions.groupby(["Season", "Component"], sort=False)
        .agg(
            Mean_flux_min_W_m2=("Mean_flux_W_m2", "min"),
            Mean_flux_max_W_m2=("Mean_flux_W_m2", "max"),
            Signed_share_min_percent=("Signed_absolute_budget_share_percent", "min"),
            Signed_share_max_percent=("Signed_absolute_budget_share_percent", "max"),
            Role_share_min_percent=("Role_specific_share_percent", "min"),
            Role_share_max_percent=("Role_specific_share_percent", "max"),
        )
        .reset_index()
    )

    robustness = pd.DataFrame([
        {
            "Scientific_statement": "Spring and summer shortwave-dominated heat input",
            "Criterion": "Qsw is the largest positive component in every OAT scenario in both seasons",
            "Scenarios_satisfying": int(((dominance.Spring_dominant_heat_input == "Qsw") & (dominance.Summer_dominant_heat_input == "Qsw")).sum()),
            "Scenarios_total": len(dominance),
            "Classification": "robust" if ((dominance.Spring_dominant_heat_input == "Qsw") & (dominance.Summer_dominant_heat_input == "Qsw")).all() else "conditionally robust",
        },
        {
            "Scientific_statement": "Summer latent to winter longwave dominant-cooling shift",
            "Criterion": "Qe is dominant cooling in summer and Qlw is dominant cooling in winter in every OAT scenario",
            "Scenarios_satisfying": int(((dominance.Summer_dominant_cooling_term == "Qe") & (dominance.Winter_dominant_cooling_term == "Qlw")).sum()),
            "Scenarios_total": len(dominance),
            "Classification": "robust" if ((dominance.Summer_dominant_cooling_term == "Qe") & (dominance.Winter_dominant_cooling_term == "Qlw")).all() else "conditionally robust",
        },
    ])
    return dominance, contributions, ranges, robustness


def locate_inputs(package: Path) -> tuple[Path, Path]:
    raw_path = package / "data" / "raw_or_input" / "full_year" / "meteorology_water_temperature_2024_10min.csv"
    arc_path = package / "results" / "reproduced" / "Figure6" / "Figure6_calculated_data_revised.csv"
    missing = [path for path in (raw_path, arc_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required input(s) not found. Place the private annual input as documented in README.md "
            "and run Figure6.py first: " + ", ".join(str(path) for path in missing)
        )
    return raw_path, arc_path


def load_joint_inputs(package: Path, stage1):
    raw_path, arc_path = locate_inputs(package)
    raw = pd.read_csv(raw_path, index_col=0, parse_dates=True, encoding="utf-8-sig").sort_index()
    raw = raw.loc[:, ~raw.columns.astype(str).str.startswith("Unnamed:")]
    arc = pd.read_csv(arc_path, index_col=0, parse_dates=True, encoding="utf-8-sig").sort_index()
    if not raw.index.equals(arc.index):
        raise ValueError("Raw and archived Figure6 timestamps do not match")
    rad_col = next(c for c in raw.columns if "辐射" in c and "累计" not in c)
    rh_col = next(c for c in raw.columns if "湿度" in c)
    Rsw = pd.to_numeric(raw[rad_col], errors="coerce").clip(lower=0)
    if "Ta_C" in arc.columns:
        Ta = pd.to_numeric(arc["Ta_C"], errors="coerce")
    else:
        ta_col = next(c for c in raw.columns if "大气温度" in c)
        Ta = pd.to_numeric(raw[ta_col], errors="coerce")
    Ts = pd.to_numeric(raw["25cm"], errors="coerce")
    RH = pd.to_numeric(raw[rh_col], errors="coerce") / 100
    ea = stage1.calc_vapor_pressure(Ta) * RH
    tk = Ta + 273.15
    outgoing_unit = stage1.SIGMA * (Ts + 273.15) ** 4
    base = {k: pd.to_numeric(arc[k], errors="coerce") for k in ["Q_sw", "Q_lw", "Q_e", "Q_h", "Q_net"]}
    return raw_path, arc_path, raw.index, Rsw, tk, ea, outgoing_unit, base


def mean_for_mask(series: pd.Series, mask: np.ndarray) -> float:
    return float(series.loc[mask].mean())


def interpolate_over_invalid(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Linearly fill timestamps excluded by the archived Qnet validity mask."""
    arr = np.asarray(values, dtype=float)
    one_dimensional = arr.ndim == 1
    if one_dimensional:
        arr = arr[:, None]
    out = arr.copy()
    x = np.arange(len(out))
    for j in range(out.shape[1]):
        good = valid & np.isfinite(out[:, j])
        out[~valid, j] = np.interp(x[~valid], x[good], out[good, j])
    return out[:, 0] if one_dimensional else out


def build_longwave_response(index, Rsw, tk, ea, outgoing_unit, base, stage1):
    rso_grid = np.linspace(*BOUNDS["rso_scale"], 81)
    cloud_grid = np.linspace(*BOUNDS["cloud_scale"], 81)
    valid = base["Q_net"].notna().to_numpy()
    season = season_of(index)
    masks = {s: valid & (season == s) for s in SEASONS}
    masks["AnnualValid"] = valid
    metrics = ["AnnualValid", *SEASONS, "AnnualContinuous"]
    atm_grids = {m: np.empty((len(rso_grid), len(cloud_grid))) for m in metrics}
    tk_arr = tk.to_numpy(float)
    ea_arr = ea.to_numpy(float)
    for i, rso in enumerate(rso_grid):
        c = stage1.infer_cloud(index, Rsw, float(rso)).to_numpy(float)
        c2 = np.clip(c[:, None] * cloud_grid[None, :], 0, 1)
        emiss = np.clip(stage1.ATM_CLEAR_COEFF * (ea_arr[:, None] / tk_arr[:, None]) ** (1 / 7) * (1 + stage1.CLOUD_COEFF * c2**2), 0, 0.98)
        incoming = emiss * stage1.SIGMA * tk_arr[:, None] ** 4
        for metric in ["AnnualValid", *SEASONS]:
            mask = masks[metric]
            atm_grids[metric][i, :] = np.nanmean(incoming[mask, :], axis=0)
        incoming_continuous = interpolate_over_invalid(incoming, valid)
        atm_grids["AnnualContinuous"][i, :] = np.mean(incoming_continuous[1:, :], axis=0)
    interpolators = {
        m: RegularGridInterpolator((rso_grid, cloud_grid), grid, bounds_error=True)
        for m, grid in atm_grids.items()
    }
    outgoing = {}
    for metric in ["AnnualValid", *SEASONS]:
        outgoing[metric] = float(outgoing_unit.loc[masks[metric]].mean())
    outgoing_continuous = interpolate_over_invalid(outgoing_unit.to_numpy(float), valid)
    outgoing["AnnualContinuous"] = float(np.mean(outgoing_continuous[1:]))
    return interpolators, outgoing, masks, rso_grid, cloud_grid


def component_base_means(Rsw, base, masks):
    means = {}
    for metric in ["AnnualValid", *SEASONS]:
        mask = masks[metric]
        means[metric] = {
            "Rsw": float(Rsw.loc[mask].mean()),
            "Qe_base": float(base["Q_e"].loc[mask].mean()),
            "Qh_base": float(base["Q_h"].loc[mask].mean()),
        }
    valid = masks["AnnualValid"]
    rsw_continuous = interpolate_over_invalid(Rsw.to_numpy(float), valid)
    means["AnnualContinuous"] = {
        "Rsw": float(np.mean(rsw_continuous[1:])),
        "Qe_base": float(base["Q_e"].interpolate(method="time", limit_area="inside").iloc[1:].mean()),
        "Qh_base": float(base["Q_h"].interpolate(method="time", limit_area="inside").iloc[1:].mean()),
    }
    return means


def validate_longwave_response(index, Rsw, tk, ea, valid, lw_interp, stage1, n=12) -> float:
    rng = np.random.default_rng(SEED + 2)
    max_error = 0.0
    tk_arr = tk.to_numpy(float)
    ea_arr = ea.to_numpy(float)
    season = season_of(index)
    for rso, cloud in zip(rng.uniform(*BOUNDS["rso_scale"], n), rng.uniform(*BOUNDS["cloud_scale"], n)):
        c = stage1.infer_cloud(index, Rsw, float(rso)).to_numpy(float)
        c = np.clip(c * cloud, 0, 1)
        emiss = np.clip(stage1.ATM_CLEAR_COEFF * (ea_arr / tk_arr) ** (1 / 7) * (1 + stage1.CLOUD_COEFF * c**2), 0, 0.98)
        incoming = emiss * stage1.SIGMA * tk_arr**4
        direct = {"AnnualValid": float(np.mean(incoming[valid]))}
        for s in SEASONS:
            direct[s] = float(np.mean(incoming[valid & (season == s)]))
        direct["AnnualContinuous"] = float(np.mean(interpolate_over_invalid(incoming, valid)[1:]))
        point = np.array([[rso, cloud]])
        for metric, expected in direct.items():
            max_error = max(max_error, abs(float(lw_interp[metric](point)[0]) - expected))
    return max_error


def evaluate_samples(samples: pd.DataFrame, metric: str, lw_interp, outgoing, base_means) -> pd.DataFrame:
    points = samples[["rso_scale", "cloud_scale"]].to_numpy(float)
    qsw = (1 - samples["albedo"].to_numpy()) * base_means[metric]["Rsw"]
    qlw = lw_interp[metric](points) - samples["emissivity"].to_numpy() * outgoing[metric]
    wind_factor = 5 ** (samples["wind_exponent"].to_numpy() - BASELINE["wind_exponent"])
    qe = base_means[metric]["Qe_base"] * samples["CE"].to_numpy() / BASELINE["CE"] * wind_factor
    qh = base_means[metric]["Qh_base"] * samples["CH"].to_numpy() / BASELINE["CH"] * wind_factor
    qnet = qsw + qlw + qe + qh
    return pd.DataFrame({"Qsw": qsw, "Qlw": qlw, "Qe": qe, "Qh": qh, "Qnet": qnet})


def lhs_samples(n: int, linked: bool = False) -> pd.DataFrame:
    names = list(BOUNDS)
    if not linked:
        unit = qmc.LatinHypercube(d=len(names), seed=SEED).random(n)
        scaled = qmc.scale(unit, [BOUNDS[n][0] for n in names], [BOUNDS[n][1] for n in names])
        return pd.DataFrame(scaled, columns=names)
    unit = qmc.LatinHypercube(d=5, seed=SEED + 1).random(n)
    out = pd.DataFrame(index=np.arange(n))
    for j, name in enumerate(["albedo", "emissivity", "wind_exponent"]):
        lo, hi = BOUNDS[name]
        out[name] = lo + unit[:, j] * (hi - lo)
    transfer_u = unit[:, 3]
    cloud_u = unit[:, 4]
    for name in ["CE", "CH"]:
        lo, hi = BOUNDS[name]
        out[name] = lo + transfer_u * (hi - lo)
    for name in ["rso_scale", "cloud_scale"]:
        lo, hi = BOUNDS[name]
        out[name] = lo + cloud_u * (hi - lo)
    return out[list(BOUNDS)]


def corner_samples() -> pd.DataFrame:
    names = list(BOUNDS)
    values = [[BOUNDS[n][0], BOUNDS[n][1]] for n in names]
    return pd.DataFrame(list(itertools.product(*values)), columns=names)


def quantile_summary(values: np.ndarray, ensemble: str, metric: str) -> dict:
    q = np.quantile(values, [0.025, 0.5, 0.975])
    return {
        "Ensemble": ensemble,
        "Metric": metric,
        "Mean": float(np.mean(values)),
        "Std": float(np.std(values, ddof=1)),
        "P2_5": float(q[0]),
        "P50": float(q[1]),
        "P97_5": float(q[2]),
        "Minimum": float(np.min(values)),
        "Maximum": float(np.max(values)),
    }


def joint_uncertainty(package: Path, stage1_script: Path, output: Path):
    stage1 = import_stage1(stage1_script)
    raw_path, arc_path, index, Rsw, tk, ea, outgoing_unit, base = load_joint_inputs(package, stage1)
    lw_interp, outgoing, masks, rso_grid, cloud_grid = build_longwave_response(index, Rsw, tk, ea, outgoing_unit, base, stage1)
    base_means = component_base_means(Rsw, base, masks)
    baseline_eval = evaluate_samples(pd.DataFrame([BASELINE]), "AnnualContinuous", lw_interp, outgoing, base_means).iloc[0]
    baseline_qnet_error = abs(float(baseline_eval["Qnet"]) - BASELINE_CONTINUOUS_QNET_W_M2)
    response_validation_error = validate_longwave_response(
        index, Rsw, tk, ea, masks["AnnualValid"], lw_interp, stage1
    )

    ensembles = {
        "independent_uniform_LHS": lhs_samples(N_SAMPLES, linked=False),
        "linked_rank_stress": lhs_samples(N_SAMPLES, linked=True),
        "parameter_box_corners": corner_samples(),
    }
    summaries, dominance_rows, corner_long = [], [], []
    annual_export = ensembles["independent_uniform_LHS"].copy()

    for ensemble_name, samples in ensembles.items():
        results = {}
        for metric in ["AnnualValid", *SEASONS, "AnnualContinuous"]:
            results[metric] = evaluate_samples(samples, metric, lw_interp, outgoing, base_means)
        annual_residual = results["AnnualContinuous"]["Qnet"].to_numpy() - BASELINE_STORAGE_TENDENCY_W_M2
        for metric_name, values in [
            ("Annual mean Qnet (valid timestamps), W m-2", results["AnnualValid"]["Qnet"].to_numpy()),
            ("Annual continuous Qnet, W m-2", results["AnnualContinuous"]["Qnet"].to_numpy()),
            ("Annual continuous residual, W m-2", annual_residual),
        ]:
            row = quantile_summary(values, ensemble_name, metric_name)
            row["Probability_gt_zero_percent"] = float(100 * np.mean(values > 0))
            row["Probability_abs_le_5_W_m2_percent"] = float(100 * np.mean(np.abs(values) <= 5))
            summaries.append(row)

        spring = results["Spring"]
        summer = results["Summer"]
        winter = results["Winter"]
        spring_sw = spring["Qsw"].to_numpy() >= spring[["Qlw", "Qe", "Qh"]].clip(lower=0).max(axis=1).to_numpy()
        summer_sw = summer["Qsw"].to_numpy() >= summer[["Qlw", "Qe", "Qh"]].clip(lower=0).max(axis=1).to_numpy()
        summer_latent = np.abs(summer["Qe"].to_numpy()) >= np.maximum(np.abs(summer["Qlw"].to_numpy()), np.abs(summer["Qh"].to_numpy()))
        winter_longwave = np.abs(winter["Qlw"].to_numpy()) >= np.maximum(np.abs(winter["Qe"].to_numpy()), np.abs(winter["Qh"].to_numpy()))
        for label, condition in [
            ("Spring Qsw dominant heat input", spring_sw),
            ("Summer Qsw dominant heat input", summer_sw),
            ("Summer Qe dominant cooling", summer_latent),
            ("Winter Qlw dominant cooling", winter_longwave),
            ("All four core ranking criteria", spring_sw & summer_sw & summer_latent & winter_longwave),
        ]:
            dominance_rows.append({
                "Ensemble": ensemble_name,
                "Criterion": label,
                "Satisfied_count": int(condition.sum()),
                "Total_count": len(condition),
                "Satisfied_percent": float(100 * condition.mean()),
            })

        if ensemble_name == "independent_uniform_LHS":
            annual_export["Annual_Qnet_valid_W_m2"] = results["AnnualValid"]["Qnet"].to_numpy()
            annual_export["Annual_Qnet_continuous_W_m2"] = results["AnnualContinuous"]["Qnet"].to_numpy()
            annual_export["Annual_residual_continuous_W_m2"] = annual_residual
        if ensemble_name == "parameter_box_corners":
            for metric in [*SEASONS, "AnnualContinuous"]:
                tmp = pd.concat([samples.reset_index(drop=True), results[metric]], axis=1)
                tmp.insert(0, "Metric", metric)
                if metric == "AnnualContinuous":
                    tmp["Residual_W_m2"] = tmp["Qnet"] - BASELINE_STORAGE_TENDENCY_W_M2
                corner_long.append(tmp)

    annual_export.to_csv(output / "joint_uncertainty_samples_annual.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(summaries)
    dominance = pd.DataFrame(dominance_rows)
    corners = pd.concat(corner_long, ignore_index=True)
    summary.to_csv(output / "joint_uncertainty_summary.csv", index=False, encoding="utf-8-sig")
    dominance.to_csv(output / "joint_dominance_probabilities.csv", index=False, encoding="utf-8-sig")
    corners.to_csv(output / "joint_corner_envelope.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"Parameter": n, "Baseline": BASELINE[n], "Lower": BOUNDS[n][0], "Upper": BOUNDS[n][1], "Distribution": "Uniform within existing stage-1 range", "Correlation_main": "Independent", "Correlation_stress": "CE=CH linked ranks; Rso scale=cloud scale linked ranks"}
        for n in BOUNDS
    ]).to_csv(output / "joint_parameter_assumptions.csv", index=False, encoding="utf-8-sig")
    checks = {
        "raw_path": str(raw_path), "raw_sha256": sha256(raw_path),
        "figure6_path": str(arc_path), "figure6_sha256": sha256(arc_path),
        "stage1_script": str(stage1_script), "stage1_script_sha256": sha256(stage1_script),
        "n_lhs_samples": N_SAMPLES, "seed": SEED,
        "response_grid_shape": [len(rso_grid), len(cloud_grid)],
        "baseline_continuous_qnet_reference_W_m2": BASELINE_CONTINUOUS_QNET_W_M2,
        "baseline_continuous_residual_reference_W_m2": BASELINE_CONTINUOUS_RESIDUAL_W_M2,
        "baseline_joint_reconstruction_Qnet_W_m2": float(baseline_eval["Qnet"]),
        "baseline_joint_reconstruction_abs_error_W_m2": baseline_qnet_error,
        "longwave_response_surface_validation_max_abs_error_W_m2": response_validation_error,
    }
    return summary, dominance, corners, annual_export, checks


def mld_robustness(stage1_results: Path) -> pd.DataFrame:
    mld = pd.read_csv(stage1_results / "MLD_threshold_summary.csv")
    cmp = pd.read_csv(stage1_results / "interpolation_MLD_comparison.csv")
    rows = []
    for method in ["PCHIP", "linear"]:
        for threshold in [0.1, 0.2, 0.3]:
            x = mld[(mld.Method == method) & np.isclose(mld.Threshold_C, threshold)].set_index("Period")
            ordering = x.loc[["Spring", "Summer", "Autumn", "Winter"], "Mean_MLD_m"].to_dict()
            regime_ok = ordering["Spring"] < ordering["Summer"] < ordering["Autumn"] and ordering["Spring"] < ordering["Winter"]
            rows.append({
                "Method": method,
                "Threshold_C": threshold,
                "Spring_mean_MLD_m": ordering["Spring"],
                "Summer_mean_MLD_m": ordering["Summer"],
                "Autumn_mean_MLD_m": ordering["Autumn"],
                "Winter_mean_MLD_m": ordering["Winter"],
                "Summer_daily_amplitude_median_m": x.loc["Summer", "Summer_daily_amplitude_median_m"],
                "Spring_bottom_reached_fraction_percent": x.loc["Spring", "Bottom_reached_fraction_percent"],
                "Seasonal_regime_criterion_preserved": bool(regime_ok),
            })
    out = pd.DataFrame(rows)
    out["Baseline_0.2C_PCHIP_annual_MLD_MAE_linear_m"] = float(cmp[(np.isclose(cmp.Threshold_C, 0.2)) & cmp.Period.eq("Annual")]["Mean_absolute_difference_m"].iloc[0])
    out["Baseline_0.2C_PCHIP_annual_MLD_max_difference_linear_m"] = float(cmp[(np.isclose(cmp.Threshold_C, 0.2)) & cmp.Period.eq("Annual")]["Maximum_absolute_difference_m"].iloc[0])
    out["Regime_preserved_flag"] = out["Seasonal_regime_criterion_preserved"].astype(int)
    return out


def make_figures(output: Path, ranges: pd.DataFrame, annual_samples: pd.DataFrame, summary: pd.DataFrame):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": 0.8})
    colors = {"Qsw": "#D89C20", "Qlw": "#496A9A", "Qe": "#2A9D8F", "Qh": "#8C6D9B"}
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.6), sharex=True)
    for ax, season in zip(axes.ravel(), SEASONS):
        d = ranges[ranges.Season.eq(season)].set_index("Component").loc[COMPONENTS]
        y = np.arange(len(COMPONENTS))
        lo, hi = d.Mean_flux_min_W_m2.to_numpy(), d.Mean_flux_max_W_m2.to_numpy()
        ax.hlines(y, lo, hi, color=[colors[c] for c in COMPONENTS], linewidth=5, alpha=0.8)
        ax.scatter(lo, y, color=[colors[c] for c in COMPONENTS], s=18)
        ax.scatter(hi, y, color=[colors[c] for c in COMPONENTS], s=18)
        ax.axvline(0, color="0.35", lw=0.8)
        ax.set_yticks(y, COMPONENTS)
        ax.set_title(season)
        ax.grid(axis="x", color="0.9", lw=0.6)
    axes[1, 0].set_xlabel("Seasonal mean heat flux (W m$^{-2}$)")
    axes[1, 1].set_xlabel("Seasonal mean heat flux (W m$^{-2}$)")
    fig.suptitle("OAT scenario envelopes preserve the core seasonal component rankings", y=0.995, fontsize=11)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(output / f"Figure_U1_OAT_component_robustness.{ext}", dpi=400 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7))
    for ax, col, baseline, title in [
        (axes[0], "Annual_Qnet_continuous_W_m2", BASELINE_CONTINUOUS_QNET_W_M2, "Annual mean surface Qnet"),
        (axes[1], "Annual_residual_continuous_W_m2", BASELINE_CONTINUOUS_RESIDUAL_W_M2, "Annual mean energy residual"),
    ]:
        vals = annual_samples[col].to_numpy()
        ax.hist(vals, bins=70, density=True, color="#5B8DB8", alpha=0.85, edgecolor="none")
        ax.axvline(baseline, color="#B23A48", lw=1.4, label="baseline")
        ax.axvline(0, color="0.2", lw=0.8, ls="--")
        ax.set_title(title)
        ax.set_xlabel("W m$^{-2}$")
        ax.set_ylabel("Density")
        ax.legend(frameon=False)
    fig.suptitle("Joint parameter scenario propagation (independent uniform LHS)", y=0.99, fontsize=11)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(output / f"Figure_U2_joint_uncertainty.{ext}", dpi=400 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", help="Stage-1 OAT result directory")
    parser.add_argument("--stage1-script", help="Stage-1 OAT Python script")
    parser.add_argument("--package", help="Repository root; defaults to the detected public-package root")
    parser.add_argument("--output", help="Output directory; defaults to results/reproduced/joint_uncertainty")
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    package = Path(args.package).resolve() if args.package else repository_root
    stage1_results = (
        Path(args.stage1).resolve()
        if args.stage1
        else package / "results" / "reproduced" / "sensitivity_oat"
    )
    stage1_script = (
        Path(args.stage1_script).resolve()
        if args.stage1_script
        else package / "code" / "analysis" / "uncertainty_sensitivity_analysis.py"
    )
    output = (
        Path(args.output).resolve()
        if args.output
        else package / "results" / "reproduced" / "joint_uncertainty"
    )
    output.mkdir(parents=True, exist_ok=True)

    dominance, contributions, ranges, robust = oat_robustness(stage1_results)
    dominance.to_csv(output / "OAT_flux_ranking_robustness.csv", index=False, encoding="utf-8-sig")
    contributions.to_csv(output / "OAT_seasonal_component_contributions.csv", index=False, encoding="utf-8-sig")
    ranges.to_csv(output / "OAT_seasonal_contribution_ranges.csv", index=False, encoding="utf-8-sig")
    robust.to_csv(output / "core_conclusion_robustness.csv", index=False, encoding="utf-8-sig")

    mld = mld_robustness(stage1_results)
    mld.to_csv(output / "MLD_regime_robustness.csv", index=False, encoding="utf-8-sig")

    summary, joint_dom, corners, annual_samples, checks = joint_uncertainty(
        package, stage1_script, output
    )
    make_figures(output, ranges, annual_samples, summary)

    checks.update({
        "stage1_seasonal_sha256": sha256(stage1_results / "surface_flux_OAT_seasonal.csv"),
        "stage1_mld_summary_sha256": sha256(stage1_results / "MLD_threshold_summary.csv"),
        "oat_scenarios": int(len(dominance)),
        "all_oat_core_rankings_preserved": bool((~dominance.Baseline_core_ranking_changes).all()),
        "all_mld_regime_criteria_preserved": bool(mld.Seasonal_regime_criterion_preserved.all()),
        "original_files_modified": False,
    })
    (output / "quality_checks_stage2.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")



if __name__ == "__main__":
    main()
