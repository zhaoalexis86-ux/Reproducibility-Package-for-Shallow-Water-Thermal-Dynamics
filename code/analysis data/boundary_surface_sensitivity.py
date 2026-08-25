"""Stage-2 energy-budget sensitivity analysis for the 2024 concrete basin.

This script never changes the retained baseline Qnet calculation. It reads the
existing Figure 6 flux components, perturbs one parameter at a time, and reports
how each scenario changes annual Qnet and the diagnostic energy residual.
Concrete calculations are order-of-magnitude scenarios only and are not added
to Qnet or used to force energy-budget closure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Basin geometry and stage-1 result
A_SURFACE = 260.0
A_BOTTOM = 260.0
A_WALL = 184.8
A_WETTED = 444.8
Q_RESIDUAL_BASELINE = 52.717504735157995  # W m-2, surface-area basis
DELTA_H_ANNUAL_J_M2 = -4.753581788840919e6

# Retained Figure 6 parameters
ALBEDO_BASE = 0.07
EMISS_W_BASE = 0.97
CE_BASE = 1.3e-3
CH_BASE = 1.3e-3
ATM_CLEAR_COEFF_BASE = 0.642
CLOUD_COEFF_BASE = 0.17
WIND_EXP_BASE = 1.0 / 7.0
SIGMA = 5.67e-8

DT_BOUNDARY_K = [1, 2, 3, 5, 8, 10]
K_CONCRETE_W_MK = [1.0, 1.5, 2.0, 2.5]
L_CONCRETE_M = [0.15, 0.20, 0.25, 0.30]

SOURCES = {
    "concrete": "https://doi.org/10.1016/S0008-8846(02)00965-1",
    "concrete_moisture": "https://doi.org/10.1177/014362449201300105",
    "albedo": "https://doi.org/10.1016/j.ecolind.2023.109905",
    "emissivity": "https://doi.org/10.1175/1520-0450(1972)011%3C1391:LDOWSE%3E2.0.CO;2",
    "transfer": "https://doi.org/10.1029/2009JD012839",
    "atmospheric": "https://doi.org/10.1029/WR011i005p00742",
    "cloud": "https://doi.org/10.1016/j.solener.2010.01.012",
    "wind": "https://doi.org/10.1175/1520-0450(1994)033%3C0757:DTPLWP%3E2.0.CO;2",
}


def calc_vapor_pressure(temp_c: pd.Series) -> pd.Series:
    return 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5)) * 100.0


def integrate_flux(series: pd.Series, dt_s: pd.Series) -> tuple[float, float]:
    analysis = series.astype(float).interpolate(method="time", limit_area="inside")
    if analysis.isna().any():
        raise ValueError("Scenario flux retains missing values at an analysis boundary.")
    energy_j_m2 = float((analysis.iloc[1:] * dt_s.iloc[1:]).sum())
    duration_s = float(dt_s.iloc[1:].sum())
    return energy_j_m2, energy_j_m2 / duration_s


def scenario_result(
    df: pd.DataFrame,
    dt_s: pd.Series,
    qnet: pd.Series,
    category: str,
    parameter: str,
    baseline_value: float,
    test_value: float,
    units: str,
    range_label: str,
    source: str,
) -> dict[str, float | str]:
    energy, mean_qnet = integrate_flux(qnet, dt_s)
    duration_s = float(dt_s.iloc[1:].sum())
    residual_energy = energy - DELTA_H_ANNUAL_J_M2
    residual_mean = residual_energy / duration_s
    return {
        "Category": category,
        "Parameter": parameter,
        "Baseline_value": baseline_value,
        "Test_value": test_value,
        "Units": units,
        "Range_label": range_label,
        "Annual_mean_Qnet_W_m2": mean_qnet,
        "Cumulative_Qnet_MJ_m2": energy / 1e6,
        "Delta_mean_Qnet_W_m2": mean_qnet - df.attrs["baseline_mean_qnet"],
        "Annual_mean_residual_W_m2": residual_mean,
        "Cumulative_residual_MJ_m2": residual_energy / 1e6,
        "Residual_reduction_percent": (
            (Q_RESIDUAL_BASELINE - residual_mean) / Q_RESIDUAL_BASELINE * 100.0
        ),
        "Source_or_basis": source,
        "Status": "Literature-informed scenario; not a site measurement",
    }


def boundary_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    q_required = Q_RESIDUAL_BASELINE * A_SURFACE / A_WETTED
    required = pd.DataFrame({
        "DeltaT_boundary_K": DT_BOUNDARY_K,
        "q_required_W_m2_wetted_boundary": q_required,
        "U_required_W_m2_K": [q_required / value for value in DT_BOUNDARY_K],
        "Interpretation": "Required effective boundary exchange; not a measured U-value",
    })

    concrete_rows = []
    for thickness in L_CONCRETE_M:
        for conductivity in K_CONCRETE_W_MK:
            for delta_t in DT_BOUNDARY_K:
                q_cond = conductivity * delta_t / thickness
                concrete_rows.append({
                    "k_concrete_W_m_K": conductivity,
                    "L_concrete_m": thickness,
                    "DeltaT_K": delta_t,
                    "q_cond_W_m2": q_cond,
                    "q_required_W_m2": q_required,
                    "q_cond_over_q_required": q_cond / q_required,
                    "Scenario_status": "Assumption only; no site wall thickness or k measurement",
                    "Source": SOURCES["concrete"],
                })
    concrete = pd.DataFrame(concrete_rows)

    summary_rows = []
    for thickness in L_CONCRETE_M:
        for conductivity in K_CONCRETE_W_MK:
            u_slab = conductivity / thickness
            summary_rows.append({
                "k_concrete_W_m_K": conductivity,
                "L_concrete_m": thickness,
                "Slab_k_over_L_W_m2_K": u_slab,
                "DeltaT_needed_for_q_required_K": q_required / u_slab,
                "q_at_1K_W_m2": u_slab,
                "q_at_3K_W_m2": u_slab * 3.0,
                "q_at_5K_W_m2": u_slab * 5.0,
                "q_at_10K_W_m2": u_slab * 10.0,
                "Interpretation": "1-D concrete-only scale; excludes water-side, soil-side, and transient resistances",
            })
    concrete_summary = pd.DataFrame(summary_rows)
    return required, concrete, concrete_summary


def surface_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    dt_s = df.index.to_series().diff().dt.total_seconds()
    baseline_energy, baseline_mean = integrate_flux(df["Q_net"], dt_s)
    df.attrs["baseline_mean_qnet"] = baseline_mean
    df.attrs["baseline_energy"] = baseline_energy

    ts = df["25cm"].astype(float)
    ta_col = [c for c in df.columns if "气温" in c or "大气温度" in c][0]
    rh_col = [c for c in df.columns if "湿度" in c][0]
    ta = df[ta_col].astype(float)
    rh = df[rh_col].astype(float) / 100.0
    ea = calc_vapor_pressure(ta) * rh
    cloud = df["Cloud_Cover_C"].astype(float)
    tk = ta + 273.15
    outgoing_unit = SIGMA * (ts + 273.15) ** 4

    components = df[["Q_sw", "Q_lw", "Q_e", "Q_h"]].astype(float)
    rows: list[dict[str, float | str]] = []

    def total(q_sw=None, q_lw=None, q_e=None, q_h=None):
        return (
            components["Q_sw"] if q_sw is None else q_sw
        ) + (
            components["Q_lw"] if q_lw is None else q_lw
        ) + (
            components["Q_e"] if q_e is None else q_e
        ) + (
            components["Q_h"] if q_h is None else q_h
        )

    # Baseline row
    rows.append(scenario_result(
        df, dt_s, df["Q_net"], "Baseline", "All retained parameters", 1.0, 1.0,
        "dimensionless", "Retained Figure 6", "Original calculation",
    ))

    for value in [0.05, 0.07, 0.10, 0.14]:
        q_sw = components["Q_sw"] * (1.0 - value) / (1.0 - ALBEDO_BASE)
        rows.append(scenario_result(
            df, dt_s, total(q_sw=q_sw), "Shortwave", "Water albedo", ALBEDO_BASE, value,
            "dimensionless", "0.05-0.14", SOURCES["albedo"],
        ))

    for value in [0.96, 0.97, 0.98, 0.99]:
        q_lw = components["Q_lw"] + (EMISS_W_BASE - value) * outgoing_unit
        rows.append(scenario_result(
            df, dt_s, total(q_lw=q_lw), "Longwave", "Water emissivity", EMISS_W_BASE, value,
            "dimensionless", "0.96-0.99", SOURCES["emissivity"],
        ))

    for value in [1.0e-3, 1.3e-3, 1.5e-3, 1.9e-3]:
        q_e = components["Q_e"] * value / CE_BASE
        rows.append(scenario_result(
            df, dt_s, total(q_e=q_e), "Turbulent", "Latent transfer coefficient CE",
            CE_BASE, value, "dimensionless", "1.0e-3 to 1.9e-3", SOURCES["transfer"],
        ))
        q_h = components["Q_h"] * value / CH_BASE
        rows.append(scenario_result(
            df, dt_s, total(q_h=q_h), "Turbulent", "Sensible transfer coefficient CH",
            CH_BASE, value, "dimensionless", "1.0e-3 to 1.9e-3", SOURCES["transfer"],
        ))

    # Brutsaert leading coefficient, varied one-at-a-time by +/-10%.
    for value in [ATM_CLEAR_COEFF_BASE * 0.9, ATM_CLEAR_COEFF_BASE, ATM_CLEAR_COEFF_BASE * 1.1]:
        emiss_clear = value * np.power(ea / tk, 1.0 / 7.0)
        emiss_all = np.clip(emiss_clear * (1.0 + CLOUD_COEFF_BASE * cloud**2), 0.0, 0.98)
        q_an = emiss_all * SIGMA * tk**4
        q_lw = q_an - EMISS_W_BASE * outgoing_unit
        rows.append(scenario_result(
            df, dt_s, total(q_lw=q_lw), "Atmospheric longwave",
            "Brutsaert leading coefficient", ATM_CLEAR_COEFF_BASE, value, "dimensionless",
            "+/-10% scenario around 0.642", SOURCES["atmospheric"],
        ))

    # Cloud-cover scaling tests uncertainty in locally inferred cloud fraction.
    for value in [0.8, 1.0, 1.2]:
        cloud_test = np.clip(cloud * value, 0.0, 1.0)
        emiss_clear = ATM_CLEAR_COEFF_BASE * np.power(ea / tk, 1.0 / 7.0)
        emiss_all = np.clip(emiss_clear * (1.0 + CLOUD_COEFF_BASE * cloud_test**2), 0.0, 0.98)
        q_an = emiss_all * SIGMA * tk**4
        q_lw = q_an - EMISS_W_BASE * outgoing_unit
        rows.append(scenario_result(
            df, dt_s, total(q_lw=q_lw), "Atmospheric longwave", "Inferred cloud-cover scale",
            1.0, value, "multiplier", "+/-20% scenario", SOURCES["cloud"],
        ))

    for value in [0.08, 0.11, WIND_EXP_BASE, 0.20]:
        wind_ratio = 5.0 ** (value - WIND_EXP_BASE)
        q_e = components["Q_e"] * wind_ratio
        q_h = components["Q_h"] * wind_ratio
        rows.append(scenario_result(
            df, dt_s, total(q_e=q_e, q_h=q_h), "Wind conversion",
            "2-m to 10-m power-law exponent", WIND_EXP_BASE, value, "dimensionless",
            "0.08-0.20; 0.11 +/- 0.03 measured over water", SOURCES["wind"],
        ))

    result = pd.DataFrame(rows)
    # Remove duplicate baseline-valued rows within each sensitivity parameter.
    return result


def make_concrete_figure(concrete: pd.DataFrame, output_dir: Path) -> None:
    q_required = float(concrete["q_required_W_m2"].iloc[0])
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)
    vmin, vmax = 0.0, max(4.0, concrete["q_cond_over_q_required"].max())
    image = None
    for axis, thickness in zip(axes.ravel(), L_CONCRETE_M):
        subset = concrete[concrete["L_concrete_m"] == thickness]
        grid = subset.pivot(index="k_concrete_W_m_K", columns="DeltaT_K", values="q_cond_over_q_required")
        image = axis.imshow(grid.values, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        axis.set_xticks(range(len(grid.columns)), [str(value) for value in grid.columns])
        axis.set_yticks(range(len(grid.index)), [f"{value:.1f}" for value in grid.index])
        axis.set_xlabel(r"Assumed $\Delta T$ (K)")
        axis.set_ylabel(r"Assumed $k$ (W m$^{-1}$ K$^{-1}$)")
        axis.set_title(f"Assumed concrete thickness = {thickness:.2f} m", fontsize=10)
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                value = grid.iloc[row, col]
                color = "white" if value > vmax * 0.48 else "black"
                axis.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=8, color=color)
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.9, pad=0.02)
    colorbar.set_label(r"$q_{cond}/q_{required}$")
    fig.suptitle(
        f"Concrete-only 1-D conduction scale relative to {q_required:.2f} W m$^{{-2}}$ required",
        fontsize=11,
    )
    fig.savefig(output_dir / "Figure_concrete_conduction_sensitivity.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / "Figure_concrete_conduction_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


def make_surface_figure(surface: pd.DataFrame, output_dir: Path) -> None:
    nonbase = surface[surface["Category"] != "Baseline"].copy()
    grouped = []
    for parameter, block in nonbase.groupby("Parameter", sort=False):
        grouped.append({
            "Parameter": parameter,
            "Minimum": block["Delta_mean_Qnet_W_m2"].min(),
            "Maximum": block["Delta_mean_Qnet_W_m2"].max(),
            "Best_reduction": block["Residual_reduction_percent"].max(),
        })
    summary = pd.DataFrame(grouped).sort_values("Best_reduction")
    y = np.arange(len(summary))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.3), gridspec_kw={"width_ratios": [1.2, 1]})
    axes[0].hlines(y, summary["Minimum"], summary["Maximum"], color="#376B95", linewidth=5)
    axes[0].scatter(summary["Minimum"], y, color="#D95F4C", s=24, zorder=3)
    axes[0].scatter(summary["Maximum"], y, color="#2F8F6B", s=24, zorder=3)
    axes[0].axvline(0.0, color="0.25", linewidth=0.8)
    axes[0].set_yticks(y, summary["Parameter"])
    axes[0].set_xlabel(r"Change in annual mean $Q_{net}$ (W m$^{-2}$)")
    axes[0].set_title("(a) OAT response range", loc="left", fontweight="bold", fontsize=10)
    axes[1].barh(y, summary["Best_reduction"], color="#D9A62E")
    axes[1].axvline(100.0, color="0.25", linestyle="--", linewidth=0.8)
    axes[1].set_yticks(y, [])
    axes[1].set_xlabel("Residual reduction (%)")
    axes[1].set_title("(b) Best case within each range", loc="left", fontweight="bold", fontsize=10)
    for axis in axes:
        axis.grid(False)
        axis.tick_params(direction="in")
    fig.tight_layout()
    fig.savefig(output_dir / "Figure_surface_flux_sensitivity.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / "Figure_surface_flux_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


def resolve_default_input(script_dir: Path) -> Path:
    repository_root = script_dir.parents[1]
    candidate = repository_root / "results" / "reproduced" / "Figure6" / "Figure6_calculated_data_revised.csv"
    if candidate.exists():
        return candidate
    raise FileNotFoundError("Run code/figures/Figure6.py first or provide --input PATH.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    input_path = args.input.resolve() if args.input else resolve_default_input(script_dir)
    repository_root = script_dir.parents[1]
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else repository_root / "results" / "reproduced" / "boundary_diagnostic"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, index_col=0, parse_dates=True, encoding="utf-8-sig").sort_index()
    required = ["25cm", "Q_sw", "Q_lw", "Q_e", "Q_h", "Q_net", "Cloud_Cover_C"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    required_table, concrete, concrete_summary = boundary_tables()
    surface = surface_sensitivity(df)
    oat_summary_rows = []
    for parameter, block in surface[surface["Category"] != "Baseline"].groupby("Parameter", sort=False):
        oat_summary_rows.append({
            "Parameter": parameter,
            "Minimum_Delta_Qnet_W_m2": block["Delta_mean_Qnet_W_m2"].min(),
            "Maximum_Delta_Qnet_W_m2": block["Delta_mean_Qnet_W_m2"].max(),
            "Maximum_residual_reduction_percent": block["Residual_reduction_percent"].max(),
            "Minimum_residual_remaining_W_m2": block["Annual_mean_residual_W_m2"].min(),
            "Eliminates_residual": bool((block["Annual_mean_residual_W_m2"].abs() < 1.0).any()),
        })
    oat_summary = pd.DataFrame(oat_summary_rows).sort_values(
        "Maximum_residual_reduction_percent", ascending=False
    )
    literature = pd.DataFrame([
        ["Concrete thermal conductivity", "1.0-2.5 W m-1 K-1", "Scenario subset within experimentally reported variability; not measured at this site", SOURCES["concrete"]],
        ["Concrete moisture dependence", "Context only", "Moisture and density affect k; reinforces scenario treatment", SOURCES["concrete_moisture"]],
        ["Concrete thickness", "0.15, 0.20, 0.25, 0.30 m", "Assumed scenarios because construction drawings were unavailable", "No site measurement"],
        ["Water albedo", "0.05-0.14", "Literature-informed open-water/turbid-water scenarios; baseline 0.07 retained", SOURCES["albedo"]],
        ["Water emissivity", "0.96-0.99", "Laboratory/field-informed scenarios; baseline 0.97 retained", SOURCES["emissivity"]],
        ["CE and CH", "1.0e-3 to 1.9e-3", "Covers constant and higher literature values; baseline 1.3e-3 retained", SOURCES["transfer"]],
        ["Brutsaert coefficient", "0.642 +/- 10%", "Sensitivity bracket for an empirical atmospheric-emissivity parameter; not a measured local range", SOURCES["atmospheric"]],
        ["Inferred cloud cover", "0.8x, 1.0x, 1.2x", "Sensitivity bracket around the Luo-type shortwave inference; not measured cloud fraction", SOURCES["cloud"]],
        ["Wind-profile exponent", "0.08, 0.11, 1/7, 0.20", "Includes Hsu et al. over-water estimate 0.11 +/- 0.03 and wider scenario", SOURCES["wind"]],
    ], columns=["Parameter", "Range_or_values", "Status_and_basis", "Source"])

    baseline_row = surface[surface["Category"] == "Baseline"].iloc[0]
    checks = {
        "input_file": input_path.name,
        "expected_relative_input": "results/reproduced/Figure6/Figure6_calculated_data_revised.csv",
        "baseline_annual_mean_qnet_W_m2": float(baseline_row["Annual_mean_Qnet_W_m2"]),
        "baseline_cumulative_qnet_MJ_m2": float(baseline_row["Cumulative_Qnet_MJ_m2"]),
        "baseline_mean_residual_W_m2": float(baseline_row["Annual_mean_residual_W_m2"]),
        "q_boundary_required_W_m2": float(required_table["q_required_W_m2_wetted_boundary"].iloc[0]),
        "whole_basin_residual_power_kW": Q_RESIDUAL_BASELINE * A_SURFACE / 1000.0,
        "geometry_m2": {
            "surface": A_SURFACE,
            "bottom": A_BOTTOM,
            "wetted_walls": A_WALL,
            "total_wetted_concrete": A_WETTED,
        },
        "interpretive_constraint": (
            "Concrete q_cond values are scenario scales only. They are not added to Qnet, "
            "not used to close the budget, and not applied independently to the bottom-soil boundary."
        ),
    }

    required_table.to_csv(output_dir / "boundary_required_U_table.csv", index=False, encoding="utf-8-sig")
    concrete.to_csv(output_dir / "concrete_conduction_sensitivity.csv", index=False, encoding="utf-8-sig")
    concrete_summary.to_csv(output_dir / "concrete_scenario_summary.csv", index=False, encoding="utf-8-sig")
    surface.to_csv(output_dir / "surface_flux_OAT_sensitivity.csv", index=False, encoding="utf-8-sig")
    oat_summary.to_csv(output_dir / "surface_flux_OAT_summary.csv", index=False, encoding="utf-8-sig")
    literature.to_csv(output_dir / "literature_parameter_basis.csv", index=False, encoding="utf-8-sig")
    (output_dir / "stage2_quality_control.json").write_text(
        json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "stage2_workbook_data.json").write_text(
        json.dumps({
            "required": required_table.to_dict(orient="records"),
            "concrete": concrete.to_dict(orient="records"),
            "concrete_summary": concrete_summary.to_dict(orient="records"),
            "surface": surface.to_dict(orient="records"),
            "oat_summary": oat_summary.to_dict(orient="records"),
            "literature": literature.to_dict(orient="records"),
            "quality": checks,
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    make_concrete_figure(concrete, output_dir)
    make_surface_figure(surface, output_dir)

    print(json.dumps(checks, indent=2, ensure_ascii=False))
    print("\nBoundary requirement:\n", required_table.to_string(index=False))
    best = surface[surface["Category"] != "Baseline"].nlargest(10, "Residual_reduction_percent")
    print("\nLargest OAT residual reductions:\n", best[[
        "Parameter", "Test_value", "Delta_mean_Qnet_W_m2",
        "Annual_mean_residual_W_m2", "Residual_reduction_percent",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
