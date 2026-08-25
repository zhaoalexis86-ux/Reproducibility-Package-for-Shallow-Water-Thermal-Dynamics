"""Quantitative annual energy-budget closure analysis for the 2024 CSITE basin.

The script reads the retained Figure 6 Qnet result and does not recalculate or
adjust any original surface heat-flux component. Short gaps are interpolated
only on analysis copies so that time integration and storage differentiation
remain continuous; the raw columns are retained in the exported time series.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import simpson


CP_WATER = 4186.0  # J kg-1 K-1; identical to Figure 6
WATER_DEPTH = 2.8  # m; identical to Figure 6
DEPTH_COLUMNS = ["25cm", "75cm", "125cm", "175cm", "225cm"]
DEPTHS_M = np.array([0.25, 0.75, 1.25, 1.75, 2.25], dtype=float)


def calc_water_density(temp_c: np.ndarray) -> np.ndarray:
    """Freshwater density (kg m-3), identical to the retained Figure 6 code."""
    return 1000.0 * (
        1.0
        - (temp_c + 288.9414) * (temp_c - 3.9863) ** 2
        / (508929.2 * (temp_c + 68.12963))
    )


def calculate_heat_content(temperatures: pd.DataFrame) -> pd.Series:
    """Calculate H = integral[rho(T) Cp T dz] over 0-2.8 m."""
    depths_extended = np.r_[0.0, DEPTHS_M, WATER_DEPTH]
    result = np.empty(len(temperatures), dtype=float)
    for row_index, values in enumerate(temperatures.to_numpy(dtype=float)):
        temps_extended = np.r_[values[0], values, values[-1]]
        energy_density = calc_water_density(temps_extended) * CP_WATER * temps_extended
        result[row_index] = simpson(y=energy_density, x=depths_extended)
    return pd.Series(result, index=temperatures.index, name="Heat_Content_recalculated_J_m2")


def describe_residual(series: pd.Series, scope: str) -> dict[str, float | int | str]:
    values = series.dropna().to_numpy(dtype=float)
    return {
        "Scope": scope,
        "N": int(values.size),
        "Mean_W_m2": float(np.mean(values)),
        "Median_W_m2": float(np.median(values)),
        "SD_W_m2": float(np.std(values, ddof=1)),
        "Minimum_W_m2": float(np.min(values)),
        "P05_W_m2": float(np.percentile(values, 5)),
        "P25_W_m2": float(np.percentile(values, 25)),
        "P75_W_m2": float(np.percentile(values, 75)),
        "P95_W_m2": float(np.percentile(values, 95)),
        "Maximum_W_m2": float(np.max(values)),
        "MAE_W_m2": float(np.mean(np.abs(values))),
        "RMSE_W_m2": float(np.sqrt(np.mean(values**2))),
        "Positive_fraction_percent": float(np.mean(values > 0) * 100.0),
    }


def period_budget(intervals: pd.DataFrame, label: str) -> dict[str, float | int | str]:
    duration = float(intervals["dt_s"].sum())
    e_qnet = float(intervals["E_Qnet_J_m2"].sum())
    delta_h = float(intervals["Delta_H_J_m2"].sum())
    e_residual = float(intervals["E_residual_J_m2"].sum())
    return {
        "Period": label,
        "Start": intervals.index.min(),
        "End": intervals.index.max(),
        "Duration_days": duration / 86400.0,
        "Interval_count": int(len(intervals)),
        "Qnet_filled_points": int(intervals["Qnet_was_filled"].sum()),
        "E_Qnet_MJ_m2": e_qnet / 1e6,
        "Delta_H_MJ_m2": delta_h / 1e6,
        "E_residual_MJ_m2": e_residual / 1e6,
        "Mean_Qnet_W_m2": e_qnet / duration,
        "Mean_storage_change_W_m2": delta_h / duration,
        "Mean_residual_W_m2": e_residual / duration,
        "Storage_to_Qnet_percent": delta_h / e_qnet * 100.0,
        "Residual_fraction_percent": e_residual / e_qnet * 100.0,
    }


def make_figure(timeseries: pd.DataFrame, monthly: pd.DataFrame, output_dir: Path) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.6,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    blue = "#376B95"
    red = "#D95F4C"
    gold = "#D9A62E"
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), gridspec_kw={"hspace": 0.32})

    ax = axes[0]
    ax.plot(
        timeseries.index,
        timeseries["Cumulative_Qnet_MJ_m2"],
        color=blue,
        linewidth=1.6,
        label=r"Cumulative $Q_{net}$",
    )
    ax.plot(
        timeseries.index,
        timeseries["Cumulative_storage_change_MJ_m2"],
        color=red,
        linewidth=1.6,
        label=r"Cumulative $\Delta H$",
    )
    ax.axhline(0.0, color="0.25", linewidth=0.7, linestyle="--")
    ax.set_ylabel(r"Cumulative energy (MJ m$^{-2}$)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.text(0.0, 1.02, "(a)", transform=ax.transAxes, fontweight="bold", va="bottom")

    ax = axes[1]
    x = np.arange(len(monthly))
    width = 0.25
    ax.bar(x - width, monthly["Mean_Qnet_W_m2"], width, color=blue, label=r"$Q_{net}$")
    ax.bar(
        x,
        monthly["Mean_storage_change_W_m2"],
        width,
        color=red,
        label=r"$\Delta H/\Delta t$",
    )
    ax.bar(
        x + width,
        monthly["Mean_residual_W_m2"],
        width,
        color=gold,
        label=r"$Q_{res}$",
    )
    ax.axhline(0.0, color="0.25", linewidth=0.7)
    ax.set_xticks(x, [pd.Timestamp(value).strftime("%b") for value in monthly["Start"]])
    ax.set_ylabel(r"Monthly mean flux (W m$^{-2}$)")
    ax.set_xlabel("Month in 2024")
    ax.legend(frameon=False, loc="upper right", ncol=3)
    ax.text(0.0, 1.02, "(b)", transform=ax.transAxes, fontweight="bold", va="bottom")

    for axis in axes:
        axis.grid(False)
        axis.tick_params(top=False, right=False)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.97, bottom=0.09)
    fig.savefig(output_dir / "Figure_energy_closure.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / "Figure_energy_closure.pdf", bbox_inches="tight")
    plt.close(fig)


def resolve_default_input(script_dir: Path) -> Path:
    repository_root = script_dir.parents[1]
    candidate = repository_root / "results" / "reproduced" / "Figure6" / "Figure6_calculated_data_revised.csv"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        "Default Figure 6 input was not found. Run code/figures/Figure6.py first "
        "or provide --input PATH."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Existing Figure 6 calculated CSV")
    parser.add_argument("--output-dir", type=Path, help="Output directory; defaults to script folder")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    input_path = args.input.resolve() if args.input else resolve_default_input(script_dir)
    repository_root = script_dir.parents[1]
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else repository_root / "results" / "reproduced" / "energy_budget"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, index_col=0, parse_dates=True, encoding="utf-8-sig").sort_index()
    required = DEPTH_COLUMNS + ["Heat_Content_Jm2", "Q_net"]
    missing_columns = [column for column in required if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")
    if df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError("Timestamps must be unique and monotonically increasing.")

    dt_all = df.index.to_series().diff().dt.total_seconds()
    if not (dt_all.iloc[1:] > 0).all():
        raise ValueError("All time intervals must be positive.")

    temperature_raw = df[DEPTH_COLUMNS].astype(float)
    temperature_filled = temperature_raw.interpolate(method="time", limit_area="inside")
    if temperature_filled.isna().any().any():
        raise ValueError("Temperature gaps remain at the analysis boundaries.")

    heat_content = calculate_heat_content(temperature_filled)
    valid_h = df["Heat_Content_Jm2"].notna()
    heat_content_validation_error = float(
        np.max(np.abs(heat_content.loc[valid_h] - df.loc[valid_h, "Heat_Content_Jm2"]))
    )
    if heat_content_validation_error > 1e-5:
        raise ValueError(f"Heat-content reproduction failed: max error={heat_content_validation_error}")

    qnet_raw = df["Q_net"].astype(float)
    qnet_filled = qnet_raw.interpolate(method="time", limit_area="inside")
    if qnet_filled.isna().any():
        raise ValueError("Qnet gaps remain at the analysis boundaries.")

    result = pd.DataFrame(index=df.index)
    result.index.name = "Date"
    result["Qnet_raw_W_m2"] = qnet_raw
    result["Qnet_analysis_W_m2"] = qnet_filled
    result["Qnet_was_filled"] = qnet_raw.isna()
    result["Heat_Content_existing_J_m2"] = df["Heat_Content_Jm2"]
    result["Heat_Content_analysis_J_m2"] = heat_content
    result["Temperature_was_filled"] = temperature_raw.isna().any(axis=1)
    result["dt_s"] = result.index.to_series().diff().dt.total_seconds()
    result["Delta_H_J_m2"] = result["Heat_Content_analysis_J_m2"].diff()
    result["Storage_change_W_m2"] = result["Delta_H_J_m2"] / result["dt_s"]
    result["Q_residual_W_m2"] = result["Qnet_analysis_W_m2"] - result["Storage_change_W_m2"]
    result["E_Qnet_J_m2"] = result["Qnet_analysis_W_m2"] * result["dt_s"]
    result["E_residual_J_m2"] = result["Q_residual_W_m2"] * result["dt_s"]
    result["Cumulative_Qnet_MJ_m2"] = result["E_Qnet_J_m2"].fillna(0.0).cumsum() / 1e6
    result["Cumulative_storage_change_MJ_m2"] = (
        result["Heat_Content_analysis_J_m2"] - result["Heat_Content_analysis_J_m2"].iloc[0]
    ) / 1e6
    result["Cumulative_residual_MJ_m2"] = result["E_residual_J_m2"].fillna(0.0).cumsum() / 1e6

    intervals = result.iloc[1:].copy()
    annual = pd.DataFrame([period_budget(intervals, "2024 observation period")])
    monthly_rows = []
    for period, group in intervals.groupby(intervals.index.to_period("M")):
        monthly_rows.append(period_budget(group, str(period)))
    monthly = pd.DataFrame(monthly_rows)

    daily_residual = (
        intervals["E_residual_J_m2"].resample("D").sum()
        / intervals["dt_s"].resample("D").sum()
    )
    monthly_residual = monthly.set_index("Period")["Mean_residual_W_m2"]
    residual_statistics = pd.DataFrame([
        describe_residual(intervals["Q_residual_W_m2"], "10-minute intervals"),
        describe_residual(daily_residual, "Daily energy-weighted mean"),
        describe_residual(monthly_residual, "Monthly energy-weighted mean"),
    ])

    raw_available_energy = float((qnet_raw.iloc[1:] * intervals["dt_s"]).sum())
    interpolated_energy = float(intervals["E_Qnet_J_m2"].sum())
    quality = {
        "source_file": input_path.name,
        "expected_relative_source": "results/reproduced/Figure6/Figure6_calculated_data_revised.csv",
        "records": int(len(df)),
        "start": str(df.index[0]),
        "end": str(df.index[-1]),
        "nominal_timestep_seconds": float(dt_all.iloc[1:].median()),
        "qnet_missing_points": int(qnet_raw.isna().sum()),
        "temperature_missing_points": int(temperature_raw.isna().any(axis=1).sum()),
        "missing_fraction_percent": float(qnet_raw.isna().mean() * 100.0),
        "heat_content_max_reproduction_error_J_m2": heat_content_validation_error,
        "qnet_energy_raw_available_only_MJ_m2": raw_available_energy / 1e6,
        "qnet_energy_after_short_gap_interpolation_MJ_m2": interpolated_energy / 1e6,
        "gap_fill_energy_difference_MJ_m2": (interpolated_energy - raw_available_energy) / 1e6,
        "integration_convention": (
            "Right-endpoint discrete sum over observed intervals: E=sum(Q(t_i)*[t_i-t_(i-1)]). "
            "No interval is extrapolated beyond 2024-12-31 23:50."
        ),
        "monthly_assignment": "Each interval is assigned to the calendar month of its ending timestamp.",
        "residual_definition": "Q_residual = Qnet_analysis - Delta_H/dt; positive values are unaccounted heat loss.",
        "temperature_gap_treatment": "Time-linear interpolation on an analysis copy only; raw values retained.",
        "qnet_gap_treatment": "Time-linear interpolation on an analysis copy only; original Qnet values unchanged.",
    }

    result.to_csv(output_dir / "energy_budget_timeseries.csv", encoding="utf-8-sig")
    annual.to_csv(output_dir / "annual_energy_budget.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output_dir / "monthly_energy_budget.csv", index=False, encoding="utf-8-sig")
    residual_statistics.to_csv(
        output_dir / "residual_flux_statistics.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "energy_budget_quality_control.json").write_text(
        json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    make_figure(result, monthly, output_dir)

    print(annual.to_string(index=False))
    print("\nResidual statistics:\n", residual_statistics.to_string(index=False))
    print("\nQuality control:\n", json.dumps(quality, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
