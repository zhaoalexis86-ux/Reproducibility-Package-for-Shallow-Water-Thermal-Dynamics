"""Reproduce the selected daily quantitative-linkage statistics.

This public-data workflow uses the archived daily analysis-ready dataset.  It
does not reconstruct the underlying 10-min observations and intentionally
excludes the exploratory nighttime, lagged, and correlation-matrix analyses.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "processed" / "analysis" / "daily_linkage_data.csv"
OUTPUT_DIR = ROOT / "results" / "reproduced" / "quantitative_linkage"


def main() -> None:
    data = pd.read_csv(INPUT, parse_dates=["Date"])
    storage = data[["Daily_mean_Qnet_W_m2", "Daily_mean_S_W_m2"]].dropna()
    fit = linregress(storage["Daily_mean_Qnet_W_m2"], storage["Daily_mean_S_W_m2"])
    predicted = fit.intercept + fit.slope * storage["Daily_mean_Qnet_W_m2"]
    rmse = float(np.sqrt(np.mean((storage["Daily_mean_S_W_m2"] - predicted) ** 2)))

    strat = data[["Daytime_net_heat_MJ_m2", "Daily_max_St_J_m2"]].dropna()
    rho, p_value = spearmanr(strat["Daytime_net_heat_MJ_m2"], strat["Daily_max_St_J_m2"])

    selected = pd.DataFrame(
        [
            {
                "analysis": "Daily Qnet-S linear regression",
                "n": len(storage),
                "slope": fit.slope,
                "intercept_W_m2": fit.intercept,
                "R_squared": fit.rvalue**2,
                "RMSE_W_m2": rmse,
            },
            {
                "analysis": "Daytime-integrated Qnet vs same-day maximum St",
                "n": len(strat),
                "Spearman_rho": rho,
                "p_value_archived_for_completeness": p_value,
            },
        ]
    )

    seasonal_rows = []
    for season, group in data.groupby("Season", sort=False):
        valid = group[["Daytime_net_heat_MJ_m2", "Daily_max_St_J_m2"]].dropna()
        season_rho, season_p = spearmanr(valid.iloc[:, 0], valid.iloc[:, 1])
        seasonal_rows.append(
            {
                "Season": season,
                "n": len(valid),
                "Spearman_rho": season_rho,
                "p_value_archived_for_completeness": season_p,
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected.to_csv(OUTPUT_DIR / "public_recalculated_linkage_statistics.csv", index=False)
    pd.DataFrame(seasonal_rows).to_csv(
        OUTPUT_DIR / "public_recalculated_seasonal_correlations.csv", index=False
    )
    print("Public quantitative-linkage verification completed.")


if __name__ == "__main__":
    main()
