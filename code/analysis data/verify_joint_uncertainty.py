"""Verify archived joint-uncertainty sample bounds and reported quantiles.

This script reads the archived 100,000-sample output. It does not redraw the
Latin-hypercube ensemble or change the original uncertainty implementation.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ASSUMPTIONS = ROOT / "data" / "processed" / "analysis" / "joint_parameter_assumptions.csv"
SAMPLES = ROOT / "data" / "processed" / "analysis" / "joint_uncertainty_samples_annual.csv"
OUTPUT_DIR = ROOT / "results" / "reproduced" / "joint_uncertainty"
OUTPUT = OUTPUT_DIR / "public_verification_summary.csv"


def main() -> None:
    assumptions = pd.read_csv(ASSUMPTIONS)
    samples = pd.read_csv(SAMPLES)
    if len(samples) != 100_000:
        raise ValueError(f"Expected 100000 samples, found {len(samples)}")

    for row in assumptions.itertuples(index=False):
        series = samples[row.Parameter]
        if not series.between(row.Lower, row.Upper, inclusive="both").all():
            raise ValueError(f"Samples for {row.Parameter} exceed the archived bounds")

    rows = []
    for column, label in [
        ("Annual_Qnet_continuous_W_m2", "Annual mean Qnet"),
        ("Annual_residual_continuous_W_m2", "Annual residual"),
    ]:
        q = samples[column].quantile([0.025, 0.5, 0.975])
        rows.append(
            {
                "Metric": label,
                "n": len(samples),
                "p2.5_W_m2": q.loc[0.025],
                "median_W_m2": q.loc[0.5],
                "p97.5_W_m2": q.loc[0.975],
                "interpretation": "uncertainty propagation within prescribed parameter bounds",
            }
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)
    print("Archived joint-uncertainty samples verified.")


if __name__ == "__main__":
    main()
