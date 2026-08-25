import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import PchipInterpolator
import subprocess

# ============================================================
# 1. File path and output settings
# ============================================================

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
# Private full-year workbook. Public representative-profile values are provided separately.
INPUT_FILE = REPOSITORY_ROOT / "data" / "raw_or_input" / "full_year" / "figure4_temperature_profiles.xlsx"
SHEET_NAME = "整合 (2)"

OUTPUT_DIR = REPOSITORY_ROOT / "results" / "reproduced" / "Figure4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_STEM = OUTPUT_DIR / "Figure4"

INKSCAPE_EXE = None  # Optional; SVG is always saved, EMF conversion is skipped when unavailable.

# ============================================================
# 2. Representative profile times
# ============================================================

PROFILE_TIMES = {
    "Winter mixed": "2024-12-26 06:00",
    "Spring persistent": "2024-05-28 05:20",
    "Summer diel": "2024-08-18 15:00",
    "Autumn decay": "2024-10-25 15:00",
}

# 夜间（只用于 Summer）
NIGHT_TIME = "2024-08-19 02:10"

# ============================================================
# 3. Column settings
# ============================================================

TIME_COL = "Time"

TEMP_COLS = {
    "25cm": 0.25,
    "75cm": 0.75,
    "125cm": 1.25,
    "175cm": 1.75,
    "225cm": 2.25,
}

DEPTHS = np.array(list(TEMP_COLS.values()), dtype=float)
TEMP_COLUMNS = list(TEMP_COLS.keys())

# ============================================================
# 4. Color settings
# ============================================================

COLORS = [
    "#376795",  # Winter
    "#72BCD5",  # Spring
    "#E76254",  # Summer
    "#FFD06F",  # Autumn
]

# ============================================================
# 5. Matplotlib style
# ============================================================

# 为保证期刊最终页面缩放后的可读性，在不改变图幅和布局的前提下放大字体。
FIGURE_SIZE = (8.6, 4.6)
AXIS_LABEL_FONTSIZE = 14
TICK_LABEL_FONTSIZE = 13
TITLE_FONTSIZE = 13
ANNOTATION_FONTSIZE = 13
PANEL_LABEL_FONTSIZE = 14
OUTPUT_DPI = 800

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 12,
    "axes.labelsize": AXIS_LABEL_FONTSIZE,
    "axes.titlesize": TITLE_FONTSIZE,
    "xtick.labelsize": TICK_LABEL_FONTSIZE,
    "ytick.labelsize": TICK_LABEL_FONTSIZE,
    "axes.linewidth": 1.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 5.0,
    "ytick.major.size": 5.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.minor.size": 3.0,
    "ytick.minor.size": 3.0,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none"
})

# ============================================================
# 6. Read data
# ============================================================

df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)

df[TIME_COL] = pd.to_datetime(df[TIME_COL])
df = df.sort_values(TIME_COL).reset_index(drop=True)

for col in TEMP_COLUMNS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=TEMP_COLUMNS, how="all").reset_index(drop=True)

# ============================================================
# 7. Helper
# ============================================================

def get_nearest_profile(data, target_time, time_col, temp_cols):
    target_time = pd.to_datetime(target_time)
    idx = (data[time_col] - target_time).abs().idxmin()
    row = data.loc[idx]
    return row[time_col], row[temp_cols].astype(float).values

# ============================================================
# 8. Extract profiles
# ============================================================

profiles = []

for i, (label, target_time) in enumerate(PROFILE_TIMES.items()):

    actual_time, temps = get_nearest_profile(df, target_time, TIME_COL, TEMP_COLUMNS)

    valid = np.isfinite(temps)
    z_valid = DEPTHS[valid]
    t_valid = temps[valid]

    z_smooth = np.linspace(z_valid.min(), z_valid.max(), 200)
    pchip = PchipInterpolator(z_valid, t_valid)
    t_smooth = pchip(z_smooth)

    delta_t = t_valid[0] - t_valid[-1]

    profiles.append({
        "label": label,
        "actual_time": actual_time,
        "z_valid": z_valid,
        "t_valid": t_valid,
        "z_smooth": z_smooth,
        "t_smooth": t_smooth,
        "delta_t": delta_t,
        "color": COLORS[i],
    })

# ============================================================
# 9. Plot
# ============================================================

fig, axes = plt.subplots(1, 4, figsize=FIGURE_SIZE, sharex=True, sharey=True)

panel_labels = ["(a)", "(b)", "(c)", "(d)"]

for ax, prof, p_lab in zip(axes, profiles, panel_labels):

    color = prof["color"]

    # smooth
    ax.plot(prof["t_smooth"], prof["z_smooth"],
            color=color, linewidth=1.8)

    # points
    ax.scatter(prof["t_valid"], prof["z_valid"],
                color=color, s=18, edgecolor="white", linewidth=0.45)

    # ---------------------------
    # ONLY panel (c): add night
    # ---------------------------
    if prof["label"] == "Summer diel":

        night_time, night_t = get_nearest_profile(
            df, NIGHT_TIME, TIME_COL, TEMP_COLUMNS
        )

        valid_n = np.isfinite(night_t)
        z_n = DEPTHS[valid_n]
        t_n = night_t[valid_n]

        z_smooth_n = np.linspace(z_n.min(), z_n.max(), 200)
        pchip_n = PchipInterpolator(z_n, t_n)
        t_smooth_n = pchip_n(z_smooth_n)

        delta_n = t_n[0] - t_n[-1]

        # night line (dashed)
        ax.plot(t_smooth_n, z_smooth_n,
                color=color, linewidth=1.8, alpha=0.7, linestyle='--')

        # night scatter
        ax.scatter(t_n, z_n,
                   color=color, s=18, alpha=0.7,
                   edgecolor="white", linewidth=0.45)

        # title (3 lines)
        ax.set_title(
            "Summer diel\n"
            f"Day: {prof['actual_time'].strftime('%Y-%m-%d %H:%M')}\n"
            f"Night: {night_time.strftime('%Y-%m-%d %H:%M')}",
            pad=10, fontsize=TITLE_FONTSIZE
        )

        # two-line ΔT (no default ΔT)
        ax.text(0.06, 0.10,
                r"$\Delta T_{day}$ = " + f"{prof['delta_t']:.2f} °C"
                + "\n"
                + r"$\Delta T_{night}$ = " + f"{delta_n:.2f} °C",
                transform=ax.transAxes,
                fontsize=ANNOTATION_FONTSIZE,
                va="bottom")

    else:
        # default ΔT (single line)
        ax.text(0.06, 0.10,
                r"$\Delta T$ = " + f"{prof['delta_t']:.2f} °C",
                transform=ax.transAxes, fontsize=ANNOTATION_FONTSIZE)

        # title with trailing empty line to match 3-line height
        ax.set_title(
            f"{prof['label']}\n{prof['actual_time'].strftime('%Y-%m-%d %H:%M')}\n",
            fontsize=TITLE_FONTSIZE, pad=10
        )

    # panel label
    ax.text(0.5, -0.1, p_lab,
            transform=ax.transAxes,
            ha="center", va="top", fontweight="bold",
            fontsize=PANEL_LABEL_FONTSIZE)

    # axes
    ax.set_ylim(2.8, 0)
    ax.set_xlim(5, 36)

    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    ax.tick_params(top=True, bottom=False,
                   direction="out", length=5)

axes[0].set_ylabel("Depth (m)")

fig.text(0.53, 1.035, "Water temperature (°C)", ha="center",
         fontsize=AXIS_LABEL_FONTSIZE)

fig.subplots_adjust(left=0.08, right=0.985,
                    top=0.78, bottom=0.18,
                    wspace=0.18)

# ============================================================
# 10. Save
# ============================================================

png_file = f"{OUTPUT_STEM}.png"
svg_file = f"{OUTPUT_STEM}.svg"
emf_file = f"{OUTPUT_STEM}.emf"
eps_file = OUTPUT_DIR / "Figure4.eps"

fig.savefig(png_file, dpi=OUTPUT_DPI, bbox_inches="tight")
fig.savefig(svg_file, bbox_inches="tight")
fig.savefig(eps_file, format="eps", bbox_inches="tight")

if INKSCAPE_EXE:
    try:
        subprocess.run([
            INKSCAPE_EXE,
            svg_file,
            "--export-type=emf",
            f"--export-filename={emf_file}"
        ], check=True)
        print("Saved:", png_file, svg_file, emf_file)
    except Exception as e:
        print("[WARN] EMF conversion skipped:", e)
else:
    print("Saved:", png_file, svg_file)
