import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.integrate import simpson
from pathlib import Path
import subprocess
from matplotlib.lines import Line2D
import re

# ================= 路径配置区域（仅路径经过公开包适配） =================
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = REPOSITORY_ROOT / "data" / "raw_or_input" / "full_year" / "meteorology_water_temperature_2024_10min.csv"
OUTPUT_DIR = REPOSITORY_ROOT / "results" / "reproduced" / "Figure6"

OUTPUT_STEM = OUTPUT_DIR / "Figure6"
OUTPUT_CSV = OUTPUT_DIR / "Figure6_calculated_data_revised.csv"
OUTPUT_SEASONAL_CSV = OUTPUT_DIR / "Figure6_seasonal_summary_revised.csv"

INKSCAPE_EXE = None  # Optional; SVG is always saved, EMF conversion is skipped when unavailable.

PLOT_START = "2024-01-01"
PLOT_END = "2024-12-31 23:59:59"

LATITUDE = 30.47
LONGITUDE = 114.31
STANDARD_MERIDIAN = 120.0

WATER_DEPTH = 2.8  # m，全水深近似积分范围

SIGMA = 5.67e-8
EMISS_W = 0.97
ALBEDO = 0.07
R_D = 287.058
C_P_AIR = 1005
C_P_WATER = 4186
L_V = 2.45e6
CE = 1.3e-3
CH = 1.3e-3

# ================= 可直接微调的出版格式参数 =================
FIGURE_SIZE = (7.2, 4.8)       # 保持原图幅比例与尺寸
AXIS_LABEL_FONTSIZE = 15       # 在上一 revised 版基础上再放大 2 pt
TICK_LABEL_FONTSIZE = 12       # 在上一 revised 版基础上再放大 1 pt
LEGEND_FONTSIZE = 11.5         # 再放大 1 pt，同时保留五列图例布局
PANEL_LABEL_FONTSIZE = 15      # 在上一 revised 版基础上再放大 2 pt
DATE_TICK_INTERVAL_MONTHS = 2  # 保持原月份刻度间隔
DATE_TICK_FORMAT = "%Y-%m"     # 保持原日期格式
OUTPUT_DPI = 800

# ================= 字体 =================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 12,
    "axes.linewidth": 0.9,
    "axes.labelsize": AXIS_LABEL_FONTSIZE,
    "axes.titlesize": 12,
    "xtick.labelsize": TICK_LABEL_FONTSIZE,
    "ytick.labelsize": TICK_LABEL_FONTSIZE,
    "legend.fontsize": LEGEND_FONTSIZE,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none"
})

# ================= 配色 =================
COLOR_RED = "#E76354"
COLOR_YELLOW = "#F7D06A"
COLOR_LIGHT_BLUE = "#72BCD4"
COLOR_DEEP_BLUE = "#376B95"

COLOR_QSW = COLOR_RED
COLOR_QLW = COLOR_YELLOW
COLOR_QE  = COLOR_LIGHT_BLUE
COLOR_QH  = COLOR_DEEP_BLUE

COLOR_QNET = COLOR_DEEP_BLUE
COLOR_S = COLOR_RED

COLOR_GREY = "0.75"


def save_figure_all(fig, output_stem, inkscape_exe=None, dpi=600):
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    png_path = output_stem.with_suffix(".png")
    svg_path = output_stem.with_suffix(".svg")
    emf_path = output_stem.with_suffix(".emf")

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    print(f"[OK] PNG已保存: {png_path}")

    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    print(f"[OK] SVG已保存: {svg_path}")

    eps_path = output_stem.parent / "Figure6.eps"
    fig.savefig(eps_path, format="eps", bbox_inches="tight")
    print(f"[OK] EPS已保存: {eps_path}")

    if inkscape_exe is not None:
        inkscape_path = Path(inkscape_exe)
        if inkscape_path.exists():
            try:
                subprocess.run(
                    [
                        str(inkscape_path),
                        str(svg_path),
                        f"--export-filename={str(emf_path)}"
                    ],
                    check=True
                )
                print(f"[OK] EMF已保存: {emf_path}")
            except subprocess.CalledProcessError as e:
                print("[WARN] Inkscape转换EMF失败。")
                print(e)
        else:
            print(f"[WARN] 未找到 Inkscape: {inkscape_exe}")
            print("   已保存 PNG 和 SVG，EMF 未生成。")
    else:
        print("[WARN] 未提供 Inkscape 路径，仅保存 PNG 和 SVG。")


def calc_vapor_pressure(temp_c):
    """饱和水汽压，单位 Pa"""
    return 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5)) * 100


def calc_water_density(temp_c):
    """淡水密度，单位 kg/m3"""
    return 1000 * (
        1 - (temp_c + 288.9414) * (temp_c - 3.9863) ** 2
        / (508929.2 * (temp_c + 68.12963))
    )


def calculate_theoretical_solar_radiation(timestamps):
    """理论晴空短波辐射，单位 W/m²"""
    doy = timestamps.dayofyear.values
    hour = timestamps.hour.values
    minute = timestamps.minute.values

    b = 2 * np.pi * (doy - 81) / 364.0
    eot = 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)

    local_time = hour + minute / 60.0
    lon_correction = (LONGITUDE - STANDARD_MERIDIAN) * 4.0 / 60.0
    tst = local_time + lon_correction + eot / 60.0

    omega = (tst - 12.0) * 15.0 * np.pi / 180.0
    delta = np.arcsin(0.39795 * np.cos(0.98563 * (doy - 173) * np.pi / 180.0))

    lat_rad = LATITUDE * np.pi / 180.0
    sin_elevation = (
        np.sin(lat_rad) * np.sin(delta)
        + np.cos(lat_rad) * np.cos(delta) * np.cos(omega)
    )
    sin_elevation = np.maximum(sin_elevation, 0)

    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365.0)
    r_ext = 1367.0 * dr * sin_elevation
    r_so = 0.75 * r_ext

    return r_so


def get_depth_columns(df):
    """
    自动识别水温深度列。
    支持列名格式：25cm、75cm、125cm、175cm、225cm。
    返回 depths_m 和 col_names。
    """
    depth_col_pairs = []

    for col in df.columns:
        col_str = str(col).strip()

        match_cm = re.fullmatch(r"(\d+(\.\d+)?)\s*cm", col_str, flags=re.IGNORECASE)
        if match_cm:
            depth_m = float(match_cm.group(1)) / 100.0
            depth_col_pairs.append((depth_m, col))
            continue

        # 兼容旧版数字列名，如 0.25、0.75、1.25
        try:
            depth_m = float(col_str)
            if 0 < depth_m <= 5:
                depth_col_pairs.append((depth_m, col))
        except ValueError:
            pass

    depth_col_pairs = sorted(depth_col_pairs, key=lambda x: x[0])

    if len(depth_col_pairs) == 0:
        raise ValueError("未识别到水温深度列，请检查列名是否为 25cm、75cm 等格式。")

    depths = np.array([p[0] for p in depth_col_pairs])
    col_names = [p[1] for p in depth_col_pairs]

    return depths, col_names


def assign_season(month):
    if month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Autumn"
    else:
        return "Winter"


def compute_seasonal_summary(df):
    """
    输出季节平均热通量和基于平均绝对幅值的贡献率。
    括号外：带符号季节均值。
    括号内：mean absolute magnitude contribution。
    """
    flux_cols = ["Q_sw", "Q_lw", "Q_e", "Q_h"]

    df_daily = df[flux_cols + ["Q_net", "S"]].resample("D").mean()
    df_daily["Season"] = df_daily.index.month.map(assign_season)

    rows = []

    for season in ["Spring", "Summer", "Autumn", "Winter"]:
        sub = df_daily[df_daily["Season"] == season].copy()

        signed_means = sub[flux_cols].mean()
        abs_means = sub[flux_cols].abs().mean()
        contribution = abs_means / abs_means.sum() * 100

        row = {
            "Season": season,
            "Days": len(sub),
            "Q_sw_mean": signed_means["Q_sw"],
            "Q_sw_contribution_%": contribution["Q_sw"],
            "Q_lw_mean": signed_means["Q_lw"],
            "Q_lw_contribution_%": contribution["Q_lw"],
            "Q_e_mean": signed_means["Q_e"],
            "Q_e_contribution_%": contribution["Q_e"],
            "Q_h_mean": signed_means["Q_h"],
            "Q_h_contribution_%": contribution["Q_h"],
            "Q_net_mean": sub["Q_net"].mean(),
            "S_mean": sub["S"].mean()
        }
        rows.append(row)

    sub = df_daily.copy()
    signed_means = sub[flux_cols].mean()
    abs_means = sub[flux_cols].abs().mean()
    contribution = abs_means / abs_means.sum() * 100

    rows.append({
        "Season": "Annual average",
        "Days": len(sub),
        "Q_sw_mean": signed_means["Q_sw"],
        "Q_sw_contribution_%": contribution["Q_sw"],
        "Q_lw_mean": signed_means["Q_lw"],
        "Q_lw_contribution_%": contribution["Q_lw"],
        "Q_e_mean": signed_means["Q_e"],
        "Q_e_contribution_%": contribution["Q_e"],
        "Q_h_mean": signed_means["Q_h"],
        "Q_h_contribution_%": contribution["Q_h"],
        "Q_net_mean": sub["Q_net"].mean(),
        "S_mean": sub["S"].mean()
    })

    summary = pd.DataFrame(rows)

    # 保留一位小数，便于和论文表格对应
    numeric_cols = summary.select_dtypes(include=[np.number]).columns
    summary[numeric_cols] = summary[numeric_cols].round(1)

    summary.to_csv(OUTPUT_SEASONAL_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] 季节热通量汇总表已保存: {OUTPUT_SEASONAL_CSV}")

    print("\n>>> 季节热通量汇总预览：")
    print(summary)

    return summary


def compute_heat_budget():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(">>> 正在计算 2024 年热通量与热收支...")

    try:
        df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True, encoding="gb18030")

    # Remove spreadsheet-export helper columns that are not physical variables.
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")]

    df = df.loc[PLOT_START:PLOT_END].copy()
    df = df.sort_index()

    depths, col_names = get_depth_columns(df)
    col_surf = col_names[0]

    print(">>> 识别到的水温深度列：")
    for d, c in zip(depths, col_names):
        print(f"    {c} -> {d:.2f} m")

    col_air = [c for c in df.columns if "气温" in c or "大气温度" in c][0]
    col_rh = [c for c in df.columns if "湿度" in c][0]
    col_wind = [c for c in df.columns if "风速" in c][0]
    col_rad = [c for c in df.columns if "辐射" in c and "累计" not in c][0]
    col_press = [c for c in df.columns if "气压" in c][0]

    Ta = df[col_air].astype(float)
    Ts = df[col_surf].astype(float)
    RH = df[col_rh].astype(float) / 100.0

    # 使用实测 2 m 风速，并换算至 U10
    U_2m = df[col_wind].astype(float).clip(lower=0)
    U10 = U_2m * (10 / 2) ** (1 / 7)

    df["U_2m"] = U_2m
    df["U10"] = U10

    # 短波辐射非负处理
    R_sw_obs = df[col_rad].astype(float).clip(lower=0)

    # 原代码气压计算逻辑保留：hPa -> Pa
    P_atm_actual = df[col_press].astype(float) * 100.0

    R_so_theo = calculate_theoretical_solar_radiation(df.index)
    df["Rso_theoretical"] = R_so_theo

    # 云量由实测短波与理论晴空短波反推
    cloud_cover = np.full(len(df), np.nan)
    daytime_mask = R_so_theo > 20

    ratio = R_sw_obs[daytime_mask] / R_so_theo[daytime_mask]
    ratio = np.clip(ratio, 0, 1)
    cloud_cover[daytime_mask] = np.sqrt(1 - ratio)

    df["Cloud_Cover_C"] = cloud_cover
    df["Cloud_Cover_C"] = df["Cloud_Cover_C"].interpolate(method="time").bfill().ffill()
    C = df["Cloud_Cover_C"].values

    ea_sat = calc_vapor_pressure(Ta)
    es_sat = calc_vapor_pressure(Ts)
    ea = ea_sat * RH

    # 净短波
    Q_sw = R_sw_obs * (1 - ALBEDO)

    # 净长波
    Tk = Ta + 273.15
    emiss_a_clear = 0.642 * np.power((ea / Tk), 1 / 7)
    emiss_a_allsky = emiss_a_clear * (1 + 0.17 * C**2)
    emiss_a_allsky = np.clip(emiss_a_allsky, 0, 0.98)

    Q_an = emiss_a_allsky * SIGMA * (Ta + 273.15) ** 4
    Q_br = EMISS_W * SIGMA * (Ts + 273.15) ** 4
    Q_lw = Q_an - Q_br

    # 空气密度
    Tv = Tk * (1 + 0.378 * ea / P_atm_actual)
    rho_a_dynamic = P_atm_actual / (R_D * Tv)

    # 潜热与感热，全部保留带符号形式
    const_le = rho_a_dynamic * L_V * CE * (0.622 / P_atm_actual)
    Q_h = rho_a_dynamic * C_P_AIR * CH * U10 * (Ta - Ts)
    Q_e = const_le * U10 * (ea - es_sat)

    # 净热通量：所有分量均为带符号代数和
    Q_net = Q_sw + Q_lw + Q_h + Q_e

    # ================= 热储量计算 =================
    # 将最浅层温度外推代表 0–最浅层，将最深层温度外推代表 最深层–2.8 m
    temp_matrix = df[col_names].astype(float).values
    heat_content = []

    for i in range(len(df)):
        temps = temp_matrix[i]

        if np.isnan(temps).any():
            heat_content.append(np.nan)
        else:
            depths_ext = np.r_[0.0, depths, WATER_DEPTH]
            temps_ext = np.r_[temps[0], temps, temps[-1]]

            rhos = calc_water_density(temps_ext)
            energy_density = rhos * C_P_WATER * temps_ext

            H = simpson(y=energy_density, x=depths_ext)
            heat_content.append(H)

    df["Heat_Content_Jm2"] = heat_content

    # 根据实际时间间隔计算 S
    dt_seconds = df.index.to_series().diff().dt.total_seconds().values
    df["S"] = df["Heat_Content_Jm2"].diff() / dt_seconds

    # 保存各通量
    df["Q_sw"] = Q_sw
    df["Q_lw"] = Q_lw
    df["Q_h"] = Q_h
    df["Q_e"] = Q_e
    df["Q_net"] = Q_net

    df.to_csv(OUTPUT_CSV, encoding="utf-8-sig")
    print(f"[OK] 详细数据已保存: {OUTPUT_CSV}")

    compute_seasonal_summary(df)
    plot_heat_budget(df)


def plot_heat_budget(df):
    print(">>> 正在生成热通量组合图...")

    df_daily = df[["Q_sw", "Q_lw", "Q_h", "Q_e", "Q_net", "S"]].resample("D").mean()

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=FIGURE_SIZE,
        sharex=True,
        gridspec_kw={
            "height_ratios": [1, 1],
            "hspace": 0.24
        }
    )

    # ================= (a) 热通量分量 =================
    ax1.plot(
        df.index,
        df["Q_sw"],
        color=COLOR_GREY,
        alpha=0.45,
        linewidth=0.25,
        zorder=1
    )

    ax1.plot(df_daily.index, df_daily["Q_sw"], color=COLOR_QSW, linewidth=1.35, zorder=3)
    ax1.plot(df_daily.index, df_daily["Q_lw"], color=COLOR_QLW, linewidth=1.35, zorder=3)
    ax1.plot(df_daily.index, df_daily["Q_e"],  color=COLOR_QE,  linewidth=1.35, zorder=3)
    ax1.plot(df_daily.index, df_daily["Q_h"],  color=COLOR_QH,  linewidth=1.35, zorder=3)

    ax1.axhline(0, color="0.25", linestyle="--", linewidth=0.8)
    ax1.set_ylabel("Heat flux (W/m²)")
    ax1.set_ylim(-500, 1000)

    ax1.text(
        0.0, 1.02, "(a)",
        transform=ax1.transAxes,
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="bold",
        va="bottom",
        ha="left"
    )

    handles1 = [
        Line2D([0], [0], color=COLOR_QSW, lw=1.35, label=r"$Q_{sw}$"),
        Line2D([0], [0], color=COLOR_QLW, lw=1.35, label=r"$Q_{lw}$"),
        Line2D([0], [0], color=COLOR_QE,  lw=1.35, label=r"$Q_e$"),
        Line2D([0], [0], color=COLOR_QH,  lw=1.35, label=r"$Q_h$"),
        Line2D([0], [0], color=COLOR_GREY, alpha=0.45, lw=1.0, label=r"Instantaneous $Q_{sw}$")
    ]

    ax1.legend(
        handles=handles1,
        loc="lower left",
        ncol=5,
        frameon=False,
        handlelength=1.6,
        columnspacing=0.75,
        borderpad=0.2
    )

    # ================= (b) 热收支 =================
    ax2.plot(
        df.index,
        df["Q_net"],
        color=COLOR_GREY,
        alpha=0.50,
        linewidth=0.25,
        zorder=1
    )

    ax2.plot(df_daily.index, df_daily["Q_net"], color=COLOR_QNET, linewidth=1.5, zorder=3)
    ax2.plot(df_daily.index, df_daily["S"],     color=COLOR_S,    linewidth=1.5, zorder=3)

    ax2.axhline(0, color="0.25", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("Heat budget (W/m²)")
    ax2.set_xlabel("Date", labelpad=10)
    ax2.tick_params(axis="x", pad=8)
    ax2.set_ylim(-500, 1000)

    ax2.text(
        0.0, 1.02, "(b)",
        transform=ax2.transAxes,
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="bold",
        va="bottom",
        ha="left"
    )

    handles2 = [
        Line2D([0], [0], color=COLOR_QNET, lw=1.5, label=r"$Q_{net}$"),
        Line2D([0], [0], color=COLOR_S, lw=1.5, label=r"$S$"),
        Line2D([0], [0], color=COLOR_GREY, alpha=0.50, lw=1.0, label=r"Instantaneous $Q_{net}$")
    ]

    ax2.legend(
        handles=handles2,
        loc="upper right",
        ncol=3,
        frameon=False,
        handlelength=1.6,
        columnspacing=0.9,
        borderpad=0.2
    )

    # ================= X轴格式 =================
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=DATE_TICK_INTERVAL_MONTHS))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter(DATE_TICK_FORMAT))
    ax2.set_xlim(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"))
    plt.setp(ax2.get_xticklabels(), rotation=0, ha="center")

    ax2.tick_params(axis="x", pad=6)

    ax1.grid(False)
    ax2.grid(False)

    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_zorder(100)

    fig.subplots_adjust(
        left=0.13,
        right=0.97,
        top=0.96,
        bottom=0.12,
        hspace=0.24
    )

    save_figure_all(
        fig,
        output_stem=OUTPUT_STEM,
        inkscape_exe=INKSCAPE_EXE,
        dpi=OUTPUT_DPI
    )
print("[OK] Figure 6 已完成数据计算及 PNG、SVG 导出。")


if __name__ == "__main__":
    compute_heat_budget()
