import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from matplotlib.lines import Line2D

# ================= 路径配置区域（仅路径经过公开包适配） =================
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
# Private full-year input. The public example demonstrates its schema only.
INPUT_FILE = REPOSITORY_ROOT / "data" / "raw_or_input" / "full_year" / "meteorology_water_temperature_2024_10min.csv"

OUTPUT_DIR = REPOSITORY_ROOT / "results" / "reproduced" / "Figure3"
OUTPUT_STEM = OUTPUT_DIR / "Figure3"

TIME_COL_CANDIDATES = ["Date", "date", "Time", "Datetime", "timestamp"]

# 实际五个水温深度
TEMP_COLS = ["25cm", "75cm", "125cm", "175cm", "225cm"]
DEPTHS = np.array([0.25, 0.75, 1.25, 1.75, 2.25], dtype=float)

SURF_COL = "25cm"
BOT_COL = "225cm"

MLD_COL = "MLD_0.2_PCHIP_5depth"
MLD_THRESHOLD = 0.2

# 有效绘图深度范围
DEPTH_MIN = 0.25
DEPTH_MAX = 2.25

PLOT_START = "2024-01-01"
PLOT_END = "2024-12-31"

CMAP_NAME = "RdYlBu_r"
OUTPUT_DPI = 1000

# ================= 可直接微调的出版格式参数 =================
FIGURE_SIZE = (7.5, 6.5)       # 保持原图幅比例与尺寸
AXIS_LABEL_FONTSIZE = 15       # 在上一 revised 版基础上再放大 2 pt
TICK_LABEL_FONTSIZE = 12       # 在上一 revised 版基础上再放大 1 pt
LEGEND_FONTSIZE = 13           # 在上一 revised 版基础上再放大 1 pt
PANEL_LABEL_FONTSIZE = 15      # 在上一 revised 版基础上再放大 2 pt
DATE_TICK_INTERVAL_MONTHS = 2  # 原每月刻度略密，改为每两月
DATE_TICK_FORMAT = "%Y-%m"     # 与其他全年时间轴保持一致
DATE_TICK_ROTATION = 0

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
    "ps.fonttype": 42
})

# ================= 配色 =================
COLOR_RED = "#E76354"
COLOR_LIGHT_BLUE = "#72BCD4"
COLOR_DEEP_BLUE = "#376B95"

COLOR_SURF = COLOR_RED
COLOR_BOT = COLOR_DEEP_BLUE
COLOR_DT = COLOR_LIGHT_BLUE
COLOR_MLD = "black"


def save_figure_all(fig, output_stem, dpi=1000):
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    png_path = output_stem.with_suffix(".png")
    tif_path = output_stem.with_suffix(".tif")
    pdf_path = output_stem.with_suffix(".pdf")

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    print(f"[OK] PNG已保存: {png_path}")

    fig.savefig(tif_path, dpi=dpi, bbox_inches="tight")
    print(f"[OK] TIF已保存: {tif_path}")

    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    print(f"[OK] PDF已保存: {pdf_path}")

    eps_path = output_stem.parent / "Figure3.eps"
    fig.savefig(eps_path, format="eps", bbox_inches="tight")
    print(f"[OK] EPS已保存: {eps_path}")


def save_plot_data_csv(df_raw, df_daily, time_col, output_stem):
    """
    输出绘图所使用的数据：
    1. daily_mean：用于(a)表底层温度、ΔT和(b)水温等值线图
    2. raw_mld：用于(c)原始高频MLD曲线
    """
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    daily_out = df_daily.copy()
    daily_out = daily_out.reset_index().rename(columns={time_col: "Date"})
    daily_out["Surface"] = daily_out[SURF_COL]
    daily_out["Bottom"] = daily_out[BOT_COL]
    daily_out["Delta_T"] = daily_out[SURF_COL] - daily_out[BOT_COL]
    daily_out["MLD"] = np.nan
    daily_out["data_type"] = "daily_mean"

    raw_out = df_raw[[time_col] + TEMP_COLS + [MLD_COL]].copy()
    raw_out = raw_out.rename(columns={time_col: "Date", MLD_COL: "MLD"})
    # Keep the instantaneous surface/bottom values in the exported audit data.
    # They are not required by panel (c), but retaining them makes the reproduced
    # CSV numerically identical to the dataset archived with the final figure.
    raw_out["Surface"] = raw_out[SURF_COL]
    raw_out["Bottom"] = raw_out[BOT_COL]
    # The archived final plotting CSV stores simultaneous missing surface and
    # bottom readings as zero temperature difference.
    raw_out["Delta_T"] = (raw_out[SURF_COL] - raw_out[BOT_COL]).fillna(0.0)
    raw_out["data_type"] = "raw_mld"

    common_cols = ["data_type", "Date"] + TEMP_COLS + ["Surface", "Bottom", "Delta_T", "MLD"]

    plot_data = pd.concat(
        [
            daily_out[common_cols],
            raw_out[common_cols]
        ],
        ignore_index=True
    )

    csv_path = output_stem.with_suffix(".csv")
    plot_data.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 绘图数据CSV已保存: {csv_path}")


def find_time_col(df):
    for c in TIME_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise KeyError("未找到时间列，请检查是否存在 Date / Time / Datetime 等列。")


def load_data(path):
    path = Path(path)

    if path.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="gb18030")
    else:
        df = pd.read_excel(path)

    return df


def calc_mld_pchip_one_profile(temps, depths, threshold=0.2, max_depth=2.25):
    """
    基于一个时刻的五层水温计算 MLD。
    判据：相对于近表层水温，水温首次降低 threshold °C 的深度。
    若全剖面未达到阈值，则记为 max_depth。
    若数据不足，则返回 NaN。
    """
    temps = np.asarray(temps, dtype=float)
    depths = np.asarray(depths, dtype=float)

    valid = np.isfinite(temps) & np.isfinite(depths)

    if valid.sum() < 2:
        return np.nan

    temps_v = temps[valid]
    depths_v = depths[valid]

    # 按深度排序
    order = np.argsort(depths_v)
    depths_v = depths_v[order]
    temps_v = temps_v[order]

    surface_temp = temps_v[0]
    target_temp = surface_temp - threshold

    # 如果全剖面都没有低于阈值温度，则认为混合层达到最大有效观测深度
    if np.nanmin(temps_v) > target_temp:
        return max_depth

    # 若第一个点就已经满足，保守返回最浅观测深度
    if temps_v[0] <= target_temp:
        return depths_v[0]

    # PCHIP: depth -> temperature
    try:
        f = PchipInterpolator(depths_v, temps_v, extrapolate=False)

        # 寻找首次跨越 target_temp 的相邻深度区间
        diff = temps_v - target_temp

        for i in range(len(depths_v) - 1):
            d1, d2 = depths_v[i], depths_v[i + 1]
            y1, y2 = diff[i], diff[i + 1]

            # 正好等于阈值
            if y1 == 0:
                return d1

            # 出现跨越
            if y1 * y2 <= 0:
                root = brentq(lambda z: float(f(z) - target_temp), d1, d2)
                return float(root)

        return max_depth

    except Exception:
        # 若 PCHIP 因异常失败，则退回线性插值
        diff = temps_v - target_temp

        for i in range(len(depths_v) - 1):
            y1, y2 = diff[i], diff[i + 1]

            if y1 * y2 <= 0:
                d1, d2 = depths_v[i], depths_v[i + 1]
                t1, t2 = temps_v[i], temps_v[i + 1]

                if t2 == t1:
                    return d2

                return float(d1 + (target_temp - t1) * (d2 - d1) / (t2 - t1))

        return max_depth


def add_panel_label(ax, label, x=0.005, y=1.01):
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="bold",
        va="bottom",
        ha="left"
    )


def main():
    # ================= 1. 读取数据 =================
    df = load_data(INPUT_FILE)

    time_col = find_time_col(df)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    # 检查五个深度列
    missing_cols = [c for c in TEMP_COLS if c not in df.columns]
    if missing_cols:
        raise KeyError(f"缺少以下水温列：{missing_cols}")

    # 时间截取
    df = df[(df[time_col] >= PLOT_START) & (df[time_col] <= PLOT_END)].copy()

    # 数值转换
    for c in TEMP_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ================= 2. 基于五深度重新计算 MLD =================
    print(">>> 正在基于五个实测深度重新计算 MLD...")

    df[MLD_COL] = df[TEMP_COLS].apply(
        lambda row: calc_mld_pchip_one_profile(
            row.values,
            DEPTHS,
            threshold=MLD_THRESHOLD,
            max_depth=DEPTH_MAX
        ),
        axis=1
    )

    # ================= 3. 日均数据用于 (a)(b) =================
    df_daily = df.set_index(time_col)[TEMP_COLS].resample("D").mean()

    # 导出绘图所用数据
    save_plot_data_csv(df, df_daily, time_col, OUTPUT_STEM)

    X_daily = df_daily.index
    Y = DEPTHS.copy()
    Z = df_daily[TEMP_COLS].to_numpy(dtype=float).T

    surf = df_daily[SURF_COL]
    bot = df_daily[BOT_COL]
    delta_t = surf - bot

    # (c) 原始 MLD 序列
    X_raw = df[time_col]
    mld_raw = df[MLD_COL]

    # 色阶范围
    z_min = np.nanmin(Z)
    z_max = np.nanmax(Z)

    levels = np.linspace(np.floor(z_min), np.ceil(z_max), 80)

    # 色带刻度：5个整数标注
    cbar_ticks = np.linspace(np.floor(z_min), np.ceil(z_max), 5)
    cbar_ticks = np.round(cbar_ticks).astype(int)

    # ================= 4. 创建画布 =================
    fig = plt.figure(figsize=FIGURE_SIZE)

    gs = GridSpec(
        3, 2,
        width_ratios=[40, 0.8],
        height_ratios=[1.0, 1.05, 1.05],
        hspace=0.20,
        wspace=0.04
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)

    cax = fig.add_subplot(gs[1, 1])

    # ================= (a) 表层、底层水温和 ΔT =================
    ax1.plot(
        X_daily,
        surf,
        color=COLOR_SURF,
        linewidth=1.8,
        label="Surface"
    )

    ax1.plot(
        X_daily,
        bot,
        color=COLOR_BOT,
        linewidth=1.8,
        label="Bottom"
    )

    ax1.fill_between(
        X_daily,
        surf,
        bot,
        color="0.75",
        alpha=0.16,
        linewidth=0
    )

    ax1.set_ylabel("Temperature (°C)")

    ax1_r = ax1.twinx()

    ax1_r.plot(
        X_daily,
        delta_t,
        color=COLOR_DT,
        linewidth=1.4,
        linestyle="--",
        label=r"$\Delta T$"
    )

    ax1_r.set_ylabel(r"$\Delta T$ (°C)", color=COLOR_DT)
    ax1_r.tick_params(axis="y", labelcolor=COLOR_DT)

    handles_top = [
        Line2D([0], [0], color=COLOR_SURF, lw=1.8, label="0.25 m"),
        Line2D([0], [0], color=COLOR_BOT, lw=1.8, label="2.25 m"),
        Line2D([0], [0], color=COLOR_DT, lw=1.4, linestyle="--", label=r"$\Delta T$")
    ]

    ax1.legend(
        handles=handles_top,
        loc="upper left",
        frameon=False,
        ncol=3,
        handlelength=1.6,
        columnspacing=0.8,
        borderpad=0.2
    )

    add_panel_label(ax1, "(a)", x=0.0, y=1.02)

    # ================= (b) 水温时深分布 =================
    cf = ax2.contourf(
        X_daily,
        Y,
        Z,
        levels=levels,
        cmap=CMAP_NAME
    )

    ax2.invert_yaxis()
    ax2.set_ylabel("Depth (m)")
    ax2.set_ylim(DEPTH_MAX, DEPTH_MIN)
    ax2.set_yticks([0.25, 0.75, 1.25, 1.75, 2.25])

    add_panel_label(ax2, "(b)", x=0.005, y=1.01)

    cbar = fig.colorbar(
        cf,
        cax=cax,
        ticks=cbar_ticks
    )

    cbar.set_label("Temperature (°C)", fontsize=AXIS_LABEL_FONTSIZE)
    cbar.ax.set_yticklabels([f"{t:d}" for t in cbar_ticks])

    cbar.ax.tick_params(
        labelsize=TICK_LABEL_FONTSIZE,
        length=0
    )

    # ================= (c) MLD =================
    plot_step = 12

    ax3.plot(
        X_raw.iloc[::plot_step],
        mld_raw.iloc[::plot_step],
        color=COLOR_MLD,
        linewidth=0.3,
        alpha=0.75
    )

    ax3.set_ylabel("MLD (m)")
    ax3.set_xlabel("Date")
    ax3.set_ylim(DEPTH_MAX, DEPTH_MIN)
    ax3.set_yticks([0.25, 0.75, 1.25, 1.75, 2.25])

    add_panel_label(ax3, "(c)", x=0.005, y=1.01)

    # ================= X轴统一 =================
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=DATE_TICK_INTERVAL_MONTHS))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter(DATE_TICK_FORMAT))
    plt.setp(ax3.get_xticklabels(), rotation=DATE_TICK_ROTATION, ha="center")
    ax3.tick_params(axis="x", pad=6)

    ax1.tick_params(axis="x", labelbottom=False)
    ax2.tick_params(axis="x", labelbottom=False)

    ax1.set_xlim(pd.Timestamp(PLOT_START), pd.Timestamp(PLOT_END))

    # ================= 统一样式 =================
    for ax in [ax1, ax2, ax3, ax1_r]:
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_zorder(100)

    fig.subplots_adjust(
        left=0.12,
        right=0.92,
        top=0.97,
        bottom=0.10,
        hspace=0.20,
        wspace=0.04
    )

    # ================= 保存 =================
    save_figure_all(
        fig,
        output_stem=OUTPUT_STEM,
        dpi=OUTPUT_DPI
    )
print("[OK] Figure 3 已完成 PNG、TIF、PDF 和 CSV 导出。")


if __name__ == "__main__":
    main()
