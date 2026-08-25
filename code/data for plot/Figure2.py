import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ================= 路径配置区域（仅路径经过公开包适配） =================
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
# Private full-year input. The public example demonstrates its schema only.
INPUT_FILE = REPOSITORY_ROOT / "data" / "raw_or_input" / "full_year" / "figure2_meteorology_full_source.csv"

OUTPUT_DIR = REPOSITORY_ROOT / "results" / "reproduced" / "Figure2"
OUTPUT_STEM = OUTPUT_DIR / "Figure2"

MASK_START = "2025-08-31 00:00"
MASK_END = "2025-09-08 23:50"

PLOT_START = "2024-01-01"
PLOT_END = "2024-12-31"

DAILY_FREQ = "D"
OUTPUT_DPI = 1000

# ================= 可直接微调的出版格式参数 =================
FIGURE_SIZE = (7.3, 8.4)       # 保持原图幅比例与尺寸
AXIS_LABEL_FONTSIZE = 15       # 最终页面缩放下保持清晰
TICK_LABEL_FONTSIZE = 12       # 在上一 revised 版基础上再放大 1 pt
LEGEND_FONTSIZE = 12           # 在上一 revised 版基础上再放大 1 pt
PANEL_LABEL_FONTSIZE = 15      # 在上一 revised 版基础上再放大 2 pt
DATE_TICK_INTERVAL_MONTHS = 2  # 保持原月份刻度间隔
DATE_TICK_FORMAT = "%Y-%m"     # 保持原日期格式

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
    "ps.fonttype": 42
})

# ================= 配色 =================
COLOR_AIR = "#E76254"          # 气温
COLOR_RAD = "#EFAA58"          # 辐射
COLOR_RH = "#72BCD5"           # 湿度
COLOR_WIND = "#376795"         # 风速

COLOR_AIR_FILL = "#F4A59C"
COLOR_RAD_FILL = "#F7D06F"
COLOR_RH_FILL = "#AADCE0"
COLOR_WIND_FILL = "#A7C4DD"


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

    eps_path = output_stem.parent / "Figure2.eps"
    fig.savefig(eps_path, format="eps", bbox_inches="tight")
    print(f"[OK] EPS已保存: {eps_path}")


def save_plot_data_csv(df_plot, output_stem):
    output_stem = Path(output_stem)
    csv_path = output_stem.with_suffix(".csv")
    df_to_save = df_plot.copy()
    df_to_save.index.name = "Date"
    df_to_save.to_csv(csv_path, encoding="utf-8-sig")
    print(f"[OK] 绘图数据CSV已保存: {csv_path}")


def find_column(columns, include_keywords, exclude_keywords=None):
    exclude_keywords = exclude_keywords or []
    for c in columns:
        if all(k in c for k in include_keywords) and all(k not in c for k in exclude_keywords):
            return c
    raise KeyError(f"未找到符合条件的列：include={include_keywords}, exclude={exclude_keywords}")


def load_data(file_path):
    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True, encoding="gb18030")
    return df.sort_index()


def prepare_daily_series(df, col_air, col_rh, col_wind, col_rad):
    df = df.copy()

    # 风速故障期置空；当前绘制2024年，通常不会影响
    df.loc[MASK_START:MASK_END, col_wind] = np.nan

    daily = pd.DataFrame({
        "air_mean": df[col_air].resample(DAILY_FREQ).mean(),
        "air_max":  df[col_air].resample(DAILY_FREQ).max(),
        "air_min":  df[col_air].resample(DAILY_FREQ).min(),

        "rad_mean": df[col_rad].resample(DAILY_FREQ).mean(),
        "rad_max":  df[col_rad].resample(DAILY_FREQ).max(),
        "rad_min":  df[col_rad].resample(DAILY_FREQ).min(),

        "rh_mean": df[col_rh].resample(DAILY_FREQ).mean(),
        "rh_max":  df[col_rh].resample(DAILY_FREQ).max(),
        "rh_min":  df[col_rh].resample(DAILY_FREQ).min(),

        "wind_mean": df[col_wind].resample(DAILY_FREQ).mean(),
        "wind_max":  df[col_wind].resample(DAILY_FREQ).max(),
        "wind_min":  df[col_wind].resample(DAILY_FREQ).min()
    })

    return daily


def add_panel_label(ax, label):
    ax.text(
        0.0, 1.02, label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="bold",
        va="bottom",
        ha="left"
    )


def add_simple_legend(ax, line_color, fill_color, loc="upper left"):
    handles = [
        Patch(
            facecolor=fill_color,
            edgecolor="none",
            alpha=0.32,
            label="Daily max–min range"
        ),
        Line2D(
            [0], [0],
            color=line_color,
            lw=1.5,
            label="Daily mean"
        )
    ]

    ax.legend(
        handles=handles,
        loc=loc,
        frameon=False,
        ncol=2,
        handlelength=1.6,
        columnspacing=0.8,
        borderpad=0.2
    )


def plot_daily_range_panel(ax, x, y_mean, y_min, y_max, line_color, fill_color, ylabel, label, ylim=None, legend_loc="upper left"):
    ax.fill_between(
        x,
        y_min.values,
        y_max.values,
        color=fill_color,
        alpha=0.32,
        linewidth=0
    )

    ax.plot(
        x,
        y_mean.values,
        color=line_color,
        linewidth=1.5
    )

    ax.set_ylabel(ylabel, labelpad=8, fontsize=12)

    if ylim is not None:
        ax.set_ylim(*ylim)

    add_simple_legend(ax, line_color, fill_color, loc=legend_loc)
    add_panel_label(ax, label)


def plot_meteorological_forcing():
    print(">>> 正在绘制 Figure 2: Meteorological forcing with four panels ...")

    df = load_data(INPUT_FILE)

    col_air = find_column(df.columns, include_keywords=["气温"])
    col_rh = find_column(df.columns, include_keywords=["湿度"])
    col_wind = find_column(df.columns, include_keywords=["风速"])
    col_rad = find_column(df.columns, include_keywords=["辐射"], exclude_keywords=["累计"])

    df = df.loc[PLOT_START:PLOT_END].copy()
    daily = prepare_daily_series(df, col_air, col_rh, col_wind, col_rad)

    # 导出绘图所用数据
    save_plot_data_csv(daily, OUTPUT_STEM)

    # ================= 画布 =================
    fig, axes = plt.subplots(
        4, 1,
        figsize=FIGURE_SIZE,
        sharex=True,
        gridspec_kw={
            "height_ratios": [1, 1, 1, 1],
            "hspace": 0.2
        }
    )

    ax1, ax2, ax3, ax4 = axes

    # ================= (a) 气温 =================
    air_min = np.nanmin(daily["air_min"].values)
    air_max = np.nanmax(daily["air_max"].values)
    air_ylim = (
        np.floor(air_min / 5) * 5,
        np.ceil(air_max / 5) * 5
    )

    plot_daily_range_panel(
        ax=ax1,
        x=daily.index,
        y_mean=daily["air_mean"],
        y_min=daily["air_min"],
        y_max=daily["air_max"],
        line_color=COLOR_AIR,
        fill_color=COLOR_AIR_FILL,
        ylabel=r"Air temperature ($^\circ$C)",
        label="(a)",
        ylim=air_ylim,
        legend_loc="upper left"
    )

    # ================= (b) 辐射 =================
    rad_max = np.nanmax(daily["rad_max"].values)
    rad_ylim = (0, rad_max * 1.12)

    plot_daily_range_panel(
        ax=ax2,
        x=daily.index,
        y_mean=daily["rad_mean"],
        y_min=daily["rad_min"],
        y_max=daily["rad_max"],
        line_color=COLOR_RAD,
        fill_color=COLOR_RAD_FILL,
        ylabel=r"Solar radiation (W/m$^{2}$)",
        label="(b)",
        ylim=rad_ylim,
        legend_loc="upper left"
    )

    # ================= (c) 湿度 =================
    plot_daily_range_panel(
        ax=ax3,
        x=daily.index,
        y_mean=daily["rh_mean"],
        y_min=daily["rh_min"],
        y_max=daily["rh_max"],
        line_color=COLOR_RH,
        fill_color=COLOR_RH_FILL,
        ylabel="Relative humidity (%)",
        label="(c)",
        ylim=(0, 100),
        legend_loc="lower left"
    )

    # ================= (d) 风速 =================
    wind_max = np.nanmax(daily["wind_max"].values)
    wind_ylim = (0, wind_max * 1.15)

    plot_daily_range_panel(
        ax=ax4,
        x=daily.index,
        y_mean=daily["wind_mean"],
        y_min=daily["wind_min"],
        y_max=daily["wind_max"],
        line_color=COLOR_WIND,
        fill_color=COLOR_WIND_FILL,
        ylabel=r"Wind speed (m/s)",
        label="(d)",
        ylim=wind_ylim,
        legend_loc="upper left"
    )

    ax4.set_xlabel("Date", labelpad=10)

    # Restore the original major-tick density on all four y axes.
    ax1.yaxis.set_major_locator(MultipleLocator(10))
    ax2.yaxis.set_major_locator(MultipleLocator(250))
    ax3.yaxis.set_major_locator(MultipleLocator(20))
    ax4.yaxis.set_major_locator(MultipleLocator(2.5))

    # ================= X轴格式 =================
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=DATE_TICK_INTERVAL_MONTHS))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter(DATE_TICK_FORMAT))
    plt.setp(ax4.get_xticklabels(), rotation=0, ha="center")
    ax4.tick_params(axis="x", pad=6)

    # ================= 统一左轴标签位置 =================
    for ax in axes:
        ax.yaxis.set_label_coords(-0.075, 0.5)

    # ================= 统一样式 =================
    for ax in fig.get_axes():
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_zorder(100)

    ax1.set_xlim(daily.index.min(), daily.index.max())

    fig.subplots_adjust(
        left=0.15,
        right=0.97,
        top=0.975,
        bottom=0.12,
        hspace=0.2
    )

    save_figure_all(
        fig,
        output_stem=OUTPUT_STEM,
        dpi=OUTPUT_DPI
    )
print("[OK] Figure 2 四幅子图版本已完成。")


if __name__ == "__main__":
    plot_meteorological_forcing()
