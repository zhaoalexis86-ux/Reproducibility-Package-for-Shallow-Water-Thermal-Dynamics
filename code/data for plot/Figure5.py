import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from matplotlib.lines import Line2D

# =========================================================
# Step 2. 读取 Step 1 指标结果并绘制 Figure 4
# =========================================================

# ================= 路径配置区域（仅路径经过公开包适配） =================
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
INPUT_INDICES = REPOSITORY_ROOT / "data" / "raw_or_input" / "full_year" / "figure5_indices_5depth_source.csv"

OUTPUT_DIR = REPOSITORY_ROOT / "results" / "reproduced" / "Figure5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_STEM = OUTPUT_DIR / "Figure5"
OUTPUT_DPI = 1000

# ================= 可直接微调的出版格式参数 =================
FIGURE_SIZE = (7.3, 5.2)       # 保持原图幅比例与尺寸
AXIS_LABEL_FONTSIZE = 15       # 在上一 revised 版基础上再放大 2 pt
TICK_LABEL_FONTSIZE = 12       # 在上一 revised 版基础上再放大 1 pt
LEGEND_FONTSIZE = 12           # 在上一 revised 版基础上再放大 1 pt
PANEL_LABEL_FONTSIZE = 15      # 在上一 revised 版基础上再放大 2 pt
DATE_TICK_INTERVAL_MONTHS = 2  # 保持原月份刻度间隔
DATE_TICK_FORMAT = "%Y-%m"     # 保持原日期格式

TIME_COL = "Date"

PLOT_START = "2024-01-01"
PLOT_END = "2024-12-31 23:59:59"

GAP_START = None
GAP_END = None

# 坐标轴范围
ST_YLIM = (0, 16)
N2_YLIM = (0, 0.045)
LOGRIB_YLIM = (-1.2, 8)

# 平滑设置
RESAMPLE_FREQ = "1h"  # pandas 3 uses lowercase offset aliases
ST_N2_ROLLING_HOURS = 24
RIB_MEDIAN_HOURS = 24

# ================= 字体与风格 =================
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
COLOR_DEEP_BLUE = "#376B95"
COLOR_LIGHT_BLUE = "#72BCD4"

COLOR_INST_GRAY = "#D9D9D9"
LW_INST_GRAY = 0.5
ALPHA_INST_GRAY = 0.8

COLOR_ST = COLOR_RED
COLOR_N2 = COLOR_DEEP_BLUE
COLOR_RIB = COLOR_LIGHT_BLUE


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

    eps_path = output_stem.parent / "Figure5.eps"
    fig.savefig(eps_path, format="eps", bbox_inches="tight")
    print(f"[OK] EPS已保存: {eps_path}")


def save_plot_data_csv(df_hourly, output_stem):
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    csv_path = output_stem.with_suffix(".csv")

    df_out = df_hourly[[
        TIME_COL,
        "St(J/m2)",
        "N2_max(1/s2)",
        "log10_Rib",
        "St_24h",
        "N2_max_24h",
        "log10_Rib_med24h"
    ]].copy()

    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 绘图数据CSV已保存: {csv_path}")


def load_indices(file_path):
    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="gb18030")

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL]).sort_values(TIME_COL).reset_index(drop=True)
    return df


def main():
    print(">>> 正在读取 Figure 5 稳定性指标文件...")
    df = load_indices(INPUT_INDICES)

    start = pd.Timestamp(PLOT_START)
    end = pd.Timestamp(PLOT_END)
    df = df[(df[TIME_COL] >= start) & (df[TIME_COL] <= end)].copy()

    required_cols = ["St(J/m2)", "N2_max(1/s2)", "log10_Rib"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"指标文件缺少以下列：{missing_cols}")

    # ================= 1. 小时重采样和平滑 =================
    df_hourly = (
        df.set_index(TIME_COL)[["St(J/m2)", "N2_max(1/s2)", "log10_Rib"]]
        .resample(RESAMPLE_FREQ)
        .mean()
        .reset_index()
    )

    df_hourly["St_24h"] = (
        df_hourly["St(J/m2)"]
        .rolling(window=ST_N2_ROLLING_HOURS, center=True, min_periods=1)
        .mean()
    )

    df_hourly["N2_max_24h"] = (
        df_hourly["N2_max(1/s2)"]
        .rolling(window=ST_N2_ROLLING_HOURS, center=True, min_periods=1)
        .mean()
    )

    df_hourly["log10_Rib_med24h"] = (
        df_hourly["log10_Rib"]
        .rolling(window=RIB_MEDIAN_HOURS, center=True, min_periods=1)
        .median()
    )

    # 导出绘图所用数据
    save_plot_data_csv(df_hourly, OUTPUT_STEM)

    # ================= 2. 创建画布 =================
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=FIGURE_SIZE,
        sharex=True,
        gridspec_kw={
            "height_ratios": [1.20, 0.82],
            "hspace": 0.15
        }
    )

    # ================= (a) St + N2 =================
    ax1.plot(
        df_hourly[TIME_COL],
        df_hourly["St_24h"],
        color=COLOR_ST,
        linewidth=1.8
    )
    ax1.set_ylabel("Schmidt stability (J/m²)", color=COLOR_ST, labelpad=6)
    ax1.tick_params(axis="y", labelcolor=COLOR_ST)
    ax1.set_ylim(*ST_YLIM)

    ax1_r = ax1.twinx()
    ax1_r.plot(
        df_hourly[TIME_COL],
        df_hourly["N2_max_24h"],
        color=COLOR_N2,
        linewidth=1.8,
        linestyle="--"
    )
    # 数学排版可避免部分本地字体缺少 Unicode 上标负号字形；显示结果为 s⁻²。
    ax1_r.set_ylabel(r"Maximum $N^2$ (s$^{-2}$)", color=COLOR_N2, labelpad=6)
    ax1_r.tick_params(axis="y", labelcolor=COLOR_N2)
    ax1_r.set_ylim(*N2_YLIM)

    handles_top = [
        Line2D([0], [0], color=COLOR_ST, lw=1.8, label="Schmidt stability"),
        Line2D([0], [0], color=COLOR_N2, lw=1.8, linestyle="--", label=r"Maximum $N^2$")
    ]
    ax1.legend(
        handles=handles_top,
        loc="upper left",
        bbox_to_anchor=(0.00, 1.02),
        frameon=False,
        ncol=1,
        handlelength=1.8,
        borderpad=0.1,
        labelspacing=0.20
    )

    ax1.text(
        0.0, 1.02, "(a)",
        transform=ax1.transAxes,
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="bold",
        va="bottom",
        ha="left"
    )

    # ================= (b) log10(Rib) =================
    ax2.plot(
        df_hourly[TIME_COL],
        df_hourly["log10_Rib"],
        color=COLOR_INST_GRAY,
        linewidth=LW_INST_GRAY,
        alpha=ALPHA_INST_GRAY,
        zorder=1
    )

    ax2.plot(
        df_hourly[TIME_COL],
        df_hourly["log10_Rib_med24h"],
        color=COLOR_RIB,
        linewidth=1.8,
        alpha=0.95,
        zorder=2
    )

    ax2.set_ylabel(r"$\log_{10}(Rib)$")
    ax2.set_xlabel("Date")
    ax2.set_ylim(*LOGRIB_YLIM)

    ax2.text(
        0.0, 1.02, "(b)",
        transform=ax2.transAxes,
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="bold",
        va="bottom",
        ha="left"
    )

    # ================= 可选空档期遮罩 =================
    if GAP_START is not None and GAP_END is not None:
        gap_start = pd.Timestamp(GAP_START)
        gap_end = pd.Timestamp(GAP_END)
        for ax in [ax1, ax2]:
            ax.axvspan(gap_start, gap_end, color="white", zorder=5)

    # ================= X 轴格式 =================
    ax2.set_xlim(start, end)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=DATE_TICK_INTERVAL_MONTHS))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter(DATE_TICK_FORMAT))
    plt.setp(ax2.get_xticklabels(), rotation=0, ha="center")

    # ================= 统一样式 =================
    for ax in [ax1, ax2, ax1_r]:
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_zorder(100)

    plt.tight_layout()

    save_figure_all(
        fig,
        output_stem=OUTPUT_STEM,
        dpi=OUTPUT_DPI
    )
print("[OK] Figure 5 已完成 PNG、TIF、PDF 和 CSV 导出。")


if __name__ == "__main__":
    main()
