"""
visualization.py
================
Publication-quality visualisation module for the Carhart Four-Factor Fund
Analysis Pipeline.

Responsibility
--------------
Generates a full set of charts and tables from all upstream pipeline outputs.
Every chart is saved as a high-resolution PNG (300 dpi) to ``outputs/figures/``.
All charts follow a consistent dark-themed, premium design system.

Chart inventory
---------------
  §1  plot_nav_comparison          — Wealth index curves (Active vs Passive)
  §2  plot_return_distributions    — Violin + box plots of monthly returns
  §3  plot_alpha_distribution      — Alpha histogram with BH-FDR significance markers
  §4  plot_factor_time_series      — Market factor (MF) and WML over time
  §5  plot_sharpe_heatmap          — Sharpe ratios as annotated heatmap by fund
  §6  plot_er_vs_alpha             — Scatter: expense ratio vs annualised alpha
  §7  plot_risk_return_scatter     — Ann. Volatility vs CAGR bubble chart
  §8  plot_capture_ratios          — Up/Down-capture bar chart (Active vs Passive)
  §9  plot_sub_period_performance  — Grouped bar: Pre-COVID / COVID / Post-COVID CAGR
  §10 plot_drawdown_comparison     — Max drawdown bar chart ranked by depth
  §11 plot_rolling_sharpe          — 12-month rolling Sharpe for representative funds
  §12 plot_alpha_significance_grid — Grid of fund alphas with CI bars
  §13 generate_all_charts          — Orchestrator: renders and saves all charts
  §14 Self-test (python src/visualization.py)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for server/script runs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Path fix for direct execution ─────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    COL_ALPHA,
    COL_CAGR,
    COL_DATE,
    COL_ER,
    COL_EXCESS_RETURN,
    COL_FUND_NAME,
    COL_FUND_TYPE,
    COL_MF,
    COL_NAV,
    COL_P_ALPHA,
    COL_RETURN,
    COL_SHARPE,
    COL_WML,
    FIGURES_DIR,
    FUND_TYPE_ACTIVE,
    FUND_TYPE_PASSIVE,
    MONTHS_PER_YEAR,
)
from src.utils import ensure_dir, setup_logger

logger = setup_logger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

# Colour palette
_C_ACTIVE   = "#4FC3F7"   # sky blue   — Active funds
_C_PASSIVE  = "#FF8A65"   # warm coral — Passive funds
_C_ACCENT   = "#A5D6A7"   # mint green — accent / third category
_C_GOLD     = "#FFD54F"   # amber gold — highlights
_C_SIG      = "#EF5350"   # vivid red  — statistical significance markers
_C_GRID     = "#2A2A3E"   # subtle grid
_C_BG       = "#0F0F1A"   # near-black background
_C_SURFACE  = "#1A1A2E"   # card surface
_C_TEXT     = "#E0E0E0"   # primary text
_C_SUBTEXT  = "#9E9E9E"   # secondary text

_DPI         = 300
_FONT_FAMILY = "DejaVu Sans"
_ALPHA_SIG   = 0.05        # significance level for markers

def _apply_style(fig: plt.Figure, ax_list: list[plt.Axes]) -> None:
    """Apply the dark premium design system to a figure and its axes."""
    fig.patch.set_facecolor(_C_BG)
    for ax in ax_list:
        ax.set_facecolor(_C_SURFACE)
        ax.tick_params(colors=_C_TEXT, labelsize=9)
        ax.xaxis.label.set_color(_C_TEXT)
        ax.yaxis.label.set_color(_C_TEXT)
        ax.title.set_color(_C_TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(_C_GRID)
        ax.grid(True, color=_C_GRID, linewidth=0.5, alpha=0.7, linestyle="--")
        ax.set_axisbelow(True)


def _save(fig: plt.Figure, filename: str, tight: bool = True) -> Path:
    """Save figure to the figures directory and close it."""
    ensure_dir(FIGURES_DIR)
    path = FIGURES_DIR / filename
    if tight:
        fig.savefig(path, dpi=_DPI, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    else:
        fig.savefig(path, dpi=_DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"  Saved: {filename}")
    return path


def _legend(ax: plt.Axes, **kwargs) -> None:
    """Styled legend for dark background."""
    leg = ax.legend(
        facecolor=_C_SURFACE, edgecolor=_C_GRID,
        labelcolor=_C_TEXT, fontsize=9, **kwargs
    )
    leg.get_frame().set_alpha(0.9)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — WEALTH INDEX COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def plot_nav_comparison(
    enriched_df: pd.DataFrame,
    n_active: int = 5,
    n_passive: int = 5,
) -> Path:
    """
    Plot NAV-normalised cumulative wealth index for representative funds.

    Shows the top-N active and top-N passive funds by Sharpe Ratio, with their
    equal-weighted group average overlaid as a thick reference line.

    Parameters
    ----------
    enriched_df : pd.DataFrame
        Enriched master frame with Date, Fund_Name, Fund_Type, Cumulative_Return.
    n_active, n_passive : int
        Number of individual funds to show per group.
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    _apply_style(fig, [ax])

    for ft, color, label, n in [
        (FUND_TYPE_ACTIVE,  _C_ACTIVE,  "Active",  n_active),
        (FUND_TYPE_PASSIVE, _C_PASSIVE, "Passive", n_passive),
    ]:
        sub = enriched_df[enriched_df[COL_FUND_TYPE] == ft].copy()
        sub[COL_DATE] = pd.to_datetime(sub[COL_DATE])

        # Compute equal-weighted group average wealth index per month
        group_avg = (
            sub.groupby(COL_DATE)["Cumulative_Return"]
            .mean()
            .sort_index()
        )

        # Individual fund lines (thin, semi-transparent)
        fund_sharpes = (
            sub.groupby(COL_FUND_NAME)[COL_RETURN]
            .apply(lambda r: r.mean() / r.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR)
                   if r.std(ddof=1) > 0 else 0)
            .nlargest(n)
            .index.tolist()
        )
        for fund in fund_sharpes:
            fd = sub[sub[COL_FUND_NAME] == fund].sort_values(COL_DATE)
            ax.plot(fd[COL_DATE], fd["Cumulative_Return"],
                    color=color, linewidth=0.8, alpha=0.35)

        # Group average — thick
        ax.plot(group_avg.index, group_avg.values,
                color=color, linewidth=2.8, label=f"{label} (avg)", zorder=5)

    # Annotations
    ax.axhline(1.0, color=_C_SUBTEXT, linewidth=0.8, linestyle=":", alpha=0.8)
    ax.set_title("Cumulative Wealth Index — Active vs Passive Funds (2013–2026)",
                 fontsize=14, fontweight="bold", pad=14, color=_C_TEXT)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Wealth Index (₹1 invested = 1.0)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}×"))
    _legend(ax, loc="upper left")

    return _save(fig, "01_wealth_index_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — RETURN DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_return_distributions(enriched_df: pd.DataFrame) -> Path:
    """
    Side-by-side violin + box plots of monthly return distributions for
    Active vs Passive funds.
    """
    act_rets = enriched_df[enriched_df[COL_FUND_TYPE] == FUND_TYPE_ACTIVE][COL_RETURN].dropna() * 100
    pas_rets = enriched_df[enriched_df[COL_FUND_TYPE] == FUND_TYPE_PASSIVE][COL_RETURN].dropna() * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    _apply_style(fig, list(axes))

    for ax, data, color, label in [
        (axes[0], act_rets, _C_ACTIVE,  FUND_TYPE_ACTIVE),
        (axes[1], pas_rets, _C_PASSIVE, FUND_TYPE_PASSIVE),
    ]:
        parts = ax.violinplot(data, positions=[0], showmedians=False,
                              showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_edgecolor(_C_TEXT)
            pc.set_alpha(0.45)

        bp = ax.boxplot(data, positions=[0], widths=0.15,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color=_C_GOLD, linewidth=2),
                        boxprops=dict(facecolor=color, alpha=0.7,
                                      edgecolor=_C_TEXT),
                        whiskerprops=dict(color=_C_TEXT, linewidth=1.2),
                        capprops=dict(color=_C_TEXT, linewidth=1.5))

        # Key stats
        ax.axhline(data.mean(), color=_C_GOLD, linewidth=1.5,
                   linestyle="--", alpha=0.9, label=f"Mean: {data.mean():.2f}%")
        ax.axhline(0, color=_C_SUBTEXT, linewidth=0.8, linestyle=":",
                   alpha=0.8)

        ax.set_title(f"{label}\nMonthly Return Distribution",
                     fontsize=12, fontweight="bold", color=_C_TEXT)
        ax.set_ylabel("Monthly Return (%)", fontsize=10)
        ax.set_xticks([])
        _legend(ax, loc="upper right")

        # Stats box
        stats_text = (
            f"N = {len(data):,}\n"
            f"Mean = {data.mean():.2f}%\n"
            f"Std  = {data.std():.2f}%\n"
            f"Min  = {data.min():.1f}%\n"
            f"Max  = {data.max():.1f}%"
        )
        ax.text(0.97, 0.03, stats_text, transform=ax.transAxes,
                fontsize=8.5, color=_C_TEXT, ha="right", va="bottom",
                bbox=dict(facecolor=_C_BG, edgecolor=_C_GRID,
                          boxstyle="round,pad=0.4", alpha=0.85))

    fig.suptitle("Monthly Return Distributions — Active vs Passive Funds",
                 fontsize=14, fontweight="bold", color=_C_TEXT, y=1.01)
    fig.tight_layout()
    return _save(fig, "02_return_distributions.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ALPHA DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

def plot_alpha_distribution(
    reg_results_df: pd.DataFrame,
    stat_tests_df: Optional[pd.DataFrame] = None,
) -> Path:
    """
    Histogram of annualised alpha for Active and Passive funds, with vertical
    significance markers and BH-FDR corrected significance colouring.
    """
    ok = reg_results_df[reg_results_df["Regression_Status"] == "OK"].copy()
    ok["Alpha_Ann_Pct"] = ((1 + ok[COL_ALPHA]) ** MONTHS_PER_YEAR - 1) * 100

    # Merge BH-FDR significance if available
    if stat_tests_df is not None and "Sig_BH_FDR" in stat_tests_df.columns:
        ok = ok.merge(
            stat_tests_df[[COL_FUND_NAME, "Sig_BH_FDR"]],
            on=COL_FUND_NAME, how="left"
        )
    else:
        ok["Sig_BH_FDR"] = ok[COL_P_ALPHA] < _ALPHA_SIG

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _apply_style(fig, list(axes))

    for ax, ft, color in [
        (axes[0], FUND_TYPE_ACTIVE,  _C_ACTIVE),
        (axes[1], FUND_TYPE_PASSIVE, _C_PASSIVE),
    ]:
        sub    = ok[ok[COL_FUND_TYPE] == ft]
        alphas = sub["Alpha_Ann_Pct"]
        sig    = sub.get("Sig_BH_FDR", pd.Series([False] * len(sub)))

        bins = np.linspace(alphas.min() - 0.5, alphas.max() + 0.5, 18)

        # Non-significant bars
        ax.hist(alphas[~sig], bins=bins, color=color, alpha=0.55,
                edgecolor=_C_TEXT, linewidth=0.5, label="Not sig. (BH-FDR)")
        # Significant bars
        ax.hist(alphas[sig], bins=bins, color=_C_GOLD, alpha=0.85,
                edgecolor=_C_TEXT, linewidth=0.5, label="Sig. (BH-FDR)")

        ax.axvline(0, color=_C_SUBTEXT, linewidth=1.2,
                   linestyle="--", alpha=0.8)
        ax.axvline(alphas.mean(), color=_C_SIG, linewidth=2,
                   linestyle="-", alpha=0.9,
                   label=f"Mean: {alphas.mean():.2f}%")

        ax.set_title(f"{ft} Funds — Annualised Alpha Distribution",
                     fontsize=11, fontweight="bold", color=_C_TEXT)
        ax.set_xlabel("Alpha (% per annum)", fontsize=10)
        ax.set_ylabel("Number of Funds", fontsize=10)
        _legend(ax, loc="upper right")

    fig.suptitle("Jensen's Alpha Distribution — BH-FDR Significance Highlighted",
                 fontsize=13, fontweight="bold", color=_C_TEXT, y=1.02)
    fig.tight_layout()
    return _save(fig, "03_alpha_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — FACTOR TIME SERIES
# ══════════════════════════════════════════════════════════════════════════════

def plot_factor_time_series(reg_input_df: pd.DataFrame) -> Path:
    """
    Time series of the market factor (MF) and momentum factor (WML) with
    a 12-month rolling mean overlay.
    """
    df = reg_input_df.copy()
    df[COL_DATE] = pd.to_datetime(df[COL_DATE])
    factor_monthly = (
        df.groupby(COL_DATE)[[COL_MF, COL_WML, "RF"]]
        .mean()
        .sort_index()
    )
    factor_monthly *= 100   # → percentage

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    _apply_style(fig, list(axes))

    spec = [
        (axes[0], COL_MF,   "Market Factor MF (Rm − Rf)",   _C_ACTIVE),
        (axes[1], COL_WML,  "Momentum Factor WML",            _C_PASSIVE),
        (axes[2], "RF",      "Risk-Free Rate RF (monthly)",    _C_ACCENT),
    ]

    for ax, col, title, color in spec:
        if col not in factor_monthly.columns:
            continue
        vals = factor_monthly[col]
        roll = vals.rolling(12, min_periods=6).mean()

        ax.fill_between(vals.index, vals.values, 0,
                        where=(vals >= 0), alpha=0.35,
                        color=color, interpolate=True)
        ax.fill_between(vals.index, vals.values, 0,
                        where=(vals < 0), alpha=0.35,
                        color=_C_SIG, interpolate=True)
        ax.plot(vals.index, vals.values, color=color,
                linewidth=0.9, alpha=0.7, label="Monthly")
        ax.plot(roll.index, roll.values, color=_C_GOLD,
                linewidth=2.0, label="12M rolling avg")
        ax.axhline(0, color=_C_SUBTEXT, linewidth=0.8,
                   linestyle=":", alpha=0.8)
        ax.set_title(title, fontsize=10, fontweight="bold",
                     color=_C_TEXT, pad=6)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
        _legend(ax, loc="upper right")

    axes[-1].set_xlabel("Date", fontsize=10)
    fig.suptitle("Synthetic Factor Time Series (2013–2026)",
                 fontsize=13, fontweight="bold", color=_C_TEXT, y=1.01)
    fig.tight_layout()
    return _save(fig, "04_factor_time_series.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — SHARPE RATIO HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

def plot_sharpe_heatmap(metrics_df: pd.DataFrame) -> Path:
    """
    Horizontal bar chart of Sharpe ratios for all funds, sorted within each
    type, colour-coded by fund type.
    """
    df = metrics_df[[COL_FUND_NAME, COL_FUND_TYPE, COL_SHARPE, COL_ER]].dropna(
        subset=[COL_SHARPE]
    ).copy()
    df = df.sort_values([COL_FUND_TYPE, COL_SHARPE], ascending=[True, True])

    short_names = [
        n.replace(" - Direct Plan", "").replace(" Direct Plan", "")
         .replace(" - Growth", "").replace(" Growth", "")[:45]
        for n in df[COL_FUND_NAME]
    ]

    colors = [_C_ACTIVE if ft == FUND_TYPE_ACTIVE else _C_PASSIVE
              for ft in df[COL_FUND_TYPE]]

    fig, ax = plt.subplots(figsize=(13, max(10, len(df) * 0.32)))
    _apply_style(fig, [ax])

    bars = ax.barh(range(len(df)), df[COL_SHARPE].values,
                   color=colors, edgecolor=_C_BG, linewidth=0.5, height=0.75)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(short_names, fontsize=7.5, color=_C_TEXT)
    ax.axvline(0, color=_C_SUBTEXT, linewidth=0.8, linestyle="--", alpha=0.7)

    # Type separator line
    n_passive = (df[COL_FUND_TYPE] == FUND_TYPE_PASSIVE).sum()
    ax.axhline(n_passive - 0.5, color=_C_GOLD, linewidth=1.2,
               linestyle="--", alpha=0.8)
    ax.text(ax.get_xlim()[1] * 0.98, n_passive - 0.5,
            "  ← Active / Passive →  ",
            ha="right", va="center", fontsize=8,
            color=_C_GOLD, fontweight="bold")

    # Patch legend
    act_patch = mpatches.Patch(color=_C_ACTIVE,  label=FUND_TYPE_ACTIVE)
    pas_patch = mpatches.Patch(color=_C_PASSIVE, label=FUND_TYPE_PASSIVE)
    _legend(ax, handles=[act_patch, pas_patch], loc="lower right")

    ax.set_xlabel("Annualised Sharpe Ratio", fontsize=11)
    ax.set_title("Sharpe Ratios — All Funds (Sorted by Type)",
                 fontsize=13, fontweight="bold", color=_C_TEXT, pad=12)

    return _save(fig, "05_sharpe_ratio_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — EXPENSE RATIO VS ALPHA SCATTER
# ══════════════════════════════════════════════════════════════════════════════

def plot_er_vs_alpha(
    reg_results_df: pd.DataFrame,
    stat_tests_df: Optional[pd.DataFrame] = None,
) -> Path:
    """
    Scatter plot of Expense Ratio vs Annualised Alpha.
    Fits a linear trend line per fund type.  Significant funds (BH-FDR)
    are plotted with a bright gold border.
    """
    import scipy.stats as scipy_stats

    ok = reg_results_df[reg_results_df["Regression_Status"] == "OK"].copy()
    ok["Alpha_Ann_Pct"] = ((1 + ok[COL_ALPHA]) ** MONTHS_PER_YEAR - 1) * 100

    if stat_tests_df is not None and "Sig_BH_FDR" in stat_tests_df.columns:
        ok = ok.merge(stat_tests_df[[COL_FUND_NAME, "Sig_BH_FDR"]],
                      on=COL_FUND_NAME, how="left")
    else:
        ok["Sig_BH_FDR"] = ok[COL_P_ALPHA] < _ALPHA_SIG

    fig, ax = plt.subplots(figsize=(12, 7))
    _apply_style(fig, [ax])

    for ft, color, marker in [
        (FUND_TYPE_ACTIVE,  _C_ACTIVE,  "o"),
        (FUND_TYPE_PASSIVE, _C_PASSIVE, "s"),
    ]:
        sub = ok[ok[COL_FUND_TYPE] == ft]

        # Non-significant
        ns = sub[~sub["Sig_BH_FDR"]]
        ax.scatter(ns[COL_ER], ns["Alpha_Ann_Pct"], color=color,
                   marker=marker, s=60, alpha=0.6, edgecolors=_C_BG,
                   linewidths=0.5, label=f"{ft} (not sig.)")

        # Significant
        sg = sub[sub["Sig_BH_FDR"]]
        ax.scatter(sg[COL_ER], sg["Alpha_Ann_Pct"], color=color,
                   marker=marker, s=100, alpha=0.95, edgecolors=_C_GOLD,
                   linewidths=1.5, label=f"{ft} (sig. BH-FDR)")

        # OLS trend line
        xy = sub[[COL_ER, "Alpha_Ann_Pct"]].dropna()
        if len(xy) >= 3:
            slope, intercept, r, p, _ = scipy_stats.linregress(
                xy[COL_ER], xy["Alpha_Ann_Pct"]
            )
            x_range = np.linspace(xy[COL_ER].min(), xy[COL_ER].max(), 80)
            ax.plot(x_range, intercept + slope * x_range,
                    color=color, linewidth=1.8, linestyle="--", alpha=0.85)
            ax.text(xy[COL_ER].max(), intercept + slope * xy[COL_ER].max(),
                    f"  ρ={r:.2f}", color=color, fontsize=8.5, va="center")

    ax.axhline(0, color=_C_SUBTEXT, linewidth=0.9,
               linestyle=":", alpha=0.8)
    ax.set_xlabel("Expense Ratio (% p.a.)", fontsize=11)
    ax.set_ylabel("Jensen's Alpha (% p.a.)", fontsize=11)
    ax.set_title("Expense Ratio vs Annualised Alpha — Nanigian (2019) Test",
                 fontsize=13, fontweight="bold", color=_C_TEXT, pad=12)
    _legend(ax, loc="upper left")

    return _save(fig, "06_er_vs_alpha_scatter.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — RISK-RETURN BUBBLE CHART
# ══════════════════════════════════════════════════════════════════════════════

def plot_risk_return_scatter(metrics_df: pd.DataFrame) -> Path:
    """
    Annualised Volatility vs CAGR bubble chart.
    Bubble size = |Max Drawdown|, colour = fund type.
    """
    df = metrics_df[
        [COL_FUND_NAME, COL_FUND_TYPE, "Annualized_Volatility",
         COL_CAGR, "Max_Drawdown", COL_ER]
    ].dropna().copy()

    df["vol_pct"]  = df["Annualized_Volatility"] * 100
    df["cagr_pct"] = df[COL_CAGR] * 100
    df["mdd_size"] = df["Max_Drawdown"].abs() * 1200   # scale for bubble

    fig, ax = plt.subplots(figsize=(13, 7))
    _apply_style(fig, [ax])

    for ft, color in [(FUND_TYPE_ACTIVE, _C_ACTIVE),
                      (FUND_TYPE_PASSIVE, _C_PASSIVE)]:
        sub = df[df[COL_FUND_TYPE] == ft]
        ax.scatter(sub["vol_pct"], sub["cagr_pct"],
                   s=sub["mdd_size"], c=color, alpha=0.65,
                   edgecolors=_C_TEXT, linewidths=0.6,
                   label=ft, zorder=3)

        # Label top performers
        top = sub.nlargest(3, "cagr_pct")
        for _, row in top.iterrows():
            short_name = row[COL_FUND_NAME].split("-")[0].strip()[:20]
            ax.annotate(short_name,
                        xy=(row["vol_pct"], row["cagr_pct"]),
                        xytext=(6, 4), textcoords="offset points",
                        fontsize=7, color=_C_TEXT, alpha=0.85)

    ax.set_xlabel("Annualised Volatility (%)", fontsize=11)
    ax.set_ylabel("CAGR (%)", fontsize=11)
    ax.set_title("Risk-Return Profile — Bubble size = |Max Drawdown|",
                 fontsize=13, fontweight="bold", color=_C_TEXT, pad=12)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    _legend(ax, loc="lower right")

    return _save(fig, "07_risk_return_scatter.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — CAPTURE RATIOS
# ══════════════════════════════════════════════════════════════════════════════

def plot_capture_ratios(metrics_df: pd.DataFrame) -> Path:
    """
    Grouped bar chart of Up-Capture and Down-Capture ratios (mean per type).
    Ideal region: Up-Capture > 100%, Down-Capture < 100%.
    """
    cols = ["Up_Capture", "Down_Capture"]
    if not all(c in metrics_df.columns for c in cols):
        logger.warning("  Capture ratio columns missing — skipping chart.")
        return FIGURES_DIR / "08_capture_ratios_SKIPPED.png"

    summary = (
        metrics_df.groupby(COL_FUND_TYPE)[cols]
        .mean()
        .reindex([FUND_TYPE_ACTIVE, FUND_TYPE_PASSIVE])
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_style(fig, [ax])

    x = np.arange(2)   # Up / Down
    w = 0.30
    colors = [_C_ACTIVE, _C_PASSIVE]

    for i, (ft, color) in enumerate(
        [(FUND_TYPE_ACTIVE, _C_ACTIVE), (FUND_TYPE_PASSIVE, _C_PASSIVE)]
    ):
        vals = summary.loc[ft, cols].values
        bars = ax.bar(x + (i - 0.5) * w, vals, width=w,
                      color=color, alpha=0.85, edgecolor=_C_BG,
                      linewidth=0.5, label=ft, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{val:.1f}%", ha="center", va="bottom",
                    fontsize=9, color=_C_TEXT, fontweight="bold")

    ax.axhline(100, color=_C_GOLD, linewidth=1.5,
               linestyle="--", alpha=0.9, label="100% benchmark")
    ax.set_xticks(x)
    ax.set_xticklabels(["Up-Market Capture", "Down-Market Capture"],
                       fontsize=11, color=_C_TEXT)
    ax.set_ylabel("Capture Ratio (%)", fontsize=11)
    ax.set_title("Market Capture Ratios vs Market Factor (MF)",
                 fontsize=13, fontweight="bold", color=_C_TEXT, pad=12)
    _legend(ax, loc="lower right")

    return _save(fig, "08_capture_ratios.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — SUB-PERIOD PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

def plot_sub_period_performance(metrics_df: pd.DataFrame) -> Path:
    """
    Grouped bar chart comparing Active vs Passive mean CAGR across
    Pre-COVID, COVID, and Post-COVID periods.
    """
    period_cols = {
        "Pre-COVID\n(2013–2019)": "CAGR_Pre_COVID",
        "COVID\n(2020–2021)":     "CAGR_COVID",
        "Post-COVID\n(2022–2026)":"CAGR_Post_COVID",
    }

    avail = {k: v for k, v in period_cols.items() if v in metrics_df.columns}
    if not avail:
        logger.warning("  Sub-period CAGR columns missing — skipping chart.")
        return FIGURES_DIR / "09_sub_period_SKIPPED.png"

    labels = list(avail.keys())
    cols   = list(avail.values())

    act_vals = [
        metrics_df[metrics_df[COL_FUND_TYPE] == FUND_TYPE_ACTIVE][c].mean() * 100
        for c in cols
    ]
    pas_vals = [
        metrics_df[metrics_df[COL_FUND_TYPE] == FUND_TYPE_PASSIVE][c].mean() * 100
        for c in cols
    ]

    x = np.arange(len(labels))
    w = 0.33
    fig, ax = plt.subplots(figsize=(11, 6))
    _apply_style(fig, [ax])

    b1 = ax.bar(x - w / 2, act_vals, width=w, color=_C_ACTIVE, alpha=0.85,
                edgecolor=_C_BG, linewidth=0.5, label=FUND_TYPE_ACTIVE, zorder=3)
    b2 = ax.bar(x + w / 2, pas_vals, width=w, color=_C_PASSIVE, alpha=0.85,
                edgecolor=_C_BG, linewidth=0.5, label=FUND_TYPE_PASSIVE, zorder=3)

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + 0.3 if h >= 0 else h - 1.2,
                    f"{h:.1f}%", ha="center", va="bottom",
                    fontsize=9, color=_C_TEXT, fontweight="bold")

    ax.axhline(0, color=_C_SUBTEXT, linewidth=0.8, linestyle=":", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, color=_C_TEXT)
    ax.set_ylabel("Mean CAGR (%)", fontsize=11)
    ax.set_title("Sub-Period Performance — Active vs Passive CAGR",
                 fontsize=13, fontweight="bold", color=_C_TEXT, pad=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    _legend(ax, loc="upper right")

    return _save(fig, "09_sub_period_performance.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — DRAWDOWN COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def plot_drawdown_comparison(metrics_df: pd.DataFrame) -> Path:
    """
    Horizontal bar chart of maximum drawdown for all funds, sorted by depth.
    Colour-coded by fund type.
    """
    df = metrics_df[[COL_FUND_NAME, COL_FUND_TYPE, "Max_Drawdown"]].dropna().copy()
    df = df.sort_values("Max_Drawdown", ascending=True)   # most negative at top

    short_names = [
        n.replace(" - Direct Plan", "").replace(" Direct Plan", "")
         .replace(" - Growth", "")[:45]
        for n in df[COL_FUND_NAME]
    ]
    colors = [_C_ACTIVE if ft == FUND_TYPE_ACTIVE else _C_PASSIVE
              for ft in df[COL_FUND_TYPE]]

    fig, ax = plt.subplots(figsize=(13, max(9, len(df) * 0.31)))
    _apply_style(fig, [ax])

    ax.barh(range(len(df)), df["Max_Drawdown"].values * 100,
            color=colors, edgecolor=_C_BG, linewidth=0.4, height=0.75)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(short_names, fontsize=7.5, color=_C_TEXT)
    ax.axvline(0, color=_C_SUBTEXT, linewidth=0.8, linestyle=":", alpha=0.8)

    # Group mean lines
    for ft, color in [(FUND_TYPE_ACTIVE, _C_ACTIVE),
                      (FUND_TYPE_PASSIVE, _C_PASSIVE)]:
        mean_mdd = df[df[COL_FUND_TYPE] == ft]["Max_Drawdown"].mean() * 100
        ax.axvline(mean_mdd, color=color, linewidth=2.0,
                   linestyle="--", alpha=0.85,
                   label=f"{ft} mean: {mean_mdd:.1f}%")

    act_patch  = mpatches.Patch(color=_C_ACTIVE,  label=FUND_TYPE_ACTIVE)
    pas_patch  = mpatches.Patch(color=_C_PASSIVE, label=FUND_TYPE_PASSIVE)
    _legend(ax, handles=[act_patch, pas_patch], loc="lower right")

    ax.set_xlabel("Maximum Drawdown (%)", fontsize=11)
    ax.set_title("Maximum Drawdown — All Funds",
                 fontsize=13, fontweight="bold", color=_C_TEXT, pad=12)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    return _save(fig, "10_drawdown_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — ROLLING SHARPE RATIO
# ══════════════════════════════════════════════════════════════════════════════

def plot_rolling_sharpe(
    enriched_df: pd.DataFrame,
    window: int = 24,
    n_funds: int = 4,
) -> Path:
    """
    12/24-month rolling Sharpe ratio for the top-N active and passive funds
    by overall Sharpe, compared to their group average.
    """
    df = enriched_df.copy()
    df[COL_DATE] = pd.to_datetime(df[COL_DATE])

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    _apply_style(fig, list(axes))

    for ax, ft, color, title in [
        (axes[0], FUND_TYPE_ACTIVE,  _C_ACTIVE,
         f"Active Funds — {window}M Rolling Sharpe"),
        (axes[1], FUND_TYPE_PASSIVE, _C_PASSIVE,
         f"Passive Funds — {window}M Rolling Sharpe"),
    ]:
        sub = df[df[COL_FUND_TYPE] == ft]

        # Pick top-N funds by overall Sharpe
        sharpes = (
            sub.groupby(COL_FUND_NAME)[COL_RETURN]
            .apply(lambda r: r.mean() / r.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR)
                   if r.std(ddof=1) > 0 else 0)
            .nlargest(n_funds)
            .index.tolist()
        )

        for fund in sharpes:
            fd = sub[sub[COL_FUND_NAME] == fund].sort_values(COL_DATE)
            fd = fd.set_index(COL_DATE)[COL_EXCESS_RETURN].dropna()
            roll_sharpe = (
                fd.rolling(window, min_periods=window // 2)
                .apply(lambda x: (x.mean() / x.std(ddof=1)) * np.sqrt(MONTHS_PER_YEAR)
                       if x.std(ddof=1) > 0 else np.nan, raw=True)
            )
            short = fund.split("-")[0].strip()[:30]
            ax.plot(roll_sharpe.index, roll_sharpe.values,
                    linewidth=1.1, alpha=0.65, label=short)

        # Group average rolling Sharpe
        grp_exc = (
            sub.groupby(COL_DATE)[COL_EXCESS_RETURN]
            .mean()
            .sort_index()
        )
        grp_roll = (
            grp_exc.rolling(window, min_periods=window // 2)
            .apply(lambda x: (x.mean() / x.std(ddof=1)) * np.sqrt(MONTHS_PER_YEAR)
                   if x.std(ddof=1) > 0 else np.nan, raw=True)
        )
        ax.plot(grp_roll.index, grp_roll.values,
                color=color, linewidth=2.8, alpha=0.95, label="Group avg", zorder=5)
        ax.axhline(0, color=_C_SUBTEXT, linewidth=0.8,
                   linestyle=":", alpha=0.8)

        ax.set_title(title, fontsize=11, fontweight="bold",
                     color=_C_TEXT, pad=8)
        ax.set_ylabel("Sharpe Ratio", fontsize=10)
        _legend(ax, loc="upper left", ncol=2)

    axes[-1].set_xlabel("Date", fontsize=10)
    fig.suptitle(f"{window}-Month Rolling Sharpe Ratio — Active vs Passive",
                 fontsize=13, fontweight="bold", color=_C_TEXT, y=1.01)
    fig.tight_layout()
    return _save(fig, "11_rolling_sharpe.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — ALPHA SIGNIFICANCE GRID (Forest Plot)
# ══════════════════════════════════════════════════════════════════════════════

def plot_alpha_significance_grid(
    reg_results_df: pd.DataFrame,
    stat_tests_df: Optional[pd.DataFrame] = None,
) -> Path:
    """
    Forest plot: per-fund alpha point estimates with HAC confidence intervals.
    Colour indicates BH-FDR significance and alpha sign.
    Sorted within each fund type by alpha magnitude.
    """
    ok = reg_results_df[reg_results_df["Regression_Status"] == "OK"].copy()
    ok["Alpha_Ann"] = ((1 + ok[COL_ALPHA]) ** MONTHS_PER_YEAR - 1) * 100

    # Approximate 95% CI from t-stat and alpha (ann):
    # se_monthly ≈ alpha_monthly / T_Alpha
    # se_ann ≈ se_monthly * 12 (rough, ignoring compounding)
    ok["SE_Ann"]  = ok["Alpha_Ann"].abs() / (ok["T_Alpha"].abs().replace(0, np.nan))
    ok["CI_Lo"]   = ok["Alpha_Ann"] - 1.96 * ok["SE_Ann"]
    ok["CI_Hi"]   = ok["Alpha_Ann"] + 1.96 * ok["SE_Ann"]

    if stat_tests_df is not None and "Sig_BH_FDR" in stat_tests_df.columns:
        ok = ok.merge(stat_tests_df[[COL_FUND_NAME, "Sig_BH_FDR"]],
                      on=COL_FUND_NAME, how="left")
    else:
        ok["Sig_BH_FDR"] = ok[COL_P_ALPHA] < _ALPHA_SIG

    ok = ok.sort_values([COL_FUND_TYPE, "Alpha_Ann"], ascending=[True, True])

    short_names = [
        n.replace(" - Direct Plan", "").replace(" Direct Plan", "")
         .replace(" - Growth", "")[:40]
        for n in ok[COL_FUND_NAME]
    ]

    fig, ax = plt.subplots(figsize=(12, max(10, len(ok) * 0.33)))
    _apply_style(fig, [ax])

    for i, (_, row) in enumerate(ok.iterrows()):
        sig = row.get("Sig_BH_FDR", False)
        pos = row["Alpha_Ann"] > 0
        color = (_C_GOLD if (sig and pos) else
                 _C_SIG  if (sig and not pos) else
                 _C_SUBTEXT)

        ax.plot([row["CI_Lo"], row["CI_Hi"]], [i, i],
                color=color, linewidth=1.2, alpha=0.7, zorder=2)
        ax.scatter(row["Alpha_Ann"], i,
                   color=color, s=45, zorder=3, alpha=0.95,
                   edgecolors=_C_BG, linewidths=0.4)

    ax.axvline(0, color=_C_TEXT, linewidth=1.0,
               linestyle="--", alpha=0.9)

    # Type separator
    n_passive = (ok[COL_FUND_TYPE] == FUND_TYPE_PASSIVE).sum()
    ax.axhline(n_passive - 0.5, color=_C_ACCENT, linewidth=1.2,
               linestyle="-", alpha=0.7)
    ax.text(ax.get_xlim()[1], n_passive - 0.5,
            "  Active →", ha="left", va="center",
            fontsize=7.5, color=_C_ACCENT)

    ax.set_yticks(range(len(ok)))
    ax.set_yticklabels(short_names, fontsize=7.5, color=_C_TEXT)
    ax.set_xlabel("Annualised Alpha (%) with 95% CI", fontsize=11)
    ax.set_title("Forest Plot — Jensen's Alpha by Fund (BH-FDR corrected)",
                 fontsize=12, fontweight="bold", color=_C_TEXT, pad=12)

    # Legend
    handles = [
        mpatches.Patch(color=_C_GOLD,    label="Sig. positive α (BH-FDR)"),
        mpatches.Patch(color=_C_SIG,     label="Sig. negative α (BH-FDR)"),
        mpatches.Patch(color=_C_SUBTEXT, label="Not significant"),
    ]
    _legend(ax, handles=handles, loc="lower right")

    return _save(fig, "12_alpha_significance_grid.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — MASTER ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_all_charts(
    enriched_df: pd.DataFrame,
    reg_results_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    reg_input_df: pd.DataFrame,
    stat_tests_df: Optional[pd.DataFrame] = None,
) -> dict[str, Path]:
    """
    Render and save all 12 charts.

    Parameters
    ----------
    enriched_df : pd.DataFrame
        Enriched master frame (all funds, all months).
    reg_results_df : pd.DataFrame
        Per-fund regression results.
    metrics_df : pd.DataFrame
        Per-fund performance metrics.
    reg_input_df : pd.DataFrame
        Factor-merged regression input (for factor time series).
    stat_tests_df : pd.DataFrame, optional
        Multiple-comparison corrected alpha results (for BH-FDR colouring).

    Returns
    -------
    dict[str, Path]
        Mapping of chart name → saved file path.
    """
    logger.info("\n" + "=" * 68)
    logger.info("  VISUALIZATION PHASE — generate_all_charts")
    logger.info("=" * 68)

    ensure_dir(FIGURES_DIR)
    saved: dict[str, Path] = {}

    # Attach MF column to enriched if missing (needed for capture ratios)
    if COL_MF not in enriched_df.columns or enriched_df[COL_MF].isna().all():
        mf_series = reg_input_df.groupby(COL_DATE)[COL_MF].mean().reset_index()
        enriched_df = enriched_df.merge(mf_series, on=COL_DATE, how="left",
                                        suffixes=("", "_reg"))
        if f"{COL_MF}_reg" in enriched_df.columns:
            enriched_df[COL_MF] = enriched_df[COL_MF].fillna(enriched_df[f"{COL_MF}_reg"])
            enriched_df.drop(columns=[f"{COL_MF}_reg"], inplace=True)

    # Also ensure Excess_Return is present
    if COL_EXCESS_RETURN not in enriched_df.columns:
        if "RF" in enriched_df.columns:
            enriched_df[COL_EXCESS_RETURN] = enriched_df[COL_RETURN] - enriched_df["RF"]

    charts = [
        ("01_wealth_index",      lambda: plot_nav_comparison(enriched_df)),
        ("02_return_dist",       lambda: plot_return_distributions(enriched_df)),
        ("03_alpha_dist",        lambda: plot_alpha_distribution(reg_results_df, stat_tests_df)),
        ("04_factor_ts",         lambda: plot_factor_time_series(reg_input_df)),
        ("05_sharpe_heatmap",    lambda: plot_sharpe_heatmap(metrics_df)),
        ("06_er_vs_alpha",       lambda: plot_er_vs_alpha(reg_results_df, stat_tests_df)),
        ("07_risk_return",       lambda: plot_risk_return_scatter(metrics_df)),
        ("08_capture_ratios",    lambda: plot_capture_ratios(metrics_df)),
        ("09_sub_period",        lambda: plot_sub_period_performance(metrics_df)),
        ("10_drawdown",          lambda: plot_drawdown_comparison(metrics_df)),
        ("11_rolling_sharpe",    lambda: plot_rolling_sharpe(enriched_df)),
        ("12_alpha_forest",      lambda: plot_alpha_significance_grid(reg_results_df, stat_tests_df)),
    ]

    for name, fn in charts:
        try:
            path = fn()
            saved[name] = path
            logger.info(f"  ✅  {name}")
        except Exception as exc:
            logger.error(f"  ❌  {name}: {exc}")

    logger.info(
        f"\n  ╔══════ VISUALIZATION SUMMARY ══════╗"
    )
    logger.info(
        f"  ║  Charts generated : {len(saved):>4} / {len(charts)}        ║"
    )
    logger.info(
        f"  ║  Output directory : outputs/figures/       ║"
    )
    logger.info(
        f"  ╚═══════════════════════════════════╝"
    )
    logger.info("\n  Visualization complete.  Proceed to main.py.")
    logger.info("=" * 68 + "\n")

    return saved


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 14 — SELF-TEST (python src/visualization.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _log = setup_logger("viz_selftest", level="INFO")
    _log.info("\n" + "═" * 68)
    _log.info("  VISUALIZATION SELF-TEST")
    _log.info("═" * 68)

    try:
        from src.config import (
            CLEANED_DATA_DIR, REGRESSION_RESULTS_DIR,
            PERFORMANCE_METRICS_FILE, STATISTICAL_TESTS_FILE,
        )

        enriched   = pd.read_csv(CLEANED_DATA_DIR / "master_enriched.csv",
                                  parse_dates=["Date"])
        reg_input  = pd.read_csv(REGRESSION_RESULTS_DIR / "regression_input.csv",
                                  parse_dates=["Date"])
        reg_res    = pd.read_csv(REGRESSION_RESULTS_DIR / "carhart_regression_summary.csv")
        metrics    = pd.read_csv(PERFORMANCE_METRICS_FILE)
        stat_tests = pd.read_csv(STATISTICAL_TESTS_FILE) \
            if STATISTICAL_TESTS_FILE.exists() else None

        _log.info(
            f"  Loaded:  enriched={enriched.shape}  reg_res={reg_res.shape}  "
            f"metrics={metrics.shape}"
        )

        saved = generate_all_charts(
            enriched_df=enriched,
            reg_results_df=reg_res,
            metrics_df=metrics,
            reg_input_df=reg_input,
            stat_tests_df=stat_tests,
        )

        # ── Assertions ─────────────────────────────────────────────────────────
        _log.info("\n── Assertions ──")

        assert len(saved) >= 10, \
            f"Expected ≥ 10 charts, got {len(saved)}"
        _log.info(f"  ✅  {len(saved)} charts generated.")

        for name, path in saved.items():
            assert path.exists(), f"Chart file missing: {path}"
            assert path.stat().st_size > 10_000, \
                f"Chart file suspiciously small: {path.name} ({path.stat().st_size} bytes)"
        _log.info(f"  ✅  All chart files exist and are non-trivial size.")

        _log.info(f"\n  Chart files:")
        for name, path in sorted(saved.items()):
            size_kb = path.stat().st_size / 1024
            _log.info(f"    {path.name:<45}  {size_kb:>7.1f} KB")

        _log.info("\n✅  All visualization self-tests passed.")

    except Exception as exc:
        _log.error(f"❌  Self-test FAILED: {exc}")
        import traceback
        traceback.print_exc()
        raise

    _log.info("═" * 68 + "\n")
