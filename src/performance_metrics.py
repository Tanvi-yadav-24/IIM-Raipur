"""
performance_metrics.py
======================
Comprehensive performance metric computation for the Carhart Four-Factor
Fund Analysis Pipeline.

Responsibility
--------------
Accepts the enriched master DataFrame (from ``return_calculations.py``) and
the per-fund regression results (from ``regression_analysis.py``) to produce
a single, publication-ready per-fund metrics table covering:

  • Return metrics       — CAGR, total return, best/worst month, hit rate
  • Risk metrics         — volatility, max drawdown (with dates & duration),
                           VaR (95%/99%), CVaR (ES), downside deviation
  • Risk-adjusted ratios — Sharpe, Sortino, Calmar, Treynor, Information Ratio
  • Market capture ratios— up-capture, down-capture vs MF proxy
  • Regression outputs   — alpha (annualised), beta, R², significance
  • Sub-period analysis  — Pre-COVID (2013–2019), COVID (2020–2021),
                           Post-COVID (2022–2026)
  • Expense analysis     — gross alpha (alpha + ER), ER quartile, net alpha

All metrics are computed per-fund and then aggregated into Active vs Passive
group-level summary tables for the research report.

Module layout
-------------
  §1  _compute_drawdown_series       — drawdown timeseries from NAV
  §2  compute_drawdown_analysis      — max DD, start/end dates, duration, recovery
  §3  compute_var_cvar               — historical VaR and CVaR at 95%/99%
  §4  compute_capture_ratios         — up/down capture vs market factor
  §5  compute_sub_period_cagr        — CAGR for pre/during/post-COVID windows
  §6  compute_fund_metrics           — all metrics for a single fund
  §7  build_metrics_table            — orchestrator over all funds
  §8  compute_group_summary          — Active vs Passive aggregate comparison
  §9  rank_funds                     — fund ranking tables by metric
  §10 build_full_performance_report  — master orchestrator
  §11 Self-test (python src/performance_metrics.py)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats as stats

# ── Path fix for direct execution ─────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    ALPHA_SIGNIFICANCE,
    COL_ADJ_R2,
    COL_ALPHA,
    COL_ANN_RETURN,
    COL_ANN_VOL,
    COL_BETA_MF,
    COL_BETA_WML,
    COL_CAGR,
    COL_DATE,
    COL_ER,
    COL_EXCESS_RETURN,
    COL_FUND_NAME,
    COL_FUND_TYPE,
    COL_MAX_DRAWDOWN,
    COL_MF,
    COL_N_OBS,
    COL_NAV,
    COL_P_ALPHA,
    COL_R2,
    COL_REG_STATUS,
    COL_RETURN,
    COL_RF,
    COL_SHARPE,
    COL_T_ALPHA,
    COL_WML,
    FUND_TYPE_ACTIVE,
    FUND_TYPE_PASSIVE,
    MONTHS_PER_YEAR,
    PERFORMANCE_METRICS_FILE,
)
from src.utils import (
    annualize_return,
    annualize_volatility,
    compute_cagr,
    compute_max_drawdown,
    compute_sharpe_ratio,
    save_dataframe,
    setup_logger,
)

logger = setup_logger(__name__)

# ── Sub-period definitions ─────────────────────────────────────────────────────
_PERIODS: dict[str, tuple[str, str]] = {
    "Pre_COVID":  ("2013-01-01", "2019-12-31"),
    "COVID":      ("2020-01-01", "2021-12-31"),
    "Post_COVID": ("2022-01-01", "2026-04-30"),
}

# Confidence levels for VaR / CVaR
_VAR_LEVELS: list[float] = [0.95, 0.99]

# Minimum observations to compute a ratio (otherwise NaN)
_MIN_OBS_RATIO = 12

# Column name constants for this module
_COL_TOTAL_RETURN   = "Total_Return"
_COL_BEST_MONTH     = "Best_Month"
_COL_WORST_MONTH    = "Worst_Month"
_COL_HIT_RATE       = "Hit_Rate"
_COL_MAX_DD         = "Max_Drawdown"
_COL_MAX_DD_START   = "Max_DD_Start"
_COL_MAX_DD_END     = "Max_DD_End"
_COL_MAX_DD_DUR     = "Max_DD_Duration_Months"
_COL_MAX_DD_RECOV   = "Max_DD_Recovery_Months"
_COL_VAR_95         = "VaR_95"
_COL_VAR_99         = "VaR_99"
_COL_CVAR_95        = "CVaR_95"
_COL_CVAR_99        = "CVaR_99"
_COL_DOWNSIDE_DEV   = "Downside_Deviation"
_COL_SORTINO        = "Sortino_Ratio"
_COL_CALMAR         = "Calmar_Ratio"
_COL_TREYNOR        = "Treynor_Ratio"
_COL_INFO_RATIO     = "Information_Ratio"
_COL_UP_CAPTURE     = "Up_Capture"
_COL_DOWN_CAPTURE   = "Down_Capture"
_COL_ALPHA_ANN      = "Alpha_Annual_Pct"
_COL_ALPHA_SIG      = "Alpha_Significant"
_COL_ALPHA_GROSS    = "Alpha_Gross_Annual_Pct"
_COL_ER_QUARTILE    = "ER_Quartile"
_COL_START_DATE     = "Start_Date"
_COL_END_DATE       = "End_Date"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DRAWDOWN SERIES
# ══════════════════════════════════════════════════════════════════════════════

def _compute_drawdown_series(nav: pd.Series) -> pd.Series:
    """
    Compute the running drawdown series from a NAV / price index.

    Drawdown_t = (NAV_t − max(NAV_{0..t})) / max(NAV_{0..t})

    Parameters
    ----------
    nav : pd.Series
        NAV values indexed by date (ascending).

    Returns
    -------
    pd.Series
        Drawdown values in [−1, 0] (negative means loss from peak).
    """
    running_peak = nav.cummax()
    return (nav - running_peak) / running_peak


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DRAWDOWN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def compute_drawdown_analysis(
    fund_df: pd.DataFrame,
) -> dict:
    """
    Full maximum drawdown analysis: depth, start, end, duration, recovery.

    The maximum drawdown (MDD) is defined as the largest peak-to-trough
    decline in NAV over the entire observation period.

    Additional metrics
    ------------------
    * MDD start : date of the last NAV peak before the trough.
    * MDD end   : date of the trough (minimum NAV point).
    * Duration  : number of calendar months from peak to trough.
    * Recovery  : number of months from trough to first recovery above peak
                  (NaN if fund never fully recovered by the end of data).

    Parameters
    ----------
    fund_df : pd.DataFrame
        Single-fund DataFrame with ``Date`` and ``NAV`` columns, sorted by date.

    Returns
    -------
    dict
        Keys: Max_Drawdown, Max_DD_Start, Max_DD_End, Max_DD_Duration_Months,
        Max_DD_Recovery_Months.
    """
    nav = fund_df.set_index(COL_DATE)[COL_NAV].sort_index()

    if nav.empty or nav.isna().all():
        return {
            _COL_MAX_DD:     np.nan,
            _COL_MAX_DD_START: pd.NaT,
            _COL_MAX_DD_END:   pd.NaT,
            _COL_MAX_DD_DUR:   np.nan,
            _COL_MAX_DD_RECOV: np.nan,
        }

    dd_series    = _compute_drawdown_series(nav)
    mdd_value    = float(dd_series.min())   # most negative = largest drawdown
    trough_date  = dd_series.idxmin()

    # Peak = last date before trough where NAV equalled its running maximum
    pre_trough   = nav.loc[:trough_date]
    peak_date    = pre_trough.idxmax()

    # Duration peak → trough (in months, approx)
    duration_m   = (
        (trough_date.year - peak_date.year) * 12
        + (trough_date.month - peak_date.month)
    )

    # Recovery: first date after trough where NAV ≥ peak NAV
    peak_nav     = float(nav.loc[peak_date])
    post_trough  = nav.loc[trough_date:]
    recovered    = post_trough[post_trough >= peak_nav]
    if recovered.empty:
        recovery_m = np.nan   # not yet recovered
    else:
        recov_date = recovered.index[0]
        recovery_m = (
            (recov_date.year  - trough_date.year)  * 12
            + (recov_date.month - trough_date.month)
        )

    return {
        _COL_MAX_DD:       round(mdd_value, 6),
        _COL_MAX_DD_START: peak_date,
        _COL_MAX_DD_END:   trough_date,
        _COL_MAX_DD_DUR:   int(duration_m),
        _COL_MAX_DD_RECOV: int(recovery_m) if not np.isnan(recovery_m) else np.nan,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — VALUE-AT-RISK AND CONDITIONAL VAR
# ══════════════════════════════════════════════════════════════════════════════

def compute_var_cvar(
    returns: pd.Series,
    levels: list[float] = _VAR_LEVELS,
) -> dict:
    """
    Compute historical (non-parametric) VaR and CVaR at each confidence level.

    Historical VaR
    --------------
    VaR_α = −quantile(returns, 1 − α)
    e.g. VaR_95 = −5th percentile of monthly returns.

    CVaR (Expected Shortfall, ES)
    -----------------------------
    CVaR_α = −mean(returns  ≤  −VaR_α)
    The average loss in the worst (1−α) fraction of months.

    Parameters
    ----------
    returns : pd.Series
        Monthly simple return series (decimal).
    levels : list[float]
        Confidence levels, e.g. [0.95, 0.99].

    Returns
    -------
    dict
        Keys: VaR_95, CVaR_95, VaR_99, CVaR_99.

    Notes
    -----
    * All values are returned as POSITIVE numbers representing the magnitude
      of potential loss (e.g. VaR_95 = 0.08 means an 8% loss at 95% CI).
    * NaN is returned if the return series has fewer than _MIN_OBS_RATIO obs.
    """
    result: dict = {}
    clean = returns.dropna()

    for lvl in levels:
        pct_label = int(lvl * 100)
        var_key   = f"VaR_{pct_label}"
        cvar_key  = f"CVaR_{pct_label}"

        if len(clean) < _MIN_OBS_RATIO:
            result[var_key]  = np.nan
            result[cvar_key] = np.nan
            continue

        var_threshold = float(np.percentile(clean, (1 - lvl) * 100))
        var_value     = -var_threshold   # positive loss magnitude

        tail_losses   = clean[clean <= var_threshold]
        cvar_value    = float(-tail_losses.mean()) if not tail_losses.empty else var_value

        result[var_key]  = round(var_value,  6)
        result[cvar_key] = round(cvar_value, 6)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CAPTURE RATIOS
# ══════════════════════════════════════════════════════════════════════════════

def compute_capture_ratios(
    fund_returns: pd.Series,
    market_returns: pd.Series,
) -> dict:
    """
    Compute up-market and down-market capture ratios vs the market factor.

    Up-Capture   = mean(fund return | market > 0) / mean(market return | market > 0)
    Down-Capture = mean(fund return | market < 0) / mean(market return | market < 0)

    Interpretation
    --------------
    Up-Capture > 100%   → fund outperforms market in up months (good).
    Down-Capture < 100% → fund loses less than market in down months (good).
    A skilled active manager should have Up-Capture > Down-Capture.

    Parameters
    ----------
    fund_returns : pd.Series
        Monthly simple returns for one fund.
    market_returns : pd.Series
        Monthly market factor returns (MF = Rm − Rf), same index.

    Returns
    -------
    dict
        Keys: Up_Capture, Down_Capture (both as ratios, not percentages).

    Notes
    -----
    * Series must share an aligned index (or be aligned by caller).
    * Months where either series is NaN are dropped.
    * If fewer than 5 up or down months exist, NaN is returned.
    """
    aligned  = pd.concat(
        [fund_returns.rename("fund"), market_returns.rename("mkt")], axis=1
    ).dropna()

    up_mask  = aligned["mkt"] > 0
    dn_mask  = aligned["mkt"] < 0

    def _capture(fund_col: pd.Series, mkt_col: pd.Series) -> float:
        if len(mkt_col) < 5 or mkt_col.mean() == 0:
            return np.nan
        return float(fund_col.mean() / mkt_col.mean())

    up_cap = _capture(aligned.loc[up_mask,  "fund"], aligned.loc[up_mask,  "mkt"])
    dn_cap = _capture(aligned.loc[dn_mask,  "fund"], aligned.loc[dn_mask,  "mkt"])

    return {
        _COL_UP_CAPTURE:   round(up_cap * 100, 2) if not np.isnan(up_cap) else np.nan,
        _COL_DOWN_CAPTURE: round(dn_cap * 100, 2) if not np.isnan(dn_cap) else np.nan,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — SUB-PERIOD CAGR
# ══════════════════════════════════════════════════════════════════════════════

def compute_sub_period_cagr(
    fund_df: pd.DataFrame,
    periods: dict[str, tuple[str, str]] = _PERIODS,
) -> dict:
    """
    Compute CAGR for each defined sub-period (Pre-COVID, COVID, Post-COVID).

    CAGR for a sub-period is computed from the first and last available NAV
    within that period.  If fewer than 2 observations exist in a period,
    NaN is returned for that period.

    Parameters
    ----------
    fund_df : pd.DataFrame
        Single-fund DataFrame with ``Date`` and ``NAV`` columns.
    periods : dict[str, tuple[str, str]]
        Mapping of period label → (start_date, end_date) strings.

    Returns
    -------
    dict
        Keys: CAGR_<period_label> (decimal form).
    """
    result: dict = {}
    nav_by_date = fund_df.set_index(COL_DATE)[COL_NAV].sort_index()

    for label, (start, end) in periods.items():
        col_key = f"CAGR_{label}"
        mask    = (nav_by_date.index >= pd.Timestamp(start)) & \
                  (nav_by_date.index <= pd.Timestamp(end))
        sub_nav = nav_by_date.loc[mask].dropna()

        if len(sub_nav) < 2:
            result[col_key] = np.nan
            continue

        n_months    = len(sub_nav) - 1
        start_nav   = float(sub_nav.iloc[0])
        end_nav     = float(sub_nav.iloc[-1])
        cagr_val    = compute_cagr(start_nav, end_nav, n_months)
        result[col_key] = round(cagr_val, 6)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — SINGLE-FUND COMPREHENSIVE METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_fund_metrics(
    fund_df: pd.DataFrame,
    market_df: Optional[pd.DataFrame],
    reg_row: Optional[pd.Series],
) -> dict:
    """
    Compute all performance metrics for a single fund.

    Parameters
    ----------
    fund_df : pd.DataFrame
        Single-fund subset of the enriched master frame.  Must contain:
        Date, NAV, Monthly_Return, Excess_Return, Expense_Ratio, MF.
    market_df : pd.DataFrame or None
        Monthly market factor table (Year_Month, MF) for capture ratios.
        If None, capture ratios are set to NaN.
    reg_row : pd.Series or None
        Row from the per-fund regression results DataFrame.
        If None, regression fields are set to NaN.

    Returns
    -------
    dict
        All metrics for this fund (one future row in the master metrics table).
    """
    fund_df = fund_df.sort_values(COL_DATE).copy()
    fund_name = fund_df[COL_FUND_NAME].iloc[0]
    fund_type = fund_df[COL_FUND_TYPE].iloc[0]

    rets    = fund_df[COL_RETURN].dropna()
    exc_ret = fund_df[COL_EXCESS_RETURN].dropna()
    nav_s   = fund_df.set_index(COL_DATE)[COL_NAV].sort_index().dropna()

    m: dict = {
        COL_FUND_NAME:  fund_name,
        COL_FUND_TYPE:  fund_type,
        _COL_START_DATE: fund_df[COL_DATE].min(),
        _COL_END_DATE:   fund_df[COL_DATE].max(),
        COL_N_OBS:       len(rets),
        COL_ER:          fund_df[COL_ER].iloc[0] if COL_ER in fund_df.columns else np.nan,
    }

    # ── Guard: skip if insufficient data ──────────────────────────────────────
    if len(rets) < _MIN_OBS_RATIO:
        return m

    # ── §6a: Return metrics ───────────────────────────────────────────────────
    m[_COL_TOTAL_RETURN] = round(float((1 + rets).prod() - 1), 6)
    m[COL_ANN_RETURN]    = round(annualize_return(rets.mean()), 6)
    m[_COL_BEST_MONTH]   = round(float(rets.max()), 6)
    m[_COL_WORST_MONTH]  = round(float(rets.min()), 6)
    m[_COL_HIT_RATE]     = round(float((rets > 0).mean()), 6)   # fraction of positive months

    # CAGR from first/last NAV
    if len(nav_s) >= 2:
        m[COL_CAGR] = round(
            compute_cagr(float(nav_s.iloc[0]), float(nav_s.iloc[-1]), len(nav_s) - 1),
            6,
        )
    else:
        m[COL_CAGR] = np.nan

    # ── §6b: Risk metrics ──────────────────────────────────────────────────────
    m[COL_ANN_VOL] = round(annualize_volatility(rets.std(ddof=1)), 6)

    # Downside deviation (annualised, using 0 as the minimum acceptable return)
    downside = rets[rets < 0]
    if len(downside) >= 2:
        m[_COL_DOWNSIDE_DEV] = round(
            float(np.sqrt(((downside ** 2).mean()) * MONTHS_PER_YEAR)), 6
        )
    else:
        m[_COL_DOWNSIDE_DEV] = np.nan

    # Drawdown analysis
    dd_metrics = compute_drawdown_analysis(fund_df)
    m.update(dd_metrics)

    # VaR & CVaR
    var_cvar = compute_var_cvar(rets)
    m.update(var_cvar)

    # ── §6c: Risk-adjusted ratios ─────────────────────────────────────────────

    # Sharpe Ratio (annualised, using excess return)
    m[COL_SHARPE] = round(compute_sharpe_ratio(exc_ret, annualize=True), 6)

    # Sortino Ratio = Ann. excess return / Ann. downside deviation
    if m.get(_COL_DOWNSIDE_DEV, np.nan) and not np.isnan(m.get(_COL_DOWNSIDE_DEV, np.nan)) \
            and m[_COL_DOWNSIDE_DEV] > 0:
        ann_excess = annualize_return(exc_ret.mean())
        m[_COL_SORTINO] = round(ann_excess / m[_COL_DOWNSIDE_DEV], 6)
    else:
        m[_COL_SORTINO] = np.nan

    # Calmar Ratio = CAGR / |Max Drawdown|
    mdd = m.get(_COL_MAX_DD, np.nan)
    cagr_val = m.get(COL_CAGR, np.nan)
    if not np.isnan(mdd) and mdd < 0 and not np.isnan(cagr_val):
        m[_COL_CALMAR] = round(cagr_val / abs(mdd), 6)
    else:
        m[_COL_CALMAR] = np.nan

    # Treynor Ratio = Ann. excess return / Beta_MF
    # Beta comes from regression results
    beta_mf = float(reg_row[COL_BETA_MF]) if reg_row is not None and COL_BETA_MF in reg_row and not pd.isna(reg_row[COL_BETA_MF]) else np.nan
    if not np.isnan(beta_mf) and beta_mf != 0:
        ann_exc = annualize_return(exc_ret.mean())
        m[_COL_TREYNOR] = round(ann_exc / beta_mf, 6)
    else:
        m[_COL_TREYNOR] = np.nan

    # Information Ratio = Alpha / Tracking Error (vs market)
    # Tracking error = std(fund_return - MF) — how consistently fund deviates from market
    if COL_MF in fund_df.columns:
        aligned  = fund_df[[COL_RETURN, COL_MF]].dropna()
        if len(aligned) >= _MIN_OBS_RATIO:
            active_return  = aligned[COL_RETURN] - aligned[COL_MF]
            tracking_error = annualize_volatility(active_return.std(ddof=1))
            mean_active    = annualize_return(active_return.mean())
            if tracking_error > 0:
                m[_COL_INFO_RATIO] = round(mean_active / tracking_error, 6)
            else:
                m[_COL_INFO_RATIO] = np.nan
        else:
            m[_COL_INFO_RATIO] = np.nan
    else:
        m[_COL_INFO_RATIO] = np.nan

    # ── §6d: Capture ratios ────────────────────────────────────────────────────
    if market_df is not None and COL_MF in fund_df.columns:
        aligned_mf = fund_df[[COL_DATE, COL_RETURN, COL_MF]].dropna()
        if len(aligned_mf) >= _MIN_OBS_RATIO:
            cap = compute_capture_ratios(
                aligned_mf[COL_RETURN], aligned_mf[COL_MF]
            )
            m.update(cap)
        else:
            m[_COL_UP_CAPTURE]   = np.nan
            m[_COL_DOWN_CAPTURE] = np.nan
    else:
        m[_COL_UP_CAPTURE]   = np.nan
        m[_COL_DOWN_CAPTURE] = np.nan

    # ── §6e: Sub-period CAGR ──────────────────────────────────────────────────
    sub_cagr = compute_sub_period_cagr(fund_df)
    m.update(sub_cagr)

    # ── §6f: Regression fields ────────────────────────────────────────────────
    if reg_row is not None and reg_row[COL_REG_STATUS] == "OK":
        raw_alpha  = float(reg_row[COL_ALPHA])
        ann_alpha  = (1 + raw_alpha) ** 12 - 1
        er_val     = m[COL_ER]
        gross_ann  = ann_alpha + (er_val / 100.0 if not np.isnan(er_val) else 0.0)

        m[_COL_ALPHA_ANN]   = round(ann_alpha * 100, 4)   # in %
        m[COL_T_ALPHA]      = round(float(reg_row[COL_T_ALPHA]), 4)
        m[COL_P_ALPHA]      = round(float(reg_row[COL_P_ALPHA]), 6)
        m[_COL_ALPHA_SIG]   = bool(reg_row["Alpha_Significant"])
        m[_COL_ALPHA_GROSS] = round(gross_ann * 100, 4)   # in %
        m[COL_BETA_MF]      = round(float(reg_row[COL_BETA_MF]), 6) if not pd.isna(reg_row[COL_BETA_MF]) else np.nan
        m[COL_BETA_WML]     = round(float(reg_row[COL_BETA_WML]), 6) if not pd.isna(reg_row.get(COL_BETA_WML, np.nan)) else np.nan
        m[COL_R2]           = round(float(reg_row[COL_R2]), 6)
        m[COL_ADJ_R2]       = round(float(reg_row[COL_ADJ_R2]), 6)
    else:
        for col in [_COL_ALPHA_ANN, COL_T_ALPHA, COL_P_ALPHA, _COL_ALPHA_SIG,
                    _COL_ALPHA_GROSS, COL_BETA_MF, COL_BETA_WML, COL_R2, COL_ADJ_R2]:
            m[col] = np.nan

    return m


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — BUILD MASTER METRICS TABLE
# ══════════════════════════════════════════════════════════════════════════════

def build_metrics_table(
    enriched_df: pd.DataFrame,
    reg_results_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute all performance metrics for every fund in the enriched master frame
    and return a consolidated wide-format metrics table (one row per fund).

    Parameters
    ----------
    enriched_df : pd.DataFrame
        Full enriched master frame from ``return_calculations.enrich_master_frame``.
        Must contain: Fund_Name, Fund_Type, Date, NAV, Monthly_Return,
        Excess_Return, Expense_Ratio, MF.
    reg_results_df : pd.DataFrame
        Per-fund regression results from ``regression_analysis.run_fund_regressions``.

    Returns
    -------
    pd.DataFrame
        Metrics table: one row per fund, columns = all metrics.
        Sorted by Fund_Type then Fund_Name.
    """
    logger.info("  §7 Building per-fund metrics table …")

    # Build a regression results lookup
    reg_lookup: dict[str, pd.Series] = {
        row[COL_FUND_NAME]: row
        for _, row in reg_results_df.iterrows()
    }

    # Build market factor series (indexed by date) for capture ratios
    market_series = enriched_df.groupby(COL_DATE)[COL_MF].mean().dropna()
    market_df_tmp = market_series.reset_index().rename(columns={COL_DATE: "Date"})

    funds  = sorted(enriched_df[COL_FUND_NAME].unique())
    n      = len(funds)
    records: list[dict] = []

    for i, fund_name in enumerate(funds, 1):
        fund_df  = enriched_df[enriched_df[COL_FUND_NAME] == fund_name].copy()
        reg_row  = reg_lookup.get(fund_name)

        metrics  = compute_fund_metrics(fund_df, market_df_tmp, reg_row)
        records.append(metrics)

        if i % 10 == 0 or i == n:
            logger.info(
                f"  Progress: {i}/{n} funds processed  |  "
                f"last: {fund_name[:50]}"
            )

    metrics_df = pd.DataFrame(records)

    # ── Add expense-ratio quartile within each fund type ─────────────────────
    metrics_df[_COL_ER_QUARTILE] = np.nan
    for ft in [FUND_TYPE_ACTIVE, FUND_TYPE_PASSIVE]:
        mask = metrics_df[COL_FUND_TYPE] == ft
        er_vals = metrics_df.loc[mask, COL_ER].dropna()
        if len(er_vals) >= 4:
            metrics_df.loc[mask, _COL_ER_QUARTILE] = pd.qcut(
                metrics_df.loc[mask, COL_ER],
                q=4,
                labels=[1, 2, 3, 4],
                duplicates="drop",
            ).astype(float)

    metrics_df = metrics_df.sort_values(
        [COL_FUND_TYPE, COL_FUND_NAME]
    ).reset_index(drop=True)

    logger.info(
        f"  Metrics table built: {len(metrics_df)} funds  ×  "
        f"{len(metrics_df.columns)} metrics."
    )
    return metrics_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — GROUP SUMMARY (ACTIVE vs PASSIVE)
# ══════════════════════════════════════════════════════════════════════════════

def compute_group_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-fund metrics into an Active vs Passive group-level summary.

    For each numeric metric, computes: mean, median, std, min, max.
    Non-numeric columns (dates, strings) are excluded.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Per-fund metrics table from ``build_metrics_table``.

    Returns
    -------
    pd.DataFrame
        Summary table with multi-level columns: (Fund_Type, statistic).

    Notes
    -----
    The returned DataFrame has one row per metric and two column groups
    (Active, Passive), each with sub-columns (mean, median, std).
    """
    logger.info("  §8 Computing Active vs Passive group summary …")

    numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = [COL_N_OBS, _COL_ER_QUARTILE]
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    summary_records: list[dict] = []
    for ft in [FUND_TYPE_ACTIVE, FUND_TYPE_PASSIVE]:
        sub = metrics_df[metrics_df[COL_FUND_TYPE] == ft]
        row: dict = {"Fund_Type": ft, "N_Funds": len(sub)}
        for col in numeric_cols:
            vals = sub[col].dropna()
            if len(vals) == 0:
                continue
            row[f"{col}_mean"]   = round(float(vals.mean()),   6)
            row[f"{col}_median"] = round(float(vals.median()), 6)
            row[f"{col}_std"]    = round(float(vals.std(ddof=1)), 6) if len(vals) > 1 else np.nan
        summary_records.append(row)

    summary_df = pd.DataFrame(summary_records)

    # ── Log a clean comparison table for key metrics ──────────────────────────
    key_metrics = [
        (COL_CAGR,         "CAGR",            "{:>+8.2%}"),
        (COL_ANN_RETURN,   "Ann. Return",      "{:>+8.2%}"),
        (COL_ANN_VOL,      "Ann. Volatility",  "{:>8.2%}"),
        (COL_SHARPE,       "Sharpe Ratio",     "{:>8.4f}"),
        (_COL_SORTINO,     "Sortino Ratio",    "{:>8.4f}"),
        (_COL_CALMAR,      "Calmar Ratio",     "{:>8.4f}"),
        (_COL_MAX_DD,      "Max Drawdown",     "{:>8.2%}"),
        (_COL_VAR_95,      "VaR 95%",          "{:>8.2%}"),
        (_COL_CVAR_95,     "CVaR 95%",         "{:>8.2%}"),
        (_COL_DOWNSIDE_DEV,"Downside Dev",     "{:>8.2%}"),
        (_COL_HIT_RATE,    "Hit Rate",         "{:>8.2%}"),
        (_COL_UP_CAPTURE,  "Up Capture",       "{:>7.1f}%"),
        (_COL_DOWN_CAPTURE,"Down Capture",     "{:>7.1f}%"),
        (_COL_ALPHA_ANN,   "Alpha (ann, %)",   "{:>+8.2f}%"),
        (COL_ER,           "Expense Ratio %",  "{:>8.2f}%"),
        (_COL_ALPHA_GROSS, "Gross Alpha %",    "{:>+8.2f}%"),
    ]

    logger.info(
        f"\n  ╔═══════════════════════════════════════════════════════════╗"
    )
    logger.info(
        f"  ║  {'Metric':<28}  {'Active':>12}  {'Passive':>12}  ║"
    )
    logger.info(
        f"  ║  {'─'*58}  ║"
    )

    act_row = summary_df[summary_df["Fund_Type"] == FUND_TYPE_ACTIVE].iloc[0] \
        if not summary_df[summary_df["Fund_Type"] == FUND_TYPE_ACTIVE].empty else {}
    pas_row = summary_df[summary_df["Fund_Type"] == FUND_TYPE_PASSIVE].iloc[0] \
        if not summary_df[summary_df["Fund_Type"] == FUND_TYPE_PASSIVE].empty else {}

    for col, label, fmt in key_metrics:
        mean_key = f"{col}_mean"
        a_val    = act_row.get(mean_key, np.nan)
        p_val    = pas_row.get(mean_key, np.nan)
        try:
            a_str = fmt.format(a_val) if not pd.isna(a_val) else "       N/A"
            p_str = fmt.format(p_val) if not pd.isna(p_val) else "       N/A"
        except Exception:
            a_str = f"{a_val:>10.4f}" if not pd.isna(a_val) else "       N/A"
            p_str = f"{p_val:>10.4f}" if not pd.isna(p_val) else "       N/A"
        logger.info(
            f"  ║  {label:<28}  {a_str:>12}  {p_str:>12}  ║"
        )

    logger.info(
        f"  ╚═══════════════════════════════════════════════════════════╝"
    )

    return summary_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — FUND RANKINGS
# ══════════════════════════════════════════════════════════════════════════════

def rank_funds(
    metrics_df: pd.DataFrame,
    rank_by: str = COL_SHARPE,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Return the top and bottom funds by a specified metric, within each group.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Per-fund metrics table.
    rank_by : str
        Column name to rank by (higher = better assumed; for drawdown
        lower magnitude = better).
    top_n : int
        Number of top funds to show per group.

    Returns
    -------
    pd.DataFrame
        Ranked table with Fund_Name, Fund_Type, the rank metric, and
        Sharpe_Ratio, CAGR, Expense_Ratio for context.
    """
    cols = [COL_FUND_NAME, COL_FUND_TYPE, rank_by, COL_SHARPE, COL_CAGR, COL_ER]
    # De-duplicate: each column should appear at most once
    seen: set = set()
    available_cols: list[str] = []
    for c in cols:
        if c in metrics_df.columns and c not in seen:
            available_cols.append(c)
            seen.add(c)
    ranked = (
        metrics_df[available_cols]
        .dropna(subset=[rank_by])
        .sort_values(rank_by, ascending=False)
        .reset_index(drop=True)
    )
    ranked.insert(0, "Rank", ranked.index + 1)
    return ranked.head(top_n * 2)   # top_n × 2 to include both fund types


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — MASTER ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def build_full_performance_report(
    enriched_df: pd.DataFrame,
    reg_results_df: pd.DataFrame,
    save_outputs: bool = True,
) -> dict:
    """
    Orchestrate the complete performance metrics pipeline.

    Steps
    -----
    1.  Build per-fund metrics table (§7).
    2.  Compute Active vs Passive group summary (§8).
    3.  Produce fund rankings by Sharpe, CAGR, alpha (§9).
    4.  Save all outputs (optional).

    Parameters
    ----------
    enriched_df : pd.DataFrame
        Full enriched master frame from ``return_calculations.py``.
    reg_results_df : pd.DataFrame
        Per-fund regression results from ``regression_analysis.py``.
    save_outputs : bool
        If True, save metrics CSV and group summary CSV.

    Returns
    -------
    dict
        Keys: "metrics_table", "group_summary",
              "ranking_sharpe", "ranking_cagr", "ranking_alpha".
    """
    logger.info("\n" + "=" * 68)
    logger.info("  PERFORMANCE METRICS PHASE — build_full_performance_report")
    logger.info("=" * 68)

    # ═══ Step 1: Per-fund metrics ══════════════════════════════════════════════
    logger.info("\n── Step 1: Computing per-fund metrics ──")
    metrics_df = build_metrics_table(enriched_df, reg_results_df)

    # ═══ Step 2: Group summary ════════════════════════════════════════════════
    logger.info("\n── Step 2: Active vs Passive group summary ──")
    group_summary = compute_group_summary(metrics_df)

    # ═══ Step 3: Rankings ════════════════════════════════════════════════════
    logger.info("\n── Step 3: Fund rankings ──")
    rank_sharpe = rank_funds(metrics_df, rank_by=COL_SHARPE, top_n=10)
    rank_cagr   = rank_funds(metrics_df, rank_by=COL_CAGR,   top_n=10)
    rank_alpha  = rank_funds(metrics_df, rank_by=_COL_ALPHA_ANN, top_n=10)

    logger.info(f"\n  Top 5 funds by Sharpe Ratio:")
    for _, row in rank_sharpe.head(5).iterrows():
        er    = row.get(COL_ER, np.nan)
        sharpe = row.get(COL_SHARPE, np.nan)
        cagr  = row.get(COL_CAGR, np.nan)
        logger.info(
            f"    {int(row['Rank']):>2}. [{row[COL_FUND_TYPE][0]}]  "
            f"{row[COL_FUND_NAME][:52]:52s}  "
            f"Sharpe={sharpe:.4f}  CAGR={cagr:.2%}  ER={er:.2f}%"
        )

    # ═══ Step 4: Save ════════════════════════════════════════════════════════
    if save_outputs:
        from src.config import REGRESSION_RESULTS_DIR
        from src.utils import ensure_dir
        ensure_dir(REGRESSION_RESULTS_DIR)

        # Convert date columns to string for CSV
        export_metrics = metrics_df.copy()
        for dc in [_COL_START_DATE, _COL_END_DATE, _COL_MAX_DD_START, _COL_MAX_DD_END]:
            if dc in export_metrics.columns:
                export_metrics[dc] = pd.to_datetime(
                    export_metrics[dc], errors="coerce"
                ).dt.strftime("%Y-%m-%d")

        save_dataframe(
            export_metrics,
            PERFORMANCE_METRICS_FILE,
            description="Per-fund performance metrics",
        )
        save_dataframe(
            group_summary,
            REGRESSION_RESULTS_DIR / "group_performance_summary.csv",
            description="Group performance summary (Active vs Passive)",
        )
        save_dataframe(
            rank_sharpe,
            REGRESSION_RESULTS_DIR / "ranking_by_sharpe.csv",
            description="Fund ranking by Sharpe",
        )
        save_dataframe(
            rank_alpha,
            REGRESSION_RESULTS_DIR / "ranking_by_alpha.csv",
            description="Fund ranking by Alpha",
        )

    # ═══ Final summary ════════════════════════════════════════════════════════
    n_metrics = len([c for c in metrics_df.columns if c not in [COL_FUND_NAME, COL_FUND_TYPE]])
    logger.info(f"\n  ╔══════ PERFORMANCE METRICS SUMMARY ══════╗")
    logger.info(f"  ║  Funds covered      : {len(metrics_df):>6}              ║")
    logger.info(f"  ║  Metrics per fund   : {n_metrics:>6}              ║")
    logger.info(f"  ║  Active funds       : {(metrics_df[COL_FUND_TYPE]==FUND_TYPE_ACTIVE).sum():>6}              ║")
    logger.info(f"  ║  Passive funds      : {(metrics_df[COL_FUND_TYPE]==FUND_TYPE_PASSIVE).sum():>6}              ║")
    logger.info(f"  ╚═════════════════════════════════════════╝")
    logger.info("\n  Performance metrics complete.  Proceed to statistical_tests.py.")
    logger.info("=" * 68 + "\n")

    return {
        "metrics_table":  metrics_df,
        "group_summary":  group_summary,
        "ranking_sharpe": rank_sharpe,
        "ranking_cagr":   rank_cagr,
        "ranking_alpha":  rank_alpha,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — SELF-TEST (python src/performance_metrics.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from src.data_loader import load_all_data
    from src.preprocessing import build_master_frame
    from src.return_calculations import enrich_master_frame
    from src.factor_merge import build_and_merge_factors
    from src.regression_analysis import run_full_regression_analysis

    _log = setup_logger("perf_metrics_selftest", level="INFO")
    _log.info("\n" + "═" * 68)
    _log.info("  PERFORMANCE METRICS SELF-TEST")
    _log.info("═" * 68)

    try:
        # ── Full pipeline ─────────────────────────────────────────────────────
        raw    = load_all_data()
        master = build_master_frame(
            passive_df=raw["passive_nav"],
            active_df=raw["active_nav"],
            er_df=raw["expense_ratios"],
            save_intermediates=False,
        )
        enriched, _, reg_df = enrich_master_frame(master_df=master, save_outputs=False)
        factor_table, merged_df = build_and_merge_factors(
            master_df=enriched, reg_df=reg_df,
            raw_factor_df=raw["factor_data"], save_outputs=False,
        )
        reg_out = run_full_regression_analysis(merged_df, save_outputs=False)
        reg_results = reg_out["fund_results"]

        # ── Run performance metrics ────────────────────────────────────────────
        report = build_full_performance_report(
            enriched_df=enriched,
            reg_results_df=reg_results,
            save_outputs=True,
        )

        metrics = report["metrics_table"]
        summary = report["group_summary"]

        _log.info("\n── Assertions ──")

        # A1: Shape — one row per fund
        assert len(metrics) == enriched[COL_FUND_NAME].nunique(), \
            f"Row count mismatch: {len(metrics)} vs {enriched[COL_FUND_NAME].nunique()}"
        _log.info(
            f"  ✅ Metrics table: {len(metrics)} funds  ×  {len(metrics.columns)} cols."
        )

        # A2: CAGR in plausible range for Indian equity funds
        cagr = metrics[COL_CAGR].dropna()
        assert (cagr > -0.5).all() and (cagr < 2.0).all(), \
            f"CAGR out of plausible range: [{cagr.min():.2%}, {cagr.max():.2%}]"
        _log.info(
            f"  ✅ CAGR plausible: [{cagr.min():.2%}, {cagr.max():.2%}]."
        )

        # A3: Max Drawdown is non-positive
        mdd = metrics[_COL_MAX_DD].dropna()
        assert (mdd <= 0).all(), f"Positive max drawdown found: {mdd.max():.4f}"
        _log.info(
            f"  ✅ Max Drawdown ≤ 0 for all funds.  "
            f"Range: [{mdd.min():.2%}, {mdd.max():.2%}]."
        )

        # A4: VaR 95 > 0 (loss magnitude)
        var95 = metrics[_COL_VAR_95].dropna()
        assert (var95 > 0).all(), "Negative VaR_95 detected"
        _log.info(
            f"  ✅ VaR_95 > 0 for all funds.  "
            f"Mean: {var95.mean():.2%}."
        )

        # A5: Hit rate in [0, 1]
        hr = metrics[_COL_HIT_RATE].dropna()
        assert (hr >= 0).all() and (hr <= 1).all(), "Hit rate out of [0,1]"
        _log.info(
            f"  ✅ Hit rate in [0,1].  "
            f"Mean: Active={metrics[metrics[COL_FUND_TYPE]==FUND_TYPE_ACTIVE][_COL_HIT_RATE].mean():.1%}  "
            f"Passive={metrics[metrics[COL_FUND_TYPE]==FUND_TYPE_PASSIVE][_COL_HIT_RATE].mean():.1%}."
        )

        # A6: Group summary has both types
        assert set(summary["Fund_Type"].tolist()) == {FUND_TYPE_ACTIVE, FUND_TYPE_PASSIVE}, \
            "Group summary missing a fund type"
        _log.info(f"  ✅ Group summary covers both Active and Passive.")

        # A7: Alpha_Annual_Pct populated for regression-OK funds
        ok_funds  = reg_results[reg_results[COL_REG_STATUS] == "OK"][COL_FUND_NAME]
        alpha_ann = metrics[metrics[COL_FUND_NAME].isin(ok_funds)][_COL_ALPHA_ANN]
        assert alpha_ann.notna().sum() >= 60, \
            f"Too few funds with Alpha_Annual_Pct: {alpha_ann.notna().sum()}"
        _log.info(
            f"  ✅ Alpha_Annual_Pct populated for {alpha_ann.notna().sum()} funds."
        )

        # A8: Sub-period CAGR columns exist
        for period_col in ["CAGR_Pre_COVID", "CAGR_COVID", "CAGR_Post_COVID"]:
            assert period_col in metrics.columns, f"Missing: {period_col}"
        _log.info("  ✅ Sub-period CAGR columns present (Pre_COVID, COVID, Post_COVID).")

        # A9: Print key metric comparison
        _log.info(f"\n  ── Key metrics: Active vs Passive ──")
        key_pairs = [
            (COL_CAGR,         "CAGR",         "{:.2%}"),
            (COL_SHARPE,       "Sharpe",        "{:.4f}"),
            (_COL_SORTINO,     "Sortino",       "{:.4f}"),
            (_COL_CALMAR,      "Calmar",        "{:.4f}"),
            (_COL_MAX_DD,      "Max DD",        "{:.2%}"),
            (_COL_DOWNSIDE_DEV,"Downside Dev",  "{:.2%}"),
            (_COL_VAR_95,      "VaR 95%",       "{:.2%}"),
            (_COL_CVAR_95,     "CVaR 95%",      "{:.2%}"),
            (_COL_HIT_RATE,    "Hit Rate",      "{:.1%}"),
            (_COL_UP_CAPTURE,  "Up Capture",    "{:.1f}%"),
            (_COL_DOWN_CAPTURE,"Down Capture",  "{:.1f}%"),
            (_COL_ALPHA_ANN,   "Alpha (ann%)",  "{:+.2f}%"),
            (COL_ER,           "Expense Ratio", "{:.2f}%"),
        ]
        _log.info(f"  {'Metric':<22}  {'Active':>12}  {'Passive':>12}")
        _log.info(f"  {'-'*48}")
        for col, label, fmt in key_pairs:
            a = metrics[metrics[COL_FUND_TYPE]==FUND_TYPE_ACTIVE][col].mean()
            p = metrics[metrics[COL_FUND_TYPE]==FUND_TYPE_PASSIVE][col].mean()
            try:
                a_s = fmt.format(a) if not pd.isna(a) else "N/A"
                p_s = fmt.format(p) if not pd.isna(p) else "N/A"
            except Exception:
                a_s, p_s = str(round(a, 4)), str(round(p, 4))
            _log.info(f"  {label:<22}  {a_s:>12}  {p_s:>12}")

        _log.info("\n✅  All performance_metrics self-tests passed.")

    except Exception as exc:
        _log.error(f"❌  Self-test FAILED: {exc}")
        import traceback
        traceback.print_exc()
        raise

    _log.info("═" * 68 + "\n")
